# MOSS-Image strategy classes: tokenization, text encoding, text encoder output caching and latents caching

import os
import random
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer

from library import flux2_vae, moss_image_utils, moss_image_cache, moss_caption
from PIL import Image
import library.accelerator_setup as accelerator_setup
import library.device_utils as device_utils
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def bf16_to_uint16(tensor: torch.Tensor) -> np.ndarray:
    """bf16 tensor -> uint16 numpy array holding the raw bits (npz cannot store bf16 directly)."""
    return tensor.to(torch.bfloat16).contiguous().cpu().view(torch.int16).numpy().view(np.uint16)


def uint16_to_float32(array: np.ndarray) -> np.ndarray:
    """uint16 bf16 bits -> float32 numpy array (exact)."""
    return (array.astype(np.uint32) << 16).view(np.float32)


def uint16_to_bf16(array: np.ndarray) -> torch.Tensor:
    """View cached BF16 bits without expanding the CPU/H2D payload to float32."""
    if array.dtype != np.uint16:
        raise ValueError("MOSS cached embeddings must contain uint16 BF16 bits")
    return torch.from_numpy(array).view(torch.bfloat16)


class MossImageTokenizeStrategy(TokenizeStrategy):
    """Tokenizes prompts with the Qwen3.5 tokenizer after applying the chat template. Returns right-padded ids and a mask."""

    def __init__(self, max_length: int = moss_image_utils.MAX_TOKEN_LENGTH, tokenizer_cache_dir: Optional[str] = None) -> None:
        if not 1 <= max_length <= moss_image_utils.MAX_TOKEN_LENGTH:
            raise ValueError(f"moss_max_token_length must be between 1 and {moss_image_utils.MAX_TOKEN_LENGTH}")
        if tokenizer_cache_dir is not None and os.path.exists(os.path.join(tokenizer_cache_dir, "tokenizer_config.json")):
            # a local tokenizer directory
            self.tokenizer = moss_image_utils.load_tokenizer(tokenizer_cache_dir)
        else:
            self.tokenizer = self._load_tokenizer(
                AutoTokenizer,
                moss_image_utils.TOKENIZER_ID,
                subfolder=moss_image_utils.TOKENIZER_SUBFOLDER,
                tokenizer_cache_dir=tokenizer_cache_dir,
            )
        self.max_length = max_length
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        """Returns [input_ids [B, L], attention_mask [B, L]] (right-padded, int64)."""
        text = [text] if isinstance(text, str) else text
        input_ids_list = moss_image_utils.tokenize_prompts(self.tokenizer, text, self.max_length)
        length = max(ids.shape[0] for ids in input_ids_list)
        input_ids = torch.full((len(text), length), self.pad_token_id, dtype=torch.int64)
        attention_mask = torch.zeros((len(text), length), dtype=torch.int64)
        for i, ids in enumerate(input_ids_list):
            input_ids[i, : ids.shape[0]] = ids
            attention_mask[i, : ids.shape[0]] = 1
        return [input_ids, attention_mask]


class MossImageTextEncodingStrategy(TextEncodingStrategy):
    """Encodes each prompt separately (no padding inside the text encoder) and right-pads the outputs."""

    def __init__(self) -> None:
        pass

    def encode_tokens(self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            models: [text_encoder]
            tokens: [input_ids [B, L], attention_mask [B, L]]
        Returns:
            [prompt_embeds [B, L_max, C] (text encoder dtype), attn_mask [B, L_max] int32]
        """
        text_encoder = models[0]
        input_ids, attention_mask = tokens
        device = next(text_encoder.parameters()).device
        input_ids_list = [input_ids[i][attention_mask[i].bool()] for i in range(input_ids.shape[0])]
        embeds_list = moss_image_utils.encode_tokens(text_encoder, input_ids_list, device)
        prompt_embeds, attn_mask = moss_image_utils.pad_text_embeds(embeds_list)
        return [prompt_embeds, attn_mask]

    def drop_cached_text_encoder_outputs(
        self,
        prompt_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        uncond_embeds: torch.Tensor,
        uncond_mask: torch.Tensor,
        caption_dropout_rates: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Apply caption dropout to cached outputs: dropped samples are replaced by the cached unconditional embeddings
        (the encoding of ""), because the DiT needs at least one valid text token.

        Args:
            prompt_embeds: [B, L, C], attn_mask: [B, L], uncond_embeds: [B, L_u, C], uncond_mask: [B, L_u]
        Returns:
            [prompt_embeds [B, L', C], attn_mask [B, L'] int32] (re-padded to the longest valid length)
        """
        if caption_dropout_rates is None or torch.all(caption_dropout_rates == 0.0).item():
            return [prompt_embeds, attn_mask]

        embeds_list = []
        for i in range(prompt_embeds.shape[0]):
            if random.random() < caption_dropout_rates[i].item():
                embeds_list.append(uncond_embeds[i][uncond_mask[i].bool()])
            else:
                embeds_list.append(prompt_embeds[i][attn_mask[i].bool()])
        prompt_embeds, attn_mask = moss_image_utils.pad_text_embeds(embeds_list)
        return [prompt_embeds, attn_mask]


class MossImageTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """Caches the text encoder outputs as bf16 bits (uint16) without padding, together with the unconditional
    embeddings (the encoding of "") so that caption dropout works without the text encoder.

    All caption candidates of an image (``enable_multiline_caption`` with separator lines) are cached in one file:

    - ``num_candidates``: int32 scalar
    - ``prompt_embeds_{k}`` uint16 [L_k, C], ``attn_mask_{k}`` int32 [L_k], ``caption_{k}`` str (the processed caption)
      for k in range(num_candidates)
    - ``uncond_embeds`` uint16 [L_u, C], ``uncond_mask`` int32 [L_u], ``caption_dropout_rate`` float32 scalar

    The caption texts are used to detect changed captions (the file is re-cached) and for debugging.
    Files of the previous format (sd-scripts 0.12.1: ``prompt_embeds`` / ``attn_mask`` without ``num_candidates``)
    are still accepted for images with a single candidate.
    """

    MOSS_IMAGE_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_moss_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: Optional[int],
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        identity=None,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)
        self._uncond_embeds: Optional[torch.Tensor] = None  # [L_u, C] bf16 on CPU
        self.identity = identity or {}
        self.expected = {}

    def prepare_info(self, info, subset, dataset):
        geometry = moss_caption.center_geometry(info) if moss_caption.is_layout(info.caption) else None
        info.moss_cache_captions = dataset.process_caption_candidates_for_cache(subset, info.caption, geometry)
        info.moss_cache_caption = info.moss_cache_captions[0]
        settings = {
            "encoding": self.identity,
            "format": getattr(subset, "caption_format", "text"),
            "length": getattr(subset, "moss_caption_length", "long"),
            "geometry": geometry,
            "dropout": getattr(info, "caption_dropout_rate", 0.0),
        }
        captions = info.moss_cache_captions
        info.moss_cache_fingerprint = moss_caption.fingerprint(captions[0] if len(captions) == 1 else captions, settings)
        self.expected[self.get_outputs_npz_path(info.absolute_path)] = info.moss_cache_fingerprint

    @property
    def supports_multiple_caption_candidates(self) -> bool:
        return True

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + self.MOSS_IMAGE_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str, captions: Optional[List[str]] = None) -> bool:
        """``captions`` are the processed caption candidates of the image. If given, the cache is valid only if the number
        of candidates and their texts match (not checked with ``skip_disk_cache_validity_check``)."""
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        try:
            with np.load(npz_path, allow_pickle=False) as npz:
                if int(npz["moss_cache_version"]) != moss_image_cache.CACHE_VERSION:
                    return False
                count = int(npz["num_candidates"]) if "num_candidates" in npz else 1
                if count < 1:
                    return False
                check_captions = captions is not None and not self.skip_disk_cache_validity_check
                if check_captions and count != len(captions):
                    return False
                pairs = [("uncond_embeds", "uncond_mask")]
                if "num_candidates" not in npz:
                    pairs.append(("prompt_embeds", "attn_mask"))
                else:
                    for k in range(count):
                        pairs.append((f"prompt_embeds_{k}", f"attn_mask_{k}"))
                        if f"caption_{k}" not in npz or (check_captions and str(npz[f"caption_{k}"]) != captions[k]):
                            return False
                for embed, mask in pairs:
                    if npz[embed].dtype != np.uint16 or npz[embed].ndim != 2 or npz[mask].shape != npz[embed].shape[:1]:
                        return False
                if "caption_dropout_rate" not in npz or "moss_fingerprint" not in npz:
                    return False
                expected = self.expected.get(npz_path)
                if expected is not None and not self.skip_disk_cache_validity_check:
                    return str(npz["moss_fingerprint"].item()) == expected
        except (ValueError, KeyError, OSError, EOFError):
            return False

        return True

    def load_outputs_npz(self, npz_path: str, candidate_index: int = 0) -> List[Any]:
        """Returns [prompt_embeds BF16 [L, C], attn_mask int32 [L], uncond_embeds BF16 [L_u, C], uncond_mask int32 [L_u],
        caption_dropout_rate float32 scalar] of the caption candidate ``candidate_index``.
        The collate function right-pads the variable length arrays with zeros."""
        with np.load(npz_path, allow_pickle=False) as data:
            if "num_candidates" in data:
                num_candidates = int(data["num_candidates"])
                embeds_key, mask_key = f"prompt_embeds_{candidate_index}", f"attn_mask_{candidate_index}"
            else:
                num_candidates = 1
                embeds_key, mask_key = "prompt_embeds", "attn_mask"
            if not 0 <= candidate_index < num_candidates:
                raise ValueError(
                    f"The cached Text Encoder outputs have {num_candidates} caption candidate(s), but candidate {candidate_index} is requested: {npz_path}."
                    " Remove --skip_cache_check to re-cache changed captions."
                )
            return [
                uint16_to_bf16(data[embeds_key]),
                data[mask_key],
                uint16_to_bf16(data["uncond_embeds"]),
                data["uncond_mask"],
                data["caption_dropout_rate"],
            ]

    def get_uncond_embeds(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy
    ) -> torch.Tensor:
        """Encode "" once (the official negative prompt) and keep it for the following batches."""
        if self._uncond_embeds is None:
            with torch.no_grad():
                tokens = tokenize_strategy.tokenize("")
                embeds, mask = text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens)
            self._uncond_embeds = embeds[0][mask[0].bool()].to(torch.bfloat16).cpu()
            logger.info(f'Encoded the unconditional prompt (""): {self._uncond_embeds.shape[0]} tokens')
        return self._uncond_embeds

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List,
        captions: List[str],
    ):
        self.cache_batch_outputs_multi(tokenize_strategy, models, text_encoding_strategy, infos, [[c] for c in captions])

    def cache_batch_outputs_multi(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List,
        captions_list: List[List[str]],
    ):
        uncond_embeds = self.get_uncond_embeds(tokenize_strategy, models, text_encoding_strategy)
        uncond_np = bf16_to_uint16(uncond_embeds)
        uncond_mask_np = np.ones(uncond_embeds.shape[0], dtype=np.int32)

        # encode all candidates of the batch at once (the text encoder runs one caption at a time anyway)
        flat_captions = [caption for captions in captions_list for caption in captions]
        tokens = tokenize_strategy.tokenize(flat_captions)
        with torch.no_grad():
            prompt_embeds, attn_mask = text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens)

        offset = 0
        for info, captions in zip(infos, captions_list):
            embeds_list = []
            mask_list = []
            for k in range(len(captions)):
                length = int(attn_mask[offset + k].sum().item())
                embeds_list.append(bf16_to_uint16(prompt_embeds[offset + k, :length]))  # trimmed, no padding
                mask_list.append(np.ones(length, dtype=np.int32))
            offset += len(captions)
            caption_dropout_rate = np.array(info.caption_dropout_rate, dtype=np.float32)

            if self.cache_to_disk:
                arrays = {
                    "num_candidates": np.array(len(captions), dtype=np.int32),
                    "uncond_embeds": uncond_np,
                    "uncond_mask": uncond_mask_np,
                    "caption_dropout_rate": caption_dropout_rate,
                    "moss_cache_version": moss_image_cache.CACHE_VERSION,
                    "moss_fingerprint": getattr(info, "moss_cache_fingerprint", moss_caption.fingerprint(
                        captions[0] if len(captions) == 1 else captions, self.identity
                    )),
                }
                for k, (embeds_k, mask_k, caption_k) in enumerate(zip(embeds_list, mask_list, captions)):
                    arrays[f"prompt_embeds_{k}"] = embeds_k
                    arrays[f"attn_mask_{k}"] = mask_k
                    arrays[f"caption_{k}"] = np.array(caption_k)
                moss_image_cache.write_cache(info.text_encoder_outputs_npz, arrays)
            else:
                # Own trimmed storage rather than retaining a padded encoder batch.
                info.text_encoder_outputs = [
                    (
                        uint16_to_bf16(embeds_k).clone(),
                        mask_k,
                        uint16_to_bf16(uncond_np).clone(),
                        uncond_mask_np,
                        caption_dropout_rate,
                    )
                    for embeds_k, mask_k in zip(embeds_list, mask_list)
                ]


class MossImageLatentsCachingStrategy(LatentsCachingStrategy):
    """Caches the raw FLUX.2 VAE latents [32, H/8, W/8]. Patchify and normalization are applied in the trainer."""

    MOSS_IMAGE_LATENTS_NPZ_SUFFIX = "_moss.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool, identity=None) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)
        self.identity = identity
        self.expected = {}

    @property
    def cache_suffix(self) -> str:
        return self.MOSS_IMAGE_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        path = os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.MOSS_IMAGE_LATENTS_NPZ_SUFFIX
        if Path(absolute_path).is_file():
            stat = Path(absolute_path).stat()
            self.expected[path] = moss_caption.fingerprint(self.identity, [image_size, stat.st_size, stat.st_mtime_ns])
        return path

    def get_image_size_from_disk_cache_path(self, absolute_path, npz_path):
        with Image.open(absolute_path) as image:
            return image.size

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        try:
            if not self._default_is_disk_cached_latents_expected(
                flux2_vae.LATENT_SCALE_FACTOR, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True
            ):
                return False
            suffix = f"_{bucket_reso[1] // 8}x{bucket_reso[0] // 8}"
            with np.load(npz_path, allow_pickle=False) as data:
                if int(data["moss_cache_version"]) != moss_image_cache.CACHE_VERSION:
                    return False
                if data["latents" + suffix].shape != (32, bucket_reso[1] // 8, bucket_reso[0] // 8):
                    return False
                expected = self.expected.get(npz_path)
                return expected is None or str(data["moss_fingerprint"].item()) == expected
        except (ValueError, KeyError, OSError, EOFError):
            return False

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._default_load_latents_from_disk(flux2_vae.LATENT_SCALE_FACTOR, npz_path, bucket_reso)

    def cache_batch_latents(self, vae: flux2_vae.AutoEncoder, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        vae_device = vae.device
        vae_dtype = vae.dtype

        def encode_by_vae(img_tensor):
            # no_grad is handled in _default_cache_batch_latents. Run outside autocast: the VAE is in float32.
            with torch.autocast(device_type=vae_device.type, enabled=False):
                return vae.encode(img_tensor.to(vae_dtype))

        if self.cache_to_disk:
            for info in image_infos:
                moss_image_cache.backup_cache(info.latents_npz)
        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )

        if self.cache_to_disk:
            for info in image_infos:
                with np.load(info.latents_npz, allow_pickle=False) as data:
                    values = dict(data)
                values.update(moss_cache_version=moss_image_cache.CACHE_VERSION, moss_fingerprint=self.expected.get(info.latents_npz, ""))
                moss_image_cache.write_cache(info.latents_npz, values, backup=False)

        if not accelerator_setup.HIGH_VRAM:
            device_utils.clean_memory_on_device(vae_device)
