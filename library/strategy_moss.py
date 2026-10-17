"""Frozen Qwen3.5 and FLUX.2 strategies for dense MOSS Image."""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from PIL import Image

from library import moss_caption
from library.strategy_base import (
    LatentsCachingStrategy, TextEncoderOutputsCachingStrategy, TextEncodingStrategy, TokenizeStrategy,
)


def component_signature(path):
    path = Path(path)
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file() and file.suffix in (".json", ".safetensors", ".model", ".jinja"):
            stat = file.stat()
            digest.update(str((str(file.relative_to(path)), stat.st_size, stat.st_mtime_ns)).encode())
            if file.suffix in (".json", ".jinja"):
                digest.update(file.read_bytes())
    return digest.hexdigest()


def encode_latents(vae, images):
    z = vae.encode(images).latent_dist.sample()
    b, c, h, w = z.shape
    z = z.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    mean = vae.bn.running_mean.to(z).view(1, -1, 1, 1)
    std = (vae.bn.running_var.float() + vae.config.batch_norm_eps).sqrt().to(z).view(1, -1, 1, 1)
    return (z - mean) / std


class MossTokenizeStrategy(TokenizeStrategy):
    def __init__(self, model_path, max_length=4096, tokenizer=None):
        if max_length < 1:
            raise ValueError("moss_max_token_length must be positive")
        root = Path(model_path)
        token_path = root / "tokenizer" if (root / "tokenizer").is_dir() else root / "text_encoder"
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(token_path, local_files_only=True)
        self.max_length = max_length
        self.cache_identity = {
            "version": 1, "max_length": max_length, "template": "user-generation-open-think-last-hidden-unpadded",
            "tokenizer": component_signature(token_path), "encoder": component_signature(root / "text_encoder"),
        }

    def tokenize(self, text):
        texts = [text] if isinstance(text, str) else text
        formatted = [self.tokenizer.apply_chat_template(
            [{"role": "user", "content": value}], tokenize=False, add_generation_prompt=True, enable_thinking=True
        ) for value in texts]
        tokens = self.tokenizer(formatted, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length)
        return [tokens["input_ids"], tokens["attention_mask"]]


class MossTextEncodingStrategy(TextEncodingStrategy):
    def encode_tokens(self, tokenize_strategy, models, tokens):
        model = models[0]
        ids, masks = tokens
        rows = []
        for row, mask in zip(ids, masks):
            # Qwen's gated delta network is evaluated without padding, as in the official pipeline.
            selected = row[mask.bool()].unsqueeze(0).to(model.device)
            rows.append(model(input_ids=selected, attention_mask=torch.ones_like(selected), use_cache=False).last_hidden_state[0])
        length = max(row.shape[0] for row in rows)
        embeds = rows[0].new_zeros((len(rows), length, rows[0].shape[-1]))
        attention = torch.zeros((len(rows), length), dtype=torch.long, device=embeds.device)
        for i, row in enumerate(rows):
            embeds[i, :len(row)] = row
            attention[i, :len(row)] = 1
        return [embeds, attention]


class MossTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    def __init__(self, cache_to_disk, batch_size, identity):
        super().__init__(cache_to_disk, batch_size, False)
        self.identity = identity
        self.expected = {}

    def get_outputs_npz_path(self, image_abs_path):
        return os.path.splitext(image_abs_path)[0] + "_moss_te.npz"

    def prepare_info(self, info, subset, dataset):
        geometry = moss_caption.center_geometry(info) if moss_caption.is_layout(info.caption) else None
        info.moss_cache_caption = dataset.process_caption(subset, info.caption, geometry)
        settings = {
            "encoding": self.identity, "format": subset.caption_format, "length": subset.moss_caption_length,
            "geometry": geometry,
        }
        info.moss_cache_fingerprint = moss_caption.fingerprint(info.moss_cache_caption, settings)
        self.expected[self.get_outputs_npz_path(info.absolute_path)] = info.moss_cache_fingerprint

    def is_disk_cached_outputs_expected(self, npz_path):
        if not self.cache_to_disk or not os.path.isfile(npz_path):
            return False
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                return (str(data["fingerprint"].item()) == self.expected.get(npz_path)
                        and data["prompt_embeds"].ndim == 2
                        and data["attn_mask"].shape == data["prompt_embeds"].shape[:1])
        except (ValueError, KeyError, OSError, EOFError):
            return False

    def load_outputs_npz(self, npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            return [data["prompt_embeds"], data["attn_mask"]]

    def cache_batch_outputs(self, tokenize_strategy, models, text_encoding_strategy, infos):
        tokens = tokenize_strategy.tokenize([info.moss_cache_caption for info in infos])
        with torch.no_grad():
            embeds, masks = text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens)
        for info, embed, mask in zip(infos, embeds, masks):
            count = int(mask.sum())
            values = [embed[:count].float().cpu().numpy(), mask[:count].cpu().numpy()]
            if self.cache_to_disk:
                np.savez(info.text_encoder_outputs_npz, prompt_embeds=values[0], attn_mask=values[1], fingerprint=info.moss_cache_fingerprint)
            else:
                info.text_encoder_outputs = values


class MossLatentsCachingStrategy(LatentsCachingStrategy):
    latent_stride = 16

    def __init__(self, cache_to_disk, batch_size, identity):
        super().__init__(cache_to_disk, batch_size, False)
        self.identity = identity
        self.expected = {}

    @property
    def cache_suffix(self):
        return "_moss.npz"

    def get_latents_npz_path(self, absolute_path, image_size):
        path = os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.cache_suffix
        stat = Path(absolute_path).stat()
        self.expected[path] = moss_caption.fingerprint(self.identity, [image_size, stat.st_size, stat.st_mtime_ns])
        return path

    def is_disk_cached_latents_expected(self, bucket_reso, npz_path, flip_aug, alpha_mask):
        try:
            if not self._default_is_disk_cached_latents_expected(16, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True):
                return False
            suffix = f"_{bucket_reso[1] // 16}x{bucket_reso[0] // 16}"
            with np.load(npz_path, allow_pickle=False) as data:
                return (str(data["moss_fingerprint"].item()) == self.expected.get(npz_path)
                        and data["latents" + suffix].shape == (128, bucket_reso[1] // 16, bucket_reso[0] // 16))
        except (ValueError, KeyError, OSError, EOFError):
            return False

    def load_latents_from_disk(self, npz_path, bucket_reso):
        return self._default_load_latents_from_disk(16, npz_path, bucket_reso)

    def get_image_size_from_disk_cache_path(self, absolute_path, npz_path):
        # A changed image must not inherit dimensions embedded in an older cache filename.
        with Image.open(absolute_path) as image:
            return image.size

    def cache_batch_latents(self, vae, infos, flip_aug, alpha_mask, random_crop):
        self._default_cache_batch_latents(lambda images: encode_latents(vae, images), vae.device, vae.dtype,
                                          infos, flip_aug, alpha_mask, random_crop, multi_resolution=True)
        if self.cache_to_disk:
            for info in infos:
                with np.load(info.latents_npz, allow_pickle=False) as data:
                    values = dict(data)
                np.savez(info.latents_npz, **values, moss_fingerprint=self.expected[info.latents_npz])
