# MOSS-Image training / sampling utilities
# The sampler follows the official `MOSSImageScheduler` (pipeline_moss_image.py): the shifted inference grid is snapped
# to the nearest sigma of the training grid (shift 1, 1000 steps) and integrated with Euler steps in float32.

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import gc
import os
import time
from typing import Optional

import torch
from accelerate import Accelerator, PartialState
from PIL import Image
from tqdm import tqdm

from library import (
    device_utils,
    flux2_vae,
    flux_train_utils,
    moss_caption,
    moss_image_models,
    moss_image_utils,
    sampling,
)
from library.device_utils import clean_memory_on_device, synchronize_device
import library.compile_utils as compile_utils
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

NUM_TRAIN_TIMESTEPS = 1000
DEFAULT_UNCOND_SKIP_LAYERS = (
    22,
)  # block skipped in the negative branch of CFG by the official pipeline

# defaults of the official pipeline, used for sample image generation
DEFAULT_SAMPLE_STEPS = 50
DEFAULT_SAMPLE_GUIDANCE_SCALE = 3.0
DEFAULT_SAMPLE_FLOW_SHIFT = 5.0
DEFAULT_SAMPLE_SIZE = 1024


# region arguments


def add_moss_image_model_arguments(parser: argparse.ArgumentParser):
    """Arguments to load the MOSS-Image models (shared by the trainer and the caching tools). `--vae` and
    `--tokenizer_cache_dir` are registered by `add_sd_models_arguments`."""
    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="path to the Qwen3.5-9B text encoder checkpoint: the first shard of the diffusers format (model-00001-of-00004.safetensors) or the ComfyUI single file"
        + " / Qwen3.5-9B Text Encoder のチェックポイント。diffusers 形式なら最初の分割ファイル（model-00001-of-00004.safetensors）、ComfyUI 形式なら単一ファイル",
    )
    parser.add_argument(
        "--moss_max_token_length",
        type=int,
        default=moss_image_utils.MAX_TOKEN_LENGTH,
        help=f"maximum number of text tokens (including the chat template), default and maximum: {moss_image_utils.MAX_TOKEN_LENGTH}"
        + f" / テキストトークンの最大数（chat template を含む）。デフォルトおよび最大値: {moss_image_utils.MAX_TOKEN_LENGTH}",
    )


def add_moss_image_training_arguments(parser: argparse.ArgumentParser):
    """MOSS-Image specific training arguments (timestep sampling, attention, compile, sampling, checkpointing)."""
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=1.0,
        help="Timestep distribution shift for rectified flow training, used by --timestep_sampling sigma / shift (default: 1.0, the training grid of MOSS-Image)"
        + " / rectified flow 学習のタイムステップ分布のシフト。--timestep_sampling sigma / shift で使われる（デフォルト: 1.0、MOSS-Image の学習グリッド）",
    )
    parser.add_argument(
        "--timestep_sampling",
        type=str,
        default="sigmoid",
        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift", "legacy"],
        help="Timestep sampling method (default: sigmoid (logit normal)). The pre-training distribution of MOSS-Image is reproduced by"
        + " `--timestep_sampling sigma --weighting_scheme logit_normal --logit_mean -1.09 --logit_std 0.95 --discrete_flow_shift 1.0`"
        + " / タイムステップのサンプリング方法（デフォルト: sigmoid（logit normal））。MOSS-Image の事前学習の分布は"
        + " `--timestep_sampling sigma --weighting_scheme logit_normal --logit_mean -1.09 --logit_std 0.95 --discrete_flow_shift 1.0` で再現できる",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help="Scale factor for sigmoid (logit_normal) timestep sampling (default: 1.0) / sigmoid（logit normal）タイムステップサンプリングのスケール（デフォルト: 1.0）",
    )
    parser.add_argument(
        "--attn_mode",
        choices=[
            "torch",
            "xformers",
            "flash",
            "sageattn",
            "sdpa",
        ],  # "sdpa" is for backward compatibility
        default=None,
        help="Attention implementation to use. Default is None (torch). xformers requires --split_attn. sageattn does not support training (inference only). This option overrides --xformers or --sdpa."
        + " / 使用するAttentionの実装。デフォルトはNone（torch）です。xformersは--split_attnの指定が必要です。sageattnはトレーニングをサポートしていません（推論のみ）。このオプションは--xformersまたは--sdpaを上書きします。",
    )
    parser.add_argument(
        "--split_attn",
        action="store_true",
        help="split attention computation to reduce memory usage / メモリ使用量を減らすためにattention時にバッチを分割する",
    )
    parser.add_argument(
        "--sample_uncond_skip_layers",
        type=str,
        default="22",
        help="comma separated DiT block indices skipped in the negative (unconditional) branch of CFG in sample image generation. Default is 22 (official). Empty string disables skipping"
        + " / サンプル画像生成で、CFG の negative（unconditional）側で飛ばす DiT ブロックのインデックス（カンマ区切り）。デフォルトは 22（公式）。空文字列で無効",
    )
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap."
        + " / 非同期転送で activation を CPU RAM に退避する（--cpu_offload_checkpointing より高速）。--cpu_offload_checkpointing および --blocks_to_swap とは併用できない",
    )

    # torch.compile (per-block compilation, Triton required) and CUDA performance switches
    compile_utils.add_compile_arguments(parser)


# endregion


# region timestep visualization


def show_timesteps(args):
    """Visualize the sampled timestep / loss weighting distribution for the current settings, then return.

    MOSS-Image reuses ``flux_train_utils.get_noisy_model_input_and_timesteps`` for sampling and
    ``compute_loss_weighting_for_sd3`` for the weighting (1 for ``logit_normal``, which matches the official training).
    """
    from library import sd3_train_utils, timestep_visualization

    legacy = args.timestep_sampling == "legacy"
    noise_scheduler = (
        load_training_schedule(moss_image_utils.resolve_model_paths(args).root)
        if legacy
        else sd3_train_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS, shift=args.discrete_flow_shift
        )
    )
    h, w = flux_train_utils.parse_show_timesteps_latent_size(
        args, vae_compression=16
    )  # 2x2 patchified latents
    device, dtype = device_utils.get_preferred_device(), torch.float32
    offset, offset_note = flux_train_utils.get_show_timesteps_offset(args)

    def sample_timesteps(bsz):
        latents = torch.zeros(
            bsz,
            moss_image_models.MOSS_IMAGE_CONFIG["latent_channels"],
            h,
            w,
            dtype=dtype,
            device=device,
        )
        noise = torch.ones_like(latents)
        tso = (
            None
            if offset is None
            else torch.full((bsz,), offset, dtype=dtype, device=device)
        )
        if legacy:
            _, _, timesteps = flow_batch(
                latents, noise_scheduler, args.min_timestep or 0, args.max_timestep
            )
        else:
            _, timesteps, _ = flux_train_utils.get_noisy_model_input_and_timesteps(
                args,
                noise_scheduler,
                latents,
                noise,
                device,
                dtype,
                timestep_sampling_offset=tso,
            )
        return timesteps

    def compute_weighting(timesteps):
        if legacy:
            return torch.ones_like(timesteps)
        sigmas = timesteps / NUM_TRAIN_TIMESTEPS
        return flux_train_utils.compute_loss_weighting_for_sd3(
            args.weighting_scheme, sigmas
        )

    header = (
        "Timestep distribution / タイムステップ分布:\n  "
        + flux_train_utils.get_timestep_sampling_info(args)
        + offset_note
        + f", resolution={args.show_timesteps_resolution} (latent {h}x{w})"
    )
    timestep_visualization.show_timestep_distribution(
        args.show_timesteps,
        sample_timesteps,
        compute_weighting,
        num_train_timesteps=noise_scheduler.config.num_train_timesteps,
        header=header,
    )


# endregion


# region sampler


def _sigma_grid(count: int, shift: float, total: int) -> torch.Tensor:
    lower = shift / (total + shift - 1)
    values = torch.linspace(1, lower, count, dtype=torch.float32)
    return shift * values / (1 + (shift - 1) * values)


def get_sampling_sigmas(
    num_inference_steps: int,
    shift: float = 5.0,
    training_shift: float = 1.0,
    num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
) -> torch.Tensor:
    """Return the sampling sigmas [steps + 1] (descending, the last value is 0) on the CPU in float32.

    The number of effective steps can be smaller than `num_inference_steps` because the shifted grid is snapped to the
    training grid and duplicates are removed (same as the official scheduler).
    """
    if num_inference_steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError("Steps and shifts must be positive.")
    total = num_train_timesteps
    sigmas = _sigma_grid(num_inference_steps, shift, total)
    training = _sigma_grid(total, training_shift, total).flip(0)  # ascending
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where(
        (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(),
        lower,
        upper,
    )
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def euler_step(
    latents: torch.Tensor,
    model_pred: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    """One Euler step in float32: x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v"""
    dt = (sigmas[step_index + 1] - sigmas[step_index]).to(torch.float32)
    return latents.float() + dt * model_pred.float()


def parse_skip_layers(value: Optional[str]) -> tuple[int, ...]:
    """Parse a comma separated list of block indices. None or an empty string disables skipping."""
    if value is None:
        return ()
    value = value.strip()
    if not value:
        return ()
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def load_training_schedule(root=None):
    values = {}
    if root is not None:
        path = Path(root) / "scheduler" / "scheduler_config.json"
        if path.is_file():
            values = json.loads(path.read_text())
    config = SimpleNamespace(
        num_train_timesteps=values.get("num_train_timesteps", NUM_TRAIN_TIMESTEPS),
        training_shift=values.get("training_shift", 1.0),
        shift=values.get("shift", 5.0),
    )
    if (
        config.num_train_timesteps < 1
        or config.training_shift <= 0
        or config.shift <= 0
    ):
        raise ValueError("MOSS timestep count and shifts must be positive")
    return SimpleNamespace(config=config)


def flow_batch(latents, scheduler, min_timestep=0, max_timestep=None):
    total = scheduler.config.num_train_timesteps
    maximum = total if max_timestep is None else max_timestep
    if not 0 <= min_timestep <= maximum <= total or min_timestep == total:
        raise ValueError(
            "Require 0 <= min_timestep <= max_timestep <= num_train_timesteps, with min_timestep < total"
        )
    grid = (
        _sigma_grid(total, scheduler.config.training_shift, total)
        .to(latents.device)
        .flip(0)
    )
    if min_timestep == maximum:
        # The common validation loop fixes min/max to one timestep and reuses the batch.
        indices = torch.full(
            (latents.shape[0],), min_timestep, device=latents.device, dtype=torch.long
        )
    else:
        indices = torch.randint(
            min_timestep, maximum, (latents.shape[0],), device=latents.device
        )
    sigmas = grid[indices]
    noise = torch.randn(latents.shape, device=latents.device, dtype=torch.float32)
    sigma = sigmas.view(-1, 1, 1, 1)
    clean = latents.float()
    return (1 - sigma) * clean + sigma * noise, noise - clean, sigmas * total


def validate_datasets(groups, *, cache_text=False, block_swap=False):
    """The trainer and pre-cache tools share caption/layout cache constraints."""
    for group in groups:
        if group is None:
            continue
        group.verify_bucket_reso_steps(16)
        for dataset in group.datasets:
            if block_swap and dataset.batch_size != 1:
                raise ValueError(
                    "MOSS block swapping requires batch_size=1; use gradient_accumulation_steps"
                )
            for info in dataset.image_data.values():
                subset = dataset.image_to_subset[info.image_key]
                layout = subset.caption_format != "text" and moss_caption.is_layout(
                    info.caption
                )
                if layout and subset.face_crop_aug_range is not None:
                    raise ValueError(
                        f"{info.absolute_path}: use bucket/random cropping for layout data, not face cropping"
                    )
                dynamic = (
                    (
                        not layout
                        and (
                            subset.shuffle_caption
                            or subset.enable_wildcard
                            or subset.token_warmup_step
                            or subset.caption_tag_dropout_rate
                        )
                    )
                    or subset.caption_dropout_every_n_epochs
                    or (layout and (subset.random_crop or subset.flip_aug))
                )
                if cache_text and dynamic:
                    raise ValueError(
                        f"{info.absolute_path}: stochastic captions/layout geometry require disabling text output caching"
                    )
        if cache_text and not group.is_text_encoder_output_cacheable(
            cache_supports_dropout=True, supports_multiple_caption_candidates=True
        ):
            raise ValueError(
                "MOSS text output caching requires deterministic caption candidates"
            )


def generate_latents(
    dit: moss_image_models.MossImage,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    negative_embeds: Optional[torch.Tensor],
    negative_mask: Optional[torch.Tensor],
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    flow_shift: float,
    uncond_skip_layers: tuple[int, ...],
    seed: Optional[int],
    device: torch.device,
    show_progress: bool = True,
    training_shift: float = 1.0,
    num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
) -> torch.Tensor:
    """Sample normalized, patchified latents [1, 128, H/16, W/16] (float32).

    The initial noise is drawn on a CPU generator in float32 and CFG / Euler updates are done in float32 like the official
    pipeline. CFG is applied when `guidance_scale != 1` and the negative embeddings are given.
    """
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    shape = (1, dit.latent_channels, height // 16, width // 16)
    latents = torch.randn(shape, generator=generator, dtype=torch.float32).to(device)

    positive = prompt_embeds.to(device, dtype=dit.dtype)
    positive_mask = prompt_mask.to(device)
    do_cfg = guidance_scale != 1.0 and negative_embeds is not None
    if do_cfg:
        negative = negative_embeds.to(device, dtype=dit.dtype)
        negative_mask = negative_mask.to(device)

    sigmas = get_sampling_sigmas(
        steps, flow_shift, training_shift, num_train_timesteps
    ).to(device)
    num_steps = len(sigmas) - 1

    with (
        torch.no_grad(),
        tqdm(
            total=num_steps, desc="Denoising steps", disable=not show_progress
        ) as pbar,
    ):
        for i in range(num_steps):
            timestep = (sigmas[i] * num_train_timesteps).expand(latents.shape[0])
            x = latents.to(dit.dtype)
            dit.prepare_block_swap_before_forward()
            prediction = dit(x, timestep, positive, positive_mask).float()
            if do_cfg:
                dit.prepare_block_swap_before_forward()
                uncond = dit(
                    x, timestep, negative, negative_mask, skip_layers=uncond_skip_layers
                ).float()
                prediction = uncond + guidance_scale * (prediction - uncond)
            latents = euler_step(latents, prediction, sigmas, i)
            pbar.update()

    return latents


# endregion


# region sample images during training


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    dit: moss_image_models.MossImage,
    vae: flux2_vae.AutoEncoder,
    text_encoder,
    tokenize_strategy,
    text_encoding_strategy,
    sample_prompts_te_outputs=None,
    prompt_replacement=None,
):
    """Sample with the shared inference loop, then restore training state even on errors."""
    if not args.sample_prompts:
        return
    if steps == 0:
        if not args.sample_at_first:
            return
    elif args.sample_every_n_epochs is not None:
        if epoch is None or epoch % args.sample_every_n_epochs:
            return
    elif (
        args.sample_every_n_steps is None
        or epoch is not None
        or steps % args.sample_every_n_steps
    ):
        return
    if not os.path.isfile(args.sample_prompts):
        logger.error(f"No prompt file: {args.sample_prompts}")
        return

    dit = accelerator.unwrap_model(dit)
    if text_encoder is not None:
        text_encoder = accelerator.unwrap_model(text_encoder)
    prompts = sampling.load_prompts(args.sample_prompts)
    state = PartialState()
    prompts = prompts[state.process_index :: state.num_processes]
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)
    rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    dit_training = dit.training
    te_training = text_encoder.training if text_encoder is not None else False
    vae_device = vae.device
    try:
        dit.eval()
        if text_encoder is not None:
            text_encoder.eval()
        dit.switch_block_swap_for_inference()
        with torch.no_grad(), accelerator.autocast():
            for prompt in prompts:
                _sample_image_inference(
                    accelerator,
                    args,
                    dit,
                    text_encoder,
                    vae,
                    tokenize_strategy,
                    text_encoding_strategy,
                    save_dir,
                    prompt,
                    epoch,
                    steps,
                    sample_prompts_te_outputs,
                    prompt_replacement,
                )
    finally:
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        vae.to(vae_device)
        if text_encoder is not None:
            text_encoder.train(te_training)
        dit.train(dit_training)
        dit.switch_block_swap_for_training()
        dit.prepare_block_swap_before_forward()
        clean_memory_on_device(accelerator.device)


def _sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    dit: moss_image_models.MossImage,
    text_encoder,
    vae: flux2_vae.AutoEncoder,
    tokenize_strategy,
    text_encoding_strategy,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
):
    """Generate a single sample image."""
    prompt = prompt_dict.get("prompt", "")
    negative_prompt = prompt_dict.get(
        "negative_prompt", ""
    )  # "" is the official default negative prompt
    sample_steps = prompt_dict.get(
        "sample_steps", getattr(args, "moss_sample_steps", DEFAULT_SAMPLE_STEPS)
    )
    width = prompt_dict.get("width", DEFAULT_SAMPLE_SIZE)
    height = prompt_dict.get("height", DEFAULT_SAMPLE_SIZE)
    scale = prompt_dict.get("scale", DEFAULT_SAMPLE_GUIDANCE_SCALE)
    seed = prompt_dict.get("seed", getattr(args, "moss_sample_seed", None))
    schedule = load_training_schedule(
        moss_image_utils.resolve_model_paths(args).root
        if args.timestep_sampling == "legacy"
        else None
    ).config
    flow_shift = float(prompt_dict.get("flow_shift", schedule.shift))
    uncond_skip_layers = parse_skip_layers(args.sample_uncond_skip_layers)

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt:
            negative_prompt = negative_prompt.replace(
                prompt_replacement[0], prompt_replacement[1]
            )

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # seed all CUDA devices for multi-GPU

    if min(width, height) <= 0 or width % 16 or height % 16:
        raise ValueError("MOSS sample width/height must be positive multiples of 16")

    logger.info(
        f"  prompt: {prompt}, negative: {negative_prompt}, size: {width}x{height}, steps: {sample_steps}, scale: {scale}, "
        f"flow_shift: {flow_shift}, uncond skip layers: {uncond_skip_layers}, seed: {seed}"
    )

    # Encode prompt: [prompt_embeds [1, L, C], attn_mask [1, L]]
    def encode_prompt(prpt):
        if sample_prompts_te_outputs and prpt in sample_prompts_te_outputs:
            return sample_prompts_te_outputs[prpt]
        if text_encoder is not None:
            tokens = tokenize_strategy.tokenize(prpt)
            return text_encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens
            )
        return None

    encoded = encode_prompt(prompt)
    if encoded is None:
        logger.warning("Cannot encode prompt, skipping sample")
        return
    prompt_embeds, prompt_mask = encoded[0], encoded[1]

    negative_embeds = negative_mask = None
    if scale != 1.0:
        neg_encoded = encode_prompt(negative_prompt)
        if neg_encoded is None:
            logger.warning(
                "Cannot encode the negative prompt, CFG is disabled for this sample"
            )
        else:
            negative_embeds, negative_mask = neg_encoded[0], neg_encoded[1]

    clean_memory_on_device(accelerator.device)
    latents = generate_latents(
        dit,
        prompt_embeds,
        prompt_mask,
        negative_embeds,
        negative_mask,
        height,
        width,
        sample_steps,
        scale,
        flow_shift,
        uncond_skip_layers,
        seed,
        accelerator.device,
        training_shift=schedule.training_shift,
        num_train_timesteps=schedule.num_train_timesteps,
    )

    # Decode latents in float32 outside autocast (the VAE is in float32)
    gc.collect()
    synchronize_device(accelerator.device)
    clean_memory_on_device(accelerator.device)
    org_vae_device = vae.device
    vae.to(accelerator.device)
    with torch.autocast(device_type=accelerator.device.type, enabled=False):
        pixels = vae.decode_from_dit(latents.to(accelerator.device, dtype=vae.dtype))
    vae.to(org_vae_device)
    clean_memory_on_device(accelerator.device)

    pixels = (
        ((pixels[0].float().clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )
    image = Image.fromarray(pixels)

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i = prompt_dict.get("enum", 0)
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    # Log to wandb if enabled
    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")
        import wandb

        wandb_tracker.log(
            {f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False
        )


# endregion
