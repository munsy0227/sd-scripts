"""Dense MOSS Image LoRA/LoKR/DoRA training with frozen Qwen3.5 and FLUX.2 VAE."""

from pathlib import Path

import torch
from torch import nn
from diffusers import AutoencoderKLFlux2
from transformers import AutoModel
from PIL import Image

import train_network
import library.args as args_util
from library import moss_caption, moss_models, moss_train_utils, sampling, strategy_base, strategy_moss
from library.device_utils import clean_memory_on_device


class _CachedTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("marker", torch.empty(0))

    @property
    def device(self):
        return self.marker.device


class MossNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_conditions = {}

    def train(self, args):
        args.network_train_unet_only = True
        if args.network_train_text_encoder_only:
            raise ValueError("MOSS keeps Qwen frozen; text-encoder adapter training is not supported")
        if args.in_json is not None or args.dataset_class is not None:
            raise ValueError("Use image directories with TXT/JSON caption sidecars for MOSS")
        if args.network_module != "lycoris.kohya":
            raise ValueError("MOSS uses --network_module lycoris.kohya (algo=lora or lokr)")
        network_args = dict(value.split("=", 1) for value in (args.network_args or []))
        network_args.setdefault("preset", "moss")
        if network_args["preset"] not in ("moss", "moss-full") or network_args.get("algo", "lora") not in ("lora", "lokr"):
            raise ValueError("Use preset=moss/moss-full and algo=lora/lokr")
        args.network_args = [f"{key}={value}" for key, value in network_args.items()]
        if args.cache_latents_to_disk:
            args.cache_latents = True
        return super().train(args)

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        if args.cache_text_encoder_outputs_to_disk:
            args.cache_text_encoder_outputs = True
        if args.fp8_base or args.fp8_base_unet or args.weighted_captions or args.train_inpainting:
            raise ValueError("MOSS supports dense floating-point models and ordinary TXT/JSON conditioning")
        if args.loss_type != "l2" or args.min_snr_gamma or args.v_parameterization or args.debiased_estimation_loss:
            raise ValueError("MOSS uses flow-matching FP32 MSE, without DDPM/SNR loss options")
        self.is_swapping_blocks = bool(args.blocks_to_swap)
        if self.is_swapping_blocks and args.cpu_offload_checkpointing:
            raise ValueError("blocks_to_swap cannot be combined with cpu_offload_checkpointing")
        for group in (train_dataset_group, val_dataset_group):
            if group is None:
                continue
            group.verify_bucket_reso_steps(16)
            for dataset in group.datasets:
                if self.is_swapping_blocks and dataset.batch_size != 1:
                    raise ValueError("MOSS block swapping requires batch_size=1; use gradient_accumulation_steps")
                for info in dataset.image_data.values():
                    subset = dataset.image_to_subset[info.image_key]
                    layout = subset.caption_format != "text" and moss_caption.is_layout(info.caption)
                    if layout and subset.face_crop_aug_range is not None:
                        raise ValueError(f"{info.absolute_path}: use bucket/random cropping for layout data, not face cropping")
                    if args.cache_text_encoder_outputs:
                        dynamic = ((not layout and (subset.shuffle_caption or subset.enable_wildcard or subset.token_warmup_step
                                   or subset.caption_tag_dropout_rate)) or subset.caption_dropout_rate
                                   or subset.caption_dropout_every_n_epochs or (layout and (subset.random_crop or subset.flip_aug)))
                        if dynamic:
                            raise ValueError(f"{info.absolute_path}: stochastic captions/layout geometry require disabling text output caching")

    def get_tokenize_strategy(self, args):
        self.tokenize_strategy = strategy_moss.MossTokenizeStrategy(args.pretrained_model_name_or_path, args.moss_max_token_length)
        return self.tokenize_strategy

    def get_tokenizers(self, strategy):
        return [strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_moss.MossLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size,
            strategy_moss.component_signature(Path(args.pretrained_model_name_or_path) / "vae"))

    def get_text_encoding_strategy(self, args):
        return strategy_moss.MossTextEncodingStrategy()

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_moss.MossTextEncoderOutputsCachingStrategy(args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size, self.tokenize_strategy.cache_identity)
        return None

    def load_target_model(self, args, weight_dtype, accelerator):
        root = Path(args.pretrained_model_name_or_path)
        encoder = AutoModel.from_pretrained(root / "text_encoder", local_files_only=True, dtype=weight_dtype).eval()
        encoder.requires_grad_(False)
        vae = AutoencoderKLFlux2.from_pretrained(root / "vae", local_files_only=True, torch_dtype=torch.float32).eval()
        vae.requires_grad_(False)
        return "moss_image", [encoder], vae, None

    def cast_vae(self, args):
        return False

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        encoder = text_encoders[0]
        device = accelerator.device if args.moss_text_encoder_device == "accelerator" else torch.device("cpu")
        encoder.to(device)
        if args.cache_text_encoder_outputs:
            vae.to("cpu")
            clean_memory_on_device(accelerator.device)
            dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)
        if args.sample_prompts:
            encoding = strategy_base.TextEncodingStrategy.get_strategy()
            with torch.no_grad():
                for entry in sampling.load_prompts(args.sample_prompts):
                    for text in (entry.get("prompt", ""), entry.get("negative_prompt", "")):
                        if text not in self.sample_conditions:
                            self.sample_conditions[text] = [v.cpu() for v in encoding.encode_tokens(
                                self.tokenize_strategy, text_encoders, self.tokenize_strategy.tokenize(text))]
        if args.cache_text_encoder_outputs:
            accelerator.wait_for_everyone()
            encoder.to("cpu")
            clean_memory_on_device(accelerator.device)

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            text_encoders[:] = [_CachedTextEncoder()]
        model = moss_models.MOSSImageTransformer.from_pretrained(
            Path(args.pretrained_model_name_or_path) / "transformer", local_files_only=True, torch_dtype=weight_dtype)
        if self.is_swapping_blocks:
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)
        return model, text_encoders

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None
        for encoder in text_encoders:
            encoder.eval()
        return text_encoders

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs

    def get_noise_scheduler(self, args, device):
        return moss_train_utils.MOSSImageScheduler.from_pretrained(Path(args.pretrained_model_name_or_path) / "scheduler", local_files_only=True)

    def encode_images_to_latents(self, args, vae, images):
        return strategy_moss.encode_latents(vae, images)

    def shift_scale_latents(self, args, latents):
        return latents

    def get_noise_pred_and_target(self, args, accelerator, noise_scheduler, latents, batch, text_encoder_conds,
                                  unet, network, weight_dtype, train_unet, is_train=True):
        noisy, target, timesteps = moss_train_utils.flow_batch(latents, noise_scheduler, args.min_timestep or 0, args.max_timestep)
        embeds, mask = text_encoder_conds
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            prediction = unet(noisy.to(weight_dtype), timesteps,
                              embeds.to(device=accelerator.device, dtype=weight_dtype), mask.to(accelerator.device)).sample
        return prediction, target, timesteps, None

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return {"modelspec.architecture": "moss_image/lora"}

    def update_metadata(self, metadata, args):
        metadata["ss_base_model_version"] = "moss_image"

    def prepare_unet_with_accelerator(self, args, accelerator, unet):
        if not self.is_swapping_blocks:
            return accelerator.prepare(unet)
        model = accelerator.prepare(unet, device_placement=[False])
        accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(model).prepare_block_swap_before_forward()
        return model

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        interval = args.sample_every_n_steps if epoch is None else args.sample_every_n_epochs
        count = global_step if epoch is None else epoch
        if not args.sample_prompts or not interval or count % interval or not accelerator.is_main_process:
            return
        model = accelerator.unwrap_model(unet)
        out = Path(args.output_dir) / "sample"
        out.mkdir(parents=True, exist_ok=True)
        vae.to(device)
        if model.offloader is not None:
            model.offloader.set_forward_only(True)
        try:
            with torch.no_grad():
                for i, entry in enumerate(sampling.load_prompts(args.sample_prompts)):
                    width, height = entry.get("width", 1024), entry.get("height", 1024)
                    if min(width, height) <= 0 or width % 16 or height % 16:
                        raise ValueError("MOSS sample width/height must be positive multiples of 16")
                    generator = torch.Generator(device=device).manual_seed(entry.get("seed", 42))
                    z = torch.randn((1, model.config.latent_channels, height // 16, width // 16), device=device, generator=generator)
                    positive = [x.to(device) for x in self.sample_conditions[entry.get("prompt", "")]]
                    negative = [x.to(device) for x in self.sample_conditions[entry.get("negative_prompt", "")]]
                    scheduler = self.get_noise_scheduler(args, device)
                    scheduler.set_timesteps(entry.get("sample_steps", 30), device)
                    for t in scheduler.timesteps:
                        if model.offloader is not None:
                            model.prepare_block_swap_before_forward()
                        with accelerator.autocast():
                            cond = model(z.to(model.dtype), t, *positive).sample.float()
                            if model.offloader is not None:
                                model.prepare_block_swap_before_forward()
                            skip = [22] if len(model.layers) > 22 else []
                            uncond = model(z.to(model.dtype), t, *negative, skip_layers=skip).sample.float()
                        velocity = uncond + entry.get("scale", 3.0) * (cond - uncond)
                        z = scheduler.step(velocity, t, z, return_dict=False)[0]
                    pixels = moss_train_utils.decode_latents(vae, z)[0].float().add(1).mul(127.5).clamp(0, 255)
                    Image.fromarray(pixels.permute(1, 2, 0).byte().cpu().numpy()).save(out / f"{args.output_name}_{global_step:08d}_{i:03d}.png")
        finally:
            if args.cache_latents:
                vae.to("cpu")
            if model.offloader is not None:
                model.offloader.set_forward_only(False)
                model.prepare_block_swap_before_forward()
            clean_memory_on_device(device)


def setup_parser():
    parser = train_network.setup_parser()
    args_util.add_dit_training_arguments(parser)
    args_util.add_moss_training_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)
    MossNetworkTrainer().train(args)
