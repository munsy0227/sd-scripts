# MOSS-Image official network trainer

This fork integrates official `dev` through `6a3290182c786541ccf292106782b51568e46da5`,
including the stage 3 trainer at `cb5c24fc724f758fcb1e63969872f2f561ac1396` and its follow-up fixes.
`moss_image_train_network.py` is the main implementation. The existing
`moss_train_network.py` subclasses it to preserve the LyCORIS command, TOML,
presets, and discrete uniform timestep sampling.

Use a separate environment with this repository's `requirements.txt` and the
local LyCORIS fork when choosing `lycoris.kohya`. See the
[dataset, cache, checkpoint and ComfyUI guide](moss_train_network.md) for local
model roots, official shards, floating-point ComfyUI files, offline tokenizers,
TXT/JSON/layout captions, and cache migration.

## Generic LoRA

```sh
accelerate launch --mixed_precision bf16 moss_image_train_network.py \
  --pretrained_model_name_or_path /models/MOSS-Image-Base \
  --dataset_config examples/moss/dataset.toml \
  --network_module networks.lora_generic --network_dim 16 --network_alpha 16 \
  --cache_latents_to_disk --cache_text_encoder_outputs_to_disk \
  --gradient_checkpointing --mixed_precision bf16 \
  --optimizer_type AdamW --learning_rate 1e-4 --max_train_steps 1000 \
  --output_dir output --output_name moss-generic --save_model_as safetensors
```

This demonstrates the interface; it is not a measured quality or VRAM recipe.
The VAE stays FP32 and Qwen weights stay BF16 with FP32 rotary buffers.
Qwen is frozen by default and text preprocessing runs on the CPU. Use
`--moss_text_encoder_device accelerator` to encode on the accelerator.
Cached prompt embeddings remain on the CPU and frozen cached encoders are
released before loading the DiT, after all dataset groups have been processed.

The official entry defaults to `--timestep_sampling sigmoid`. The upstream PR
recommends the following explicit setting to reproduce its pretraining
distribution; this integration does not validate that quality claim:

```text
--timestep_sampling sigma --weighting_scheme logit_normal
--logit_mean -1.09 --logit_std 0.95 --discrete_flow_shift 1.0
```

`--show_timesteps console` visualizes the selected distribution without model
weights. Sampling uses the same `generate_latents` implementation as minimal
inference: 50 steps, CFG 3, flow shift 5, unconditional skip layer 22 by default.
Per-prompt `--s`, `--l`, `--fs`, `--n`, `--w`, `--h`, `--d` are supported.
Sample dimensions must be positive multiples of 16.

## Adapter scopes and LyCORIS

| Selection | DiT targets in the production architecture |
| --- | ---: |
| Generic default / LyCORIS `preset=moss` | 252 |
| Generic `--network_args train_text_embedder=true` | 263 |
| LyCORIS `preset=moss-full` | 269 |

The 263-target generic scope omits the six timestep/video/modulation/output
projections included in `moss-full`. The generic alias is not a preset rename.
`networks.loha` and `networks.lokr` share the MOSS architecture configuration;
their adapter option is `train_llm_adapter=true`. Native LoKr does not provide
this fork's LyCORIS full-matrix LoKR+DoRA implementation.

For that configuration, keep the existing command or choose the same module
and preset explicitly at the official entry point:

```text
--network_module lycoris.kohya --network_args algo=lokr preset=moss-full full_matrix=True factor=4 weight_decompose=True
```

The old entry defaults to `--timestep_sampling legacy`, 30 sample steps and
sample seed 42. It preserves the old training grid and RNG draws. Selecting a
new distribution changes the training trajectory. Shared sampling now draws
initial noise on a CPU generator; old GPU-generator images can differ for the
same seed. Existing native state and portable adapter loading are preserved.

## Optional TE adapters and execution options

TE adapter training is opt-in at the official entry point:

```text
--moss_train_text_encoder --moss_text_encoder_device accelerator --network_module networks.lora_generic
```

Disable text-output caching for this mode. Latent caching remains available.
LyCORIS MOSS presets and the old command keep Qwen frozen. The upstream TE merge
hook is included, and small Qwen/model tests pass; real 9B TE training and TE
adapter use in ComfyUI remain unverified locally.

`--compile` enables per-block compilation. Do not combine it with
`--torch_compile`, or combine `--compile_fullgraph` with `--split_attn`.
`--unsloth_offload_checkpointing` enables gradient checkpointing and cannot
combine with CPU checkpoint offload or block swap. CPU checkpoint offload
saves inputs on the CPU while keeping live outputs on the model device.
Sampling restores RNG, VAE placement, and training/block-swap state on success
and failure.

Validation includes tiny CUDA BF16 save/resume through both entries, adapter
scope and lifecycle tests, CPU/unsloth checkpoint output and full gradient
comparisons, and CPU compile with the `eager` backend. Full pretrained image
quality, long runs, CUDA Inductor performance, optional attention backends,
Windows and MPS are not validated. Quantization, edit training and full-model
training remain outside this integration.
