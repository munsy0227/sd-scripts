"""Official trainer integration regressions without pretrained checkpoints."""

import copy
import os
from pathlib import Path
import subprocess
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import moss_image_train_network as official
import moss_train_network as legacy
from library.custom_offloading_utils import BlockSwapConfig
from library import (
    moss_image_models,
    moss_image_train_utils,
    strategy_base,
    strategy_moss_image,
)


def make_model(layers=4):
    return moss_image_models.MossImage(
        dim=32,
        ffn_dim=48,
        num_heads=4,
        num_layers=layers,
        text_dim=16,
        freq_dim=16,
        latent_channels=128,
    )


@pytest.mark.parametrize("module", ["lycoris.kohya", "networks.lora_generic"])
def test_convrot_dit_rejects_lycoris_before_adapter_creation(monkeypatch, module):
    model = make_model()
    model.is_convrot_int8 = True
    trainer = official.MossImageNetworkTrainer()
    args = official.setup_parser().parse_args(["--network_module", module])
    monkeypatch.setattr(official.moss_image_utils, "checkpoint_file", lambda *a: "dit.safetensors")
    monkeypatch.setattr(official.moss_image_utils, "load_moss_image_model", lambda *a, **k: model)
    accelerator = SimpleNamespace(device=torch.device("cpu"))
    if module == "lycoris.kohya":
        with pytest.raises(ValueError, match="floating-point DiT"):
            trainer.load_unet_lazily(args, torch.bfloat16, accelerator, [None])
    else:
        loaded, _ = trainer.load_unet_lazily(args, torch.bfloat16, accelerator, [None])
        assert loaded is model and trainer.dit_convrot_int8
    assert not trainer.cast_unet(args)


def test_convrot_qwen_changes_cache_identity_without_changing_dense_identity():
    from library import moss_image_utils

    args = official.setup_parser().parse_args([])
    paths = SimpleNamespace(text_encoder=None, tokenizer=None)
    dense = moss_image_utils.text_cache_identity(args, paths)
    args.convrot_int8_text_encoder = True
    quantized = moss_image_utils.text_cache_identity(args, paths)
    assert quantized == {**dense, "quantization": "convrot_int8"}


def test_quantized_text_encoder_checkpoint_workaround_selects_floating_parameter():
    encoder = torch.nn.Module()
    encoder.register_parameter("tokens", torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.int8), requires_grad=False))
    encoder.register_parameter("norm", torch.nn.Parameter(torch.ones(4), requires_grad=False))
    official.MossImageNetworkTrainer().prepare_text_encoder_grad_ckpt_workaround(0, encoder)
    assert encoder.norm.requires_grad and not encoder.tokens.requires_grad


@pytest.mark.parametrize("entry", [official, legacy])
def test_validation_reuses_cached_batch_without_dropout_or_mutation(monkeypatch, entry):
    args = entry.setup_parser().parse_args([])
    trainer = (
        entry.MossImageNetworkTrainer()
        if entry is official
        else entry.MossNetworkTrainer()
    )
    trainer.bn_running_mean, trainer.bn_running_var = torch.zeros(128), torch.ones(128)
    encoding = strategy_moss_image.MossImageTextEncodingStrategy()
    monkeypatch.setattr(strategy_base.TextEncodingStrategy, "_strategy", encoding)
    conditions = [
        torch.ones(1, 3, 16),
        torch.ones(1, 3),
        torch.full((1, 2, 16), 7.0),
        torch.ones(1, 2),
        torch.ones(1),
    ]
    originals = [value.clone() for value in conditions]
    batch = {
        "latents": torch.randn(1, 32, 4, 4),
        "text_encoder_outputs_list": conditions,
        "loss_weights": torch.ones(1),
    }
    seen = []

    def model(z, t, context, mask):
        seen.append((t.clone(), context.clone()))
        return torch.zeros_like(z)

    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    scheduler = trainer.get_noise_scheduler(args, accelerator.device)
    for timestep in (200, 800):
        args.min_timestep = args.max_timestep = timestep
        loss = trainer.process_batch(
            batch,
            [],
            model,
            None,
            None,
            scheduler,
            torch.float32,
            torch.float32,
            accelerator,
            args,
            encoding,
            None,
            is_train=False,
            train_text_encoder=False,
        )
        assert torch.isfinite(loss)
        assert batch["text_encoder_outputs_list"] is conditions and len(conditions) == 5
        for before, after in zip(originals, conditions):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(seen[-1][1], originals[0])
    assert seen[1][0].item() > seen[0][0].item()
    if entry is official:
        assert [t.item() for t, _ in seen] == [200, 800]
    trainer.process_batch(
        batch,
        [],
        model,
        None,
        None,
        scheduler,
        torch.float32,
        torch.float32,
        accelerator,
        args,
        encoding,
        None,
        is_train=True,
        train_text_encoder=False,
    )
    torch.testing.assert_close(seen[-1][1], originals[2])
    assert len(conditions) == 5


def test_legacy_training_distribution_and_rng_are_preserved():
    args = legacy.setup_parser().parse_args([])
    trainer = legacy.MossNetworkTrainer()
    scheduler = trainer.get_noise_scheduler(args, torch.device("cpu"))
    latents = torch.randn(2, 128, 2, 2)
    conds = [torch.randn(2, 3, 16), torch.ones(2, 3)]
    torch.manual_seed(123)
    expected_input, expected_target, expected_t = moss_image_train_utils.flow_batch(
        latents, scheduler
    )
    expected_rng = torch.get_rng_state().clone()
    seen = []

    def model(z, t, *_):
        seen.append(z)
        return torch.zeros_like(z)

    torch.manual_seed(123)
    _, target, timesteps, weighting = trainer.get_noise_pred_and_target(
        args,
        SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext),
        scheduler,
        latents,
        {},
        conds,
        model,
        None,
        torch.float32,
        True,
    )
    torch.testing.assert_close(seen[0], expected_input, rtol=0, atol=0)
    torch.testing.assert_close(target, expected_target, rtol=0, atol=0)
    torch.testing.assert_close(timesteps, expected_t, rtol=0, atol=0)
    assert weighting is None and torch.equal(torch.get_rng_state(), expected_rng)


def test_text_encoder_training_is_explicit_and_cache_incompatible(monkeypatch):
    monkeypatch.setattr(
        official.train_network.NetworkTrainer, "train", lambda self, args: args
    )
    trainer = official.MossImageNetworkTrainer()
    assert trainer.train(official.setup_parser().parse_args([])).network_train_unet_only
    args = official.setup_parser().parse_args(["--moss_train_text_encoder"])
    with pytest.raises(ValueError, match="accelerator"):
        trainer.train(args)
    args = official.setup_parser().parse_args(
        [
            "--moss_train_text_encoder",
            "--moss_text_encoder_device",
            "accelerator",
            "--cache_text_encoder_outputs",
            "--text_encoder",
            "te",
            "--vae",
            "vae",
        ]
    )
    assert not trainer.train(args).network_train_unet_only
    group = SimpleNamespace()
    with pytest.raises(AssertionError, match="cannot be trained with caching"):
        trainer.assert_extra_args(args, group, None)


def test_native_targets_do_not_redefine_moss_full():
    from lycoris import create_lycoris
    from networks import lora_generic

    model = make_model(layers=42)
    base = lora_generic.create_network(1.0, 4, 1.0, None, [None], model)
    text = lora_generic.create_network(
        1.0, 4, 1.0, None, [None], model, train_text_embedder="true"
    )
    moss = create_lycoris(
        model,
        preset="moss",
        algo="lokr",
        linear_dim=4,
        full_matrix=True,
        factor=4,
        weight_decompose=True,
    )
    full = create_lycoris(
        model,
        preset="moss-full",
        algo="lokr",
        linear_dim=4,
        full_matrix=True,
        factor=4,
        weight_decompose=True,
    )
    assert (
        len(base.unet_loras),
        len(text.unet_loras),
        len(moss.loras),
        len(full.loras),
    ) == (252, 263, 252, 269)
    official_paths = {
        item.lora_name.removeprefix("lora_unet_") for item in base.unet_loras
    }
    module_paths = {
        id(module): name.replace(".", "_") for name, module in model.named_modules()
    }
    preset_paths = {module_paths[id(item.org_module[0])] for item in moss.loras}
    assert official_paths == preset_paths


def test_per_block_compile_preserves_forward_and_backward():
    args = official.setup_parser().parse_args(
        ["--compile", "--compile_backend", "eager"]
    )
    trainer = official.MossImageNetworkTrainer()
    model = make_model(layers=1)
    reference = copy.deepcopy(model)
    accelerator = SimpleNamespace(
        prepare=lambda model: model, unwrap_model=lambda model: model
    )
    compiled = trainer.prepare_unet_with_accelerator(args, accelerator, model)
    x = torch.randn(1, 128, 2, 2, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    t, context, mask = torch.tensor([250.0]), torch.randn(1, 3, 16), torch.ones(1, 3)
    expected, actual = reference(ref_x, t, context, mask), compiled(x, t, context, mask)
    torch.testing.assert_close(actual, expected)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    torch._dynamo.reset()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sampling_error_restores_rng_vae_and_block_swap(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = make_model().to(device).train()
    if device == "cuda":
        model.enable_block_swap(2, BlockSwapConfig(torch.device(device), supports_backward=True))
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("sample --w 32 --h 32 --s 2")
    args = official.setup_parser().parse_args(
        [
            "--sample_prompts",
            str(prompts),
            "--sample_every_n_steps",
            "1",
            "--output_dir",
            str(tmp_path),
        ]
    )

    class VAE:
        device = torch.device("cpu")

        def to(self, target):
            self.device = torch.device(target)

    vae = VAE()

    def fail(*args, **kwargs):
        vae.to(device)
        torch.manual_seed(999)
        raise RuntimeError("sample failed")

    monkeypatch.setattr(moss_image_train_utils, "_sample_image_inference", fail)
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    accelerator = SimpleNamespace(
        device=torch.device(device),
        unwrap_model=lambda model: model,
        autocast=nullcontext,
    )
    with pytest.raises(RuntimeError, match="sample failed"):
        moss_image_train_utils.sample_images(
            accelerator, args, None, 1, model, vae, None, None, None
        )
    assert model.training and vae.device.type == "cpu"
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if gpu_rng is not None:
        assert all(
            torch.equal(a, b) for a, b in zip(gpu_rng, torch.cuda.get_rng_state_all())
        )
    x = torch.randn(1, 128, 2, 2, device=device, requires_grad=True)
    model(
        x, torch.tensor([200.0], device=device), torch.randn(1, 3, 16, device=device)
    ).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("mode", ["cpu", "unsloth"])
def test_cuda_offload_checkpointing_backward_matches_plain(mode):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = make_model(layers=2).cuda().bfloat16().train()
    reference = copy.deepcopy(model)
    model.enable_gradient_checkpointing(
        cpu_offload=mode == "cpu", unsloth_offload=mode == "unsloth"
    )
    x = torch.randn(
        1, 128, 2, 2, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    ref_x = x.detach().clone().requires_grad_(True)
    t, context = (
        torch.tensor([321.0], device="cuda"),
        torch.randn(1, 3, 16, device="cuda", dtype=torch.bfloat16),
    )
    expected, actual = reference(ref_x, t, context), model(x, t, context)
    torch.testing.assert_close(actual, expected)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for actual_param, expected_param in zip(model.parameters(), reference.parameters()):
        assert (actual_param.grad is None) == (expected_param.grad is None)
        if actual_param.grad is not None:
            torch.testing.assert_close(actual_param.grad, expected_param.grad)


@pytest.mark.parametrize("entry", ["legacy", "official"])
def test_cuda_bf16_training_loop_save_and_resume_in_separate_process(tmp_path, entry):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # Accelerate keeps mixed precision in process-global state. Isolate BF16
    # training from the FP32 tests, while reusing their real dataset/loop fixture.
    script = """
import sys
from pathlib import Path
import pytest
import moss_train_network as legacy
import moss_image_train_network as official
from test_moss_training import _training_case

with pytest.MonkeyPatch.context() as patch:
    extra = []
    if sys.argv[2] == 'official':
        patch.setattr(legacy, 'MossNetworkTrainer', official.MossImageNetworkTrainer)
        patch.setattr(legacy, 'setup_parser', official.setup_parser)
        extra = ['--network_module', 'networks.lora_generic', '--network_args', 'train_text_embedder=true']
    root = Path(sys.argv[1])
    run, _, _ = _training_case(root, patch, 'txt')
    flags = ['--mixed_precision', 'bf16', '--moss_text_encoder_device', 'accelerator'] + extra
    run(flags)
    state = root / 'output' / 'moss-step00000001-state'
    assert state.is_dir()
    run(flags + ['--resume', str(state), '--max_train_steps', '2'])
    assert (root / 'output' / 'moss-step00000002-state').is_dir()
"""
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "tests"), str(root), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), entry],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
