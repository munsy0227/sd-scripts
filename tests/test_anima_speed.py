from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import anima_train_network as training
from library import sd3_train_utils
from library.anima_models import Block
from library.attention import AttentionParams


@pytest.mark.parametrize(
    "checkpointing,swapping,unsloth,cpu_offload,expected",
    [
        (False, False, False, False, False),
        (True, False, False, False, False),
        (True, True, False, False, True),
        (True, False, True, False, True),
        (True, False, False, True, True),
    ],
)
@pytest.mark.parametrize("train_text", [False, True])
def test_only_offload_paths_force_input_gradients(
    checkpointing, swapping, unsloth, cpu_offload, expected, train_text
):
    args = training.setup_parser().parse_args(["--timestep_sampling", "sigma"])
    args.gradient_checkpointing = checkpointing
    args.cpu_offload_checkpointing = cpu_offload
    trainer = training.AnimaNetworkTrainer()
    trainer.is_swapping_blocks = swapping
    trainer._use_unsloth_offload_checkpointing = unsloth
    context = torch.randn(1, 3, 32, requires_grad=train_text)
    seen = []

    def model(x, timestep, text, **kwargs):
        seen.append((x.requires_grad, text.requires_grad))
        return x + text.mean()

    output, *_ = trainer.get_noise_pred_and_target(
        args,
        SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext),
        sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
        torch.randn(1, 16, 4, 4),
        {},
        [
            context,
            torch.ones(1, 3),
            torch.ones(1, 3, dtype=torch.long),
            torch.ones(1, 3),
        ],
        model,
        None,
        torch.float32,
        True,
    )
    assert seen == [(expected, expected or train_text)]
    if train_text:
        output.square().mean().backward()
        assert context.grad is not None and torch.isfinite(context.grad).all()


@pytest.mark.parametrize("backend", ["inductor", "eager"])
def test_precision_cast_options_only_apply_to_inductor(monkeypatch, backend):
    args = training.setup_parser().parse_args(
        ["--compile", "--compile_backend", backend]
    )
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList()
    trainer = training.AnimaNetworkTrainer()
    trainer.is_swapping_blocks = False
    trainer._use_unsloth_offload_checkpointing = False
    captured = {}
    monkeypatch.setattr(
        training.compile_utils,
        "compile_transformer",
        lambda *args, **kwargs: captured.update(kwargs),
    )
    trainer.prepare_unet_with_accelerator(
        args,
        SimpleNamespace(prepare=lambda m: m, unwrap_model=lambda m: m),
        model,
    )
    assert captured["disable_linear"] is False
    assert captured["options"] == (
        {"emulate_precision_casts": True} if backend == "inductor" else None
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("scalar", [0.0, 0.5])
def test_checkpoint_without_input_gradients_preserves_all_adapter_gradients(
    device, scalar
):
    modules = pytest.importorskip("lycoris.modules")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    torch.manual_seed(81)
    block = Block(64, 32, 4, mlp_ratio=2, use_adaln_lora=True, adaln_lora_dim=8)
    block.init_weights()
    with torch.no_grad():
        for layer in block.modules():
            if isinstance(layer, torch.nn.Linear) and not torch.count_nonzero(
                layer.weight
            ):
                layer.weight.normal_(std=0.02)
    block.to(device=device, dtype=dtype).train().requires_grad_(False)
    block.enable_gradient_checkpointing()
    adapters = []
    for name, layer in list(block.named_modules()):
        if isinstance(layer, torch.nn.Linear):
            adapter = modules.LokrModule(
                name,
                layer,
                lora_dim=100000,
                factor=4,
                full_matrix=True,
                weight_decompose=True,
                use_scalar=True,
            ).to(device)
            with torch.no_grad():
                adapter.lokr_w2.normal_(std=0.02)
                adapter.scalar.fill_(scalar)
            adapter.apply_to()
            adapters.append(adapter)
    parameters = tuple(p for adapter in adapters for p in adapter.parameters())
    x = torch.randn(1, 1, 2, 2, 64, device=device, dtype=dtype)
    embedding = torch.randn(1, 1, 64, device=device)
    context = torch.randn(1, 5, 32, device=device, dtype=dtype)
    modulation = torch.randn(1, 1, 192, device=device, dtype=dtype)
    outputs, gradients = [], []
    for input_gradients in (True, False):
        with torch.autocast(device, dtype=dtype, enabled=device == "cuda"):
            output = block(
                x.clone().requires_grad_(input_gradients),
                embedding,
                context.clone().requires_grad_(input_gradients),
                AttentionParams.create_attention_params("torch", False),
                adaln_lora_B_T_3D=modulation,
            )
        outputs.append(output)
        gradients.append(
            torch.autograd.grad(output.float().square().mean(), parameters)
        )
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for old, new in zip(*gradients):
        assert torch.isfinite(new).all()
        torch.testing.assert_close(old, new, rtol=0, atol=0)
