"""Small real Anima precision regressions; no checkpoint download is needed."""

from contextlib import nullcontext

import pytest
import torch

from library.anima_models import Anima, Block, FinalLayer
from library.attention import AttentionParams

PRECISIONS = [
    ("cpu", torch.float32),
    ("cpu", torch.bfloat16),
    ("cuda", torch.float32),
    ("cuda", torch.float16),
    ("cuda", torch.bfloat16),
]


def require_precision(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if (
        device == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        pytest.skip("CUDA BF16 required")


def autocast(device, dtype):
    return (
        torch.autocast(device, dtype=dtype) if dtype != torch.float32 else nullcontext()
    )


def autocast_state(device):
    return torch.is_autocast_enabled(device), torch.get_autocast_dtype(device)


def check_backward(output, module):
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) > 0 for g in grads)


@pytest.mark.parametrize("device,dtype", PRECISIONS)
@pytest.mark.parametrize("use_adaln_lora", [False, True])
@pytest.mark.parametrize("kind", ["block", "final"])
def test_adaln_preserves_caller_autocast(device, dtype, use_adaln_lora, kind):
    require_precision(device, dtype)
    torch.manual_seed(41)
    if kind == "block":
        module = Block(
            64, 32, 4, mlp_ratio=2, use_adaln_lora=use_adaln_lora, adaln_lora_dim=8
        )
        module.init_weights()
        modulations = [
            module.adaln_modulation_self_attn,
            module.adaln_modulation_cross_attn,
            module.adaln_modulation_mlp,
        ]
    else:
        module = FinalLayer(
            64, 2, 1, 16, use_adaln_lora=use_adaln_lora, adaln_lora_dim=8
        )
        modulations = [module.adaln_modulation]
    module.to(device=device, dtype=dtype)
    x = torch.randn(1, 1, 2, 2, 64, device=device, dtype=dtype, requires_grad=True)
    # Anima with AdaLN-LoRA retains the original FP32 timestep features.
    emb = torch.randn(1, 1, 64, device=device, dtype=torch.float32, requires_grad=True)
    extra = (
        torch.randn(1, 1, 192, device=device, dtype=dtype) if use_adaln_lora else None
    )
    use_fp32 = dtype == torch.float16
    observed = []
    hooks = [
        m[1].register_forward_pre_hook(
            lambda _m, _args: observed.append(autocast_state(device))
        )
        for m in modulations
    ]
    original_state = autocast_state(device)
    try:
        with autocast(device, dtype):
            caller_state = autocast_state(device)
            # Preserve the existing FP16 -> FP32 autocast behavior on this runtime.
            if use_fp32:
                with torch.autocast(device_type=device, dtype=torch.float32):
                    expected_state = autocast_state(device)
            else:
                expected_state = caller_state
            if kind == "block":
                context = torch.randn(1, 8, 32, device=device, dtype=dtype)
                output = module(
                    x,
                    emb,
                    context,
                    AttentionParams.create_attention_params("torch", False),
                    use_fp32,
                    adaln_lora_B_T_3D=extra,
                )
            else:
                output = module(x, emb, adaln_lora_B_T_3D=extra, use_fp32=use_fp32)
            assert autocast_state(device) == caller_state
        assert autocast_state(device) == original_state
        assert observed and all(state == expected_state for state in observed)
        check_backward(output, module)
        assert emb.grad is not None and torch.isfinite(emb.grad).all()
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize("device,dtype", PRECISIONS)
@pytest.mark.parametrize("use_adaln_lora", [False, True])
def test_anima_forward_backward_and_checkpointing(device, dtype, use_adaln_lora):
    require_precision(device, dtype)
    torch.manual_seed(29)
    model = Anima(
        max_img_h=4,
        max_img_w=4,
        max_frames=1,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=64,
        num_blocks=1,
        num_heads=4,
        mlp_ratio=2,
        crossattn_emb_channels=32,
        pos_emb_cls="rope3d",
        use_adaln_lora=use_adaln_lora,
        adaln_lora_dim=8,
        rope_enable_fps_modulation=False,
        use_llm_adapter=False,
        attn_mode="torch",
    )
    # Trained checkpoints have nonzero gates; otherwise the block backward is trivial.
    with torch.no_grad():
        for module in model.modules():
            if (
                isinstance(module, torch.nn.Linear)
                and torch.count_nonzero(module.weight) == 0
            ):
                module.weight.normal_(0, 0.02)
    model.to(device=device, dtype=dtype).train()
    parameter_schema = [
        (name, tuple(p.shape), p.dtype, id(p)) for name, p in model.named_parameters()
    ]
    timestep = torch.tensor([0.4], device=device, dtype=torch.float32)
    x = torch.randn(1, 16, 1, 4, 4, device=device, dtype=dtype)
    context = torch.randn(1, 8, 32, device=device, dtype=dtype)
    timestep_dtypes = []
    hook = model.t_embedder[0].register_forward_hook(
        lambda _m, _args, out: timestep_dtypes.append(out.dtype)
    )
    results = []
    try:
        for checkpointing in (False, True):
            if checkpointing:
                model.enable_gradient_checkpointing()
            model.zero_grad(set_to_none=True)
            original_state = autocast_state(device)
            with autocast(device, dtype):
                caller_state = autocast_state(device)
                output = model(
                    x.detach().clone().requires_grad_(True), timestep, context=context
                )
                assert autocast_state(device) == caller_state
            assert autocast_state(device) == original_state
            check_backward(output, model)
            results.append(
                (
                    output.detach().clone(),
                    {
                        n: p.grad.clone()
                        for n, p in model.named_parameters()
                        if p.grad is not None
                    },
                )
            )
        assert timestep_dtypes == [torch.float32, torch.float32]
        torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
        assert results[0][1].keys() == results[1][1].keys()
        for name in results[0][1]:
            torch.testing.assert_close(
                results[0][1][name], results[1][1][name], rtol=0, atol=0
            )
        assert parameter_schema == [
            (name, tuple(p.shape), p.dtype, id(p))
            for name, p in model.named_parameters()
        ]
    finally:
        hook.remove()
