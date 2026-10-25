import copy
from types import SimpleNamespace

import pytest
import torch
from prodigyopt import Prodigy as OriginalProdigy

import moss_image_train_network as official
from library import moss_image_models
from library.attention import AttentionParams
from library.prodigy_optimizer import Prodigy


@pytest.mark.parametrize("preset", ["moss", "moss-full"])
def test_plain_checkpoint_preserves_adapter_gradients_without_input_gradients(preset):
    from lycoris import create_lycoris

    torch.manual_seed(81)
    model = moss_image_models.MossImage(
        dim=64,
        ffn_dim=96,
        num_heads=4,
        num_layers=2,
        text_dim=24,
        freq_dim=16,
        latent_channels=8,
    ).train()
    model.requires_grad_(False)
    model.enable_gradient_checkpointing()
    network = create_lycoris(
        model,
        preset=preset,
        algo="lokr",
        linear_dim=100000,
        factor=4,
        full_matrix=True,
        weight_decompose=True,
        use_scalar=True,
    )
    network.apply_to()
    with torch.no_grad():
        for adapter in network.loras:
            adapter.lokr_w2.normal_(std=0.02)
            adapter.scalar.fill_(0.5)
    x, context = torch.randn(1, 8, 2, 2), torch.randn(1, 5, 24)
    t, mask = torch.tensor([300.0]), torch.tensor([[1, 1, 1, 0, 0]])
    results = []
    gradients = []
    for input_grad in (True, False):
        result = model(
            x.clone().requires_grad_(input_grad),
            t,
            context.clone().requires_grad_(input_grad),
            mask,
        )
        results.append(result)
        gradients.append(
            torch.autograd.grad(result.square().mean(), tuple(network.parameters()))
        )
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    for got, expected in zip(*gradients):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


def make_block():
    modules = pytest.importorskip("lycoris.modules")
    torch.manual_seed(42)
    block = moss_image_models.MossImageBlock(128, 256, 4).cuda().bfloat16().train()
    block.requires_grad_(False)
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
            ).cuda()
            with torch.no_grad():
                adapter.lokr_w2.normal_(std=0.01)
                adapter.scalar.fill_(0.5)
            adapter.apply_to()
            adapters.append(adapter)
    return block, tuple(p for adapter in adapters for p in adapter.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_inductor_masked_checkpoint_gradients_optimizer_steps_and_resume():
    reference, reference_params = make_block()
    block, params = make_block()
    holder = torch.nn.Module()
    holder.layers = torch.nn.ModuleList([block])
    args = official.setup_parser().parse_args(["--compile", "--compile_fullgraph"])
    accelerator = SimpleNamespace(prepare=lambda m: m, unwrap_model=lambda m: m)
    holder = official.MossImageNetworkTrainer().prepare_unet_with_accelerator(
        args, accelerator, holder
    )
    compiled = holder.layers[0]
    x = torch.randn(1, 37, 128, device="cuda", dtype=torch.bfloat16)
    modulation = torch.randn(1, 6, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    rope = torch.randn(1, 37, 32, device="cuda", dtype=torch.bfloat16)
    mask = torch.tensor([[1, 1, 1, 0, 0]], device="cuda")
    attention = AttentionParams.create_attention_params_from_mask(
        "torch", False, 32, mask
    )
    target = torch.randn_like(x)
    options = dict(
        lr=1.0,
        betas=(0.9, 0.99),
        safeguard_warmup=True,
        use_bias_correction=True,
        weight_decay=0.01,
    )
    old, new = OriginalProdigy(reference_params, **options), Prodigy(params, **options)
    snapshot = None

    def forward(module):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return module(x, rope, modulation, attention)

    def train_step(module, parameters, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = (forward(module).float() - target.float()).square().mean()
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in parameters
        )
        optimizer.step()
        return loss.detach()

    expected, actual = forward(reference), forward(compiled)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
    grad = torch.randn_like(expected)
    for got, wanted in zip(
        torch.autograd.grad(actual, params, grad),
        torch.autograd.grad(expected, reference_params, grad),
    ):
        torch.testing.assert_close(got, wanted, rtol=5e-2, atol=2e-3)
    for step in range(5):
        wanted = train_step(reference, reference_params, old)
        got = train_step(compiled, params, new)
        torch.testing.assert_close(got, wanted, rtol=5e-3, atol=1e-5)
        for a, b in zip(params, reference_params):
            torch.testing.assert_close(a, b, rtol=5e-3, atol=1e-5)
        if step == 1:
            snapshot = (
                [p.detach().clone() for p in params],
                copy.deepcopy(new.state_dict()),
            )
    final = [p.detach().clone() for p in params]
    with torch.no_grad():
        for p, value in zip(params, snapshot[0]):
            p.copy_(value)
    new = Prodigy(params, **options)
    new.load_state_dict(snapshot[1])
    for _ in range(3):
        train_step(compiled, params, new)
    for p, wanted in zip(params, final):
        torch.testing.assert_close(p, wanted, rtol=1e-6, atol=1e-8)
    torch._dynamo.reset()
