import copy
from unittest.mock import patch

import pytest
import torch
from prodigyopt import Prodigy as OriginalProdigy

from library.prodigy_optimizer import Prodigy


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"safeguard_warmup": True, "use_bias_correction": True, "weight_decay": 0.01},
        {"betas": (0.0, 0.99), "slice_p": 3, "growth_rate": 1.02, "d_coef": 0.5},
        {
            "betas": (0.0, 0.99),
            "beta3": 0.8,
            "decouple": False,
            "weight_decay": 0.02,
            "slice_p": 3,
        },
    ],
)
def test_prodigy_matches_original_parameters_and_resume(device, dtype, options):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "cpu" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip("CPU half dot is outside the CUDA optimization")
    if dtype == torch.float16:
        # The upstream default d0 underflows FP16 Adam state; use a finite reference case.
        options = {**options, "d0": 1e-2, "eps": 1e-4}
    torch.manual_seed(43)
    initial = [
        torch.randn(9, 7, device=device, dtype=dtype),
        torch.zeros(13, device=device, dtype=dtype),
        torch.randn((), device=device, dtype=dtype),
    ]
    reference_params = [torch.nn.Parameter(p.clone()) for p in initial]
    actual_params = [torch.nn.Parameter(p.clone()) for p in initial]
    reference = OriginalProdigy(reference_params, lr=1.0, **options)
    actual = Prodigy(actual_params, lr=1.0, **options)
    for step in range(7):
        for a, b in zip(reference_params, actual_params):
            grad = torch.randn_like(a)
            a.grad, b.grad = grad.clone(), grad.clone()
        reference.step()
        actual.step()
        for a, b in zip(reference_params, actual_params):
            assert torch.isfinite(a).all()
            assert torch.equal(a, b), f"parameter changed at step {step}"
        assert_state_equal(reference.state_dict(), actual.state_dict())
        if step == 2:
            old_state = copy.deepcopy(reference.state_dict())
            new_state = copy.deepcopy(actual.state_dict())
            reference = OriginalProdigy(reference_params, lr=1.0, **options)
            actual = Prodigy(actual_params, lr=1.0, **options)
            # Both directions must accept the other implementation's state.
            reference.load_state_dict(new_state)
            actual.load_state_dict(old_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prodigy_mixed_dtypes_missing_gradients_and_zero_lr():
    torch.manual_seed(12)
    ref = [
        torch.nn.Parameter(torch.randn(7, device="cuda", dtype=d))
        for d in [torch.float32, torch.bfloat16, torch.float64]
    ]
    got = [torch.nn.Parameter(p.clone()) for p in ref]
    old = OriginalProdigy(
        [{"params": ref[:2]}, {"params": ref[2:], "lr": 0.0}], lr=1.0, weight_decay=0.01
    )
    new = Prodigy(
        [{"params": got[:2]}, {"params": got[2:], "lr": 0.0}], lr=1.0, weight_decay=0.01
    )
    for step in range(5):
        for i, (a, b) in enumerate(zip(ref, got)):
            grad = torch.randn_like(a) if step != i else None
            a.grad = grad
            b.grad = None if grad is None else grad.clone()
        old.step()
        new.step()
        assert_state_equal(old.state_dict(), new.state_dict())
        for a, b in zip(ref, got):
            assert torch.equal(a, b)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prodigy_closure_and_step_hooks_fire_once(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    # Materialize the upstream profile wrapper before constructing the subclass.
    OriginalProdigy([torch.nn.Parameter(torch.ones(1))])
    p = torch.nn.Parameter(torch.ones(3, device=device))
    optimizer = Prodigy([p], lr=1.0)
    calls = []
    optimizer.register_step_pre_hook(lambda *args: calls.append("before"))
    optimizer.register_step_post_hook(lambda *args: calls.append("after"))

    def closure():
        calls.append("closure")
        loss = p.square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss.item() == 3.0
    assert calls == ["before", "closure", "after"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prodigy_unsupported_version_uses_original_step():
    p = torch.nn.Parameter(torch.ones(3, device="cuda"))
    optimizer = Prodigy([p], lr=1.0)
    calls = []

    def original(self, closure=None):
        calls.append(self)
        return "original"

    with patch("library.prodigy_optimizer._SUPPORTED_VERSION", False):
        with patch.object(OriginalProdigy, "step", original):
            assert optimizer.step() == "original"
    assert calls == [optimizer]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prodigy_shared_parameter_storage_uses_original_order():
    torch.manual_seed(15)
    first = torch.randn(9, device="cuda")
    second = first.clone()
    old_params = [torch.nn.Parameter(first[:7]), torch.nn.Parameter(first[2:])]
    new_params = [torch.nn.Parameter(second[:7]), torch.nn.Parameter(second[2:])]
    old, new = OriginalProdigy(old_params, lr=1.0), Prodigy(new_params, lr=1.0)
    for _ in range(3):
        for a, b in zip(old_params, new_params):
            a.grad = torch.randn_like(a)
            b.grad = a.grad.clone()
        old.step()
        new.step()
        assert torch.equal(first, second)
        assert_state_equal(old.state_dict(), new.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_prodigy_bounded_scratch_batches_preserve_exact_updates():
    from library.prodigy_optimizer import _parameter_batches

    torch.manual_seed(20)
    old_params = [torch.nn.Parameter(torch.randn(17, device="cuda")) for _ in range(6)]
    new_params = [torch.nn.Parameter(p.clone()) for p in old_params]
    old, new = OriginalProdigy(old_params, lr=1.0), Prodigy(new_params, lr=1.0)
    with patch("library.prodigy_optimizer._UPDATE_BYTES", 136):
        assert [len(batch) for batch in _parameter_batches(new_params)] == [2, 2, 2]
        for _ in range(4):
            for a, b in zip(old_params, new_params):
                a.grad = torch.randn_like(a)
                b.grad = a.grad.clone()
            old.step()
            new.step()
            assert_state_equal(old.state_dict(), new.state_dict())
            for a, b in zip(old_params, new_params):
                assert torch.equal(a, b)
