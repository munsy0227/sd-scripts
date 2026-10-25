"""Prodigy 1.1.2 with one CUDA-to-CPU statistics transfer per step.

Initialization, public options and state_dict are inherited from prodigyopt.
CPU, sharded, coupled-decay, shared-storage and multi-device parameters use its original step.
"""

# MIT License
# Copyright (c) 2023 Konstantin Mishchenko
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from importlib.metadata import version

import torch
from prodigyopt import Prodigy as OriginalProdigy

_SUPPORTED_VERSION = version("prodigyopt") == "1.1.2"
_UPDATE_BYTES = 64 * 1024 * 1024


def _parameter_batches(parameters):
    """Bound denominator scratch to 64 MiB or one larger parameter."""
    batch, size = [], 0
    for parameter in parameters:
        nbytes = parameter.numel() * parameter.element_size()
        if batch and size + nbytes > _UPDATE_BYTES:
            yield batch
            batch, size = [], 0
        batch.append(parameter)
        size += nbytes
    if batch:
        yield batch


class Prodigy(OriginalProdigy):
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        parameters = [p for group in self.param_groups for p in group["params"]]
        devices = {p.device for p in parameters}
        compatible = (
            _SUPPORTED_VERSION
            and len(devices) == 1
            and next(iter(devices)).type == "cuda"
            and self.param_groups[0]["decouple"]
            and not any(group["fsdp_in_use"] for group in self.param_groups)
            and not any(hasattr(p, "_fsdp_flattened") for p in parameters)
            and all(type(p) in (torch.nn.Parameter, torch.Tensor) for p in parameters)
            and not any(
                torch.is_tensor(group[key])
                for group in self.param_groups
                for key in ("lr", "d", "d_numerator")
            )
        )
        if compatible:
            storages = [p.untyped_storage().data_ptr() for p in parameters]
            compatible = len(set(storages)) == len(storages)
        if not compatible:
            original = OriginalProdigy.step
            # The outer optimizer wrapper has already fired this instance's hooks.
            if getattr(original, "hooked", False):
                original = original.__wrapped__
            result = original(self)
            return loss if closure is not None else result

        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        beta3 = group["beta3"] if group["beta3"] is not None else math.sqrt(beta2)
        k = group["k"]
        d, d_max = group["d"], group["d_max"]
        d_coef = group["d_coef"]
        lr = max(group["lr"] for group in self.param_groups)
        correction = (
            (1 - beta2 ** (k + 1)) ** 0.5 / (1 - beta1 ** (k + 1))
            if group["use_bias_correction"]
            else 1
        )
        dlr = d * lr * correction
        growth_rate = group["growth_rate"]
        d_numerator = group["d_numerator"] * beta3
        numerators, denominators = [], []
        coefficients = []

        for group in self.param_groups:
            d0, group_lr = group["d0"], group["lr"]
            slice_p = group["slice_p"]
            if group_lr not in (lr, 0.0):
                raise RuntimeError(
                    "Setting different lr values in different parameter groups is only supported for values of 0"
                )
            gradients, states = [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["s"] = torch.zeros_like(p.data.flatten()[::slice_p]).detach()
                    state["p0"] = (
                        p.flatten()[::slice_p].detach().clone()
                        if p.any()
                        else torch.tensor(0, device=p.device, dtype=p.dtype)
                    )
                    if beta1 > 0:
                        state["exp_avg"] = torch.zeros_like(p.data).detach()
                    state["exp_avg_sq"] = torch.zeros_like(p.data).detach()
                if group_lr > 0:
                    sliced_grad = grad.flatten()[::slice_p]
                    numerators.append(
                        torch.dot(
                            sliced_grad, state["p0"].data - p.data.flatten()[::slice_p]
                        )
                    )
                    coefficients.append((d / d0) * dlr)
                    gradients.append(grad)
                    states.append(state)
            if states:
                if beta1 > 0:
                    averages = [state["exp_avg"] for state in states]
                    torch._foreach_mul_(averages, beta1)
                    torch._foreach_add_(averages, gradients, alpha=d * (1 - beta1))
                averages_sq = [state["exp_avg_sq"] for state in states]
                torch._foreach_mul_(averages_sq, beta2)
                torch._foreach_addcmul_(
                    averages_sq, gradients, gradients, value=d * d * (1 - beta2)
                )
                sums = [state["s"] for state in states]
                torch._foreach_mul_(sums, beta3)
                alpha = (d / d0) * (d if group["safeguard_warmup"] else dlr)
                torch._foreach_add_(
                    sums, [grad.flatten()[::slice_p] for grad in gradients], alpha=alpha
                )
                denominators.extend(state["s"].abs().sum() for state in states)

        if not numerators:
            return loss
        # Keep the original Python-float summation order and precision.
        values = torch.stack(numerators + denominators).cpu().tolist()
        delta_numerator = d_denom = 0.0
        for i, coefficient in enumerate(coefficients):
            delta_numerator += coefficient * values[i]
            d_denom += values[len(numerators) + i]
        if d_denom == 0:
            return loss
        global_d_numerator = d_numerator + delta_numerator
        if lr > 0:
            d_hat = d_coef * global_d_numerator / d_denom
            if d == group["d0"]:
                d = max(d, d_hat)
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

        for group in self.param_groups:
            group["d_numerator"] = global_d_numerator
            group["d_denom"] = d_denom
            group["d"], group["d_max"], group["d_hat"] = d, d_max, d_hat
            decay, eps, k = group["weight_decay"], group["eps"], group["k"]
            active = [p for p in group["params"] if p.grad is not None]
            for batch in _parameter_batches(active):
                states = [self.state[p] for p in batch]
                for state in states:
                    state["step"] += 1
                denominator = torch._foreach_sqrt(
                    [state["exp_avg_sq"] for state in states]
                )
                torch._foreach_add_(denominator, d * eps)
                data = [p.data for p in batch]
                if decay != 0:
                    torch._foreach_add_(data, data, alpha=-decay * dlr)
                if beta1 > 0:
                    torch._foreach_addcdiv_(
                        data,
                        [state["exp_avg"] for state in states],
                        denominator,
                        value=-dlr,
                    )
                else:
                    torch._foreach_addcdiv_(
                        data, [p.grad.data for p in batch], denominator, value=-dlr * d
                    )
                del denominator
            group["k"] = k + 1
        return loss
