from unittest.mock import patch

import pytest
import torch

from library.attention import AttentionParams, attention


@pytest.mark.parametrize("lengths", [[3], [3, 3], [3, 5]])
def test_masked_attention_compiles_without_device_scalar_reads(lengths):
    torch.manual_seed(19)
    batch, image_tokens, text_tokens = len(lengths), 2, 5
    mask = torch.arange(text_tokens)[None, :] < torch.tensor(lengths)[:, None]
    params = AttentionParams.create_attention_params_from_mask(
        "torch", False, image_tokens, mask
    )
    q, k, v = [
        torch.randn(batch, image_tokens + text_tokens, 2, 4, requires_grad=True)
        for _ in range(3)
    ]

    def forward(q, k, v):
        return attention([q, k, v], attn_params=params)

    if len(set(lengths)) == 1:
        valid = image_tokens + lengths[0]
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                q[:, :valid].transpose(1, 2),
                k[:, :valid].transpose(1, 2),
                v[:, :valid].transpose(1, 2),
            )
            .transpose(1, 2)
            .flatten(2)
        )
        expected = torch.nn.functional.pad(
            expected, (0, 0, 0, image_tokens + text_tokens - valid)
        )
    else:
        key_mask = torch.nn.functional.pad(mask, (image_tokens, 0), value=True)
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=key_mask[:, None, None, :],
            )
            .transpose(1, 2)
            .flatten(2)
        )

    actual = torch.compile(forward, backend="eager", fullgraph=True)(q, k, v)
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(expected)
    for got, wanted in zip(
        torch.autograd.grad(actual, (q, k, v), grad),
        torch.autograd.grad(expected, (q, k, v), grad),
    ):
        torch.testing.assert_close(got, wanted)
    with patch("torch.all", side_effect=AssertionError("repeated GPU scalar check")):
        repeated = forward(q, k, v)
    torch.testing.assert_close(repeated, expected)
    torch._dynamo.reset()
