"""Opt-in round trip through the installed ComfyUI loader and patch engine."""

import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from library.moss_image_models import MossImage as MOSSImageTransformer


@pytest.fixture(scope="module")
def comfy():
    root = os.environ.get("MOSS_COMFYUI_PATH")
    if not root:
        pytest.skip("Set MOSS_COMFYUI_PATH for actual ComfyUI compatibility tests")
    sys.path.insert(0, root)
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import comfy.lora
    import comfy.model_base
    return SimpleNamespace(lora=comfy.lora, model_base=comfy.model_base)


def fused_state(state):
    result = {"diffusion_model." + k: v.detach().clone() for k, v in state.items() if k != "rope.bands"}
    for key in list(result):
        if key.endswith(".proj.q.weight"):
            prefix = key[:-len("q.weight")]
            result[prefix + "qkv.weight"] = torch.cat([result.pop(prefix + part + ".weight") for part in ("q", "k", "v")])
    return result


def mapping(comfy, state):
    model = object.__new__(comfy.model_base.MOSSImage)
    torch.nn.Module.__init__(model)
    model.model_config = SimpleNamespace(unet_config={"image_model": "moss_image"})
    model.state_dict = lambda: state
    return comfy.lora.model_lora_keys_unet(model, {})


def test_all_production_scope_aliases(comfy):
    from lycoris.config import MOSS_BLOCK_TARGETS, MOSS_FULL_TARGETS
    import re
    model = MOSSImageTransformer(dim=32, ffn_dim=48, num_heads=4, num_layers=42, text_dim=16, freq_dim=16)
    keys = mapping(comfy, fused_state(model.state_dict()))
    for targets, count in ((MOSS_BLOCK_TARGETS, 252), (MOSS_FULL_TARGETS, 269)):
        paths = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
                 and any(re.fullmatch(pattern, name) for pattern in targets)]
        assert len(paths) == count
        for prefix in ("lycoris_", "lora_unet_"):
            assert all(prefix + path.replace(".", "_") in keys for path in paths)


@pytest.mark.parametrize("preset", ["moss", "moss-full"])
@pytest.mark.parametrize("prefix", ["lycoris_", "lora_unet_"])
@pytest.mark.parametrize("algo,dora,output_axis", [("lora", False, False),
    ("lokr", False, False), ("lokr", True, False), ("lokr", True, True)])
@pytest.mark.parametrize("strength", [0., 0.5, 1.])
def test_portable_weights_match_lycoris_merge(comfy, tmp_path, caplog, preset, prefix, algo, dora, output_axis, strength):
    from lycoris import create_lycoris
    from safetensors.torch import load_file
    torch.manual_seed(41)
    model = MOSSImageTransformer(dim=32, ffn_dim=48, num_heads=4, num_layers=2, text_dim=16, freq_dim=16)
    original = fused_state(model.state_dict())
    net = create_lycoris(model, preset=preset, algo=algo, linear_dim=4, linear_alpha=7,
                         full_matrix=True, factor=4, weight_decompose=dora, wd_on_output=output_axis)
    net.apply_to()
    with torch.no_grad():
        for name, param in net.named_parameters():
            if name.endswith(("lora_up.weight", "lokr_w2")):
                param.normal_(0, 0.02)
    file = tmp_path / "adapter.safetensors"
    net.save_weights(str(file), dtype=torch.float32, metadata=None)
    portable = {prefix + key[len("lycoris_"):]: value for key, value in load_file(file).items()}
    patches = comfy.lora.load_lora(portable, mapping(comfy, original))
    assert len(patches) == (12 if preset == "moss" else 29)
    assert not any("not loaded" in rec.message for rec in caplog.records)
    grouped = defaultdict(list)
    for key, patch in patches.items():
        if isinstance(key, tuple):
            key, offset = key
        else:
            offset = None
        grouped[key].append((strength, patch, 1., offset, None))
    result = {key: comfy.lora.calculate_weight(grouped[key], value.clone(), key) if key in grouped else value
              for key, value in original.items()}
    net.restore()
    net.merge_to(strength)
    expected = fused_state(model.state_dict())
    for key in expected:
        torch.testing.assert_close(result[key], expected[key], rtol=2e-5, atol=2e-6, msg=key)
