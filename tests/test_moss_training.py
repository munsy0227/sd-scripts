import copy
import json
import gc
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from library import (
    flux2_vae,
    moss_image_models as moss_models,
    moss_image_train_utils as moss_train_utils,
    moss_image_utils,
)
from library import strategy_base, strategy_moss_image as strategy_moss
from library.custom_offloading_utils import BlockSwapConfig


def tiny_model():
    return moss_models.MossImage(
        dim=32,
        ffn_dim=48,
        num_heads=4,
        num_layers=2,
        text_dim=16,
        freq_dim=16,
        latent_channels=128,
    )


@pytest.mark.parametrize(
    "device,dtype", [("cpu", torch.float32), ("cuda", torch.bfloat16)]
)
@pytest.mark.parametrize("preset", ["moss", "moss-full"])
@pytest.mark.parametrize(
    "algo,dora", [("lora", False), ("lokr", False), ("lokr", True)]
)
def test_tiny_adapter_backward_checkpoint_roundtrip(
    tmp_path, device, dtype, preset, algo, dora
):
    from lycoris import create_lycoris, create_lycoris_from_weights

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(23)
    model = tiny_model().to(device=device, dtype=dtype)
    model.requires_grad_(False)
    model.enable_gradient_checkpointing()
    z = torch.randn(2, 128, 2, 2, device=device, dtype=dtype)
    text = torch.randn(2, 5, 16, device=device, dtype=dtype)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], device=device)
    t = torch.tensor([100.0, 600.0], device=device)
    baseline = model(z, t, text, mask).detach()
    base_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    kwargs = dict(preset=preset, algo=algo, linear_dim=4, linear_alpha=4)
    if algo == "lokr":
        kwargs.update(full_matrix=True, factor=4, weight_decompose=dora)
    network = create_lycoris(model, **kwargs)
    network.apply_to()
    network.to(device=device, dtype=dtype)
    torch.testing.assert_close(model(z, t, text, mask), baseline)
    opt = torch.optim.AdamW(network.parameters(), lr=0.01)
    prediction = model(z, t, text, mask)
    prediction.float().square().mean().backward()
    grads = [p.grad for p in network.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) for g in grads)
    opt.step()
    native = copy.deepcopy(network.state_dict())
    optimizer_state = copy.deepcopy(opt.state_dict())
    after = model(z, t, text, mask).detach()
    assert not torch.equal(after, baseline)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, base_weights[k], rtol=0, atol=0)
    path = tmp_path / "adapter.safetensors"
    network.save_weights(str(path), dtype=dtype, metadata=None)
    network.restore()
    restored, _ = create_lycoris_from_weights(1.0, str(path), model)
    restored.apply_to()
    restored.to(device=device, dtype=dtype)
    tolerance = 0.03 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(
        model(z, t, text, mask), after, rtol=tolerance, atol=tolerance
    )
    restored.restore()
    restored = create_lycoris(model, **kwargs)
    restored.apply_to()
    restored.to(device=device, dtype=dtype)
    restored.load_state_dict(native)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    resumed_optimizer.load_state_dict(optimizer_state)
    assert resumed_optimizer.state
    resumed_optimizer.zero_grad()
    model(z, t, text, mask).float().square().mean().backward()
    resumed_optimizer.step()
    assert torch.isfinite(model(z, t, text, mask)).all()


def test_vae_pack_normalize_decode_roundtrip():
    z = torch.arange(32 * 4 * 6, dtype=torch.float32).reshape(1, 32, 4, 6) / 100
    mean, var = torch.arange(128).float() / 128, torch.ones(128) * 2
    packed = flux2_vae.normalize_latents(flux2_vae.patchify_latents(z), mean, var)
    assert packed.shape == (1, 128, 2, 3)
    raw = flux2_vae.unpatchify_latents(flux2_vae.denormalize_latents(packed, mean, var))
    torch.testing.assert_close(raw, z)


def test_flow_target_independent_identity():
    scheduler = moss_train_utils.load_training_schedule()
    scheduler.config.training_shift = 1.7
    clean = torch.randn(4, 128, 2, 2)
    noisy, target, times = moss_train_utils.flow_batch(clean, scheduler)
    sigma = (times / 1000).view(-1, 1, 1, 1)
    torch.testing.assert_close(noisy, clean + sigma * target)
    assert noisy.dtype == target.dtype == torch.float32
    sigmas = moss_train_utils.get_sampling_sigmas(10)
    assert torch.all(sigmas[:-1] >= sigmas[1:])


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(
            tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        return "user:" + messages[0]["content"] + "assistant:<think>"

    def __call__(self, texts, return_tensors, truncation, max_length, padding=None):
        texts = [texts] if isinstance(texts, str) else texts
        rows = [[ord(c) % 120 + 1 for c in text][:max_length] for text in texts]
        ids = torch.zeros(len(rows), max_length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(row)
            mask[i, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 16)

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        assert (attention_mask is None or attention_mask.all()) and not use_cache
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(128)
        self.bn_eps = flux2_vae.BN_EPS

    @property
    def device(self):
        return self.bn.running_mean.device

    @property
    def dtype(self):
        return self.bn.running_mean.dtype

    def encode(self, images):
        z = torch.nn.functional.avg_pool2d(images, 8).repeat(1, 11, 1, 1)[:, :32]
        return z

    def patchify_and_normalize(self, z):
        return flux2_vae.normalize_latents(
            flux2_vae.patchify_latents(z),
            self.bn.running_mean,
            self.bn.running_var,
            self.bn_eps,
        )

    def decode_from_dit(self, z):
        raw = flux2_vae.unpatchify_latents(
            flux2_vae.denormalize_latents(
                z, self.bn.running_mean, self.bn.running_var, self.bn_eps
            )
        )
        return torch.nn.functional.interpolate(raw[:, :3], scale_factor=8)


def _training_case(tmp_path, monkeypatch, caption_kind):
    import moss_train_network as training

    model_root = tmp_path / "model"
    for name in ["tokenizer", "text_encoder", "transformer", "vae"]:
        (model_root / name).mkdir(parents=True)
    for name in ["text_encoder", "transformer", "vae"]:
        (model_root / name / "model.safetensors").touch()
    (model_root / "tokenizer" / "tokenizer_config.json").write_text("{}")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.full((64, 80, 3), 180, dtype=np.uint8)).save(
        image_dir / "sample.png"
    )
    extension = ".txt" if caption_kind == "txt" else ".json"
    if caption_kind == "caption":
        caption = json.dumps(
            {
                "caption": {
                    "long": "style_trigger, A waving girl.\nGreen hair.",
                    "medium": "A girl.",
                    "short": "Girl.",
                },
                "watermark": [],
                "borders": ["none"],
            },
            indent=2,
        )
    elif caption_kind == "layout":
        caption = json.dumps(
            {
                "summary": {"caption": "A waving girl."},
                "items": [
                    {
                        "label": "character",
                        "caption": "green hair",
                        "bbox": [-24, -32, 24, 32],
                    }
                ],
                "bbox_format": "yxyx",
                "bbox_coord_space": "abs_centralized",
            },
            indent=2,
        )
    else:
        caption = "style_trigger, 1girl, green hair, A waving girl."
    (image_dir / ("sample" + extension)).write_text(caption, encoding="utf-8")
    config = tmp_path / "dataset.toml"
    config.write_text(
        "[general]\ncaption_extension="
        + json.dumps(extension)
        + "\n[[datasets]]\nresolution=64\nbatch_size=1\nenable_bucket=true\nmin_bucket_reso=64\nmax_bucket_reso=128\nbucket_reso_steps=16\n[[datasets.subsets]]\nimage_dir="
        + json.dumps(str(image_dir))
        + "\n"
    )
    monkeypatch.setattr(
        moss_image_utils, "load_tokenizer", lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setattr(
        moss_image_utils, "load_text_encoder", lambda *a, **k: FakeEncoder().eval()
    )
    monkeypatch.setattr(
        training.flux2_vae, "load_flux2_vae", lambda *a, **k: FakeVAE().eval()
    )
    monkeypatch.setattr(
        moss_image_utils, "load_moss_image_model", lambda *a, **k: tiny_model()
    )
    common = [
        "--pretrained_model_name_or_path",
        str(model_root),
        "--dataset_config",
        str(config),
        "--output_dir",
        str(tmp_path / "output"),
        "--output_name",
        "moss",
        "--save_model_as",
        "safetensors",
        "--network_dim",
        "4",
        "--network_args",
        "algo=lokr",
        "full_matrix=True",
        "weight_decompose=True",
        "factor=4",
        "--optimizer_type",
        "AdamW",
        "--learning_rate",
        "0.001",
        "--lr_scheduler",
        "constant",
        "--max_train_steps",
        "1",
        "--max_data_loader_n_workers",
        "0",
        "--seed",
        "42",
        "--cache_latents_to_disk",
        "--cache_text_encoder_outputs_to_disk",
        "--gradient_checkpointing",
        "--moss_max_token_length",
        "24",
        "--save_every_n_steps",
        "1",
        "--save_state",
    ]

    def run(extra):
        for cls in [
            strategy_base.TokenizeStrategy,
            strategy_base.TextEncodingStrategy,
            strategy_base.LatentsCachingStrategy,
            strategy_base.TextEncoderOutputsCachingStrategy,
        ]:
            monkeypatch.setattr(cls, "_strategy", None)
        args = training.setup_parser().parse_args(common + extra)
        training.MossNetworkTrainer().train(args)

    return run, model_root, config


@pytest.mark.parametrize("caption_kind", ["txt", "caption", "layout"])
def test_training_loop_save_and_resume_with_real_dataset(
    tmp_path, monkeypatch, caption_kind
):
    run, _, _ = _training_case(tmp_path, monkeypatch, caption_kind)
    run([])
    assert list((tmp_path / "output").glob("*.safetensors"))
    states = sorted((tmp_path / "output").glob("*-state"))
    assert states
    run(["--resume", str(states[-1]), "--max_train_steps", "2"])
    assert (tmp_path / "output" / "moss.safetensors").is_file()
    assert set((tmp_path / "output").glob("*-state")) > set(states)
    legacy = os.environ.get("MOSS_LEGACY_STATE")
    if legacy and caption_kind == "txt":
        run(["--resume", legacy, "--max_train_steps", "3"])
        assert (tmp_path / "output" / "moss-step00000003-state").is_dir()
        portable = str(Path(legacy).parent / "moss.safetensors")
        run(["--network_weights", portable, "--output_name", "legacy-portable"])
        assert (tmp_path / "output" / "legacy-portable.safetensors").is_file()


def test_bf16_cached_dataset_with_prodigy_save_and_resume(tmp_path, monkeypatch):
    import train_network

    seen = []
    original = train_network.NetworkTrainer.process_batch

    def process_batch(self, batch, *args, **kwargs):
        outputs = batch.get("text_encoder_outputs_list")
        if outputs is not None:
            assert outputs[0].dtype == outputs[2].dtype == torch.bfloat16
            assert outputs[0].element_size() == 2
            seen.append(outputs[0].numel())
        return original(self, batch, *args, **kwargs)

    monkeypatch.setattr(train_network.NetworkTrainer, "process_batch", process_batch)
    run, _, _ = _training_case(tmp_path, monkeypatch, "txt")
    flags = ["--optimizer_type", "Prodigy", "--learning_rate", "1.0"]
    run(flags)
    state = tmp_path / "output" / "moss-step00000001-state"
    assert state.is_dir() and seen
    run(flags + ["--resume", str(state), "--max_train_steps", "2"])
    assert (tmp_path / "output" / "moss-step00000002-state").is_dir()


def test_text_cache_rejects_changed_caption_and_length(tmp_path):
    identity = {"template": "test"}
    cache = strategy_moss.MossImageTextEncoderOutputsCachingStrategy(
        True, 1, False, identity=identity
    )
    info = SimpleNamespace(
        absolute_path=str(tmp_path / "a.png"), image_key="a", caption="old caption"
    )
    subset = SimpleNamespace(caption_format="auto", moss_caption_length="long")
    dataset = SimpleNamespace(
        process_caption_candidates_for_cache=lambda subset, text, geometry: [text]
    )
    cache.prepare_info(info, subset, dataset)
    path = cache.get_outputs_npz_path(info.absolute_path)
    np.savez(
        path,
        prompt_embeds=np.zeros((2, 16), dtype=np.uint16),
        attn_mask=np.ones(2),
        uncond_embeds=np.zeros((1, 16), dtype=np.uint16),
        uncond_mask=np.ones(1),
        caption_dropout_rate=0.0,
        moss_cache_version=2,
        moss_fingerprint=info.moss_cache_fingerprint,
    )
    assert cache.is_disk_cached_outputs_expected(path)
    info.caption = "new caption"
    cache.prepare_info(info, subset, dataset)
    assert not cache.is_disk_cached_outputs_expected(path)
    info.caption = "old caption"
    subset.moss_caption_length = "short"
    cache.prepare_info(info, subset, dataset)
    assert not cache.is_disk_cached_outputs_expected(path)


def test_layout_stochastic_geometry_rejects_text_cache():
    from moss_train_network import MossNetworkTrainer, setup_parser

    args = setup_parser().parse_args(["--cache_text_encoder_outputs"])
    subset = SimpleNamespace(
        caption_format="moss_layout",
        face_crop_aug_range=None,
        shuffle_caption=False,
        enable_wildcard=False,
        token_warmup_step=0,
        caption_tag_dropout_rate=0,
        caption_dropout_rate=0,
        caption_dropout_every_n_epochs=0,
        random_crop=False,
        flip_aug=True,
    )
    info = SimpleNamespace(
        image_key="a", absolute_path="image.png", caption='{"summary":{},"items":[]}'
    )
    dataset = SimpleNamespace(
        batch_size=1, image_data={"a": info}, image_to_subset={"a": subset}
    )
    group = SimpleNamespace(datasets=[dataset], verify_bucket_reso_steps=lambda _: None)
    with pytest.raises(ValueError, match="stochastic captions/layout"):
        MossNetworkTrainer().assert_extra_args(args, group, None)


@pytest.mark.parametrize("preset", ["moss", "moss-full"])
@pytest.mark.parametrize("h2d_only,ring_size,pinned", [
    (False, 2, False), (True, 1, False), (True, 2, False), (True, 2, True),
])
def test_cuda_block_swapping_matches_unswapped_gradients(preset, h2d_only, ring_size, pinned):
    from lycoris import create_lycoris

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(91)
    baseline = moss_models.MossImage(
        dim=32,
        ffn_dim=48,
        num_heads=4,
        num_layers=6,
        text_dim=16,
        freq_dim=16,
        latent_channels=128,
    ).to(device="cuda", dtype=torch.bfloat16)
    swapped = copy.deepcopy(baseline)
    z = torch.randn(1, 128, 2, 2, device="cuda", dtype=torch.bfloat16)
    text = torch.randn(1, 4, 16, device="cuda", dtype=torch.bfloat16)
    t = torch.tensor([400.0], device="cuda")
    results = []
    native = None
    for model, swap in [(baseline, False), (swapped, True)]:
        model.requires_grad_(False)
        model.enable_gradient_checkpointing()
        net = create_lycoris(
            model,
            preset=preset,
            algo="lokr",
            linear_dim=4,
            full_matrix=True,
            factor=4,
            weight_decompose=True,
        )
        net.apply_to()
        net.to(device="cuda", dtype=torch.bfloat16)
        if native is None:
            native = copy.deepcopy(net.state_dict())
        else:
            net.load_state_dict(native)
        if swap:
            model.enable_block_swap(4, BlockSwapConfig(
                torch.device("cuda"), supports_backward=True, h2d_only=h2d_only,
                ring_size=ring_size, use_pinned_memory=pinned,
            ))
            model.prepare_block_swap_before_forward()
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        history = []
        for _ in range(2):
            opt.zero_grad()
            prediction = model(z, t, text)
            prediction.float().square().mean().backward()
            history.append(
                (
                    prediction.detach().clone(),
                    {
                        n: p.grad.detach().clone()
                        for n, p in net.named_parameters()
                        if p.grad is not None
                    },
                )
            )
            opt.step()
        results.append(history)
        if swap:
            model.prepare_block_swap_before_forward()
    for plain, swapped in zip(*results):
        torch.testing.assert_close(plain[0], swapped[0], rtol=0.03, atol=0.03)
        assert plain[1].keys() == swapped[1].keys()
        for name in plain[1]:
            torch.testing.assert_close(
                plain[1][name], swapped[1][name], rtol=0.03, atol=0.03
            )


def test_text_cache_migration_preserves_old_bytes_and_tracks_settings(tmp_path):
    cache = strategy_moss.MossImageTextEncoderOutputsCachingStrategy(
        True, 1, False, identity={"max_length": 24}
    )
    info = SimpleNamespace(
        absolute_path=str(tmp_path / "a.png"), caption="old", caption_dropout_rate=0.0
    )
    subset = SimpleNamespace(caption_format="text", moss_caption_length="long")
    dataset = SimpleNamespace(
        process_caption_candidates_for_cache=lambda subset, text, geometry: [text]
    )
    info.text_encoder_outputs_npz = cache.get_outputs_npz_path(info.absolute_path)
    np.savez(
        info.text_encoder_outputs_npz,
        prompt_embeds=np.ones((2, 16)),
        attn_mask=np.ones(2),
        fingerprint="v1",
    )
    old_bytes = open(info.text_encoder_outputs_npz, "rb").read()
    cache.prepare_info(info, subset, dataset)
    assert not cache.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)
    tokenize = strategy_moss.MossImageTokenizeStrategy.__new__(
        strategy_moss.MossImageTokenizeStrategy
    )
    tokenize.tokenizer, tokenize.max_length, tokenize.pad_token_id = (
        FakeTokenizer(),
        24,
        0,
    )
    encoding = strategy_moss.MossImageTextEncodingStrategy()
    cache.cache_batch_outputs(
        tokenize, [FakeEncoder()], encoding, [info], [info.moss_cache_caption]
    )
    assert (
        open(info.text_encoder_outputs_npz + ".legacy-v1.bak", "rb").read() == old_bytes
    )
    assert cache.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)
    outputs = cache.load_outputs_npz(info.text_encoder_outputs_npz)
    assert len(outputs) == 5 and outputs[0].dtype == torch.bfloat16
    with np.load(info.text_encoder_outputs_npz) as data:
        assert data["prompt_embeds_0"].dtype == np.uint16
    info.caption = "updated"
    cache.prepare_info(info, subset, dataset)
    assert not cache.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)
    cache.cache_batch_outputs(
        tokenize, [FakeEncoder()], encoding, [info], [info.moss_cache_caption]
    )
    assert (tmp_path / "a_moss_te.npz.legacy-v1.bak.1").is_file()
    assert (
        open(info.text_encoder_outputs_npz + ".legacy-v1.bak", "rb").read() == old_bytes
    )
    cache.identity["max_length"] = 32
    cache.prepare_info(info, subset, dataset)
    assert not cache.is_disk_cached_outputs_expected(info.text_encoder_outputs_npz)


def test_latent_migration_uses_raw_mean_latents_and_source_identity(
    tmp_path, monkeypatch
):
    from library import caching

    image = tmp_path / "a.png"
    Image.new("RGB", (80, 64)).save(image)
    cache = strategy_moss.MossImageLatentsCachingStrategy(
        True, 1, False, identity="vae-a"
    )
    path = cache.get_latents_npz_path(str(image), (80, 64))
    info = SimpleNamespace(absolute_path=str(image), latents_npz=path)
    np.savez(
        path,
        latents_4x5=np.ones((128, 4, 5)),
        original_size=[80, 64],
        crop_ltrb=[0, 0, 80, 64],
    )
    old_bytes = open(path, "rb").read()
    assert not cache.is_disk_cached_latents_expected((80, 64), path, False, False)
    monkeypatch.setattr(
        caching,
        "load_images_and_masks_for_caching",
        lambda *a: (torch.ones(1, 3, 64, 80), [None], [(80, 64)], [(0, 0, 80, 64)]),
    )
    cache.cache_batch_latents(FakeVAE().eval(), [info], False, False, False)
    assert open(path + ".legacy-v1.bak", "rb").read() == old_bytes
    assert cache.is_disk_cached_latents_expected((80, 64), path, False, False)
    raw, *_ = cache.load_latents_from_disk(path, (80, 64))
    assert raw.shape == (32, 8, 10)
    np.testing.assert_array_equal(raw, np.ones_like(raw))
    cache.identity = "vae-b"
    cache.get_latents_npz_path(str(image), (80, 64))
    assert not cache.is_disk_cached_latents_expected((80, 64), path, False, False)
    cache.identity = "vae-a"
    Image.new("RGB", (80, 64), "red").save(image)
    cache.get_latents_npz_path(str(image), (80, 64))
    assert not cache.is_disk_cached_latents_expected((80, 64), path, False, False)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("reuse", [False, True])
def test_encoder_release_after_all_datasets_and_sample_prompts(
    tmp_path, monkeypatch, device, reuse
):
    import moss_train_network as training

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    events = []

    class TrackedEncoder(FakeEncoder):
        def to(self, target, *args, **kwargs):
            events.append(str(target))
            return super().to(target, *args, **kwargs)

    root = tmp_path / "model"
    (root / "transformer").mkdir(parents=True)
    (root / "transformer" / "model.safetensors").touch()
    prompts = tmp_path / "prompts.json"
    prompts.write_text(
        json.dumps([{"prompt": "sample", "negative_prompt": "negative"}])
    )
    args = training.setup_parser().parse_args(
        [
            "--pretrained_model_name_or_path",
            str(root),
            "--cache_text_encoder_outputs",
            "--moss_text_encoder_device",
            "accelerator",
            "--sample_prompts",
            str(prompts),
        ]
    )
    trainer = training.MossNetworkTrainer()
    tokenize = strategy_moss.MossImageTokenizeStrategy.__new__(
        strategy_moss.MossImageTokenizeStrategy
    )
    tokenize.tokenizer, tokenize.max_length, tokenize.pad_token_id = (
        FakeTokenizer(),
        24,
        0,
    )
    trainer.tokenize_strategy = tokenize
    encoding = strategy_moss.MossImageTextEncodingStrategy()
    cache = strategy_moss.MossImageTextEncoderOutputsCachingStrategy(True, 1, False)
    monkeypatch.setattr(strategy_base.TextEncodingStrategy, "_strategy", encoding)
    encoder = TrackedEncoder().eval()
    reference = weakref.ref(encoder)
    encoders = [encoder]
    subset = SimpleNamespace(caption_format="text", moss_caption_length="long")
    dataset = SimpleNamespace(
        process_caption_candidates_for_cache=lambda subset, text, geometry: [text]
    )

    class Group:
        def __init__(self, index):
            self.info = SimpleNamespace(
                absolute_path=str(tmp_path / f"{index}.png"),
                caption=f"caption {index}",
                caption_dropout_rate=0.0,
            )
            self.info.text_encoder_outputs_npz = cache.get_outputs_npz_path(
                self.info.absolute_path
            )

        def new_cache_text_encoder_outputs(self, models, accelerator):
            events.append("dataset")
            cache.prepare_info(self.info, subset, dataset)
            if not cache.is_disk_cached_outputs_expected(
                self.info.text_encoder_outputs_npz
            ):
                cache.cache_batch_outputs(
                    tokenize,
                    models,
                    encoding,
                    [self.info],
                    [self.info.moss_cache_caption],
                )

    groups = [Group(0), Group(1)]
    accelerator = SimpleNamespace(
        device=torch.device(device), wait_for_everyone=lambda: None
    )
    if reuse:
        for group in groups:
            group.new_cache_text_encoder_outputs(encoders, accelerator)
        events.clear()
    for group in groups:
        trainer.cache_text_encoder_outputs_if_needed(
            args, accelerator, None, FakeVAE(), encoders, group, torch.float32
        )
    assert events.count("dataset") == 2
    assert all(
        v.device.type == "cpu"
        for values in trainer.sample_conditions.values()
        for v in values
    )
    assert cache._uncond_embeds is None or cache._uncond_embeds.device.type == "cpu"
    del encoder

    def load_dit(*args, **kwargs):
        gc.collect()
        assert reference() is None
        assert events[-1] == "cpu"
        events.append("dit")
        return tiny_model()

    monkeypatch.setattr(moss_image_utils, "load_moss_image_model", load_dit)
    trainer.is_swapping_blocks = False
    trainer.load_unet_lazily(args, torch.float32, accelerator, encoders)
    assert events[-1] == "dit"


def test_explicit_component_paths_override_root(tmp_path):
    import moss_train_network as training

    root = tmp_path / "model"
    (root / "transformer").mkdir(parents=True)
    (root / "tokenizer").mkdir()
    args = training.setup_parser().parse_args(
        [
            "--pretrained_model_name_or_path",
            str(root),
            "--text_encoder",
            "/custom/te.safetensors",
            "--vae",
            "/custom/vae.safetensors",
        ]
    )
    paths = moss_image_utils.resolve_model_paths(args)
    assert (
        paths.text_encoder == "/custom/te.safetensors"
        and paths.vae == "/custom/vae.safetensors"
    )
    assert paths.dit == root / "transformer" and paths.tokenizer == str(
        root / "tokenizer"
    )


@pytest.mark.parametrize("fused,sharded", [(True, False), (False, True)])
def test_real_dit_loader_single_file_and_official_shards(
    tmp_path, monkeypatch, fused, sharded
):
    from safetensors.torch import save_file

    original = tiny_model().eval()
    weights = {k: v.detach().clone() for k, v in original.state_dict().items()}
    if fused:
        for key in list(weights):
            if key.endswith(".proj.q.weight"):
                prefix = key[: -len("q.weight")]
                weights[prefix + "qkv.weight"] = torch.cat(
                    [weights.pop(prefix + part + ".weight") for part in "qkv"]
                )
        weights = {"model.diffusion_model." + k: v for k, v in weights.items()}
    monkeypatch.setattr(
        moss_models,
        "MOSS_IMAGE_CONFIG",
        dict(
            dim=32,
            ffn_dim=48,
            num_heads=4,
            num_layers=2,
            text_dim=16,
            freq_dim=16,
            latent_channels=128,
        ),
    )
    split = moss_image_utils.split_fused_qkv
    monkeypatch.setattr(
        moss_image_utils,
        "split_fused_qkv",
        lambda key, tensor: split(key, tensor, dim=32),
    )
    if sharded:
        keys = list(weights)
        mapping = {}
        for part, subset in enumerate([keys[::2], keys[1::2]], 1):
            name = f"diffusion_pytorch_model-{part:05d}-of-00002.safetensors"
            save_file({k: weights[k] for k in subset}, str(tmp_path / name))
            mapping.update({k: name for k in subset})
        (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": mapping})
        )
    else:
        save_file(weights, str(tmp_path / "moss.safetensors"))
    path = moss_image_utils.checkpoint_file(tmp_path, "DiT")
    restored = moss_image_utils.load_moss_image_model(
        "cpu", path, "torch", False, "cpu", torch.float32
    ).eval()
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    z, text, t = torch.randn(1, 128, 2, 2), torch.randn(1, 3, 16), torch.tensor([300.0])
    torch.testing.assert_close(
        restored(z, t, text), original(z, t, text), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "device,dtype",
    [("cpu", torch.float32), ("cpu", torch.bfloat16), ("cuda", torch.bfloat16)],
)
def test_real_qwen_text_only_loader_drops_vision_and_preserves_hidden_states(
    tmp_path, monkeypatch, device, dtype
):
    from transformers import Qwen3_5TextConfig, Qwen3_5TextModel
    from safetensors.torch import save_file

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [1, 1, 2],
        },
    )
    original = Qwen3_5TextModel(config).eval()
    # Match from_pretrained: cast persistent weights, preserving FP32 rotary
    # buffers. A blanket .bfloat16() would quantize inv_freq in the reference.
    original.load_state_dict(
        {key: value.to(dtype=dtype) for key, value in original.state_dict().items()},
        assign=True,
    )
    original.to(device)
    assert original.rotary_emb.inv_freq.dtype == torch.float32
    weights = {
        "model.language_model." + k: v.detach().cpu().clone()
        for k, v in original.state_dict().items()
    }
    weights.update(
        {"model.visual.unused": torch.ones(2), "lm_head.weight": torch.ones(2)}
    )
    path = tmp_path / "qwen.safetensors"
    save_file(weights, str(path))
    monkeypatch.setattr(moss_image_utils, "load_text_encoder_config", lambda: config)
    restored = moss_image_utils.load_text_encoder(str(path), dtype, device)
    assert isinstance(restored, Qwen3_5TextModel) and not hasattr(restored, "visual")
    ids = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    expected = moss_image_utils.encode_tokens(original, ids, device)
    with torch.autocast(device_type=device, dtype=torch.float16):
        actual = moss_image_utils.encode_tokens(restored, ids, device)
    for before, after in zip(expected, actual):
        assert after.dtype == dtype
        torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sampling_returns_to_training(tmp_path, device):
    import moss_train_network as training
    from lycoris import create_lycoris

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    trainer = training.MossNetworkTrainer()
    model = moss_models.MossImage(
        dim=32,
        ffn_dim=48,
        num_heads=4,
        num_layers=4,
        text_dim=16,
        freq_dim=16,
        latent_channels=128,
    ).to(device=device, dtype=dtype)
    model.requires_grad_(False)
    model.enable_gradient_checkpointing()
    net = create_lycoris(
        model,
        preset="moss-full",
        algo="lokr",
        linear_dim=4,
        full_matrix=True,
        factor=4,
        weight_decompose=True,
    )
    net.apply_to()
    net.to(device=device, dtype=dtype)
    if device == "cuda":
        model.enable_block_swap(2, BlockSwapConfig(torch.device(device), supports_backward=True))
        model.prepare_block_swap_before_forward()
    prompts = tmp_path / "samples.json"
    prompts.write_text(
        json.dumps([{"prompt": "sample", "width": 32, "height": 32, "sample_steps": 2}])
    )
    args = training.setup_parser().parse_args(
        [
            "--sample_prompts",
            str(prompts),
            "--sample_every_n_steps",
            "1",
            "--output_dir",
            str(tmp_path),
            "--output_name",
            "sample",
            "--cache_latents",
        ]
    )
    trainer.sample_conditions = {
        "sample": [torch.randn(1, 3, 16), torch.ones(1, 3)],
        "": [torch.randn(1, 2, 16), torch.ones(1, 2)],
    }
    accelerator = SimpleNamespace(
        device=torch.device(device),
        trackers=[],
        is_main_process=True,
        unwrap_model=lambda model: model,
        autocast=lambda: torch.autocast(
            device_type=device, dtype=dtype, enabled=device == "cuda"
        ),
    )
    vae = FakeVAE().eval()
    trainer.sample_images(
        accelerator, args, None, 1, torch.device(device), vae, None, None, model
    )
    assert list((tmp_path / "sample").glob("*.png")) and vae.device.type == "cpu"
    prediction = model(
        torch.randn(1, 128, 2, 2, device=device, dtype=dtype),
        torch.tensor([200.0], device=device),
        torch.randn(1, 3, 16, device=device, dtype=dtype),
    )
    prediction.float().square().mean().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_caption_dropout_is_training_only(monkeypatch):
    from contextlib import nullcontext
    import moss_train_network as training

    encoding = strategy_moss.MossImageTextEncodingStrategy()
    monkeypatch.setattr(strategy_base.TextEncodingStrategy, "_strategy", encoding)
    trainer = training.MossNetworkTrainer()
    seen = []

    def model(z, t, context, mask):
        seen.append(context.clone())
        return torch.zeros_like(z)

    conditions = [
        torch.ones(1, 3, 16),
        torch.ones(1, 3),
        torch.full((1, 2, 16), 7.0),
        torch.ones(1, 2),
        torch.ones(1),
    ]
    args = training.setup_parser().parse_args([])
    accelerator = SimpleNamespace(device=torch.device("cpu"), autocast=nullcontext)
    for is_train in [False, True]:
        trainer.get_noise_pred_and_target(
            args,
            accelerator,
            moss_train_utils.load_training_schedule(),
            torch.randn(1, 128, 2, 2),
            {},
            conditions,
            model,
            None,
            torch.float32,
            True,
            is_train=is_train,
        )
    torch.testing.assert_close(seen[0], conditions[0])
    torch.testing.assert_close(seen[1], conditions[2])


@pytest.mark.parametrize("caption_kind", ["txt", "caption", "layout"])
def test_cache_tools_and_trainer_reuse_the_same_files(
    tmp_path, monkeypatch, caption_kind
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from tools import cache_latents, cache_text_encoder_outputs

    run, root, config = _training_case(tmp_path, monkeypatch, caption_kind)
    common = [
        "--moss_image",
        "--pretrained_model_name_or_path",
        str(root),
        "--dataset_config",
        str(config),
        "--caption_format",
        "auto",
        "--moss_max_token_length",
        "24",
        "--max_data_loader_n_workers",
        "0",
    ]
    for tool, extra in [
        (cache_latents, []),
        (cache_text_encoder_outputs, ["--cache_text_encoder_outputs_to_disk"]),
    ]:
        for cls in [
            strategy_base.TokenizeStrategy,
            strategy_base.TextEncodingStrategy,
            strategy_base.LatentsCachingStrategy,
            strategy_base.TextEncoderOutputsCachingStrategy,
        ]:
            monkeypatch.setattr(cls, "_strategy", None)
        tool.cache_to_disk(tool.setup_parser().parse_args(common + extra))
    caches = list((tmp_path / "images").glob("*.npz"))
    assert len(caches) == 2
    before = {path: path.read_bytes() for path in caches}
    run([])
    assert all(path.read_bytes() == value for path, value in before.items())
    assert not list((tmp_path / "images").glob("*.bak*"))
