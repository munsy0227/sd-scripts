import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from library import moss_models, moss_train_utils, strategy_base, strategy_moss


def tiny_model():
    return moss_models.MOSSImageTransformer(dim=32, ffn_dim=48, num_heads=4, num_layers=2,
                                            text_dim=16, freq_dim=16, latent_channels=128)


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float32), ("cuda", torch.bfloat16)])
@pytest.mark.parametrize("preset", ["moss", "moss-full"])
@pytest.mark.parametrize("algo,dora", [("lora", False), ("lokr", False), ("lokr", True)])
def test_tiny_adapter_backward_checkpoint_roundtrip(tmp_path, device, dtype, preset, algo, dora):
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
    t = torch.tensor([100., 600.], device=device)
    baseline = model(z, t, text, mask).sample.detach()
    base_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
    kwargs = dict(preset=preset, algo=algo, linear_dim=4, linear_alpha=4)
    if algo == "lokr":
        kwargs.update(full_matrix=True, factor=4, weight_decompose=dora)
    network = create_lycoris(model, **kwargs)
    network.apply_to()
    network.to(device=device, dtype=dtype)
    torch.testing.assert_close(model(z, t, text, mask).sample, baseline)
    opt = torch.optim.AdamW(network.parameters(), lr=0.01)
    prediction = model(z, t, text, mask).sample
    prediction.float().square().mean().backward()
    grads = [p.grad for p in network.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) for g in grads)
    opt.step()
    native = copy.deepcopy(network.state_dict())
    optimizer_state = copy.deepcopy(opt.state_dict())
    after = model(z, t, text, mask).sample.detach()
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
    torch.testing.assert_close(model(z, t, text, mask).sample, after, rtol=tolerance, atol=tolerance)
    restored.restore()
    restored = create_lycoris(model, **kwargs)
    restored.apply_to()
    restored.to(device=device, dtype=dtype)
    restored.load_state_dict(native)
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    resumed_optimizer.load_state_dict(optimizer_state)
    assert resumed_optimizer.state
    resumed_optimizer.zero_grad()
    model(z, t, text, mask).sample.float().square().mean().backward()
    resumed_optimizer.step()
    assert torch.isfinite(model(z, t, text, mask).sample).all()


def test_vae_pack_normalize_decode_roundtrip():
    from library.moss_train_utils import decode_latents
    z = torch.arange(32 * 4 * 6, dtype=torch.float32).reshape(1, 32, 4, 6) / 100
    vae = SimpleNamespace(device="cpu", dtype=torch.float32,
        bn=SimpleNamespace(running_mean=torch.arange(128).float() / 128, running_var=torch.ones(128) * 2),
        config=SimpleNamespace(batch_norm_eps=1e-6),
        encode=lambda _: SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: z)),
        decode=lambda value, return_dict=False: (value / 10,))
    packed = strategy_moss.encode_latents(vae, None)
    assert packed.shape == (1, 128, 2, 3)
    torch.testing.assert_close(decode_latents(vae, packed), z / 10)


def test_flow_target_independent_identity():
    scheduler = moss_train_utils.MOSSImageScheduler(training_shift=1.7)
    clean = torch.randn(4, 128, 2, 2)
    noisy, target, times = moss_train_utils.flow_batch(clean, scheduler)
    sigma = (times / 1000).view(-1, 1, 1, 1)
    torch.testing.assert_close(noisy, clean + sigma * target)
    assert noisy.dtype == target.dtype == torch.float32
    scheduler.set_timesteps(10)
    assert torch.all(scheduler.sigmas[:-1] >= scheduler.sigmas[1:])


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
        return "user:" + messages[0]["content"] + "assistant:<think>"

    def __call__(self, texts, return_tensors, truncation, padding, max_length):
        rows = [[ord(c) % 120 + 1 for c in text][:max_length] for text in texts]
        ids = torch.zeros(len(rows), max_length, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 16)

    @property
    def device(self):
        return self.embedding.weight.device

    def forward(self, input_ids, attention_mask, use_cache):
        assert attention_mask.all() and not use_cache
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(128)
        self.config = SimpleNamespace(batch_norm_eps=1e-5)

    @property
    def device(self):
        return self.bn.running_mean.device

    @property
    def dtype(self):
        return self.bn.running_mean.dtype

    def encode(self, images):
        z = torch.nn.functional.avg_pool2d(images, 8).repeat(1, 11, 1, 1)[:, :32]
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: z))


@pytest.mark.parametrize("caption_kind", ["txt", "caption", "layout"])
def test_training_loop_save_and_resume_with_real_dataset(tmp_path, monkeypatch, caption_kind):
    import moss_train_network as training
    model_root = tmp_path / "model"
    for name in ["tokenizer", "text_encoder", "transformer", "vae"]:
        (model_root / name).mkdir(parents=True)
    moss_train_utils.MOSSImageScheduler().save_pretrained(model_root / "scheduler")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.full((64, 80, 3), 180, dtype=np.uint8)).save(image_dir / "sample.png")
    extension = ".txt" if caption_kind == "txt" else ".json"
    if caption_kind == "caption":
        caption = json.dumps({"caption": {"long": "style_trigger, A waving girl.\nGreen hair.", "medium": "A girl.", "short": "Girl."}, "watermark": [], "borders": ["none"]}, indent=2)
    elif caption_kind == "layout":
        caption = json.dumps({"summary": {"caption": "A waving girl."}, "items": [{"label": "character", "caption": "green hair", "bbox": [-24, -32, 24, 32]}], "bbox_format": "yxyx", "bbox_coord_space": "abs_centralized"}, indent=2)
    else:
        caption = "style_trigger, 1girl, green hair, A waving girl."
    (image_dir / ("sample" + extension)).write_text(caption, encoding="utf-8")
    config = tmp_path / "dataset.toml"
    config.write_text('[general]\ncaption_extension='+json.dumps(extension)+'\n[[datasets]]\nresolution=64\nbatch_size=1\nenable_bucket=true\nmin_bucket_reso=64\nmax_bucket_reso=128\nbucket_reso_steps=16\n[[datasets.subsets]]\nimage_dir='+json.dumps(str(image_dir))+'\n')
    monkeypatch.setattr(strategy_moss.AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    monkeypatch.setattr(training.AutoModel, "from_pretrained", lambda *a, **k: FakeEncoder().eval())
    monkeypatch.setattr(training.AutoencoderKLFlux2, "from_pretrained", lambda *a, **k: FakeVAE().eval())
    monkeypatch.setattr(moss_models.MOSSImageTransformer, "from_pretrained", lambda *a, **k: tiny_model())
    common = ["--pretrained_model_name_or_path", str(model_root), "--dataset_config", str(config),
              "--output_dir", str(tmp_path / "output"), "--output_name", "moss", "--save_model_as", "safetensors",
              "--network_dim", "4", "--network_args", "algo=lokr", "full_matrix=True", "weight_decompose=True", "factor=4",
              "--optimizer_type", "AdamW", "--learning_rate", "0.001", "--lr_scheduler", "constant",
              "--max_train_steps", "1", "--max_data_loader_n_workers", "0", "--seed", "42",
              "--cache_latents_to_disk", "--cache_text_encoder_outputs_to_disk", "--gradient_checkpointing",
              "--moss_max_token_length", "24", "--save_every_n_steps", "1", "--save_state"]
    def run(extra):
        for cls in [strategy_base.TokenizeStrategy, strategy_base.TextEncodingStrategy,
                    strategy_base.LatentsCachingStrategy, strategy_base.TextEncoderOutputsCachingStrategy]:
            monkeypatch.setattr(cls, "_strategy", None)
        args = training.setup_parser().parse_args(common + extra)
        training.MossNetworkTrainer().train(args)
    run([])
    assert list((tmp_path / "output").glob("*.safetensors"))
    states = sorted((tmp_path / "output").glob("*-state"))
    assert states
    run(["--resume", str(states[-1]), "--max_train_steps", "2"])
    assert (tmp_path / "output" / "moss.safetensors").is_file()
    assert set((tmp_path / "output").glob("*-state")) > set(states)


def test_text_cache_rejects_changed_caption_and_length(tmp_path):
    identity = {"template": "test"}
    cache = strategy_moss.MossTextEncoderOutputsCachingStrategy(True, 1, identity)
    info = SimpleNamespace(absolute_path=str(tmp_path / "a.png"), image_key="a", caption="old caption")
    subset = SimpleNamespace(caption_format="auto", moss_caption_length="long")
    dataset = SimpleNamespace(process_caption=lambda subset, text, geometry: text)
    cache.prepare_info(info, subset, dataset)
    path = cache.get_outputs_npz_path(info.absolute_path)
    np.savez(path, prompt_embeds=np.zeros((2, 16)), attn_mask=np.ones(2), fingerprint=info.moss_cache_fingerprint)
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
    subset = SimpleNamespace(caption_format="moss_layout", face_crop_aug_range=None, shuffle_caption=False,
        enable_wildcard=False, token_warmup_step=0, caption_tag_dropout_rate=0, caption_dropout_rate=0,
        caption_dropout_every_n_epochs=0, random_crop=False, flip_aug=True)
    info = SimpleNamespace(image_key="a", absolute_path="image.png", caption='{"summary":{},"items":[]}')
    dataset = SimpleNamespace(batch_size=1, image_data={"a":info}, image_to_subset={"a":subset})
    group = SimpleNamespace(datasets=[dataset], verify_bucket_reso_steps=lambda _: None)
    with pytest.raises(ValueError, match="stochastic captions/layout"):
        MossNetworkTrainer().assert_extra_args(args, group, None)


def test_cuda_block_swapping_matches_unswapped_gradients():
    from lycoris import create_lycoris
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(91)
    baseline = moss_models.MOSSImageTransformer(dim=32, ffn_dim=48, num_heads=4, num_layers=4,
        text_dim=16, freq_dim=16, latent_channels=128).to(device="cuda", dtype=torch.bfloat16)
    swapped = copy.deepcopy(baseline)
    z = torch.randn(1, 128, 2, 2, device="cuda", dtype=torch.bfloat16)
    text = torch.randn(1, 4, 16, device="cuda", dtype=torch.bfloat16)
    t = torch.tensor([400.], device="cuda")
    results = []
    native = None
    for model, swap in [(baseline, False), (swapped, True)]:
        model.requires_grad_(False)
        model.enable_gradient_checkpointing()
        net = create_lycoris(model, preset="moss", algo="lokr", linear_dim=4, full_matrix=True,
                            factor=4, weight_decompose=True)
        net.apply_to()
        net.to(device="cuda", dtype=torch.bfloat16)
        if native is None:
            native = copy.deepcopy(net.state_dict())
        else:
            net.load_state_dict(native)
        if swap:
            model.enable_block_swap(2, torch.device("cuda"))
            model.prepare_block_swap_before_forward()
        opt = torch.optim.SGD(net.parameters(), lr=0.01)
        history = []
        for _ in range(2):
            opt.zero_grad()
            prediction = model(z, t, text).sample
            prediction.float().square().mean().backward()
            history.append((prediction.detach().clone(), {n:p.grad.detach().clone() for n,p in net.named_parameters() if p.grad is not None}))
            opt.step()
        results.append(history)
        if swap:
            model.prepare_block_swap_before_forward()
    for plain, swapped in zip(*results):
        torch.testing.assert_close(plain[0], swapped[0], rtol=0.03, atol=0.03)
        assert plain[1].keys() == swapped[1].keys()
        for name in plain[1]:
            torch.testing.assert_close(plain[1][name], swapped[1][name], rtol=0.03, atol=0.03)
