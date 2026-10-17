import json
from types import SimpleNamespace

import numpy as np
import pytest

from library import moss_caption


def layout():
    return {"summary": {"caption": "女の子, green hair", "meta": ["anime"]},
            "items": [{"label": "character", "caption": "waving", "bbox": [-25, -50, 25, 50]}],
            "bbox_format": "yxyx", "bbox_coord_space": "abs_centralized"}


def test_anima_txt_is_unchanged(tmp_path):
    caption = "style_trigger, character_name, 1girl, solo, green hair, waving, This anime illustration depicts a girl with green hair. She waves against a plaid background."
    path = tmp_path / "日本語.txt"
    path.write_text(caption, encoding="utf-8")
    assert moss_caption.read_caption(path, "auto", "long") == caption


@pytest.mark.parametrize("length", ["long", "medium", "short"])
def test_json_selected_field_and_multiline(tmp_path, length):
    path = tmp_path / "한글.json"
    data = {"caption": {key: key + " 한국어 日本語\nsecond line" for key in ["long", "medium", "short"]},
            "watermark": [{"label": "not a prompt"}], "borders": ["left"]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8-sig")
    actual = moss_caption.read_caption(path, "auto", length)
    assert actual == data["caption"][length]
    assert isinstance(actual, moss_caption.StructuredCaption)


@pytest.mark.parametrize("text", ['{bad json', '{"caption":{"long":""}}', '{"caption":{}}', '{"summary":{}}'])
def test_invalid_json_has_source_path(text):
    with pytest.raises(ValueError, match="bad.json"):
        moss_caption.parse_caption(text, source="bad.json")


def test_json_sidecar_rejects_non_json_text(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("not JSON")
    with pytest.raises(ValueError, match="broken.json"):
        moss_caption.read_caption(path, "auto", "long")


def test_layout_resize_crop_flip_independent_geometry():
    # Source canvas 200x100. Resize 2x; crop x=100, y=20 to 200x160.
    geo = {"original": (200, 100), "resized": (400, 200), "target": (200, 160), "crop": (100, 20)}
    result = json.loads(moss_caption.render_layout(json.dumps(layout()), geo, prefix="trigger"))
    assert result["items"][0]["bbox"] == [-50, -100, 50, 100]
    assert result["summary"]["caption"].startswith("trigger ")
    source = layout()
    source["items"][0]["bbox"] = [-25, -80, 25, -20]
    result = json.loads(moss_caption.render_layout(json.dumps(source), geo, flipped=True))
    assert result["items"][0]["bbox"] == [-50, 40, 50, 100]


def test_layout_clipping_drops_invisible_items_and_keeps_schema():
    data = layout()
    data["items"] += [{"label": "text", "caption": "off canvas", "text": "hello", "bbox": [-10, 80, 10, 90]}]
    geo = {"original": (200, 100), "resized": (200, 100), "target": (100, 100), "crop": (50, 0)}
    result = json.loads(moss_caption.render_layout(json.dumps(data), geo))
    assert len(result["items"]) == 1
    assert result["summary"]["meta"] == ["anime"]
    assert result["bbox_coord_space"] == "abs_centralized"


def test_actual_random_crop_is_reported(monkeypatch):
    from library import utils
    monkeypatch.setattr(utils.random, "randint", lambda low, high: high)
    transform = {}
    image = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    cropped, _, _ = utils.trim_and_resize_if_required(True, image, (4, 4), (8, 4), transform=transform)
    np.testing.assert_array_equal(cropped, image[:, 4:])
    assert transform["crop"] == (4, 0)


def test_json_newlines_survive_dataset_processing():
    from library.dataset import BaseDataset
    subset = SimpleNamespace(caption_format="moss_caption", caption_prefix="trigger", caption_suffix=None,
        caption_dropout_rate=0, caption_dropout_every_n_epochs=0, enable_wildcard=False,
        shuffle_caption=False, token_warmup_step=0, caption_tag_dropout_rate=0, secondary_separator=None)
    dataset = SimpleNamespace(current_epoch=1, replacements={})
    text = moss_caption.StructuredCaption("first\nsecond")
    assert BaseDataset.process_caption(dataset, subset, text) == "trigger first\nsecond"


def test_cache_fingerprint_tracks_caption_and_settings():
    base = moss_caption.fingerprint("caption", {"length": "long", "crop": [0, 0]})
    assert base != moss_caption.fingerprint("edited", {"length": "long", "crop": [0, 0]})
    assert base != moss_caption.fingerprint("caption", {"length": "short", "crop": [0, 0]})
    assert base != moss_caption.fingerprint("caption", {"length": "long", "crop": [1, 0]})
