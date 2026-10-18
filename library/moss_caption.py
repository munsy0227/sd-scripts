"""MOSS caption sidecars and layout geometry. Caption contents are data only."""

import copy
import hashlib
import json
import math
from pathlib import Path


class StructuredCaption(str):
    """A natural-language JSON caption, including its intentional line breaks."""


def parse_caption(text, caption_format="auto", length="long", source="caption"):
    if caption_format == "text":
        return text
    if caption_format not in ("auto", "moss_caption", "moss_layout"):
        raise ValueError(f"{source}: invalid caption_format {caption_format!r}")
    if length not in ("long", "medium", "short"):
        raise ValueError(f"{source}: invalid moss_caption_length {length!r}")
    if caption_format == "auto" and not text.lstrip().startswith("{"):
        return text
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        if caption_format == "moss_caption" or (caption_format == "auto" and "caption" in data):
            value = data.get("caption", {}).get(length)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"caption.{length} must be a nonempty string")
            return StructuredCaption(value)
        if caption_format == "moss_layout" or "summary" in data or "items" in data:
            validate_layout(data)
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        raise ValueError("expected caption.long/medium/short or summary/items")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"{source}: {exc}") from exc


def read_caption(path, caption_format, length, enable_wildcard=False, enable_multiline_caption=False):
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"{path}: caption file is empty")
    if caption_format == "auto" and Path(path).suffix.lower() == ".json" and not text.startswith("{"):
        raise ValueError(f"{path}: expected a JSON caption object")
    value = parse_caption(text, caption_format, length, str(path))
    if is_layout(value) or isinstance(value, StructuredCaption):
        return value
    return value if enable_wildcard or enable_multiline_caption else value.splitlines()[0]


def is_layout(caption):
    if not isinstance(caption, str) or not caption.lstrip().startswith("{"):
        return False
    try:
        data = json.loads(caption)
    except ValueError:
        return False
    return isinstance(data, dict) and "summary" in data and "items" in data


def validate_layout(data):
    if not isinstance(data.get("summary"), dict) or not isinstance(data["summary"].get("caption"), str):
        raise ValueError("summary.caption must be a string")
    if not isinstance(data.get("items"), list):
        raise ValueError("items must be a list")
    if data.get("bbox_format") != "yxyx" or data.get("bbox_coord_space") != "abs_centralized":
        raise ValueError("layout requires bbox_format=yxyx and bbox_coord_space=abs_centralized")
    for item in data["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("caption"), str) or not isinstance(item.get("label"), str):
            raise ValueError("each layout item requires label and caption strings")
        box = item.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in box
        ):
            raise ValueError("bbox must contain four finite numbers")
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("bbox must have positive width and height")


def center_geometry(info):
    ow, oh = info.image_size
    rw, rh = info.resized_size
    tw, th = info.bucket_reso
    return {"original": (ow, oh), "resized": (rw, rh), "target": (tw, th), "crop": ((rw - tw) // 2, (rh - th) // 2)}


def render_layout(caption, geometry=None, flipped=False, prefix=None, suffix=None):
    data = copy.deepcopy(json.loads(caption))
    validate_layout(data)
    data["summary"]["caption"] = " ".join(x for x in (prefix, data["summary"]["caption"], suffix) if x)
    if geometry is not None:
        ow, oh = geometry["original"]
        rw, rh = geometry["resized"]
        tw, th = geometry["target"]
        cx, cy = geometry["crop"]
        items = []
        for item in data["items"]:
            y0, x0, y1, x1 = item["bbox"]
            x0, x1 = [(x + ow / 2) * rw / ow - cx for x in (x0, x1)]
            y0, y1 = [(y + oh / 2) * rh / oh - cy for y in (y0, y1)]
            if flipped:
                x0, x1 = tw - x1, tw - x0
            x0, x1 = max(0, x0), min(tw, x1)
            y0, y1 = max(0, y0), min(th, y1)
            if x0 >= x1 or y0 >= y1:
                continue
            item["bbox"] = [math.floor(y0 - th / 2), math.floor(x0 - tw / 2), math.ceil(y1 - th / 2), math.ceil(x1 - tw / 2)]
            items.append(item)
        data["items"] = items
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def fingerprint(caption, settings):
    return hashlib.sha256(json.dumps([caption, settings], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
