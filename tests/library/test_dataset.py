"""Tests for aspect-ratio bucket selection."""

import pytest

from library.dataset import BucketManager


def make_bucket_manager(no_upscale: bool) -> BucketManager:
    manager = BucketManager(
        no_upscale=no_upscale,
        max_reso=(1024, 1024),
        min_size=256,
        max_size=1024,
        reso_steps=64,
    )
    manager.make_buckets()
    return manager


@pytest.mark.parametrize(
    "image_size, expected_bucket_reso, expected_resized_size",
    [
        ((2048, 1536), (1024, 768), (1024, 768)),
        ((1536, 2048), (768, 1024), (768, 1024)),
        ((2048, 2048), (1024, 1024), (1024, 1024)),
        ((1200, 800), (1024, 704), (1056, 704)),
        ((4096, 512), (1024, 256), (2048, 256)),
    ],
)
def test_no_upscale_uses_standard_bucket_when_image_only_needs_downscaling(
    image_size, expected_bucket_reso, expected_resized_size
):
    standard_manager = make_bucket_manager(no_upscale=False)
    no_upscale_manager = make_bucket_manager(no_upscale=True)

    standard_result = standard_manager.select_bucket(*image_size)
    no_upscale_result = no_upscale_manager.select_bucket(*image_size)

    assert no_upscale_result == standard_result
    assert standard_result[:2] == (expected_bucket_reso, expected_resized_size)
    assert no_upscale_result[1][0] <= image_size[0]
    assert no_upscale_result[1][1] <= image_size[1]


@pytest.mark.parametrize(
    "image_size, expected_bucket_reso",
    [
        ((512, 768), (512, 768)),
        ((500, 750), (448, 704)),
        ((1000, 1000), (960, 960)),
    ],
)
def test_no_upscale_uses_original_size_bucket_when_standard_bucket_requires_upscaling(
    image_size, expected_bucket_reso
):
    standard_manager = make_bucket_manager(no_upscale=False)
    no_upscale_manager = make_bucket_manager(no_upscale=True)

    standard_reso, standard_resized_size, _ = standard_manager.select_bucket(*image_size)
    bucket_reso, resized_size, _ = no_upscale_manager.select_bucket(*image_size)

    assert resized_size == image_size
    assert bucket_reso == expected_bucket_reso
    assert standard_reso != bucket_reso
    assert standard_resized_size[0] > image_size[0] or standard_resized_size[1] > image_size[1]


def test_no_upscale_uses_standard_bucket_at_scale_one_boundary():
    standard_manager = make_bucket_manager(no_upscale=False)
    no_upscale_manager = make_bucket_manager(no_upscale=True)

    standard_result = standard_manager.select_bucket(1024, 768)
    no_upscale_result = no_upscale_manager.select_bucket(1024, 768)

    assert no_upscale_result == standard_result
    assert no_upscale_result[:2] == ((1024, 768), (1024, 768))
