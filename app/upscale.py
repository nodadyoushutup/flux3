"""Restore a photograph, then enlarge it to an exact pixel size.

JPEG cleanup only moves the smooth band, so grain stays with the
photograph. A 4x photographic super-resolution model then rebuilds
edges, hair, and texture. Large-scale color and shading stay on a plain
resample of that cleaned frame; the model's lightness detail is mixed
in full. Nothing in this path is a text-to-image sample.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image, ImageOps

DEBLOCK_NAME = "scunet_color_real_gan.pth"
# RealESRGAN's photo weights are trained to denoise, which reads as
# plastic skin. UltraSharp is the same RRDB family with the texture left in.
UPSCALE_NAME = "4x-UltraSharp.pth"
# Share of the deblocker's smooth band to accept. The photograph keeps
# its own grain either way.
DEBLOCK_AMOUNT = 0.45
DEBLOCK_RADIUS = 3.0
# Full lightness detail from the upscaler. Color stays with the photograph,
# so the model cannot wash the grade or repaint skin.
DETAIL_AMOUNT = 1.0
LONG_EDGE = {
    "512": 512,
    "768": 768,
    "1K": 1024,
    "1.5K": 1536,
    "2K": 2048,
    "4K": 4096,
}

_MODELS: dict[str, object] = {}


def target_size(width: int, height: int, resolution: str) -> tuple[int, int]:
    """Return the output pixels for a studio size token.

    ``source`` keeps the photograph's own size. Any other token sets the
    long edge and keeps the photograph's aspect.

    Args:
        width: Source width in pixels.
        height: Source height in pixels.
        resolution: ``source``, ``512``, ``768``, ``1K``, ``1.5K``, ``2K``,
            or ``4K``.

    Returns:
        Even width and height.

    Raises:
        ValueError: If the size token or the source size is unusable.
    """
    if width < 1 or height < 1:
        raise ValueError("source image has no size")
    token = str(resolution or "source").strip() or "source"
    if token == "source":
        return _even(width), _even(height)
    long_edge = LONG_EDGE.get(token)
    if long_edge is None:
        raise ValueError(
            "resolution must be source, 512, 768, 1K, 1.5K, 2K, or 4K"
        )
    if width >= height:
        target_width = long_edge
        target_height = max(1, round(height * long_edge / width))
    else:
        target_height = long_edge
        target_width = max(1, round(width * long_edge / height))
    return _even(target_width), _even(target_height)


def detail_radius(width: int, height: int) -> float:
    """Return the split between facial form and reconstructed texture.

    A one-pixel split only puts grain back on a soft enlargement. Edges,
    hair, and pores sit further out, and they cover more pixels as the
    frame gets larger.

    Args:
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        Gaussian radius in pixels.
    """
    return max(3.0, min(width, height) * 0.005)


def restore(blob: bytes, width: int, height: int) -> bytes:
    """Deblock ``blob`` and scale it to ``width`` by ``height``.

    Args:
        blob: Encoded source photograph.
        width: Output width in pixels.
        height: Output height in pixels.

    Returns:
        PNG bytes.

    Raises:
        ValueError: If the bytes are not an image or the size is invalid.
    """
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    image = _open(blob)
    cleaned = _deblock(image)
    base = _fit(cleaned, width, height)
    enlarged = cleaned
    passes = 0
    while (
        passes < 2
        and max(width / enlarged.width, height / enlarged.height) > 1.05
    ):
        enlarged = _enlarge(enlarged)
        passes += 1
    if passes:
        detailed = _fit(enlarged, width, height)
        base = _transfer_detail(
            base,
            detailed,
            radius=detail_radius(width, height),
            amount=DETAIL_AMOUNT,
        )
    output = io.BytesIO()
    base.save(output, format="PNG")
    return output.getvalue()


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _mix_low(
    base: Image.Image,
    donor: Image.Image,
    *,
    radius: float,
    amount: float,
) -> Image.Image:
    """Keep ``base`` texture and move its smooth band toward ``donor``."""
    import numpy as np

    base_arr = np.asarray(base, dtype=np.float32)
    mixed = base_arr + amount * (
        _gaussian(donor, radius) - _gaussian(base, radius)
    )
    return Image.fromarray(np.clip(mixed, 0, 255).astype("uint8"), mode="RGB")


def _transfer_detail(
    base: Image.Image,
    detailed: Image.Image,
    *,
    radius: float,
    amount: float,
) -> Image.Image:
    """Keep ``base`` color and add ``detailed`` lightness above ``radius``."""
    import numpy as np
    from PIL import ImageFilter

    detailed_arr = np.asarray(detailed, dtype=np.float32)
    luma = (
        detailed_arr[..., 0] * 0.2126
        + detailed_arr[..., 1] * 0.7152
        + detailed_arr[..., 2] * 0.0722
    )
    plane = Image.fromarray(np.clip(luma, 0, 255).astype("uint8"), mode="L")
    low = np.asarray(
        plane.filter(ImageFilter.GaussianBlur(radius=radius)),
        dtype=np.float32,
    )
    mixed = np.asarray(base, dtype=np.float32) + (amount * (luma - low))[..., None]
    return Image.fromarray(np.clip(mixed, 0, 255).astype("uint8"), mode="RGB")


def _gaussian(image: Image.Image, radius: float):
    import numpy as np
    from PIL import ImageFilter

    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(blurred, dtype=np.float32)


def _open(blob: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(blob))
        image.load()
    except Exception as exc:
        raise ValueError("source image could not be read") from exc
    oriented = ImageOps.exif_transpose(image) or image
    return oriented.convert("RGB")


def _deblock(image: Image.Image) -> Image.Image:
    restored = _apply(_load(DEBLOCK_NAME), image)
    if restored.size != image.size:
        restored = _fit(restored, image.width, image.height)
    return _mix_low(
        image,
        restored,
        radius=DEBLOCK_RADIUS,
        amount=DEBLOCK_AMOUNT,
    )


def _enlarge(image: Image.Image) -> Image.Image:
    return _apply(_load(UPSCALE_NAME), image)


def _load(filename: str):
    cached = _MODELS.get(filename)
    if cached is not None:
        return cached
    import torch
    from spandrel import ModelLoader

    path = Path(os.environ.get("MODELS_DIR", "/models")) / "upscale" / filename
    if not path.is_file():
        raise RuntimeError(f"missing upscale weights: {path}")
    descriptor = ModelLoader().load_from_file(str(path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    descriptor.model.to(device).eval()
    _MODELS[filename] = descriptor
    return descriptor


def _apply(descriptor, image: Image.Image) -> Image.Image:
    import torch

    tile = int(os.environ.get("UPSCALE_TILE", "512"))
    overlap = int(os.environ.get("UPSCALE_OVERLAP", "64"))
    device = next(descriptor.model.parameters()).device
    source = _tensor(image)
    _, _, height, width = source.shape
    scale = int(descriptor.scale)
    output = torch.zeros((1, 3, height * scale, width * scale))
    weight = torch.zeros((1, 1, height * scale, width * scale))
    window = _window(min(tile, height), min(tile, width), overlap)
    for top, left, bottom, right in _tiles(height, width, tile, overlap):
        crop = source[:, :, top:bottom, left:right]
        with torch.inference_mode():
            rendered = descriptor.model(crop.to(device)).float().cpu()
        out_top, out_left = top * scale, left * scale
        out_bottom, out_right = bottom * scale, right * scale
        patch = window
        if rendered.shape[-2] != patch.shape[-2] or rendered.shape[-1] != patch.shape[-1]:
            patch = _window(rendered.shape[-2] // scale, rendered.shape[-1] // scale, overlap)
            patch = patch.repeat_interleave(scale, -1).repeat_interleave(scale, -2)
        else:
            patch = window.repeat_interleave(scale, -1).repeat_interleave(scale, -2)
        output[:, :, out_top:out_bottom, out_left:out_right] += rendered * patch
        weight[:, :, out_top:out_bottom, out_left:out_right] += patch
    merged = output / weight.clamp_min(1e-6)
    return _image(merged)


def _tiles(height: int, width: int, tile: int, overlap: int):
    step = max(1, tile - overlap)
    rows = _stops(height, tile, step)
    cols = _stops(width, tile, step)
    for top in rows:
        for left in cols:
            yield top, left, min(top + tile, height), min(left + tile, width)


def _stops(length: int, tile: int, step: int) -> list[int]:
    if length <= tile:
        return [0]
    stops = list(range(0, length - tile + 1, step))
    last = length - tile
    if stops[-1] != last:
        stops.append(last)
    return stops


def _window(height: int, width: int, overlap: int):
    import torch

    window = torch.ones((1, 1, height, width))
    fade = max(1, min(overlap, height // 2, width // 2))
    ramp = torch.linspace(0, 1, fade)
    window[:, :, :fade, :] *= ramp.view(fade, 1)
    window[:, :, -fade:, :] *= ramp.flip(0).view(fade, 1)
    window[:, :, :, :fade] *= ramp.view(1, fade)
    window[:, :, :, -fade:] *= ramp.flip(0).view(1, fade)
    return window


def _tensor(image: Image.Image):
    import numpy as np
    import torch

    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _image(tensor) -> Image.Image:
    array = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).byte().numpy()
    return Image.fromarray(array, mode="RGB")


def _even(value: int) -> int:
    number = max(2, int(value))
    return number - (number % 2)
