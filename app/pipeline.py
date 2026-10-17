"""One OpenRouter image request becomes one Klein graph.

The body matches ``POST /api/v1/images`` for Flux 3: ``prompt``, ``resolution``,
``aspect_ratio``, and up to ten ``input_references``. ``size`` is the same
shorthand OpenRouter accepts. Safety settings are not part of this request.
The 9B prompt upsampler runs first. Klein then samples that caption onto the
requested canvas. Reference images, when present, condition the same sample.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

PROMPTS = json.loads((Path(__file__).with_name("prompts.json")).read_text())

MODEL_ID = "black-forest-labs/flux-3-image-klein"
ACCEPTED_MODELS = {
    MODEL_ID,
    "black-forest-labs/flux-3-image",
    "flux-3-image-klein",
    "flux-3-image",
}
UNET = "flux3-image-klein-int8-convrot.safetensors"
TEXT_ENCODER = "qwen3vl_4b_int8_convrot.safetensors"
UPSAMPLER = "qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors"
VAE = "flux2-vae.safetensors"
MAX_REFERENCES = 10

# Pixel area of each OpenRouter resolution tier.
AREAS = {
    "768": 768 * 768,
    "768sq": 768 * 768,
    "1K": 1024 * 1024,
    "1.5K": 2 * 1024 * 1024,
    "2K": 4 * 1024 * 1024,
    "4K": 16 * 1024 * 1024,
}


def canvas_size(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    if resolution not in AREAS:
        raise ValueError(f"unsupported resolution {resolution}")
    width_ratio, height_ratio = _aspect(aspect_ratio)
    area = AREAS[resolution]
    height = math.sqrt(area * height_ratio / width_ratio)
    width = height * width_ratio / height_ratio
    return _snap16(width), _snap16(height)


def megapixels(resolution: str) -> float:
    if resolution not in AREAS:
        raise ValueError(f"unsupported resolution {resolution}")
    return AREAS[resolution] / (1024 * 1024)


RESOLUTIONS = ("768", "1K", "1.5K", "2K", "4K")
ASPECTS = (
    "21:9",
    "2:1",
    "16:9",
    "3:2",
    "7:5",
    "4:3",
    "5:4",
    "1:1",
    "4:5",
    "3:4",
    "5:7",
    "2:3",
    "9:16",
    "1:2",
    "9:21",
)


def normalize_resolution(value: str) -> str:
    """Map an OpenRouter resolution tier. ``1K`` is about 1 megapixel and ``2K`` is about 4."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("resolution must be 768, 1K, 1.5K, 2K, or 4K")
    key = value.strip().lower().replace(" ", "")
    aliases = {
        "768": "768",
        "768sq": "768",
        "1k": "1K",
        "1.5k": "1.5K",
        "2k": "2K",
        "4k": "4K",
    }
    resolved = aliases.get(key)
    if resolved is None:
        raise ValueError("resolution must be 768, 1K, 1.5K, 2K, or 4K")
    return resolved


def output_size(body: dict, reference_size: tuple[int, int] | None = None) -> tuple[int, int]:
    """Canvas for an OpenRouter image body. Hosted default is 1K, aspect ``auto``."""
    resolution, pixels = _dimensions(body)
    if pixels is not None:
        return _snap16(pixels[0]), _snap16(pixels[1])
    aspect = body.get("aspect_ratio") or "auto"
    if not isinstance(aspect, str):
        raise ValueError("aspect_ratio must be a string")
    if aspect == "auto":
        aspect = nearest_aspect(*reference_size) if reference_size else "1:1"
    elif aspect not in ASPECTS:
        raise ValueError(f"aspect_ratio must be one of {', '.join(ASPECTS)}, or auto")
    return canvas_size(resolution, aspect)


def nearest_aspect(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "1:1"
    target = math.log(width / height)
    return min(ASPECTS, key=lambda name: abs(math.log(_aspect(name)[0] / _aspect(name)[1]) - target))


def _dimensions(body: dict) -> tuple[str, tuple[int, int] | None]:
    raw_resolution = body.get("resolution")
    raw_size = body.get("size")
    resolution = None if raw_resolution is None else normalize_resolution(raw_resolution)
    if raw_size is None:
        return resolution or "1K", None
    if not isinstance(raw_size, str) or not raw_size.strip():
        raise ValueError("size must be a resolution tier or WIDTHxHEIGHT")
    text = raw_size.strip().lower().replace(" ", "")
    if "x" in text:
        if raw_resolution is not None or "aspect_ratio" in body:
            raise ValueError("pixel size cannot be combined with resolution or aspect_ratio")
        width_text, _, height_text = text.partition("x")
        if not width_text.isdigit() or not height_text.isdigit():
            raise ValueError("size must be a resolution tier or WIDTHxHEIGHT")
        return resolution or "1K", (int(width_text), int(height_text))
    tier = normalize_resolution(raw_size)
    if resolution is not None and resolution != tier:
        raise ValueError("size and resolution disagree")
    return tier, None


def build_upsampler(
    *,
    user_prompt: str,
    image_names: list[str],
) -> dict:
    """Run only the 9B upsampler. The diffusion weights stay unloaded on a 6 GB GPU."""
    if len(image_names) > MAX_REFERENCES:
        raise ValueError(f"at most {MAX_REFERENCES} reference images")
    editing = bool(image_names)
    prompt = {
        "10": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": UPSAMPLER, "type": "flux3", "device": "default"},
        },
        "11": {
            "class_type": "TextGenerate",
            "inputs": {
                "clip": ["10", 0],
                "prompt": user_prompt,
                "max_length": 1024,
                "sampling_mode": "on",
                "sampling_mode.temperature": 1,
                "sampling_mode.top_k": 1,
                "sampling_mode.top_p": 0.95,
                "sampling_mode.min_p": 0,
                "sampling_mode.repetition_penalty": 1.05,
                "sampling_mode.seed": 0,
                "sampling_mode.presence_penalty": 0.0 if editing else 1.5,
                "thinking": False,
                "use_default_template": True,
                "mtp": "off",
                "system_prompt": PROMPTS["i2i" if editing else "t2i"],
            },
        },
        "12": {
            "class_type": "PreviewAny",
            "inputs": {"source": ["11", 0]},
        },
    }
    if editing:
        for index, name in enumerate(image_names):
            prompt[str(100 + index)] = {
                "class_type": "LoadImage",
                "inputs": {"image": name},
            }
        prompt["13"] = {
            "class_type": "BatchImagesNode",
            "inputs": {
                f"images.image{index}": [str(100 + index), 0]
                for index in range(len(image_names))
            },
        }
        prompt["11"]["inputs"]["image"] = ["13", 0]
    return prompt


def build_prompt(
    *,
    caption: str,
    image_names: list[str],
    width: int,
    height: int,
    seed: int,
) -> dict:
    if len(image_names) > MAX_REFERENCES:
        raise ValueError(f"at most {MAX_REFERENCES} reference images")
    editing = bool(image_names)
    steps = 16 if editing else 15
    sampler_name = "euler" if editing else "er_sde"

    prompt = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": UNET, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "flux3", "device": "default"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": caption},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "flux3"},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
        },
    }

    if editing:
        for index, name in enumerate(image_names):
            prompt[str(100 + index)] = {
                "class_type": "LoadImage",
                "inputs": {"image": name},
            }
        prompt["5"] = {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["4", 0]},
        }
        reference_inputs = {
            "positive": ["4", 0],
            "negative": ["5", 0],
            "vae": ["3", 0],
            "resolution": "resize",
            "resolution.megapixels": 1,
        }
        reference_inputs.update(
            {
                f"images.image{index}": [str(100 + index), 0]
                for index in range(len(image_names))
            }
        )
        prompt["14"] = {
            "class_type": "ReferenceMultiLatent",
            "inputs": reference_inputs,
        }
        prompt["6"] = {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        }
    else:
        prompt["5"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": ""},
        }
        prompt["6"] = {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        }
    prompt["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "seed": seed,
            "steps": steps,
            "cfg": 1,
            "sampler_name": sampler_name,
            "scheduler": "simple",
            "positive": ["14", 0] if editing else ["4", 0],
            "negative": ["14", 1] if editing else ["5", 0],
            "latent_image": ["6", 0],
            "denoise": 1,
        },
    }
    return prompt


def _aspect(aspect_ratio: str) -> tuple[float, float]:
    parts = aspect_ratio.split(":")
    if len(parts) != 2:
        raise ValueError(f"unsupported aspect_ratio {aspect_ratio}")
    width_ratio, height_ratio = float(parts[0]), float(parts[1])
    if width_ratio <= 0 or height_ratio <= 0:
        raise ValueError(f"unsupported aspect_ratio {aspect_ratio}")
    return width_ratio, height_ratio


def _snap16(value: float) -> int:
    return max(16, int(round(value / 16.0)) * 16)
