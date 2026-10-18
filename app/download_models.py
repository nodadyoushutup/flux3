"""Download the ComfyUI Flux 3 Klein files when the model volume is empty."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

REPO = "black-forest-labs/flux-3-image-klein-comfy"

# Paths match the official ComfyUI Klein templates.
FILES = (
    "diffusion_models/flux3-image-klein-int8-convrot.safetensors",
    "text_encoders/qwen3vl_4b_int8_convrot.safetensors",
    "text_encoders/qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors",
    "vae/flux2-vae.safetensors",
)


def missing(models_dir: Path) -> list[str]:
    return [name for name in FILES if not (models_dir / name).is_file()]


def ensure_models(models_dir: Path, token: str | None) -> None:
    needed = missing(models_dir)
    if not needed:
        print("models already present", flush=True)
        return
    if not token:
        joined = "\n".join(needed)
        print(
            "HF_TOKEN is unset, so the Klein weights were not downloaded.\n"
            f"Missing:\n{joined}",
            flush=True,
        )
        return
    from huggingface_hub import hf_hub_download

    for name in needed:
        print(f"downloading {name}", flush=True)
        try:
            hf_hub_download(
                repo_id=REPO,
                filename=name,
                local_dir=str(models_dir),
                token=token,
            )
        except Exception as exc:
            print(f"download failed for {name}: {exc.__class__.__name__}", flush=True)
            return
        if not (models_dir / name).is_file():
            print(f"download finished without writing {name}", flush=True)
            return


UPSCALE_FILES = {
    "scunet_color_real_gan.pth": (
        "https://github.com/cszn/KAIR/releases/download/v1.0/"
        "scunet_color_real_gan.pth"
    ),
    "4x-UltraSharp.pth": (
        "https://huggingface.co/Kim2091/UltraSharp/resolve/main/"
        "4x-UltraSharp.pth"
    ),
}


def ensure_upscale_models(models_dir: Path) -> None:
    """Download the restoration weights when the volume does not have them."""
    folder = models_dir / "upscale"
    folder.mkdir(parents=True, exist_ok=True)
    for name, url in UPSCALE_FILES.items():
        path = folder / name
        if path.is_file() and path.stat().st_size > 0:
            continue
        print(f"downloading {name}", flush=True)
        urllib.request.urlretrieve(url, path)


def main() -> None:
    models_dir = Path(os.environ.get("MODELS_DIR", "/models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    ensure_upscale_models(models_dir)
    ensure_models(models_dir, os.environ.get("HF_TOKEN") or None)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(exc, file=sys.stderr)
        raise
