"""Download the ComfyUI Flux 3 Klein files when the model volume is empty."""

from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
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


# ``expected_size`` is the published byte count for the safetensors file.
UPSCALE_FILES = {
    "4xNomos8kSCHAT-L.safetensors": (
        "https://huggingface.co/Phips/4xNomos8kSCHAT-L/resolve/main/"
        "4xNomos8kSCHAT-L.safetensors",
        165_299_372,
    ),
}


def _weights_ok(path: Path, expected_size: int | None = None) -> bool:
    """Return whether ``path`` is a complete upscale checkpoint.

    Safetensors and legacy pickles are checked by byte count. Zip
    checkpoints are checked by archive integrity.
    """
    if not path.is_file():
        return False
    size = path.stat().st_size
    if expected_size is not None:
        if size != expected_size:
            return False
    elif size < 60_000_000:
        return False
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    with path.open("rb") as handle:
        header = handle.read(9)
    if len(header) == 9 and header[8:9] == b"{":
        length = int.from_bytes(header[:8], "little")
        return 0 < length < 100_000_000
    return header[:1] == b"\x80"


def ensure_upscale_models(models_dir: Path) -> None:
    """Download the restoration weights when the volume does not have them."""
    folder = models_dir / "upscale"
    folder.mkdir(parents=True, exist_ok=True)
    for name, (url, expected_size) in UPSCALE_FILES.items():
        path = folder / name
        if _weights_ok(path, expected_size):
            continue
        print(f"downloading {name}", flush=True)
        partial = path.with_suffix(path.suffix + ".partial")
        urllib.request.urlretrieve(url, partial)
        if not _weights_ok(partial, expected_size):
            partial.unlink(missing_ok=True)
            raise SystemExit(f"{name} download was incomplete")
        partial.replace(path)


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
