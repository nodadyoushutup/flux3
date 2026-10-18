"""POST /v1/images.

Same body as OpenRouter's Flux 3 image request, without safety settings:

model, prompt, resolution, aspect_ratio, input_references, n, seed, size.
"""

from __future__ import annotations

import base64
import json
import os
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pipeline import ACCEPTED_MODELS, MAX_REFERENCES, MODEL_ID, build_prompt, build_upsampler, output_size

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
TIMEOUT = float(os.environ.get("GENERATION_TIMEOUT", "900"))
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_MAX_JOBS = 8


def _remember(job_id: str, payload: dict) -> None:
    with _JOBS_LOCK:
        _JOBS[job_id] = payload
        while len(_JOBS) > _MAX_JOBS:
            oldest = next(iter(_JOBS))
            if oldest == job_id:
                break
            _JOBS.pop(oldest, None)


def _job(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        payload = _JOBS.get(job_id)
        return None if payload is None else dict(payload)


def _job_id(path: str) -> str | None:
    for prefix in (
        "/v1/images/",
        "/api/v1/images/",
        "/v1/upscale/",
        "/api/v1/upscale/",
    ):
        if not path.startswith(prefix):
            continue
        job_id = path[len(prefix) :]
        if len(job_id) == 32 and all(char in "0123456789abcdef" for char in job_id):
            return job_id
        return ""
    return None


def _start_job(body: dict) -> str:
    if not isinstance(body, dict):
        raise ValueError("Expected a JSON object.")
    job_id = uuid.uuid4().hex
    _remember(job_id, {"id": job_id, "status": "running"})

    def run() -> None:
        try:
            image = generate(body)
        except ValueError as exc:
            _remember(
                job_id,
                {
                    "id": job_id,
                    "status": "failed",
                    "error": str(exc),
                    "http_status": 400,
                },
            )
            return
        except Exception as exc:
            print(f"job {job_id} failed: {exc}", flush=True)
            _remember(
                job_id,
                {
                    "id": job_id,
                    "status": "failed",
                    "error": str(exc),
                    "http_status": 502,
                },
            )
            return
        _remember(
            job_id,
            {
                "id": job_id,
                "status": "succeeded",
                "created": int(time.time()),
                "data": [
                    {
                        "b64_json": base64.b64encode(image).decode(),
                        "media_type": "image/png",
                    }
                ],
            },
        )

    threading.Thread(target=run, name=f"klein-{job_id[:8]}", daemon=True).start()
    return job_id


def _start_upscale(body: dict) -> str:
    if not isinstance(body, dict):
        raise ValueError("Expected a JSON object.")
    blob = _one_image(body.get("image"))
    width = body.get("width")
    height = body.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("width and height must be integers")
    job_id = uuid.uuid4().hex
    _remember(job_id, {"id": job_id, "status": "running"})

    def run() -> None:
        try:
            from upscale import restore

            image = restore(blob, width, height)
        except ValueError as exc:
            _remember(
                job_id,
                {
                    "id": job_id,
                    "status": "failed",
                    "error": str(exc),
                    "http_status": 400,
                },
            )
            return
        except Exception as exc:
            print(f"upscale {job_id} failed: {exc}", flush=True)
            _remember(
                job_id,
                {
                    "id": job_id,
                    "status": "failed",
                    "error": str(exc),
                    "http_status": 502,
                },
            )
            return
        _remember(
            job_id,
            {
                "id": job_id,
                "status": "succeeded",
                "created": int(time.time()),
                "data": [
                    {
                        "b64_json": base64.b64encode(image).decode(),
                        "media_type": "image/png",
                    }
                ],
            },
        )

    threading.Thread(target=run, name=f"upscale-{job_id[:8]}", daemon=True).start()
    return job_id


def _one_image(value) -> bytes:
    if isinstance(value, str):
        return _fetch_reference(value)
    if isinstance(value, dict):
        return _fetch_reference((value.get("image_url") or {}).get("url") or value.get("url") or "")
    raise ValueError("image is required")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/health":
            try:
                with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=2) as response:
                    ok = response.status == 200
            except (urllib.error.URLError, TimeoutError):
                ok = False
            self._json(200 if ok else 503, {"status": "ok" if ok else "starting"})
            return
        job_id = _job_id(path)
        if not job_id:
            self._json(404, {"error": "not found"})
            return
        payload = _job(job_id)
        if payload is None:
            self._json(404, {"error": "not found"})
            return
        if payload.get("status") == "failed":
            self._json(
                int(payload.get("http_status") or 502),
                {"error": str(payload.get("error") or "generation failed")},
            )
            return
        self._json(200, payload)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/v1/upscale", "/api/v1/upscale"):
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                job_id = _start_upscale(body)
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(202, {"id": job_id, "status": "running"})
            return
        if path not in ("/v1/images", "/api/v1/images"):
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            job_id = _start_job(body)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
            return
        self._json(202, {"id": job_id, "status": "running"})

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}", flush=True)

    def _json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def generate(body: dict) -> bytes:
    model = body.get("model")
    if model is not None and model not in ACCEPTED_MODELS:
        raise ValueError(f"this server runs {MODEL_ID}")
    if body.get("n", 1) != 1:
        raise ValueError("n must be 1")
    user_prompt = body.get("prompt")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("prompt is required")
    references = _references(body.get("input_references") or [])
    reference_size = _image_size(references[0]) if references else None
    width, height = output_size(body, reference_size)
    seed = body.get("seed")
    if seed is None:
        seed = random.randrange(2**31)
    if not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    names = [_upload(index, blob) for index, blob in enumerate(references)]
    caption = _upsample(user_prompt, names)
    graph = build_prompt(
        caption=caption,
        image_names=names,
        width=width,
        height=height,
        seed=seed,
    )
    prompt_id = _submit(graph)
    return _wait(prompt_id)


def _references(value) -> list[bytes]:
    if not isinstance(value, list):
        raise ValueError("input_references must be a list")
    if len(value) > MAX_REFERENCES:
        raise ValueError(f"at most {MAX_REFERENCES} reference images")
    blobs = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each input reference needs image_url.url")
        kind = item.get("type")
        if kind is not None and kind != "image_url":
            raise ValueError("each input reference type must be image_url")
        url = (item.get("image_url") or {}).get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("each input reference needs image_url.url")
        blobs.append(_fetch_reference(url))
    return blobs


def _fetch_reference(url: str) -> bytes:
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        if ";base64" not in header:
            raise ValueError("data URLs must be base64")
        return base64.b64decode(data)
    if not url.startswith(("http://", "https://")):
        raise ValueError("reference URLs must be http(s) or base64 data URLs")
    with urllib.request.urlopen(url, timeout=60) as response:
        blob = response.read(26 * 1024 * 1024)
        if len(blob) >= 26 * 1024 * 1024:
            raise ValueError("a reference image is larger than 25 MB")
        return blob


def _image_size(blob: bytes) -> tuple[int, int] | None:
    if blob.startswith(b"\x89PNG\r\n\x1a\n") and len(blob) >= 24:
        width = int.from_bytes(blob[16:20], "big")
        height = int.from_bytes(blob[20:24], "big")
        return width, height
    if blob.startswith(b"\xff\xd8"):
        return _jpeg_size(blob)
    return None


def _jpeg_size(blob: bytes) -> tuple[int, int] | None:
    index = 2
    while index + 8 < len(blob):
        if blob[index] != 0xFF:
            index += 1
            continue
        marker = blob[index + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            height = int.from_bytes(blob[index + 5 : index + 7], "big")
            width = int.from_bytes(blob[index + 7 : index + 9], "big")
            return width, height
        length = int.from_bytes(blob[index + 2 : index + 4], "big")
        index += 2 + length
    return None


def _upload(index: int, blob: bytes) -> str:
    name = f"ref-{index}.png" if blob.startswith(b"\x89PNG") else f"ref-{index}.bin"
    boundary = "flux3boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + blob + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{COMFY_URL}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read())
    filename = payload.get("name")
    if not filename:
        raise RuntimeError("ComfyUI did not accept a reference image")
    return filename


def _upsample(user_prompt: str, names: list[str]) -> str:
    prompt_id = _submit(build_upsampler(user_prompt=user_prompt, image_names=names))
    return _wait_text(prompt_id)


def _submit(graph: dict) -> str:
    raw = json.dumps({"prompt": graph}).encode()
    request = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"ComfyUI rejected the prompt: {detail}") from exc
    if payload.get("node_errors"):
        raise RuntimeError(f"ComfyUI rejected the prompt: {payload['node_errors']}")
    prompt_id = payload.get("prompt_id")
    if not prompt_id:
        raise RuntimeError("ComfyUI did not return a prompt id")
    return prompt_id


def _wait_text(prompt_id: str) -> str:
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}", timeout=30) as response:
            history = json.loads(response.read())
        item = history.get(prompt_id)
        if item:
            status = (item.get("status") or {}).get("status_str")
            if status == "error":
                raise RuntimeError("ComfyUI prompt upsampling failed")
            for node in (item.get("outputs") or {}).values():
                text = node.get("text")
                if text:
                    return text[0] if isinstance(text, list) else text
        time.sleep(1)
    raise RuntimeError("prompt upsampling timed out")


def _wait(prompt_id: str) -> bytes:
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}", timeout=30) as response:
            history = json.loads(response.read())
        item = history.get(prompt_id)
        if item:
            status = (item.get("status") or {}).get("status_str")
            if status == "error":
                raise RuntimeError("ComfyUI generation failed")
            for node in (item.get("outputs") or {}).values():
                images = node.get("images") or []
                if images:
                    return _view(images[0])
        time.sleep(1)
    raise RuntimeError("generation timed out")


def _view(image: dict) -> bytes:
    query = urllib.parse.urlencode(
        {
            "filename": image["filename"],
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        }
    )
    with urllib.request.urlopen(f"{COMFY_URL}/view?{query}", timeout=60) as response:
        return response.read()


def main() -> None:
    port = int(os.environ.get("API_PORT", "8099"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"listening on {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
