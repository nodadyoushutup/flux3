"""Use the torch deltanet path inside ComfyUI.

comfy_kitchen treats every sm_80 GPU as kernel-capable. On an A100 the CUDA
launch is rejected, so the Qwen upsampler dies before Klein samples. Forcing
the torch implementation keeps the same math and lets generation finish.
"""


def _force_torch_deltanet() -> None:
    try:
        import comfy_kitchen.gated_delta as gated_delta
    except Exception:
        return
    gated_delta.deferred_is_available = lambda *args, **kwargs: False


_force_torch_deltanet()
