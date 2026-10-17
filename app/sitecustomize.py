"""Use the torch deltanet path inside ComfyUI.

comfy_kitchen treats every sm_80 GPU as kernel-capable. On an A100 the CUDA
launch is rejected, so the Qwen upsampler dies before Klein samples. Patch the
module when ComfyUI imports it, after Comfy has chosen its CUDA allocator.
"""

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder


class _DeltanetFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "comfy_kitchen.gated_delta":
            return None
        sys.meta_path.remove(self)
        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        real_exec = spec.loader.exec_module

        def exec_module(module):
            real_exec(module)
            module.deferred_is_available = lambda *args, **kwargs: False

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _DeltanetFinder())
