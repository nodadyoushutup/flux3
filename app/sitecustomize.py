"""Use torch for ComfyUI's Qwen attention kernels.

comfy_kitchen's CUDA kernels report this A100 as supported and then fail at
launch. The torch implementations compute the same attention, so generation
can finish. The patch is applied when ComfyUI imports each module, after
Comfy has chosen its CUDA allocator.
"""

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder

_MODULES = {
    "comfy_kitchen.gated_delta",
    "comfy_kitchen.flash_attention",
}


def _force_torch(module) -> None:
    for name in (
        "deferred_is_available",
        "is_available",
        "flash_attention_decode_gqa_is_available",
    ):
        if hasattr(module, name):
            setattr(module, name, lambda *args, **kwargs: False)


class _KitchenFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _MODULES:
            return None
        sys.meta_path.remove(self)
        _MODULES.discard(fullname)
        if _MODULES:
            sys.meta_path.insert(0, _KitchenFinder())
        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        real_exec = spec.loader.exec_module

        def exec_module(module):
            real_exec(module)
            _force_torch(module)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _KitchenFinder())
