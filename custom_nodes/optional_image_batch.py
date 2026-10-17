"""Batch only the reference images that are actually connected."""

from __future__ import annotations


class OptionalImageBatch:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"image{index}": ("IMAGE",) for index in range(1, 11)}
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "batch"
    CATEGORY = "image/batch"

    def batch(self, **images):
        frames = []
        for index in range(1, 11):
            frame = images.get(f"image{index}")
            if frame is not None:
                frames.append(frame)
        if not frames:
            return (None,)
        from comfy_extras.nodes_post_processing import batch_images

        return (batch_images(frames),)


NODE_CLASS_MAPPINGS = {"OptionalImageBatch": OptionalImageBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"OptionalImageBatch": "Optional Image Batch"}
