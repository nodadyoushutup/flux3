import tempfile
import unittest
from pathlib import Path

from download_models import ensure_models, missing
from pipeline import build_prompt, build_upsampler, canvas_size, megapixels, normalize_resolution, output_size
from upscale import detail_radius
from upscale import target_size
from upscale import _mix_low
from upscale import _transfer_detail


class CanvasTest(unittest.TestCase):
    def test_square_1k(self):
        self.assertEqual(canvas_size("1K", "1:1"), (1024, 1024))

    def test_wide_2k_keeps_area(self):
        width, height = canvas_size("2K", "16:9")
        self.assertGreater(width, height)
        self.assertEqual(width % 16, 0)
        self.assertEqual(height % 16, 0)

    def test_megapixels(self):
        self.assertEqual(megapixels("1K"), 1)
        self.assertEqual(megapixels("4K"), 16)

    def test_square_2k(self):
        self.assertEqual(canvas_size("2K", "1:1"), (2048, 2048))

    def test_resolution_tiers(self):
        self.assertEqual(normalize_resolution("2k"), "2K")
        self.assertEqual(normalize_resolution("768"), "768")
        with self.assertRaises(ValueError):
            normalize_resolution("8K")

    def test_openrouter_body_defaults_to_1k_square(self):
        self.assertEqual(output_size({}), (1024, 1024))
        self.assertEqual(output_size({"resolution": "2K", "aspect_ratio": "1:1"}), (2048, 2048))
        self.assertEqual(output_size({"size": "2K", "aspect_ratio": "16:9"}), canvas_size("2K", "16:9"))
        self.assertEqual(output_size({"aspect_ratio": "auto"}, (1920, 1080)), canvas_size("1K", "16:9"))
        with self.assertRaises(ValueError):
            output_size({"size": "2048x2048", "resolution": "2K"})


class GraphTest(unittest.TestCase):
    def test_text_runs_upsampler_before_klein(self):
        upsampler = build_upsampler(user_prompt="a red fox", image_names=[])
        self.assertEqual(upsampler["11"]["inputs"]["sampling_mode.presence_penalty"], 1.5)
        self.assertNotIn("image", upsampler["11"]["inputs"])
        self.assertNotIn("1", upsampler)
        graph = build_prompt(
            caption="a red fox in snow",
            image_names=[],
            width=1024,
            height=1024,
            seed=7,
        )
        self.assertEqual(graph["4"]["inputs"]["text"], "a red fox in snow")
        self.assertNotIn("11", graph)
        self.assertEqual(graph["7"]["inputs"]["sampler_name"], "er_sde")

    def test_references_go_to_upsampler_and_klein(self):
        upsampler = build_upsampler(user_prompt="use both", image_names=["a.png", "b.png"])
        self.assertEqual(upsampler["11"]["inputs"]["image"], ["13", 0])
        self.assertEqual(upsampler["11"]["inputs"]["sampling_mode.presence_penalty"], 0)
        self.assertEqual(upsampler["13"]["inputs"]["images.image1"], ["101", 0])
        graph = build_prompt(
            caption="a composed scene",
            image_names=["a.png", "b.png"],
            width=2048,
            height=2048,
            seed=7,
        )
        self.assertEqual(graph["14"]["inputs"]["images.image0"], ["100", 0])
        self.assertEqual(graph["14"]["inputs"]["resolution.megapixels"], 1)
        self.assertEqual(graph["6"]["inputs"], {"width": 2048, "height": 2048, "batch_size": 1})
        self.assertEqual(graph["7"]["inputs"]["latent_image"], ["6", 0])
        self.assertEqual(graph["7"]["inputs"]["sampler_name"], "euler")
        self.assertNotIn("11", graph)


class DownloadTest(unittest.TestCase):
    def test_skips_when_files_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in missing(root):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            ensure_models(root, None)

    def test_missing_token_starts_anyway(self):
        with tempfile.TemporaryDirectory() as directory:
            ensure_models(Path(directory), None)
            self.assertTrue(missing(Path(directory)))


class DetailMixTest(unittest.TestCase):
    def test_radius_covers_edges_on_a_large_frame(self):
        self.assertGreater(detail_radius(2048, 2048), 8)
        self.assertGreater(detail_radius(4096, 2160), 10)

    def test_detail_keeps_photograph_color(self):
        from PIL import Image

        base = Image.new("RGB", (32, 32), (200, 30, 30))
        detailed = Image.new("RGB", (32, 32), (30, 30, 200))
        mixed = _transfer_detail(base, detailed, radius=8, amount=1)
        self.assertEqual(mixed.getpixel((16, 16)), (200, 30, 30))

    def test_detail_adds_lightness_from_the_upscaler(self):
        from PIL import Image

        base = Image.new("RGB", (64, 64), (80, 70, 60))
        detailed = Image.new("RGB", (64, 64), (80, 70, 60))
        for y in range(64):
            for x in range(30, 34):
                detailed.putpixel((x, y), (250, 250, 250))
        mixed = _transfer_detail(base, detailed, radius=3, amount=1)
        edge = mixed.getpixel((32, 32))
        corner = mixed.getpixel((4, 4))
        self.assertGreater(sum(edge), sum(corner) + 40)
        self.assertLess(abs(edge[0] - edge[1]), 30)

    def test_deblock_keeps_fine_texture(self):
        from PIL import Image

        base = Image.new("RGB", (32, 32), (40, 40, 40))
        for y in range(32):
            for x in range(0, 32, 2):
                base.putpixel((x, y), (140, 140, 140))
        donor = Image.new("RGB", (32, 32), (200, 180, 160))
        mixed = _mix_low(base, donor, radius=4, amount=1)
        bright = mixed.getpixel((0, 16))
        dark = mixed.getpixel((1, 16))
        self.assertGreater(bright[0] - dark[0], 60)
        self.assertGreater(dark[0], 80)


class TargetSizeTest(unittest.TestCase):
    def test_source_stays_even(self):
        self.assertEqual(target_size(1001, 501, "source"), (1000, 500))

    def test_2k_sets_the_long_edge(self):
        self.assertEqual(target_size(1000, 500, "2K"), (2048, 1024))
        self.assertEqual(target_size(500, 1000, "2K"), (1024, 2048))


if __name__ == "__main__":
    unittest.main()
