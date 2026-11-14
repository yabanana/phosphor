#!/usr/bin/env python3
"""Unit tests of tools/f15_shadowed_hit_check.py (no GPU): rig framing, region geometry, the generated fixture, the PNG reader."""
import json
import pathlib
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import f15_shadowed_hit_check as check  # noqa: E402


def make_png(width, height, value):
    raw = b"".join(b"\x00" + bytes([value, value, value]) * width for _ in range(height))
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


class ShadowedHitCheck(unittest.TestCase):
    def test_framing_matches_the_declared_rig(self):
        centre, distance = check.frame_rig((-1, -0.6, 0.0), (1, 0.6, 0.30))
        self.assertAlmostEqual(centre[2], 0.15)
        self.assertAlmostEqual(distance, 1.12 * (1.0 / (0.5773502691896257 * 4.0 / 3.0)) + 0.15, places=5)

    def test_regions_sit_inside_the_shadow_and_the_panes(self):
        shadow = check.shadow_rect(check.PANEL)
        for name in ("shadow_glass", "shadow_direct"):
            x0, y0, x1, y1 = check.REGIONS[name]
            self.assertTrue(shadow[0] <= x0 and x1 <= shadow[2] and shadow[1] <= y0 and y1 <= shadow[3], name)
        for name in ("lit_glass", "lit_direct"):  # outside the shadow: the sun reaches them
            x0, y0, x1, y1 = check.REGIONS[name]
            self.assertTrue(x0 > shadow[2] or x1 < shadow[0] or y1 < shadow[1] or y0 > shadow[3], name)
        gx0, gy0, gx1, gy1, _ = check.PANE_SHADOW
        x0, y0, x1, y1 = check.REGIONS["shadow_glass"]
        self.assertTrue(gx0 < x0 and x1 < gx1 and gy0 < y0 and y1 < gy1)
        self.assertFalse(check.PANE_SHADOW[0] < check.REGIONS["shadow_direct"][2] and check.REGIONS["shadow_direct"][0] < check.PANE_SHADOW[2])

    def test_region_pixels_are_inside_the_frame_and_ordered(self):
        centre, distance = check.frame_rig((-1, -0.6, 0.0), (1, 0.6, 0.30))
        for rect in check.REGIONS.values():
            x0, y0, x1, y1 = check.region_pixels(rect, centre, distance)
            self.assertTrue(0 <= x0 < x1 <= check.WIDTH and 0 <= y0 < y1 <= check.HEIGHT)

    def test_scene_is_valid_gltf_with_three_materials(self):
        with tempfile.TemporaryDirectory() as folder:
            path = check.write_scene(pathlib.Path(folder))
            document = json.loads(path.read_text())
            binary = (path.parent / "scene.bin").read_bytes()
            self.assertEqual(len(binary), document["buffers"][0]["byteLength"])
            for view in document["bufferViews"]:
                self.assertEqual(view["byteOffset"] % 4, 0)
                self.assertLessEqual(view["byteOffset"] + view["byteLength"], len(binary))
            glass = [m for m in document["materials"] if "KHR_materials_transmission" in m.get("extensions", {})]
            self.assertEqual(len(glass), 1)
            self.assertEqual(glass[0]["extensions"]["KHR_materials_ior"]["ior"], 1.0)
            self.assertNotIn("KHR_materials_volume", glass[0]["extensions"])  # thin-walled

    def test_png_reader_and_region_mean(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "a.png"
            path.write_bytes(make_png(8, 6, 100))
            image = check.read_png_gray(path)
            self.assertAlmostEqual(check.region_mean(image, (1, 1, 5, 4)), 100.0, places=4)


if __name__ == "__main__":
    unittest.main()
