#!/usr/bin/env python3
"""Unit tests of tools/f15_encoding_check.py (the glTF encodings of the thin subsurface lobe against the description file)."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import f15_encoding_check as enc  # noqa: E402


class EncodingCheck(unittest.TestCase):
    def test_encoded_scene_carries_the_extensions_and_the_plain_scene_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = enc.build_scene(pathlib.Path(tmp) / "a", True)[1]
            document = json.loads(path.read_text())
            plain = json.loads(enc.build_scene(pathlib.Path(tmp) / "b", False)[1].read_text())
        self.assertEqual(document["extensionsUsed"], ["KHR_materials_diffuse_transmission", "KHR_materials_scatter", "KHR_materials_transmission"])
        self.assertNotIn("extensionsUsed", plain)
        names = [m["name"] for m in document["materials"]]
        self.assertEqual(names, [m["name"] for m in plain["materials"]])
        self.assertEqual(names[:3], ["dt-orange", "dt-white", "scatter-teal"])
        self.assertTrue(all("extensions" not in m for m in plain["materials"]))

    def test_overrides_map_the_encodings_like_the_loader(self):
        doc = enc.overrides()
        self.assertEqual(doc["schema"], "phosphor-material-overrides")
        by_name = {o["name"]: o for o in doc["overrides"]}
        self.assertEqual(by_name["dt-orange"]["subsurface"], {"mode": "thin", "weight": 0.6, "color": [1.0, 0.5, 0.25]})
        scatter = by_name["scatter-teal"]
        self.assertAlmostEqual(scatter["subsurface"]["weight"], 0.8 * 0.5)  # transmission * scatterStrength
        self.assertEqual(scatter["subsurface"]["color"], [0.9, 0.7, 0.4])
        self.assertEqual(scatter["transmission"], {"factor": 0.8})
        # the controls perturb the first diffuse sphere only
        moved = {o["name"]: o for o in enc.overrides(delta_weight=0.02)["overrides"]}
        self.assertAlmostEqual(moved["dt-orange"]["subsurface"]["weight"], 0.62)
        self.assertEqual(moved["dt-white"], by_name["dt-white"])
        self.assertAlmostEqual(enc.overrides(delta_colour=0.02)["overrides"][0]["subsurface"]["color"][2], 0.27)

    def test_differing_pixels_counts_pixels_not_channels(self):
        a = (2, 1, 3, [bytes([1, 2, 3, 4, 5, 6])])
        self.assertEqual(enc.differing_pixels(a, a), (0, 0))
        b = (2, 1, 3, [bytes([1, 2, 9, 4, 5, 6])])
        self.assertEqual(enc.differing_pixels(a, b), (1, 6))
        c = (2, 1, 3, [bytes([2, 3, 3, 4, 5, 7])])
        self.assertEqual(enc.differing_pixels(a, c), (2, 1))
        with self.assertRaises(ValueError):
            enc.differing_pixels(a, (3, 1, 3, [bytes(9)]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
