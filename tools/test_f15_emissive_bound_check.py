#!/usr/bin/env python3
"""Tests of tools/f15_emissive_bound_check.py (no GPU).

  uv run --offline --no-project --python 3.13 --with numpy python -I tools/test_f15_emissive_bound_check.py
"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import f15_emissive_bound_check as eb  # noqa: E402


class Judge(unittest.TestCase):
    def test_pass_and_failures(self):
        self.assertEqual(eb.judge({"gpu_ms": {"max": 15.0}, "gpu_allocations": 0}), [])
        self.assertTrue(eb.judge({"gpu_ms": {"max": 250.0}, "gpu_allocations": 0}))   # negative control: a slow frame fails
        self.assertTrue(eb.judge({"gpu_ms": {"max": float("nan")}, "gpu_allocations": 0}))
        self.assertTrue(eb.judge({"gpu_ms": {"max": 1.0}, "gpu_allocations": 3}))

    def test_command_uses_the_emissive_scene(self):
        for rt in (True, False):
            c = eb.command("phosphor", "a.gltf", rt, "1280x720", "r.json")
            self.assertIn(eb.SCENARIO, c)
            self.assertEqual(c[c.index("--rt") + 1], "on" if rt else "off")


if __name__ == "__main__":
    unittest.main()
