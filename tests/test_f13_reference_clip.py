"""Portable checks of the F13 reference-clip script reconstruction (no Mitsuba, no GPU)."""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import f13_reference as reference  # noqa: E402
import f13_reference_clip as clip  # noqa: E402


def base():
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    return {"frame": 0, "camera": {"position": [0, 2.2, 7], "direction": [0, 0, -1], "up": [0, 1, 0]},
            "instances": [{"slot": s, "world": list(identity)} for s in (0, 64, 66)],
            "materials": [{"emissive": [0, 0, 0]} for _ in range(3)] + [{"emissive": [8, 0.4, 0.2]}],
            "lights": [{"direction": [0, -1, 0], "intensity": 12}], "sky": [0, 0, 0], "sun_angular_radius": 0.00465}


class ScriptTiming(unittest.TestCase):
    def test_frame_zero_is_one_fixed_step(self):
        self.assertAlmostEqual(clip.seconds(0), 1 / 60, places=6)

    def test_float_step_puts_events_on_runner_frames(self):
        # The engine sums float(1/60) in double: frame 119 is just past 2 s, 239 just past 4 s.
        self.assertLess(clip.seconds(118), clip.LIGHT_STEP)
        self.assertGreater(clip.seconds(119), clip.LIGHT_STEP)
        self.assertLess(clip.seconds(238), clip.CAMERA_CUT)
        self.assertGreater(clip.seconds(239), clip.CAMERA_CUT)

    def test_disocclusion_cut_and_occluder(self):
        before, after = clip.state("disocclusion", base(), 238), clip.state("disocclusion", base(), 239)
        self.assertNotEqual(before["camera"]["position"], after["camera"]["position"])
        self.assertEqual(after["camera"]["position"], [-1.8, 2.5, 6.5])
        occluder = {i["slot"]: i for i in after["instances"]}[66]["world"][12:15]
        self.assertEqual(occluder, [1.8, 1.25, 2.0])
        self.assertEqual({i["slot"]: i for i in clip.state("disocclusion", base(), 100)["instances"]}[66]["world"][12], 0.0)

    def test_moving_light_steps(self):
        self.assertEqual(clip.state("moving-light", base(), 118)["materials"][3]["emissive"], [8.0, 0.4, 0.2])
        dark = clip.state("moving-light", base(), 119)
        self.assertEqual(dark["materials"][3]["emissive"], [0.0, 0.0, 0.0])
        self.assertEqual(dark["lights"][0]["intensity"], 6.0)
        self.assertEqual(clip.state("moving-light", base(), 239)["materials"][3]["emissive"], [0.2, 0.5, 8.0])

    def test_static_frames_share_one_state(self):
        keys = {clip.key(clip.state("disocclusion", base(), f)) for f in range(239, 300)}
        self.assertEqual(len(keys), 1)
        self.assertEqual(len({clip.key(clip.state("mirror", base(), f)) for f in range(300)}), 1)
        self.assertEqual(len({clip.key(clip.state("moving-light", base(), f)) for f in range(10)}), 10)

    def test_unknown_scene_refused(self):
        with self.assertRaises(ValueError):
            clip.state("ao-cavity", base(), 0)

    def test_roughness_is_static(self):
        self.assertEqual(len({clip.key(clip.state("roughness", base(), f)) for f in range(300)}), 1)


class Window(unittest.TestCase):
    def test_window_covers_regions_plus_margin(self):
        regions = {"a": [0.33, 0.20, 0.67, 0.73], "b": [0.36, 0.23, 0.64, 0.50]}
        self.assertEqual(clip.window(regions, 640, 360, 2), (209, 70, 431, 265))

    def test_window_clamped_to_image(self):
        self.assertEqual(clip.window({"a": [0, 0, 1, 1]}, 64, 32, 2), (0, 0, 64, 32))


class TextureFree(unittest.TestCase):
    def test_textured_material_refused(self):
        data = {"instances": [{"material": 0}], "lights": [], "sampled_lights": [],
                "materials": [{"textures": [0, reference.INVALID, reference.INVALID, reference.INVALID, reference.INVALID],
                               "alpha_cutoff": 0, "flags": 0}]}
        with self.assertRaises(ValueError):
            reference.require_texture_free(data)


class ResumeIdentity(unittest.TestCase):
    def config(self, **changes):
        import argparse
        args = argparse.Namespace(secondary_specular="regularized", regularization_scale=8.0, spp=4096, spp_per_pass=256,
                                  window_regions=True, window_margin=8)
        for name, value in changes.items():
            setattr(args, name, value)
        return clip.render_config(args, {"floor": [0, 0, 1, 1]}, "abc")

    def test_reuse_needs_the_same_render_configuration(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            report = pathlib.Path(folder) / "reference_report.json"
            report.write_text(json.dumps({"converged": True, "render_config": self.config()}))
            self.assertTrue(clip.reusable(report, self.config(), True))
            self.assertFalse(clip.reusable(report, self.config(), False))
            for change in ({"regularization_scale": 4.0}, {"secondary_specular": "unregularized"},
                           {"window_margin": 16}, {"spp": 8192}):
                self.assertFalse(clip.reusable(report, self.config(**change), True), change)
            report.write_text(json.dumps({"converged": True}))  # older report without provenance
            self.assertFalse(clip.reusable(report, self.config(), True))


class DifferenceEstimator(unittest.TestCase):
    """The correction new = old + (render_new - render_old) with shared seed streams (synthetic renderer, no Mitsuba)."""

    @classmethod
    def setUpClass(cls):
        try:
            import numpy
        except ImportError:
            raise unittest.SkipTest("numpy not installed")
        cls.np = numpy

    def fake(self, effect, noise=0.5):
        np = self.np

        class Context:
            def __init__(inner, shift_effect):
                inner.np, inner.width, inner.height, inner.effect = np, 6, 4, shift_effect

            def render_pass(inner, seed, spp):
                rng = np.random.default_rng(seed)  # the noise is a function of the stream seed only
                return spp * (1.0 + inner.effect + noise * rng.standard_normal((4, 6, 3)) / np.sqrt(spp))
        return Context(0.0), Context(effect)

    def test_exact_difference_with_shared_streams(self):
        np = self.np
        off, on = self.fake(0.25)
        a = clip.accumulate(off, 64, 32)
        b = clip.accumulate(on, 64, 32)
        old = np.full((4, 6, 3), 3.0)
        image, differences = clip.combine(np, old, a, b)
        np.testing.assert_allclose(image, old + 0.25, atol=1e-12)
        for d in differences:
            np.testing.assert_allclose(d, 0.25, atol=1e-12)
        area = (slice(0, 4), slice(0, 6))
        stats = clip.difference_statistics(np, old, differences, area)
        self.assertAlmostEqual(stats["mean_difference"], 0.25, places=12)
        self.assertAlmostEqual(stats["relative_mean_effect"], 0.25 / 3.0, places=12)
        self.assertLess(stats["difference_noise_rms"], 1e-12)

    def test_streams_match_converge_and_shift_decorrelates(self):
        np = self.np
        self.assertEqual(clip.seed_stream(1234, 0), (1234 * 1000003) & 0x7FFFFFFF)
        self.assertEqual(clip.seed_stream(5678, 64), (5678 * 1000003 + 64) & 0x7FFFFFFF)
        self.assertNotEqual(clip.seed_stream(1234, 32, 7919), clip.seed_stream(1234, 32))
        off, on = self.fake(0.25)
        a = clip.accumulate(off, 64, 32)
        correlated = clip.accumulate(on, 64, 32)
        shifted = clip.accumulate(on, 64, 32, shift=7919)  # negative control: uncorrelated streams
        old = np.ones((4, 6, 3))
        area = (slice(0, 4), slice(0, 6))
        good = clip.difference_statistics(np, old, clip.combine(np, old, a, correlated)[1], area)
        bad = clip.difference_statistics(np, old, clip.combine(np, old, a, shifted)[1], area)
        self.assertLess(good["difference_noise_rms"], 1e-12)
        self.assertGreater(bad["difference_noise_rms"], 1e-2)
        self.assertGreater(bad["noise_over_effect"], 10 * max(good["noise_over_effect"], 1e-3))

    def test_accumulator_continues_the_same_streams(self):
        np = self.np
        _, on = self.fake(0.25)
        one = clip.accumulate(on, 96, 32)
        accumulator = clip.Accumulator(on, 32)
        accumulator.extend(32)
        accumulator.extend(96)
        for a, b in zip(one, accumulator.means()):
            np.testing.assert_allclose(a, b, atol=1e-12)

    def test_adaptive_difference_stops_at_the_first_quiet_checkpoint(self):
        import argparse
        np = self.np
        off, on = self.fake(0.25)
        slices = {"frame": (slice(0, 4), slice(0, 6))}
        args = argparse.Namespace(spp=[32, 64, 128], spp_per_pass=32, max_noise_ratio=0.15)
        _, _, spp, noise = clip.adaptive_difference([(False, off), (True, on)], args, np.ones((4, 6, 3)), slices, {"frame": 0.01})
        self.assertEqual(spp, 32)  # shared streams: the difference is exact at once
        self.assertLess(noise["frame"], 1e-12)
        on.effect = 0.0
        # Decorrelate by hand: a context whose noise ignores the stream shift would never quiet down; the loop must run to the last checkpoint.
        numpy = np

        class Loud:
            np, width, height, calls = numpy, 6, 4, 0
            def render_pass(inner, seed, spp):
                inner.calls += 1
                return spp * np.random.default_rng(seed + 7 * inner.calls).standard_normal((4, 6, 3))
        _, _, spp, noise = clip.adaptive_difference([(False, off), (True, Loud())], args, np.ones((4, 6, 3)), slices, {"frame": 0.01})
        self.assertEqual(spp, 128)
        self.assertGreater(noise["frame"], 0.0015)

    def test_pfm_round_trip_keeps_row_order(self):
        import tempfile
        np = self.np
        image = np.arange(4 * 6 * 3, dtype=np.float64).reshape(4, 6, 3)
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "a.pfm"
            clip.write_pfm(path, image)
            np.testing.assert_allclose(clip.read_pfm_array(path), image)


if __name__ == "__main__":
    unittest.main()
