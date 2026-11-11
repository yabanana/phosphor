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
            clip.state("roughness", base(), 0)


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


if __name__ == "__main__":
    unittest.main()
