#!/usr/bin/env python3
"""Tests of tools/f15_chess_check.py on synthetic images (no GPU, no corpus).

  uv run --offline --no-project --python 3.13 --with numpy python -I tools/test_f15_chess_check.py
"""
import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import f15_chess_check as cc  # noqa: E402

H, W = 90, 160


def solid(rgb):
    return np.tile(np.asarray(rgb, dtype=np.float32), (H, W, 1))


def with_region(base, region, rgb):
    image = base.copy()
    image[region] = np.asarray(rgb, dtype=np.float32)
    return image


class CommandTests(unittest.TestCase):
    def test_capture_names_and_scenarios_are_valid_bench_names(self):
        self.assertEqual(sum(1 for s in cc.CAPTURES.values() if s["group"] == "backlight"), 8)
        self.assertEqual(sum(1 for s in cc.CAPTURES.values() if s["group"] == "sidelight"), 9)
        self.assertEqual(sum(1 for s in cc.CAPTURES.values() if s["group"] == "shadow"), 9)
        for spec in cc.CAPTURES.values():
            self.assertTrue(spec["scenario"].startswith("chess-"))

    def test_capture_command_carries_the_declared_flags(self):
        binary, gltf, overrides = pathlib.Path("/b/phosphor"), pathlib.Path("/c/ABeautifulGame.gltf"), pathlib.Path("/o.json")
        command = cc.capture_command(binary, gltf, overrides, "bl-t050", pathlib.Path("/out/bl-t050"))
        text = " ".join(command)
        for needle in ("--scene /c/ABeautifulGame.gltf", "--material-overrides /o.json", "--bench 2", "--material-scenario chess-backlight/transmission=0.5",
                       "--rt on", "--reflections rt", "--lighting-denoise off", "--pipeline-sync", "--capture-linear-accumulate 112",
                       "--resolution 1920x1080", "--capture-linear-signal hdr", "--shadows off"):
            self.assertIn(needle, text)
        csm = " ".join(cc.capture_command(binary, gltf, overrides, "sh-csm-hidden", pathlib.Path("/out/x")))
        self.assertIn("--shadows csm", csm)
        self.assertIn("--shadow-map-size 8192", csm)
        self.assertNotIn("--shadow-map-size", " ".join(cc.capture_command(binary, gltf, overrides, "sh-rt-on", pathlib.Path("/out/x"))))
        self.assertIn("--rt off", csm)
        rt = " ".join(cc.capture_command(binary, gltf, overrides, "sh-rt-on", pathlib.Path("/out/x")))
        self.assertIn("--shadows rt", rt)
        self.assertIn("--rt on", rt)
        mask = " ".join(cc.capture_command(binary, gltf, overrides, "bl-mask", pathlib.Path("/out/x")))
        self.assertNotIn("--capture-linear-accumulate", mask)
        o7 = " ".join(cc.o7_command(binary, gltf, overrides, "o7-sidelight", pathlib.Path("/out/y")))
        self.assertIn("--warmup 64", o7)
        self.assertNotIn("--capture-linear", o7)


class GeometryTests(unittest.TestCase):
    def test_target_projects_to_the_centre_and_x_follows_the_camera_handedness(self):
        position, target = (0.0, 0.1, 0.5), (0.0, 0.1, 0.0)
        column, row, depth = cc.project(position, target, target, 1920, 1080)
        self.assertAlmostEqual(column, 960.0, places=3)
        self.assertAlmostEqual(row, 540.0, places=3)
        self.assertAlmostEqual(depth, 0.5)
        # looking along -Z the image right is +X; looking along +Z it is -X
        self.assertGreater(cc.project(position, target, (0.1, 0.1, 0.0), 1920, 1080)[0], 960.0)
        self.assertLess(cc.project((0, 0.1, -0.5), (0, 0.1, 0), (0.1, 0.1, 0.0), 1920, 1080)[0], 960.0)
        # a point above the target is above the centre row; the vertical field of view is 60 degrees
        _, row_up, _ = cc.project(position, target, (0.0, 0.1 + 0.5 * math.tan(math.radians(30)), 0.0), 1920, 1080)
        self.assertAlmostEqual(row_up, 0.0, places=3)

    def test_backlight_mask_geometry_check(self):
        mask = np.zeros((cc.HEIGHT, cc.WIDTH), dtype=bool)
        for x in cc.TOP_CENTRES_X:
            column, row, _ = cc.project(cc.CAMERA_BACKLIGHT[0], cc.CAMERA_BACKLIGHT[1], (x, cc.TOP_Y, cc.TOP_Z_ADVANCED.get(x, cc.TOP_Z)))
            mask[int(row) - 15:int(row) + 15, int(column) - 15:int(column) + 15] = True
        result = cc.check_backlight_mask(np, mask)
        self.assertTrue(result["pass"], result)
        shifted = np.roll(mask, 40, axis=0)
        self.assertFalse(cc.check_backlight_mask(np, shifted)["pass"])
        self.assertFalse(cc.check_backlight_mask(np, np.zeros_like(mask))["pass"])

    def test_erode_and_mask(self):
        image = np.zeros((10, 10, 3), dtype=np.float32)
        image[2:8, 2:8] = 1.0
        mask = cc.mask_from_capture(np, image, erode_pixels=1)
        self.assertEqual(int(mask.sum()), 16)
        self.assertEqual(int(cc.mask_from_capture(np, image, erode_pixels=0).sum()), 36)


def backlight_images(tint=(0.2995, 0.5, 0.45), panel=3.0, opaque=0.05, with_back_panel=True):
    mask = np.zeros((H, W), dtype=bool)
    mask[30:60, 40:120] = True
    background = solid((0.02, 0.02, 0.02))
    images = {}
    tint = np.asarray(tint, dtype=np.float32)
    for key, t in (("bl-t000", 0.0), ("bl-t025", 0.25), ("bl-t050", 0.5), ("bl-t075", 0.75), ("bl-t100", 1.0)):
        rgb = (1 - t) * opaque * tint + t * (panel * tint if with_back_panel else 0.2 * tint)
        images[key] = with_region(background, mask, rgb)
    images["bl-white"] = with_region(background, mask, (panel, panel, panel))
    images["bl-nopanel"] = with_region(background, mask, 0.2 * tint)
    return images, mask


class BacklightTests(unittest.TestCase):
    def test_a_transmitting_tinted_backdrop_passes_and_every_control_fails(self):
        images, mask = backlight_images()
        result = cc.evaluate_backlight(np, images, mask)
        self.assertTrue(result["pass"], result)
        controls = cc.backlight_controls(np, images, mask)
        self.assertEqual(set(controls), {"transmission-ignored", "order-reversed", "backdrop-removed", "tint-ignored"})
        for name, control in controls.items():
            self.assertFalse(control["pass"], name)

    def test_each_gate_discriminates_individually(self):
        images, mask = backlight_images()
        controls = cc.backlight_controls(np, images, mask)
        self.assertFalse(controls["transmission-ignored"]["gates"]["monotone"]["pass"])
        self.assertFalse(controls["order-reversed"]["gates"]["monotone"]["pass"])
        self.assertFalse(controls["backdrop-removed"]["gates"]["visible"]["pass"])
        self.assertFalse(controls["tint-ignored"]["gates"]["tint"]["pass"])
        self.assertTrue(controls["tint-ignored"]["gates"]["monotone"]["pass"])  # the others stay green: one gate per fault

    def test_a_wrong_tint_fails_the_chroma_gate(self):
        images, mask = backlight_images(tint=(0.5, 0.5, 0.5))
        images["bl-white"] = with_region(solid((0.02, 0.02, 0.02)), mask, (3.0, 3.0, 3.0))
        result = cc.evaluate_backlight(np, images, mask)
        self.assertFalse(result["gates"]["tint"]["pass"])
        self.assertAlmostEqual(result["gates"]["tint"]["chroma_rg_gg_bg"][0], 1.0, places=3)

    def test_a_dim_backdrop_that_does_not_beat_the_opaque_look_fails_monotonicity(self):
        images, mask = backlight_images(panel=0.04)
        self.assertFalse(cc.evaluate_backlight(np, images, mask)["gates"]["monotone"]["pass"])


def sidelight_scene(w050=0.02, w100=0.05, s050=0.03, s200=0.08, queen_scale=1.0):
    """Two pieces, lit on the left half (luminance 1), dark on the right (0 at weight 0); bleed fractions per capture."""
    base = np.zeros((H, W, 3), dtype=np.float32)
    king = np.zeros((H, W), dtype=bool)
    queen = np.zeros((H, W), dtype=bool)
    king[20:70, 10:60] = True
    queen[20:70, 90:140] = True
    lit = np.zeros((H, W), dtype=bool)
    lit[:, 10:35] = True
    lit[:, 90:115] = True
    unlit = (king | queen) & ~lit

    def image(fraction, queen_fraction=None):
        im = np.zeros((H, W, 3), dtype=np.float32)
        im[lit & (king | queen)] = 1.0
        im[unlit & king] = fraction
        im[unlit & queen] = fraction * queen_scale if queen_fraction is None else queen_fraction
        return im

    images = {"sl-w000": image(0.0), "sl-w050": image(w050), "sl-w100": image(w100), "sl-s050": image(s050), "sl-s200": image(s200),
              "sw-w000": image(0.0), "sw-w100": image(0.02)}
    return images, {"king": king, "queen": queen}, {"king": king, "queen": queen}


class SidelightTests(unittest.TestCase):
    def test_monotone_bleed_passes_and_the_controls_fail(self):
        images, black, white = sidelight_scene()
        result = cc.evaluate_sidelight(np, images, black, white)
        self.assertTrue(result["pass"], result)
        king = result["pieces"]["king-black"]
        self.assertAlmostEqual(king["bleed"]["sl-w100"], 0.05, places=4)
        for name, control in cc.sidelight_controls(np, images, black, white).items():
            self.assertFalse(all(p["pass"] for k, p in control["pieces"].items() if k.endswith("-black")), name)

    def test_a_scale_that_does_nothing_fails_only_the_scale_series(self):
        images, black, white = sidelight_scene(s050=0.05, s200=0.05)
        king = cc.evaluate_sidelight(np, images, black, white)["pieces"]["king-black"]
        self.assertTrue(king["weight"]["pass"])
        self.assertFalse(king["scale"]["pass"])

    def test_too_small_regions_are_reported_not_passed(self):
        images, black, white = sidelight_scene()
        black["king"] = np.zeros_like(black["king"])
        result = cc.evaluate_sidelight(np, images, black, white)
        self.assertFalse(result["pieces"]["king-black"]["pass"])
        self.assertIsNone(result["pieces"]["king-black"]["weight"]["series"])

    def test_white_view_gates(self):
        images, black, white = sidelight_scene()
        self.assertTrue(cc.evaluate_sidelight(np, images, black, white)["pieces"]["queen-white"]["pass"])
        images["sw-w100"] = images["sw-w000"]  # no rise at all: the King (gated) fails, the Queen (no fall) passes
        result = cc.evaluate_sidelight(np, images, black, white)
        self.assertFalse(result["pieces"]["king-white"]["pass"])
        self.assertTrue(result["pieces"]["queen-white"]["pass"])

    def test_split_pieces_follows_the_view_handedness(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[:, 2:8] = True
        mask[:, 12:18] = True
        black = cc.split_pieces(np, mask)
        white = cc.split_pieces(np, mask, king_left=False)
        self.assertTrue(black["king"][:, 2:8].all())
        self.assertTrue(white["king"][:, 12:18].all())


def shadow_images(t=(0.27, 0.45, 0.405), board=0.5, tint_white=False):
    region = np.zeros((H, W), dtype=bool)
    region[40:60, 50:110] = True
    hidden = solid((0.0, 0.0, 0.0))
    hidden[20:80, 20:140] = board                      # the lit board
    opaque = hidden.copy()
    opaque[region] = 0.01                              # the opaque top's shadow
    on = opaque.copy()
    on[region] = 0.01 + (board - 0.01) * np.asarray(t, dtype=np.float32)
    white = opaque.copy()
    white[region] = 0.01 + (board - 0.01) * 0.9
    return {"on": on, "opaque": opaque, "hidden": hidden, "white": white}


class ShadowTests(unittest.TestCase):
    def test_an_attenuated_tinted_shadow_passes_and_the_controls_fail(self):
        images = shadow_images()
        result = cc.evaluate_shadow(np, images)
        self.assertTrue(result["pass"], result)
        self.assertGreater(result["region_pixels"], 500)
        self.assertAlmostEqual(result["transmittance_rgb"][1], 0.45, places=2)
        for name, control in cc.shadow_controls(np, images).items():
            self.assertFalse(control["pass"], name)

    def test_the_controls_fail_for_the_right_reason(self):
        images = shadow_images()
        controls = cc.shadow_controls(np, images)
        self.assertFalse(controls["transmission-ignored"]["attenuated"])
        self.assertTrue(controls["tint-ignored"]["attenuated"])
        self.assertFalse(controls["tint-ignored"]["tinted"])

    def test_a_clear_or_wrongly_coloured_shadow_fails(self):
        clear = shadow_images(t=(0.99, 0.99, 0.99))
        self.assertFalse(cc.evaluate_shadow(np, clear)["attenuated"])
        wrong = shadow_images(t=(0.45, 0.45, 0.27))
        self.assertFalse(cc.evaluate_shadow(np, wrong)["tinted"])

    def test_region_excludes_the_tops_own_pixels_and_the_foreground_rows(self):
        images = shadow_images()
        tops = np.zeros((H, W), dtype=bool)
        tops[40:50, 50:80] = True  # the pixels of a top seen directly: hiding it uncovers the board, which must not count as receiver
        plain = cc.shadow_region(np, images["hidden"], images["opaque"])
        masked = cc.shadow_region(np, images["hidden"], images["opaque"], tops)
        self.assertTrue(plain[42:48, 55:75].all())
        self.assertFalse(masked[40 - cc.SHADOW_DILATE_PIXELS:50 + cc.SHADOW_DILATE_PIXELS, 50:80].any())
        limited = cc.shadow_region(np, images["hidden"], images["opaque"], tops, row_limit=50)
        self.assertFalse(limited[50:, :].any())
        self.assertTrue(limited[41:48, 90:105].all())

    def test_shadow_camera_row_limit_is_inside_the_frame(self):
        _, row, depth = cc.project(cc.CAMERA_SHADOW[0], cc.CAMERA_SHADOW[1], (0.0, cc.SHADOW_BOARD_Y, cc.SHADOW_ROW_LIMIT_Z), cc.WIDTH, cc.HEIGHT)
        self.assertTrue(0 < row < cc.HEIGHT)
        self.assertGreater(depth, 0)
        # the black row's shadow (z = 0.1) lies above the limit, the white row's (z = -0.2) below it
        self.assertLess(cc.project(cc.CAMERA_SHADOW[0], cc.CAMERA_SHADOW[1], (0.0, cc.SHADOW_BOARD_Y, 0.1), cc.WIDTH, cc.HEIGHT)[1], row)
        self.assertGreater(cc.project(cc.CAMERA_SHADOW[0], cc.CAMERA_SHADOW[1], (0.0, cc.SHADOW_BOARD_Y, -0.2), cc.WIDTH, cc.HEIGHT)[1], row)

    def test_region_excludes_the_body_shadow_and_needs_pixels(self):
        images = shadow_images()
        # a body shadow present in both B and H (dark in both) must not enter the region
        for k in ("opaque", "hidden", "on", "white"):
            images[k][20:30, 20:60] = 0.01
        region = cc.shadow_region(np, images["hidden"], images["opaque"])
        self.assertFalse(region[20:30, 20:60].any())
        tiny = shadow_images()
        tiny["opaque"][:] = tiny["hidden"]
        self.assertFalse(cc.evaluate_shadow(np, tiny)["pass"])


class ReportTests(unittest.TestCase):
    def report(self, transmission="approximated", approximation="rt_refraction"):
        feature = lambda name, support, approx: {"feature": name, "support": support, "approximation": approx, "source": "gltf", "label": "ratified"}  # noqa: E731
        return {"materials": {"materials": [
            {"name": "Pawn_Top_Black", "features": [feature("transmission", transmission, approximation), feature("volume", "implemented", "")]}]}}

    def test_support_rows(self):
        ok, rows = cc.evaluate_support(self.report(), {"Pawn_Top_Black": ["transmission", "volume"]})
        self.assertTrue(ok)
        self.assertEqual([r["support"] for r in rows], ["approximated", "implemented"])

    def test_unsupported_unnamed_or_missing_fails(self):
        self.assertFalse(cc.evaluate_support(self.report("unsupported", ""), {"Pawn_Top_Black": ["transmission"]})[0])
        self.assertFalse(cc.evaluate_support(self.report("approximated", ""), {"Pawn_Top_Black": ["transmission"]})[0])
        self.assertFalse(cc.evaluate_support(self.report(), {"Pawn_Top_Black": ["subsurface"]})[0])
        self.assertFalse(cc.evaluate_support(self.report(), {"King_Black": ["subsurface"]})[0])

    def test_pipeline_health(self):
        good = "PIPELINES requests 10 | archive hits 10, misses 0, unavailable 0 (miss rate 0.0%) | compiler calls 0 (0.0 ms, max 0.0 ms) | render-thread compiles 0 (0.0 ms) | fallbacks 0 (draws 0) | failures 0"
        self.assertTrue(cc.pipeline_health(good)["pass"])
        self.assertFalse(cc.pipeline_health(good.replace("misses 0", "misses 2"))["pass"])
        self.assertFalse(cc.pipeline_health(good.replace("compiles 0", "compiles 3"))["pass"])
        startup = "STARTUP first frame submitted 400 ms after launch | PIPELINES requests 9 | archive hits 9, misses 0 | render-thread compiles 2 (0.3 ms) | fallbacks 0 (draws 0)\n"
        late = cc.pipeline_health(startup + good.replace("compiles 0", "compiles 2"))
        self.assertTrue(late["pass"])  # the two startup compiles are the phase baseline
        self.assertEqual(late["startup_render_thread_compiles"], 2)
        self.assertFalse(cc.pipeline_health(startup + good.replace("compiles 0", "compiles 3"))["pass"])
        self.assertFalse(cc.pipeline_health(good.replace("draws 0", "draws 9"))["pass"])
        self.assertIsNone(cc.pipeline_health("nothing"))

    def test_verdict(self):
        green = [{"pass": True, "controls_fail": True}]
        self.assertEqual(cc.verdict(green, "green"), ("GREEN", cc.EXIT_OK))
        self.assertEqual(cc.verdict([{"pass": False, "controls_fail": True}], "green"), ("RED", cc.EXIT_UNEXPECTED))
        self.assertEqual(cc.verdict([{"pass": True, "controls_fail": False}], "green"), ("NON_DISCRIMINATING", cc.EXIT_NON_DISCRIMINATING))
        self.assertEqual(cc.verdict([{"pass": False, "controls_fail": True}], "red"), ("RED", cc.EXIT_OK))


class SssBound(unittest.TestCase):
    """Ticket 20, RT tier: the textured subsurface bound captures of the kings and queens."""

    def test_the_captures_per_view(self):
        for prefix, view in (("rw", "white"), ("rb", "black")):
            main, twin, control = (cc.CAPTURES[f"{prefix}-{k}"] for k in ("w100", "off", "w000"))
            self.assertEqual(main["scenario"], f"chess-sidelight/view={view}/zoom=3/weight=1")
            self.assertTrue(main["flags"].get("export"))  # the snapshot the Mitsuba reference reads
            self.assertIn("--debug-subsurface-walk-lights", twin["flags"]["extra"])  # the noise twin and negative control
            self.assertEqual(control["scenario"], f"chess-sidelight/view={view}/zoom=3/weight=0")  # the base model: negative control
            for spec in (main, twin, control):
                self.assertEqual(spec["group"], "sssbound")
                self.assertTrue(spec["flags"]["rt"])
                self.assertIn("rt-walk", spec["flags"]["extra"])

    def test_the_command_exports_the_snapshot_only_for_the_reference_capture(self):
        folder = pathlib.Path("/tmp/x")
        main = cc.capture_command(pathlib.Path("phosphor"), pathlib.Path("g.gltf"), pathlib.Path("o.json"), "rw-w100", folder)
        twin = cc.capture_command(pathlib.Path("phosphor"), pathlib.Path("g.gltf"), pathlib.Path("o.json"), "rw-off", folder)
        self.assertIn("--export-reference", main)
        self.assertNotIn("--export-reference", twin)
        self.assertIn("--subsurface-mode", main)
        self.assertEqual(main[main.index("--rt") + 1], "on")

    def test_a_capture_that_rendered_the_diffusion_is_a_capture_error(self):
        import json, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = pathlib.Path(tmp)
            (folder / "report.json").write_text(json.dumps({"subsurface": {"requested": "rt-walk", "effective": "diffusion", "fallback_reason": "--rt off"}}))
            with self.assertRaises(cc.fc.CaptureFailed) as raised:
                cc.require_walk(folder, "rw-w100")
            self.assertIn("--rt off", str(raised.exception))
            (folder / "report.json").write_text(json.dumps({"subsurface": {"requested": "rt-walk", "effective": "rt-walk", "fallback_reason": ""}}))
            cc.require_walk(folder, "rw-w100")

    def test_the_default_groups_do_not_run_the_rt_tier(self):
        self.assertNotIn("sssbound", ["backlight", "sidelight", "shadow"])
        self.assertTrue(any(s["group"] == "sssbound" for s in cc.CAPTURES.values()))


if __name__ == "__main__":
    unittest.main()
