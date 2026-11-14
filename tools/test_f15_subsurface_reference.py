#!/usr/bin/env python3
"""Unit tests of tools/f15_subsurface_reference.py (the Mitsuba volumetric-subsurface ground truth, ticket 17).

Expected values come from the draft text, from analytic identities or from an independent numpy random walk of a semi-infinite
medium, never from re-running the reference's own mapping.  The Mitsuba tests render tiny scenes (a few seconds).

Run offline:
  uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy python -I tools/test_f15_subsurface_reference.py
Without Mitsuba the Mitsuba tests are skipped; numpy alone runs the rest.
"""
from __future__ import annotations

import json
import math
import pathlib
import struct
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import f15_reference as base  # noqa: E402
import f15_subsurface_reference as sss  # noqa: E402

try:
    import mitsuba  # noqa: F401
    HAVE_MITSUBA = True
except ImportError:  # pragma: no cover
    HAVE_MITSUBA = False

INVALID = base.INVALID
SPHERE_RADIUS = 0.4
INDEX_MATCHED_TOLERANCE = 0.05      # measured worst |A - rho| 0.043 (rho 0.3), 0.035 (0.9); the numpy walk of the Kulla-Conty test: 0.01-0.07
ANGLE_SPREAD_BOUND = 0.10      # measured: diffuse vs 0 / 45 / 70 degrees differ by <= 6 % up to alpha 0.99 (A <= 0.36)


def f32bits(x):
    return int.from_bytes(struct.pack("<f", x), "little")


def write_sphere_ply(path, radius=SPHERE_RADIUS, slices=64, stacks=32, outward=False):
    """The engine's UV sphere (RayCastedSnapshot.write_sphere_ply of test_f15_reference.py)."""
    verts, faces = [], []
    for st in range(stacks + 1):
        theta = math.pi * st / stacks
        for sl in range(slices + 1):
            phi = 2 * math.pi * sl / slices
            n = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
            verts.append((*(c * radius for c in n), *n, sl / slices, st / stacks))
    for st in range(stacks):
        for sl in range(slices):
            a = st * (slices + 1) + sl
            b = a + slices + 1
            faces += [(a, a + 1, b), (a + 1, b + 1, b)] if outward else [(a, b, a + 1), (a + 1, b, b + 1)]
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
                 "property float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float s\n"
                 "property float t\nelement face %d\nproperty list uchar uint vertex_indices\nend_header\n"
                 % (len(verts), len(faces))).encode())
        for v in verts:
            f.write(struct.pack("<8f", *v))
        for tri in faces:
            f.write(struct.pack("<B3I", 3, *tri))


def volumetric_records(weight, color, radius, scale, g, thickness=0.8, attenuation=(1.0, 1.0, 1.0), distance=math.inf, mode=1):
    record0 = [0] * 20
    for k in (5, 6, 10, 11, 12, 17, 18):
        record0[k] = INVALID
    record1 = [0] * 20
    record1[2], record1[7], record1[8], record1[10], record1[12] = INVALID, INVALID, INVALID, INVALID, INVALID
    record1[11] = f32bits(thickness)
    for k in range(3):
        record1[13 + k] = f32bits(attenuation[k])
    record1[16] = f32bits(distance)
    record2 = [0] * 20
    record2[0] = mode
    record2[1] = f32bits(weight)
    for k in range(3):
        record2[2 + k] = f32bits(color[k])
        record2[6 + k] = f32bits(scale[k])
    record2[5] = f32bits(radius)
    record2[9] = f32bits(g)
    for k in (10, 11, 12):
        record2[k] = INVALID
    return [record0, record1, record2, [0] * 20]


def make_snapshot(directory, pieces, sun, outward=False, width=320, height=180, roughness=0.7, base_color=(0.8, 0.8, 0.8), camera=(0.0, 0.0, 4.0),
                  sky=0.0):
    """pieces: list of dicts {position, block: None | dict(weight, color, radius, scale, g)}; one material per piece."""
    root = pathlib.Path(directory)
    write_sphere_ply(root / "mesh_0.ply", outward=outward)
    materials, records, instances = [], [], []
    for k, piece in enumerate(pieces):
        block = piece.get("block")
        bits = 0
        if block is not None:
            bits = sss.SUBSURFACE_BIT | sss.VOLUME_BIT
            records += volumetric_records(block["weight"], block["color"], block["radius"], block["scale"], block["g"])
        else:
            records += [[0] * 20] * 4
        materials.append({"base": [*base_color, 1], "emissive": [0, 0, 0], "metallic": 0.0, "roughness": roughness, "normal_scale": 1,
                          "occlusion_strength": 1, "alpha_cutoff": 0, "flags": 0, "feature_bits": bits,
                          "extension_index": 4 * k, "textures": [INVALID] * 5})
        x, y, z = piece["position"]
        instances.append({"slot": k, "generation": 1, "flags": 7, "mesh": 0, "material": k,
                          "world": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]})
    data = {"schema": 1, "state": "TEST", "linear": True, "units": "metres", "frame": 0, "revisions": [0, 0, 0],
            "camera": {"position": list(camera), "direction": [0, 0, -1], "up": [0, 1, 0], "fov_y_radians": math.radians(60),
                       "near": 0.05, "far": 0, "width": width, "height": height, "jitter_pixels": [0, 0],
                       "projection": "perspective_reverse_z_infinite"},
            "sky": [sky] * 3, "sun_angular_radius": 0.00465, "textures": [], "materials": materials,
            "material_extensions": {"version": 1, "records_per_material": 4, "records": records},
            "mesh_version": 1, "meshes": ["mesh_0.ply"], "instances": instances, "lights": [], "sampled_lights": []}
    if sun:
        toward, irradiance = sun
        data["lights"].append({"type": 0, "position": [0, 0, 0], "direction": [float(-c) for c in toward], "color": [1, 1, 1],
                               "intensity": irradiance, "range": 0, "inner": 0, "outer": 0})
    (root / "scene.json").write_text(json.dumps(data))
    return root


def unit(*v):
    v = np.array(v, dtype=np.float64)
    return v / np.linalg.norm(v)


SUN = (unit(-0.9, 0.25, 0.35), 3.0)
SKIN = dict(weight=1.0, color=[0.85, 0.55, 0.45], radius=0.08, scale=[1.0, 0.5, 0.25], g=0.0)


class Mapping(unittest.TestCase):
    def test_kulla_conty_inversion_hand_values(self):
        # rho = 0: s = 4.09712 - sqrt(9.59217) = 1.0000 (within the fit), rho_ss = 0.  rho = 0.5: worked from the draft text.
        self.assertAlmostEqual(float(sss.single_scatter_albedo(0.0)), 0.0, places=12)
        rho = 0.5
        s = 4.09712 + 4.20863 * rho - math.sqrt(9.59217 + 41.6808 * rho + 17.7126 * rho * rho)
        self.assertAlmostEqual(s, 4.09712 + 2.104315 - math.sqrt(9.59217 + 20.8404 + 4.42815), places=9)
        self.assertAlmostEqual(float(sss.single_scatter_albedo(0.5)), 1.0 - s * s, places=12)
        self.assertGreater(float(sss.single_scatter_albedo(0.5)), 0.9)
        self.assertAlmostEqual(float(sss.single_scatter_albedo(1.0)), 1.0, places=4)
        # with anisotropy: (1 - s^2) / (1 - g s^2)
        self.assertAlmostEqual(float(sss.single_scatter_albedo(0.5, 0.3)), (1.0 - s * s) / (1.0 - 0.3 * s * s), places=12)
        values = [float(sss.single_scatter_albedo(r)) for r in np.linspace(0.0, 1.0, 21)]
        self.assertTrue(all(b > a for a, b in zip(values, values[1:])))

    def test_inversion_against_an_independent_semi_infinite_random_walk(self):
        # Independent numpy walk (no code shared): a semi-infinite medium with a matched boundary, normal incidence, the escaping
        # fraction.  The fit targets the diffuse albedo, so it is only close: it is measured to be below the target by 0.01-0.07
        # at every albedo and anisotropy and it is monotone.
        rng = np.random.default_rng(7)

        def escaping_fraction(albedo, g, n=40000):
            mu, w, out, alive = np.ones(n), np.ones(n), 0.0, np.ones(n, dtype=bool)
            z = -np.log(1.0 - rng.random(n))
            while alive.any():
                idx = np.flatnonzero(alive)
                w[idx] *= albedo
                xi = rng.random(idx.size)
                if abs(g) < 1e-9:
                    c = 1.0 - 2.0 * xi
                else:
                    t = (1 - g * g) / (1 - g + 2 * g * xi)
                    c = (1 + g * g - t * t) / (2 * g)
                phi = rng.random(idx.size) * 2 * math.pi
                m = mu[idx]
                mu[idx] = m * c + np.sqrt(np.maximum(0, 1 - m * m)) * np.sqrt(np.maximum(0, 1 - c * c)) * np.cos(phi)
                z[idx] += -np.log(1.0 - rng.random(idx.size)) * mu[idx]
                escaped = z[idx] < 0
                out += w[idx[escaped]].sum()
                alive[idx[escaped | (w[idx] < 1e-6)]] = False
            return out / n

        previous = {}
        for g in (0.0, 0.3, -0.3):
            for rho in (0.3, 0.6, 0.85, 0.95):
                measured = escaping_fraction(float(sss.single_scatter_albedo(rho, g)), g)
                self.assertLess(measured, rho + 0.01)
                self.assertGreater(measured, rho - 0.07, f"g {g} rho {rho}: {measured}")
                if g in previous:
                    self.assertGreater(measured, previous[g])
                previous[g] = measured

    def test_medium_coefficients(self):
        block = dict(color=[0.30, 0.40, 0.20], radius=0.08, radius_scale=[1.0, 0.5, 0.25], anisotropy=0.0, weight=1.0)
        alpha_b = sss.fixture_alpha_b()
        medium = sss.medium_coefficients(block, None, alpha_b)
        np.testing.assert_allclose(medium["mfp"], [0.08, 0.04, 0.02])
        np.testing.assert_allclose(medium["sigma_t"], [12.5, 25.0, 50.0])
        np.testing.assert_allclose(medium["sigma_s"] + medium["sigma_a"], medium["sigma_t"])
        # the mapping is the boundary-aware inversion, not the draft's: the observed albedo reached is the colour
        np.testing.assert_allclose(medium["albedo_reached"], block["color"], atol=2e-4)
        self.assertFalse(any(medium["saturated"]))
        self.assertTrue((medium["albedo"] > np.asarray(sss.single_scatter_albedo(np.array(block["color"])))).all())
        # the radius override scales every channel
        np.testing.assert_allclose(sss.medium_coefficients(block, 0.16, alpha_b)["mfp"], [0.16, 0.08, 0.04])
        with self.assertRaises(ValueError):
            sss.medium_coefficients({**block, "radius_scale": [1.0, 0.0, 0.25]}, None, alpha_b)
        with self.assertRaises(ValueError):
            sss.medium_coefficients(block)  # the boundary alpha is mandatory
        with self.assertRaises(ValueError):
            sss.medium_coefficients(block, None, 0.123)  # no curve for another boundary: regenerate the table
        # a colour above the energy the boundary lets out saturates at alpha = 1 and says so
        bright = sss.medium_coefficients({**block, "color": [0.9, 0.9, 0.9]}, None, alpha_b)
        self.assertTrue(all(bright["saturated"]))
        np.testing.assert_allclose(bright["albedo"], 1.0)


    def test_blocks_are_decoded_from_the_records_and_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            from f12_reference import snapshot as load
            root = make_snapshot(tmp, [dict(position=(0, 0, 0), block=SKIN), dict(position=(1, 0, 0), block=None),
                                       dict(position=(-1, 0, 0), block={**SKIN, "weight": 0.0})], SUN)
            _, data = load(root)
            blocks = sss.scene_blocks(data)
            self.assertEqual(list(blocks), [0])  # weight 0 and the plain piece have no medium
            b = blocks[0]
            self.assertEqual((b["weight"], b["radius"], b["anisotropy"]), (1.0, np.float32(0.08), 0.0))
            np.testing.assert_allclose(b["color"], [0.85, 0.55, 0.45], rtol=1e-6)
            # refusals
            def refuses(patch, message):
                bad = json.loads(json.dumps(data))
                patch(bad)
                with self.assertRaisesRegex(ValueError, message):
                    sss.scene_blocks(bad)
            records = lambda d: d["material_extensions"]["records"]  # noqa: E731
            refuses(lambda d: records(d)[2].__setitem__(0, 0), "thin")
            refuses(lambda d: records(d)[2].__setitem__(10, 3), "texture")
            refuses(lambda d: d["materials"][0].__setitem__("feature_bits", 128 | 64 | 2), "another material feature")
            refuses(lambda d: records(d)[1].__setitem__(11, 0), "thickness")
            refuses(lambda d: (records(d)[1].__setitem__(13, f32bits(0.5)), records(d)[1].__setitem__(16, f32bits(2.0))), "attenuation")
            refuses(lambda d: records(d)[2].__setitem__(5, 0), "positive radius")
            refuses(lambda d: records(d)[2].__setitem__(1, f32bits(1.5)), "weight")
            refuses(lambda d: records(d)[2].__setitem__(9, f32bits(1.0)), "anisotropy")

    def test_off_and_diffusion_off_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            from f12_reference import snapshot as load
            _, data = load(make_snapshot(tmp, [dict(position=(0, 0, 0), block={**SKIN, "weight": 0.5}),
                                               dict(position=(1, 0, 0), block=None)], SUN))
            blocks = sss.scene_blocks(data)
            off = sss.off_data(data)
            self.assertEqual([m["feature_bits"] for m in off["materials"]], [0, 0])
            self.assertFalse(base.uses_subsurface(off))
            diffusion = sss.diffusion_off_data(data, blocks)
            np.testing.assert_allclose(diffusion["materials"][0]["base"][:3], [0.5 * 0.8 + 0.5 * c for c in blocks[0]["color"]],
                                       rtol=1e-6)
            self.assertEqual(diffusion["materials"][1]["base"], data["materials"][1]["base"])
            self.assertEqual(sss.nominal_radius({0: {"radius": 1.0}, 1: {"radius": 2.0}, 2: {"radius": 8.0}}), 2.0)


class ObservedAlbedoTable(unittest.TestCase):
    """The boundary-aware inversion: the committed table, its inversion and the checks of the measurement itself."""

    def curve(self, g=0.0, ior=sss.BOUNDARY_IOR):
        return sss.curve_for(g, sss.fixture_alpha_b(), ior)

    def test_table_covers_the_fixture_combinations_and_is_monotone_within_noise(self):
        for g in (0.0, 0.3):
            for ior in (sss.BOUNDARY_IOR, 1.0):
                points = self.curve(g, ior)["points"]
                self.assertEqual([p["albedo"] for p in points], sss.ALBEDO_GRID)
                self.assertAlmostEqual(points[0]["A"], 0.0, delta=1e-3)
                # (the index-matched curve above 0.99 is truncated by the path depth bound and noisy: not used by the inversion)
                usable = [p for p in points if ior != 1.0 or p["albedo"] <= 0.99]
                for lo, hi in zip(usable, usable[1:]):
                    self.assertGreater(hi["A"] + 3 * (lo["noise"] + hi["noise"]) + 0.002, lo["A"], (g, ior, lo["albedo"]))
                # the measurement noise is small against the 10 % bound it serves (largest near the A_max plateau)
                self.assertLess(max(p["noise"] for p in usable if p["albedo"] <= 0.99), 0.004)
                self.assertLess(max(p["noise"] for p in usable), 0.04)   # heavy tails of the plateau; pooled by the isotonic fit

    def test_inversion_reproduces_the_table_and_saturates_above_the_energy_the_boundary_returns(self):
        a_max = sss.observed_albedo(1.0, self.curve())
        self.assertGreater(a_max, 0.3)
        self.assertLess(a_max, 0.6)       # the lossy single-scattering GGX boundary lets out less than 0.6
        for g in (0.0, 0.3):
            curve = self.curve(g)
            a_max = sss.observed_albedo(1.0, curve)
            for target in (0.02, 0.1, 0.2, 0.3, 0.4, 0.95 * a_max):
                alpha, reached, saturated = sss.invert_observed_albedo(target, curve)
                self.assertFalse(saturated)
                self.assertAlmostEqual(reached, target, delta=1e-6)
                self.assertAlmostEqual(sss.observed_albedo(alpha, curve), target, delta=1e-6)
            # every tabulated point is returned by its own A within the interpolation
            for p in curve["points"][1:-1]:
                alpha, reached, _ = sss.invert_observed_albedo(p["A"], curve)
                self.assertAlmostEqual(sss.observed_albedo(alpha, curve), min(p["A"], a_max), delta=1e-6)
        self.assertEqual(sss.invert_observed_albedo(0.0, curve), (0.0, 0.0, False))
        alpha, reached, saturated = sss.invert_observed_albedo(0.9, curve)
        self.assertEqual((alpha, saturated), (1.0, True))
        self.assertAlmostEqual(reached, sss.observed_albedo(1.0, curve))

    def test_the_draft_mapping_fails_the_openpbr_relation_under_this_boundary(self):
        # Negative control: the Kulla-Conty alpha(C) (boundary ignored) behind the ior 1.5 rough boundary returns about half of C
        # for the fixture colours; the boundary-aware inversion returns C wherever the boundary allows it.
        for g in (0.0, 0.3):
            curve = self.curve(g)
            for colour in (0.85, 0.55, 0.45):
                draft = sss.observed_albedo(float(sss.single_scatter_albedo(colour, g)), curve)
                self.assertLess(draft / colour, 0.62, (g, colour, draft))
            self.assertGreater(0.85 / sss.observed_albedo(float(sss.single_scatter_albedo(0.85, g)), curve), 1.6)  # the ~2x miss
            alpha, reached, saturated = sss.invert_observed_albedo(0.30, curve)
            self.assertAlmostEqual(reached, 0.30, delta=1e-6)
            self.assertGreater(alpha, float(sss.single_scatter_albedo(0.30, g)))

    def test_index_matched_boundary_agrees_with_the_draft_relation(self):
        # A check of the measurement itself: without an interface (ior 1) the observed albedo under diffuse incidence is what the
        # boundary-blind van de Hulst / Kulla-Conty relation describes (g = 0; the fit is of the diffuse albedo).
        curve = self.curve(0.0, 1.0)
        worst = 0.0
        for rho in (0.1, 0.3, 0.5, 0.7, 0.9):  # above, alpha -> 1 paths exceed the 1e5 depth bound
            predicted = sss.observed_albedo(float(sss.single_scatter_albedo(rho)), curve)
            worst = max(worst, abs(predicted - rho))
        self.assertLess(worst, INDEX_MATCHED_TOLERANCE)

    def test_angle_dependence_is_recorded_and_bounded(self):
        # The table is built for diffuse incidence; the per-direction values at 0 / 45 / 70 degrees (reciprocity) differ by the
        # declared approximation of using the diffuse value.
        for p in self.curve()["points"]:
            if p["A"] < 0.05 or p["albedo"] > 0.99:   # (the plateau above is dominated by noise)
                continue
            spread = max(abs(v - p["A"]) for v in p["A_angle"].values())
            self.assertLess(spread / p["A"], ANGLE_SPREAD_BOUND, p)

    def test_a_fresh_measurement_reproduces_a_table_point_within_the_noise(self):
        if not HAVE_MITSUBA:
            self.skipTest("mitsuba not installed")
        mi_, dr = sss.mitsuba_modules(2)
        masks, res = sss.angle_masks(64), 64
        alpha_b = sss.fixture_alpha_b()
        spec = [sss.measure_albedo_point(mi_, dr, 0.0, 0.0, alpha_b, sss.BOUNDARY_IOR, 64, 5 + h, res) for h in range(2)]
        full = [sss.measure_albedo_point(mi_, dr, 0.99, 0.0, alpha_b, sss.BOUNDARY_IOR, 64, 9 + h, res) for h in range(2)]
        measured = sss.observed_albedos(0.5 * (full[0] + full[1]), 0.5 * (spec[0] + spec[1]), masks)["diffuse"]
        tabulated = next(p["A"] for p in self.curve()["points"] if p["albedo"] == 0.99)
        self.assertAlmostEqual(measured, tabulated, delta=0.02)


class MeshOrientation(unittest.TestCase):
    def test_signed_volume_tells_the_winding_and_a_mirror_flips_it(self):
        from f12_reference import snapshot as load
        for outward in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root, data = load(make_snapshot(tmp, [dict(position=(1, 2, 3), block=None)], SUN, outward=outward))
                self.assertEqual(sss.mesh_orientation(root, data, data["instances"][0]), 1 if outward else -1)
                mirrored = json.loads(json.dumps(data["instances"][0]))
                mirrored["world"][0] = -1.0
                self.assertEqual(sss.mesh_orientation(root, data, mirrored), -1 if outward else 1)


class DiffusionRegime(unittest.TestCase):
    def test_regime_is_ten_percent_of_the_piece_size_on_the_largest_channel(self):
        self.assertEqual(sss.DIFFUSION_REGIME_FRACTION, 0.1)
        # skin radius 0.08 x scale 1.0 = 0.08 m on a 0.8 m piece: exactly at the limit (in); 0.16 m: out
        for radius, expected in ((0.04, True), (0.08, True), (0.16, False), (0.24, False)):
            mfp = max(np.float32(radius) * np.float32(s) for s in (1.0, 0.5, 0.25))
            self.assertEqual(bool(mfp <= sss.DIFFUSION_REGIME_FRACTION * np.float32(0.8) * (1 + 1e-4)), expected)


class Pooling(unittest.TestCase):
    def test_box_pool_uses_only_valid_pixels_and_keeps_a_constant(self):
        image = np.ones((9, 9, 3)) * 2.0
        image[0, :] = 100.0  # an invalid row must not leak in
        valid = np.ones((9, 9), dtype=bool)
        valid[0, :] = False
        pooled = sss.box_pool(image, valid, 5)
        np.testing.assert_allclose(pooled[1:], 2.0)
        self.assertTrue((pooled[0] <= 2.0 + 1e-12).all())  # invalid pixels receive the mean of their valid neighbours
        np.testing.assert_allclose(sss.box_pool(image, valid, 1)[valid], 2.0)

    def test_pooling_a_smooth_ramp_changes_it_by_less_than_the_gate_bound(self):
        # The image of a piece is smooth at the scale of the mean free path (>= 4 px for the luminance-dominant channel of the
        # fixtures): a 5 x 5 mean of a ramp of that scale (a logistic of width 4 px) moves it by far less than the 10 % bound.
        x = np.arange(120, dtype=np.float64)
        ramp = 0.02 + 0.8 / (1.0 + np.exp(-(x - 60.0) / 4.0))
        image = np.tile(ramp[None, :, None], (40, 1, 3))
        pooled = sss.box_pool(image, np.ones((40, 120), dtype=bool), 5)
        keep = slice(4, -4)
        floor = 0.05 * ramp.mean()
        error = np.abs(pooled[10, keep, 0] - ramp[keep]) / np.maximum(ramp[keep], floor)
        self.assertLess(float(error.mean()), 0.01)
        self.assertLess(float(error.max()), 0.05)

    def test_interior_mask_keeps_the_piece_without_its_ring(self):
        grid = np.full((20, 20), -1)
        grid[5:15, 5:15] = 3
        interior = sss.interior_mask(grid, 3)
        self.assertEqual(int(interior.sum()), 8 * 8)
        self.assertFalse(interior[5, 5] or interior[14, 9])


@unittest.skipUnless(HAVE_MITSUBA, "mitsuba not installed")
class MitsubaConventions(unittest.TestCase):
    def test_a_diffuse_sphere_agrees_with_lambert_under_the_snapshot_camera_and_sun(self):
        # Camera, film crop, sun direction and irradiance conventions of the scene dictionary against the analytic Lambert radiance
        # E a max(N.L, 0) / pi of the same snapshot on the interior pixels (area averaging vs pixel-centre rays: smooth image).
        from f12_reference import snapshot as load
        import mitsuba as mi
        with tempfile.TemporaryDirectory() as tmp:
            root, data = load(make_snapshot(tmp, [dict(position=(0, 0, 0), block=None)], SUN, width=640, height=360))
            cast = base.cast_primary_rays(root, data, 2)
            mi_, dr = sss.mitsuba_modules(2)
            description = {"type": "scene", "integrator": {"type": "path", "max_depth": 2},
                           "sensor": sss.camera_dict(mi_, data, (0, 0, 640, 360), 64),
                           "piece": {"type": "ply", "filename": str(root / "mesh_0.ply"), "face_normals": False,
                                     "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.8, 0.8, 0.8]}}},
                           **sss.sun_emitters(data)}
            image = np.array(mi_.render(mi_.load_dict(description), spp=64))
            n = cast[2].reshape(360, 640, 3)
            lambert = 3.0 * 0.8 / math.pi * np.maximum((n * SUN[0]).sum(axis=-1), 0.0)
            interior = sss.interior_mask(cast[3], 0)
            lit = interior & (lambert > 0.05)
            relative = np.abs(image[..., 1] - lambert)[lit] / lambert[lit]
            self.assertLess(float(relative.mean()), 0.02)
            self.assertGreater(int(lit.sum()), 800)

    def test_a_non_scattering_medium_leaves_only_the_boundary_reflection(self):
        # albedo 0: every refracted path is absorbed, so the image is the boundary's own specular reflection of the sun, which the
        # Appendix B specular lobe of the base model approximates (Schlick vs the exact Fresnel, correlated vs separable Smith).
        from f12_reference import snapshot as load
        with tempfile.TemporaryDirectory() as tmp:
            root, data = load(make_snapshot(tmp, [dict(position=(0, 0, 0), block=SKIN)], (unit(-0.5, 0.3, 0.8), 3.0), width=320, height=180,
                                            roughness=0.5))
            cast = base.cast_primary_rays(root, data, 2)
            crop = sss.piece_crop(cast[3], 0)
            alpha = float(base.effective_alpha(0.5))
            image = sss.render_grey(root, data, 0, crop, alpha, 25.0, 0.0, 0.0, 256, 0, 2)
            x0, y0, x1, y1 = crop
            off = base.render_frame(root, sss.off_data(data), base.APPENDIX_B, threads=2, cast=cast)[0]
            # The off model includes the Lambert diffuse; compare against its specular part: bsdf with base 0.
            plain = sss.off_data(data)
            plain["materials"][0]["base"] = [0.0, 0.0, 0.0, 1.0]
            specular = base.render_frame(root, plain, base.APPENDIX_B, threads=2, cast=cast)[0]
            interior = sss.interior_mask(cast[3], 0)[y0:y1, x0:x1]
            reference = specular[y0:y1, x0:x1, 1]
            region = interior & (reference > 0.2 * reference.max())
            self.assertGreater(int(region.sum()), 50)
            ratio = float(image[..., 1][region].sum() / reference[region].sum())
            self.assertGreater(ratio, 0.8)
            self.assertLess(ratio, 1.25)
            self.assertLess(float(off.max()), 10.0)

    def test_end_to_end_render_writes_the_reference_files_and_composes_the_weight(self):
        from f12_reference import read_pfm
        with tempfile.TemporaryDirectory() as tmp:
            root = make_snapshot(pathlib.Path(tmp), [dict(position=(-0.6, 0, 0), block={**SKIN, "weight": 0.5, "radius": 0.04}),
                                                    dict(position=(0.6, 0, 0), block=None)], SUN, width=320, height=180)
            out = pathlib.Path(tmp) / "out"
            code = sss.main(["render", str(root), "--out", str(out), "--controls", "subsurface-off", "diffusion-off",
                             "radius-ignored", "subsurface-leak", "--spp", "64", "--pool", "3", "--threads", "2",
                             "--cache-dir", str(pathlib.Path(tmp) / "cache")])
            self.assertEqual(code, 0)
            report = json.loads((out / "reference.json").read_text())
            self.assertEqual(sorted(report["controls"]), ["diffusion-off", "radius-ignored", "subsurface-leak", "subsurface-off"])
            reference = read_pfm(out / "reference.pfm")
            off = read_pfm(out / "control-subsurface-off.pfm")
            masks = np.load(out / "masks.npz")
            plain = masks["sphere"] == 1
            np.testing.assert_array_equal(reference[plain], off[plain])  # the piece without a block is the base model
            piece = masks["roi"] & (masks["sphere"] == 0)
            self.assertGreater(float(np.abs(reference[piece] - off[piece]).mean()), 1e-3)  # the weighted medium changes the piece
            self.assertEqual(report["spheres"][0]["weight"], 0.5)
            # mfp max 0.04 m on a 0.8 m piece: inside the 10 % diffusion regime of the orchestrator decision
            self.assertTrue(report["spheres"][0]["diffusion_regime"])
            self.assertAlmostEqual(report["spheres"][0]["subsurface"]["mfp_max"], 0.04, places=6)
            self.assertNotIn("weight", report["spheres"][1])
            self.assertIn("residual_noise_per_piece", report["ground_truth"])
            # The nominal radius of a one-piece scene is its own radius: radius-ignored equals the reference.
            np.testing.assert_allclose(read_pfm(out / "control-radius-ignored.pfm")[piece], reference[piece], rtol=1e-6)
            self.assertGreater(float(np.abs(read_pfm(out / "control-subsurface-leak.pfm")[piece] - reference[piece]).mean()), 1e-3)


    def test_no_ground_truth_skips_the_random_walk_of_the_reference_but_keeps_the_leak_control(self):
        from f12_reference import read_pfm
        with tempfile.TemporaryDirectory() as tmp:
            root = make_snapshot(pathlib.Path(tmp), [dict(position=(0.0, 0, 0), block={**SKIN, "radius": 0.0005})], SUN, width=160,
                                 height=90)
            out = pathlib.Path(tmp) / "out"
            code = sss.main(["render", str(root), "--out", str(out), "--controls", "subsurface-leak", "--spp", "32", "--pool", "3",
                             "--threads", "2", "--cache-dir", str(pathlib.Path(tmp) / "cache"), "--no-ground-truth"])
            self.assertEqual(code, 0)
            report = json.loads((out / "reference.json").read_text())
            self.assertFalse(report["ground_truth"]["ground_truth_rendered"])
            reference = read_pfm(out / "reference.pfm")
            leak = read_pfm(out / "control-subsurface-leak.pfm")
            masks = np.load(out / "masks.npz")
            piece = masks["roi"] & (masks["sphere"] == 0)
            self.assertGreater(float(np.abs(leak[piece] - reference[piece]).mean()), 1e-3)
            self.assertEqual(len(list((pathlib.Path(tmp) / "cache").glob("piece-*.npz"))), 3)  # the leak control only: three channels


if __name__ == "__main__":
    unittest.main(verbosity=2)
