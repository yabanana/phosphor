#!/usr/bin/env python3
"""Unit tests of tools/f15_reference.py (the glTF Appendix B reference).

Expected values come from the specification or from analytic identities, or from
an independent Monte Carlo estimate written in wi-space (the reference
integrates in half-vector space), never from re-running the reference formula.

Run offline:
  uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy python -I tools/test_f15_reference.py
Without Mitsuba the ray-casting test is skipped; numpy alone runs the rest.
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
import f15_reference as ref  # noqa: E402

try:
    import mitsuba  # noqa: F401
    HAVE_MITSUBA = True
except ImportError:  # pragma: no cover
    HAVE_MITSUBA = False


def unit(*v):
    v = np.array(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def spec_f0(f0, vh):
    return f0 + (1 - f0) * (1 - vh) ** 5


class RoughnessConventions(unittest.TestCase):
    def test_clamp_and_alpha(self):
        self.assertAlmostEqual(float(ref.effective_alpha(0.0)), 0.002)    # clamp 0.04 -> 0.0016 -> floor 0.002
        self.assertAlmostEqual(float(ref.effective_alpha(0.04)), 0.002)
        self.assertAlmostEqual(float(ref.effective_alpha(0.1)), 0.01)
        self.assertAlmostEqual(float(ref.effective_alpha(1.0)), 1.0)
        self.assertAlmostEqual(float(ref.effective_alpha(2.0)), 1.0)
        self.assertAlmostEqual(float(ref.effective_alpha(0.5, ref.CONTROLS["alpha-roughness"])), 0.5)
        self.assertAlmostEqual(ref.DIELECTRIC_F0, 0.04)


class AppendixBMix(unittest.TestCase):
    def setUp(self):
        self.n = np.array([[0.0, 0.0, 1.0]])
        self.v = np.array([unit(0.3, 0.1, 1.0)])
        self.l = np.array([unit(-0.5, 0.2, 0.7)])
        self.base = np.array([[0.8, 0.35, 0.15]])
        self.roughness = np.array([0.5])

    def parts(self):
        n, v, l = self.n[0], self.v[0], self.l[0]
        alpha = 0.25
        h = unit(*(v + l))
        nl, nv, nh, vh = l @ n, v @ n, h @ n, v @ h
        d = alpha ** 2 / (math.pi * (nh * nh * (alpha ** 2 - 1) + 1) ** 2)
        vis = 1 / (2 * (nv * math.sqrt(alpha ** 2 + (1 - alpha ** 2) * nl ** 2) +
                        nl * math.sqrt(alpha ** 2 + (1 - alpha ** 2) * nv ** 2)))
        return d * vis, vh

    def test_dielectric_reduction(self):
        spec, vh = self.parts()
        fr = spec_f0(0.04, vh)
        expected = (1 - fr) * self.base[0] / math.pi + fr * spec
        got = ref.bsdf(self.base, np.array([0.0]), self.roughness, self.n, self.v, self.l)[0]
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_metal_reduction(self):
        spec, vh = self.parts()
        expected = spec_f0(self.base[0], vh) * spec
        got = ref.bsdf(self.base, np.array([1.0]), self.roughness, self.n, self.v, self.l)[0]
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_linear_mix_at_intermediate_metallic(self):
        diel = ref.bsdf(self.base, np.array([0.0]), self.roughness, self.n, self.v, self.l)[0]
        metal = ref.bsdf(self.base, np.array([1.0]), self.roughness, self.n, self.v, self.l)[0]
        for m in (0.25, 0.5, 0.75):
            got = ref.bsdf(self.base, np.array([m]), self.roughness, self.n, self.v, self.l)[0]
            np.testing.assert_allclose(got, (1 - m) * diel + m * metal, rtol=1e-12)

    def test_engine_blend_differs_at_intermediate_metallic_only(self):
        blend = ref.CONTROLS["engine-f0-blend"]
        for m, same in ((0.0, True), (1.0, True), (0.5, False)):
            a = ref.bsdf(self.base, np.array([m]), self.roughness, self.n, self.v, self.l)
            b = ref.bsdf(self.base, np.array([m]), self.roughness, self.n, self.v, self.l, blend)
            self.assertEqual(bool(np.allclose(a, b, rtol=1e-9)), same, f"metallic {m}")

    def test_below_horizon_is_black(self):
        l = np.array([unit(0.0, 0.8, -0.2)])
        self.assertTrue(np.all(ref.bsdf(self.base, np.array([0.0]), self.roughness, self.n, self.v, l) == 0))


class EnvironmentQuadrature(unittest.TestCase):
    def test_lambert_white_returns_albedo_times_radiance(self):
        # Quadrature of (1/pi) cos over the hemisphere is exactly 1 for any view angle.
        for mu in (1.0, 0.7, 0.3, 0.05):
            self.assertAlmostEqual(ref._diffuse_integral(mu, lambda vh: np.ones_like(vh)), 1.0, places=3)
        table = ref.environment_table(0.5)
        np.testing.assert_allclose(table["Cu"], 1.0, atol=1e-3)
        # White Lambert dielectric (diffuse only, no specular, no Fresnel weighting): L_o = albedo * L_env.
        white = np.ones((1, 3)) * 0.6
        lo = ref.environment_radiance(white, np.array([0.0]), np.array([1.0]), np.array([0.8]), 2.0,
                                      ref.CONTROLS["diffuse-unweighted"])
        # unweighted diffuse + specular: strictly more than albedo * L, by the specular albedo only (<= 0.4 L)
        self.assertGreater(float(lo[0, 0]), 0.6 * 2.0)
        self.assertLess(float(lo[0, 0]), 0.6 * 2.0 + 0.4 * 2.0)

    def independent_directional_albedo(self, roughness, mu, samples=4_000_000, seed=7):
        """E(mu) = integral of D Vis cos over wi by cosine-weighted Monte Carlo (wi space)."""
        rng = np.random.default_rng(seed)
        alpha = max(roughness ** 2, 0.002)
        u1, u2 = rng.random(samples), rng.random(samples)
        r, phi = np.sqrt(u1), 2 * math.pi * u2
        wi = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)], axis=1)
        wo = np.array([math.sqrt(1 - mu * mu), 0, mu])
        h = wi + wo
        h /= np.linalg.norm(h, axis=1, keepdims=True)
        a2 = alpha * alpha
        d = a2 / (math.pi * (h[:, 2] ** 2 * (a2 - 1) + 1) ** 2)
        vis = 1 / (2 * (mu * np.sqrt(a2 + (1 - a2) * wi[:, 2] ** 2) + wi[:, 2] * math.sqrt(a2 + (1 - a2) * mu * mu)))
        # pdf = cos / pi, integrand f cos -> estimate = mean(f * pi)
        return float(np.mean(d * vis * math.pi))

    def test_ggx_directional_albedo_matches_monte_carlo(self):
        for roughness, mu in ((0.5, 0.9), (0.5, 0.4), (0.8, 0.25), (1.0, 0.6), (0.35, 0.8)):
            table = ref.environment_table(roughness)
            mus = np.array([mu])
            albedo = float(np.interp(mus, table["mu"], table["A"] + table["B"])[0])
            estimate = self.independent_directional_albedo(roughness, mu)
            self.assertLess(abs(albedo - estimate) / estimate, 0.015, f"roughness {roughness} mu {mu}: {albedo} vs {estimate}")

    def test_white_furnace_metal_and_smooth_dielectric(self):
        white = np.ones((1, 3))
        for roughness in (0.1, 0.5, 1.0):
            for mu in (1.0, 0.5, 0.1):
                out = ref.environment_radiance(white, np.array([1.0]), np.array([roughness]), np.array([mu]), 1.0)
                self.assertLessEqual(float(out.max()), 1.0 + 1e-3, (roughness, mu))  # a white conductor never gains
            out = ref.environment_radiance(white, np.array([0.0]), np.array([roughness]), np.array([1.0]), 1.0)
            self.assertLessEqual(float(out.max()), 1.0 + 1e-3, roughness)  # head-on dielectric conserves energy
        smooth_metal = ref.environment_radiance(white, np.array([1.0]), np.array([0.1]), np.array([1.0]), 1.0)
        self.assertGreater(float(smooth_metal.min()), 0.97)
        gain = ref.environment_radiance(white, np.array([0.0]), np.array([0.2]), np.array([1.0]), 1.0,
                                        ref.CONTROLS["diffuse-unweighted"])
        self.assertGreater(float(gain.max()), 1.005)

    def test_literal_appendix_b_dielectric_gains_energy_at_grazing(self):
        # The specification itself says the fresnel_mix coupling "breaks energy conservation": diffuse is
        # weighted by 1 - F(V.H) of the *incident* direction while the specular lobe carries F -> 1 at a
        # grazing view.  The reference reports the literal model; this documents the property.
        white = np.ones((1, 3))
        out = ref.environment_radiance(white, np.array([0.0]), np.array([0.1]), np.array([0.2]), 1.0)
        self.assertGreater(float(out.max()), 1.1)


class SunDisk(unittest.TestCase):
    def test_spiral_covers_the_disk_uniformly(self):
        l = unit(0.2, 0.5, 0.8)
        radius = 0.05
        samples = ref.sun_disk_samples(l, radius, 128)
        angles = np.arccos(np.clip(samples @ l, -1, 1))
        self.assertTrue(np.all(angles <= radius + 1e-12))
        # Uniform on the disk: mean of cos(angle) = (1 + cos r) / 2 over solid angle.
        self.assertAlmostEqual(float(np.mean(samples @ l)), (1 + math.cos(radius)) / 2, places=4)
        np.testing.assert_allclose(np.linalg.norm(samples, axis=1), 1.0, atol=1e-12)

    def test_tiny_radius_reduces_to_a_point_light(self):
        n = np.array([unit(0.1, 0.2, 1.0), unit(-0.3, 0.1, 1.0)])
        v = np.array([unit(0.0, 0.0, 1.0), unit(0.1, -0.1, 1.0)])
        toward = unit(-0.4, 0.3, 0.9)
        base = np.array([[0.8, 0.35, 0.15]] * 2)
        metallic = np.array([0.0, 0.5])
        for roughness in (0.1, 0.5, 1.0):
            rough = np.array([roughness] * 2)
            irradiance = np.array([3.0, 3.0, 3.0])
            got, _ = ref.sun_radiance(base, metallic, rough, n, v, toward, irradiance, 1e-7)
            expected = ref.bsdf(base, metallic, rough, n, v, np.broadcast_to(toward, n.shape)) * \
                (n @ toward)[:, None] * irradiance
            np.testing.assert_allclose(got, expected, rtol=1e-6)

    def test_narrow_rule_and_counts(self):
        # Mirror geometry (R = L) with alpha < 2 r -> 128 samples; wide lobe -> point evaluation.
        n = np.array([[0.0, 0.0, 1.0]] * 2)
        toward = unit(0.0, 0.3, 1.0)
        v = np.array([unit(0.0, -0.3, 1.0)] * 2)
        count, narrow = ref.sun_quadrature(n, v, toward, np.array([0.04, 0.7]), 0.00465)
        self.assertEqual(list(count), [128, 1])  # alpha 0.002 < 2 r vs 0.49
        self.assertEqual(list(narrow), [True, False])
        count, _ = ref.sun_quadrature(n, v, toward, np.array([0.2, 0.2]), 0.00465)
        self.assertEqual(list(count), [32, 32])  # alpha 0.04: narrow but not < 2 r = 0.0093

    def test_disk_integral_converges_to_fine_quadrature(self):
        # Mirror-aligned smooth surface: the 128-sample rule must match a 20000-sample spiral.
        n = np.array([[0.0, 0.0, 1.0]])
        toward = unit(0.0, 0.3, 1.0)
        v = np.array([unit(0.0, -0.3, 1.0)])
        base = np.array([[1.0, 1.0, 1.0]])
        radius = 0.00465
        got, _ = ref.sun_radiance(base, np.array([1.0]), np.array([0.1]), n, v, toward, np.ones(3), radius)
        samples = ref.sun_disk_samples(toward, radius, 20000)
        total = np.zeros(3)
        for s in samples:
            total += ref.bsdf(base, np.array([1.0]), np.array([0.1]), n, v, s[None, :])[0] * max(0.0, s[2])
        total /= len(samples)
        np.testing.assert_allclose(got[0], total, rtol=0.02)


@unittest.skipUnless(HAVE_MITSUBA, "mitsuba not installed")
class RayCastedSnapshot(unittest.TestCase):
    @staticmethod
    def write_sphere_ply(path, radius=0.4, slices=48, stacks=24):
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
                faces += [(a, b, a + 1), (a + 1, b, b + 1)]
        with open(path, "wb") as f:
            f.write(("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
                     "property float z\nproperty float nx\nproperty float ny\nproperty float nz\nproperty float s\n"
                     "property float t\nelement face %d\nproperty list uchar uint vertex_indices\nend_header\n"
                     % (len(verts), len(faces))).encode())
            for v in verts:
                f.write(struct.pack("<8f", *v))
            for tri in faces:
                f.write(struct.pack("<B3I", 3, *tri))

    def snapshot(self, directory, sky, sun, metallic, roughness, base):
        root = pathlib.Path(directory)
        self.write_sphere_ply(root / "mesh_0.ply")
        world = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  # column-major identity
        data = {"schema": 1, "state": "TEST", "linear": True, "units": "metres", "frame": 0, "revisions": [0, 0, 0],
                "camera": {"position": [0, 0, 3], "direction": [0, 0, -1], "up": [0, 1, 0],
                           "fov_y_radians": math.radians(60), "near": 0.05, "far": 0, "width": 256, "height": 144,
                           "jitter_pixels": [0, 0], "projection": "perspective_reverse_z_infinite"},
                "sky": [sky] * 3, "sun_angular_radius": 0.00465, "textures": [],
                "materials": [{"base": [*base, 1], "emissive": [0, 0, 0], "metallic": metallic, "roughness": roughness,
                               "normal_scale": 1, "occlusion_strength": 1, "alpha_cutoff": 0, "flags": 0,
                               "textures": [ref.INVALID] * 5}],
                "meshes": ["mesh_0.ply"],
                "instances": [{"slot": 0, "generation": 1, "flags": 7, "mesh": 0, "material": 0, "world": world}],
                "lights": [], "sampled_lights": []}
        if sun:
            data["lights"].append({"type": 0, "position": [0, 0, 0], "direction": list(-sun[0]),
                                   "color": [1, 1, 1], "intensity": sun[1], "range": 0, "inner": 0, "outer": 0})
        (root / "scene.json").write_text(json.dumps(data))
        return root

    def test_furnace_sphere_and_centre_pixel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.snapshot(tmp, 1.0, None, 1.0, 0.5, [1.0, 1.0, 1.0])
            from f12_reference import snapshot as load
            root, data = load(root)
            image, masks, spheres, terms = ref.render_frame(root, data, threads=2)
            self.assertEqual(terms, ["environment-quadrature"])
            self.assertGreater(int(masks["roi"].sum()), 500)
            self.assertLessEqual(float(image[masks["roi"]].max()), 1.0 + 1e-3)
            # Background shows the environment itself.
            self.assertAlmostEqual(float(image[0, 0, 0]), 1.0, places=6)
            # The centre pixel looks along -z at a normal very near +z: mu ~ 1.
            table = ref.environment_table(0.5)
            expected = float(np.interp(1.0, table["mu"], table["A"] + table["B"]))  # white metal, f = 1
            self.assertAlmostEqual(float(image[72, 128, 0]), expected, delta=2e-3)

    def test_sun_lit_hemisphere_and_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            toward = unit(-0.8, 0.35, -0.3)
            root = self.snapshot(tmp, 0.0, (toward, 3.0), 0.0, 0.1, [0.8, 0.35, 0.15])
            from f12_reference import snapshot as load
            root, data = load(root)
            image, masks, spheres, terms = ref.render_frame(root, data, threads=2)
            self.assertEqual(terms, ["sun-disk"])
            lit = masks["roi"] & masks["lit"]
            self.assertGreater(int(lit.sum()), 100)
            unlit = masks["roi"] & ~masks["lit"]
            # No light reaches N.L <= 0 (shading normal of the facets): essentially black.
            self.assertLess(float(image[unlit].max()), 1e-6)
            # A lit diffuse-dominated pixel: rough dielectric ~ albedo/pi * E * cos (+ small specular).
            self.assertGreater(float(image[lit].max()), 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
