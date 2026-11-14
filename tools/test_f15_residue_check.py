#!/usr/bin/env python3
"""Unit tests of tools/f15_fixture_scene.py and tools/f15_residue_check.py (no GPU): glTF writer, winding, regions, ring detection."""
import json
import math
import pathlib
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import f15_fixture_scene as fx  # noqa: E402
import f15_residue_check as check  # noqa: E402


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def make_png(width, height, pixel):
    raw = b"".join(b"\x00" + b"".join(bytes(pixel(x, y)) for x in range(width)) for y in range(height))
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


class FixtureScene(unittest.TestCase):
    def test_sphere_is_counter_clockwise_seen_from_outside(self):
        scene = fx.Scene()
        scene.sphere((0, 0, 0), 1.0, 16, scene.material("m"))
        positions, normals, indices, _ = scene.primitives[0]
        for t in range(0, len(indices), 3):
            a, b, c = (positions[i] for i in indices[t:t + 3])
            face = cross(sub(b, a), sub(c, a))
            if dot(face, face) < 1e-18:
                continue  # the degenerate triangles at the poles
            centre = tuple((a[k] + b[k] + c[k]) / 3 for k in range(3))
            self.assertGreater(dot(face, centre), 0.0)

    def test_quad_normal_follows_the_winding(self):
        scene = fx.Scene()
        m = scene.material("m")
        scene.quad([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], m)
        scene.quad([(1, 0, 0), (0, 0, 0), (0, 1, 0), (1, 1, 0)], m)
        self.assertEqual(scene.primitives[0][1][0], (0.0, 0.0, 1.0))
        self.assertEqual(scene.primitives[1][1][0], (0.0, 0.0, -1.0))

    def test_gltf_is_consistent(self):
        scene = fx.Scene()
        glass = scene.material("glass", extensions={"KHR_materials_transmission": {"transmissionFactor": 1.0}})
        scene.quad([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)], glass)
        scene.sphere((0, 0, 1), 0.5, 8, scene.material("sphere", double_sided=True))
        with tempfile.TemporaryDirectory() as folder:
            path = scene.write(folder)
            document = json.loads(path.read_text())
            blob = (pathlib.Path(folder) / "scene.bin").read_bytes()
        self.assertEqual(document["buffers"][0]["byteLength"], len(blob))
        self.assertEqual(document["extensionsUsed"], ["KHR_materials_transmission"])
        self.assertTrue(document["materials"][1]["doubleSided"])
        self.assertNotIn("doubleSided", document["materials"][0])
        for view in document["bufferViews"]:
            self.assertEqual(view["byteOffset"] % 4, 0)
            self.assertLessEqual(view["byteOffset"] + view["byteLength"], len(blob))


class Cases(unittest.TestCase):
    def test_mirror_regions_lie_on_the_wall_the_floor_shows(self):
        scene, = (check.mirror_scene(tempfile.mkdtemp())[0],)
        boxes = check.mirror_boxes(scene)
        for name, (x0, y0, x1, y1) in boxes.items():
            self.assertTrue(0 <= x0 < x1 <= fx.WIDTH and 0 <= y0 < y1 <= fx.HEIGHT, name)
        # the mirror image sits below the direct wall region on the screen
        self.assertGreater(boxes["mirror_shadow"][1], boxes["direct_shadow"][3])
        shadow = (-0.307, -0.136, 0.193, 0.264)  # sun shadow of the panel (height 0.3) on the wall
        x0, y0, x1, y1 = check.MIRROR_REGIONS["shadow"]
        self.assertTrue(shadow[0] <= x0 and x1 <= shadow[2] and shadow[1] <= y0 and y1 <= shadow[3])
        self.assertLessEqual(check.MIRROR_REGIONS["shadow"][3], 0.11)  # the floor mirror shows the wall up to y = 0.11

    def test_backface_scene_has_a_plane_facing_away_and_a_pane_over_it(self):
        scene, _ = check.backface_scene(tempfile.mkdtemp())
        wall, plane, floor, glass = (p[1][0] for p in scene.primitives)
        self.assertEqual(wall, (0.0, 0.0, 1.0))
        self.assertEqual(plane, (0.0, 0.0, -1.0))  # its front faces away from the camera: the back faces the viewer
        self.assertEqual(floor, (0.0, 1.0, 0.0))
        self.assertFalse(scene.materials[1].get("doubleSided", False))
        glass_x = (0.45, 0.85)
        x0, _, x1, _ = check.BACKFACE_REGIONS["glass"]
        self.assertTrue(glass_x[0] < x0 and x1 < glass_x[1])
        for box in check.backface_boxes(scene).values():
            self.assertTrue(0 <= box[0] < box[2] <= fx.WIDTH and 0 <= box[1] < box[3] <= fx.HEIGHT)

    def test_ring_pixels_are_the_silhouette(self):
        size, centre, radius = 40, 20, 12
        def pixel(x, y):
            return (200, 200, 200) if (x - centre) ** 2 + (y - centre) ** 2 <= radius * radius else (10, 10, 10)
        with tempfile.TemporaryDirectory() as folder:
            png = pathlib.Path(folder) / "disc.png"
            png.write_bytes(make_png(size, size, pixel))
            image = fx.read_png_gray(png)
        ring = check.ring_pixels(image)
        self.assertGreater(len(ring), 30)
        for x, y in ring:
            self.assertLessEqual(math.hypot(x - centre, y - centre), radius + 0.01)
            self.assertGreater(math.hypot(x - centre, y - centre), radius - 1.6)

    def test_grazing_sphere_triangles_have_back_facing_shading_normals_at_the_horizon(self):
        """The cause of C22-D3 from the fixture itself: some visible pixel has shading N.V < 0 while the geometric N.V > 0."""
        scene, _ = check.grazing_scene(tempfile.mkdtemp())
        bmin, bmax = scene.bounds()
        centre, distance = fx.frame_rig(bmin, bmax)
        camera = (centre[0], centre[1], centre[2] + distance)
        found = False
        positions, normals, indices, _ = scene.primitives[0]
        for t in range(0, len(indices), 3):
            ids = indices[t:t + 3]
            for corner in range(3):  # a vertex whose analytic normal faces away from the camera inside a triangle whose plane faces it
                v = tuple(camera[k] - positions[ids[corner]][k] for k in range(3))
                n = normals[ids[corner]]
                a, b, c = (positions[i] for i in ids)
                face = cross(sub(b, a), sub(c, a))
                if dot(n, v) < 0 < dot(face, tuple(camera[k] - a[k] for k in range(3))):
                    found = True
        self.assertTrue(found)

    def test_furnace_scene_is_smooth_solid_glass_with_an_opaque_twin_of_the_same_framing(self):
        glass, _ = check.furnace_scene(tempfile.mkdtemp())
        twin, _ = check.furnace_scene(tempfile.mkdtemp(), glass=False)
        self.assertEqual(len(glass.primitives), len(check.IORS))
        self.assertEqual(glass.bounds(), twin.bounds())  # the corpus rig frames both identically: the twin's silhouette is the glass's
        for material, ior in zip(glass.materials, check.IORS):
            extensions = material["extensions"]
            self.assertEqual(extensions["KHR_materials_ior"]["ior"], ior)
            self.assertEqual(extensions["KHR_materials_transmission"]["transmissionFactor"], 1.0)
            self.assertGreater(extensions["KHR_materials_volume"]["thicknessFactor"], 0.0)  # a solid, not a thin-walled pane
            self.assertEqual(material["pbrMetallicRoughness"]["baseColorFactor"], [1.0, 1.0, 1.0, 1])
            self.assertEqual(material["pbrMetallicRoughness"]["roughnessFactor"], 0.0)
            self.assertNotIn("attenuationDistance", extensions["KHR_materials_volume"])  # nothing absorbs: a furnace
        for material in twin.materials:
            self.assertNotIn("extensions", material)
        for centre in range(len(check.IORS)):
            x = (centre - 0.5 * (len(check.IORS) - 1)) * check.GLASS_SPACING
            self.assertTrue(any(abs(p[0] - (x + 1.0)) < 1e-6 for p in glass.primitives[centre][0]))

    def test_absorbing_control_only_adds_a_medium(self):
        plain, _ = check.furnace_scene(tempfile.mkdtemp())
        absorbing, _ = check.furnace_scene(tempfile.mkdtemp(), absorbing=True)
        self.assertEqual(plain.bounds(), absorbing.bounds())
        for a, b in zip(plain.materials, absorbing.materials):
            volume = b["extensions"]["KHR_materials_volume"]
            self.assertEqual(volume["attenuationColor"], [0.5, 0.5, 0.5])
            self.assertEqual({k: v for k, v in volume.items() if k not in check.ABSORBING}, a["extensions"]["KHR_materials_volume"])

    def test_furnace_gate_demands_no_dark_pixel_and_a_flat_mean(self):
        good = {"dark_disc_pixels": 0, "low_fraction": 0.0001, "mean_ratio": 1.0, "ring_mean_ratio": 0.99}
        self.assertTrue(check.furnace_gate(good))
        self.assertFalse(check.furnace_gate({**good, "dark_disc_pixels": 1}))
        self.assertFalse(check.furnace_gate({**good, "low_fraction": 0.05}))
        self.assertFalse(check.furnace_gate({**good, "mean_ratio": 0.9}))
        self.assertFalse(check.furnace_gate({**good, "ring_mean_ratio": 1.1}))


if __name__ == "__main__":
    unittest.main()
