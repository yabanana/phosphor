"""Portable checks of the external cooked-probe writer (F13.5; no GPU)."""
import pathlib
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import probe_pfm_author as author  # noqa: E402


class FaceTable(unittest.TestCase):
    def test_face_centres_are_the_major_axes(self):
        axes = {"px": (1, 0, 0), "nx": (-1, 0, 0), "py": (0, 1, 0), "ny": (0, -1, 0), "pz": (0, 0, 1), "nz": (0, 0, -1)}
        for face, axis in axes.items():
            self.assertEqual(tuple(round(c, 9) for c in author.face_direction(face, 0.0, 0.0)), axis)

    def test_shared_edges_meet(self):
        # Every pair of adjacent faces shares its edge directions exactly.
        pairs = [(("px", -1, None), ("pz", 1, None)), (("nx", 1, None), ("pz", -1, None)),
                 (("px", 1, None), ("nz", -1, None)), (("nx", -1, None), ("nz", 1, None))]
        for (fa, sa, _), (fb, sb, _) in pairs:
            for t in (-0.75, 0.0, 0.5):
                a, b = author.face_direction(fa, sa, t), author.face_direction(fb, sb, t)
                for x, y in zip(a, b):
                    self.assertAlmostEqual(x, y, places=12)
        # Side faces: t = -1 (top row) points up, the +Y face's t = +1 edge faces +Z.
        for face in ("px", "nx", "pz", "nz"):
            self.assertGreater(author.face_direction(face, 0.0, -1.0)[1], 0)
        self.assertGreater(author.face_direction("py", 0.0, 1.0)[2], 0)
        self.assertLess(author.face_direction("ny", 0.0, 1.0)[2], 0)


class Files(unittest.TestCase):
    def test_pfm_rows_are_bottom_up_and_gradient_is_vertical(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp) / "probe"
            author.main(["--out", str(out)])
            data = (out / "px.pfm").read_bytes()
            head = b"PF\n64 64\n-1.0\n"
            self.assertTrue(data.startswith(head))
            body = data[len(head):]
            self.assertEqual(len(body), 64 * 64 * 12)
            first_row_green = struct.unpack_from("<3f", body, 32 * 12)[1]   # file row 0 = bottom
            last_row_green = struct.unpack_from("<3f", body, (63 * 64 + 32) * 12)[1]
            self.assertGreater(last_row_green, first_row_green)  # the top of a side face is brighter
            self.assertEqual(sorted(p.name for p in out.glob("*.pfm")), sorted(f + ".pfm" for f in author.FACES))
            with self.assertRaises(FileExistsError):
                author.main(["--out", str(out)])

    def test_top_down_row_order_is_the_flipped_negative(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, bad = pathlib.Path(tmp) / "good", pathlib.Path(tmp) / "bad"
            author.main(["--out", str(good)])
            author.main(["--out", str(bad), "--row-order", "top-down"])
            head = len(b"PF\n64 64\n-1.0\n")
            a, b = (good / "px.pfm").read_bytes()[head:], (bad / "px.pfm").read_bytes()[head:]
            row = 64 * 12
            self.assertEqual(a[:row], b[63 * row:])  # same rows, opposite order
            self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
