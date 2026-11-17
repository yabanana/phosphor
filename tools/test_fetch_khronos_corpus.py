#!/usr/bin/env python3
"""Unit tests of the offline `--verify` mode of tools/fetch_khronos_corpus.py (stdlib only, tiny fake corpus, no network)."""
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fetch_khronos_corpus as f  # noqa: E402

GLTF = json.dumps({"asset": {"version": "2.0"}, "extensionsUsed": ["KHR_materials_clearcoat"]}).encode()
FILES = {"glTF/Fake.gltf": GLTF, "glTF/Fake.bin": b"\x00\x01\x02\x03", "glTF/Tex.png": b"\x89PNG-fake",
         "README.md": b"readme", "LICENSE.md": b"licence", "metadata.json": b"{}", "screenshot/screenshot.jpg": b"jpg-fake"}


def entry(path, data):
    return {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def build(root, manifest_path, files=None):
    """Write the fake model folder under `root` and its manifest at `manifest_path`."""
    files = dict(FILES if files is None else files)
    for path, data in files.items():
        (root / "Fake" / path).parent.mkdir(parents=True, exist_ok=True)
        (root / "Fake" / path).write_bytes(data)
    manifest = {"commit": "abc", "models": {"Fake": {
        "tier": "A", "license": [{"license": "CC0-1.0"}],
        "files": [entry(p, d) for p, d in files.items() if p.startswith("glTF/")],
        "license_files": [entry(p, d) for p, d in files.items() if p in ("README.md", "LICENSE.md", "metadata.json")],
        "screenshots": [entry(p, d) for p, d in files.items() if p.startswith("screenshot/")]}}}
    manifest_path.write_text(json.dumps(manifest))


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = pathlib.Path(self.tmp.name)
        self.corpus = root / "corpus"
        self.corpus.mkdir()
        self.manifest = root / "manifest.json"
        build(self.corpus, self.manifest)

    def problems(self):
        return f.verify(self.corpus, self.manifest)

    def test_matching_corpus_passes(self):
        self.assertEqual(self.problems(), [])

    def test_hash_mismatch(self):
        (self.corpus / "Fake/glTF/Fake.bin").write_bytes(b"\x00\x01\x02\x04")
        self.assertTrue(any("sha256 mismatch" in p and "Fake.bin" in p for p in self.problems()))

    def test_size_mismatch(self):
        (self.corpus / "Fake/glTF/Tex.png").write_bytes(b"short")
        self.assertTrue(any("size" in p for p in self.problems()))

    def test_missing_file_and_missing_model(self):
        (self.corpus / "Fake/glTF/Tex.png").unlink()
        self.assertTrue(any("Tex.png: missing" in p for p in self.problems()))
        (self.corpus / "Fake/README.md").unlink()
        self.assertTrue(any("README.md: missing" in p for p in self.problems()))
        for path in sorted((self.corpus / "Fake").rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        (self.corpus / "Fake").rmdir()
        self.assertTrue(any("model folder missing" in p for p in self.problems()))

    def test_extra_files(self):
        (self.corpus / "Fake/glTF/Fake.webp").write_bytes(b"x")
        (self.corpus / "Other").mkdir()
        (self.corpus / "Other/note.txt").write_text("x")
        problems = self.problems()
        self.assertTrue(any("Fake.webp: extra" in p for p in problems))
        self.assertTrue(any("Other/note.txt: extra" in p for p in problems))

    def test_incomplete_licence_record(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["models"]["Fake"]["license_files"] = manifest["models"]["Fake"]["license_files"][:1]
        self.manifest.write_text(json.dumps(manifest))
        self.assertTrue(any("licence record incomplete" in p for p in self.problems()))
        manifest["models"]["Fake"].pop("license")
        self.manifest.write_text(json.dumps(manifest))
        self.assertTrue(any("licence record incomplete" in p for p in self.problems()))

    def test_banned_file_kind_and_extensions(self):
        with tempfile.TemporaryDirectory() as other:
            root = pathlib.Path(other)
            files = dict(FILES)
            files["glTF/Tex.webp"] = b"webp"
            files["glTF/Fake.gltf"] = json.dumps({"extensionsRequired": ["KHR_draco_mesh_compression"]}).encode()
            build(root, root / "manifest.json", files)
            (root / "manifest.json").rename(root / "m.json")
            problems = f.verify(root, root / "m.json")
            self.assertTrue(any("Tex.webp: file kind not allowed" in p for p in problems))
            self.assertTrue(any("KHR_draco_mesh_compression" in p for p in problems))

    def test_command_line_exit_codes(self):
        script = str(pathlib.Path(f.__file__))
        ok = subprocess.run([sys.executable, "-I", script, "--verify", str(self.corpus), str(self.manifest)], capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stdout)
        (self.corpus / "Fake/glTF/Fake.bin").write_bytes(b"bad")
        bad = subprocess.run([sys.executable, "-I", script, "--verify", str(self.corpus), str(self.manifest)], capture_output=True, text=True)
        self.assertEqual(bad.returncode, 1)
        self.assertIn("FAIL", bad.stdout)

    def test_committed_manifest_is_well_formed(self):
        manifest = json.loads(f.MANIFEST.read_text())
        self.assertEqual(manifest["commit"], f.COMMIT)
        for name, model in manifest["models"].items():
            for record in model["files"]:
                self.assertIn(pathlib.PurePosixPath(record["path"]).suffix.lower(), f.ALLOWED_SUFFIXES, name)
            self.assertTrue(model["license"], name)
            self.assertFalse(model.get("skipped"), f"{name}: the default variant needed skipping, re-check Draco/BasisU/WebP")


if __name__ == "__main__":
    unittest.main()
