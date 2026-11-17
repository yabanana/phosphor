#!/usr/bin/env python3
"""Unit tests of tools/f15_corpus.py and tools/f15_corpus_page.py (stdlib only; no GPU, no corpus needed: a fake model is built in a temp folder)."""
import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import f15_corpus as c  # noqa: E402
import f15_corpus_page as page  # noqa: E402

LOG = """[INFO] [2026-11-14 10:00:00] x.cpp:1: hello
[WARN] [2026-11-14 10:00:00] /a/src/scene/gltf_loader.cpp:433: glTF BLEND material rendered opaque until F16 transparency is implemented
2026-11-14 11:01:29.137 phosphor[29722:8568902] Render Command Encoder Set Depth Stencil State Validation.
Redundant call to setDepthStencilState.
2026-11-14 11:01:29.137 phosphor[29722:8568902] Render Command Encoder Set Viewport Validation
Redundant call to setViewport.
PIPELINES requests 148 | archive hits 145, misses 3, unavailable 0 (miss rate 2.0%) | compiler calls 3 (1.0 ms, max 1.0 ms) | render-thread compiles 2 (1.0 ms) | fallbacks 0 (draws 0) | failures 0 | reloads 0 (failed 0)
EXIT 0
"""
REFUSED = ("[ERROR] [2026-11-14 10:00:00] /a/src/scene/gltf_loader.cpp:152: glTF /m.gltf requires extension(s) that cannot be honoured: "
           "KHR_materials_unlit (not understood by this loader); refusing to load\n[ERROR] [2026-11-14 10:00:00] /a/src/main.cpp:33: Fatal: Required scene failed to load: /m.gltf\nEXIT 1\n")


class LogTest(unittest.TestCase):
    def test_baseline_notices_and_loader_warnings_are_classified(self):
        scan = c.scan_log(LOG)
        self.assertEqual(scan["other_messages"], [])
        self.assertEqual(scan["validation_failures"], [])
        self.assertEqual(scan["known_notices"]["Redundant call to setViewport."], 1)
        self.assertEqual(len(scan["loader_messages"]), 1)
        self.assertIn("BLEND", scan["loader_messages"][0])

    def test_an_unknown_error_or_validation_failure_is_not_swallowed(self):
        scan = c.scan_log(LOG + "[ERROR] [t] /a/src/platform/metal/x.cpp:9: boom\nfailed assertion `Draw Errors Validation'\n")
        self.assertTrue(any("boom" in line for line in scan["other_messages"]))
        self.assertTrue(scan["validation_failures"])

    def test_pipelines_summary(self):
        found = c.PIPELINES_LINE.findall(LOG)[-1]
        self.assertEqual((found[1], found[2], found[5], found[6]), ("145", "3", "0", "0"))
        self.assertEqual(c.fallback_draws(LOG), 0)
        self.assertIsNone(c.fallback_draws("nothing"))

    def test_refusal_reason(self):
        reason = c.refusal_reason(REFUSED)
        self.assertIn("KHR_materials_unlit", reason)
        self.assertTrue(reason.endswith("refusing to load"))
        self.assertIsNone(c.refusal_reason(LOG))

    def test_gpu_guard_names_other_jobs(self):
        ps = "100 /bin/zsh\n200 /x/build/release/phosphor\n300 /x/soc_bench\n400 /x/f9_spike\n500 /x/phosphor-helper\n"
        self.assertEqual(c.running_gpu_processes(ps), ["200:phosphor", "300:soc_bench", "400:f9_spike"])

    def test_rig_line(self):
        line = "CORPUS_RIG: 27 instances, bounds min (-7.3578 -6.2500 -0.0600) max (3.1000 6.8545 1.0500), camera distance 13.2657"
        match = c.RIG_LINE.search(line)
        self.assertEqual(match.group(1), "27")
        self.assertEqual(match.group(4), "13.2657")


class CommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.corpus = pathlib.Path(self.tmp.name)
        folder = self.corpus / "ClearCoatTest" / "glTF"
        folder.mkdir(parents=True)
        (folder / "ClearCoatTest.gltf").write_text(json.dumps({
            "asset": {"version": "2.0"}, "extensionsUsed": ["KHR_materials_clearcoat", "KHR_texture_transform"], "extensionsRequired": ["KHR_texture_transform"],
            "images": [{"uri": "a.png"}, {"uri": "b.JPG"}],
            "materials": [{"extensions": {"KHR_materials_clearcoat": {"clearcoatTexture": {"index": 0, "extensions": {"KHR_texture_transform": {"scale": [2, 2]}}}}}},
                          {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0, "extensions": {"KHR_texture_transform": {}}}}}]}))

    def test_capture_command_declares_rig_and_guards_fallbacks(self):
        command = c.capture_command(pathlib.Path("/b/phosphor"), self.corpus, "ClearCoatTest", pathlib.Path("/o"))
        text = " ".join(command)
        for flag in ("--corpus-rig 0,0", "--bench 4", "--pipeline-sync", "--tonemap aces", "--rt on", "--reflections rt", "--warmup 48", "--frames 48",
                     "--capture /o/engine.png", "--resolution 1024x768"):
            self.assertIn(flag, text)
        self.assertNotIn("--no-pipeline-archive", text)
        validation = c.capture_command(pathlib.Path("/b/phosphor"), self.corpus, "ClearCoatTest", pathlib.Path("/o"), validation=True)
        self.assertIn("--no-pipeline-archive", validation)

    def test_static_audit_reads_the_gltf_only(self):
        audit = c.audit_model(self.corpus, "ClearCoatTest")
        self.assertEqual(audit["extensions_required"], ["KHR_texture_transform"])
        self.assertEqual(audit["material_extensions"], {"KHR_materials_clearcoat": 1})
        self.assertEqual(audit["texture_transform_uses"], 2)
        self.assertEqual(audit["images"], ["png", "jpg"])
        self.assertFalse(c.audit_model(self.corpus, "ToyCar")["present"])

    def test_missing_corpus_exits_2(self):
        self.assertEqual(c.main(["--static", "--corpus", str(self.corpus / "absent"), "--out", str(self.corpus / "out")]), 2)

    def test_every_model_has_a_view_and_a_statement(self):
        for name, model in c.MODELS.items():
            self.assertEqual(len(model["view"]), 2, name)
            self.assertTrue(model["shows"], name)
        manifest = json.loads((pathlib.Path(c.REPO) / "tools/khronos_corpus.json").read_text())
        self.assertEqual(set(c.MODELS), set(manifest["models"]))


class PageTest(unittest.TestCase):
    def test_validation_status_texts(self):
        self.assertEqual(page.validation_status(None), "not run")
        clean = {"status": "rendered", "known_notices": {"Redundant call to setViewport.": 96}, "other_messages": [], "validation_failures": []}
        self.assertIn("96 redundant state calls", page.validation_status(clean))
        self.assertIn("unclassified", page.validation_status({**clean, "other_messages": ["x"]}))
        self.assertIn("validation failures", page.validation_status({**clean, "validation_failures": ["y"]}))
        self.assertIn("refused", page.validation_status({"status": "refused"}))

    def test_notes_are_consistent_with_the_models(self):
        notes = json.loads((pathlib.Path(c.REPO) / "tools/f15_corpus_notes.json").read_text())
        notes.pop("_comment")
        self.assertTrue(set(notes) <= set(c.MODELS))
        for name, note in notes.items():
            for defect in note.get("defects", []):
                self.assertIn(defect["status"], {"open", "fixed", "by-design", "named-approximation"}, name)
                self.assertTrue(defect["hypothesis"], name)

    def test_page_refuses_tracked_folders(self):
        with self.assertRaises(SystemExit):
            page.main(["--run", str(c.REPO), "--corpus", str(c.REPO), "--page", str(c.REPO / "docs/page")])


if __name__ == "__main__":
    unittest.main()
