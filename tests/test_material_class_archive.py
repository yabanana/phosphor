#!/usr/bin/env python3
"""tools/material_class_archive.py: the targeted archive update keeps history untouched, adds exactly the plan and never duplicates (F15.2)."""
import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import material_class_archive as mca  # noqa: E402

BIN = 0x1000


def constant(index, value, kind="ConstantUInt"):
    return {"id": {"data": index}, "id_type": "FunctionConstantIndex", "value": {"data": value}, "value_type": kind}


def corpus(library_label, specialisations, kernel="visibility_resolve", path="/somewhere/phosphor.metallib"):
    """A tiny Metal serializer corpus: one library function `kernel` and one compute pipeline per specialisation."""
    functions = {"library_function_descriptors": [{"label": "LIB-" + library_label, "library": library_label, "name": kernel}],
                 "specialized_function_descriptors": []}
    pipelines = []
    for constants in specialisations:
        label = "SPEC-" + "-".join(f"{i}_{v}" for i, v in constants)
        functions["specialized_function_descriptors"].append(
            {"constant_values": [constant(i, v) for i, v in constants], "function_descriptor": "fnd:LIB-" + library_label, "label": label})
        pipelines.append({"compute_function_descriptor": "fnd:" + label, "required_threads_per_threadgroup": [0, 0, 0]})
    return {"function_descriptors": functions, "generator": "MetalFramework",
            "libraries": [{"label": library_label, "path": path}],
            "pipeline_descriptors": {"compute_pipeline_descriptors": pipelines, "render_pipeline_descriptors": []}, "version": {"major": 1}}


def plan_for(entries, budget=64):
    pipelines = [{"kernel": "visibility_resolve", "mask": m, "bin_mode": b, "bin_value": v, "sss": False,
                  "constants": [{"index": 22, "value": m}, {"index": 24, "value": v}]} for m, b, v in entries]
    return {"schema": "phosphor.material-class-plan", "version": 1, "budgets": {"archived_permutations": budget, "active_classes": 16},
            "permutations": sorted({m for m, _, _ in entries} | {1}), "pipelines": pipelines}


class MaterialClassArchive(unittest.TestCase):
    def setUp(self):
        self.archive = corpus("OLD", [[(22, 1)], [(22, 1)]], path=mca.PLACEHOLDER)
        # History keeps the same pipeline twice with different records (as the real archive does): not merged away.
        self.archive["pipeline_descriptors"]["compute_pipeline_descriptors"][1]["required_threads_per_threadgroup"] = [1, 1, 1]  # one permutation, a duplicated identity from history
        self.plan = plan_for([(3, 1, BIN | 3), (511, 2, 0x2000)])
        self.harvest = corpus("NEW", [[(22, 3), (24, BIN | 3)], [(22, 511), (24, 0x2000)], [(22, 1)]])

    def test_update_adds_exactly_the_plan_and_keeps_history(self):
        before = copy.deepcopy(self.archive)
        merged = mca.update(self.plan, self.archive, [copy.deepcopy(self.harvest)], 24)
        old = merged["pipeline_descriptors"]["compute_pipeline_descriptors"]
        self.assertEqual(len(old), len(before["pipeline_descriptors"]["compute_pipeline_descriptors"]) + 2)
        self.assertEqual(old[:2], before["pipeline_descriptors"]["compute_pipeline_descriptors"])  # history is a prefix
        problems, report = mca.check(self.plan, merged)
        self.assertEqual(problems, [])
        self.assertEqual(report["plan_pipelines"], 2)
        self.assertEqual(report["archived_permutations"], 3)  # 1, 3, 511

    def test_history_duplicates_are_not_added_to(self):
        merged = mca.update(self.plan, self.archive, [copy.deepcopy(self.harvest)], 24)
        self.assertEqual(mca.duplicate_identities(merged), mca.duplicate_identities(self.archive))

    def test_a_second_update_is_a_no_op(self):
        once = mca.update(self.plan, self.archive, [copy.deepcopy(self.harvest)], 24)
        twice = mca.update(self.plan, once, [copy.deepcopy(self.harvest)], 24)
        self.assertEqual(json.dumps(once, sort_keys=True), json.dumps(twice, sort_keys=True))

    def test_two_harvests_of_one_class_are_kept_once(self):
        merged = mca.update(self.plan, self.archive, [copy.deepcopy(self.harvest), copy.deepcopy(self.harvest)], 24)
        self.assertEqual(mca.check(self.plan, merged)[0], [])

    def test_a_harvest_outside_the_plan_is_rejected(self):
        stray = corpus("NEW", [[(22, 5), (24, BIN | 5)]])
        with self.assertRaises(ValueError):
            mca.update(self.plan, self.archive, [stray], 24)

    def test_check_reports_missing_and_budget_and_stray_permutations(self):
        problems, _ = mca.check(self.plan, self.archive)
        self.assertEqual(len([p for p in problems if "occurs 0 times" in p]), 2)
        tight = plan_for([(3, 1, BIN | 3), (511, 2, 0x2000)], budget=2)
        full = mca.update(self.plan, self.archive, [copy.deepcopy(self.harvest)], 24)
        problems, _ = mca.check(tight, full)
        self.assertTrue(any("budget" in p for p in problems))
        narrow = dict(self.plan, permutations=[1])
        problems, _ = mca.check(narrow, full)
        self.assertTrue(any("outside the plan" in p for p in problems))

    def test_the_committed_archive_holds_the_generated_plan(self):
        import os
        plan = pathlib.Path(os.environ.get("PHOSPHOR_CLASS_PLAN", REPO / "build/release/generated/pipeline/material_class_plan.json"))
        if not plan.is_file():
            self.skipTest("no generated plan (configure the build first)")
        result = subprocess.run([sys.executable, str(REPO / "tools/material_class_archive.py"), "check", "--plan", str(plan),
                                 "--archive", str(REPO / "shaders/pipelines.mtl4-json")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_the_writer_is_a_fixed_point_of_the_committed_archive(self):
        text = (REPO / "shaders/pipelines.mtl4-json").read_text()
        self.assertEqual(json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n", text)


if __name__ == "__main__":
    unittest.main()
