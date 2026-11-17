#!/usr/bin/env python3
"""Unit tests of tools/filter_pipeline_harvest.py (stdlib only)."""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import filter_pipeline_harvest as f  # noqa: E402


def constant(index, value):
    return {"id_type": "FunctionConstantIndex", "id": {"data": index}, "value_type": "ConstantUInt",
            "value": {"data": value}}


CORPUS = {
    "version": 1, "generator": "test", "libraries": [{"path": "@PHOSPHOR_METALLIB@"}],
    "function_descriptors": {
        "library_function_descriptors": [{"label": "LVS", "name": "vs"}, {"label": "LFS", "name": "fs"},
                                         {"label": "LCS", "name": "cs"}, {"label": "LUNUSED", "name": "x"}],
        "specialized_function_descriptors": [
            {"label": "S22", "function_descriptor": "fnd:LFS", "constant_values": [constant(21, 2), constant(22, 1)]},
            {"label": "S21", "function_descriptor": "fnd:LFS", "constant_values": [constant(21, 2)]},
            {"label": "C22", "function_descriptor": "fnd:LCS", "constant_values": [constant(22, 511)]}]},
    "pipeline_descriptors": {
        "render_pipeline_descriptors": [
            {"vertex_function_descriptor": "fnd:LVS", "fragment_function_descriptor": "fnd:S22", "label": "A"},
            {"vertex_function_descriptor": "fnd:LVS", "fragment_function_descriptor": "fnd:S21", "label": "B"}],
        "compute_pipeline_descriptors": [{"compute_function_descriptor": "fnd:C22"},
                                         {"compute_function_descriptor": "fnd:LCS"}],
        "tile_render_pipeline_descriptors": []}}


class FilterHarvest(unittest.TestCase):
    def test_keeps_only_pipelines_specialised_with_the_constant_and_their_functions(self):
        kept = f.filter_harvest(CORPUS, 22)
        self.assertEqual([d["label"] for d in kept["pipeline_descriptors"]["render_pipeline_descriptors"]], ["A"])
        self.assertEqual(kept["pipeline_descriptors"]["compute_pipeline_descriptors"],
                         [{"compute_function_descriptor": "fnd:C22"}])
        self.assertEqual(kept["pipeline_descriptors"]["tile_render_pipeline_descriptors"], [])
        functions = kept["function_descriptors"]
        self.assertEqual(sorted(d["label"] for d in functions["specialized_function_descriptors"]), ["C22", "S22"])
        self.assertEqual(sorted(d["label"] for d in functions["library_function_descriptors"]), ["LCS", "LFS", "LVS"])
        self.assertEqual(kept["libraries"], CORPUS["libraries"])

    def test_a_new_kernel_without_the_constant_is_kept_by_name(self):
        kept = f.filter_harvest(CORPUS, 22, functions_by_name=("cs",))
        self.assertEqual(kept["pipeline_descriptors"]["compute_pipeline_descriptors"],
                         [{"compute_function_descriptor": "fnd:C22"}, {"compute_function_descriptor": "fnd:LCS"}])
        self.assertEqual(sorted(d["label"] for d in kept["function_descriptors"]["library_function_descriptors"]), ["LCS", "LFS", "LVS"])

    def test_other_constant_and_unknown_references(self):
        self.assertEqual([d["label"] for d in f.filter_harvest(CORPUS, 21)["pipeline_descriptors"]
                          ["render_pipeline_descriptors"]], ["A", "B"])
        broken = {**CORPUS, "pipeline_descriptors": {"render_pipeline_descriptors": [
            {"vertex_function_descriptor": "fnd:NOPE", "fragment_function_descriptor": "fnd:S22"}]}}
        with self.assertRaises(ValueError):
            f.filter_harvest(broken, 22)


class SkipPresent(unittest.TestCase):
    def test_pipelines_present_under_other_labels_are_dropped(self):
        import copy
        relabelled = copy.deepcopy(CORPUS)  # the same pipelines after a shader edit: every label differs
        text = json.dumps(relabelled)
        for old in ("LVS", "LFS", "LCS", "LUNUSED", "S22", "S21", "C22"):
            text = text.replace('"' + old + '"', '"X' + old + '"').replace('fnd:' + old + '"', 'fnd:X' + old + '"')
        relabelled = json.loads(text)
        kept = f.filter_harvest(relabelled, 22, present=CORPUS)
        self.assertEqual(sum(len(v) for v in kept["pipeline_descriptors"].values()), 0)
        changed = copy.deepcopy(relabelled)
        changed["pipeline_descriptors"]["render_pipeline_descriptors"][0]["label"] = "A2"  # different state
        kept = f.filter_harvest(changed, 22, present=CORPUS)
        self.assertEqual(len(kept["pipeline_descriptors"]["render_pipeline_descriptors"]), 1)
        self.assertEqual(kept["pipeline_descriptors"]["compute_pipeline_descriptors"], [])


if __name__ == "__main__":
    unittest.main()
