import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("merge_pipeline_harvests", Path(__file__).resolve().parents[1] / "tools/merge_pipeline_harvests.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HarvestMerge(unittest.TestCase):
    def test_changed_library_id_reuses_same_function_label(self):
        def corpus(lib, name="trace"):
            return {"libraries": [{"label": lib, "path": "/build/phosphor.metallib"}], "function_descriptors": {"library_function_descriptors": [{"label": "f", "library": lib, "name": name}]}}
        old = module.normalize_library(corpus("old"), "old")
        new = module.normalize_library(corpus("new"), "old")
        self.assertEqual(module.merge(old, new), old)
        with self.assertRaisesRegex(ValueError, "conflicting labeled"):
            module.merge(old, module.normalize_library(corpus("new", "other"), "old"))
        external = corpus("external");external["libraries"][0]["path"] = "/Framework/private.metallib"
        with self.assertRaisesRegex(ValueError, "only engine"):
            module.normalize_library(external, "old")

    def test_unlabeled_formats_and_linkage_survive(self):
        old = {"compute": [{"compute_function_descriptor": "fnd:trace"}], "render": [{"color_attachments": [{"pixel_format": "RGBA16Float"}]}]}
        new = {"compute": [{"compute_function_descriptor": "fnd:trace", "static_linking_descriptor": {"function_descriptors": ["fnd:alpha"]}}], "render": [{"color_attachments": [{"pixel_format": "RGBA32Float"}]}]}
        merged = module.merge(old, new)
        self.assertEqual(len(merged["compute"]), 2)
        self.assertEqual(len(merged["render"]), 2)
        self.assertEqual(module.merge(merged, old), merged)

    def test_nested_scalar_metadata_is_not_a_descriptor_array(self):
        corpus = {"version": {"major": 0, "minor": 1}, "libraries": [{"label": "a", "path": "@PHOSPHOR_METALLIB@"}]}
        self.assertEqual(module.merge(corpus, corpus), corpus)
        with self.assertRaisesRegex(ValueError, "metadata at root.version.minor"):
            module.merge(corpus, {"version": {"major": 0, "minor": 2}})
        with self.assertRaisesRegex(ValueError, "types"):
            module.merge({"pipelines": []}, {"pipelines": {}})


if __name__ == "__main__":
    unittest.main()
