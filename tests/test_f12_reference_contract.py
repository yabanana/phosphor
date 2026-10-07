"""Written only: independent offline role/MASK/triangle fixtures, no render."""
import importlib.util
import json
import pathlib
import tempfile
import types
import unittest

path=pathlib.Path(__file__).resolve().parents[1]/"tools"/"f12_reference.py"
spec=importlib.util.spec_from_file_location("f12_reference_contract",path)
reference=importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class ReferenceContract(unittest.TestCase):
    def material(self):
        return {"base":[1,1,1,0.5],"emissive":[8,4,2],"alpha_cutoff":0.25}

    def test_mask_filters_emission_not_only_bsdf(self):
        material=self.material()
        # Independent scalar oracle: factor0.5 * filteredLOD0 alpha0.25=0.125
        # below0.25 cutoff. The previous emissive-only factor would emit8W.
        self.assertEqual(reference.masked_emission(material,[1,1,1],0.25),(0,0,0))
        self.assertNotEqual((8,4,2),reference.masked_emission(material,[1,1,1],0.25))
        # Equality at the cutoff survives, as engine alpha>=cutoff does.
        self.assertEqual(reference.masked_emission(material,[0.5,0.5,0.5],0.5),(4,2,1))
        # Opaque does not consult alpha.
        material["alpha_cutoff"]=0
        self.assertEqual(reference.masked_emission(material,[1,1,1],0),(8,4,2))

    def test_ray_roles_cannot_be_overridden_by_model_difference_option(self):
        reference.validate_reference_roles({"instances":[{"flags":16|3}]})
        for flags in (16|1,16|2,16):
            with self.assertRaisesRegex(ValueError,"visibility/shadow roles"):
                reference.validate_reference_roles({"instances":[{"flags":flags}]})
            with tempfile.TemporaryDirectory() as folder:
                snapshot={"schema":1,"linear":True,"instances":[{"flags":flags}],
                          "meshes":[],"textures":[],"materials":[],"sampled_lights":[]}
                (pathlib.Path(folder)/"scene.json").write_text(json.dumps(snapshot))
                args=types.SimpleNamespace(snapshot=folder,allow_model_differences=True)
                # render validates roles BEFORE importing NumPy/Mitsuba. This
                # control therefore needs no renderer and cannot silently pass.
                with self.assertRaisesRegex(ValueError,"visibility/shadow roles"):
                    reference.render(args)

    def test_triangle_dedup_uses_material_metadata_provenance(self):
        standalone={"type":6,"flags":0,"material":reference.INVALID}
        extracted={"type":6,"flags":2,"material":0}
        self.assertFalse(reference.extracted_mesh_triangle(standalone,1))
        self.assertTrue(reference.extracted_mesh_triangle(extracted,1))
        self.assertFalse(reference.extracted_mesh_triangle({"type":3,"material":0},1))
        with tempfile.TemporaryDirectory() as folder:
            snapshot={"schema":1,"linear":True,"instances":[],"meshes":[],"textures":[],
                      "materials":[],"sampled_lights":[standalone]}
            (pathlib.Path(folder)/"scene.json").write_text(json.dumps(snapshot))
            with self.assertRaisesRegex(ValueError,"standalone triangle"):
                reference.snapshot(folder)
            standalone["geometry"]="light_triangle_0.ply"
            (pathlib.Path(folder)/"light_triangle_0.ply").write_text("fixture geometry resource")
            (pathlib.Path(folder)/"scene.json").write_text(json.dumps(snapshot))
            _,loaded=reference.snapshot(folder)
            self.assertEqual(loaded["sampled_lights"][0]["geometry"],"light_triangle_0.ply")


if __name__=="__main__":
    unittest.main()
