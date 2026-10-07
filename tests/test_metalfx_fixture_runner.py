"""WRITTEN ONLY. Independent closed-form oracles; no renderer is launched."""
import sys
import copy
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from metalfx_denoise_fixture_check import constant_oracle,impulse_oracle,guide_oracle,leaks_at_exit,frozen_prewarm_budget,make_cases,frozen_auto_exposure_experiment,exposure_mode_oracle

class SDKOracleTests(unittest.TestCase):
    def test_constant_scalar_unit_hypotheses(self):
        physical=[.5,.25,.125];scaled=[v/64 for v in physical]
        a=constant_oracle(scaled,physical,physical,1/64)
        self.assertTrue(a["passed"]);self.assertTrue(a["sdk_preexposed_hypothesis"]);self.assertFalse(a["sdk_physical_hypothesis"])
        b=constant_oracle(physical,[32,16,8],physical,1/64)
        self.assertFalse(b["passed"]);self.assertTrue(b["sdk_physical_hypothesis"])
    def test_wide_hdr_actual_float32_restore(self):
        physical=[368640,128,64];scaled=[5760,2,1]
        self.assertTrue(constant_oracle(scaled,physical,physical,1/64)["passed"])
        with self.assertRaises(ValueError):constant_oracle([float("inf"),2,1],physical,physical,1/64)
    def test_blurred_nonunit_energy_impulse_metamorphism(self):
        # The independent observed operator has energy1.25, not an identity.
        blurred=[.1,.2,.65,.2,.1];scaled=[v/64 for v in blurred]
        self.assertTrue(impulse_oracle(blurred,scaled)["passed"])
        shifted=list(scaled);shifted[0]=0;shifted[4]+=.1/64
        self.assertFalse(impulse_oracle(blurred,shifted)["passed"])
        with self.assertRaises(ValueError):impulse_oracle([0]*5,[0]*5)
    def test_wrong_motion_normal_roughness_packed_input_rejected(self):
        sample={"color":[.5,.25,.125],"normal":[0,0,1],"roughness":.5,"depth":.025,
            "motion":[0,0],"diffuseR":.6,"specularR":.04,"exposure":1,"hitDistance":0,"reactive":0,"strength":0}
        record={"input_width":64,"input_height":64,"frame":8,"phase":0,"scenario":"constant","preExposure":1,
            "input_samples":[dict(sample) for _ in range(4)],"packed_samples":[dict(sample) for _ in range(4)]}
        self.assertTrue(guide_oracle(record))
        for key,value in (("normal",[0,0,-1]),("motion",[1,0]),("roughness",.25),("exposure",2)):
            corrupt={**record,"packed_samples":[dict(sample) for _ in range(4)]};corrupt["packed_samples"][0][key]=value
            with self.assertRaises(ValueError):guide_oracle(corrupt)
    def test_process_exit_lifetime_diagnostic_needs_actual_tool_summary(self):
        self.assertFalse(leaks_at_exit("retirements_submitted=8")["passed"])
        self.assertTrue(leaks_at_exit("Process 123: 0 leaks for 0 total leaked bytes.")["passed"])
        self.assertFalse(leaks_at_exit("Process 123: 1 leak for 80 total leaked bytes.")["passed"])

class SDKPrewarmProtocolTests(unittest.TestCase):
    def plan(self,budget,command_budget=None):
        return {"prewarm_ms":budget,"cases":[{"prewarm_ms":budget,"command":["phosphor","--denoised-fixture-prewarm-ms",str(budget if command_budget is None else command_budget)]}]}
    def test_process_timeout_is_checked_against_actual_frozen_wait(self):
        with self.assertRaises(ValueError):frozen_prewarm_budget(self.plan(120000),2)
        self.assertEqual(frozen_prewarm_budget(self.plan(60000),90),60000)
    def test_a_case_cannot_hide_a_different_wait_in_its_command(self):
        with self.assertRaises(ValueError):frozen_prewarm_budget(self.plan(60000,120000),90)
        with self.assertRaises(ValueError):frozen_prewarm_budget(self.plan(0),90)
        self.assertEqual(frozen_prewarm_budget(self.plan(120000),300),120000)

class SDKAutomaticExposureProtocolTests(unittest.TestCase):
    def plan(self):
        case=make_cases(Path("out"),96,"native",True)[0]
        case["command"]=["phosphor","--frames","96","--denoised-fixture","wide-hdr",
            "--denoised-fixture-pre-exposed","--denoised-fixture-auto-exposure"]
        return {"gateway":"native","auto_exposure_experiment":True,"cases":[case]}
    def test_only_one_wide_hdr_case_and_existing_manual_suite_unchanged(self):
        self.assertTrue(frozen_auto_exposure_experiment(self.plan()))
        manual=make_cases(Path("out"),96,"native")
        self.assertEqual(len(manual),7);self.assertFalse(any(c.get("auto_exposure") for c in manual))
        for frames,gateway in ((95,"native"),(192,"native"),(96,"expected-missing")):
            with self.assertRaises(ValueError):make_cases(Path("out"),frames,gateway,True)
    def test_manifest_cannot_silently_change_units_frames_or_actual_mode(self):
        for key,value in (("physical_target",[5760,2,1]),("pre_exposure",1),("packed_target",[90,2,1]),("frames",192),("expected_exit",1)):
            bad=copy.deepcopy(self.plan());bad["cases"][0][key]=value
            with self.assertRaises(ValueError):frozen_auto_exposure_experiment(bad)
        bad=copy.deepcopy(self.plan());bad["cases"][0]["command"].remove("--denoised-fixture-auto-exposure")
        with self.assertRaises(ValueError):frozen_auto_exposure_experiment(bad)
        bad=copy.deepcopy(self.plan());bad["cases"][0]["command"][2]="192"
        with self.assertRaises(ValueError):frozen_auto_exposure_experiment(bad)
        bad=copy.deepcopy(self.plan());bad["cases"].append(copy.deepcopy(bad["cases"][0]))
        with self.assertRaises(ValueError):frozen_auto_exposure_experiment(bad)
    def test_actual_descriptor_mode_and_manual_texture_semantics_are_required(self):
        record={"auto_exposure_requested":True,"exposure_descriptor_configured":True,
            "auto_exposure_enabled":True,"exposure_mode":"sdk-auto","provided_manual_exposure_texture_value":1,
            "manual_exposure_texture_ignored":True,"packed_exposure_is_provided_manual_value":True}
        self.assertTrue(exposure_mode_oracle(record,True))
        for key,value in (("auto_exposure_enabled",False),("exposure_descriptor_configured",False),
            ("manual_exposure_texture_ignored",False),("provided_manual_exposure_texture_value",.015625),
            ("packed_exposure_is_provided_manual_value",False)):
            with self.assertRaises(ValueError):exposure_mode_oracle({**record,key:value},True)
        with self.assertRaises(ValueError):exposure_mode_oracle(record,False)
    def test_auto_mode_does_not_excuse_radiometric_loss_or_change_output_equation(self):
        physical=[368640,128,64];sdk=[5760,2,1]
        self.assertTrue(constant_oracle(sdk,physical,physical,1/64)["passed"])
        damaged=[31.98,.82,.5]
        self.assertFalse(constant_oracle(damaged,[v*64 for v in damaged],physical,1/64)["passed"])
        self.assertFalse(constant_oracle(sdk,sdk,physical,1/64)["passed"])

if __name__=="__main__":unittest.main()
