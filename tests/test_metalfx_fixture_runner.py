"""WRITTEN ONLY. Independent closed-form oracles; no renderer is launched."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from metalfx_denoise_fixture_check import constant_oracle,impulse_oracle,guide_oracle,leaks_at_exit,frozen_prewarm_budget

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

if __name__=="__main__":unittest.main()
