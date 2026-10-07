"""CPU-only report admission tests; never launch a renderer."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from f13_f14_check import evaluate_sdk_policy,make_cases
class NativeRadiometryPolicyTests(unittest.TestCase):
    def case(self):return {"sdk_policy":"fallback","sdk_fallback_reason":"unqualified-radiometry"}
    def report(self):return {"denoise_requested":"metalfx","denoise_effective":"custom",
        "denoise_fallback":"UnqualifiedRadiometricDomain: custom Float32 selected before graph encoding",
        "denoise_radiometric_domain":"unqualified-scene-linear","denoise_native_production_qualified":False,
        "denoise_native_factory_requests":0,"denoise_native_encoded_frames":0}
    def test_new_plan_freezes_the_specific_radiometry_reason(self):
        case=next(c for c in make_cases(Path("out"),8,"fallback") if c.get("sdk_policy"))
        self.assertEqual(case["sdk_fallback_reason"],"unqualified-radiometry")
        self.assertEqual(evaluate_sdk_policy(case,self.report()),[])
    def test_missing_factory_and_radiometry_cannot_substitute_for_each_other(self):
        missing={**self.report(),"denoise_fallback":"Denoised gateway factory is absent"}
        self.assertTrue(evaluate_sdk_policy(self.case(),missing))
        legacy={"sdk_policy":"fallback"}
        self.assertEqual(evaluate_sdk_policy(legacy,missing),[])
        self.assertTrue(evaluate_sdk_policy(legacy,self.report()))
    def test_native_work_or_native_claim_fails_the_pregraph_custom_contract(self):
        for key,value in (("denoise_effective","metalfx"),("denoise_native_factory_requests",1),
            ("denoise_native_encoded_frames",1),("denoise_native_production_qualified",True),
            ("denoise_radiometric_domain","controlled-fixture-diagnostic")):
            self.assertTrue(evaluate_sdk_policy(self.case(),{**self.report(),key:value}))
    def test_absent_or_invalid_counter_evidence_does_not_mean_zero(self):
        for value in (None,False,"0",-1):
            self.assertTrue(evaluate_sdk_policy(self.case(),{**self.report(),"denoise_native_encoded_frames":value}))
    def test_native_effective_string_without_admission_and_work_cannot_pass(self):
        self.assertTrue(evaluate_sdk_policy({"sdk_policy":"native"},{**self.report(),"denoise_effective":"metalfx"}))
if __name__=="__main__":unittest.main()
