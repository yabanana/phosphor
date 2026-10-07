"""WRITTEN ONLY. Independent closed-form oracles; no renderer is launched."""
import sys
import copy
import json
import math
import tempfile
from array import array
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from metalfx_denoise_fixture_check import constant_oracle,impulse_oracle,guide_oracle,leaks_at_exit,frozen_prewarm_budget,make_cases,frozen_auto_exposure_experiment,exposure_mode_oracle,frozen_manual_exposure_control,MANUAL_EXPOSURE_REQUESTED_FP32,MANUAL_EXPOSURE_R16
import metalfx_denoise_fixture_check as runner

class SDKNumpyEquivalenceTests(unittest.TestCase):
    def both(self,call):
        numpy=runner._NUMPY
        if numpy is None:self.skipTest("optional NumPy unavailable; standard-library tests still run")
        with patch.object(runner,"_NUMPY",None):slow=call()
        with patch.object(runner,"_NUMPY",numpy):fast=call()
        return slow,fast
    def assert_numeric_equal(self,a,b):
        if isinstance(a,dict):
            self.assertEqual(set(a),set(b))
            for k in a:self.assert_numeric_equal(a[k],b[k])
        elif isinstance(a,float) and math.isnan(a):self.assertTrue(math.isnan(b))
        else:self.assertEqual(a,b)
    def test_constant_float32_float64_and_subnormal_equations_exact(self):
        for kind in ("f","d"):
            for target,factor in (([.5,.25,.125],1),([.5,.25,.125],1/64),([368640,128,64],1/64),
                                  ([1e-40,0.,-0.],1),([0.,0.,0.],1)):
                sdk=array(kind,[v*factor for v in target]*7);restored=array(kind,target*7)
                a,b=self.both(lambda:runner.constant_oracle(sdk,restored,target,factor));self.assert_numeric_equal(a,b)
        tiny=float.fromhex("0x0.0000000000001p-1022")
        a,b=self.both(lambda:runner.constant_oracle(array("d",[tiny]*3),array("d",[tiny]*3),[tiny]*3,1));self.assert_numeric_equal(a,b)
    def test_threshold_neighbors_keep_identical_decisions(self):
        for threshold in (runner.GATES["constant_relative_max"],runner.GATES["restore_relative_max"]):
            center=1+threshold
            for value in (math.nextafter(center,-math.inf),center,math.nextafter(center,math.inf)):
                for factor in (1,1/64):
                    sdk=array("d",[value*factor]*3);restored=array("d",[value]*3)
                    a,b=self.both(lambda:runner.constant_oracle(sdk,restored,[1]*3,factor));self.assert_numeric_equal(a,b)
                    a,b=self.both(lambda:runner.constant_oracle(array("d",[factor]*3),restored,[1]*3,factor));self.assert_numeric_equal(a,b)
    def test_relative_max_preserves_double_order_and_nonfinite_semantics(self):
        for a,b in (([0.,-0.,1e-320],[0.,1e-320,0.]),([1e308,-1e308],[1e308,1e308]),
                    ([math.nan,1.],[1.,1.]),([1.,math.nan],[1.,1.]),([math.inf],[math.inf])):
            a,b=array("d",a),array("d",b)
            slow,fast=self.both(lambda:runner.relative_max(a,b));self.assert_numeric_equal(slow,fast)
    def test_invalid_radiance_and_layout_rejected_in_both_backends(self):
        numpy=runner._NUMPY
        for backend in (None,numpy):
            with patch.object(runner,"_NUMPY",backend):
                for data in ([],[True],["1"],[None],array("d",[math.nan]),array("d",[math.inf]),array("d",[-1e-320])):
                    with self.assertRaises(ValueError):runner.pixels(data)
                for factor in (0,-1,math.nan,math.inf,True):
                    with self.assertRaises(ValueError):runner.constant_oracle(array("d",[1]*3),array("d",[1]*3),[1]*3,factor)
                with self.assertRaises(ValueError):runner.constant_oracle(array("d",[1]*2),array("d",[1]*2),[1]*3,1)
                with self.assertRaises(ValueError):runner.relative_max(array("d"),array("d"))
                with self.assertRaises(ValueError):runner.relative_max(array("d",[1]),array("d",[1,2]))

class SDKAnalyzeOnlyTests(unittest.TestCase):
    def fixture(self,base,lifecycle=False):
        out=base/"original";folder=out/"case";folder.mkdir(parents=True)
        case={"name":"case","scenario":"lifecycle" if lifecycle else "constant","frames":300 if lifecycle else 96,
              "preexposed":False,"expected_exit":0,"expected_state":"FIXTURE_CHECKS_PASSED","prewarm_ms":120000,
              "capture":str(folder/"actual-sdk"),"log":str(folder/"renderer.log"),"lifecycle":lifecycle,
              "command":["/not-needed/phosphor","--denoised-fixture-prewarm-ms","120000"]}
        if lifecycle:case["leaks_tool"]="/not-needed/leaks"
        manifest={"thresholds":runner.GATES,"hard_stop":"STOP_AFTER_F14","frozen_before_run":True,
                  "gateway":"native","prewarm_ms":120000,"cases":[case],"binary_sha256":"b"*64}
        path=out/"sdk-frozen-manifest.json";path.write_bytes(runner.json_bytes(manifest))
        provenance={"source_sha":"a"*64,"binary_sha":"b"*64,"manifest_sha":runner.sha(path),
                    "timeout_seconds":600,"frozen_before_first_case":True}
        (out/"sdk-execution-provenance.json").write_bytes(runner.json_bytes(provenance))
        command=case["command"] if not lifecycle else [case["leaks_tool"],"--atExit","--",*case["command"]]
        status={"command":command,"expected_exit":0,"returncode":1 if lifecycle else 0,"exit_marker":0,
                "binary_sha256":"b"*64,"failures":["process exit 1, expected 0"] if lifecycle else [],"timed_out":False,"signal":None}
        (folder/"renderer.log.status.json").write_bytes(runner.json_bytes(status))
        (folder/"renderer.log").write_text("EXIT 0\n"+("Process 7: 20 leaks for 12800 total leaked bytes.\n" if lifecycle else ""))
        (out/"sdk-results.json").write_text("PREEXISTING RAW RESULT\n")
        return out,path,provenance
    def test_reanalysis_has_separate_identity_never_relaunches_or_rehashes_gpu_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,path,provenance=self.fixture(Path(tmp));original={p:p.read_bytes() for p in out.rglob("*") if p.is_file()}
            with patch.object(runner,"run_checked",side_effect=AssertionError("renderer must not run")),\
                 patch.object(runner,"source_hash",side_effect=AssertionError("GPU source must not be relabelled")),\
                 patch.object(runner,"gpu_lock",side_effect=AssertionError("no GPU scheduling")),\
                 patch.object(runner,"evaluate_capture",return_value={"errors":[],"native_numerical_proof":True}):
                result=runner.analyze_existing(out,path,Path(tmp)/"analysis")
            self.assertTrue(result["passed"]);self.assertEqual(result["provenance"],provenance)
            self.assertEqual(result["gpu_processes_launched"],0);self.assertIn("sha256",result["analyzer"])
            self.assertTrue(all(p.read_bytes()==data for p,data in original.items()))
            with self.assertRaises(ValueError):runner.analyze_existing(out,path,Path(tmp)/"analysis")
            bad=json.loads((out/"sdk-execution-provenance.json").read_text());bad["manifest_sha"]="c"*64
            (out/"sdk-execution-provenance.json").write_bytes(runner.json_bytes(bad))
            with self.assertRaises(ValueError):runner.analyze_existing(out,path,Path(tmp)/"different")
    def test_completed_native_frames_do_not_override_original_exit_or_leak_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,path,_=self.fixture(Path(tmp),True)
            with patch.object(runner,"evaluate_capture",return_value={"errors":[],"native_numerical_proof":True}):
                result=runner.analyze_existing(out,path,Path(tmp)/"analysis")
            self.assertFalse(result["passed"]);row=result["results"][0]
            self.assertTrue(row["evaluation"]["native_numerical_proof"])
            self.assertEqual(row["evaluation"]["process_at_exit"]["leaks"],20)
            self.assertEqual(row["evaluation"]["process_at_exit"]["leaked_bytes"],12800)

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

class SDKManualExposureProtocolTests(unittest.TestCase):
    def plan(self):
        case=make_cases(Path("out"),96,"native",False,True)[0]
        case["command"]=["phosphor","--frames","96","--denoised-fixture","wide-hdr",
            "--denoised-fixture-pre-exposed","--denoised-fixture-manual-exposure-control"]
        return {"gateway":"native","auto_exposure_experiment":False,"manual_exposure_control":True,"cases":[case]}
    def test_exact_r16_quantization_and_one_parameter_experiment(self):
        self.assertEqual(MANUAL_EXPOSURE_R16,1456*2**-24)
        self.assertLess(abs(MANUAL_EXPOSURE_REQUESTED_FP32-.5/5760),(.5/5760)*2**-24)
        self.assertEqual(MANUAL_EXPOSURE_R16*5760,.4998779296875)
        self.assertTrue(frozen_manual_exposure_control(self.plan()))
        self.assertFalse(frozen_auto_exposure_experiment(self.plan()))
        with self.assertRaises(ValueError):make_cases(Path("out"),96,"native",True,True)
        for key,value in (("expected_manual_exposure_r16",0),("requested_manual_exposure_fp32",1),
            ("physical_target",[.5,.25,.125]),("pre_exposure",1),("frames",192)):
            bad=copy.deepcopy(self.plan());bad["cases"][0][key]=value
            with self.assertRaises(ValueError):frozen_manual_exposure_control(bad)
    def test_actual_manual_texel_must_be_read_back_and_not_flushed(self):
        record={"auto_exposure_requested":False,"exposure_descriptor_configured":True,
            "auto_exposure_enabled":False,"exposure_mode":"manual","provided_manual_exposure_texture_value":MANUAL_EXPOSURE_R16,
            "manual_exposure_texture_ignored":False,"packed_exposure_is_provided_manual_value":True,
            "manual_exposure_control":True,"requested_manual_exposure_fp32":MANUAL_EXPOSURE_REQUESTED_FP32,
            "expected_manual_exposure_r16":MANUAL_EXPOSURE_R16,"provided_manual_exposure_fp32":MANUAL_EXPOSURE_R16,
            "manual_exposure_prequantized":True,"manual_exposure_basis":"packed-input-color","actual_manual_exposure_readback":True,
            "actual_provided_manual_exposure_texture_value":MANUAL_EXPOSURE_R16}
        self.assertTrue(exposure_mode_oracle(record,False,True,True))
        for key,value in (("actual_manual_exposure_readback",False),("actual_provided_manual_exposure_texture_value",0),
            ("provided_manual_exposure_fp32",MANUAL_EXPOSURE_REQUESTED_FP32),("manual_exposure_prequantized",False),
            ("manual_exposure_basis","physical-radiance"),
            ("provided_manual_exposure_texture_value",MANUAL_EXPOSURE_REQUESTED_FP32),("manual_exposure_texture_ignored",True)):
            with self.assertRaises(ValueError):exposure_mode_oracle({**record,key:value},False,True,True)
    def test_gpu_input_texel_check_is_independent_of_metadata(self):
        authored={"color":[368640,128,64],"normal":[0,0,1],"roughness":.5,"depth":.025,
            "motion":[0,0],"diffuseR":.6,"specularR":.04,"exposure":1,"hitDistance":0,"reactive":0,"strength":0}
        packed={**authored,"color":[5760,2,1],"exposure":MANUAL_EXPOSURE_R16}
        record={"input_width":64,"input_height":64,"frame":8,"phase":0,"scenario":"wide-hdr","preExposure":1/64,
            "input_samples":[dict(authored) for _ in range(4)],"packed_samples":[dict(packed) for _ in range(4)]}
        self.assertTrue(guide_oracle(record,True))
        for value in (0,1,MANUAL_EXPOSURE_REQUESTED_FP32):
            bad=copy.deepcopy(record);bad["packed_samples"][2]["exposure"]=value
            with self.assertRaises(ValueError):guide_oracle(bad,True)

if __name__=="__main__":unittest.main()
