"""Written-only independent closed-form temporal fixtures. No GPU execution."""
import importlib.util
import math
from pathlib import Path
import sys
import unittest

path=Path(__file__).resolve().parents[1]/"tools"/"temporal_light_metrics.py"
spec=importlib.util.spec_from_file_location("temporal_light_metrics",path)
metrics=importlib.util.module_from_spec(spec);sys.modules[spec.name]=metrics;spec.loader.exec_module(metrics)


def config(events=(),require_change=False):
    return {"linear":True,"frozen_before_run":True,"reconstruction_support_radius":1,"events":list(events),
            "rois":[{"name":"fixture","xywh":[0,0,2,2],"scale":1,"require_change":require_change,
                     "thresholds":{"linear_rmse_p95_max":0.06,"linear_rmse_max":0.15,
                                   "residual_flicker_mean_max":0.02,"support_ghost_fraction_max":0.25,
                                   "recovery_frames_max":4,"recovery_linear_rmse":0.06}}]}


def d4(events=()):
    c=config(events);c["rois"][0]["ghost_metric"]="error-change";return c


class ErrorChangeGhostD4(unittest.TestCase):
    """Owner decision D4: ghost on the change of the error (F13.6 ROIs only)."""
    def images(self,values):return [metrics.solid(2,2,float(v)) for v in values]
    def key(self,result):return result["regions"]["fixture"]["metrics"]["error_change_ghost_fraction"]
    def test_one_frame_delay_at_a_step_fails(self):
        ref=self.images([0,0,1,1,1,1]);late=self.images([0,0,0,1,1,1])
        result=metrics.analyse_clips(ref,late,d4())
        self.assertAlmostEqual(self.key(result),1);self.assertFalse(result["regions"]["fixture"]["checks"]["ghost_support"])
    def test_exponential_history_at_a_step_fails(self):
        ref=self.images([0,0,1,1,1,1,1,1]);alpha=.2;value=0.0;ema=[]
        for r in [0,0,1,1,1,1,1,1]:value=value+alpha*(r-value);ema.append(value)
        result=metrics.analyse_clips(ref,self.images(ema),d4())
        # Clip-level regression over the step and the next frame: (1-alpha^2)/2.
        self.assertAlmostEqual(self.key(result),(1-alpha*alpha)/2,places=6);self.assertFalse(result["regions"]["fixture"]["checks"]["ghost_support"])
    def test_static_bias_under_steady_drift_passes_and_fails_the_legacy_metric(self):
        ref=self.images([.1*i for i in range(8)]);biased=self.images([.1*i-.05 for i in range(8)])
        result=metrics.analyse_clips(ref,biased,d4())
        self.assertEqual(self.key(result),0);self.assertTrue(result["regions"]["fixture"]["checks"]["ghost_support"])
        legacy=metrics.analyse_clips(ref,biased,config())
        self.assertGreater(legacy["regions"]["fixture"]["metrics"]["support_ghost_fraction_max"],.25)
        self.assertFalse(legacy["regions"]["fixture"]["checks"]["ghost_support"])
    def test_independent_flicker_is_not_ghost(self):
        ref=self.images([0,0,1,1,1,1]);noisy=self.images([.01,-.01,1.01,.99,1.01,.99])
        self.assertLess(self.key(metrics.analyse_clips(ref,noisy,d4())),.25)
    def test_unknown_ghost_definition_is_refused(self):
        c=config();c["rois"][0]["ghost_metric"]="lenient"
        with self.assertRaises(ValueError):metrics.analyse_clips(self.images([0,1]),self.images([0,1]),c)
    def test_default_remains_the_f8_metric(self):
        self.assertEqual(metrics.ghost_metric_key(config()["rois"][0]),"support_ghost_fraction_max")


class TemporalFixtures(unittest.TestCase):
    def images(self,values):return [metrics.solid(2,2,float(v)) for v in values]
    def test_identical_linear_step_has_zero_error_and_immediate_recovery(self):
        ref=self.images([0,0,1,1,1,1])
        c=config([{"name":"step","frame":2,"window":4}],True)
        result=metrics.analyse_clips(ref,ref,c)
        self.assertTrue(result["passed"])
        self.assertEqual(result["regions"]["fixture"]["metrics"]["recovery_frames"]["step"],0)
        self.assertEqual(result["regions"]["fixture"]["metrics"]["support_ghost_fraction_max"],0)

    def test_known_one_frame_lag_at_uniform_step_is_detected_independently(self):
        ref=self.images([0,0,1,1,1,1]);lag=self.images([0,0,0,1,1,1])
        result=metrics.analyse_clips(ref,lag,config([{"name":"cut","frame":2,"window":4}],True))
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["regions"]["fixture"]["metrics"]["support_ghost_fraction_max"],1)
        self.assertEqual(result["regions"]["fixture"]["metrics"]["recovery_frames"]["cut"],1)

    def test_corrupt_history_recovery_uses_stable_window_not_one_lucky_frame(self):
        ref=self.images([0,0,1,1,1,1,1,1,1])
        candidate=self.images([0,0,0,0.5,0.75,0.875,0.9375,0.96875,1])
        result=metrics.analyse_clips(ref,candidate,config([{"name":"cut","frame":2,"window":7}]))
        # Exact step error .5^n crosses .06 only at frame7; offset7-2=5.
        self.assertEqual(result["regions"]["fixture"]["metrics"]["recovery_frames"]["cut"],5)
        self.assertFalse(result["regions"]["fixture"]["checks"]["recovery"])
        lucky=self.images([0,0,1,0,0,0,0,0,0])
        result=metrics.analyse_clips(ref,lucky,config([{"name":"cut","frame":2,"window":7}]))
        self.assertIsNone(result["regions"]["fixture"]["metrics"]["recovery_frames"]["cut"])

    def test_signed_bias_and_residual_flicker_closed_form(self):
        ref=self.images([1,1,1,1]);candidate=self.images([1.1,0.9,1.1,0.9])
        result=metrics.analyse_clips(ref,candidate,config())
        rows=result["regions"]["fixture"]["frames"]
        self.assertAlmostEqual(rows[1]["residual_flicker"],0.2,places=6)
        self.assertAlmostEqual(result["regions"]["fixture"]["metrics"]["signed_bias_mean"],0,places=6)
        self.assertFalse(result["regions"]["fixture"]["checks"]["flicker"])
        self.assertFalse(metrics.analyse_clips(ref,ref,config(require_change=True))["passed"])

    def test_nan_and_resolution_mismatch_never_pass(self):
        with self.assertRaises(ValueError):
            metrics.analyse_clips(self.images([1,1]),[metrics.solid(3,2,1),metrics.solid(3,2,1)],config())
        with self.assertRaises(ValueError):
            metrics.analyse_clips(self.images([1,1]),self.images([1,math.nan]),config())

    def test_homogeneous_fog_exact_independent_oracle(self):
        value=metrics.homogeneous_fog(math.log(2),[1,2,0],1)
        self.assertAlmostEqual(value["transmittance"],0.5)
        self.assertAlmostEqual(value["radiance"][0],0.5/math.log(2))
        self.assertAlmostEqual(value["radiance"][1],1/math.log(2))
        self.assertEqual(metrics.homogeneous_fog(0,[1,2,3],2)["radiance"],[2,4,6])


if __name__=="__main__":unittest.main()
