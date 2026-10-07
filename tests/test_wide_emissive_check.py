"""Independent analytic HDR oracle tests. No GPU or renderer process."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from wide_emissive_check import evaluate,TARGET,GAMMA16
class WideEmissiveOracleTests(unittest.TestCase):
    def test_full_image_constant_reference_passes(self):
        self.assertTrue(evaluate(list(TARGET)*64)["passed"])
        self.assertEqual(evaluate(list(TARGET)*64)["pixels"],64)
    def test_old_half_saturation_overflow_and_black_repair_are_detected(self):
        for old in ((65504,128,64),(float("inf"),128,64),(0,0,0)):
            self.assertFalse(evaluate(list(old)*64)["passed"])
    def test_small_channels_and_single_bad_pixel_cannot_hide_behind_red(self):
        for c in range(3):
            image=list(TARGET)*64;image[63*3+c]+=TARGET[c]*GAMMA16*2
            self.assertFalse(evaluate(image)["passed"])
        with self.assertRaises(ValueError):evaluate([])
        with self.assertRaises(ValueError):evaluate([1,2])
if __name__=="__main__":unittest.main()
