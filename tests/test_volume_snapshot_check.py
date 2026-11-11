"""WRITTEN ONLY: authored scalar fixtures, not GPU evidence or test results."""
import copy
import math
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from volume_snapshot_check import VOLUME_GATES,analytic_component,check_record,volume_words,temporal_record

def words(floats):return list(struct.unpack("<"+"I"*len(floats),struct.pack("<"+"f"*len(floats),*floats)))
def record_fixture():
    a=[0.0]*96;a[11]=.01;a[24:27]=[.8,-.6,0];a[28:31]=[100]*3
    a[32:40]=[0,1,0,.0045,.0025,.003,.005,.25]
    aw=words(a);aw[86]=7;aw[87]=9;aw[90]=1
    f=[0.0]*112
    for i in (0,5,15,16,21,26,31):f[i]=1
    f[10]=-1;f[72]=1;f[73]=9
    fw=words(f);fw[64:67]=[1,1,2];fw[99]=11
    blocks={"encoding":"little-endian-u32-shared-scalar-ABI","expected_atmosphere":aw,"submitted_atmosphere":list(aw),"fog":fw}
    settings={"fixture_extinction_m_inv":.01,"fixture_source":[.01,.02,.03]}
    cases=[]
    for kind in ("transmittance","multiscattering","sky_view"):
        epoch=9 if kind=="sky_view" else 7
        cases.append({"kind":kind,"pixel":[0,0],"index":0,"expected":[.5]*3+[1],"actual":[.5]*3+[1],
            "expected_epoch":epoch,"actual_epoch":epoch,"tolerance":VOLUME_GATES[kind],"passed":True})
    for kind,count in (("solar_disk",3),("lunar_disk",3),("star_field",64)):
        for index in range(count):
            row={"kind":kind,"pixel":[index,0],"index":index,"expected_epoch":9,"actual_epoch":9,
                "tolerance":VOLUME_GATES[kind],"passed":True}
            row["expected"]=analytic_component(row,blocks,settings) or [0,0,0,1]
            row["actual"]=list(row["expected"]);cases.append(row)
    return {"schema":"phosphor.volume-oracle.v1","kind":"f14-volume","certification":False,
        "provenance":{"source_sha":"s","binary_sha":"b","manifest":"m","shader_generation":2,"source_hash_at_prepare":"abc"},
        "parameter_blocks":blocks,"settings":settings,"cases":cases,"gpu_counters":{"nonfinite":0,"invalid_units":0,"invalid_history":0,"history_reused":0},
        "passed":True,"numeric_failure_count":0,"epoch_mismatch_count":0,"corruption_armed":False}

class VolumeSnapshotTests(unittest.TestCase):
    def test_complete_parameter_blocks_and_real_rows(self):
        record=record_fixture();check=check_record(record,{"source_sha":"s","binary_sha":"b","manifest":"m"})
        self.assertFalse(check["failed"]);self.assertTrue(check["solar_wide"])
        with self.assertRaises(ValueError):volume_words([0]*95,96)
        bad=copy.deepcopy(record);bad["cases"]=[]
        with self.assertRaises(ValueError):check_record(bad,{})
    def test_independent_lunar_radiance_and_star_ray_coverage(self):
        record=record_fixture()
        centre=record["cases"][6]
        self.assertAlmostEqual(centre["expected"][0],.01*.6/((2/3)*math.pi*math.sin(.0045)**2),places=4)
        self.assertEqual(record["cases"][7]["expected"],[0,0,0,1])
        self.assertEqual(record["cases"][8]["expected"],[0,0,0,1])
        centre["expected"][0]*=.25
        with self.assertRaisesRegex(ValueError,"independent analytic equation"):check_record(record,{})
        record=record_fixture();record["cases"].pop()
        with self.assertRaisesRegex(ValueError,"star_field producer samples missing"):check_record(record,{})
        record=record_fixture();record["cases"][-1]=record["cases"][-2]
        with self.assertRaisesRegex(ValueError,"duplicate star_field ray"):check_record(record,{})
    def test_independent_homogeneous_eight_metre_prefix(self):
        record=record_fixture();row={"kind":"homogeneous_fog","index":1}
        answer=analytic_component(row,record["parameter_blocks"],record["settings"])
        self.assertAlmostEqual(answer[3],math.exp(-.08),places=12)
        for value,source in zip(answer[:3],(.01,.02,.03)):
            self.assertAlmostEqual(value,source*(1-math.exp(-.08))/.01,places=12)
    def test_epoch_negative_and_fake_pass_rejected(self):
        record=record_fixture();record["cases"][0]["actual_epoch"]=6
        with self.assertRaises(ValueError):check_record(record,{},negative=True)
        record["cases"][0]["passed"]=False;record["passed"]=False
        record["numeric_failure_count"]=record["epoch_mismatch_count"]=1
        self.assertTrue(check_record(record,{},negative=True)["failed"])
    def test_half_saturation_and_wrong_away_sun_are_real_mismatches(self):
        for index,value in ((3,65504),(4,200)):
            record=record_fixture();record["cases"][index]["actual"][0]=value
            record["cases"][index]["passed"]=False;record["passed"]=False;record["numeric_failure_count"]=1
            self.assertTrue(check_record(record,{},negative=True)["failed"])
    def test_history_control_needs_natural_reuse_and_actual_clock(self):
        record=record_fixture()
        record["clock"]={"seconds":0.,"delta_seconds":0.,"day_fraction":.5,"epoch":2,"reset":False,"frozen_base":True}
        record["history_eligible"]={"fog":True,"clouds":False}
        self.assertFalse(temporal_record(record))
        record["gpu_counters"]["history_reused"]=12
        self.assertTrue(temporal_record(record))
        record["corruption_requested"]=2;record["corruption_armed"]=True
        record["clock"]["reset"]=True
        with self.assertRaises(ValueError):temporal_record(record)
        record["clock"]["reset"]=False;record["history_eligible"]["fog"]=False
        with self.assertRaises(ValueError):temporal_record(record)
        record["history_eligible"]["fog"]=True;record["clock"]["seconds"]=8
        with self.assertRaises(ValueError):temporal_record(record)

if __name__=="__main__":unittest.main()
