"""WRITTEN ONLY: authored scalar fixtures, not GPU evidence or test results."""
import copy
import math
from pathlib import Path
import struct
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import json
import tempfile
from volume_snapshot_check import VOLUME_GATES,FOG_SLICING_GATES,analytic_component,camera_geometry_evidence,check_record,fog_slicing_evidence,fog_slicing_sample,volume_words,temporal_record

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

def perspective_blocks(width=64,height=36,near=0.1,far=500.0,camera=(0.0,1.7,0.0)):
    """Fog block of a 60-degree reverse-Z infinite camera looking down -Z."""
    t=math.tan(math.radians(30));aspect=width/height;inv=[0.0]*16
    # inverse(P*V) for V = translate(-camera): world = camera + (x*a*t, y*t, -1)/(z/n)
    # h = (x*a*t + cx*z/n, y*t + cy*z/n, -1 + cz*z/n, z/n), column-major m[4c+r]
    inv[0]=aspect*t;inv[5]=t;inv[14]=-1;inv[11]=1/near
    inv[8]=camera[0]/near;inv[9]=camera[1]/near;inv[10]=camera[2]/near
    view=[1,0,0,0, 0,1,0,0, 0,0,1,0, -camera[0],-camera[1],-camera[2],1]
    f=[0.0]*112;f[0:16]=inv;f[16:32]=view;f[72]=near;f[73]=far;f[92:95]=list(camera)
    fw=words(f);fw[64:67]=[64,36,32];fw[68]=width;fw[69]=height
    return f,fw
class FogSlicingTests(unittest.TestCase):
    def snapshot(self,falloff,perturb=0.0,rows=None):
        f,fw=perspective_blocks();f[75]=falloff;fw=words(f);fw[64:67]=[64,36,32];fw[68]=64;fw[69]=36
        f=list(volume_words(fw,112)) # the evaluator sees float32 parameters
        params={"width":64,"height":36,"grid":[64,36,32],"inverse":f[0:16],"view":f[16:32],"camera":f[92:95],"near":f[72],"far":f[73]}
        samples=[]
        for y in range(0,36,3):
            for x in range(0,64,3):
                # Depths spanning every band: a plane at view depth 0.5 m to 400 m, sky above row 6.
                d=0.0 if y<6 else 0.1/(0.5*(800**((y-6)/30)))
                row=[x,y,d,.4,.8,1.7,0,0,0]
                _,exact,model=fog_slicing_sample(params,row,.01,f[75],[1,2,3])
                row[6:9]=[m*(1+perturb) for m in model];samples.append(row)
        record={"parameter_blocks":{"fog":fw},"settings":{"fixture_extinction_m_inv":.01,"fixture_source":[.01,.02,.03]},
                "fog_apply":{"schema":"phosphor.fog-apply-probe.v1","width":64,"height":36,"fixture_height_falloff_m_inv":falloff,"samples":rows or samples}}
        return record
    def evaluate(self,record,falloff):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"frame-000000-view-0.json";path.write_text(json.dumps(record))
            old=FOG_SLICING_GATES["minimum_band_samples"];FOG_SLICING_GATES["minimum_band_samples"]=1
            try:return fog_slicing_evidence([path],falloff)
            finally:FOG_SLICING_GATES["minimum_band_samples"]=old
    def test_model_output_passes_every_band_and_discretisation_is_reported(self):
        for falloff in (0.0,0.05):
            result=self.evaluate(self.snapshot(falloff),falloff)
            self.assertTrue(result["passed"],result["errors"])
            self.assertEqual(sum(b["samples"]>0 for b in result["bands"]),len(result["bands"]))
            self.assertTrue(all(b["max_implementation_error"]<1e-3 for b in result["bands"]))
            self.assertTrue(all(b["max_discretisation_relative"]<FOG_SLICING_GATES["discretisation_relative"] for b in result["bands"]))
    def test_homogeneous_column_model_is_the_exact_exponential(self):
        f,_=perspective_blocks()
        params={"width":64,"height":36,"grid":[64,36,32],"inverse":f[0:16],"view":f[16:32],"camera":f[92:95],"near":f[72],"far":f[73]}
        # A pixel on a column centre: the model equals the continuous medium.
        band,exact,model=fog_slicing_sample(params,[32,18,0.0,0,0,0,0,0,0],.01,0.0,[1,2,3])
        self.assertEqual(band,5)
        for e,m in zip(exact,model):self.assertAlmostEqual(e,m,places=12)
        self.assertAlmostEqual(exact[0],1-math.exp(-.01*(500-.1)),places=4)
    def test_a_one_percent_radiance_error_fails(self):
        result=self.evaluate(self.snapshot(0.0,perturb=0.01),0.0)
        self.assertFalse(result["passed"])
    def test_wrong_fixture_falloff_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"falloff"):self.evaluate(self.snapshot(0.05),0.0)

class CameraGeometryTests(unittest.TestCase):
    def record(self,frame,altitude,sun):
        a=[0.0]*96;a[0:3]=[0,-6360000,0];a[3]=6360000;a[24:27]=sun;a[40:43]=[0,altitude,0]
        # inverse VP whose view-centre ray points down -Z from the camera.
        a[44:60]=[1,0,0,0, 0,1,0,0, 0,altitude,0,1, 0,0,-1,0]
        return {"frame":frame,"parameter_blocks":{"submitted_atmosphere":words(a)}}
    def run_geometry(self,records,spec):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for r in records:
                path=Path(tmp)/f"frame-{r['frame']:06d}-view-0.json";path.write_text(json.dumps(r));paths.append(path)
            return camera_geometry_evidence(paths,spec)
    def test_flight_must_cross_the_layer(self):
        spec={"flight_altitude":{"below":1500.0,"above":4000.0,"minimum_frames":2}}
        up=[self.record(i,1000+i*400,[0,1,0]) for i in range(10)]
        self.assertTrue(self.run_geometry(up,spec)["passed"])
        self.assertFalse(self.run_geometry(up[:3],spec)["passed"])
        base={"flight_altitude":{"below":1500.0,"above":4000.0,"minimum_frames":2,"require":["below","inside"]}}
        self.assertTrue(self.run_geometry(up[:5],base)["passed"])
        self.assertFalse(self.run_geometry(up[:2],base)["passed"])
    def test_backlit_sun_offset(self):
        e=math.radians(8);good=[self.record(0,2,[0,-math.sin(e),-math.cos(e)])]
        spec={"backlit_sun":{"offset_degrees":8.0,"tolerance_degrees":0.5}}
        self.assertTrue(self.run_geometry(good,spec)["passed"])
        self.assertFalse(self.run_geometry([self.record(0,2,[0,1,0])],spec)["passed"])

if __name__=="__main__":unittest.main()
