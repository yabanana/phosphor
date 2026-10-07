"""WRITTEN ONLY. Authored analytical/mocked clips; never launch a renderer.

Mock history records exercise the verifier. They are not GPU readback evidence.
"""
from array import array
import copy
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from temporal_light_metrics import Image
from f13_ao_temporal_check import visibility,receiver,load_protocol,validate_metadata,physical_reference,quality

PROVENANCE={"source_sha":"authored-source","binary_sha":"mock-binary","metallib_sha":None,"manifest_sha":"mock-manifest"}
def records():
    result=[]
    for i in range(64):
        post=i>=32;length=min((i-32 if post else i)+1,32)
        result.append({"schema":"phos.f13-ao-temporal.v1","frame":i,"view":0,"signal":9,"width":128,"height":96,
            "fixture_ordinal":i,"wall_x":8 if post else .8,"radius":2,
            "camera":{"position":[0,0,3],"front":[0,0,-1],"up":[0,1,0],"fov_y":math.pi/3},
            "plane":{"z":0,"size":8,"normal":[0,0,1]},"provenance":dict(PROVENANCE),
            "history":{"frame":i,"view":0,"pixels":12288,"valid":12288,"reused":12288 if length>1 else 0,
                "max_length":length,"invalid":0,"invalid_output":0,"epoch":2 if post else 1,"revision":2 if post else 1,
                "reset":i in (0,32),"atrous_iterations":3}})
    return result

def image(reference):
    # Outside the tested oracle domain these are mock scalar pixels, not a
    # fabricated full-scene physical reference used by the production runner.
    data=array("f",[1])*(128*96*3)
    for (x,y),value in reference.items():
        base=(y*128+x)*3
        for j in range(3):data[base+j]=value
    return Image(128,96,data)

class PhysicalAOTests(unittest.TestCase):
    def test_closed_form_disk_cap_and_correct_both_side_far_endpoint(self):
        self.assertAlmostEqual(visibility(0,1,2),1-(math.pi/3-math.sqrt(3)/4)/math.pi,places=14)
        self.assertEqual(visibility(0,2,2),1)
        self.assertEqual(visibility(0,-8,2),1) # Behind a receiver is not total occlusion.
        self.assertEqual(visibility(0,-1,2),visibility(0,1,2))
        with self.assertRaises(ValueError):visibility(.8,.8,2)
    def test_exact_pixel_center_projection_y_down(self):
        camera={"position":[0,0,3],"front":[0,0,-1],"up":[0,1,0],"fov_y":math.pi/3}
        p=receiver(camera,63,47,128,96,0)
        self.assertAlmostEqual(p[0],-3*math.tan(math.pi/6)/96,places=14)
        self.assertAlmostEqual(p[1],3*math.tan(math.pi/6)/96,places=14)
        self.assertEqual(p[2],0)
    def test_fixed_halo_rejects_nonidentifiable_geometry_and_camera(self):
        protocol,_=load_protocol();m=records()[0]
        values=physical_reference(m,protocol);self.assertEqual(len(values),44*48)
        bad=copy.deepcopy(m);bad["camera"]["position"][0]=.001
        with self.assertRaises(ValueError):physical_reference(bad,protocol)
        bad=copy.deepcopy(m);bad["camera"]["fov_y"]=math.pi*.9
        with self.assertRaises(ValueError):physical_reference(bad,protocol)
    def test_black_and_lagging_outputs_fail_frozen_caps(self):
        protocol,_=load_protocol();references=validate_metadata(records(),protocol,PROVENANCE)
        self.assertTrue(quality([image(r) for r in references],references,protocol)["passed"])
        black=[Image(128,96,array("f",[0])*(128*96*3)) for _ in range(64)]
        self.assertFalse(quality(black,references,protocol)["passed"])
        for lag in (1,5):
            candidate=[image(references[max(0,i-lag)]) for i in range(64)]
            result=quality(candidate,references,protocol);self.assertFalse(result["passed"])
            self.assertFalse(result["checks"]["ghost_support"])
            if lag==5:self.assertEqual(result["metrics"]["persistent_recovery_frames"],5)
    def test_omitted_reset_foreign_frame_and_history_reuse_are_rejected(self):
        protocol,_=load_protocol()
        for ordinal,key,value in ((32,"reset",False),(32,"max_length",8),(20,"view",1),(20,"frame",19),(20,"reused",0),(20,"valid",1),(20,"invalid_output",1)):
            bad=records();bad[ordinal]["history"][key]=value
            with self.assertRaises(ValueError):validate_metadata(bad,protocol,PROVENANCE)
    def test_late_rebound_below_other_caps_still_fails_persistent_recovery(self):
        protocol,_=load_protocol();references=validate_metadata(records(),protocol,PROVENANCE)
        candidate=[image(r) for r in references];x,y,w,h=protocol["roi"]["xywh"]
        for yy in range(y,y+h):
            for xx in range(x,x+w):
                for c in range(3):candidate[48].rgb[(yy*128+xx)*3+c]=.93
        result=quality(candidate,references,protocol)
        self.assertFalse(result["passed"]);self.assertFalse(result["checks"]["persistent_recovery"])
        self.assertFalse(result["checks"]["event_ghost"])
        self.assertTrue(all(v for k,v in result["checks"].items() if k not in ("persistent_recovery","event_ghost")))
    def test_one_pixel_partial_rebound_escapes_rms_but_not_event_anchor(self):
        protocol,_=load_protocol();references=validate_metadata(records(),protocol,PROVENANCE)
        candidate=[image(r) for r in references];x,y,_,_=protocol["roi"]["xywh"]
        for c in range(3):candidate[48].rgb[(y*128+x)*3+c]=.95
        result=quality(candidate,references,protocol)
        self.assertFalse(result["passed"]);self.assertFalse(result["checks"]["event_ghost"])
        self.assertTrue(result["checks"]["persistent_recovery"])
        self.assertTrue(all(v for k,v in result["checks"].items() if k!="event_ghost"))
    def test_rgb_scalar_and_actual_provenance_join_required(self):
        protocol,_=load_protocol();m=records();references=validate_metadata(m,protocol,PROVENANCE)
        bad=records();bad[0]["provenance"]["binary_sha"]="other-binary"
        with self.assertRaises(ValueError):validate_metadata(bad,protocol,PROVENANCE)
        candidate=[image(r) for r in references];candidate[0].rgb[1]=0
        with self.assertRaises(ValueError):quality(candidate,references,protocol)

if __name__=="__main__":unittest.main()
