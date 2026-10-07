#!/usr/bin/env python3
"""Freeze thin-wall ROIs from geometry only; never opens candidate images."""
import argparse, json, pathlib
import numpy as np
from f12_oracle_validate import load_reference, geometry, ray, intersect, erode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['reference-script','snapshot','protocol','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();ref=load_reference(args.reference_script);root,data=ref.snapshot(args.snapshot)
    protocol=json.loads(pathlib.Path(args.protocol).read_text())['thin_wall'];camera=data['camera']
    if [camera['width'],camera['height']]!=protocol['capture_resolution']:raise ValueError('use the frozen256x144 thin-wall capture size')
    triangles,slots,_=geometry(root,data);wall=[];emitters=[]
    for instance in data['instances']:
        points=triangles[slots==instance['slot']].reshape(-1,3);size=points.max(axis=0)-points.min(axis=0);centre=(points.max(axis=0)+points.min(axis=0))/2
        if np.allclose(size,[.01,4,3.3],atol=2e-5) and np.allclose(centre,[0,2,-.35],atol=2e-5):wall.append(instance['slot'])
        if any(c>0 for c in data['materials'][instance['material']]['emissive']):emitters.append(instance['slot'])
    if len(wall)!=1 or len(emitters)!=1:raise ValueError('snapshot does not have the known unique closed wall and panel')
    emitter_triangles=triangles[np.isin(slots,emitters)];areas=np.linalg.norm(np.cross(emitter_triangles[:,1]-emitter_triangles[:,0],emitter_triangles[:,2]-emitter_triangles[:,0]),axis=1)/2
    light=np.average(emitter_triangles.mean(axis=1),axis=0,weights=areas)
    if light[0]>=-.5 or data['lights'] or any(data['sky']):raise ValueError('unsupported extra light/sky or wrong compartment emitter')
    h,w=camera['height'],camera['width'];bright=np.zeros((h,w),bool);dark=bright.copy();labels=np.full((h,w),-1,dtype=np.int32);surface_counts={}
    for y in range(h):
        for x in range(w):
            origin,direction=ray(camera,x+.5,y+.5);hit=intersect(triangles,origin,direction)
            if not hit:continue
            i,t,_=hit;slot=int(slots[i]);labels[y,x]=slot
            if slot in emitters:continue
            point=origin+direction*t;n=np.cross(triangles[i,1]-triangles[i,0],triangles[i,2]-triangles[i,0]);n/=np.linalg.norm(n)
            if np.dot(n,-direction)<=0 or point[2]>=-.1:continue
            floor=abs(point[1])<1e-5 and n[1]>.99
            back=abs(point[2]+2)<1e-5 and n[2]>.99
            left=abs(point[0]+2)<1e-5 and n[0]>.99
            right=abs(point[0]-2)<1e-5 and n[0]<-.99
            wall_right=slot==wall[0] and n[0]>.99
            start=point+n*1e-5;delta=light-start;distance=np.linalg.norm(delta);block=intersect(triangles,start,delta/distance,distance-1e-5)
            bright[y,x]=(floor or back or left) and point[0]<-.1 and block is None
            dark[y,x]=(floor or back or right or wall_right) and point[0]>=.004 and block is not None and slots[block[0]]==wall[0]
    masks={'bright_control':erode(bright,2),'dark_behind_wall':erode(dark,2)}
    counts={name:int(mask.sum()) for name,mask in masks.items()}
    for name,mask in masks.items():surface_counts[name]={str(slot):int(np.count_nonzero(mask&(labels==slot))) for slot in np.unique(labels[mask])}
    passed=all(n>=protocol['minimum_pixels_each_mask'] for n in counts.values())
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=False);np.savez_compressed(out/'masks.npz',**masks);np.save(out/'primary_labels.npy',labels)
    report={'schema':1,'state':'GEOMETRIC_REGIONS_FROZEN_NO_CANDIDATE_READ','snapshot_sha256':ref.scene_digest(root),'counts':counts,'per_surface_counts':surface_counts,'wall_slot':int(wall[0]),'emitter_slot':int(emitters[0]),'emitter_centre':light.tolist(),'coverage_passed':passed,'candidate_pixels_read':False}
    (out/'regions.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
