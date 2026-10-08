#!/usr/bin/env python3
"""Independent CPU test of captured F12 receiver self-intersections."""
import argparse, ctypes, json, math, pathlib
import numpy as np
from f12_oracle_validate import load_reference, geometry, ray, intersect


def offset(point,normal):
    point=np.asarray(point,dtype=np.float32);normal=np.asarray(normal,dtype=np.float32)
    integer=(np.float32(256)*normal).astype(np.int32)
    bits=point.view(np.int32)
    shifted=(bits+np.where(point<0,-integer,integer)).astype(np.int32).view(np.float32)
    return np.where(np.abs(point)<np.float32(1/32),point+normal*np.float32(1/65536),shifted)


def corrected(point,eye,triangle):
    """Simulate float32 guide fix; double geometry below is the independent oracle."""
    point,eye,triangle=[np.asarray(x,dtype=np.float32) for x in [point,eye,triangle]]
    n=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);n/=np.linalg.norm(n)
    direction=point-eye;denominator=np.dot(direction,n)
    if not np.isfinite(denominator) or abs(denominator)<1e-10:return None
    t=np.dot(triangle[0]-eye,n)/denominator
    if not np.isfinite(t) or t<=0:return None
    result=eye+direction*t
    return result-n*np.dot(result-triangle[0],n)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-script',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--masks',required=True);p.add_argument('--positions',required=True);p.add_argument('--normals',required=True);p.add_argument('--output',required=True);p.add_argument('--barycentric-library');a=p.parse_args()
    ref=load_reference(a.reference_script);root,data=ref.snapshot(a.snapshot);triangles,slots,_=geometry(root,data);positions=ref.read_pfm(a.positions);normals=ref.read_pfm(a.normals);masks=dict(np.load(a.masks));names=[n for n in masks if n not in ['interior','occluded_floor']]
    barycentric=None
    if a.barycentric_library:
        library=ctypes.CDLL(str(pathlib.Path(a.barycentric_library).resolve()))
        barycentric=library.f12ReceiverWeights
        pointer=ctypes.POINTER(ctypes.c_float)
        barycentric.argtypes=[pointer,ctypes.c_float,ctypes.c_float,ctypes.c_float,ctypes.c_float,pointer]
        barycentric.restype=ctypes.c_uint
    def geometry_position(triangle,x,y):
        if barycentric is None:return corrected(positions[y,x],data['camera']['position'],triangle)
        camera=data['camera'];eye=np.asarray(camera['position'],dtype=np.float32)
        forward=np.asarray(camera['direction'],dtype=np.float32);forward/=np.linalg.norm(forward)
        right=np.cross(forward,np.asarray(camera['up'],dtype=np.float32));right/=np.linalg.norm(right);up=np.cross(right,forward)
        relative=np.asarray(triangle,dtype=np.float32)-eye
        sy=np.float32(1/math.tan(camera['fov_y_radians']/2));sx=sy/np.float32(camera['width']/camera['height'])
        clip=np.zeros((3,4),dtype=np.float32);clip[:,0]=(relative@right)*sx;clip[:,1]=(relative@up)*sy;clip[:,3]=relative@forward
        weights=np.zeros(3,dtype=np.float32)
        if not barycentric(clip.ctypes.data_as(pointer),x+.5,y+.5,camera['width'],camera['height'],weights.ctypes.data_as(pointer)):return None
        t=np.asarray(triangle,dtype=np.float32)
        return t[0]*weights[0]+t[1]*weights[1]+t[2]*weights[2]
    results={};corrected_positions=positions.copy()
    for name in names:
        result={'pixels':int(masks[name].sum()),'rays':0,'before_self_hit':0,'after_self_hit':0,'before_backface':0,'after_backface':0,'after_front_hit':0,'after_miss':0,'max_before_plane_error':0.,'max_after_plane_error':0.,'max_corrected_position_error':0.,'minimum_cosine':1.}
        for y,x in np.argwhere(masks[name]):
            ro,rd=ray(data['camera'],x+.5,y+.5);triangle_index,distance,_=intersect(triangles,ro,rd);triangle=triangles[triangle_index];primary_slot=slots[triangle_index]
            exact=ro+rd*distance;true_n=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);true_n/=np.linalg.norm(true_n)
            point=positions[y,x];n=normals[y,x].astype(float);n/=np.linalg.norm(n)
            after=geometry_position(triangle,x,y);assert after is not None;corrected_positions[y,x]=after
            result['max_before_plane_error']=max(result['max_before_plane_error'],abs(float(np.dot(point-exact,true_n))))
            result['max_after_plane_error']=max(result['max_after_plane_error'],abs(float(np.dot(after-exact,true_n))))
            result['max_corrected_position_error']=max(result['max_corrected_position_error'],float(np.max(np.abs(after-exact))))
            tangent=np.cross([0,0,1] if abs(n[2])<.999 else [0,1,0],n);tangent/=np.linalg.norm(tangent);bitangent=np.cross(n,tangent)
            # An independent deterministic cosine quadrature. No copied shader RNG.
            for k in range(16):
                u=(k+.5)/16;angle=2*math.pi*((k*.6180339887498949)%1)
                direction=tangent*(math.sqrt(u)*math.cos(angle))+bitangent*(math.sqrt(u)*math.sin(angle))+n*math.sqrt(1-u)
                result['rays']+=1;result['minimum_cosine']=min(result['minimum_cosine'],float(np.dot(n,direction)))
                for label,start in [('before',point),('after',after)]:
                    hit=intersect(triangles,offset(start,n).astype(float),direction,100.)
                    if hit:
                        i,t,_=hit;face_n=np.cross(triangles[i,1]-triangles[i,0],triangles[i,2]-triangles[i,0])
                        back=np.dot(face_n,-direction)<=0
                        if slots[i]==primary_slot and t<.01:result[f'{label}_self_hit']+=1
                        if back:result[f'{label}_backface']+=1
                        elif label=='after':result['after_front_hit']+=1
                    elif label=='after':result['after_miss']+=1
        results[name]=result
    report={'schema':1,'snapshot_sha256':ref.scene_digest(root),'position_method':'shared visibilityBarycentrics C++' if barycentric else 'float32 ray-plane projection','quadrature':'16 independent stratified cosine directions per eroded pixel','ray_offset':'float32 Waechter/Binder, unchanged production constants','regions':results,'passed':all(v['after_self_hit']==0 and v['after_backface']==0 and v['max_after_plane_error']<2e-6 for v in results.values())}
    out=pathlib.Path(a.output);out.mkdir(exist_ok=False,parents=True);(out/'receiver_probe.json').write_text(json.dumps(report,indent=2)+'\n');ref.write_pfm(out/'corrected_positions.pfm',corrected_positions);print(json.dumps(report,indent=2));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
