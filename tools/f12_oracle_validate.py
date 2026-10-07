#!/usr/bin/env python3
"""CPU-only independent geometry checks and frozen Cornell masks for F12."""
import argparse, copy, hashlib, importlib.util, json, math, pathlib, struct, time
import numpy as np


def load_reference(path):
    spec = importlib.util.spec_from_file_location('f12_reference_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ply(path):
    with pathlib.Path(path).open('rb') as stream:
        lines = []
        while True:
            line = stream.readline().decode('ascii').strip()
            lines.append(line)
            if line == 'end_header': break
        assert 'format binary_little_endian 1.0' in lines
        nv = int(next(x for x in lines if x.startswith('element vertex')).split()[-1])
        nf = int(next(x for x in lines if x.startswith('element face')).split()[-1])
        vertices = np.frombuffer(stream.read(nv*32), '<f4').reshape(nv,8).astype(float)
        faces = []
        for _ in range(nf):
            assert stream.read(1) == b'\x03'
            faces.append(struct.unpack('<III', stream.read(12)))
        assert not stream.read()
    return vertices, np.asarray(faces)


def geometry(root, data):
    triangles, slots, uvs = [], [], []
    meshes = [ply(root/p) for p in data['meshes']]
    for ins in data['instances']:
        v, f = meshes[ins['mesh']]
        m = np.asarray(ins['world']).reshape(4,4,order='F')
        world = np.c_[v[:,:3],np.ones(len(v))] @ m.T
        triangles.extend(world[f,:3]); slots.extend([ins['slot']]*len(f));uvs.extend(v[f,6:8])
    return np.asarray(triangles), np.asarray(slots), np.asarray(uvs)


def intersect(tris, origin, direction, max_t=1e30):
    e1, e2 = tris[:,1]-tris[:,0], tris[:,2]-tris[:,0]
    p = np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,p)
    valid=np.abs(det)>1e-10;inv=np.zeros_like(det);inv[valid]=1/det[valid]
    tvec=origin-tris[:,0];u=np.einsum('ij,ij->i',tvec,p)*inv
    q=np.cross(tvec,e1);v=q@direction*inv;t=np.einsum('ij,ij->i',e2,q)*inv
    valid &= (u>=-1e-7)&(v>=-1e-7)&(u+v<=1+1e-7)&(t>1e-6)&(t<max_t)
    t[~valid]=np.inf;i=int(np.argmin(t))
    return (i,float(t[i]),np.array([1-u[i]-v[i],u[i],v[i]])) if np.isfinite(t[i]) else None


def ray(camera, x, y):
    forward=np.asarray(camera['direction']);forward/=np.linalg.norm(forward)
    right=np.cross(forward,camera['up']);right/=np.linalg.norm(right);up=np.cross(right,forward)
    sx=(2*x/camera['width']-1)*camera['width']/camera['height']*math.tan(camera['fov_y_radians']/2)
    sy=(1-2*y/camera['height'])*math.tan(camera['fov_y_radians']/2)
    d=forward+sx*right+sy*up;d/=np.linalg.norm(d)
    o=np.asarray(camera['position'])+d*(camera['near']/np.dot(d,forward))
    return o,d


def erode(mask, radius):
    result=mask.copy();padded=np.pad(mask,radius)
    for dy in range(2*radius+1):
        for dx in range(2*radius+1):result &= padded[dy:dy+mask.shape[0],dx:dx+mask.shape[1]]
    return result


def native_constants(scene_dict, data):
    """Exact Cornell-only specialization; do not replace textured scenes."""
    def clone(value):
        return {k:clone(v) for k,v in value.items()} if isinstance(value,dict) else [clone(v) for v in value] if isinstance(value,list) else value
    result=clone(scene_dict)
    active={i['material'] for i in data['instances']}
    for index in active:
        material=data['materials'][index]
        assert all(t==0xffffffff for t in material['textures'])
        assert material['alpha_cutoff']==0 and material['flags']==0
        value=(np.asarray(material['base'][:3],dtype=np.float32)*np.float32(1-material['metallic'])).tolist()
        result[f'material_{index}']={'type':'diffuse','reflectance':{'type':'rgb','value':value}}
    for instance in data['instances']:
        shape=result[f"instance_{instance['slot']}"]
        if 'emitter' in shape:
            shape['emitter']['radiance']={'type':'rgb','value':data['materials'][instance['material']]['emissive']}
    return result


def validate(args):
    ref=load_reference(args.reference_script);root,data=ref.snapshot(args.snapshot)
    protocol=json.loads(pathlib.Path(args.protocol).read_text())
    assert ref.scene_digest(root)==protocol['snapshot_sha256']
    out=pathlib.Path(args.output);out.mkdir(parents=True,exist_ok=True)
    mi,sd,diffs=ref.build_scene(root,data,'diffuse',False);assert not diffs
    import drjit as dr
    dr.set_thread_count(2)
    scene=mi.load_dict(sd);native=mi.load_dict(native_constants(sd,data));sensor=scene.sensors()[0]
    triangles,slots,uvs=geometry(root,data);camera=data['camera'];w,h=camera['width'],camera['height']
    labels=np.full((h,w),-1);points=np.zeros((h,w,3));errors=np.zeros((h,w,4));ties=[];stats={'pixels':w*h,'hit_mismatches':0,'max_ray_direction_error':0.,'max_ray_origin_error':0.,'max_hit_position_error':0.,'max_uv_error':0.,'max_normal_error':0.,'max_constant_bsdf_error':0.}
    for y in range(h):
        for x in range(w):
            origin,direction=ray(camera,x+.5,y+.5)
            mr,_=sensor.sample_ray(0,0,[(x+.5)/w,(y+.5)/h],[0,0]);si=scene.ray_intersect(mr)
            stats['max_ray_direction_error']=max(stats['max_ray_direction_error'],float(np.max(np.abs(np.asarray(mr.d)-direction))))
            stats['max_ray_origin_error']=max(stats['max_ray_origin_error'],float(np.max(np.abs(np.asarray(mr.o)-origin))))
            hit=intersect(triangles,origin,direction)
            if bool(hit)!=bool(si.is_valid()):stats['hit_mismatches']+=1;continue
            if not hit:continue
            tri,t,bary=hit;slot=int(slots[tri]);normal=np.cross(triangles[tri,1]-triangles[tri,0],triangles[tri,2]-triangles[tri,0]);normal/=np.linalg.norm(normal)
            if si.shape.id()!=f'instance_{slot}':
                stats['hit_mismatches']+=1;ties.append({'pixel':[x,y],'cpu_slot':slot,'mitsuba_shape':si.shape.id()})
            position=origin+t*direction
            stats['max_hit_position_error']=max(stats['max_hit_position_error'],float(np.max(np.abs(position-np.asarray(si.p)))))
            stats['max_uv_error']=max(stats['max_uv_error'],float(np.max(np.abs(bary@uvs[tri]-np.asarray(si.uv)))))
            stats['max_normal_error']=max(stats['max_normal_error'],float(np.max(np.abs(normal-np.asarray(si.n)))))
            errors[y,x]=[si.shape.id()!=f'instance_{slot}',np.max(np.abs(position-np.asarray(si.p))),np.max(np.abs(bary@uvs[tri]-np.asarray(si.uv))),np.max(np.abs(normal-np.asarray(si.n)))]
            if np.dot(normal,-direction)>0:labels[y,x]=slot;points[y,x]=position
            ni=native.ray_intersect(mr);context=mi.BSDFContext();wo=mi.Vector3f(.2,.3,math.sqrt(.87))
            a=np.asarray(si.bsdf().eval(context,si,wo));b=np.asarray(ni.bsdf().eval(context,ni,wo))
            stats['max_constant_bsdf_error']=max(stats['max_constant_bsdf_error'],float(np.max(np.abs(a-b))))
    masks={name:erode(labels==slot,protocol['regions']['erode_pixels']) for name,slot in protocol['regions']['slots'].items()}
    masks['interior']=np.logical_or.reduce(list(masks.values()))
    light_triangles=triangles[slots==5];areas=np.linalg.norm(np.cross(light_triangles[:,1]-light_triangles[:,0],light_triangles[:,2]-light_triangles[:,0]),axis=1)/2
    centre=np.mean(light_triangles.reshape(-1,3),axis=0);boxes=triangles[np.isin(slots,[64,65])]
    leak=np.zeros((h,w),bool)
    for y,x in np.argwhere(masks['floor']):
        origin=points[y,x]+np.array([0,1e-5,0]);d=centre-origin;distance=np.linalg.norm(d);d/=distance
        leak[y,x]=intersect(boxes,origin,d,distance-1e-5) is not None
    masks['occluded_floor']=leak
    counts={name:int(mask.sum()) for name,mask in masks.items()}
    np.savez_compressed(out/'masks.npz',**masks)
    np.save(out/'primary_labels.npy',labels)
    stats.update({'edge_ties':ties,'roi_geometry_max_errors':errors[masks['interior']].max(axis=0).tolist(),'leak_roi_sufficient':counts['occluded_floor']>=16,'region_pixels':counts,'emitter_count':len(scene.emitters()),'emitter_area_m2':float(areas.sum()),'declared_Le':data['materials'][8]['emissive'],'snapshot_sha256':ref.scene_digest(root),'reference_script_sha256':hashlib.sha256(pathlib.Path(args.reference_script).read_bytes()).hexdigest()})
    stats['passed']=errors[masks['interior'],0].max()==0 and stats['max_ray_direction_error']<1e-6 and stats['max_ray_origin_error']<1e-6 and errors[masks['interior'],1].max()<1e-4 and errors[masks['interior'],2].max()<1e-5 and errors[masks['interior'],3].max()<1e-6 and stats['max_constant_bsdf_error']<1e-7 and len(scene.emitters())==1 and abs(areas.sum()-.8)<1e-6 and all(counts[name]>=16 for name in protocol['regions']['slots'])
    (out/'geometry_validation.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2))
    return 0 if stats['passed'] else 1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference-script',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    return validate(p.parse_args())
if __name__=='__main__':raise SystemExit(main())
