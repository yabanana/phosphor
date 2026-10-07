#!/usr/bin/env python3
"""F12 independent CPU offline reference. WRITTEN, NOT EXECUTED in delivery.

Consumes exportOfflineReference's exact same-frame snapshot. Requires an already
installed Mitsuba3 scalar_rgb and NumPy; never installs/downloads dependencies.
PFM/EXR are Float32 LINEAR, no gamma/exposure/tone mapping. Diffuse transport is
the exact F12 secondary-material reduction; principled is an explicitly different
full-material oracle and cannot certify the engine GGX without adapter validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

INVALID = 0xFFFFFFFF


def np_module():
    import numpy as np
    return np


def read_pfm(path):
    np = np_module()
    with pathlib.Path(path).open("rb") as stream:
        if stream.readline().strip() != b"PF":
            raise ValueError(f"{path}: expected RGB Float32 PFM")
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        if width < 1 or height < 1 or scale == 0:
            raise ValueError("invalid PFM header")
        raw = stream.read()
    if len(raw) != width * height * 12:
        raise ValueError(f"{path}: truncated or padded PFM")
    dtype = "<f4" if scale < 0 else ">f4"
    image = np.frombuffer(raw, dtype=dtype).reshape(height, width, 3)[::-1].astype(np.float32)
    image *= abs(scale)
    if not np.isfinite(image).all():
        raise ValueError(f"{path}: nonfinite linear pixels")
    return image


def write_pfm(path, image):
    np = np_module()
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError("invalid RGB linear image")
    with pathlib.Path(path).open("wb") as stream:
        stream.write(f"PF\n{image.shape[1]} {image.shape[0]}\n-1.0\n".encode())
        stream.write(image[::-1].astype("<f4").tobytes())


def snapshot(path):
    root = pathlib.Path(path).resolve()
    data = json.loads((root / "scene.json").read_text())
    if data.get("schema") != 1 or data.get("linear") is not True:
        raise ValueError("unsupported or non-linear F12 snapshot")
    # Imported names are data, not instructions; reject traversal/symlink escapes.
    for name in data["meshes"] + [t[k] for t in data["textures"] for k in ("rgb", "alpha")]:
        if not (root / name).resolve().is_relative_to(root):
            raise ValueError("snapshot resource escaped its directory")
        if not (root / name).is_file():
            raise ValueError(f"missing exact snapshot resource {name}")
    return root, data


def scene_digest(root):
    digest = hashlib.sha256()
    for file in sorted(p for p in root.iterdir() if p.is_file()):
        if file.name == "scene.json" or file.suffix in (".pfm", ".ply"):
            digest.update(file.name.encode())
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def build_scene(root, data, material_model, allow_differences):
    import mitsuba as mi
    mi.set_variant("scalar_rgb")  # Independent CPU path, never the Phosphor GPU.
    np = np_module()
    Transform = getattr(mi, "ScalarAffineTransform4f", None) or getattr(mi, "ScalarTransform4f")
    textures = {t["id"]: (read_pfm(root / t["rgb"]), read_pfm(root / t["alpha"])) for t in data["textures"]}
    differences = []
    if material_model == "principled":
        differences.append("Mitsuba Disney principled differs from engine correlated-GGX/Schlick")
    if any(l["type"] != 0 and l["range"] > 0 for l in data["lights"]):
        differences.append("Mitsuba punctual emitter has no engine finite-range quartic attenuation")
    if any(l["type"] == 2 for l in data["lights"]):
        differences.append("Mitsuba spotlight angular interpolation differs from engine smoothstep")
    if any(l["range"] > 0 for l in data["sampled_lights"]):
        differences.append("Mitsuba area/punctual emitter does not use sampled-light range window")
    if any(m["flags"] & 1 and any(e > 0 for e in m["emissive"]) for m in data["materials"]):
        differences.append("Mitsuba area emission is one-sided; engine double-sided emissive must be adapted")
    if differences and not allow_differences:
        raise ValueError("reference model differences require explicit experimental override: " + "; ".join(differences))

    def bilinear(texture_id, uv, fallback, alpha=False):
        if texture_id == INVALID:
            return np.asarray(fallback, dtype=np.float32)
        image = textures[texture_id][1 if alpha else 0]
        h, w = image.shape[:2]
        x, y = float(uv[0]) * w - 0.5, float(uv[1]) * h - 0.5
        ix, iy = math.floor(x), math.floor(y)
        fx, fy = x - ix, y - iy
        value = ((image[iy % h, ix % w] * (1-fx) + image[iy % h, (ix+1) % w] * fx) * (1-fy) +
                 (image[(iy+1) % h, ix % w] * (1-fx) + image[(iy+1) % h, (ix+1) % w] * fx) * fy)
        # Shader converts the FILTERED sample to half before factor multiplication.
        return value.astype(np.float16).astype(np.float32)

    class SnapshotTexture(mi.Texture):
        def __init__(self, props):
            super().__init__(props)
            self.material = data["materials"][int(props["material"])]
            self.semantic = str(props["semantic"])

        def values(self, si):
            m, uv = self.material, si.uv
            base = np.asarray(m["base"][:3], dtype=np.float32) * bilinear(m["textures"][0], uv, [1,1,1])
            mr = bilinear(m["textures"][2], uv, [1,1,1])
            metallic = min(1.0, max(0.0, m["metallic"] * float(mr[2])))
            if self.semantic == "base": return base
            if self.semantic == "diffuse": return base * (1-metallic)
            if self.semantic == "metallic": return np.full(3, metallic)
            if self.semantic == "roughness": return np.full(3, min(1.0, max(0.04, m["roughness"] * float(mr[1]))))
            if self.semantic == "emissive":
                return np.asarray(m["emissive"], dtype=np.float32) * bilinear(m["textures"][4], uv, [1,1,1])
            if self.semantic == "mask":
                alpha = m["base"][3] * float(bilinear(m["textures"][0], uv, [1,1,1], alpha=True)[0])
                return np.full(3, float(alpha >= m["alpha_cutoff"]))
            if self.semantic == "normal":
                normal = bilinear(m["textures"][1], uv, [0.5,0.5,1]) * 2-1
                normal[:2] *= m["normal_scale"]
                return normal * 0.5+0.5
            raise ValueError(self.semantic)

        def eval(self, si, active=True): return mi.Color3f(self.values(si))
        def eval_3(self, si, active=True): return mi.Color3f(self.values(si))
        def eval_1(self, si, active=True): return float(self.values(si)[0])
        def mean(self): return 0.5  # Only an emitter/BSDF sampling heuristic, never radiance.
        def is_spatially_varying(self): return True
        def to_string(self): return f"PhosphorSnapshotTexture[{self.semantic}]"

    mi.register_texture("phosphor_snapshot", lambda props: SnapshotTexture(props))

    def tex(material_id, semantic):
        return {"type": "phosphor_snapshot", "material": material_id, "semantic": semantic}

    camera = data["camera"]
    position = np.asarray(camera["position"])
    scene = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 8, "rr_depth": 5},
        "sensor": {
            "type": "perspective", "fov": math.degrees(camera["fov_y_radians"]), "fov_axis": "y",
            "near_clip": camera["near"], "far_clip": 1e30,
            "principal_point_offset_x": -camera.get("jitter_pixels",[0,0])[0]/camera["width"],
            "principal_point_offset_y": -camera.get("jitter_pixels",[0,0])[1]/camera["height"],
            "to_world": Transform().look_at(origin=position.tolist(),
                target=(position+np.asarray(camera["direction"])).tolist(), up=camera["up"]),
            "sampler": {"type": "independent", "sample_count": 64},
            "film": {"type": "hdrfilm", "width": camera["width"], "height": camera["height"],
                     "pixel_format": "rgb", "component_format": "float32",
                     "rfilter": {"type": "box"}},
        },
    }
    for material_id, material in enumerate(data["materials"]):
        if material_model == "diffuse":
            bsdf = {"type": "diffuse", "reflectance": tex(material_id, "diffuse")}
        else:
            bsdf = {"type": "principled", "base_color": tex(material_id, "base"),
                    "metallic": tex(material_id, "metallic"), "roughness": tex(material_id, "roughness"),
                    "specular": 0.5, "spec_trans": 0.0, "clearcoat": 0.0, "sheen": 0.0}
            if material["textures"][1] != INVALID:
                bsdf = {"type": "normalmap", "normalmap": tex(material_id, "normal"), "nested": bsdf}
        if material["flags"] & 1:
            bsdf = {"type": "twosided", "nested": bsdf}
        if material["alpha_cutoff"] > 0:
            bsdf = {"type": "mask", "opacity": tex(material_id, "mask"), "nested": bsdf}
        scene[f"material_{material_id}"] = bsdf
    for instance in data["instances"]:
        material_id = instance["material"]
        material = data["materials"][material_id]
        shape = {"type": "ply", "filename": str(root/data["meshes"][instance["mesh"]]),
                 "to_world": Transform(np.asarray(instance["world"]).reshape(4,4,order="F")),
                 "face_normals": material_model == "diffuse",
                 "bsdf": {"type": "ref", "id": f"material_{material_id}"}}
        if any(value > 0 for value in material["emissive"]):
            shape["emitter"] = {"type": "area", "radiance": tex(material_id, "emissive")}
        scene[f"instance_{instance['slot']}"] = shape
    # Analytic suns are not duplicated by the F11 sampled list. When the sampled
    # list exists it owns punctual/area emitters, as in the GI shader bridge.
    for index, light in enumerate(data["lights"]):
        if data["sampled_lights"] and light["type"] != 0:
            continue
        power = (np.asarray(light["color"])*light["intensity"]).tolist()
        if light["type"] == 0:
            angular_radius = data.get("sun_angular_radius",0.0)
            if angular_radius > 0:
                # Independently trace a distant physical disk with the same
                # angular radius and perpendicular irradiance. Finite-distance
                # approximation (1e6m) requires the tester's units/penumbra check.
                d = np.asarray(light["direction"]);d /= np.linalg.norm(d)
                helper = np.asarray([0,0,1] if abs(d[2]) < 0.99 else [0,1,0])
                u = np.cross(helper,d);u /= np.linalg.norm(u);v = np.cross(d,u)
                distance = 1e6;radius = distance*math.tan(angular_radius)
                matrix = np.eye(4);matrix[:3,0]=u*radius;matrix[:3,1]=v*radius
                matrix[:3,2]=d;matrix[:3,3]=-d*distance
                Le = (np.asarray(power)/(math.pi*math.sin(angular_radius)**2)).tolist()
                emitter = {"type":"disk","to_world":Transform(matrix),
                           "bsdf":{"type":"diffuse","reflectance":{"type":"rgb","value":[0,0,0]}},
                           "emitter":{"type":"area","radiance":{"type":"rgb","value":Le}}}
            else:
                emitter = {"type": "directional", "direction": light["direction"],
                           "irradiance": {"type": "rgb", "value": power}}
        elif light["type"] == 1:
            emitter = {"type": "point", "position": light["position"], "intensity": {"type": "rgb", "value": power}}
        else:
            p = np.asarray(light["position"])
            d = np.asarray(light["direction"])
            up = [0,1,0] if abs(d[1]) < 0.99 else [1,0,0]
            emitter = {"type": "spot", "to_world": Transform().look_at(origin=p.tolist(),target=(p+d).tolist(),up=up),
                       "intensity": {"type": "rgb", "value": power}, "cutoff_angle": math.degrees(light["outer"]),
                       "beam_width": math.degrees(light["inner"])}
        scene[f"analytic_light_{index}"] = emitter
    for index, light in enumerate(data["sampled_lights"]):
        t = light["type"]
        if t == 6:
            # Mesh emissive exists already, with exact textures; sampling list
            # triangles are distributions over THAT geometry, never extra lights.
            continue
        if t in (1,2):
            p = np.asarray(light["position"]);d = np.asarray(light["u"])
            if t == 1:
                item = {"type": "point", "position": p.tolist(), "intensity": {"type": "rgb","value": light["emission"]}}
            else:
                up = [0,1,0] if abs(d[1]/np.linalg.norm(d)) < 0.99 else [1,0,0]
                item = {"type": "spot", "to_world": Transform().look_at(origin=p.tolist(),target=(p+d).tolist(),up=up),
                        "intensity": {"type":"rgb","value":light["emission"]},
                        "cutoff_angle":math.degrees(light["outer"]), "beam_width":math.degrees(light["inner"])}
        else:
            p = np.asarray(light["position"]);u = np.asarray(light["u"]);v = np.asarray(light["v"])
            if t == 5:
                item = {"type": "cylinder", "p0": (p-u).tolist(), "p1": (p+u).tolist(), "radius": light["radius"]}
            else:
                normal = np.cross(u,v);normal /= np.linalg.norm(normal)
                if t == 4:
                    u = u*light["radius"];v = v*light["radius"]
                matrix = np.eye(4);matrix[:3,0]=u;matrix[:3,1]=v;matrix[:3,2]=normal;matrix[:3,3]=p
                item = {"type": "rectangle" if t == 3 else "disk", "to_world": Transform(matrix)}
            item["emitter"] = {"type": "area", "radiance": {"type":"rgb","value":light["emission"]}}
            item["bsdf"] = {"type": "diffuse", "reflectance": {"type":"rgb","value":[0,0,0]}}
        scene[f"sampled_light_{index}"] = item
    if any(v > 0 for v in data["sky"]):
        scene["sky"] = {"type": "constant", "radiance": {"type":"rgb","value":data["sky"]}}
    return mi, scene, differences


def metric(reference, candidate, region=None):
    np = np_module()
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate dimensions differ; automatic rescaling is forbidden")
    if region:
        x,y,w,h = region
        if min(x,y) < 0 or min(w,h) < 1 or x+w > reference.shape[1] or y+h > reference.shape[0]:
            raise ValueError("comparison region outside linear images")
        reference,candidate = reference[y:y+h,x:x+w],candidate[y:y+h,x:x+w]
    delta = candidate-reference
    scale = max(float(np.sqrt(np.mean(reference*reference))),1e-6)
    luminance = np.asarray([0.2126,0.7152,0.0722])
    mean_reference = float(np.mean(reference@luminance))
    return {"rmse": float(np.sqrt(np.mean(delta*delta))),
            "relative_rmse": float(np.sqrt(np.mean(delta*delta)))/scale,
            "relative_signed_bias": float(np.mean(delta@luminance))/max(abs(mean_reference),1e-6),
            "mean_reference": mean_reference, "mean_candidate": float(np.mean(candidate@luminance)),
            "maximum_abs_error":float(np.max(np.abs(delta)))}


def render(args):
    np = np_module()
    root,data = snapshot(args.snapshot)
    mi,scene_dict,differences = build_scene(root,data,args.material_model,args.allow_model_differences)
    out = pathlib.Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    previous = None
    records = []
    for spp in args.spp:
        scene_dict["integrator"]["max_depth"] = args.max_depth
        scene = mi.load_dict(scene_dict)
        full = np.asarray(mi.render(scene,spp=spp,seed=args.seed),dtype=np.float32)
        if args.signal == "indirect":
            scene_dict["integrator"]["max_depth"] = 2
            direct_scene = mi.load_dict(scene_dict)
            direct = np.asarray(mi.render(direct_scene,spp=spp,seed=args.seed),dtype=np.float32)
            image = full-direct  # signed Monte Carlo estimate; never clamp to hide noise
        else:
            image = full
        write_pfm(out/f"reference_{spp}.pfm",image)
        mi.Bitmap(image).convert(mi.Bitmap.PixelFormat.RGB,mi.Struct.Type.Float32,False).write(str(out/f"reference_{spp}.exr"))
        record = {"spp":spp,"linear_pfm":f"reference_{spp}.pfm","linear_exr":f"reference_{spp}.exr"}
        if previous is not None:
            record["previous_difference"] = metric(image,previous)
        records.append(record);previous=image
    convergence = records[-1].get("previous_difference",{}).get("relative_rmse")
    converged = convergence is not None and convergence <= args.max_convergence_rmse
    report = {"schema":1,"state":"REFERENCE_CANDIDATE_CONVERGED" if converged and not differences else "NON_ACCEPTED_REFERENCE",
              "snapshot_sha256":scene_digest(root),"renderer":"Mitsuba3","renderer_version":mi.__version__,
              "variant":"scalar_rgb","seed":args.seed,"material_model":args.material_model,"signal":args.signal,
              "max_depth":args.max_depth,"model_differences":differences,"records":records,
              "threshold_relative_convergence_rmse":args.max_convergence_rmse,"converged":converged,
              "validation":"renderer/plugin/UV/units implementation still requires tester validation"}
    (out/"reference_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0 if converged and not differences else 1


def compare(args):
    reference,candidate = read_pfm(args.reference),read_pfm(args.candidate)
    manifest = json.loads(pathlib.Path(args.regions).read_text())
    if not manifest.get("regions"):
        raise ValueError("named frozen regions required; global RMSE alone does not certify leak/bias")
    regions = {r["name"]:metric(reference,candidate,r["xywh"]) for r in manifest["regions"]}
    accepted = all(r["relative_rmse"] <= args.max_relative_rmse and
                   abs(r["relative_signed_bias"]) <= args.max_relative_bias for r in regions.values())
    report = {"schema":1,"passed":accepted,"regions":regions,"global":metric(reference,candidate),
              "thresholds":{"relative_rmse":args.max_relative_rmse,"relative_bias":args.max_relative_bias},
              "state":"QUALITY_CHECK_ONLY_NOT_PHASE_ACCEPTANCE"}
    pathlib.Path(args.report).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0 if accepted else 1


def plan(args):
    # This freezes the required corpus. It renders nothing, invents no reference,
    # and requires actual engine exports for every listed state.
    states = [{"name":"cornell_diffuse","frames":[0]},
              {"name":"thin_walls","frames":[0],"negative":"disabled distance moments must expose leak"},
              {"name":"probe_inside_wall","frames":[0,16,64],"negative":"disabled classification/relocation"},
              {"name":"moving_sun","frames":[0,1,8,32,128],"negative":"stale light revision"},
              {"name":"moving_emissive","frames":[0,1,8,32,128],"negative":"stale material/transform revision"},
              {"name":"cache_collision","frames":[0,1,8],"negative":"omit full-key comparison"},
              {"name":"disocclusion","frames":[0,1,8,32],"negative":"inject foreign view history"}]
    record = {"schema":1,"state":"NOT_EXECUTED","source":"same-frame engine export",
              "units":"metres","images":"Float32 linear PFM/EXR","signal":"indirect diffuse reflected radiance",
              "spp":[64,256,1024,4096],"preset":"EXPERIMENTAL_UNMEASURED",
              "convergence_max_relative_rmse":args.max_convergence_rmse,"scenarios":states}
    pathlib.Path(args.output).write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("render");p.add_argument("snapshot");p.add_argument("--output",required=True)
    p.add_argument("--spp",type=int,nargs="+",default=[64,256,1024,4096]);p.add_argument("--seed",type=int,default=1234)
    p.add_argument("--max-depth",type=int,default=8);p.add_argument("--signal",choices=["indirect","total"],default="indirect")
    p.add_argument("--material-model",choices=["diffuse","principled"],default="diffuse")
    p.add_argument("--allow-model-differences",action="store_true")
    p.add_argument("--max-convergence-rmse",type=float,default=0.02);p.set_defaults(function=render)
    p = sub.add_parser("compare");p.add_argument("reference");p.add_argument("candidate");p.add_argument("--regions",required=True)
    p.add_argument("--report",required=True);p.add_argument("--max-relative-rmse",type=float,required=True)
    p.add_argument("--max-relative-bias",type=float,required=True);p.set_defaults(function=compare)
    p = sub.add_parser("plan");p.add_argument("--output",required=True)
    p.add_argument("--max-convergence-rmse",type=float,default=0.02);p.set_defaults(function=plan)
    args = parser.parse_args(argv)
    if hasattr(args,"spp") and (len(args.spp) < 2 or any(n < 1 for n in args.spp) or args.spp != sorted(set(args.spp))):
        parser.error("at least two strictly increasing positive SPP checkpoints required")
    if hasattr(args,"max_depth") and args.max_depth < 3:
        parser.error("indirect reference needs max_depth >=3")
    for field in ("max_convergence_rmse","max_relative_rmse","max_relative_bias"):
        if hasattr(args,field) and (not math.isfinite(getattr(args,field)) or getattr(args,field) < 0):
            parser.error(f"{field} must be finite and nonnegative")
    try:
        return args.function(args)
    except (ImportError,ValueError,RuntimeError,OSError) as error:
        print(f"F12 reference failure: {error}",file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
