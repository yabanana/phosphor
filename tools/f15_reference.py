#!/usr/bin/env python3
"""F15.3 independent reference of the glTF 2.0 Appendix B base model.

Written from the specification (Khronos glTF 2.0, Appendix B "BRDF
Implementation"), not translated from the MSL.  It consumes the snapshot of
`exportOfflineReference` (geometry, world matrices, camera, sun, environment,
material factors) of the material-validation bench (`--material-scenario`) and
never reads a renderer image to choose rays, samples or masks.  Mitsuba 3 is
used only to cast pixel-centre rays on the snapshot meshes (hit position,
interpolated shading normal, shape); the BSDF and every integral are this
file's own.

Model (Appendix B):
  dielectric = mix(diffuse(baseColor), specular, F(f0 = 0.04))        [ior 1.5]
  metal      = specular * F(f0 = baseColor)
  material   = mix(dielectric, metal, metallic)
  specular   = D_GGX * Vis_SmithCorrelated, alpha = roughness^2
  F          = Schlick, f0 + (1 - f0) (1 - |V.H|)^5
  diffuse    = baseColor / pi

Declared engine conventions (every item is part of the JSON report):
- roughness clamped to [0.04, 1], alpha = max(roughness^2, 0.002), N.V >= 1e-4.
- Sun: a disk of uniform radiance and angular radius r (the snapshot's
  sun_angular_radius).  Diffuse is evaluated at the disk centre; the specular
  term is the irradiance-normalised mean of f_s cos over the disk with the
  engine's declared rule (`sunDiskSpecular` in shaders/material_shading.h): a
  point evaluation when the lobe is wide, else a deterministic uniform
  solid-angle golden-angle spiral of 32 samples (128 when alpha < 2r) at the
  pixel centre.  `narrow` pixels are reported as the quadrature mask.
- Environment: one constant isotropic radiance L (the snapshot's sky).  The
  outgoing radiance is the exact numerical quadrature over the hemisphere of
  BSDF * cos * L, not a split-sum fit (half-vector parametrisation with the
  GGX CDF, midpoint rule; the table is a function of roughness and N.V only).
- Camera: one ray through each pixel centre, no reconstruction filter.
- Shadows: none (the scenarios declare `--shadows off`), no inter-reflection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from f12_reference import np_module, scene_digest, snapshot, write_pfm  # noqa: E402

np = np_module()

MIN_ROUGHNESS, MIN_ALPHA, MIN_NDOTV = 0.04, 0.002, 1e-4
DIELECTRIC_IOR = 1.5
DIELECTRIC_F0 = ((1 - DIELECTRIC_IOR) / (1 + DIELECTRIC_IOR)) ** 2  # 0.04
GOLDEN_ANGLE = 2.39996323  # the engine's constant (float32 literal)
INVALID = 0xFFFFFFFF

MODEL = {
    "base": "glTF 2.0 Appendix B: mix(mix(diffuse, specular, F(0.04)), specular * F(baseColor), metallic)",
    "specular": "GGX D, height-correlated Smith visibility, alpha = max(roughness^2, 0.002), roughness in [0.04, 1]",
    "fresnel": "Schlick on |V.H|",
    "sun": "disk centre for diffuse; specular = mean over a point or a deterministic 32/128-sample golden-angle "
           "uniform-solid-angle spiral (engine sunDiskSpecular rule); narrow pixels form the quadrature mask",
    "environment": "exact hemisphere quadrature of BSDF * cos * L_env (constant isotropic), not a split-sum fit",
    "camera": "pixel-centre rays, no reconstruction filter",
    "shadows": "none",
}

# The declared model plus the deliberately wrong variants used as negative controls.
APPENDIX_B = {"alpha": "squared", "fresnel": "schlick", "weighting": "appendix-b"}
CONTROLS = {
    "alpha-roughness": {**APPENDIX_B, "alpha": "linear"},          # alpha = roughness instead of roughness^2
    "fresnel-off": {**APPENDIX_B, "fresnel": "off"},               # F = f0 for every angle
    "diffuse-unweighted": {**APPENDIX_B, "weighting": "unweighted-diffuse"},  # diffuse not scaled by (1 - F)
    "engine-f0-blend": {**APPENDIX_B, "weighting": "engine-blend"},  # F of the metallic-blended f0 weights diffuse
}


# --- BSDF -----------------------------------------------------------------

def clamp_roughness(roughness):
    return np.clip(np.asarray(roughness, dtype=np.float64), MIN_ROUGHNESS, 1.0)


def effective_alpha(roughness, model=APPENDIX_B):
    r = clamp_roughness(roughness)
    return np.maximum(r * r if model["alpha"] == "squared" else r, MIN_ALPHA)


def ggx_d(n_dot_h, alpha):
    a2 = alpha * alpha
    d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
    return a2 / (math.pi * d * d)


def smith_vis(n_dot_v, n_dot_l, alpha):
    """Appendix B Vis = 1 / (2 (|NV| sqrt(a^2 + (1-a^2) NL^2) + |NL| sqrt(a^2 + (1-a^2) NV^2)))."""
    a2 = alpha * alpha
    denominator = 2.0 * (np.abs(n_dot_v) * np.sqrt(a2 + (1 - a2) * n_dot_l * n_dot_l) +
                         np.abs(n_dot_l) * np.sqrt(a2 + (1 - a2) * n_dot_v * n_dot_v))
    return 1.0 / np.maximum(denominator, 1e-12)


def schlick_weight(v_dot_h, model=APPENDIX_B):
    if model["fresnel"] == "off":
        return np.zeros_like(np.asarray(v_dot_h, dtype=np.float64))
    return (1.0 - np.clip(v_dot_h, 0.0, 1.0)) ** 5


def dot(a, b):
    return np.sum(a * b, axis=-1)


def normalize(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-300)


def lobe_weights(base, metallic, v_dot_h, model=APPENDIX_B):
    """(diffuse multiplier per channel, specular multiplier per channel) at one V.H.

    f = diffuse_multiplier * base/pi-style term + specular_multiplier * D * Vis; the
    diffuse multiplier already contains (1 - metallic) and the Fresnel weighting.
    """
    w = schlick_weight(v_dot_h, model)[..., None]
    m = np.asarray(metallic, dtype=np.float64)[..., None]
    f_dielectric = DIELECTRIC_F0 + (1 - DIELECTRIC_F0) * w
    f_metal = base + (1 - base) * w
    if model["weighting"] == "engine-blend":
        f0 = DIELECTRIC_F0 * (1 - m) + base * m
        f = f0 + (1 - f0) * w
        return (1 - f) * (1 - m), f
    diffuse = (1 - m) * (1.0 if model["weighting"] == "unweighted-diffuse" else (1 - f_dielectric))
    specular = (1 - m) * f_dielectric + m * f_metal
    return diffuse, specular


def bsdf(base, metallic, roughness, n, v, l, model=APPENDIX_B):
    """BSDF value f (without the cosine) for world unit vectors; (n, 3) arrays or broadcastable."""
    alpha = effective_alpha(roughness, model)
    n_dot_l = dot(n, l)
    n_dot_v = np.maximum(dot(n, v), MIN_NDOTV)
    h = normalize(v + l)
    n_dot_h = np.clip(dot(n, h), 0.0, 1.0)
    v_dot_h = dot(v, h)
    diffuse_w, specular_w = lobe_weights(base, metallic, v_dot_h, model)
    spec = (ggx_d(n_dot_h, alpha) * smith_vis(n_dot_v, np.maximum(n_dot_l, 0.0), alpha))[..., None]
    value = diffuse_w * base / math.pi + specular_w * spec
    return np.where((n_dot_l > 0)[..., None], value, 0.0)


# --- sun disk ---------------------------------------------------------------

def sun_quadrature(n, v, l, roughness, radius, model=APPENDIX_B):
    """Per-pixel sample count and `narrow` flag of the engine's declared disk rule."""
    alpha = effective_alpha(roughness, model)
    n_dot_v = dot(n, v)
    reflected = 2.0 * n_dot_v[..., None] * n - v
    separation = np.arccos(np.clip(dot(reflected, l), -1.0, 1.0))
    narrow = (alpha < 0.05) & (separation < 4.0 * radius + 16.0 * alpha)
    count = np.where(~narrow, 1, np.where(alpha < 2.0 * radius, 128, 32))
    return count, narrow


def sun_disk_samples(l, radius, count):
    """Deterministic uniform solid-angle golden-angle spiral over the disk of centre l (unit)."""
    l = np.asarray(l, dtype=np.float64)
    if count == 1:
        return l[None, :]
    helper = np.array([0.0, 0.0, 1.0]) if abs(l[2]) < 0.999 else np.array([0.0, 1.0, 0.0])
    t = np.cross(helper, l)
    t /= np.linalg.norm(t)
    b = np.cross(l, t)
    k = np.arange(count, dtype=np.float64)
    z = 1.0 - (k + 0.5) / count * (1.0 - math.cos(radius))
    s = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = GOLDEN_ANGLE * k
    return t[None, :] * (s * np.cos(phi))[:, None] + b[None, :] * (s * np.sin(phi))[:, None] + l[None, :] * z[:, None]


def sun_radiance(base, metallic, roughness, n, v, toward_sun, irradiance, radius, model=APPENDIX_B):
    """Outgoing radiance per pixel from the sun disk; returns (radiance (n,3), narrow (n,))."""
    n_pixels = n.shape[0]
    l = np.asarray(toward_sun, dtype=np.float64)
    l = l / np.linalg.norm(l)
    count, narrow = sun_quadrature(n, v, l, roughness, radius, model)
    alpha = effective_alpha(roughness, model)
    n_dot_v = np.maximum(dot(n, v), MIN_NDOTV)
    # Diffuse: centre of the disk.
    n_dot_l = np.clip(dot(n, np.broadcast_to(l, n.shape)), 0.0, 1.0)
    h_c = normalize(v + l)
    diffuse_w, _ = lobe_weights(base, metallic, dot(v, h_c), model)
    total = np.where((n_dot_l > 0)[..., None], diffuse_w * base / math.pi * n_dot_l[..., None], 0.0)
    # Specular: mean of f_spec cos over the samples (count = 1 reduces to the point evaluation).
    spec_sum = np.zeros((n_pixels, 3))
    for c in np.unique(count):
        sel = np.flatnonzero(count == c)
        samples = sun_disk_samples(l, radius, int(c))
        for sample in samples:
            lk = np.broadcast_to(sample, (sel.size, 3))
            nl = np.clip(dot(n[sel], lk), 0.0, 1.0)
            h = normalize(v[sel] + lk)
            nh = np.clip(dot(n[sel], h), 0.0, 1.0)
            _, specular_w = lobe_weights(base[sel], metallic[sel], dot(v[sel], h), model)
            term = (ggx_d(nh, alpha[sel]) * smith_vis(n_dot_v[sel], nl, alpha[sel]) * nl)[:, None] * specular_w
            spec_sum[sel] += np.where((nl > 0)[:, None], term, 0.0) / int(c)
    return (total + spec_sum) * np.asarray(irradiance, dtype=np.float64), narrow


# --- environment ------------------------------------------------------------

_TABLES = {}
MU_NODES = 257
U_NODES, PHI_NODES = 512, 256
DIFFUSE_MU_NODES, DIFFUSE_PHI_NODES = 128, 128


def _diffuse_integral(mu, weight_of_vh):
    """integral over wi of weight(V.H) * cos(theta_i) / pi for wo at cos(theta_o) = mu."""
    x, wgt = np.polynomial.legendre.leggauss(DIFFUSE_MU_NODES)
    cos_i = 0.5 * (x + 1.0)  # Gauss-Legendre in cos(theta_i) over [0, 1]
    wgt = 0.5 * wgt
    phi = (np.arange(DIFFUSE_PHI_NODES) + 0.5) * (2 * math.pi / DIFFUSE_PHI_NODES)
    sin_i = np.sqrt(1 - cos_i ** 2)
    wo = np.array([math.sqrt(max(0.0, 1 - mu * mu)), 0.0, mu])
    wi_dot_wo = sin_i[:, None] * np.cos(phi)[None, :] * wo[0] + cos_i[:, None] * wo[2]
    vh = np.sqrt(np.clip((1 + wi_dot_wo) / 2, 0.0, 1.0))
    integrand = weight_of_vh(vh) * cos_i[:, None] / math.pi
    return float(np.sum(integrand.mean(axis=1) * 2 * math.pi * wgt))


def _spec_integrals(mu, alpha, model):
    """(A, B): integrals of D*Vis*cos*(1-w) and D*Vis*cos*w over wi, via the GGX CDF in half-vector space."""
    u = (np.arange(U_NODES) + 0.5) / U_NODES
    phi = (np.arange(PHI_NODES) + 0.5) * (2 * math.pi / PHI_NODES)
    cos_h = np.sqrt((1 - u) / (1 + (alpha * alpha - 1) * u))
    sin_h = np.sqrt(np.maximum(0.0, 1 - cos_h ** 2))
    wo = np.array([math.sqrt(max(0.0, 1 - mu * mu)), 0.0, mu])
    vh = sin_h[:, None] * np.cos(phi)[None, :] * wo[0] + cos_h[:, None] * wo[2]
    wi_z = 2 * vh * cos_h[:, None] - wo[2]  # z of wi = 2 (wo.h) h - wo
    valid = (wi_z > 0) & (vh > 0)
    # With h sampled from D cos(theta_h): integral = mean( Vis * cos_i * 4 (wo.h) / cos_h * g ).
    vis = smith_vis(mu, np.maximum(wi_z, 0.0), alpha)
    base = np.where(valid, vis * wi_z * 4 * vh / cos_h[:, None], 0.0)
    w = schlick_weight(vh, model)
    return float(np.mean(base * (1 - w))), float(np.mean(base * w))


def environment_table(roughness, model=APPENDIX_B):
    """Directional-albedo components over N.V for one roughness: dict of arrays on a uniform mu grid."""
    alpha = float(effective_alpha(roughness, model))
    key = (round(alpha, 12), model["fresnel"], model["alpha"] == "squared")
    if key not in _TABLES:
        mus = np.linspace(MIN_NDOTV, 1.0, MU_NODES)
        a = np.empty(MU_NODES)
        b = np.empty(MU_NODES)
        cd = np.empty(MU_NODES)
        cu = np.empty(MU_NODES)
        for i, mu in enumerate(mus):
            a[i], b[i] = _spec_integrals(float(mu), alpha, model)
            cd[i] = _diffuse_integral(float(mu), lambda vh: 1.0 - DIELECTRIC_F0 - (1 - DIELECTRIC_F0) * schlick_weight(vh, model))
            cu[i] = _diffuse_integral(float(mu), lambda vh: np.ones_like(vh))
        _TABLES[key] = {"mu": mus, "A": a, "B": b, "Cd": cd, "Cu": cu}
    return _TABLES[key]


def environment_radiance(base, metallic, roughness, n_dot_v, radiance, model=APPENDIX_B):
    """Outgoing radiance (n, 3) under a constant isotropic environment of radiance `radiance`."""
    n_pixels = len(n_dot_v)
    result = np.zeros((n_pixels, 3))
    mu = np.clip(n_dot_v, MIN_NDOTV, 1.0)
    for r in np.unique(np.asarray(roughness, dtype=np.float64)):
        sel = np.flatnonzero(np.asarray(roughness) == r)
        t = environment_table(r, model)
        a = np.interp(mu[sel], t["mu"], t["A"])[:, None]
        b = np.interp(mu[sel], t["mu"], t["B"])[:, None]
        cd = np.interp(mu[sel], t["mu"], t["Cd"])[:, None]
        cu = np.interp(mu[sel], t["mu"], t["Cu"])[:, None]
        m = np.asarray(metallic, dtype=np.float64)[sel, None]
        c = base[sel]
        if model["weighting"] == "engine-blend":
            f0 = DIELECTRIC_F0 * (1 - m) + c * m
            result[sel] = (1 - m) * c * cu + f0 * a + b
            continue
        diffuse = c * (cu if model["weighting"] == "unweighted-diffuse" else cd)
        dielectric = diffuse + DIELECTRIC_F0 * a + b
        metal = c * a + b
        result[sel] = (1 - m) * dielectric + m * metal
    return result * radiance


# --- ray casting ------------------------------------------------------------

def cast_primary_rays(root, data, threads):
    """Pixel-centre primary hits with Mitsuba: (valid, position, shading normal, instance index)."""
    import drjit as dr
    import mitsuba as mi

    mi.set_variant("llvm_ad_rgb")
    dr.set_flag(dr.JitFlag.FastMath, False)
    dr.set_thread_count(threads)
    transform = getattr(mi, "ScalarAffineTransform4f", None) or mi.ScalarTransform4f
    description = {"type": "scene"}
    for index, instance in enumerate(data["instances"]):
        description[f"instance_{index}"] = {
            "type": "ply", "filename": str(root / data["meshes"][instance["mesh"]]),
            "face_normals": False,
            "to_world": transform(np.asarray(instance["world"], dtype=np.float64).reshape(4, 4, order="F"))}
    scene = mi.load_dict(description)
    identifiers = np.asarray(dr.reinterpret_array(mi.UInt32, scene.shapes_dr()))
    index_of = {int(i): k for k, (shape, i) in enumerate(zip(scene.shapes(), identifiers))}
    names = [shape.id() for shape in scene.shapes()]
    camera = data["camera"]
    if any(camera.get("jitter_pixels", [0, 0])):
        raise ValueError("pixel-centre reference requires an unjittered camera")
    width, height = int(camera["width"]), int(camera["height"])
    forward = np.asarray(camera["direction"], dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(camera["up"], dtype=np.float64))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    scale = height / (2 * math.tan(camera["fov_y_radians"] / 2))
    pixel = np.arange(width * height)
    x = (pixel % width + 0.5 - width / 2) / scale
    y = (height / 2 - (pixel // width + 0.5)) / scale
    direction = forward[None, :] + right[None, :] * x[:, None] + up[None, :] * y[:, None]
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    ray = mi.Ray3f(mi.Point3f(*[float(c) for c in camera["position"]]),
                   mi.Vector3f(mi.Float(direction[:, 0].copy()), mi.Float(direction[:, 1].copy()),
                               mi.Float(direction[:, 2].copy())))
    si = scene.ray_intersect(ray)
    valid = np.asarray(si.is_valid())
    position = np.stack([np.asarray(si.p.x), np.asarray(si.p.y), np.asarray(si.p.z)], axis=1).astype(np.float64)
    normal = np.stack([np.asarray(si.sh_frame.n.x), np.asarray(si.sh_frame.n.y), np.asarray(si.sh_frame.n.z)],
                      axis=1).astype(np.float64)
    shape_id = np.asarray(dr.reinterpret_array(mi.UInt32, si.shape)).astype(np.int64)
    instance = np.full(width * height, -1, dtype=np.int64)
    for key, value in index_of.items():
        instance[valid & (shape_id == key)] = int(names[value].split("_")[1])
    valid &= instance >= 0
    return valid.reshape(height, width), position, normal, instance.reshape(height, width), (width, height)


# --- frame ------------------------------------------------------------------

def sphere_table(data):
    return [{"instance": k, "slot": inst["slot"], "base": data["materials"][inst["material"]]["base"][:3],
             "metallic": float(data["materials"][inst["material"]]["metallic"]),
             "roughness": float(data["materials"][inst["material"]]["roughness"])}
            for k, inst in enumerate(data["instances"])]


def require_declared_scene(data):
    for instance in data["instances"]:
        material = data["materials"][instance["material"]]
        if any(index != INVALID for index in material["textures"]):
            raise ValueError("instanced material samples a texture; the Appendix B reference is texture-free")
        if any(v > 0 for v in material["emissive"]) or material["alpha_cutoff"] > 0:
            raise ValueError("emissive or alpha-masked material is outside the declared model")
    if any(light["type"] != 0 for light in data["lights"]) or len(data["lights"]) > 1:
        raise ValueError("at most one directional light (the sun) is modelled")
    sky = data["sky"]
    if not (sky[0] == sky[1] == sky[2]):
        raise ValueError("the environment must be isotropic and grey")
    if data["sampled_lights"]:
        raise ValueError("sampled (area) lights are outside the declared model")


def render_frame(root, data, model=APPENDIX_B, threads=None, cast=None):
    """Reference radiance image (h, w, 3) float32 plus masks and the per-sphere table."""
    require_declared_scene(data)
    valid, position, normal, instance, (width, height) = cast or cast_primary_rays(
        root, data, threads or min(8, os.cpu_count() or 1))
    spheres = sphere_table(data)
    flat = valid.reshape(-1)
    inst = instance.reshape(-1)
    camera = np.asarray(data["camera"]["position"], dtype=np.float64)
    pick = np.flatnonzero(flat)
    n = normalize(normal[pick])
    v = normalize(camera[None, :] - position[pick])
    base = np.array([spheres[k]["base"] for k in inst[pick]], dtype=np.float64)
    metallic = np.clip(np.array([spheres[k]["metallic"] for k in inst[pick]]), 0.0, 1.0)
    roughness = np.array([spheres[k]["roughness"] for k in inst[pick]])
    radiance = np.zeros((pick.size, 3))
    narrow = np.zeros(pick.size, dtype=bool)
    lit = np.zeros(pick.size, dtype=bool)
    terms = []
    sky = float(data["sky"][0])
    sun_radius = float(data.get("sun_angular_radius", 0.0))
    for light in data["lights"]:
        toward = -normalize(np.asarray(light["direction"], dtype=np.float64))
        irradiance = np.asarray(light["color"], dtype=np.float64) * float(light["intensity"])
        if sun_radius <= 0:
            raise ValueError("a sun requires a positive angular radius")
        sun, narrow = sun_radiance(base, metallic, roughness, n, v, toward, irradiance, sun_radius, model)
        radiance += sun
        lit = dot(n, np.broadcast_to(toward, n.shape)) > 0
        terms.append("sun-disk")
    if sky > 0:
        radiance += environment_radiance(base, metallic, roughness, dot(n, v), sky, model)
        terms.append("environment-quadrature")
    image = np.full((height * width, 3), sky, dtype=np.float64)  # pixels without geometry show the environment
    image[pick] = radiance
    # ROI: sphere pixels whose 4-neighbours belong to the same sphere (silhouette excluded).
    sphere_id = instance.copy()
    same = np.ones((height, width), dtype=bool)
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        same &= np.roll(sphere_id, shift, axis=axis) == sphere_id
    same[0, :] = same[-1, :] = same[:, 0] = same[:, -1] = False
    roi = valid & same
    narrow_full = np.zeros(height * width, dtype=bool)
    narrow_full[pick] = narrow
    lit_full = np.zeros(height * width, dtype=bool)
    lit_full[pick] = lit
    masks = {"roi": roi, "sphere": sphere_id.astype(np.int32), "narrow": narrow_full.reshape(height, width),
             "lit": lit_full.reshape(height, width)}
    return image.reshape(height, width, 3).astype(np.float32), masks, spheres, terms


def sha(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command_render(args):
    root, data = snapshot(args.snapshot)
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cast = cast_primary_rays(root, data, args.threads)
    controls = {"": APPENDIX_B, **{name: CONTROLS[name] for name in args.controls}}
    report = {"schema": 1, "tool": "tools/f15_reference.py", "tool_sha256": sha(__file__), "model": MODEL,
              "snapshot_sha256": scene_digest(root), "frame": data.get("frame"),
              "sun_angular_radius": data.get("sun_angular_radius"), "sky": data["sky"],
              "lights": data["lights"], "controls": {}, "deterministic": True}
    for name, model in controls.items():
        image, masks, spheres, terms = render_frame(root, data, model, cast=cast)
        stem = "reference" if not name else f"control-{name}"
        write_pfm(out / f"{stem}.pfm", image)
        if not name:
            np.savez_compressed(out / "masks.npz", **masks)
            report.update(terms=terms, spheres=spheres, model_variant=model, width=image.shape[1],
                          height=image.shape[0], maximum_radiance=float(image.max()),
                          roi_pixels=int(masks["roi"].sum()), quadrature_mask_pixels=int(masks["narrow"].sum()))
        else:
            report["controls"][name] = {"model_variant": model, "file": f"{stem}.pfm"}
    (out / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"reference": str(out / "reference.pfm"), "terms": report["terms"],
                      "roi_pixels": report["roi_pixels"]}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("render", help="reference image, masks and negative controls for one exported snapshot")
    p.add_argument("snapshot")
    p.add_argument("--out", required=True)
    p.add_argument("--controls", nargs="*", default=[], choices=sorted(CONTROLS))
    p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    p.set_defaults(function=command_render)
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"F15 reference failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
