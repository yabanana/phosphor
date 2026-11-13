#!/usr/bin/env python3
"""F15.3 (ticket 13) ground truth of KHR_materials_transmission + KHR_materials_volume: a Mitsuba 3.9.1 path-traced
reference of the exported snapshot of a `transmission-*` scenario (src/testbench/transmission_validation.*).

What is compared and how (decision 11: transmission vs Mitsuba ground truth on the glass mask, static accumulation):
- Primary rays are pixel-CENTRE rays built from the snapshot camera exactly like tools/f15_reference.py (the engine
  samples the pixel centre; Mitsuba's own film would jitter over the pixel and blur the sharp refracted checkerboard).
  They are traced with `integrator.sample` (path integrator; `--medium stochastic`: volpath with real media) from a sampler seeded here.
  Only the pixels of the glass mask are rendered (primary hit on a glass instance, silhouette band excluded).
- Glass, from the material extension records of the snapshot (record 0 ior; record 1 transmissionFactor word 9,
  thicknessFactor word 11, attenuationColor 13-15, attenuationDistance 16; textures refused):
    * thicknessFactor > 0 (a CLOSED volume, single- or double-sided): Mitsuba `dielectric` (roughness 0) or
      `roughdielectric` (GGX, alpha = max(clamp(roughness, 0.04, 1)^2, 0.002), the engine's clamp), int_ior = ior,
      ext_ior = 1, specular transmittance = baseColor (the glTF tint of the BTDF), the interior a HOMOGENEOUS, purely
      ABSORBING medium sigma_t = -ln(attenuationColor) / attenuationDistance (albedo 0).  By default the medium is the
      expectation of Mitsuba's own: exp(-sigma_t * length) applied by a BSDF wrapper (`gltf_absorbing`) to the interior
      segment that ends at each hit seen from inside; volpath's medium kills absorbed paths at random (10-25 % per-pixel
      noise at the fixtures' attenuations), and stays selectable (`--medium stochastic`) as the test oracle.
    * thicknessFactor == 0 (thin-walled, "two surfaces back to back"): the glTF thin-walled interface, NOT Mitsuba's
      `thindielectric`: glTF layers ONE Fresnel-chosen sample per surface (reflect F, otherwise straight transmission
      tinted by baseColor, no refraction), whereas `thindielectric` models the slab's internal inter-reflections
      (T = (1 - F) / (1 + F), 4 % darker at normal incidence for ior 1.5, 8 % through a thin sphere's two surfaces).  A
      tiny Python BSDF (`gltf_thin`, two delta lobes) implements the glTF rule with the Schlick Fresnel at N.V.  Rough
      thin-walled glass has NO ground truth here (Mitsuba has no rough thin dielectric): it is refused.
- Transmission factor t < 1 (interpretation, declared): t mixes the diffuse base at the camera-VISIBLE surface only,
  glTF `mix(diffuse, specular_btdf * baseColor, t)` under the Fresnel layer; the interfaces refracted rays cross later
  are full dielectric interfaces.  image(t) = (1 - t) I0 + t I1 with I0 the opaque Appendix B dielectric (exact
  environment quadrature of tools/f15_reference.py: the image of today's engine, "transmission-ignored") and I1 the
  Mitsuba image with t = 1 everywhere.  The mix is linear in t in the glTF model, so this is exact for it.
- Environment: a constant emitter of the snapshot's sky radiance; emissive instances: one-sided Mitsuba area emitters
  of the material's emissive radiance on a black diffuse base (the fixtures have nothing diffuse behind the glass).

Differences between the glTF model and Mitsuba, all measured by `model-differences` (analytic sphere tracer below,
independent of Mitsuba) and reported with the harness:
  1. Fresnel: Mitsuba exact unpolarised; glTF Schlick (also on the leaving side, with the transmitted cosine; TIR = 1).
  2. Crossings: Mitsuba unbounded (max_depth 64, no Russian roulette); the engine stops at 4 interface events.
  3. Thin-walled: handled exactly (the custom BSDF); Mitsuba's `thindielectric` would differ (above).
  4. Tint: applied at EVERY transmitting interface event (Mitsuba `specular_transmittance` and the custom BSDF), so a
     two-crossing body is tinted baseColor^2; the gated volume fixtures use a white base (ambiguity not gated).
  5. Rough solids: Mitsuba's Walter rough dielectric (exact Fresnel, height-correlated Smith) against glTF's
     Schlick + separable Smith G_T: not measurable analytically here; the bound absorbs it (declared).

Negative controls (scene or model variants that a defective engine would match):
  transmission-ignored (the opaque Appendix B dielectric: analytic environment quadrature + for smooth glass the traced mirror
  reflection of the scene, M - F L_sky, so that an opaque cube sees the checkerboard in its top face like the engine's RT), transmission-not-tinted (tint = white), volume-ignored (no interior
  medium), ior-ignored (no refraction bend: solids become gltf_thin interfaces with their media kept), thin-as-volume
  (thin-walled objects refract like a solid, no medium), backface-rejected (back faces of the closed glass meshes removed
  from the scene: the exit is never found; medium dropped), emitter-double-counted (emissive radiance x 2),
  back-glass-black (glass more than 0.5 m behind the front-most glass is a black absorber).

`--declared` renders the declared `--rt off` model (probes/environment along the refracted ray; decision 09) in closed
form from the DFG table: L (diffuse mix (1 - E) + spec) with E = f0 A + B, the transmitted part (1 - E) t tint T, T =
exp(-sigma_t * volumeThickness_world) (the raster hint is the only path length without RT; 1 for thin-walled).

Usage:
  uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy python -I tools/f15_transmission_reference.py render SNAPSHOT --out DIR [--controls ...] [--spp N]
  ... model-differences SNAPSHOT
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import struct
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from f12_reference import np_module, scene_digest, snapshot, write_pfm  # noqa: E402
from f15_common import sha256_file  # noqa: E402

np = np_module()

INVALID = 0xFFFFFFFF
FEATURE_SPECULAR_IOR, FEATURE_TRANSMISSION, FEATURE_VOLUME = 1, 32, 64
EXTENSION_RECORD_WORDS = 20
MIN_ROUGHNESS, MIN_ALPHA = 0.04, 0.002
EROSION = 2                 # glass mask: pixels whose (2 * EROSION + 1)^2 neighbourhood is one glass instance
MAX_DEPTH = 64              # volpath path depth (interface events + medium events): effectively unbounded
ENGINE_CROSSINGS = 4        # the engine's declared bound of interface events along one transport sample
MIN_SIGMA = 1e-4            # 1/m: absorption floor of a channel the attenuation colour leaves at 1 (exp(-1e-4 * 10 m) = 0.999)

CONTROLS = ("transmission-ignored", "transmission-not-tinted", "volume-ignored", "ior-ignored", "thin-as-volume",
            "backface-rejected", "emitter-double-counted", "back-glass-black")
DECLARED_CONTROLS = ("declared-transmission-ignored", "declared-not-tinted", "declared-volume-ignored")

MODEL = {
    "reference": "Mitsuba 3.9.1 path integrator, pixel-centre rays through integrator.sample, only the glass-mask pixels",
    "solid": "dielectric / roughdielectric (GGX, alpha = max(clamp(r, .04, 1)^2, .002)), int_ior = ior, ext_ior = 1, "
             "specular transmittance = baseColor, interior = homogeneous absorbing medium (sigma_t = -ln(c) / d)",
    "thin_walled": "gltf_thin: per surface reflect F (Schlick at N.V, f0 = ((ior - 1) / (ior + 1))^2) else straight "
                   "transmission x baseColor (roughness 0 only)",
    "transmission_factor": "image(t) = (1 - t) I0 + t I1, I0 = opaque Appendix B dielectric (exact environment quadrature), "
                           "I1 = the t = 1 Mitsuba image; t applies at the camera-visible surface only",
    "environment": "constant emitter, radiance = snapshot sky; emitters: one-sided area emitters on a black diffuse base",
    "declared_rt_off": "L ((1 - E)((1 - t) base + t tint T) + E), E = clamp(f0 A + B, 0, 1) from the DFG table, "
                       "T = exp(-sigma_t thickness_world)",
}


# --- snapshot -> glass table ------------------------------------------------------------------------------------------

def f32(word):
    return struct.unpack("<f", struct.pack("<I", int(word) & 0xFFFFFFFF))[0]


def instance_matrix(instance):
    return np.asarray(instance["world"], dtype=np.float64).reshape(4, 4, order="F")


def sigma_t(color, distance):
    """sigma_t = -ln(attenuationColor) / attenuationDistance per channel (KHR_materials_volume); 0 without attenuation."""
    color = np.asarray(color, dtype=np.float64)
    if not math.isfinite(distance) or distance <= 0:
        return np.zeros(3)
    return np.maximum(-np.log(np.clip(color, 1e-6, 1.0)) / distance, 0.0)


def ior_f0(ior):
    return ((ior - 1.0) / (ior + 1.0)) ** 2


def mitsuba_alpha(roughness):
    """The engine's alpha: max(clamp(roughness, 0.04, 1)^2, 0.002)."""
    r = min(max(float(roughness), MIN_ROUGHNESS), 1.0)
    return max(r * r, MIN_ALPHA)


def glass_table(data, allow_rough_thin=False):
    """One dict per instance whose material carries KHR_materials_transmission (factor > 0), keyed by instance index.
    Raises for the combinations the reference does not model (textures, ior 0, rough thin-walled, other features)."""
    table = {}
    for k, instance in enumerate(data["instances"]):
        material = data["materials"][instance["material"]]
        bits = int(material.get("feature_bits", 0))
        if not bits & FEATURE_TRANSMISSION:
            if bits & FEATURE_VOLUME:
                raise ValueError("a volume without transmission has no effect and is outside the declared model")
            continue
        if bits & ~(FEATURE_SPECULAR_IOR | FEATURE_TRANSMISSION | FEATURE_VOLUME):
            raise ValueError(f"glass material feature bits {bits:#x} are outside the transmission/volume/ior model")
        if any(index != INVALID for index in material["textures"]):
            raise ValueError("glass samples a base texture; the reference is texture-free")
        first = int(material["extension_index"])
        records = data["material_extensions"]["records"]
        record0, record1 = records[first], records[first + 1]
        if len(record0) != EXTENSION_RECORD_WORDS or len(record1) != EXTENSION_RECORD_WORDS:
            raise ValueError("a material extension record must hold 20 words")
        if int(record1[10]) != INVALID or int(record1[12]) != INVALID:
            raise ValueError("a transmission or thickness texture is set; the reference is texture-free")
        ior = f32(record0[0])
        if not ior >= 1.0:
            raise ValueError(f"ior {ior}: the reference needs ior >= 1 (ior 0 is the permanent Fresnel = 1 mode)")
        entry = {"instance": k, "base": [float(v) for v in material["base"][:3]], "roughness": float(material["roughness"]),
                 "ior": ior, "transmission": f32(record1[9]), "thickness": f32(record1[11]),
                 "attenuation_color": [f32(record1[13]), f32(record1[14]), f32(record1[15])],
                 "attenuation_distance": f32(record1[16]), "double_sided": bool(int(material["flags"]) & 1),
                 "metallic": float(material["metallic"])}
        entry["thin"] = not entry["thickness"] > 0.0
        entry["sigma_t"] = sigma_t(entry["attenuation_color"], entry["attenuation_distance"])
        entry["has_medium"] = bool((entry["sigma_t"] > 0).any()) and not entry["thin"]
        if entry["metallic"] != 0.0:
            raise ValueError("metallic glass is outside the declared model (metals absorb all refracted light)")
        if entry["thin"] and entry["roughness"] > 0.0 and not allow_rough_thin:
            raise ValueError("rough thin-walled glass has no ground truth (Mitsuba has no rough thin dielectric)")
        entry["label"] = (f"{'thin' if entry['thin'] else 'volume'} r={entry['roughness']:g} ior={ior:g} t={entry['transmission']:g}"
                          f"{'' if entry['base'] == [1.0, 1.0, 1.0] else ' tint=' + ','.join(f'{c:g}' for c in entry['base'])}"
                          f"{'' if not entry['has_medium'] else ' atten=' + ','.join(f'{c:g}' for c in entry['attenuation_color']) + '@' + format(entry['attenuation_distance'], 'g')}")
        table[k] = entry
    if not table:
        raise ValueError("the snapshot has no transmissive material")
    return table


def emitters(data):
    """Instances with an emissive material: {instance index: radiance (3)}."""
    out = {}
    for k, instance in enumerate(data["instances"]):
        material = data["materials"][instance["material"]]
        if any(v > 0 for v in material["emissive"]):
            out[k] = [float(v) for v in material["emissive"]]
    return out


def check_scene(data):
    if any(float(l["intensity"]) > 0 for l in data["lights"]) or any(l["type"] != 6 for l in data["sampled_lights"]):
        raise ValueError("the transmission reference models the environment and emissive meshes only (no analytic lights)")
    sky = data["sky"]
    if not (sky[0] == sky[1] == sky[2]):
        raise ValueError("the environment must be isotropic and grey")
    for instance in data["instances"]:
        material = data["materials"][instance["material"]]
        if float(material["alpha_cutoff"]) > 0:
            raise ValueError("alpha-masked material is outside the declared model")


# --- camera -----------------------------------------------------------------------------------------------------------

def camera_directions(data):
    """Unit pixel-centre directions (h * w, 3) of the snapshot camera (the formula of f15_reference.cast_primary_rays)."""
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
    return np.asarray(camera["position"], dtype=np.float64), direction


# --- Fresnel and the analytic sphere tracer (independent of Mitsuba) ---------------------------------------------------

def fresnel_exact(cos_i, eta):
    """Unpolarised dielectric Fresnel; eta = n_transmitted / n_incident; total internal reflection = 1."""
    cos_i = np.clip(cos_i, 0.0, 1.0)
    sin2_t = (1.0 - cos_i * cos_i) / (eta * eta)
    cos_t = np.sqrt(np.clip(1.0 - sin2_t, 0.0, None))
    rs = (cos_i - eta * cos_t) / np.maximum(cos_i + eta * cos_t, 1e-12)
    rp = (eta * cos_i - cos_t) / np.maximum(eta * cos_i + cos_t, 1e-12)
    return np.where(sin2_t >= 1.0, 1.0, 0.5 * (rs * rs + rp * rp))


def fresnel_schlick(cos_i, eta):
    """glTF: F = f0 + (1 - f0)(1 - |V.H|)^5 with V.H = cos of the incident angle entering (eta >= 1) and of the
    TRANSMITTED angle leaving (eta < 1: KHR_materials_volume F_TR-), total internal reflection = 1."""
    cos_i = np.clip(cos_i, 0.0, 1.0)
    f0 = ((eta - 1.0) / (eta + 1.0)) ** 2
    sin2_t = (1.0 - cos_i * cos_i) / (eta * eta)
    cos_t = np.sqrt(np.clip(1.0 - sin2_t, 0.0, None))
    c = np.where(eta >= 1.0, cos_i, cos_t)
    return np.where(sin2_t >= 1.0, 1.0, f0 + (1.0 - f0) * (1.0 - c) ** 5)


def backdrop_rectangles(data):
    """Axis-aligned (facing +Z) emissive rectangles of the snapshot: rows (z, cx, cy, half width, half height, radiance rgb)."""
    rows = []
    for k, radiance in emitters(data).items():
        m = instance_matrix(data["instances"][k])
        centre, du, dv = m[:3, 3], m[:3, 0] * 0.5, m[:3, 2] * 0.5   # unit plane: x across, z along (after the +90 degree turn: world Y)
        if abs(du[0]) < 1e-9 or abs(abs(dv[1]) - np.linalg.norm(dv)) > 1e-6:
            raise ValueError("emitters must be rectangles facing +Z")
        rows.append((centre[2], centre[0], centre[1], abs(du[0]), abs(dv[1]), *radiance))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 8)


def exit_radiance(origin, direction, rectangles, sky):
    """Radiance seen by rays that left the glass: the nearest emissive rectangle facing the ray, else the environment."""
    out = np.broadcast_to(np.asarray(sky, dtype=np.float64), (len(origin), 3)).copy()
    nearest = np.full(len(origin), np.inf)
    for z, cx, cy, hw, hh, *radiance in rectangles:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (z - origin[:, 2]) / direction[:, 2]
        px, py = origin[:, 0] + t * direction[:, 0], origin[:, 1] + t * direction[:, 1]
        hit = (direction[:, 2] < 0) & (t > 1e-9) & (np.abs(px - cx) <= hw) & (np.abs(py - cy) <= hh) & (t < nearest)
        nearest[hit] = t[hit]
        out[hit] = radiance
    return out


def trace_sphere(origin, direction, sphere, glass, rectangles, sky, fresnel="exact", max_events=None, on_limit="env",
                 prune=1e-5, max_iterations=60):
    """Expected radiance of pixel-centre rays (n, 3) through ONE smooth sphere {"centre", "radius"} of the glass dict
    `glass` (ior, base = tint, sigma_t, thin): the exact expectation of the Fresnel-chosen reflect/refract tree (every
    branch followed, weights pruned below `prune`), Beer-Lambert along the inside segments.  Other geometry is ignored
    except the emissive rectangles seen after the glass.  `fresnel`: "exact" (Mitsuba) or "schlick" (glTF);
    `max_events`: interface events allowed per path (None = unbounded); a path that would exceed them ends with the
    environment (`on_limit` "env") or black ("zero")."""
    fresnel_function = fresnel_exact if fresnel == "exact" else fresnel_schlick
    centre, radius = np.asarray(sphere["centre"], dtype=np.float64), float(sphere["radius"])
    ior, tint, sigma, thin = float(glass["ior"]), np.asarray(glass["base"], dtype=np.float64), glass["sigma_t"], glass["thin"]
    n = len(origin)
    result = np.zeros((n, 3))
    sky_vector = np.asarray(sky, dtype=np.float64)
    o, d = origin.copy(), direction.copy()
    w = np.ones((n, 3))
    inside = np.zeros(n, dtype=bool)
    events = np.zeros(n, dtype=np.int64)
    pixel = np.arange(n)
    eps = 1e-7 * radius
    for _ in range(max_iterations):
        if not len(o):
            break
        oc = o - centre
        b = np.sum(oc * d, axis=1)
        disc = b * b - (np.sum(oc * oc, axis=1) - radius * radius)
        root = np.sqrt(np.clip(disc, 0.0, None))
        t0, t1 = -b - root, -b + root
        if thin:
            t = np.where(t0 > eps, t0, t1)
            hit = (disc > 0) & (t > eps)
        else:
            t = np.where(inside, t1, t0)
            hit = (disc > 0) & (t > eps)
        # rays that leave: the emissive rectangles or the environment, weighted
        miss = ~hit
        if miss.any():
            np.add.at(result, pixel[miss], w[miss] * exit_radiance(o[miss], d[miss], rectangles, sky_vector))
        o, d, w, inside, events, pixel, t = o[hit], d[hit], w[hit], inside[hit], events[hit], pixel[hit], t[hit]
        if not len(o):
            break
        if not thin and sigma.any():  # Beer-Lambert over the inside segment that just ended at this hit
            w = w * np.where(inside[:, None], np.exp(-sigma[None, :] * t[:, None]), 1.0)
        p = o + t[:, None] * d
        normal = (p - centre) / radius
        d_dot_n = np.sum(d * normal, axis=1)
        entering = d_dot_n < 0
        n_facing = np.where(entering[:, None], normal, -normal)
        cos_i = np.abs(d_dot_n)
        events = events + 1
        if max_events is not None:
            over = events > max_events
            if over.any():
                if on_limit == "env":
                    np.add.at(result, pixel[over], w[over] * sky_vector)
                keep = ~over
                o, d, w, inside, events, pixel, p, n_facing, cos_i, entering = (a[keep] for a in (o, d, w, inside, events, pixel, p, n_facing, cos_i, entering))
                if not len(o):
                    break
        reflect_d = d + 2.0 * cos_i[:, None] * n_facing
        if thin:
            eta = np.full(len(o), ior)
            f = fresnel_function(cos_i, eta)
            transmit_d, transmit_inside = d, inside
            transmit_weight = (1.0 - f)[:, None] * tint[None, :]
            ok = np.ones(len(o), dtype=bool)
        else:
            eta = np.where(entering, ior, 1.0 / ior)
            f = fresnel_function(cos_i, eta)
            ratio = 1.0 / eta
            sin2_t = ratio * ratio * (1.0 - cos_i * cos_i)
            cos_t = np.sqrt(np.clip(1.0 - sin2_t, 0.0, None))
            transmit_d = ratio[:, None] * d + (ratio * cos_i - cos_t)[:, None] * n_facing
            transmit_inside = entering
            transmit_weight = (1.0 - f)[:, None] * tint[None, :]
            ok = sin2_t < 1.0
        new_o = np.concatenate([p, p[ok]])
        new_d = np.concatenate([reflect_d, transmit_d[ok]])
        new_w = np.concatenate([w * f[:, None], w[ok] * transmit_weight[ok]])
        new_inside = np.concatenate([inside, transmit_inside[ok]])
        new_events = np.concatenate([events, events[ok]])
        new_pixel = np.concatenate([pixel, pixel[ok]])
        keep = new_w.max(axis=1) > prune
        o, d, w, inside, events, pixel = new_o[keep], new_d[keep], new_w[keep], new_inside[keep], new_events[keep], new_pixel[keep]
    return result


def sphere_of(root, data, k):
    """{"centre", "radius"} of instance k when its mesh is a sphere about the object origin under a uniform scale, else None."""
    from f15_reference import read_ply
    instance = data["instances"][k]
    m = instance_matrix(instance)
    scale = np.linalg.norm(m[:3, :3], axis=0)
    if scale.max() - scale.min() > 1e-6:
        return None
    properties, _ = read_ply(root / data["meshes"][instance["mesh"]])
    distance = np.sqrt(properties["x"] ** 2 + properties["y"] ** 2 + properties["z"] ** 2)
    if (distance.max() - distance.min()) > 1e-3 * distance.max():
        return None
    return {"centre": m[:3, 3], "radius": float(distance.max() * scale[0])}


# --- glass mask and opaque dielectric ------------------------------------------------------------------------------------

def glass_masks(cast, glass):
    """(instance map (h, w) of glass pixels, -1 elsewhere; ROI (h, w) bool: the pixels whose (2 EROSION + 1)^2 neighbourhood
    is the same glass instance)."""
    valid, _, _, instance, _ = cast
    inst = np.where(valid & np.isin(instance, list(glass)), instance, -1)
    same = inst >= 0
    h, w = inst.shape
    for dy in range(-EROSION, EROSION + 1):
        for dx in range(-EROSION, EROSION + 1):
            same &= np.roll(inst, (dy, dx), axis=(0, 1)) == inst
    same[:EROSION, :] = same[-EROSION:, :] = False
    same[:, :EROSION] = same[:, -EROSION:] = False
    return inst, same


def traced_specular_correction(root, data, glass, origin, directions, pick, ids, n_dot_v, threads, out_dir):
    """The scene the opaque dielectric reflects: for the SMOOTH glass objects the specular layer is a mirror weighted by the
    Schlick F(N.V) (every glass object is an opaque mirror, one deterministic path per pixel), so the traced reflection of
    the emitters, the other objects and the environment is M and the correction to the uniform-environment image I0 is
    M - F(N.V) L_sky (the quadrature already holds F L_sky).  Rough objects get 0 (their blurred reflection of the backdrop
    stays unmodelled, declared)."""
    scene, mi, dr = build_scene(root, data, glass, "mirror-only", threads, out_dir)
    mirror, _, _ = render_pixels(scene, mi, dr, origin, directions[pick], 1, 7)
    f0 = np.array([ior_f0(glass[k]["ior"]) for k in ids])
    f = f0 + (1.0 - f0) * (1.0 - n_dot_v) ** 5
    correction = mirror - f[:, None] * float(data["sky"][0])
    smooth = np.array([glass[k]["roughness"] <= 0.0 for k in ids])
    return np.where(smooth[:, None], correction, 0.0)


def opaque_dielectric(data, glass, instance_ids, n_dot_v):
    """I0: the opaque Appendix B dielectric of tools/f15_reference.py (exact environment quadrature, KHR_materials_ior f0)
    of the glass materials under the constant environment: what the engine renders while it ignores transmission."""
    import f15_reference as ref
    n = len(instance_ids)
    base = np.array([glass[k]["base"] for k in instance_ids], dtype=np.float64)
    roughness = np.array([glass[k]["roughness"] for k in instance_ids], dtype=np.float64)
    f0 = np.array([[ref.ior_factor(glass[k]["ior"])] * 3 for k in instance_ids], dtype=np.float64)
    dielectric = {"f0": f0, "weight": np.ones(n), "variant": "extension"}
    return ref.environment_radiance(base, np.zeros(n), roughness, n_dot_v, float(data["sky"][0]), ref.SPECULAR_EXT, dielectric)


# --- Mitsuba scene --------------------------------------------------------------------------------------------------------

_REGISTERED = []


def register_gltf_thin(mi, dr):
    """The glTF thin-walled interface as a Mitsuba BSDF plugin: reflect with probability F (Schlick at N.V, mirror), else
    straight transmission tinted by `tint`; two delta lobes, no inter-reflection."""
    if _REGISTERED:
        return

    class GltfThin(mi.BSDF):
        def __init__(self, props):
            mi.BSDF.__init__(self, props)
            self.f0 = float(props.get("f0", 0.04))
            self.tint = mi.Color3f(float(props.get("tint_r", 1.0)), float(props.get("tint_g", 1.0)), float(props.get("tint_b", 1.0)))
            sides = mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
            reflection = mi.BSDFFlags.DeltaReflection | sides
            transmission = mi.BSDFFlags.DeltaTransmission | sides
            self.m_components = [reflection, transmission]
            self.m_flags = reflection | transmission

        def sample(self, ctx, si, sample1, sample2, active):
            cos_i = dr.abs(mi.Frame3f.cos_theta(si.wi))
            f = self.f0 + (1.0 - self.f0) * (1.0 - cos_i) ** 5
            reflect = sample1 < f
            bs = dr.zeros(mi.BSDFSample3f)
            bs.wo = dr.select(reflect, mi.Vector3f(-si.wi.x, -si.wi.y, si.wi.z), -si.wi)
            bs.pdf = dr.select(reflect, f, 1.0 - f)
            bs.eta = 1.0
            bs.sampled_type = dr.select(reflect, mi.UInt32(+mi.BSDFFlags.DeltaReflection), mi.UInt32(+mi.BSDFFlags.DeltaTransmission))
            bs.sampled_component = dr.select(reflect, mi.UInt32(0), mi.UInt32(1))
            weight = dr.select(reflect, mi.Color3f(1.0), self.tint)
            return bs, dr.select(active, weight, mi.Color3f(0.0))

        def eval(self, ctx, si, wo, active=True):
            return mi.Color3f(0.0)

        def pdf(self, ctx, si, wo, active=True):
            return mi.Float(0.0)

        def eval_pdf(self, ctx, si, wo, active=True):
            return mi.Color3f(0.0), mi.Float(0.0)

        def traverse(self, callback):
            pass

        def to_string(self):
            return "GltfThin[]"

    class GltfAbsorbing(mi.BSDF):
        """Beer-Lambert absorption of a homogeneous, purely absorbing interior as a BSDF wrapper: the segment that ends at a hit
        approached from INSIDE (shading cosine < 0) began at the previous interaction on the same closed glass mesh, so its
        length is `si.t` and the nested BSDF's weight is multiplied by exp(-sigma_t si.t).  Same expectation as Mitsuba's
        homogeneous medium with albedo 0 (volpath samples the absorption by killing paths: 5-25 % pixel noise at the
        fixtures' attenuation, against none here); `volpath` stays available (--medium stochastic) as the independent
        confirmation, see the tests."""

        def __init__(self, props):
            mi.BSDF.__init__(self, props)
            self.sigma = mi.Color3f(float(props.get("sigma_r", 0.0)), float(props.get("sigma_g", 0.0)), float(props.get("sigma_b", 0.0)))
            self.nested = props["nested"]
            self.m_flags = self.nested.flags()
            self.m_components = [self.nested.flags(i) for i in range(self.nested.component_count())]

        def attenuation(self, si):
            inside = mi.Frame3f.cos_theta(si.wi) < 0.0
            return dr.select(inside, dr.exp(-self.sigma * si.t), mi.Color3f(1.0))

        def sample(self, ctx, si, sample1, sample2, active):
            bs, value = self.nested.sample(ctx, si, sample1, sample2, active)
            return bs, value * self.attenuation(si)

        def eval(self, ctx, si, wo, active=True):
            return self.nested.eval(ctx, si, wo, active) * self.attenuation(si)

        def pdf(self, ctx, si, wo, active=True):
            return self.nested.pdf(ctx, si, wo, active)

        def eval_pdf(self, ctx, si, wo, active=True):
            value, pdf = self.nested.eval_pdf(ctx, si, wo, active)
            return value * self.attenuation(si), pdf

        def traverse(self, callback):
            pass

        def to_string(self):
            return "GltfAbsorbing[]"

    class GltfMirror(mi.BSDF):
        """The specular layer of an OPAQUE glTF dielectric as a mirror: always reflect, weighted by the Schlick Fresnel at N.V."""

        def __init__(self, props):
            mi.BSDF.__init__(self, props)
            self.f0 = float(props.get("f0", 0.04))
            flags = mi.BSDFFlags.DeltaReflection | mi.BSDFFlags.FrontSide | mi.BSDFFlags.BackSide
            self.m_components = [flags]
            self.m_flags = flags

        def sample(self, ctx, si, sample1, sample2, active):
            cos_i = dr.abs(mi.Frame3f.cos_theta(si.wi))
            f = self.f0 + (1.0 - self.f0) * (1.0 - cos_i) ** 5
            bs = dr.zeros(mi.BSDFSample3f)
            bs.wo = mi.Vector3f(-si.wi.x, -si.wi.y, si.wi.z)
            bs.pdf = 1.0
            bs.eta = 1.0
            bs.sampled_type = mi.UInt32(+mi.BSDFFlags.DeltaReflection)
            bs.sampled_component = mi.UInt32(0)
            return bs, dr.select(active, mi.Color3f(f), mi.Color3f(0.0))

        def eval(self, ctx, si, wo, active=True):
            return mi.Color3f(0.0)

        def pdf(self, ctx, si, wo, active=True):
            return mi.Float(0.0)

        def eval_pdf(self, ctx, si, wo, active=True):
            return mi.Color3f(0.0), mi.Float(0.0)

        def traverse(self, callback):
            pass

        def to_string(self):
            return "GltfMirror[]"

    mi.register_bsdf("gltf_mirror", lambda props: GltfMirror(props))
    mi.register_bsdf("gltf_thin", lambda props: GltfThin(props))
    mi.register_bsdf("gltf_absorbing", lambda props: GltfAbsorbing(props))
    _REGISTERED.append(True)


def write_ply(path, properties, faces):
    """Binary little-endian triangle PLY with the exporter's vertex layout (x y z nx ny nz s t)."""
    names = ("x", "y", "z", "nx", "ny", "nz", "s", "t")
    columns = np.stack([properties[n] for n in names], axis=1).astype("<f4")
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(columns) +
              "".join(f"property float {n}\n" for n in names) +
              "element face %d\nproperty list uchar uint vertex_indices\nend_header\n" % len(faces))
    face_dtype = np.dtype([("n", "u1"), ("i", "<u4", 3)])
    table = np.zeros(len(faces), dtype=face_dtype)
    table["n"] = 3
    table["i"] = faces
    with open(path, "wb") as stream:
        stream.write(header.encode())
        stream.write(columns.tobytes())
        stream.write(table.tobytes())


def front_facing_mesh(root, data, k, camera, out_dir):
    """The PLY of instance k restricted to the triangles that face the camera (world space): the model of back-face
    rejection (the exit surface of a closed glass mesh does not exist for the transport)."""
    from f15_reference import read_ply
    instance = data["instances"][k]
    properties, faces = read_ply(root / data["meshes"][instance["mesh"]])
    m = instance_matrix(instance)
    p = np.stack([properties["x"], properties["y"], properties["z"]], axis=1).astype(np.float64)
    world = p @ m[:3, :3].T + m[:3, 3]
    a, b, c = world[faces[:, 0]], world[faces[:, 1]], world[faces[:, 2]]
    normal = np.cross(b - a, c - a)
    centroid = (a + b + c) / 3.0
    keep = np.einsum("ij,ij->i", normal, centroid - camera[None, :]) < 0.0
    path = out_dir / f"front_facing_{k}.ply"
    write_ply(path, properties, faces[keep])
    return path


def mitsuba_modules(threads):
    import drjit as dr
    import mitsuba as mi
    mi.set_variant("llvm_ad_rgb")
    dr.set_flag(dr.JitFlag.FastMath, False)
    dr.set_thread_count(threads)
    return mi, dr


def build_scene(root, data, glass, control, threads, out_dir, medium="analytic"):
    """The Mitsuba scene of the snapshot with the glass of the model variant `control` ("" = the ground truth).
    `medium`: "analytic" (path integrator, Beer-Lambert in the `gltf_absorbing` wrapper) or "stochastic" (volpath with a
    homogeneous absorbing interior medium)."""
    mi, dr = mitsuba_modules(threads)
    register_gltf_thin(mi, dr)
    transform = getattr(mi, "ScalarAffineTransform4f", None) or mi.ScalarTransform4f
    camera_position = np.asarray(data["camera"]["position"], dtype=np.float64)
    front = max((instance_matrix(data["instances"][k])[2, 3] for k in glass), default=0.0)
    description = {"type": "scene",
                   "integrator": {"type": "volpath" if medium == "stochastic" else "path", "max_depth": MAX_DEPTH,
                                  "rr_depth": MAX_DEPTH + 1},
                   "sensor": {"type": "perspective", "fov": 45.0, "film": {"type": "hdrfilm", "width": 8, "height": 8},
                              "sampler": {"type": "independent", "sample_count": 1}}}
    radiance = emitters(data)
    scale_emission = 2.0 if control == "emitter-double-counted" else 1.0
    for k, instance in enumerate(data["instances"]):
        shape = {"type": "ply", "filename": str(root / data["meshes"][instance["mesh"]]), "face_normals": False,
                 "to_world": transform(instance_matrix(instance))}
        if k in glass:
            g = glass[k]
            tint = [1.0, 1.0, 1.0] if control == "transmission-not-tinted" else g["base"]
            solid = not g["thin"]
            if control == "thin-as-volume" and g["thin"]:
                bsdf = {"type": "dielectric", "int_ior": g["ior"], "ext_ior": 1.0,
                        "specular_transmittance": {"type": "rgb", "value": tint}}
            elif control == "ior-ignored" and solid:
                bsdf = {"type": "gltf_thin", "f0": ior_f0(g["ior"]), "tint_r": tint[0], "tint_g": tint[1], "tint_b": tint[2]}
            elif g["thin"]:
                bsdf = {"type": "gltf_thin", "f0": ior_f0(g["ior"]), "tint_r": tint[0], "tint_g": tint[1], "tint_b": tint[2]}
            elif g["roughness"] <= 0.0:
                bsdf = {"type": "dielectric", "int_ior": g["ior"], "ext_ior": 1.0,
                        "specular_transmittance": {"type": "rgb", "value": tint}}
            else:
                bsdf = {"type": "roughdielectric", "distribution": "ggx", "alpha": mitsuba_alpha(g["roughness"]),
                        "int_ior": g["ior"], "ext_ior": 1.0, "specular_transmittance": {"type": "rgb", "value": tint}}
            if control == "mirror-only":
                bsdf = {"type": "gltf_mirror", "f0": ior_f0(g["ior"])}
            if control == "back-glass-black" and instance_matrix(instance)[2, 3] < front - 0.5:
                bsdf = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.0, 0.0, 0.0]}}
            shape["bsdf"] = bsdf
            if solid and g["has_medium"] and control not in ("volume-ignored", "backface-rejected", "back-glass-black", "mirror-only"):
                if medium == "stochastic":
                    # sigma_t reaches Mitsuba as a reflectance-like RGB in [0, 1] times a scale (the maximum channel).
                    sigma = np.maximum(g["sigma_t"], MIN_SIGMA)  # a zero channel makes volpath's chromatic sampling divide by 0
                    peak = float(sigma.max())
                    shape["interior"] = {"type": "homogeneous", "albedo": {"type": "rgb", "value": [0.0, 0.0, 0.0]},
                                         "sigma_t": {"type": "rgb", "value": [float(v) / peak for v in sigma]}, "scale": peak}
                else:
                    shape["bsdf"] = {"type": "gltf_absorbing", "sigma_r": float(g["sigma_t"][0]), "sigma_g": float(g["sigma_t"][1]),
                                     "sigma_b": float(g["sigma_t"][2]), "nested": bsdf}
            if control == "backface-rejected" and solid:
                shape["filename"] = str(front_facing_mesh(root, data, k, camera_position, out_dir))
        else:
            shape["bsdf"] = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.0, 0.0, 0.0]}}
            if k in radiance:
                shape["emitter"] = {"type": "area", "radiance": {"type": "rgb", "value": [v * scale_emission for v in radiance[k]]}}
        description[f"instance_{k}"] = shape
    sky = [float(v) for v in data["sky"]]
    if any(v > 0 for v in sky):
        description["sky"] = {"type": "constant", "radiance": {"type": "rgb", "value": sky}}
    return mi.load_dict(description), mi, dr


def render_pixels(scene, mi, dr, origin, directions, spp, seed, batch_rays=1_000_000):
    """(mean (n, 3), standard error of the mean (n, 3), non-finite sample count) of `spp` pixel-centre samples per ray.
    Each `integrator.sample` call traces a whole wavefront (the Python-side tracing cost is per call, not per ray), so every
    pixel ray is replicated `rep` times per call: samples = rep * calls >= spp."""
    integrator = scene.integrator()
    n = len(directions)
    rep = max(1, min(spp, batch_rays // max(n, 1)))
    pixels_per_chunk = max(1, min(n, batch_rays // rep))
    calls = -(-spp // rep)
    samples = rep * calls
    total = np.zeros((n, 3))
    total2 = np.zeros((n, 3))
    bad = 0
    for start in range(0, n, pixels_per_chunk):
        stop = min(n, start + pixels_per_chunk)
        count = stop - start
        d = np.repeat(directions[start:stop], rep, axis=0)
        rays = count * rep
        ray = mi.RayDifferential3f(mi.Point3f(mi.Float(np.full(rays, origin[0])), mi.Float(np.full(rays, origin[1])),
                                              mi.Float(np.full(rays, origin[2]))),
                                   mi.Vector3f(mi.Float(d[:, 0].copy()), mi.Float(d[:, 1].copy()), mi.Float(d[:, 2].copy())))
        sampler = mi.load_dict({"type": "independent", "sample_count": 1})
        sampler.seed(seed + start, rays)
        for _ in range(calls):
            value, _, _ = integrator.sample(scene, sampler, ray)
            rgb = np.stack([np.asarray(value[c], dtype=np.float64) for c in range(3)], axis=1)
            finite = np.isfinite(rgb).all(axis=1)
            bad += int((~finite).sum())
            rgb[~finite] = 0.0
            rgb = rgb.reshape(count, rep, 3)
            total[start:stop] += rgb.sum(axis=1)
            total2[start:stop] += (rgb * rgb).sum(axis=1)
    mean = total / samples
    variance = np.maximum(total2 / samples - mean * mean, 0.0) * samples / max(samples - 1, 1)
    return mean, np.sqrt(variance / samples), bad


# --- rendering one scene ---------------------------------------------------------------------------------------------------

def blend_transmission_factor(i0, i1, t):
    """image(t) = (1 - t) I0 + t I1 (t per pixel, shape (n,))."""
    return (1.0 - t)[:, None] * i0 + t[:, None] * i1


def render_variant(root, data, glass, cast, pick, control, spp, seed, threads, out_dir, i0, medium="analytic"):
    """Radiance (len(pick), 3) of the glass-mask pixels `pick` (flat indices) for one model variant, with its
    standard error and render time."""
    t_pixel = np.array([glass[k]["transmission"] for k in cast_instance_flat(cast)[pick]], dtype=np.float64)
    if control == "transmission-ignored":
        return i0.copy(), np.zeros_like(i0), {"seconds": 0.0, "non_finite": 0}
    scene, mi, dr = build_scene(root, data, glass, control, threads, out_dir, medium)
    origin, directions = camera_directions(data)
    started = time.monotonic()
    i1, error, bad = render_pixels(scene, mi, dr, origin, directions[pick], spp, seed)
    seconds = time.monotonic() - started
    image = blend_transmission_factor(i0, i1, t_pixel)
    return image, error * t_pixel[:, None], {"seconds": seconds, "non_finite": bad}


def cast_instance_flat(cast):
    return cast[3].reshape(-1)


def command_render(args):
    import f15_reference as ref
    root, data = snapshot(args.snapshot)
    check_scene(data)
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    glass = glass_table(data)
    cast = ref.cast_primary_rays(root, data, args.threads)
    instance_map, roi = glass_masks(cast, glass)
    height, width = roi.shape
    flat_roi = roi.reshape(-1)
    pick = np.flatnonzero(flat_roi)
    if not pick.size:
        raise ValueError("no glass pixel survives the silhouette erosion")
    camera = np.asarray(data["camera"]["position"], dtype=np.float64)
    position, normal = cast[1][pick], ref.normalize(cast[2][pick])
    view = ref.normalize(camera[None, :] - position)
    n_dot_v = np.clip(ref.dot(normal, view), 1e-4, 1.0)
    ids = cast_instance_flat(cast)[pick]
    sky = [float(v) for v in data["sky"]]
    objects = []
    for k, entry in sorted(glass.items()):
        objects.append({**{key: entry[key] for key in ("instance", "label", "roughness", "ior", "transmission", "thickness",
                                                       "attenuation_color", "attenuation_distance", "thin", "base")},
                        "pixels": int((ids == k).sum())})
    i0 = opaque_dielectric(data, glass, ids, n_dot_v)
    origin, directions = camera_directions(data)
    i0 = i0 + traced_specular_correction(root, data, glass, origin, directions, pick, ids, n_dot_v, args.threads, out)
    report = {"schema": 1, "tool": "tools/f15_transmission_reference.py", "tool_sha256": sha256_file(__file__), "model": MODEL,
              "snapshot_sha256": scene_digest(root), "frame": data.get("frame"), "sky": data["sky"], "kind": "mitsuba",
              "spp": args.spp, "max_depth": MAX_DEPTH, "medium": args.medium, "erosion_pixels": EROSION, "engine_crossings": ENGINE_CROSSINGS,
              "width": width, "height": height, "roi_pixels": int(pick.size), "objects": objects, "controls": {},
              "terms": ["environment", "emitters", "glass-path-traced"], "interpretations": [
                  "tint applies at every transmitting interface event", "transmission factor applies at the camera-visible surface only",
                  "thin-walled = glTF single Fresnel sample per surface (gltf_thin), not Mitsuba thindielectric"]}
    names = [""] + list(args.controls)
    for name in names:
        started = time.monotonic()
        image_pixels, error_pixels, info = render_variant(root, data, glass, cast, pick, name, args.spp, args.seed, args.threads, out, i0,
                                                          args.medium)
        image = np.broadcast_to(np.asarray(sky, dtype=np.float32), (height * width, 3)).copy()
        image[pick] = image_pixels
        stem = "reference" if not name else f"control-{name}"
        write_pfm(out / f"{stem}.pfm", image.reshape(height, width, 3))
        floor = 0.05 * float(image_pixels.mean())
        relative = (error_pixels / np.maximum(image_pixels, floor)).max(axis=1)
        noise = {"mean_relative_standard_error": float(relative.mean()), "p95_relative_standard_error": float(np.percentile(relative, 95)),
                 "per_object": {str(k): float(relative[ids == k].mean()) for k in glass if (ids == k).any()}}
        entry = {"file": f"{stem}.pfm", "seconds": info["seconds"], "non_finite_samples": info["non_finite"], "noise": noise}
        if name:
            report["controls"][name] = entry
        else:
            report.update(reference=entry, maximum_radiance=float(image_pixels.max()))
            np.savez_compressed(out / "masks.npz", roi=roi, sphere=np.where(instance_map >= 0, instance_map, -1).astype(np.int32),
                                glass=instance_map >= 0)
        print(json.dumps({"variant": name or "reference", **entry}), flush=True)
    (out / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


# --- declared --rt off model ------------------------------------------------------------------------------------------------

DECLARED_VARIANTS = {"spec": {}, "declared-transmission-ignored": {"t": 0.0}, "declared-not-tinted": {"tint": False},
                     "declared-volume-ignored": {"volume": False}}


def declared_pixel(base, roughness, ior, t, thickness, att_color, att_distance, n_dot_v, sky, variant="spec"):
    """The declared --rt off model of one pixel set (arrays (n, ...)): see the module docstring."""
    import f15_reference as ref
    options = DECLARED_VARIANTS[variant]
    a, b = ref.dfg_lookup(np.clip(n_dot_v, 1e-4, 1.0), np.clip(roughness, MIN_ROUGHNESS, 1.0))
    f0 = ref.ior_factor(ior)
    spec = np.clip(f0 * a + b, 0.0, 1.0)[:, None]
    tint = base if options.get("tint", True) else np.ones_like(base)
    absorb = np.ones_like(base)
    if options.get("volume", True):
        for k in range(len(base)):
            if thickness[k] > 0:
                absorb[k] = np.exp(-sigma_t(att_color[k], float(att_distance[k])) * thickness[k])
    t = np.full(len(base), options["t"]) if "t" in options else t
    diffuse_mix = (1.0 - t)[:, None] * base + t[:, None] * tint * absorb
    return float(sky) * (diffuse_mix * (1.0 - spec) + spec)


def command_declared(args):
    import f15_reference as ref
    root, data = snapshot(args.snapshot)
    check_scene(data)
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    glass = glass_table(data, allow_rough_thin=True)
    cast = ref.cast_primary_rays(root, data, args.threads)
    instance_map, roi = glass_masks(cast, glass)
    height, width = roi.shape
    pick = np.flatnonzero(roi.reshape(-1))
    camera = np.asarray(data["camera"]["position"], dtype=np.float64)
    normal = ref.normalize(cast[2][pick])
    view = ref.normalize(camera[None, :] - cast[1][pick])
    n_dot_v = np.clip(ref.dot(normal, view), 1e-4, 1.0)
    ids = cast_instance_flat(cast)[pick]
    arrays = dict(
        base=np.array([glass[k]["base"] for k in ids]), roughness=np.array([glass[k]["roughness"] for k in ids]),
        ior=np.array([glass[k]["ior"] for k in ids]), t=np.array([glass[k]["transmission"] for k in ids]),
        # the volume thickness in WORLD units: mesh-local thicknessFactor times the instance scale along the thin axis
        thickness=np.array([glass[k]["thickness"] * float(np.linalg.norm(instance_matrix(data["instances"][k])[:3, :3], axis=0).min())
                            for k in ids]),
        att_color=np.array([glass[k]["attenuation_color"] for k in ids]), att_distance=np.array([glass[k]["attenuation_distance"] for k in ids]))
    sky = float(data["sky"][0])
    report = {"schema": 1, "tool": "tools/f15_transmission_reference.py", "tool_sha256": sha256_file(__file__), "model": MODEL,
              "snapshot_sha256": scene_digest(root), "kind": "declared", "width": width, "height": height,
              "roi_pixels": int(pick.size), "controls": {}, "terms": ["environment-declared"], "erosion_pixels": EROSION,
              "objects": [{"instance": k, "label": g["label"], "pixels": int((ids == k).sum())} for k, g in sorted(glass.items())]}
    for name in ["spec"] + list(args.controls):
        values = declared_pixel(arrays["base"], arrays["roughness"], arrays["ior"], arrays["t"], arrays["thickness"],
                                arrays["att_color"], arrays["att_distance"], n_dot_v, sky, name)
        image = np.full((height * width, 3), sky, dtype=np.float32)
        image[pick] = values
        stem = "reference" if name == "spec" else f"control-{name}"
        write_pfm(out / f"{stem}.pfm", image.reshape(height, width, 3))
        entry = {"file": f"{stem}.pfm"}
        if name == "spec":
            report["reference"] = entry
            report["maximum_radiance"] = float(values.max())
            np.savez_compressed(out / "masks.npz", roi=roi, sphere=np.where(instance_map >= 0, instance_map, -1).astype(np.int32),
                                glass=instance_map >= 0)
        else:
            report["controls"][name] = entry
    (out / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"reference": str(out / "reference.pfm"), "roi_pixels": report["roi_pixels"]}))
    return 0


# --- model differences (analytic, no Mitsuba path tracing) -----------------------------------------------------------------

def command_model_differences(args):
    import f15_reference as ref
    root, data = snapshot(args.snapshot)
    check_scene(data)
    glass = glass_table(data)
    cast = ref.cast_primary_rays(root, data, args.threads)
    _, roi = glass_masks(cast, glass)
    origin, directions = camera_directions(data)
    rectangles = backdrop_rectangles(data)
    sky = [float(v) for v in data["sky"]]
    flat_instance = cast_instance_flat(cast)
    rows = []
    for k, g in sorted(glass.items()):
        pixels = np.flatnonzero(roi.reshape(-1) & (flat_instance == k))
        sphere = sphere_of(root, data, k)
        if sphere is None or g["roughness"] > 0 or not pixels.size:
            continue  # smooth spheres only (a cube, a pane or a rough lens has no analytic tree here)
        o = np.broadcast_to(origin, (len(pixels), 3)).copy()
        d = directions[pixels]
        baseline = trace_sphere(o, d, sphere, g, rectangles, sky, "exact", None)
        floor = 0.05 * float(baseline.mean())

        def stat(image):
            e = (np.abs(image - baseline) / np.maximum(baseline, floor)).max(axis=1)
            return {"mean": float(e.mean()), "p95": float(np.percentile(e, 95))}

        variants = {"schlick": trace_sphere(o, d, sphere, g, rectangles, sky, "schlick", None),
                    "events4-env": trace_sphere(o, d, sphere, g, rectangles, sky, "exact", ENGINE_CROSSINGS, "env"),
                    "events4-zero": trace_sphere(o, d, sphere, g, rectangles, sky, "exact", ENGINE_CROSSINGS, "zero"),
                    "schlick+events4-zero": trace_sphere(o, d, sphere, g, rectangles, sky, "schlick", ENGINE_CROSSINGS, "zero")}
        rows.append({"instance": k, "label": g["label"], "pixels": int(pixels.size), **{n: stat(v) for n, v in variants.items()}})
    print(json.dumps({"model_differences_vs_exact_unbounded": rows}, indent=1))
    return 0


# --- command line -----------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("render", help="Mitsuba reference image, glass masks and negative controls of one snapshot")
    p.add_argument("snapshot")
    p.add_argument("--out", required=True)
    p.add_argument("--controls", nargs="*", default=[], choices=CONTROLS)
    p.add_argument("--spp", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--medium", choices=["analytic", "stochastic"], default="analytic",
                   help="Beer-Lambert as a BSDF wrapper (default, noise-free) or volpath's absorbing medium (noisy: the test oracle)")
    p.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    p.set_defaults(function=command_render)
    p = sub.add_parser("declared", help="the declared --rt off model (closed form) and its negative controls")
    p.add_argument("snapshot")
    p.add_argument("--out", required=True)
    p.add_argument("--controls", nargs="*", default=[], choices=[c for c in DECLARED_VARIANTS if c != "spec"])
    p.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    p.set_defaults(function=command_declared)
    p = sub.add_parser("model-differences", help="analytic measure of the glTF-vs-Mitsuba model differences on the snapshot's smooth spheres")
    p.add_argument("snapshot")
    p.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    p.set_defaults(function=command_model_differences)
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"F15 transmission reference failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
