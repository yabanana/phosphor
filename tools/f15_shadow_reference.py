#!/usr/bin/env python3
"""F15.3 (tickets 14 and 15) ground truth of attenuated, coloured shadows through transmissive casters: the irradiance of a
diffuse RECEIVER (the floor) lit by one light (the sun, or one point light) through planar glass / water casters, from the
exported snapshot of a `transmissive-shadow-*` scenario (src/testbench/transmissive_shadow_validation.*).

What is compared and how (decision 11 transmission bound on the RECEIVER mask, ADR-0004 amendment):
- Receiver pixels: primary hits (pixel-centre rays, tools/f15_reference.py `cast_primary_rays`) on the floor, i.e. the single large
  horizontal plane of the scene.  The camera never sees the floor through a caster (bench test), so the primary hit is the floor.
- Unshadowed radiance: tools/f15_reference.py `sun_radiance` / `point_radiance` (the Appendix B model the engine implements; the
  floor has no extension, roughness 1, no environment, no GI).  The shadowed radiance is that value times the RGB transmittance
  T(P) of the light path from the receiver point P to the light (diffuse and specular lobes see the same visibility).
- SHADOW REGIONS (the mask): the receiver pixels whose shadow ray (towards the light centre) first hits caster k AND whose 8
  neighbours on a ring of world radius `--margin` around P in the floor plane (shadow rays recomputed) also first hit caster k.  The
  margin is the declared filtering envelope of the technique under test (RT: sun penumbra + the shadow mask filter; CSM: the
  PCSS filter disk and the bias slop); pixels nearer a shadow boundary, or covered by no caster, are not gated.  The gate is the
  decision-11 transmission bound per caster region, with the absolute floor 0.05 * mean over the whole mask of the GROUND TRUTH
  (also for the controls and the pending twin: a candidate is never judged against a floor derived from itself).
- Ground truth T(P): Mitsuba 3.9.1.  Rays leave P inside a cone around the light, are followed through the casters with Mitsuba's
  own BSDFs (`dielectric` for closed solids, `gltf_thin` for thin-walled panes, the `gltf_absorbing` Beer-Lambert wrapper for the
  interior medium; all of tools/f15_transmission_reference.py), refract and reflect with Mitsuba's exact Fresnel, and are
  weighted by the BSDF sample weights; a ray that ESCAPES the scene contributes its throughput when it hits the light (the sun
  disk: inside the angular radius; a point light: a stand-in emitter sphere of radius POINT_LIGHT_RADIUS around it).  The floor
  and every opaque object are black absorbers (an opaque caster has T = 0; a path that reflects back to the floor dies: the
  engine has no secondary bounce of the shadow ray).  The sampling cone is CONE_FACTOR_* times wider than the light's (1 for the
  sun: a parallel slab returns every ray to the sun's direction; 1.3 for a point light's stand-in sphere: the exit ray is shifted
  sideways by millimetres, the extra directions collect the rays that the shift moves into the source) and the estimator is
  weighted by the solid-angle ratio (the exact irradiance fraction of a source of that size).  The reflect/refract choice at each
  interface comes from a per-depth radical-inverse sequence over the cone sample index (a low-discrepancy prefix: the rays that
  reach the light are the first samples of the spiral), randomly shifted per receiver point.  This IS the physical answer for planar casters: through a parallel slab the
  irradiance at P equals the straight-line transmittance up to the lateral shift of the ray (mm for these slabs), the model
  the engine declares.  Caustics (curved casters) are not modelled by the engine: the curved key is report only and its
  reference is the declared model (`--no-ground-truth`).
- DECLARED MODEL (the analytic straight-line model of ADR-0004, independent of Mitsuba's BSDFs; Mitsuba supplies only the
  geometry: the list of surfaces the straight shadow ray crosses): per transmissive instance k, the first surface it meets is the
  ENTRY surface with factor transmissionFactor * tint * (1 - F), every further surface of the instance an interior interface with
  factor (1 - F) (orchestrator decision of ticket 13: factor and tint at the entry surface only); F = Schlick(f0(ior), cos of the
  STRAIGHT ray to the surface normal); Beer-Lambert exp(-sigma_t * L) with L = signed sum of the entry/exit distances along the straight
  ray (entry surfaces count -t, exit surfaces +t: order independent), sigma_t = -ln(attenuationColor) / attenuationDistance;
  an opaque caster gives 0.  `model-differences` (and `model.pfm` of every render) states its difference with the Mitsuba truth.

Negative controls (images of defective models; each must exceed the gate on the fixture it is listed for):
  shadow-ignored (T = 1: no shadow; the curved key), transmittance-ignored (opaque casters: today's engine; T = 0 behind any caster), tint-ignored (the declared model with tint white),
  fresnel-ignored (F = 0), beer-lambert-ignored (no medium), double-counted (the declared transmittance applied twice).

Usage:
  uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy python -I tools/f15_shadow_reference.py render SNAPSHOT --out DIR --margin M [--controls ...] [--samples N]
  ... model-differences SNAPSHOT --margin M
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from f12_reference import np_module, scene_digest, snapshot, write_pfm  # noqa: E402
from f15_common import sha256_file  # noqa: E402
import f15_transmission_reference as tref  # noqa: E402

np = np_module()

SAMPLES = 512              # cone samples per receiver pixel, in two independent halves (the noise estimate)
CONE_FACTOR_SUN = 1.0      # sampling cone half angle / light angular radius: a parallel slab returns every ray to the sun's direction
CONE_FACTOR_POINT = 1.3    # a point light: the lateral shift of the exit ray moves the target by millimetres, the margin covers it
POINT_LIGHT_RADIUS = 0.08  # m, the stand-in sphere of a point light (a ray escaping the casters must find it; >> the lateral shift of 1 cm, << the casters)
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)  # bases of the radical-inverse sequence of each path depth
MAX_BOUNCES = 12           # path vertices followed through the casters (a slab needs 2 to 6)
MAX_HITS = 10              # surfaces a straight shadow ray may cross (declared model)
SHADOW_EPS = 1e-3          # m, shadow ray origin offset along the floor normal
LIT_BAND = 0.5            # m, the declared LIT reference region of the ratio gate: floor outside every shadow and envelope, within this distance of the shadows
RING = 8                   # ring neighbours of the region test
FLOOR_MIN_SCALE = 5.0      # the receiver is the instance whose world X and Z scales are both at least this
GOLDEN_ANGLE = 2.39996322972865332

CONTROLS = ("shadow-ignored", "transmittance-ignored", "tint-ignored", "fresnel-ignored", "beer-lambert-ignored", "double-counted")

MODEL = {
    "reference": "Mitsuba 3.9.1: rays from each receiver point through the casters (Mitsuba BSDFs and exact Fresnel), sun disk / point-light "
                 "stand-in sphere as the target, stratified reflect/refract choice, black floor and opaque objects",
    "radiance": "tools/f15_reference.py sun_radiance / point_radiance of the floor (Appendix B) times the RGB transmittance T(P)",
    "declared_model": "straight shadow ray: entry surface transmissionFactor * tint * (1 - F), interior interfaces (1 - F), F Schlick at the "
                      "straight-ray angle, Beer-Lambert from the signed sum of entry/exit distances, opaque casters 0",
    "mask": "floor pixels whose shadow ray and 8 ring neighbours at the world margin first hit the same caster",
    "floor": "ground-truth mean of the mask * 0.05",
}


# --- scene description ---------------------------------------------------------------------------------------------------

def instance_scales(data, k):
    return np.linalg.norm(tref.instance_matrix(data["instances"][k])[:3, :3], axis=0)


def floor_instance(data):
    """The receiver: the single instance whose world X and Z scales are both >= FLOOR_MIN_SCALE (the large floor plane)."""
    found = [k for k in range(len(data["instances"]))
             if instance_scales(data, k)[0] >= FLOOR_MIN_SCALE and instance_scales(data, k)[2] >= FLOOR_MIN_SCALE]
    if len(found) != 1:
        raise ValueError(f"the scene must hold exactly one large floor plane as the receiver (found {len(found)})")
    return found[0]


def caster_table(data, glass, floor):
    """{instance: description} of every non-floor instance: the glass entries of tref.glass_table plus the opaque casters."""
    out = {}
    for k, instance in enumerate(data["instances"]):
        if k == floor:
            continue
        material = data["materials"][instance["material"]]
        if any(v > 0 for v in material["emissive"]):
            raise ValueError("emissive objects are outside the transmissive-shadow model")
        if k in glass:
            out[k] = {"instance": k, "opaque": False, **glass[k]}
        else:
            out[k] = {"instance": k, "opaque": True, "label": "opaque", "thin": False, "base": [float(v) for v in material["base"][:3]],
                      "roughness": float(material["roughness"]), "ior": 1.5, "transmission": 0.0, "thickness": 0.0,
                      "attenuation_color": [1.0, 1.0, 1.0], "attenuation_distance": math.inf, "has_medium": False,
                      "sigma_t": np.zeros(3)}
    return out


def single_light(data):
    """The one light of the scene: {"kind": "sun", toward, radius, irradiance} or {"kind": "point", position, range, intensity, color}."""
    lights = data["lights"]
    if len(lights) != 1 or lights[0]["type"] not in (0, 1):
        raise ValueError("the transmissive-shadow model needs exactly one light, the sun or one white point light")
    light = lights[0]
    color = np.asarray(light["color"], dtype=np.float64)
    if not (color[0] == color[1] == color[2]):
        raise ValueError("the light must be white")
    if any(float(v) > 0 for v in data["sky"]):
        raise ValueError("the environment must be 0 (only the light reaches the receiver)")
    if light["type"] == 0:
        radius = float(data.get("sun_angular_radius", 0.0))
        if radius <= 0:
            raise ValueError("a sun requires a positive angular radius")
        direction = np.asarray(light["direction"], dtype=np.float64)
        return {"kind": "sun", "toward": -direction / np.linalg.norm(direction), "radius": radius,
                "irradiance": color * float(light["intensity"]), "raw": light}
    return {"kind": "point", "position": np.asarray(light["position"], dtype=np.float64), "raw": light}


def light_axes(light, points):
    """(unit directions towards the light (n, 3), angular radius of the light per point (n,)) for receiver points."""
    n = len(points)
    if light["kind"] == "sun":
        return np.broadcast_to(light["toward"], (n, 3)).copy(), np.full(n, light["radius"])
    delta = light["position"][None, :] - points
    distance = np.linalg.norm(delta, axis=1)
    return delta / distance[:, None], np.arcsin(np.minimum(POINT_LIGHT_RADIUS / distance, 1.0))


# --- Mitsuba helpers -----------------------------------------------------------------------------------------------------------

def to_mi(mi, array):
    """(n, 3) float array -> Point3f/Vector3f components as Float arrays."""
    array = np.ascontiguousarray(array, dtype=np.float32)
    return mi.Float(array[:, 0].copy()), mi.Float(array[:, 1].copy()), mi.Float(array[:, 2].copy())


def make_ray(mi, origin, direction):
    ox, oy, oz = to_mi(mi, origin)
    dx, dy, dz = to_mi(mi, direction)
    return mi.RayDifferential3f(mi.Point3f(ox, oy, oz), mi.Vector3f(dx, dy, dz))


def np3(value):
    return np.stack([np.asarray(value.x), np.asarray(value.y), np.asarray(value.z)], axis=1).astype(np.float64)


def instance_of_shapes(scene, mi, dr):
    """{shape id (uint32 of the shape pointer): instance index} of the scene built by tref.build_scene (shapes named instance_k)."""
    identifiers = np.asarray(dr.reinterpret_array(mi.UInt32, scene.shapes_dr()))
    return {int(i): int(shape.id().split("_")[1]) for shape, i in zip(scene.shapes(), identifiers)}


def hit_instance(si, dr, mi, index_of, count):
    valid = np.asarray(si.is_valid())
    ids = np.asarray(dr.reinterpret_array(mi.UInt32, si.shape)).astype(np.int64)
    out = np.full(count, -1, dtype=np.int64)
    for key, instance in index_of.items():
        out[valid & (ids == key)] = instance
    return out


def first_hit_instance(scene, mi, dr, index_of, origin, direction):
    """Instance index of the first surface hit by each ray, -1 for none."""
    si = scene.ray_intersect(make_ray(mi, origin, direction))
    return hit_instance(si, dr, mi, index_of, len(origin))


# --- regions ---------------------------------------------------------------------------------------------------------------------

def floor_frame(normals):
    """Two unit tangents of each floor normal."""
    helper = np.where(np.abs(normals[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    t1 = np.cross(helper, normals)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return t1, np.cross(normals, t1)


def shadow_regions(scene, mi, dr, index_of, light, position, normal, margin):
    """(class (n,), roi (n,)): class = instance first hit by the shadow ray of each receiver point (-1 none); roi = the class is a caster
    and the 8 ring neighbours at world distance `margin` in the floor plane have the same class."""
    direction, _ = light_axes(light, position)
    cls = first_hit_instance(scene, mi, dr, index_of, position + SHADOW_EPS * normal, direction)
    roi = cls >= 0
    pick = np.flatnonzero(roi)
    if margin > 0 and pick.size:
        t1, t2 = floor_frame(normal[pick])
        for k in range(RING):
            angle = 2.0 * math.pi * k / RING
            offset = margin * (math.cos(angle) * t1 + math.sin(angle) * t2)
            moved = position[pick] + offset
            d, _ = light_axes(light, moved)
            neighbour = first_hit_instance(scene, mi, dr, index_of, moved + SHADOW_EPS * normal[pick], d)
            roi[pick] &= neighbour == cls[pick]
    return cls, roi


def lit_region(scene, mi, dr, index_of, light, position, normal, cls, margin):
    """Boolean (n,): the declared lit reference region next to the shadows: receiver points whose shadow ray and 8 ring neighbours
    (world margin) hit nothing, inside the x/z bounding box of the shadow footprints grown by LIT_BAND."""
    shadowed = cls >= 0
    if not shadowed.any():
        return np.zeros(len(position), dtype=bool)
    low = position[shadowed][:, [0, 2]].min(axis=0) - LIT_BAND
    high = position[shadowed][:, [0, 2]].max(axis=0) + LIT_BAND
    inside = (cls < 0) & (position[:, 0] >= low[0]) & (position[:, 0] <= high[0]) & (position[:, 2] >= low[1]) & (position[:, 2] <= high[1])
    pick = np.flatnonzero(inside)
    keep = inside.copy()
    if pick.size:
        t1, t2 = floor_frame(normal[pick])
        for k in range(RING):
            angle = 2.0 * math.pi * k / RING
            moved = position[pick] + margin * (math.cos(angle) * t1 + math.sin(angle) * t2)
            d, _ = light_axes(light, moved)
            keep[pick] &= first_hit_instance(scene, mi, dr, index_of, moved + SHADOW_EPS * normal[pick], d) < 0
    return keep


# --- the Mitsuba estimator -------------------------------------------------------------------------------------------------------

def cone_directions(axis, half_angle, count, phase):
    """(n, count, 3) unit directions: a golden-angle spiral, uniform in solid angle over the cone of `half_angle` around `axis`."""
    n = len(axis)
    helper = np.where(np.abs(axis[:, 2:3]) < 0.999, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 1.0, 0.0]]))
    t = np.cross(helper, axis)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    b = np.cross(axis, t)
    k = np.arange(count, dtype=np.float64)[None, :]
    z = 1.0 - (k + 0.5) / count * (1.0 - np.cos(half_angle))[:, None]
    s = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = GOLDEN_ANGLE * k + phase
    return (t[:, None, :] * (s * np.cos(phi))[..., None] + b[:, None, :] * (s * np.sin(phi))[..., None] + axis[:, None, :] * z[..., None])


def escaped_hits_light(light, origin, direction):
    """1.0 for the escaped rays that hit the light: the sun disk (angle to the centre <= radius) or the stand-in sphere of a point light."""
    if light["kind"] == "sun":
        return (direction @ light["toward"] >= np.cos(light["radius"])).astype(np.float64)
    oc = origin - light["position"][None, :]
    b = np.einsum("ij,ij->i", oc, direction)
    c = np.einsum("ij,ij->i", oc, oc) - POINT_LIGHT_RADIUS ** 2
    disc = b * b - c
    return ((disc >= 0) & (-b - np.sqrt(np.maximum(disc, 0.0)) > 0)).astype(np.float64)  # the origin is outside the sphere


def radical_inverse(index, base):
    """Van der Corput radical inverse of integer indices in `base`."""
    result = np.zeros(len(index))
    factor = 1.0 / base
    i = np.asarray(index, dtype=np.int64).copy()
    while i.any():
        result += (i % base) * factor
        i //= base
        factor /= base
    return result


def trace_cone_batch(scene, mi, dr, light, origin, axis, half_angle_light, samples, rng, phase):
    """Transmittance estimate (n, 3) of `origin` -> light through the scene, `samples` cone directions per point."""
    n = len(origin)
    factor = CONE_FACTOR_SUN if light["kind"] == "sun" else CONE_FACTOR_POINT
    half_big = np.minimum(factor * half_angle_light, 1.2)
    shift = rng.random((n, len(PRIMES)))
    directions = cone_directions(axis, half_big, samples, phase).reshape(n * samples, 3)
    lane = np.arange(n * samples)
    o = np.repeat(origin, samples, axis=0)
    d = directions
    throughput = np.ones((n * samples, 3))
    total = np.zeros((n * samples, 3))
    sample_index = lane % samples
    alive = lane.copy()
    ctx = mi.BSDFContext()
    for depth in range(MAX_BOUNCES):
        if not alive.size:
            break
        ray = make_ray(mi, o, d)
        si = scene.ray_intersect(ray)
        hit = np.asarray(si.is_valid())
        miss = ~hit
        if miss.any():
            weight = escaped_hits_light(light, o[miss], d[miss])
            total[alive[miss]] += throughput[miss] * weight[:, None]
        if not hit.any():
            break
        u1 = (radical_inverse(sample_index[alive], PRIMES[depth]) + shift[alive // samples, depth]) % 1.0
        u2 = rng.random((alive.size, 2))
        bsdf = si.bsdf(ray)
        bs, value = bsdf.sample(ctx, si, mi.Float(u1.astype(np.float32)),
                                mi.Point2f(mi.Float(u2[:, 0].astype(np.float32)), mi.Float(u2[:, 1].astype(np.float32))), mi.Bool(hit))
        weight = np.stack([np.asarray(value[c]) for c in range(3)], axis=1).astype(np.float64)
        next_ray = si.spawn_ray(si.to_world(bs.wo))
        new_o, new_d = np3(next_ray.o), np3(next_ray.d)
        pdf = np.asarray(bs.pdf)
        keep = hit & (pdf > 0) & (weight.max(axis=1) > 0) & np.isfinite(weight).all(axis=1)
        throughput = (throughput * weight)[keep]
        o, d = new_o[keep], new_d[keep]
        alive = alive[keep]
    solid_angle_ratio = (1.0 - np.cos(half_big)) / (1.0 - np.cos(half_angle_light))
    estimate = total.reshape(n, samples, 3).mean(axis=1) * solid_angle_ratio[:, None]
    return estimate


def mitsuba_transmittance(scene, mi, dr, light, position, normal, samples, seed, batch_rays=400_000):
    """(T (n, 3) from `samples` cone samples, half-sample noise (n, 3) = |T_a - T_b| / 2) at receiver points."""
    n = len(position)
    origin = position + SHADOW_EPS * normal
    axis, half = light_axes(light, position)
    half_samples = max(samples // 2, 1)
    per_chunk = max(1, batch_rays // half_samples)
    out_a, out_b = np.zeros((n, 3)), np.zeros((n, 3))
    for start in range(0, n, per_chunk):
        stop = min(n, start + per_chunk)
        for which, target in enumerate((out_a, out_b)):
            rng = np.random.default_rng(seed + 7919 * which + start)
            target[start:stop] = trace_cone_batch(scene, mi, dr, light, origin[start:stop], axis[start:stop], half[start:stop], half_samples, rng,
                                                  0.5 * which)
    return 0.5 * (out_a + out_b), 0.5 * np.abs(out_a - out_b)


# --- the declared straight-line model ---------------------------------------------------------------------------------------------

def straight_hits(scene, mi, dr, index_of, origin, direction, count=MAX_HITS):
    """Surfaces crossed by straight rays: (instance (n, H) -1 none, absolute distance (n, H), cosine of the ray with the geometric normal)."""
    n = len(origin)
    instance = np.full((n, count), -1, dtype=np.int64)
    distance = np.zeros((n, count))
    cosine = np.zeros((n, count))
    o = origin.copy()
    travelled = np.zeros(n)
    live = np.ones(n, dtype=bool)
    step = 1e-4
    for h in range(count):
        if not live.any():
            break
        si = scene.ray_intersect(make_ray(mi, o, direction))
        valid = np.asarray(si.is_valid()) & live
        ids = hit_instance(si, dr, mi, index_of, n)
        t = np.asarray(si.t).astype(np.float64)
        geometric = np3(si.n)
        point = np3(si.p)
        instance[valid, h] = ids[valid]
        distance[valid, h] = travelled[valid] + t[valid]
        cosine[valid, h] = np.einsum("ij,ij->i", direction[valid], geometric[valid])
        o = np.where(valid[:, None], point + step * direction, o)
        travelled = np.where(valid, travelled + t + step, travelled)
        live = valid
    return instance, distance, cosine


def declared_transmittance(hits, casters, variant="spec"):
    """(T (n, 3)) of the declared straight-line model; `variant` one of CONTROLS (or "spec")."""
    instance, distance, cosine = hits
    n = len(instance)
    t_total = np.ones((n, 3))
    any_caster = np.zeros(n, dtype=bool)
    for k, c in casters.items():
        mine = instance == k
        present = mine.any(axis=1)
        any_caster |= present
        if variant == "transmittance-ignored":
            continue
        if c["opaque"]:
            t_total[present] = 0.0
            continue
        eta = c["ior"]
        fresnel = np.zeros_like(cosine) if variant == "fresnel-ignored" else tref.fresnel_schlick(np.abs(cosine), eta)
        factor = np.where(mine, 1.0 - fresnel, 1.0)                       # every surface of the instance
        tint = np.ones(3) if variant == "tint-ignored" else np.asarray(c["base"], dtype=np.float64)
        entry_extra = c["transmission"] * tint                              # entry surface: transmission factor and tint
        t_k = np.prod(factor, axis=1)[:, None] * np.where(present[:, None], entry_extra[None, :], 1.0)
        if c["has_medium"] and variant != "beer-lambert-ignored" and not c["thin"]:
            length = np.where(mine, np.where(cosine > 0.0, distance, -distance), 0.0).sum(axis=1)  # exits +t, entries -t
            t_k = t_k * np.exp(-np.outer(np.maximum(length, 0.0), c["sigma_t"]))
        t_total = t_total * np.where(present[:, None], t_k, 1.0)
    if variant == "transmittance-ignored":
        t_total = np.where(any_caster[:, None], 0.0, 1.0) * np.ones((n, 3))
    if variant == "shadow-ignored":
        t_total = np.ones((n, 3))
    if variant == "double-counted":
        t_total = t_total * t_total
    return t_total


# --- command: render ----------------------------------------------------------------------------------------------------------------

def relative_stats(np_, candidate, reference, roi_ids, floor_value):
    """Per-caster mean / p95 of the relative error max over RGB (floor from the truth), for the model-differences report."""
    denominator = np_.maximum(reference, floor_value)
    error = (np_.abs(candidate - reference) / denominator).max(axis=1)
    out = {}
    for k in sorted(set(roi_ids.tolist())):
        e = error[roi_ids == k]
        out[str(k)] = {"mean": float(e.mean()), "p95": float(np_.percentile(e, 95)), "pixels": int(e.size)}
    return out


def prepare(args):
    import f15_reference as ref
    root, data = snapshot(args.snapshot)
    glass = tref.glass_table(data, allow_rough_thin=True)
    floor = floor_instance(data)
    casters = caster_table(data, glass, floor)
    light = single_light(data)
    cast = ref.cast_primary_rays(root, data, args.threads)
    valid, position, normal, instance, (width, height) = cast
    out = pathlib.Path(args.out).resolve() if getattr(args, "out", None) else None
    scene, mi, dr = tref.build_scene(root, data, glass, "", args.threads, out if out else pathlib.Path("."))
    index_of = instance_of_shapes(scene, mi, dr)
    return {"ref": ref, "root": root, "data": data, "glass": glass, "floor": floor, "casters": casters, "light": light, "cast": cast,
            "scene": scene, "mi": mi, "dr": dr, "index_of": index_of, "width": width, "height": height}


def unshadowed(context, pick):
    ref, data = context["ref"], context["data"]
    valid, position, normal, instance, _ = context["cast"]
    material = data["materials"][data["instances"][context["floor"]]["material"]]
    p = position[pick]
    n = ref.normalize(normal[pick])
    camera = np.asarray(data["camera"]["position"], dtype=np.float64)
    v = ref.normalize(camera[None, :] - p)
    base = np.tile(np.asarray(material["base"][:3], dtype=np.float64), (len(p), 1))
    metallic = np.zeros(len(p))
    roughness = np.full(len(p), float(material["roughness"]))
    light = context["light"]
    if light["kind"] == "sun":
        value, _ = ref.sun_radiance(base, metallic, roughness, n, v, light["toward"], light["irradiance"], light["radius"], ref.APPENDIX_B)
    else:
        value, _ = ref.point_radiance(base, metallic, roughness, n, v, p, light["raw"], ref.APPENDIX_B)
    return value, p, n


def command_render(args):
    context = prepare(args)
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data, casters, light = context["data"], context["casters"], context["light"]
    valid, position, normal, instance, _ = context["cast"]
    width, height = context["width"], context["height"]
    flat_instance = instance.reshape(-1)
    pick_floor = np.flatnonzero(valid.reshape(-1) & (flat_instance == context["floor"]))
    if not pick_floor.size:
        raise ValueError("the camera does not see the floor")
    radiance, p, n = unshadowed(context, pick_floor)
    scene, mi, dr, index_of = context["scene"], context["mi"], context["dr"], context["index_of"]
    cls, roi_floor = shadow_regions(scene, mi, dr, index_of, light, p, n, args.margin)
    roi_pick = np.flatnonzero(roi_floor)
    if not roi_pick.size:
        raise ValueError("no receiver pixel lies inside a shadow region of a caster (margin too wide, or the camera hides the shadows)")
    region_ids = cls[roi_pick]
    lit_pick = np.flatnonzero(lit_region(scene, mi, dr, index_of, light, p, n, cls, args.margin))
    if not lit_pick.size:
        raise ValueError("no lit reference region next to the shadows")
    direction, _ = light_axes(light, p[roi_pick])
    origin = p[roi_pick] + SHADOW_EPS * n[roi_pick]
    hits = straight_hits(scene, mi, dr, index_of, origin, direction)
    started = time.monotonic()
    if args.no_ground_truth:
        truth = declared_transmittance(hits, casters, "spec")
        noise = np.zeros_like(truth)
        kind = "declared-model"
    else:
        truth, noise = mitsuba_transmittance(scene, mi, dr, light, p[roi_pick], n[roi_pick], args.samples, args.seed)
        kind = "mitsuba"
    seconds = time.monotonic() - started
    shadowed = radiance[roi_pick] * truth
    floor_value = 0.05 * float(shadowed.mean())
    snapshot_digest = scene_digest(context["root"])

    def write_image(name, values):
        image = np.zeros((height * width, 3), dtype=np.float32)
        image[pick_floor] = radiance                       # outside the mask: the unshadowed floor (not gated)
        image[pick_floor[roi_pick]] = values
        write_pfm(out / f"{name}.pfm", image.reshape(height, width, 3))

    write_image("reference", shadowed)
    write_image("lit", radiance[roi_pick])   # the unshadowed floor radiance everywhere (T = reference / lit inside the mask)
    objects = []
    for k, c in sorted(casters.items()):
        count = int((region_ids == k).sum())
        objects.append({"instance": k, "label": c["label"], "pixels": count, "opaque": c["opaque"], "thin": c["thin"], "ior": c["ior"],
                        "base": c["base"], "transmission": c["transmission"], "thickness": c["thickness"],
                        "attenuation_color": c["attenuation_color"],
                        "attenuation_distance": c["attenuation_distance"] if math.isfinite(c["attenuation_distance"]) else None,
                        "roughness": c["roughness"]})
    relative_noise = (noise * radiance[roi_pick] / np.maximum(shadowed, floor_value)).max(axis=1)
    report = {"schema": 1, "tool": "tools/f15_shadow_reference.py", "tool_sha256": sha256_file(__file__), "model": MODEL,
              "snapshot_sha256": snapshot_digest, "frame": data.get("frame"), "kind": kind, "samples": args.samples, "margin_m": args.margin,
              "cone_factor": {"sun": CONE_FACTOR_SUN, "point": CONE_FACTOR_POINT}, "point_light_radius": POINT_LIGHT_RADIUS, "max_bounces": MAX_BOUNCES, "seed": args.seed,
              "light": {"kind": light["kind"], **({"toward": light["toward"].tolist(), "angular_radius": light["radius"]}
                                                  if light["kind"] == "sun" else {"position": light["position"].tolist()})},
              "floor_instance": context["floor"], "width": width, "height": height, "roi_pixels": int(roi_pick.size),
              "floor_pixels": int(pick_floor.size), "lit_pixels": int(lit_pick.size), "lit_band_m": LIT_BAND, "objects": objects, "controls": {}, "maximum_radiance": float(shadowed.max()),
              "terms": ["light-through-casters", "receiver-radiance"] + (["mitsuba-bsdf-paths"] if kind == "mitsuba" else ["declared-model"]),
              "reference": {"file": "reference.pfm", "seconds": seconds,
                            "noise": {"mean_relative_standard_error": float(relative_noise.mean()),
                                      "p95_relative_standard_error": float(np.percentile(relative_noise, 95))}},
              "interpretations": [
                  "transmission factor and tint at the entry surface only, interior crossings pure dielectric (ticket 13 orchestrator decision)",
                  "irradiance fraction of the light through the caster along Mitsuba's refracted paths; straight-line up to the lateral shift",
                  "reflected paths that return to the floor die (no secondary bounce of the shadow ray)"]}
    model = declared_transmittance(hits, casters, "spec")
    write_image("model", radiance[roi_pick] * model)
    report["model"] = {"file": "model.pfm", "vs_truth": relative_stats(np, radiance[roi_pick] * model, shadowed, region_ids, floor_value)}
    for name in args.controls:
        values = radiance[roi_pick] * declared_transmittance(hits, casters, name)
        write_image(f"control-{name}", values)
        report["controls"][name] = {"file": f"control-{name}.pfm"}
        print(json.dumps({"variant": name}), flush=True)
    np.savez_compressed(out / "masks.npz", roi=_scatter_mask(np, pick_floor[roi_pick], height, width),
                        sphere=_scatter_ids(np, pick_floor[roi_pick], region_ids, height, width),
                        glass=_scatter_mask(np, pick_floor[roi_pick], height, width),
                        lit=_scatter_mask(np, pick_floor[lit_pick], height, width))
    (out / "reference.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(json.dumps({"variant": "reference", "seconds": seconds, "roi_pixels": int(roi_pick.size), "kind": kind}), flush=True)
    return 0


def _scatter_mask(np_, flat_indices, height, width):
    mask = np_.zeros(height * width, dtype=bool)
    mask[flat_indices] = True
    return mask.reshape(height, width)


def _scatter_ids(np_, flat_indices, ids, height, width):
    out = np_.full(height * width, -1, dtype=np.int32)
    out[flat_indices] = ids
    return out.reshape(height, width)


# --- command: model-differences ------------------------------------------------------------------------------------------------------

def command_model_differences(args):
    args.out = None
    context = prepare(args)
    data, casters, light = context["data"], context["casters"], context["light"]
    valid, position, normal, instance, _ = context["cast"]
    pick_floor = np.flatnonzero(valid.reshape(-1) & (instance.reshape(-1) == context["floor"]))
    radiance, p, n = unshadowed(context, pick_floor)
    scene, mi, dr, index_of = context["scene"], context["mi"], context["dr"], context["index_of"]
    cls, roi_floor = shadow_regions(scene, mi, dr, index_of, light, p, n, args.margin)
    roi_pick = np.flatnonzero(roi_floor)
    direction, _ = light_axes(light, p[roi_pick])
    hits = straight_hits(scene, mi, dr, index_of, p[roi_pick] + SHADOW_EPS * n[roi_pick], direction)
    truth, _ = mitsuba_transmittance(scene, mi, dr, light, p[roi_pick], n[roi_pick], args.samples, args.seed)
    base = radiance[roi_pick]
    reference = base * truth
    floor_value = 0.05 * float(reference.mean())
    ids = cls[roi_pick]
    rows = {}
    for variant in ("spec",) + CONTROLS:
        rows[variant] = relative_stats(np, base * declared_transmittance(hits, casters, variant), reference, ids, floor_value)
    print(json.dumps({"declared_model_vs_mitsuba": rows, "floor": floor_value, "margin_m": args.margin}, indent=1))
    return 0


# --- command line -------------------------------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("render", help="reference image, shadow-region masks and negative controls of one snapshot")
    p.add_argument("snapshot")
    p.add_argument("--out", required=True)
    p.add_argument("--controls", nargs="*", default=[], choices=CONTROLS)
    p.add_argument("--samples", type=int, default=SAMPLES)
    p.add_argument("--margin", type=float, required=True, help="world radius (m) of the ring around each receiver point (the declared filtering envelope)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-ground-truth", action="store_true", help="use the declared straight-line model as the reference (curved casters: caustics are excluded)")
    p.add_argument("--threads", type=int, default=int(os.environ.get("F15_REFERENCE_THREADS", min(4, os.cpu_count() or 1))))
    p.set_defaults(function=command_render)
    p = sub.add_parser("model-differences", help="the declared straight-line model against the Mitsuba ground truth on the snapshot")
    p.add_argument("snapshot")
    p.add_argument("--margin", type=float, required=True)
    p.add_argument("--samples", type=int, default=SAMPLES)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=int(os.environ.get("F15_REFERENCE_THREADS", min(4, os.cpu_count() or 1))))
    p.set_defaults(function=command_model_differences)
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"F15 shadow reference failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
