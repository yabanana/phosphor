#!/usr/bin/env python3
"""F15.3 (ticket 17) ground truth of the VOLUMETRIC subsurface mode: Mitsuba 3.9.1 volumetric path tracing.

The reference of decision 10 / ADR-0005: each subsurface piece of an exported snapshot (`exportOfflineReference` of the
material-validation bench) is rendered as a dielectric BOUNDARY (the engine's material roughness, GGX alpha = roughness^2 under the
engine's clamp, ior 1.5) enclosing a HOMOGENEOUS scattering medium, by a random walk (`volpath`).  Written from the OpenPBR semantics
of the decision and the draft KHR_materials_scatter text (research/khr-materials.md sections 10.3), never from the MSL or from any
diffusion profile; Mitsuba does the transport, the BSDF of the boundary and the ray casting, this file maps the parameters.

Medium mapping (per colour channel c, all of it documented here and tested in tools/test_f15_subsurface_reference.py):
  mean free path      mfp_c = radius * radiusScale_c                (metres, record 2 words 5 and 6-8)
  extinction          sigma_t_c = 1 / mfp_c                         (1/m; the OpenPBR "subsurface_radius" IS the mean free path)
  multiple-scatter    C_c = colour_c                                (record 2 words 2-4)
  single-scatter      alpha_c = the root of  A(alpha; g, alpha_b) = C_c  (boundary-aware inversion, next paragraph)
  albedo
  scattering/absorp.  sigma_s = sigma_t alpha, sigma_a = sigma_t (1 - alpha)   (Mitsuba `homogeneous`: albedo = alpha, sigma_t)
  phase function     Henyey-Greenstein with the anisotropy g (Mitsuba `hg`: positive g = forward scattering)

Boundary-aware inversion (orchestrator decision 2026-11-14, ticket 17).  OpenPBR defines subsurface_color as "the observed RGB
reflection albedo color taking into account all orders of multiple scattering": the fraction of the energy transmitted through the
interface that is transmitted back out and observed, E_multi-scatter = (1 - E_spec) C, for a semi-infinite homogeneous slab.  It does not
prescribe alpha(C); it says the closer the inversion is to that relation the closer it is to the ground truth.  The draft
KHR_materials_scatter uses the Kulla-Conty 2017 fit (`single_scatter_albedo`, kept as the draft's approximation: s = 4.09712 +
4.20863 rho - sqrt(9.59217 + 41.6808 rho + 17.7126 rho^2), alpha = (1 - s^2) / (1 - g s^2)), which like van de Hulst ignores the
dielectric boundary: behind the ior 1.5 boundary of this reference light is trapped by internal reflection and absorbed, so the observed
albedo falls far below C (negative control in the tests: about 2x).  The ground truth therefore measures, with Mitsuba itself,
  A(alpha; g, alpha_b) = (R_total - R_spec) / (1 - R_spec)
for a very large homogeneous sphere (1000 mean free paths in radius, the same `roughdielectric` boundary builder as the pieces) under a
constant environment of radiance 1: the orthographic image of the sphere averaged over its disc is the cosine-weighted (diffuse) incidence
hemispherical albedo (disc area = cos-weighted hemisphere), R_total is that mean, R_spec the mean of the same sphere with an absorbing
medium (alpha = 0: the first-surface specular reflection alone) and 1 - R_spec the energy transmitted through the interface (the
OpenPBR E_spec).  The per-direction value follows from the radial position of the pixel (reciprocity: radiance seen at theta under a
constant environment = directional albedo at theta); the table stores the diffuse value and the values at 0, 45 and 70 degrees, the
measured angle dependence is the declared approximation of using the diffuse value (see `tools/f15_subsurface_albedo.json`,
regenerate with `f15_subsurface_reference.py albedo-table`).  alpha(C) is the monotone inversion (bisection on the table interpolated
linearly in sqrt(1 - alpha)).  MEASURED LIMIT: Mitsuba's `roughdielectric` is a single-scattering GGX BSDF and loses energy at every interaction, so a non-absorbing
medium (alpha = 1) returns only A_max(alpha_b): 0.980 (roughness 0.15), 0.948 (0.25), 0.876 (0.35), 0.717 (0.5), 0.503 (0.7) in the table (a longer
roulette depth, 20000, on a 30-mfp sphere reads 0.993, 0.960, 0.887, 0.724, 0.507; noise 0.0001-0.0004), measured with
Russian roulette from depth 5000 (the first table, roulette from 64, read 0.45 at 0.7: the long walks were under-sampled).  Colours above
A_max are not reachable by any medium under this boundary and saturate at alpha = 1 (reported per piece in reference.json `albedo_target` /
`albedo_reached` / `albedo_saturated`).  The volumetric fixtures therefore use roughness 0.35, the largest roughness whose A_max exceeds the
brightest fixture colour (0.85) by 3 table sigmas (orchestrator decision 2026-11-14); roughness 0.7 subsurface has no valid ground truth.

The volume block of the material (record 1: thickness, attenuation) only marks the object as a solid, as the validation of the
material description demands (volumetric subsurface requires a volume with thickness > 0); a non-neutral attenuation is refused.

Boundary: Mitsuba `roughdielectric` (GGX, alpha = effective alpha of the engine's clamp, int_ior 1.5, ext_ior 1), the material's own
roughness, as decision 11 demands.  Lights: the sun of the snapshot as a Mitsuba directional emitter of the same irradiance (a delta
source is exact for the diffuse transport; the specular term is the rough boundary's own reflection).  No environment (the fixtures are
sun-only), no shadows between pieces and no inter-reflection: every piece is rendered ALONE in its own scene (the engine's captures run
with `--shadows off`, the analytic references have no inter-reflection).

Composition with the base model (OpenPBR: base substrate = mix(diffuse, subsurface, weight) under one specular coat): the image of a
piece with weight w is  (1 - w) * OFF + w * SSS  where OFF is the base model of the same snapshot (tools/f15_reference.py, Appendix B,
the `subsurface-off` control) and SSS the Mitsuba render of the weight-1 piece (it contains the boundary's specular reflection, as OFF
contains the Appendix B one).  The mix is linear in w, so one random walk per distinct medium serves every weight.

Convergence: a medium is rendered per colour channel as a GREY medium (sigma_t, albedo of that channel) because the chromatic estimator of
`volpath` has fireflies; the sun reaches the medium only through the boundary BTDF at the boundary vertices of the walk, so the walk is
noisy (relative per-path deviation about 5 with the default roughness).  Each grey render is split in two independent halves (seeds); the
halves give the noise of the reference: reference.json holds, per piece, the expected mean and p95 relative luminance error that the
residual noise alone would produce against the gate's absolute floor.  The image is smooth at the scale of the mean free path, so the
mean of a `pool` x `pool` window of the piece's interior pixels is the reference value (default 5 x 5; the window never crosses the
silhouette: only pixels whose 8 neighbours belong to the same piece are averaged); the pooling is declared in the report and a negative
control of the unit tests shows it moves a noise-free analytic image by less than the gate's bound.

Controls (each rendered on the reference side; the runner requires the gate to FAIL for every one):
  subsurface-off    the base model (subsurface ignored: what the engine does today with a volumetric block)
  diffusion-off     the colour replaces the diffuse albedo but the irradiance is not diffused (weight-mix of base colour and colour)
  radius-ignored    every piece rendered with the NOMINAL radius (median of the subsurface radii of the scene): the sweep is lost
  subsurface-leak   every piece rendered with LEAK_RADIUS (a sub-pixel radius that is not skipped)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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
import f15_reference as base  # noqa: E402

np = np_module()

GROUND_TRUTH_VERSION = 4
DEFAULT_THREADS = int(os.environ.get("F15_REFERENCE_THREADS", min(4, os.cpu_count() or 1)))
DEFAULT_SPP = int(os.environ.get("F15_SUBSURFACE_SPP", 2048))      # per grey render (two halves of spp / 2)
DEFAULT_POOL = int(os.environ.get("F15_SUBSURFACE_POOL", 5))       # odd window of the interior pooling
DEFAULT_DEPTH = 100000                                             # path depth bound (volpath counts surface and medium events)
RR_DEPTH = 1000                                                    # Russian roulette from this depth on (64 under-sampled the long walks near alpha = 1)
SUBSURFACE_BIT = base.SUBSURFACE_FEATURE_BIT   # 128
VOLUME_BIT = 64                                # MaterialFeature::Volume
VOLUME_RECORD = 1                              # GPUMaterialExt1 words 11 thickness, 13-15 attenuation colour, 16 attenuation distance
BOUNDARY_IOR = 1.5
DIFFUSION_REGIME_FRACTION = 0.1                # orchestrator decision: gated only when max-channel mfp <= 10 % of the piece size (diameter)
LEAK_RADIUS = 0.08                             # metres: the radius of the subsurface-leak control (not sub-pixel at any fixture distance)
CONTROLS = ("subsurface-off", "diffusion-off", "radius-ignored", "subsurface-leak")
LUMA = (0.2126, 0.7152, 0.0722)


# --- parameter mapping ------------------------------------------------------------------------------------------------------

def single_scatter_albedo(rho_ms, g=0.0):
    """Single-scattering albedo of the draft KHR_materials_scatter (Kulla-Conty 2017) from the multiple-scattering colour."""
    rho = np.asarray(rho_ms, dtype=np.float64)
    s = 4.09712 + 4.20863 * rho - np.sqrt(9.59217 + 41.6808 * rho + 17.7126 * rho * rho)
    s = np.where(rho <= 0.0, 1.0, np.clip(s, 0.0, 1.0))
    return (1.0 - s * s) / (1.0 - g * s * s)


ALBEDO_TABLE_PATH = REPO / "tools" / "f15_subsurface_albedo.json"
MEASURE_SIGMA_T = 100.0         # extinction of the measurement sphere of radius 1: 100 mean free paths (A is stable from 10 to 100)
MEASURE_RR_DEPTH = 5000         # roulette depth of the measurement (64 / 1000 bias the plateau low: 0.44 / 0.49 against 0.507)
MEASURE_SPP = 64                # per half (two independent halves)
MEASURE_RES = 128
MEASURE_SEED = 20261114
# single-scatter albedos of the table, dense towards 1 (A is smooth in sqrt(1 - alpha))
ALBEDO_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995,
               0.9998, 0.9999, 0.99995, 0.99999, 0.999999, 1.0]
ANGLES_DEG = (0.0, 45.0, 70.0)
_table_cache = {}


def boundary_bsdf(alpha, ior=BOUNDARY_IOR):
    """The boundary of every medium of this reference (pieces and measurement spheres alike): GGX rough dielectric.  ior == 1 gives
    the index-matched control boundary (no interface at all)."""
    if ior == 1.0:
        return {"type": "null"}
    return {"type": "roughdielectric", "distribution": "ggx", "alpha": float(alpha), "int_ior": float(ior), "ext_ior": 1.0}


def measure_albedo_point(mi, dr, albedo, g, alpha, ior, spp, seed, res=MEASURE_RES, sigma_t=MEASURE_SIGMA_T):
    """Orthographic image of a sphere of radius 1 (homogeneous medium of the given single-scatter albedo behind the boundary) under a
    constant environment of radiance 1: float64 (res, res)."""
    transform = getattr(mi, "ScalarAffineTransform4f", None) or mi.ScalarTransform4f
    description = {
        "type": "scene", "integrator": {"type": "volpath", "max_depth": DEFAULT_DEPTH, "rr_depth": MEASURE_RR_DEPTH},
        "sensor": {"type": "orthographic", "to_world": transform().look_at(origin=[0, 0, 5], target=[0, 0, 0], up=[0, 1, 0]),
                   "sampler": {"type": "independent", "sample_count": int(spp)},
                   "film": {"type": "hdrfilm", "width": res, "height": res, "pixel_format": "rgb", "component_format": "float32",
                            "rfilter": {"type": "box"}}},
        "env": {"type": "constant", "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]}},
        "piece": {"type": "sphere", "radius": 1.0, "bsdf": boundary_bsdf(alpha, ior),
                  "interior": {"type": "homogeneous", "albedo": {"type": "rgb", "value": [float(albedo)] * 3},
                               "sigma_t": {"type": "rgb", "value": [1.0, 1.0, 1.0]}, "scale": float(sigma_t),
                               "phase": {"type": "hg", "g": float(g)}}}}
    image = mi.render(mi.load_dict(description), spp=int(spp), seed=int(seed))
    dr.eval(image)
    return np.array(image, dtype=np.float64)[..., 0]


def disc_radius(res=MEASURE_RES):
    ys, xs = np.mgrid[0:res, 0:res]
    return np.hypot((xs + 0.5 - res / 2.0) / (res / 2.0), (ys + 0.5 - res / 2.0) / (res / 2.0))


def angle_masks(res=MEASURE_RES):
    """Pixel masks of the disc (cosine-weighted diffuse incidence) and of the viewing angles 0, 45 and 70 degrees (sin theta = r)."""
    r = disc_radius(res)
    out = {"diffuse": r < 0.99, "0": r < 0.12}
    for deg in ANGLES_DEG[1:]:
        centre = math.sin(math.radians(deg))
        out[f"{deg:g}"] = (r >= centre - 0.03) & (r <= centre + 0.03)
    return out


def observed_albedos(total, spec, masks):
    """A = (R_total - R_spec) / (1 - R_spec) per mask from the mean images of the medium and of the absorbing medium."""
    return {k: (float(total[m].mean()) - float(spec[m].mean())) / (1.0 - float(spec[m].mean())) for k, m in masks.items()}


def measure_albedo_curve(g, alpha_b, ior, threads, albedos=ALBEDO_GRID, spp=MEASURE_SPP, log=None):
    """The observed albedo A over the single-scatter albedos for one (g, boundary).  Every point is two independent halves (spp each);
    the noise is the half-difference.  Returns a JSON-able dict."""
    mi, dr = mitsuba_modules(threads)
    masks = angle_masks()
    spec = [measure_albedo_point(mi, dr, 0.0, g, alpha_b, ior, spp, MEASURE_SEED + 101 * h) for h in range(2)]
    specs_mean = 0.5 * (spec[0] + spec[1])
    points = []
    for i, albedo in enumerate(albedos):
        halves = [measure_albedo_point(mi, dr, albedo, g, alpha_b, ior, spp, MEASURE_SEED + 7 + 977 * i + 101 * h) for h in range(2)]
        total = 0.5 * (halves[0] + halves[1])
        a_mean = observed_albedos(total, specs_mean, masks)
        a_halves = [observed_albedos(h, specs_mean, masks)["diffuse"] for h in halves]
        points.append({"albedo": albedo, "A": a_mean["diffuse"], "noise": abs(a_halves[0] - a_halves[1]) / 2.0,
                       "A_angle": {k: a_mean[k] for k in a_mean if k != "diffuse"}})
        if log:
            log(f"g {g} alpha_b {alpha_b} ior {ior} albedo {albedo}: A {a_mean['diffuse']:.4f} +- {points[-1]['noise']:.4f}")
    r_spec = observed_albedos(specs_mean, np.zeros_like(specs_mean), masks)
    return {"g": g, "alpha_b": alpha_b, "ior": ior, "R_spec": {k: v for k, v in r_spec.items()}, "points": points}


ALBEDO_CURVES_PATH = REPO / "tools" / "f15_subsurface_albedo_curves.json"   # ticket 18: more boundary roughnesses (`albedo-curves`)


def load_albedo_table():
    """The measured curves: the fixtures' table plus (when present) the curves of the other roughnesses (ticket 18, the rt-walk LUT)."""
    key = str(ALBEDO_TABLE_PATH)
    if key not in _table_cache:
        table = json.loads(ALBEDO_TABLE_PATH.read_text())
        if ALBEDO_CURVES_PATH.exists():
            table = {**table, "curves": table["curves"] + json.loads(ALBEDO_CURVES_PATH.read_text())["curves"]}
        _table_cache[key] = table
    return _table_cache[key]


def curve_for(g, alpha_b, ior=BOUNDARY_IOR, table=None):
    table = table or load_albedo_table()
    for curve in table["curves"]:
        if abs(curve["g"] - g) < 1e-6 and abs(curve["alpha_b"] - alpha_b) < 1e-4 and abs(curve["ior"] - ior) < 1e-6:
            return curve
    raise ValueError(f"no observed-albedo curve for g {g}, boundary alpha {alpha_b}, ior {ior}: regenerate "
                     f"{ALBEDO_TABLE_PATH.name} (f15_subsurface_reference.py albedo-table) for this combination")


def _monotone_curve(curve):
    """(x = sqrt(1 - alpha), A) with A made non-decreasing in alpha by weighted isotonic regression (pool adjacent violators, weight
    1 / max(noise, 0.003)^2): the noise of the points near the A_max plateau (heavy tails of the long walks, up to 0.04) cannot reverse
    the curve nor inflate A_max the way a running maximum would."""
    alphas = np.array([p["albedo"] for p in curve["points"]])
    values = [float(p["A"]) for p in curve["points"]]
    weights = [1.0 / max(float(p["noise"]), 0.003) ** 2 for p in curve["points"]]
    blocks = []  # [value, weight, count]
    for v, w in zip(values, weights):
        blocks.append([v, w, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    a = np.concatenate([np.full(c, v) for v, _, c in blocks])
    return np.sqrt(1.0 - alphas), a


def observed_albedo(alpha, curve):
    """A(alpha) interpolated from the table (linear in sqrt(1 - alpha))."""
    x, a = _monotone_curve(curve)
    return float(np.interp(math.sqrt(max(1.0 - float(alpha), 0.0)), x[::-1], a[::-1]))


def invert_observed_albedo(target, curve):
    """(alpha, reached A, saturated): the single-scatter albedo whose observed albedo is `target`, by bisection on the interpolated
    table.  A target above the largest A of the curve (energy lost by the boundary: A_max < 1) saturates at alpha = 1."""
    a_max = observed_albedo(1.0, curve)
    if target <= 0.0:
        return 0.0, 0.0, False
    if target >= a_max:
        return 1.0, a_max, bool(target > a_max)
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if observed_albedo(mid, curve) < target:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    return alpha, observed_albedo(alpha, curve), False


def medium_coefficients(block, radius=None, alpha_b=None):
    """Per-channel medium of a volumetric block: sigma_t (1/m), single-scatter albedo, sigma_s, sigma_a, mean free path (m), g, plus
    the target colour, the observed albedo it reaches and whether the boundary saturates it.  alpha_b = the boundary's GGX alpha."""
    radius = block["radius"] if radius is None else radius
    if alpha_b is None:
        raise ValueError("the boundary-aware inversion needs the boundary alpha (alpha_b)")
    mfp = np.array([radius * s for s in block["radius_scale"]], dtype=np.float64)
    if (mfp <= 0).any():
        raise ValueError("a volumetric piece needs a positive mean free path in every channel")
    sigma_t = 1.0 / mfp
    curve = curve_for(float(block["anisotropy"]), float(alpha_b))
    inverted = [invert_observed_albedo(float(c), curve) for c in block["color"]]
    albedo = np.array([i[0] for i in inverted])
    return {"sigma_t": sigma_t, "albedo": albedo, "sigma_s": sigma_t * albedo, "sigma_a": sigma_t * (1.0 - albedo),
            "mfp": mfp, "g": float(block["anisotropy"]), "albedo_target": [float(c) for c in block["color"]],
            "albedo_reached": [i[1] for i in inverted], "saturated": [i[2] for i in inverted]}


def material_volumetric(data, material):
    """The volumetric subsurface block of a material (dict) or None without the subsurface bit.  Raises on thin mode (ticket 16's model),
    textures, another feature, a volume block that attenuates, a weight outside [0, 1], a negative colour or a non-positive radius."""
    bits = int(material.get("feature_bits", 0))
    if not bits & SUBSURFACE_BIT:
        if bits:
            raise ValueError(f"material feature bits {bits:#x}: only the volumetric subsurface block is modelled here")
        return None
    if bits & ~(SUBSURFACE_BIT | VOLUME_BIT):
        raise ValueError(f"volumetric subsurface combined with another material feature ({bits:#x}) is outside the declared model")
    records = data["material_extensions"]["records"]
    words = records[int(material["extension_index"]) + base.SUBSURFACE_RECORD]
    if len(words) != base.EXTENSION_RECORD_WORDS:
        raise ValueError("a material extension record must hold 20 words")
    if int(words[0]) != 1:
        raise ValueError("a thin subsurface block belongs to the ticket 16 model (tools/f15_reference.py)")
    if any(int(words[k]) != base.INVALID for k in (10, 11, 12)):
        raise ValueError("a subsurface texture is set; the reference is texture-free")
    volume = records[int(material["extension_index"]) + VOLUME_RECORD]
    thickness, attenuation, distance = base.f32(volume[11]), [base.f32(volume[k]) for k in (13, 14, 15)], base.f32(volume[16])
    if not thickness > 0.0:
        raise ValueError("volumetric subsurface requires a volume with thickness > 0")
    if any(a != 1.0 for a in attenuation) and math.isfinite(distance):
        raise ValueError("the volume attenuation of a subsurface piece is not modelled (a non-neutral colour with a finite distance)")
    block = {"weight": base.f32(words[1]), "color": [base.f32(words[k]) for k in (2, 3, 4)], "radius": base.f32(words[5]),
             "radius_scale": [base.f32(words[k]) for k in (6, 7, 8)], "anisotropy": base.f32(words[9]),
             "thickness": thickness}
    if not 0.0 <= block["weight"] <= 1.0 or any(c < 0.0 for c in block["color"]):
        raise ValueError("subsurface weight outside [0, 1] or a negative colour")
    if not (block["radius"] > 0.0 and all(s > 0.0 for s in block["radius_scale"])):
        raise ValueError("a volumetric piece needs a positive radius and radius scale")
    if not -1.0 < block["anisotropy"] < 1.0:
        raise ValueError("subsurface anisotropy outside (-1, 1)")
    return block


def scene_blocks(data):
    """{instance index: block} of the volumetric pieces of the scene (weight > 0 only)."""
    out = {}
    for k, instance in enumerate(data["instances"]):
        block = material_volumetric(data, data["materials"][instance["material"]])
        if block is not None and block["weight"] > 0.0:
            out[k] = block
    return out


def off_data(data):
    """The snapshot with every volumetric block removed: the base model of the same pieces (the `subsurface-off` control)."""
    out = copy.deepcopy(data)
    for material in out["materials"]:
        if int(material.get("feature_bits", 0)) & SUBSURFACE_BIT:
            material["feature_bits"] = int(material["feature_bits"]) & ~(SUBSURFACE_BIT | VOLUME_BIT)
    return out


def diffusion_off_data(data, blocks):
    """The snapshot where the subsurface colour replaces the diffuse albedo by the weight mix and nothing is diffused."""
    out = off_data(data)
    taken = set()
    for k, block in blocks.items():
        index = data["instances"][k]["material"]
        if index in taken:
            raise ValueError("the diffusion-off control needs one material per volumetric piece")
        taken.add(index)
        w = block["weight"]
        mixed = [(1.0 - w) * data["materials"][index]["base"][c] + w * block["color"][c] for c in range(3)]
        out["materials"][index]["base"] = mixed + [out["materials"][index]["base"][3]]
    return out


def nominal_radius(blocks):
    """The median subsurface radius of the scene (the radius a sweep-blind implementation would use everywhere)."""
    return float(np.median([b["radius"] for b in blocks.values()]))


def series_label(block):
    """Pieces that differ only in radius (and weight is equal) share a sweep series."""
    return "weight {:g} colour {} scale {} g {:g}".format(block["weight"], ",".join(f"{c:g}" for c in block["color"]),
                                                          ",".join(f"{s:g}" for s in block["radius_scale"]), block["anisotropy"])


# --- Mitsuba -------------------------------------------------------------------------------------------------------------------

def mitsuba_modules(threads):
    import drjit as dr
    import mitsuba as mi
    mi.set_variant("llvm_ad_rgb")
    dr.set_flag(dr.JitFlag.FastMath, False)
    dr.set_thread_count(threads)
    return mi, dr


def camera_dict(mi, data, crop, spp):
    camera = data["camera"]
    if any(camera.get("jitter_pixels", [0, 0])):
        raise ValueError("the reference requires an unjittered camera")
    position = np.asarray(camera["position"], dtype=np.float64)
    transform = getattr(mi, "ScalarAffineTransform4f", None) or mi.ScalarTransform4f
    x0, y0, x1, y1 = crop
    return {"type": "perspective", "fov": math.degrees(camera["fov_y_radians"]), "fov_axis": "y",
            "near_clip": camera["near"], "far_clip": 1e30,
            "to_world": transform().look_at(origin=position.tolist(), target=(position + np.asarray(camera["direction"])).tolist(),
                                            up=camera["up"]),
            "sampler": {"type": "independent", "sample_count": spp},
            "film": {"type": "hdrfilm", "width": int(camera["width"]), "height": int(camera["height"]), "pixel_format": "rgb",
                     "component_format": "float32", "crop_offset_x": int(x0), "crop_offset_y": int(y0),
                     "crop_width": int(x1 - x0), "crop_height": int(y1 - y0), "rfilter": {"type": "box"}}}


def sun_emitters(data):
    out, count = {}, 0
    for light in data["lights"]:
        if light["type"] != 0:
            raise ValueError("the volumetric reference models the sun only (a delta source reaches the medium through the boundary)")
        irradiance = [float(c) * float(light["intensity"]) for c in light["color"]]
        out[f"sun_{count}"] = {"type": "directional", "direction": [float(c) for c in light["direction"]],
                               "irradiance": {"type": "rgb", "value": irradiance}}
        count += 1
    if not count:
        raise ValueError("the volumetric reference needs a sun")
    return out


def mesh_orientation(root, data, instance):
    """+1 when the world-space triangles of the instance wind counter-clockwise seen from OUTSIDE (positive signed volume, the engine's
    glTF convention), -1 when they wind the other way (Mitsuba then needs `flip_normals` for the medium to lie inside)."""
    vertices, faces = base.read_ply(root / data["meshes"][instance["mesh"]])
    p = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float64)
    world = base.instance_matrix(instance)
    p = p @ world[:3, :3].T + world[:3, 3]
    a, b, c = p[faces[:, 0]], p[faces[:, 1]], p[faces[:, 2]]
    volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0
    return 1 if volume > 0.0 else -1


def render_grey(root, data, instance_index, crop, alpha, sigma_t, albedo, g, spp, seed, threads, depth=DEFAULT_DEPTH, medium=True):
    """One Mitsuba render of one piece alone in its scene with a GREY medium: float32 (crop h, crop w, 3)."""
    mi, dr = mitsuba_modules(threads)
    transform = getattr(mi, "ScalarAffineTransform4f", None) or mi.ScalarTransform4f
    instance = data["instances"][instance_index]
    shape = {"type": "ply", "filename": str(root / data["meshes"][instance["mesh"]]), "face_normals": False,
             "flip_normals": mesh_orientation(root, data, instance) < 0,
             "to_world": transform(np.asarray(instance["world"], dtype=np.float64).reshape(4, 4, order="F")),
             "bsdf": boundary_bsdf(alpha)}
    if medium:
        shape["interior"] = {"type": "homogeneous", "albedo": {"type": "rgb", "value": [float(albedo)] * 3},
                             "sigma_t": {"type": "rgb", "value": [1.0, 1.0, 1.0]}, "scale": float(sigma_t),
                             "phase": {"type": "hg", "g": float(g)}}
    description = {"type": "scene",
                   "integrator": {"type": "volpath", "max_depth": int(depth), "rr_depth": RR_DEPTH},
                   "sensor": camera_dict(mi, data, crop, spp), "piece": shape, **sun_emitters(data)}
    scene = mi.load_dict(description)
    image = mi.render(scene, spp=spp, seed=int(seed))
    dr.eval(image)
    return np.array(image, dtype=np.float32)


def piece_crop(instance_grid, instance_index, margin=2):
    """(x0, y0, x1, y1) bounding box of a piece on the film, grown by `margin` pixels and clamped."""
    ys, xs = np.nonzero(instance_grid == instance_index)
    if not ys.size:
        raise ValueError(f"instance {instance_index} is not visible")
    h, w = instance_grid.shape
    return (max(int(xs.min()) - margin, 0), max(int(ys.min()) - margin, 0), min(int(xs.max()) + 1 + margin, w),
            min(int(ys.max()) + 1 + margin, h))


def interior_mask(instance_grid, instance_index):
    """Pixels of the piece whose 8 neighbours belong to the piece too (what the pooling is allowed to average)."""
    same = instance_grid == instance_index
    keep = same.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            keep &= np.roll(np.roll(same, dy, axis=0), dx, axis=1)
    keep[0, :] = keep[-1, :] = keep[:, 0] = keep[:, -1] = False
    return keep


def box_pool(image, valid, window):
    """Mean of `image` over the `window` x `window` neighbourhood restricted to `valid` pixels (NaN-free: 0 where none is valid)."""
    if window <= 1:
        return np.where(valid[..., None], image, 0.0)
    half = window // 2

    def box_sum(a):
        padded = np.pad(a, [(half + 1, half), (half + 1, half)] + [(0, 0)] * (a.ndim - 2))
        c = padded.cumsum(axis=0).cumsum(axis=1)
        h, w = a.shape[:2]
        return c[window:window + h, window:window + w] - c[:h, window:window + w] - c[window:window + h, :w] + c[:h, :w]

    total = box_sum(np.where(valid[..., None], image, 0.0))
    count = box_sum(valid.astype(np.float64))[..., None]
    return np.where(count > 0, total / np.maximum(count, 1.0), 0.0)


def cache_directory(arg=None):
    path = pathlib.Path(arg or os.environ.get("F15_REFERENCE_CACHE") or REPO / "build" / "f15-reference-cache")
    path.mkdir(parents=True, exist_ok=True)
    return path


def render_piece_channels(root, data, instance_index, crop, alpha, medium, spp, threads, cache, seed=1):
    """(halves a, halves b) float32 (crop h, crop w, 3): the two independent halves (spp / 2 each) of the three grey channel renders
    of one piece, cached by everything that determines them."""
    sun = sun_emitters(data)
    instance = data["instances"][instance_index]
    results = []
    for channel in range(3):
        key = {"v": GROUND_TRUTH_VERSION, "mitsuba": "3.9.1", "mesh": sha256_file(root / data["meshes"][instance["mesh"]]),
               "world": [round(float(x), 7) for x in instance["world"]], "camera": data["camera"], "crop": list(crop),
               "alpha": round(float(alpha), 9), "ior": BOUNDARY_IOR, "sigma_t": round(float(medium["sigma_t"][channel]), 7),
               "albedo": round(float(medium["albedo"][channel]), 9), "g": round(medium["g"], 9), "spp": int(spp), "seed": seed,
               "sun": sun, "depth": DEFAULT_DEPTH, "rr": RR_DEPTH}
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:24]
        path = cache / f"piece-{digest}.npz"
        if path.is_file():
            with np.load(path) as stored:
                results.append((stored["a"], stored["b"]))
            continue
        halves = []
        for half in range(2):
            image = render_grey(root, data, instance_index, crop, alpha, medium["sigma_t"][channel], medium["albedo"][channel],
                                medium["g"], max(spp // 2, 1), seed + 7919 * half, threads)
            halves.append(image[..., 0])  # grey scene: the three colour channels carry the same radiance
        np.savez_compressed(path, a=halves[0], b=halves[1])
        results.append((halves[0], halves[1]))
    a = np.stack([r[0] for r in results], axis=-1)
    b = np.stack([r[1] for r in results], axis=-1)
    return a, b


def luminance(image):
    return image[..., 0] * LUMA[0] + image[..., 1] * LUMA[1] + image[..., 2] * LUMA[2]


# --- rendering of the scene ----------------------------------------------------------------------------------------------------

def sss_image_of_piece(root, data, cast, k, block, radius, alpha, spp, pool, threads, cache):
    """(pooled SSS image placed on the full film, pooled halves for the noise estimate) of one piece with the given radius."""
    instance_grid = cast[3]
    crop = piece_crop(instance_grid, k)
    medium = medium_coefficients(block, radius, alpha)
    a, b = render_piece_channels(root, data, k, crop, alpha, medium, spp, threads, cache)
    x0, y0, x1, y1 = crop
    interior = interior_mask(instance_grid, k)[y0:y1, x0:x1]
    pooled_a, pooled_b = box_pool(a, interior, pool), box_pool(b, interior, pool)
    height, width = instance_grid.shape
    full = np.zeros((height, width, 3), dtype=np.float32)
    half_a, half_b = np.zeros_like(full), np.zeros_like(full)
    full[y0:y1, x0:x1] = 0.5 * (pooled_a + pooled_b)
    half_a[y0:y1, x0:x1], half_b[y0:y1, x0:x1] = pooled_a, pooled_b
    return full, half_a, half_b


def composite(off, sss_by_piece, blocks, instance_grid):
    """(1 - w) OFF + w SSS on the pixels of every subsurface piece."""
    image = off.copy()
    for k, block in blocks.items():
        mask = instance_grid == k
        w = block["weight"]
        image[mask] = (1.0 - w) * off[mask] + w * sss_by_piece[k][mask]
    return image


def noise_report(halves, blocks, masks, off_luma, weights_only=True):
    """Expected relative luminance error produced by the residual noise alone, per piece: mean and p95 of
    |Y_a - Y_b| / 2 / max(Y, floor) over the piece's ROI (|Y_a - Y_b| / 2 is the expected magnitude of the noise of the averaged image)
    where Y is the composite luminance with the weight applied (noise scales with w)."""
    out = {}
    for k, (half_a, half_b, composite_y) in halves.items():
        roi = masks["roi"] & (masks["sphere"] == k)
        w = blocks[k]["weight"]
        floor = 0.05 * float(composite_y[roi].mean())
        noise = w * np.abs(luminance(half_a) - luminance(half_b))[roi] / 2.0 / np.maximum(composite_y[roi], floor)
        out[k] = {"mean": float(noise.mean()), "p95": float(np.percentile(noise, 95)), "pixels": int(roi.sum())}
    return out


def command_render(args):
    root, data = snapshot(args.snapshot)
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if data["sky"][0] != 0 or data["sky"][1] != 0 or data["sky"][2] != 0:
        raise ValueError("the volumetric reference fixtures are sun-only (environment 0)")
    blocks = scene_blocks(data)
    plain = off_data(data)
    cast = base.cast_primary_rays(root, plain, args.threads)
    threads = args.threads
    cache = cache_directory(args.cache_dir)
    instance_grid = cast[3]
    # The base model of the same snapshot: the `subsurface-off` control and the weight-0 part of every piece.
    off, masks, spheres, terms = base.render_frame(root, plain, base.APPENDIX_B, threads=threads, cast=cast)
    alpha_of = lambda k: float(base.effective_alpha(data["materials"][data["instances"][k]["material"]]["roughness"]))  # noqa: E731
    started = time.time()
    sss, halves = {}, {}
    nominal = nominal_radius(blocks) if blocks else 0.0
    need = {} if args.no_ground_truth else {"reference": {k: blocks[k]["radius"] for k in blocks}}
    if "radius-ignored" in args.controls:
        need["radius-ignored"] = {k: nominal for k in blocks}
    if "subsurface-leak" in args.controls:
        need["subsurface-leak"] = {k: LEAK_RADIUS for k in blocks}
    images = {}
    for name, radii in need.items():
        pieces = {}
        for k, block in blocks.items():
            full, half_a, half_b = sss_image_of_piece(root, data, cast, k, block, radii[k], alpha_of(k), args.spp, args.pool, threads,
                                                      cache)
            pieces[k] = full
            if name == "reference":
                halves[k] = (half_a, half_b)
        images[name] = composite(off, pieces, blocks, instance_grid)
    reference = images.get("reference", off)
    reference_y = luminance(reference)
    noise = noise_report({k: (a, b, reference_y) for k, (a, b) in halves.items()}, blocks, masks, luminance(off))
    for k, entry in enumerate(spheres):
        if k in blocks:
            b = blocks[k]
            entry.update(subsurface={"weight": b["weight"], "color": b["color"], "radius": b["radius"],
                                     "radius_scale": b["radius_scale"], "anisotropy": b["anisotropy"],
                                     "mfp": [b["radius"] * s for s in b["radius_scale"]], "piece_size": b["thickness"],
                                     "single_scatter_albedo": [float(x) for x in medium_coefficients(b, None, alpha_of(k))["albedo"]],
                                     "albedo_reached": medium_coefficients(b, None, alpha_of(k))["albedo_reached"],
                                     "albedo_saturated": medium_coefficients(b, None, alpha_of(k))["saturated"],
                                     "mfp_max": max(b["radius"] * s for s in b["radius_scale"])},
                         diffusion_regime=bool(max(b["radius"] * s for s in b["radius_scale"])
                                               <= DIFFUSION_REGIME_FRACTION * b["thickness"] * (1.0 + 1e-4)),
                         weight=b["weight"], radius=b["radius"], series=series_label(b), group=f"weight {b['weight']:g}",
                         boundary_roughness=float(data["materials"][data["instances"][k]["material"]]["roughness"]))
    report = {"schema": 1, "tool": "tools/f15_subsurface_reference.py", "tool_sha256": sha256_file(__file__),
              "tool_sha256_base": sha256_file(base.__file__), "model": "volumetric subsurface: Mitsuba 3.9.1 volpath random walk",
              "ground_truth": {"version": GROUND_TRUTH_VERSION, "integrator": "volpath", "max_depth": DEFAULT_DEPTH, "rr_depth": RR_DEPTH,
                               "spp_per_grey_render": args.spp, "halves": 2, "pool_window": args.pool, "boundary_ior": BOUNDARY_IOR,
                               "boundary": "roughdielectric ggx, alpha = clamp-aware roughness^2 of the material",
                               "nominal_radius": nominal, "leak_radius": LEAK_RADIUS,
                               "ground_truth_rendered": not args.no_ground_truth,
                               "residual_noise_per_piece": {str(k): v for k, v in noise.items()},
                               "seconds": round(time.time() - started, 1)},
              "snapshot_sha256": scene_digest(root), "frame": data.get("frame"), "sun_angular_radius": data.get("sun_angular_radius"),
              "sky": data["sky"], "lights": data["lights"], "visibility": "none", "controls": {}, "deterministic": False,
              "terms": ["sun-delta-mitsuba-volpath" if blocks else "sun-disk"]}
    write_pfm(out / "reference.pfm", reference)
    np.savez_compressed(out / "masks.npz", **masks)
    controls = {"subsurface-off": off, "radius-ignored": images.get("radius-ignored"), "subsurface-leak": images.get("subsurface-leak")}
    if "diffusion-off" in args.controls:
        controls["diffusion-off"] = base.render_frame(root, diffusion_off_data(data, blocks), base.APPENDIX_B, threads=threads,
                                                      cast=cast)[0]
    for name in args.controls:
        image = controls.get(name)
        if image is None:
            raise ValueError(f"unknown or unavailable control {name}")
        write_pfm(out / f"control-{name}.pfm", image)
        report["controls"][name] = {"model_variant": name, "file": f"control-{name}.pfm"}
    report.update(spheres=spheres, width=reference.shape[1], height=reference.shape[0], maximum_radiance=float(reference.max()),
                  roi_pixels=int(masks["roi"].sum()), quadrature_mask_pixels=int(masks["narrow"].sum()))
    (out / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"reference": str(out / "reference.pfm"), "pieces": len(blocks), "roi_pixels": report["roi_pixels"],
                      "seconds": report["ground_truth"]["seconds"]}))
    return 0


FIXTURE_ROUGHNESS = 0.35        # volumetric fixtures (orchestrator decision 2026-11-14: the largest roughness whose A_max exceeds 0.85 + 3 sigma)
LEGACY_ROUGHNESS = 0.7          # the first fixtures: A_max 0.507, the reference cannot represent colours above it (kept as information)


def fixture_alpha_b():
    """GGX alpha of the fixtures' boundary (the roughness of every volumetric scene)."""
    return float(base.effective_alpha(FIXTURE_ROUGHNESS))


def command_albedo_table(args):
    """Measure the observed-albedo curves of the fixtures' (g, boundary) combinations, the index-matched control curve and A at
    alpha = 1 against the boundary roughness (0.7 included)."""
    alpha_b = fixture_alpha_b()
    started = time.time()
    curves = []
    log = lambda m: print(m, flush=True)  # noqa: E731
    for g in (0.0, 0.3):
        curves.append(measure_albedo_curve(g, alpha_b, BOUNDARY_IOR, args.threads, spp=args.spp, log=log))
    # index-matched control (g = 0, no interface; the grid stops where the walks need no roulette tuning): validates the measurement
    curves.append(measure_albedo_curve(0.0, alpha_b, 1.0, args.threads, albedos=[a for a in ALBEDO_GRID if a <= 0.99], spp=args.spp, log=log))
    mi, dr = mitsuba_modules(args.threads)
    masks = angle_masks()
    a_max = []
    for rough in (0.04, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0):
        alpha = float(base.effective_alpha(rough))
        row = {"roughness": rough, "alpha_b": alpha}
        for g in (0.0, 0.3):
            spec = np.mean([measure_albedo_point(mi, dr, 0.0, g, alpha, BOUNDARY_IOR, args.spp, MEASURE_SEED + 101 * h) for h in range(4)], axis=0)
            vals = [observed_albedos(measure_albedo_point(mi, dr, 1.0, g, alpha, BOUNDARY_IOR, args.spp, MEASURE_SEED + 977 * h), spec, masks)["diffuse"]
                    for h in range(4)]
            row[f"g{g:g}"] = {"A": float(np.mean(vals)), "noise": float(np.std(vals, ddof=1) / 2.0)}
        a_max.append(row)
        print("A at albedo 1", row, flush=True)
    table = {"schema": 2, "tool": "tools/f15_subsurface_reference.py albedo-table", "mitsuba": "3.9.1",
             "definition": "A = (R_total - R_spec) / (1 - R_spec): sphere of radius 1, sigma_t 100 (100 mfp), roughdielectric (or null for "
                           "ior 1) boundary, constant environment 1, orthographic view; 'A' = disc mean = cosine-weighted diffuse incidence; "
                           "'A_angle' = viewing/incidence angle 0, 45, 70 degrees (reciprocity)",
             "sigma_t": MEASURE_SIGMA_T, "spp_per_half": args.spp, "halves": 2, "resolution": MEASURE_RES, "seed": MEASURE_SEED,
             "integrator": {"type": "volpath", "max_depth": DEFAULT_DEPTH, "rr_depth": MEASURE_RR_DEPTH}, "seconds": round(time.time() - started, 1),
             "curves": curves, "A_at_albedo_1_by_roughness": a_max}
    pathlib.Path(args.out).write_text(json.dumps(table, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


def command_albedo_curves(args):
    """F15.1 (ticket 18): observed-albedo curves for more boundary roughnesses (the engine's rt-walk LUT, tools/f15_subsurface_lut.py).
    One curve per (roughness, g) is appended to --out after it is measured (resumable: a curve already in the file is skipped)."""
    out = pathlib.Path(args.out)
    doc = json.loads(out.read_text()) if out.exists() else {"schema": 2, "tool": "tools/f15_subsurface_reference.py albedo-curves", "mitsuba": "3.9.1",
                                                           "sigma_t": MEASURE_SIGMA_T, "spp_per_half": args.spp, "halves": 2, "resolution": MEASURE_RES,
                                                           "seed": MEASURE_SEED, "integrator": {"type": "volpath", "max_depth": DEFAULT_DEPTH, "rr_depth": MEASURE_RR_DEPTH},
                                                           "definition": load_albedo_table()["definition"], "curves": []}
    log = lambda m: print(m, flush=True)  # noqa: E731
    for rough in args.roughness:
        alpha = float(base.effective_alpha(rough))
        for g in args.g:
            if any(abs(c["g"] - g) < 1e-6 and abs(c["alpha_b"] - alpha) < 1e-4 and abs(c["ior"] - BOUNDARY_IOR) < 1e-6 for c in doc["curves"]):
                continue
            started = time.time()
            curve = measure_albedo_curve(g, alpha, BOUNDARY_IOR, args.threads, spp=args.spp, log=log)
            curve["roughness"] = rough
            curve["seconds"] = round(time.time() - started, 1)
            doc["curves"].append(curve)
            out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("render", help="ground-truth image, masks and controls of one exported snapshot")
    p.add_argument("snapshot")
    p.add_argument("--out", required=True)
    p.add_argument("--controls", nargs="*", default=[], choices=CONTROLS)
    p.add_argument("--visibility", choices=["none", "rays"], default="none", help="accepted for the runner's call; the sun is unshadowed")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--spp", type=int, default=DEFAULT_SPP, help="samples per grey render (split in two halves)")
    p.add_argument("--pool", type=int, default=DEFAULT_POOL, help="odd pooling window over the piece interior (1 = none)")
    p.add_argument("--no-ground-truth", action="store_true",
                   help="skip the random walk of the reference itself (reference.pfm = the base model; the masks and the controls that "
                        "need renders are still produced): for keys whose gate does not read the ground truth (the sub-pixel pair)")
    p.add_argument("--cache-dir", default=None, help="cache of the per-piece renders (default $F15_REFERENCE_CACHE or build/f15-reference-cache)")
    p.set_defaults(function=command_render)
    t = sub.add_parser("albedo-table", help="measure the observed-albedo table (boundary-aware inversion) with Mitsuba")
    t.add_argument("--out", default=str(ALBEDO_TABLE_PATH))
    t.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    t.add_argument("--spp", type=int, default=MEASURE_SPP, help="samples per half image")
    t.set_defaults(function=command_albedo_table)
    c = sub.add_parser("albedo-curves", help="measure observed-albedo curves for more roughnesses (ticket 18, the rt-walk LUT)")
    c.add_argument("--out", required=True)
    c.add_argument("--roughness", type=float, nargs="+", required=True)
    c.add_argument("--g", type=float, nargs="+", default=[0.0, 0.3])
    c.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    c.add_argument("--spp", type=int, default=MEASURE_SPP, help="samples per half image")
    c.set_defaults(function=command_albedo_curves)
    args = parser.parse_args(argv)
    if getattr(args, "pool", 1) % 2 == 0 or getattr(args, "pool", 1) < 1:
        parser.error("--pool must be odd")
    try:
        return args.function(args)
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"F15 subsurface reference failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
