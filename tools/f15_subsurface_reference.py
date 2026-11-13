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
  multiple-scatter    rho_ms_c = colour_c                           (record 2 words 2-4)
  single-scatter      s = 4.09712 + 4.20863 rho - sqrt(9.59217 + 41.6808 rho + 17.7126 rho^2)   (clamped to [0, 1]; rho = 0 -> s = 1)
  albedo              rho_ss_c = (1 - s^2) / (1 - g s^2)            (the Kulla-Conty 2017 inversion of the draft; g = word 9)
  scattering/absorp.  sigma_s = sigma_t rho_ss, sigma_a = sigma_t (1 - rho_ss)   (Mitsuba `homogeneous`: albedo = rho_ss, sigma_t)
  phase function     Henyey-Greenstein with the anisotropy g (Mitsuba `hg`: positive g = forward scattering)
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

GROUND_TRUTH_VERSION = 2
DEFAULT_THREADS = int(os.environ.get("F15_REFERENCE_THREADS", min(4, os.cpu_count() or 1)))
DEFAULT_SPP = int(os.environ.get("F15_SUBSURFACE_SPP", 4096))      # per grey render (two halves of spp / 2)
DEFAULT_POOL = int(os.environ.get("F15_SUBSURFACE_POOL", 5))       # odd window of the interior pooling
DEFAULT_DEPTH = 100000                                             # path depth bound (volpath counts surface and medium events)
RR_DEPTH = 64
SUBSURFACE_BIT = base.SUBSURFACE_FEATURE_BIT   # 128
VOLUME_BIT = 64                                # MaterialFeature::Volume
VOLUME_RECORD = 1                              # GPUMaterialExt1 words 11 thickness, 13-15 attenuation colour, 16 attenuation distance
BOUNDARY_IOR = 1.5
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


def medium_coefficients(block, radius=None):
    """Per-channel medium of a volumetric block: sigma_t (1/m), single-scatter albedo, sigma_s, sigma_a, mean free path (m), g."""
    radius = block["radius"] if radius is None else radius
    mfp = np.array([radius * s for s in block["radius_scale"]], dtype=np.float64)
    if (mfp <= 0).any():
        raise ValueError("a volumetric piece needs a positive mean free path in every channel")
    sigma_t = 1.0 / mfp
    albedo = single_scatter_albedo(block["color"], block["anisotropy"])
    return {"sigma_t": sigma_t, "albedo": albedo, "sigma_s": sigma_t * albedo, "sigma_a": sigma_t * (1.0 - albedo),
            "mfp": mfp, "g": float(block["anisotropy"])}


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
             "bsdf": {"type": "roughdielectric", "distribution": "ggx", "alpha": float(alpha), "int_ior": BOUNDARY_IOR,
                      "ext_ior": 1.0}}
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
    medium = medium_coefficients(block, radius)
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
    need = {"reference": {k: blocks[k]["radius"] for k in blocks}}
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
                                     "mfp": [b["radius"] * s for s in b["radius_scale"]]},
                         weight=b["weight"], radius=b["radius"], series=series_label(b), group=f"weight {b['weight']:g}")
    report = {"schema": 1, "tool": "tools/f15_subsurface_reference.py", "tool_sha256": sha256_file(__file__),
              "tool_sha256_base": sha256_file(base.__file__), "model": "volumetric subsurface: Mitsuba 3.9.1 volpath random walk",
              "ground_truth": {"version": GROUND_TRUTH_VERSION, "integrator": "volpath", "max_depth": DEFAULT_DEPTH, "rr_depth": RR_DEPTH,
                               "spp_per_grey_render": args.spp, "halves": 2, "pool_window": args.pool, "boundary_ior": BOUNDARY_IOR,
                               "boundary": "roughdielectric ggx, alpha = clamp-aware roughness^2 of the material",
                               "nominal_radius": nominal, "leak_radius": LEAK_RADIUS,
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
    p.add_argument("--cache-dir", default=None, help="cache of the per-piece renders (default $F15_REFERENCE_CACHE or build/f15-reference-cache)")
    p.set_defaults(function=command_render)
    args = parser.parse_args(argv)
    if args.pool % 2 == 0 or args.pool < 1:
        parser.error("--pool must be odd")
    try:
        return args.function(args)
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"F15 subsurface reference failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
