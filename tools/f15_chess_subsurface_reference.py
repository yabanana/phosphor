#!/usr/bin/env python3
"""F15.3 (ticket 20): Mitsuba 3.9.1 ground truth of the TEXTURED volumetric subsurface of the chess kings and queens.

The pieces of the Open Chess Set carry the MaterialX values of the ticket (subsurface colour = base colour texture, mean free path = base colour
texture x scale 0.003 / 0.001 m, weight = 1: the `*_shared_scattering.jpg` weight map is not in the pinned Khronos subset, fixtures/chess/README.md,
roughness from the ORM texture).  A texture is a surface function, not a volume: the reference is the LOCALLY HOMOGENEOUS medium the engine
also assumes (the medium parameters of a pixel are those of its entry texel), made of K classes per piece:
  1. every pixel of a piece (a Mitsuba camera ray cast against the piece alone) gets its texel: colour C = colour x colour texture(uv), the
     mean free path per channel mfp_c = radius x radiusScale_c x radius texture_c(uv), the roughness r = roughness x ORM.g(uv) (clamped like the
     engine: [0.04, 1]; the boundary alpha is max(r^2, 0.002)), and the metallic factor (pixels with metallic > 0.05 are outside the ROI: the
     model has no metal);
  2. the pixels are clustered into K classes by k-means (deterministic seed) on (C_r, C_g, C_b, r); a class is the medium of its centre: colour,
     mean free path and roughness (snapped to the nearest roughness the boundary-aware albedo table measured, tools/f15_subsurface_lut.py);
  3. every class is rendered with Mitsuba `volpath` as the WHOLE piece: `roughdielectric` GGX boundary (ior 1.5, alpha of the class), homogeneous
     interior (sigma_t = 1 / mfp_c, the boundary-aware single-scatter albedo of the class colour, HG g), the sun as a directional emitter; one
     GREY medium per colour channel (the chromatic estimator has fireflies), two independent halves, cached by content;
  4. the image is each pixel's class render (its own channel renders), then averaged over the 5 x 5 interior window like the sphere reference
     (tools/f15_subsurface_reference.py box_pool; the window never crosses a silhouette).
The mapping of an exported texture to a pixel is the engine's own (uv, bilinear, origin top-left), taken from the snapshot's decoded texels; the
classes, their quantisation error and the snapped roughness are the declared approximations of this reference (reference.json lists them).
Everything rendered alone per piece (the sun and the camera of the snapshot, no other piece: the capture runs --shadows off).

  uv run --offline --no-project --python 3.13 --with mitsuba==3.9.1 --with numpy python -I tools/f15_chess_subsurface_reference.py render \
      SNAPSHOT --out DIR [--classes K] [--spp N] [--pool 5] [--threads 4] [--only MATERIAL_INDEX ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_subsurface_reference as sub  # noqa: E402
from f12_reference import read_pfm, snapshot, write_pfm  # noqa: E402
from f15_common import sha256_file  # noqa: E402
import f15_reference as base  # noqa: E402

np = sub.np
VERSION = 1
KNOT_ROUGHNESS = (0.15, 0.25, 0.35, 0.5, 0.7)   # the roughnesses of the albedo table (colour axis of tools/f15_subsurface_lut.py)
METALLIC_LIMIT = 0.05
MIN_ROUGHNESS, MIN_ALPHA = 0.04, 0.002


def piece_blocks(data):
    """{instance index: (block, material index)} of the subsurface pieces with their texture ids (weight tex, colour tex, radius tex)."""
    out = {}
    records = data["material_extensions"]["records"]
    for k, instance in enumerate(data["instances"]):
        m = data["materials"][instance["material"]]
        if not int(m.get("feature_bits", 0)) & sub.SUBSURFACE_BIT:
            continue
        words = records[int(m["extension_index"]) + base.SUBSURFACE_RECORD]
        if int(words[0]) != 1:
            raise ValueError("a thin subsurface block is not modelled here")
        out[k] = {"material": instance["material"], "weight": base.f32(words[1]), "color": [base.f32(words[i]) for i in (2, 3, 4)],
                  "radius": base.f32(words[5]), "radius_scale": [base.f32(words[i]) for i in (6, 7, 8)], "anisotropy": base.f32(words[9]),
                  "weight_tex": int(words[10]), "color_tex": int(words[11]), "radius_tex": int(words[12])}
    return out


def load_textures(root, data):
    return {t["id"]: read_pfm(root / t["rgb"]) for t in data["textures"]}


def sample_bilinear(image, uv):
    """Engine sampling: uv (n, 2), origin top-left, repeat, bilinear on texel centres; float32 (n, 3)."""
    h, w = image.shape[:2]
    x, y = uv[:, 0].astype(np.float64) * w - 0.5, uv[:, 1].astype(np.float64) * h - 0.5
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = (x - ix)[:, None], (y - iy)[:, None]
    def tex(dx, dy):
        return image[(iy + dy) % h, (ix + dx) % w]
    return (tex(0, 0) * (1 - fx) + tex(1, 0) * fx) * (1 - fy) + (tex(0, 1) * (1 - fx) + tex(1, 1) * fx) * fy


def piece_scene(mi, root, data, instance_index):
    transform = getattr(mi, "ScalarAffineTransform4f", None) or mi.ScalarTransform4f
    instance = data["instances"][instance_index]
    return mi.load_dict({"type": "scene", "piece": {
        "type": "ply", "filename": str(root / data["meshes"][instance["mesh"]]), "face_normals": False,
        "to_world": transform(np.asarray(instance["world"], dtype=np.float64).reshape(4, 4, order="F"))}})


def cast_piece(root, data, instance_index, threads):
    """(valid (h, w), uv (h, w, 2)) of the pixel-centre camera rays against the piece alone (Mitsuba)."""
    mi, dr = sub.mitsuba_modules(threads)
    scene = piece_scene(mi, root, data, instance_index)
    camera = data["camera"]
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
                   mi.Vector3f(mi.Float(direction[:, 0].copy()), mi.Float(direction[:, 1].copy()), mi.Float(direction[:, 2].copy())))
    si = scene.ray_intersect(ray)
    valid = np.asarray(si.is_valid())
    uv = np.stack([np.asarray(si.uv.x), np.asarray(si.uv.y)], axis=1).astype(np.float64)
    return valid.reshape(height, width), uv.reshape(height, width, 2)


def kmeans(features, k, seed=7, iterations=40):
    """Deterministic k-means (k-means++ seeding from a fixed generator): (labels, centres)."""
    rng = np.random.default_rng(seed)
    n = len(features)
    k = min(k, n)
    centres = [features[rng.integers(n)]]
    for _ in range(1, k):
        d = np.min([np.sum((features - c) ** 2, axis=1) for c in centres], axis=0)
        centres.append(features[rng.choice(n, p=d / d.sum())] if d.sum() > 0 else features[rng.integers(n)])
    centres = np.array(centres)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iterations):
        labels = np.argmin(((features[:, None, :] - centres[None]) ** 2).sum(axis=2), axis=1)
        new = np.array([features[labels == c].mean(axis=0) if (labels == c).any() else centres[c] for c in range(k)])
        if np.allclose(new, centres):
            break
        centres = new
    return labels, centres


def snap_roughness(r):
    return min(KNOT_ROUGHNESS, key=lambda knot: abs(knot - r))


def class_medium(color, mfp, roughness, g):
    """Mitsuba medium of one class: (alpha_b of the snapped roughness, sigma_t (3), single-scatter albedo (3), reached A, saturated)."""
    knot = snap_roughness(roughness)
    alpha_b = float(base.effective_alpha(knot))
    curve = sub.curve_for(float(g), alpha_b)
    inverted = [sub.invert_observed_albedo(float(c), curve) for c in color]
    return {"alpha_b": alpha_b, "roughness_knot": knot, "sigma_t": [1.0 / float(m) for m in mfp], "albedo": [i[0] for i in inverted],
            "albedo_reached": [i[1] for i in inverted], "saturated": [bool(i[2]) for i in inverted], "g": float(g)}


def render_class(root, data, instance_index, crop, medium, spp, threads, cache):
    """(a, b): the two halves, float32 (crop h, crop w, 3) of the three grey channel renders of one class (cached by content)."""
    sun = sub.sun_emitters(data)
    instance = data["instances"][instance_index]
    results = []
    for channel in range(3):
        key = {"v": VERSION, "gt": sub.GROUND_TRUTH_VERSION, "mesh": sha256_file(root / data["meshes"][instance["mesh"]]),
               "world": [round(float(x), 7) for x in instance["world"]], "camera": data["camera"], "crop": list(crop),
               "alpha": round(medium["alpha_b"], 9), "ior": sub.BOUNDARY_IOR, "sigma_t": round(medium["sigma_t"][channel], 6),
               "albedo": round(medium["albedo"][channel], 9), "g": round(medium["g"], 9), "spp": int(spp), "sun": sun, "rr": sub.RR_DEPTH}
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:24]
        path = cache / f"chess-{digest}.npz"
        if path.is_file():
            with np.load(path) as stored:
                results.append((stored["a"], stored["b"]))
            continue
        halves = []
        for half in range(2):
            image = sub.render_grey(root, data, instance_index, crop, medium["alpha_b"], medium["sigma_t"][channel], medium["albedo"][channel],
                                    medium["g"], max(spp // 2, 1), 1 + 7919 * half, threads)
            halves.append(image[..., 0])
        np.savez_compressed(path, a=halves[0], b=halves[1])
        results.append((halves[0], halves[1]))
    return np.stack([r[0] for r in results], axis=-1), np.stack([r[1] for r in results], axis=-1)


def command_render(args):
    root, data = snapshot(args.snapshot)
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = sub.cache_directory(args.cache_dir)
    blocks = piece_blocks(data)
    if args.only:
        blocks = {k: b for k, b in blocks.items() if data["instances"][k]["material"] in args.only}
    if not blocks:
        raise ValueError("no subsurface piece in the snapshot (or none selected)")
    textures = load_textures(root, data)
    scene_grid = base.cast_primary_rays(root, data, args.threads)[3]
    camera = data["camera"]
    height, width = int(camera["height"]), int(camera["width"])
    started = time.time()
    reference = np.zeros((height, width, 3), dtype=np.float32)
    half_a, half_b = np.zeros_like(reference), np.zeros_like(reference)
    grid = np.full((height, width), -1, dtype=np.int32)
    roi = np.zeros((height, width), dtype=bool)
    pieces = {}
    for k, block in blocks.items():
        material = data["materials"][block["material"]]
        alone, uv = cast_piece(root, data, k, args.threads)
        # the pixels of the piece: those where the snapshot's own cast (every instance) hits it (other pieces may occlude it)
        valid = alone & (scene_grid == k)
        grid[valid] = k
        ys, xs = np.nonzero(valid)
        crop = (max(int(xs.min()) - 2, 0), max(int(ys.min()) - 2, 0), min(int(xs.max()) + 3, width), min(int(ys.max()) + 3, height))
        uv_pixels = uv[valid]
        def tex_or(tid, fallback):
            return sample_bilinear(textures[tid], uv_pixels) if tid != base.INVALID else np.tile(np.asarray(fallback, dtype=np.float32), (len(uv_pixels), 1))
        colour = np.asarray(block["color"], dtype=np.float32)[None, :] * tex_or(block["color_tex"], (1, 1, 1))
        radius_scale = np.asarray(block["radius_scale"], dtype=np.float32)[None, :] * tex_or(block["radius_tex"], (1, 1, 1))
        mfp = np.maximum(block["radius"] * radius_scale, 1e-6)
        mr_id = material["textures"][2]
        orm = tex_or(mr_id, (1, 1, 1))
        roughness = np.clip(material["roughness"] * orm[:, 1], MIN_ROUGHNESS, 1.0)
        metallic = material["metallic"] * orm[:, 2]
        features = np.concatenate([colour, roughness[:, None]], axis=1).astype(np.float64)
        labels, centres = kmeans(features, args.classes)
        classes = []
        pooled_a = np.zeros((int(valid.sum()), 3), dtype=np.float32)
        pooled_b = np.zeros_like(pooled_a)
        for c in range(len(centres)):
            members = labels == c
            if not members.any():
                continue
            medium = class_medium(colour[members].mean(axis=0), mfp[members].mean(axis=0), float(roughness[members].mean()), block["anisotropy"])
            a, b = render_class(root, data, k, crop, medium, args.spp, args.threads, cache)
            x0, y0, x1, y1 = crop
            idx = np.flatnonzero(members)
            py, px = ys[idx] - y0, xs[idx] - x0
            pooled_a[idx], pooled_b[idx] = a[py, px], b[py, px]
            classes.append({"pixels": int(members.sum()), "colour": [float(v) for v in colour[members].mean(axis=0)],
                            "mfp": [float(v) for v in mfp[members].mean(axis=0)], "roughness": float(roughness[members].mean()), **medium})
        image_a, image_b = np.zeros((height, width, 3), dtype=np.float32), np.zeros((height, width, 3), dtype=np.float32)
        image_a[ys, xs], image_b[ys, xs] = pooled_a, pooled_b
        own = grid == k
        interior = sub.interior_mask(grid, k)
        x0, y0, x1, y1 = crop
        pooled_a = sub.box_pool(image_a[y0:y1, x0:x1], interior[y0:y1, x0:x1], args.pool)
        pooled_b = sub.box_pool(image_b[y0:y1, x0:x1], interior[y0:y1, x0:x1], args.pool)
        inside = own[y0:y1, x0:x1]
        reference[y0:y1, x0:x1][inside] = 0.5 * (pooled_a + pooled_b)[inside]
        half_a[y0:y1, x0:x1][inside], half_b[y0:y1, x0:x1][inside] = pooled_a[inside], pooled_b[inside]
        metallic_full = np.zeros((height, width), dtype=np.float32)
        metallic_full[ys, xs] = metallic
        piece_roi = interior & (metallic_full < METALLIC_LIMIT)
        roi |= piece_roi
        luma = sub.luminance(reference)
        floor = 0.05 * float(luma[piece_roi].mean()) if piece_roi.any() else 0.0
        noise = np.abs(sub.luminance(half_a) - sub.luminance(half_b))[piece_roi] / 2.0 / np.maximum(luma[piece_roi], max(floor, 1e-9))
        pieces[str(k)] = {"instance": k, "material": block["material"], "weight": block["weight"], "pixels": int(own.sum()), "roi_pixels": int(piece_roi.sum()),
                          "excluded_metallic_pixels": int((interior & (metallic_full >= METALLIC_LIMIT)).sum()), "classes": classes,
                          "noise_mean": float(noise.mean()) if noise.size else None, "noise_p95": float(np.percentile(noise, 95)) if noise.size else None,
                          "mean_mfp": [float(v) for v in mfp.mean(axis=0)], "mean_colour": [float(v) for v in colour.mean(axis=0)],
                          "weight_texture": "constant 1: the scattering map is not in the pinned subset" if block["weight_tex"] == base.INVALID else "texture"}
    write_pfm(out / "reference.pfm", reference)
    np.savez_compressed(out / "masks.npz", sphere=grid, roi=roi)
    report = {"schema": 1, "tool": "tools/f15_chess_subsurface_reference.py", "tool_sha256": sha256_file(__file__), "version": VERSION,
              "model": "textured volumetric subsurface: locally homogeneous Mitsuba 3.9.1 volpath classes (k-means)", "classes_per_piece": args.classes,
              "spp_per_grey_render": args.spp, "pool_window": args.pool, "boundary_ior": sub.BOUNDARY_IOR, "knot_roughness": list(KNOT_ROUGHNESS),
              "snapshot_sha256": sub.scene_digest(root), "pieces": pieces, "width": width, "height": height,
              "seconds": round(time.time() - started, 1)}
    (out / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"reference": str(out / "reference.pfm"), "pieces": len(pieces), "seconds": report["seconds"]}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub_parsers = parser.add_subparsers(dest="command", required=True)
    p = sub_parsers.add_parser("render")
    p.add_argument("snapshot")
    p.add_argument("--out", required=True)
    p.add_argument("--classes", type=int, default=8)
    p.add_argument("--spp", type=int, default=2048, help="samples per grey render (two halves)")
    p.add_argument("--pool", type=int, default=5)
    p.add_argument("--threads", type=int, default=sub.DEFAULT_THREADS)
    p.add_argument("--only", type=int, nargs="*", default=[], help="material indices of the pieces to render")
    p.add_argument("--cache-dir", default=None)
    p.set_defaults(function=command_render)
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f"F15 chess subsurface reference failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
