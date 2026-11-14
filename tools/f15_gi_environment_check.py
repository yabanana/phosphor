#!/usr/bin/env python3
"""F15 (F13.1 residue found in F15, owner decision 2026-11-14): the declared environment reaches primary surfaces in EVERY GI mode.

Defect: with any GI mode on, the camera-visible shading dropped the declared environment term (environment x albedo x (1 - metallic), added
only when GI is off) and the transported GI never supplied it either: every probe / candidate ray that escaped the scene returned black ("no
direct environment in the INDIRECT atlas").  A wall the sun does not reach read 16/255 with `--gi ddgi` against 194 with `--gi off`.

Ownership rule (written where it is enforced: shaders/gi_common.h `giEscapeRadiance`):
  GI off      the resolve / composite adds the analytic term environment x albedo x (1 - metallic) (unoccluded, declared, no AO here).
  GI on       the TRANSPORT owns the environment: a ray that escapes the scene (no hit within maxDistance) returns the declared radiance, so the
              sky is occluded by geometry.  Camera surfaces receive it through the same irradiance they receive every other indirect light
              (ddgi atlas / ReSTIR reservoir); secondary hits receive it through the atlas lookup (the former explicit sky ray of
              giSecondaryRadiance would count it twice and is gone).  The GI-off analytic term is NOT added on top (REFLECT_COMPOSE_GI and
              `useGI` strip it), so no pixel owns the environment diffuse twice.

Fixture (generated here): a wall (albedo 0.8, a diffuse dielectric), a small dark panel high in front of it whose sun shadow falls on the wall
and blocks almost no sky ("open": reference albedo L V, V ~ 0.99), and a dark slot (a shelf above and a floor below, 0.8 deep) whose wall
region the camera sees and whose shelf hides the sun ("occluded": V ~ 0.25: an unoccluded additive term would over-light it by 1 / V).  The sun never reaches either region
(checked by the CPU reference), so the linear HDR value is the environment (plus the small dielectric specular ambient and the dark
occluders' bounce) alone.  The CPU reference integrates the cosine-weighted sky visibility of each region against the same rectangles.

Gates (linear HDR mean of the region / albedo L V, per mode):
  open      every mode             in [OPEN_GATE]
  occluded  ddgi (0.5 m probes) / cache / restir  in [occluded gate]  (the sky is blocked: an unoccluded term reads 1 / V times too bright)
  occluded  off, ddgi-default-grid  informational: the GI-off term is unoccluded by design (ratio ~ 1 / V); the default probe grid is coarser
            than the slot (see MODE_SETUP).  Recorded, not gated
Negative control (`--control-binary`, the pre-fix build): must fail the OPEN gate in every GI mode (reads ~ 0 against 1).

Exit codes: 0 gates hold (and every control fails as it must), 1 a gate failed, 2 usage / geometry check failed, 3 a control did not fail,
4 GPU busy, 6 capture failed.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_fixture_scene as fx  # noqa: E402

corpus = fx.corpus

SCALE = 8.0          # the rig frames by bounds (the image is scale invariant); 8 m lets the default probe grid resolve the scene
L_ENV = 0.5          # CorpusRig::environmentRadiance
ALBEDO = 0.8
OCCLUDER_ALBEDO = 0.01
TOWARD_SUN = (-0.45, 0.55, 0.70)  # the rig at yaw 0, pitch 0 (unnormalised: the rig's towardSun is unit to 0.5 %)
SUN_CONE = 0.05      # radians: the sun disk and the quarter-pixel margin (the region must lie in the umbra of this cone)
MODES = ("off", "ddgi", "ddgi-default-grid", "cache", "restir")
# mode -> (--gi value, extra arguments, gate the occluded region).  The slot (2.4 m tall at this scale) is thinner than the default probe grid
# spacing (~1.3 m for a 19 m wall: two probe layers), so DDGI resolves it only with a finer grid: `ddgi` uses 0.5 m probes (the atlas then reads
# the slot like the per-pixel modes); `ddgi-default-grid` records the coarse-grid number (the lookup blends probes outside the slot: a grid
# resolution property, measured, not a defect and not gated).
MODE_SETUP = {"off": ("off", [], False), "ddgi": ("ddgi", ["--gi-spacing", "0.5", "--gi-grid", "32x16x32"], True),
              "ddgi-default-grid": ("ddgi", [], False), "cache": ("cache", [], True), "restir": ("restir", [], True)}
OPEN_GATE = (0.90, 1.15)
OCCLUDED_GATE = (0.80, 1.30)
CONTROL_OPEN_MAX = 0.5  # the pre-fix GI modes read ~0 of the reference
# axis-aligned double-sided rectangles: name -> (axis, constant, (a0, a1), (b0, b1)); axis 'z' = plane z=c spanning x in a, y in b;
# axis 'y' = plane y=c spanning x in a, z in b.
OCCLUDERS = {
    "panel": ("z", 0.50, (-0.90, -0.70), (0.25, 0.45)),
    "shelf": ("y", 0.16, (0.0, 1.0), (0.0, 0.80)),
    "floor": ("y", -0.14, (0.0, 1.0), (0.0, 0.80)),
}
WALL = (-1.2, -0.7, 1.2, 0.7)
REGIONS = {  # wall rectangles (x0, y0, x1, y1) at z = 0
    "open": (-0.54, -0.10, -0.42, 0.02),
    "occluded": (0.40, -0.06, 0.70, 0.04),
}


# --- CPU reference ---------------------------------------------------------------------------------------------------------------------

def hit_rectangle(origin, direction, occluder, tmax=math.inf):
    """True when the ray origin + t direction (0 < t < tmax) crosses the double-sided axis-aligned rectangle."""
    axis, constant, (a0, a1), (b0, b1) = occluder
    index = 2 if axis == "z" else 1
    if abs(direction[index]) < 1e-12:
        return False
    t = (constant - origin[index]) / direction[index]
    if not 1e-9 < t < tmax:
        return False
    x = origin[0] + t * direction[0]
    other = origin[1] + t * direction[1] if axis == "z" else origin[2] + t * direction[2]
    return a0 <= x <= a1 and b0 <= other <= b1


def occluded_ray(origin, direction, occluders=None, tmax=math.inf):
    return any(hit_rectangle(origin, direction, o, tmax) for o in (OCCLUDERS if occluders is None else occluders).values())


def sky_visibility(point, occluders=None, n=64):
    """Cosine-weighted fraction of the hemisphere above a +z facing surface point that sees the sky (stratified n x n)."""
    open_count = 0
    for i in range(n):
        for j in range(n):
            u1, u2 = (i + 0.5) / n, (j + 0.5) / n
            r, phi = math.sqrt(u1), 2.0 * math.pi * u2
            direction = (r * math.cos(phi), r * math.sin(phi), math.sqrt(1.0 - u1))
            if not occluded_ray(point, direction, occluders):
                open_count += 1
    return open_count / (n * n)


def sun_directions(cone=SUN_CONE):
    length = math.sqrt(sum(c * c for c in TOWARD_SUN))
    axis = tuple(c / length for c in TOWARD_SUN)
    helper = (0.0, 0.0, 1.0)
    t = (axis[1] * helper[2] - axis[2] * helper[1], axis[2] * helper[0] - axis[0] * helper[2], axis[0] * helper[1] - axis[1] * helper[0])
    tl = math.sqrt(sum(c * c for c in t))
    t = tuple(c / tl for c in t)
    b = (axis[1] * t[2] - axis[2] * t[1], axis[2] * t[0] - axis[0] * t[2], axis[0] * t[1] - axis[1] * t[0])
    out = [axis]
    for k in range(8):
        phi = 2.0 * math.pi * k / 8
        out.append(tuple(math.cos(cone) * axis[i] + math.sin(cone) * (math.cos(phi) * t[i] + math.sin(phi) * b[i]) for i in range(3)))
    return out


def sun_blocked(point, occluders=None):
    """The sun (axis and a ring at SUN_CONE) is blocked from the point."""
    return all(occluded_ray(point, d, occluders) for d in sun_directions())


def sample_points(rect, n=5):
    x0, y0, x1, y1 = rect
    return [(x0 + (x1 - x0) * (i + 0.5) / n, y0 + (y1 - y0) * (j + 0.5) / n, 0.0) for i in range(n) for j in range(n)]


def scene_bounds():
    """Bounds of the unscaled scene (wall, panel, shelf), as the rig frames them."""
    lo, hi = [WALL[0], WALL[1], 0.0], [WALL[2], WALL[3], 0.0]
    for axis, c, (a0, a1), (b0, b1) in OCCLUDERS.values():
        corners = [(a, c, b) if axis == "y" else (a, b, c) for a in (a0, a1) for b in (b0, b1)]
        for p in corners:
            for i in range(3):
                lo[i], hi[i] = min(lo[i], p[i]), max(hi[i], p[i])
    return tuple(lo), tuple(hi)


def camera_position():
    lo, hi = scene_bounds()
    centre, distance = fx.frame_rig(lo, hi)
    return (centre[0], centre[1], centre[2] + distance)


def region_reference(name, occluders=None):
    """Mean sky visibility V over the region, the sun/camera checks, and the reference radiance albedo L V."""
    points = sample_points(REGIONS[name])
    camera = camera_position()
    visibilities = [sky_visibility(p, occluders) for p in points]
    sun_ok = all(sun_blocked(p, occluders) for p in points)
    camera_ok = True
    for p in points:
        d = tuple(p[i] - camera[i] for i in range(3))
        length = math.sqrt(sum(c * c for c in d))
        if occluded_ray(camera, tuple(c / length for c in d), occluders, tmax=length * (1.0 - 1e-6)):
            camera_ok = False
    v = sum(visibilities) / len(visibilities)
    return {"visibility": v, "visibility_min": min(visibilities), "visibility_max": max(visibilities), "sun_blocked": sun_ok,
            "camera_sees_wall": camera_ok, "reference": ALBEDO * L_ENV * v}


# --- scene -----------------------------------------------------------------------------------------------------------------------------

def build_scene():
    scene = fx.Scene()
    wall = scene.material("wall", (ALBEDO,) * 3, 0.0, 1.0)
    dark = scene.material("dark occluder", (OCCLUDER_ALBEDO,) * 3, 0.0, 1.0, double_sided=True)
    s = SCALE
    x0, y0, x1, y1 = WALL
    scene.quad([(x0 * s, y0 * s, 0.0), (x1 * s, y0 * s, 0.0), (x1 * s, y1 * s, 0.0), (x0 * s, y1 * s, 0.0)], wall)
    for axis, c, (a0, a1), (b0, b1) in OCCLUDERS.values():
        if axis == "z":  # front side towards +z (the camera)
            corners = [(a0, b0, c), (a1, b0, c), (a1, b1, c), (a0, b1, c)]
        else:            # horizontal shelf: front side towards -y (its underside faces the wall region)
            corners = [(a0, c, b1), (a1, c, b1), (a1, c, b0), (a0, c, b0)]
        scene.quad([tuple(v * s for v in p) for p in corners], dark)
    return scene


def region_boxes():
    lo, hi = scene_bounds()
    centre, distance = fx.frame_rig(lo, hi)
    return {name: fx.region_pixels(rect, 0.0, centre, distance) for name, rect in REGIONS.items()}


# --- measurement and gates -------------------------------------------------------------------------------------------------------------

def region_means(pfm_path, boxes):
    from temporal_light_metrics import read_pfm
    image = read_pfm(pfm_path)
    out = {}
    for name, (x0, y0, x1, y1) in boxes.items():
        total, count = 0.0, 0
        for y in range(max(y0, 0), min(y1, image.height)):
            for x in range(max(x0, 0), min(x1, image.width)):
                r, g, b = image.pixel(x, y)
                total += 0.2126 * r + 0.7152 * g + 0.0722 * b
                count += 1
        out[name] = total / max(count, 1)
    return out


def ratios(means, references):
    return {name: means[name] / max(references[name]["reference"], 1e-9) for name in references}


def gate_failures(mode, ratio):
    """Names of the gates a mode's ratios violate (occluded is informational with GI off)."""
    failures = []
    if not OPEN_GATE[0] <= ratio["open"] <= OPEN_GATE[1]:
        failures.append("open")
    if MODE_SETUP[mode][2] and not OCCLUDED_GATE[0] <= ratio["occluded"] <= OCCLUDED_GATE[1]:
        failures.append("occluded")
    return failures


def control_fails(mode, ratio):
    """A pre-fix build must lose the environment in every GI mode (open ratio below CONTROL_OPEN_MAX); GI off was never broken."""
    if mode == "off":
        return True  # nothing to prove: the declared term existed
    return ratio["open"] < CONTROL_OPEN_MAX


def capture_mode(binary, scene_path, mode, folder, boxes):
    pfm = pathlib.Path(folder) / "engine.pfm"
    gi, extra, _ = MODE_SETUP[mode]
    fx.capture(binary, scene_path, folder, {"--gi": gi},
               ["--capture-linear", str(pfm), "--capture-linear-signal", "hdr", "--capture-linear-frame", str(corpus.FRAMES - 1), *extra],
               name="gi-environment")
    return region_means(pfm, boxes)


def run(args):
    references = {name: region_reference(name) for name in REGIONS}
    for name, ref in references.items():
        if not (ref["sun_blocked"] and ref["camera_sees_wall"]):
            print(f"geometry check failed for region {name}: {ref}")
            return 2, {}
    scene_path = build_scene().write(args.out / "scene")
    boxes = region_boxes()
    report = {"scale": SCALE, "references": references, "boxes": boxes, "modes": {}}
    code = 0
    for mode in args.modes:
        entry = {}
        means = capture_mode(args.binary.resolve(), scene_path, mode, args.out / mode / "fixed", boxes)
        ratio = ratios(means, references)
        failures = gate_failures(mode, ratio)
        entry["fixed"] = {"means": means, "ratio": ratio, "failed_gates": failures}
        if failures:
            code = code or 1
        if args.control_binary:
            cmeans = capture_mode(args.control_binary.resolve(), scene_path, mode, args.out / mode / "control", boxes)
            cratio = ratios(cmeans, references)
            entry["control"] = {"means": cmeans, "ratio": cratio, "fails_as_required": control_fails(mode, cratio)}
            if not entry["control"]["fails_as_required"] and not failures:
                code = code or 3
        report["modes"][mode] = entry
    return code, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--control-binary", type=pathlib.Path, help="the pre-fix build: every GI mode must fail the open gate")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-gi-environment")
    args = parser.parse_args(argv)
    try:
        code, report = run(args)
    except corpus.GpuBusy:
        print("GPU busy")
        return 4
    except corpus.CaptureFailed as error:
        print(error)
        return 6
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
