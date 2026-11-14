#!/usr/bin/env python3
"""F13 residues found in F15 (owner decision 2026-11-14: fixed inside F15): fixtures with negative controls.

Every case generates its own scene (tools/f15_fixture_scene.py), renders it under the declared corpus rig (`--corpus-rig 0,0`: sun 3 W/m2 from the
upper left behind the camera, constant environment 0.5) and gates a measured number.  `--control-binary` is a build WITHOUT the fix: it must fail the
case's gate (exit 3 when it does not: the fixture would not discriminate).

Cases:
  grazing   C22-D3: smooth spheres, silhouette ring.  A ring pixel must not be darker than the same pixel with reflections off (the composite
            removes the diffuse weight 1 - E(N.V) the specular estimate then has to give back).
  mirror-env  a perfect mirror floor seeing a wall in the sun shadow of an opaque panel, GI off: the wall seen in the mirror must carry the same
            ambient light as the wall seen directly (mirror / direct ratio), shadowed and lit.
  backface  a mirror floor and a thin glass pane over a single-sided plane seen from BEHIND (culled by the rasteriser, a lit wall behind it): the transport
            ray must go on past the plane and show the wall as the camera sees it (mirror / direct and glass / direct ratios).
  all       every case

Exit codes: 0 gates hold (and every control fails as it must), 1 a gate failed, 2 usage, 3 a control did not fail (non-discriminating),
4 GPU busy, 5 internal error, 6 capture failed.
"""
from __future__ import annotations

import argparse
import json
import re
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_fixture_scene as fx  # noqa: E402

corpus = fx.corpus


# --- grazing (C22-D3) ----------------------------------------------------------------------------------------------------------------------

SPHERES = ((-2.4, 0.15), (0.0, 0.40), (2.4, 0.80))  # (x, roughness)
DARK_RATIO = 0.70  # a ring pixel is "dark" when its luminance is below this fraction of the reflections-off pixel


def grazing_scene(folder, segments=48):
    scene = fx.Scene()
    for x, roughness in SPHERES:
        material = scene.material(f"sphere r{roughness}", (0.8, 0.8, 0.8), 0.0, roughness)
        scene.sphere((x, 0.0, 0.0), 1.0, segments, material)
    return scene, scene.write(folder)


def ring_pixels(image):
    """Silhouette pixels: not the background (the top-left corner colour) with a background 4-neighbour."""
    width, height, channels, rows = image
    background = tuple(rows[0][0:3])

    def is_background(x, y):
        return tuple(rows[y][x * channels:x * channels + 3]) == background
    ring = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if is_background(x, y):
                continue
            if is_background(x - 1, y) or is_background(x + 1, y) or is_background(x, y - 1) or is_background(x, y + 1):
                ring.append((x, y))
    return ring


def measure_grazing(binary, scene_path, out):
    result = {}
    off = fx.read_png_gray(fx.capture(binary, scene_path, out / "off", {"--reflections": "off"}))
    ring = ring_pixels(off)
    result["ring_pixels"] = len(ring)
    for mode in ("rt", "ssr"):
        image = fx.read_png_gray(fx.capture(binary, scene_path, out / mode, {"--reflections": mode}))
        dark = [(x, y) for x, y in ring if fx.luminance(image, x, y) < DARK_RATIO * fx.luminance(off, x, y)]
        ratios = [fx.luminance(image, x, y) / max(fx.luminance(off, x, y), 1.0) for x, y in ring]
        result[mode] = {"dark_ring_pixels": len(dark), "min_ratio_to_off": round(min(ratios), 3), "first": dark[:6]}
    return result


def case_grazing(binary, control, out):
    scene, path = grazing_scene(out / "grazing" / "scene")
    fixed = measure_grazing(binary, path, out / "grazing" / "fixed")
    ok = fixed["rt"]["dark_ring_pixels"] == 0 and fixed["ssr"]["dark_ring_pixels"] == 0
    report = {"fixed": fixed, "gates_hold": ok}
    control_fails = None
    if control:
        report["control"] = measure_grazing(control, path, out / "grazing" / "control")
        control_fails = report["control"]["rt"]["dark_ring_pixels"] > 0
        report["control_fails_as_required"] = control_fails
    return ok, control_fails, report


# --- mirror seeing a shadowed wall (the reflection environment term) ---------------------------------------------------------------------

MIRROR_GATE = (0.85, 1.15)
MIRROR_CONTROL_MAX = 0.5
FLOOR_Y = -0.6


def mirror_scene(folder):
    scene = fx.Scene()
    wall = scene.material("wall", (0.8, 0.8, 0.8), 0.0, 1.0)
    panel = scene.material("panel", (0.5, 0.5, 0.5), 0.0, 1.0)
    mirror = scene.material("mirror", (1.0, 1.0, 1.0), 1.0, 0.0)  # a metal of baseColor 1: F = 1 at every angle, smooth (roughness clamps at .04)
    scene.quad([(-1.0, -0.6, 0.0), (1.0, -0.6, 0.0), (1.0, 0.9, 0.0), (-1.0, 0.9, 0.0)], wall)
    scene.quad([(-0.5, 0.1, 0.3), (0.0, 0.1, 0.3), (0.0, 0.5, 0.3), (-0.5, 0.5, 0.3)], panel)
    scene.quad([(-1.6, FLOOR_Y, 2.2), (1.6, FLOOR_Y, 2.2), (1.6, FLOOR_Y, 0.0), (-1.6, FLOOR_Y, 0.0)], mirror)
    return scene, scene.write(folder)


# wall rectangles (x0, y0, x1, y1) at z = 0: the sun shadow of the panel covers x -0.31..0.19, y -0.14..0.26 (panel height 0.3 shifts it by
# -0.3 * towardSun.xy / towardSun.z); the lit region is far from it.  Both lie in the part of the wall the floor mirror shows (y <= 0.11).
MIRROR_REGIONS = {"shadow": (-0.20, -0.07, 0.10, 0.06), "lit": (0.40, -0.07, 0.90, 0.06)}


def mirror_boxes(scene):
    bmin, bmax = scene.bounds()
    centre, distance = fx.frame_rig(bmin, bmax)
    boxes = {}
    for name, (x0, y0, x1, y1) in MIRROR_REGIONS.items():
        boxes["direct_" + name] = fx.region_pixels((x0, y0, x1, y1), 0.0, centre, distance)
        # the mirror image of the wall: reflected about the floor plane, seen at the wall's depth
        boxes["mirror_" + name] = fx.region_pixels((x0, 2 * FLOOR_Y - y1, x1, 2 * FLOOR_Y - y0), 0.0, centre, distance)
    return boxes


def measure_mirror(binary, scene, path, out):
    image = fx.read_png_gray(fx.capture(binary, path, out, {"--reflections": "rt", "--gi": "off"}))
    boxes = mirror_boxes(scene)
    means = {name: fx.region_mean(image, box) for name, box in boxes.items()}
    return {"means": means, "shadow_ratio": means["mirror_shadow"] / max(means["direct_shadow"], 1e-6),
            "lit_ratio": means["mirror_lit"] / max(means["direct_lit"], 1e-6)}


def case_mirror_env(binary, control, out):
    scene, path = mirror_scene(out / "mirror-env" / "scene")
    fixed = measure_mirror(binary, scene, path, out / "mirror-env" / "fixed")
    ok = all(MIRROR_GATE[0] <= fixed[k] <= MIRROR_GATE[1] for k in ("shadow_ratio", "lit_ratio"))
    report = {"fixed": fixed, "gates_hold": ok}
    control_fails = None
    if control:
        report["control"] = measure_mirror(control, scene, path, out / "mirror-env" / "control")
        control_fails = report["control"]["shadow_ratio"] < MIRROR_CONTROL_MAX
        report["control_fails_as_required"] = control_fails
    return ok, control_fails, report


# --- transport rays that meet the BACK of a single-sided opaque surface ------------------------------------------------------------------

BACKFACE_GATE = (0.85, 1.15)
BACKFACE_CONTROL_MAX = 0.5
GLASS_THIN = {"KHR_materials_transmission": {"transmissionFactor": 1.0}, "KHR_materials_ior": {"ior": 1.0}}


def backface_scene(folder):
    """A lit wall W2 (z = -0.4), a single-sided red plane W1 (z = 0) whose FRONT faces away from the camera and the mirror (the rasteriser culls
    it: the camera sees W2 through it), a perfect mirror floor and a thin glass pane (IOR 1: no bend, no Fresnel) in front of W1."""
    scene = fx.Scene()
    w2 = scene.material("lit wall", (0.8, 0.8, 0.8), 0.0, 1.0)
    w1 = scene.material("single-sided plane, back to the viewer", (0.8, 0.1, 0.1), 0.0, 1.0)
    mirror = scene.material("mirror", (1.0, 1.0, 1.0), 1.0, 0.0)
    glass = scene.material("glass", (1.0, 1.0, 1.0), 0.0, 0.0, GLASS_THIN)
    scene.quad([(-1.0, -0.6, -0.4), (1.0, -0.6, -0.4), (1.0, 0.9, -0.4), (-1.0, 0.9, -0.4)], w2)
    scene.quad([(1.0, -0.6, 0.0), (-1.0, -0.6, 0.0), (-1.0, 0.9, 0.0), (1.0, 0.9, 0.0)], w1)  # reversed winding: the normal is -z
    scene.quad([(-1.6, FLOOR_Y, 2.2), (1.6, FLOOR_Y, 2.2), (1.6, FLOOR_Y, -0.4), (-1.6, FLOOR_Y, -0.4)], mirror)
    scene.quad([(0.45, -0.4, 0.15), (0.85, -0.4, 0.15), (0.85, 0.1, 0.15), (0.45, 0.1, 0.15)], glass)
    return scene, scene.write(folder)


BACKFACE_ROWS = (-0.12, 0.02)
BACKFACE_REGIONS = {"direct": (-0.70, BACKFACE_ROWS[0], -0.30, BACKFACE_ROWS[1]), "glass": (0.55, BACKFACE_ROWS[0], 0.80, BACKFACE_ROWS[1])}


def backface_boxes(scene):
    bmin, bmax = scene.bounds()
    centre, distance = fx.frame_rig(bmin, bmax)
    x0, y0, x1, y1 = BACKFACE_REGIONS["direct"]
    return {"direct": fx.region_pixels((x0, y0, x1, y1), -0.4, centre, distance),
            "glass": fx.region_pixels(BACKFACE_REGIONS["glass"], -0.4, centre, distance),
            "mirror": fx.region_pixels((x0, 2 * FLOOR_Y - y1, x1, 2 * FLOOR_Y - y0), -0.4, centre, distance)}


def measure_backface(binary, scene, path, out):
    image = fx.read_png_gray(fx.capture(binary, path, out, {"--reflections": "rt", "--gi": "off"}, ["--debug-lighting", "1"]))
    means = {name: fx.region_mean(image, box) for name, box in backface_boxes(scene).items()}
    log = (out / "engine.log").read_text()
    skipped = [int(m) for m in re.findall(r"back-face continuations slot \d+ \| culled single-sided back faces passed (\d+)", log)]
    return {"means": means, "mirror_ratio": means["mirror"] / max(means["direct"], 1e-6), "glass_ratio": means["glass"] / max(means["direct"], 1e-6),
            "continuations_reported": max(skipped) if skipped else 0}


def case_backface(binary, control, out):
    scene, path = backface_scene(out / "backface" / "scene")
    fixed = measure_backface(binary, scene, path, out / "backface" / "fixed")
    ok = all(BACKFACE_GATE[0] <= fixed[k] <= BACKFACE_GATE[1] for k in ("mirror_ratio", "glass_ratio")) and fixed["continuations_reported"] > 0
    report = {"fixed": fixed, "gates_hold": ok}
    control_fails = None
    if control:
        report["control"] = measure_backface(control, scene, path, out / "backface" / "control")
        c = report["control"]
        control_fails = c["mirror_ratio"] < BACKFACE_CONTROL_MAX and c["glass_ratio"] < BACKFACE_CONTROL_MAX
        report["control_fails_as_required"] = control_fails
    return ok, control_fails, report


CASES = {"grazing": case_grazing, "mirror-env": case_mirror_env, "backface": case_backface}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--control-binary", type=pathlib.Path, help="a build WITHOUT the fixes: every gate it is asked about must fail")
    parser.add_argument("--case", default="all", choices=["all", *CASES])
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-residue")
    args = parser.parse_args(argv)
    names = list(CASES) if args.case == "all" else [args.case]
    reports, exit_code = {}, 0
    try:
        for name in names:
            ok, control_fails, report = CASES[name](args.binary.resolve(), args.control_binary.resolve() if args.control_binary else None, args.out)
            reports[name] = report
            if not ok:
                exit_code = exit_code or 1
            elif args.control_binary and not control_fails:
                exit_code = exit_code or 3
    except corpus.GpuBusy:
        print("GPU busy")
        return 4
    except corpus.CaptureFailed as error:
        print(error)
        return 6
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
