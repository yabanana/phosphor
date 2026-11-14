#!/usr/bin/env python3
"""F13 residues found in F15 (owner decision 2026-11-14: fixed inside F15): fixtures with negative controls.

Every case generates its own scene (tools/f15_fixture_scene.py), renders it under the declared corpus rig (`--corpus-rig 0,0`: sun 3 W/m2 from the
upper left behind the camera, constant environment 0.5) and gates a measured number.  `--control-binary` is a build WITHOUT the fix: it must fail the
case's gate (exit 3 when it does not: the fixture would not discriminate).

Cases:
  grazing   C22-D3: smooth spheres, silhouette ring.  A ring pixel must not be darker than the same pixel with reflections off (the composite
            removes the diffuse weight 1 - E(N.V) the specular estimate then has to give back).
  all       every case

Exit codes: 0 gates hold (and every control fails as it must), 1 a gate failed, 2 usage, 3 a control did not fail (non-discriminating),
4 GPU busy, 5 internal error, 6 capture failed.
"""
from __future__ import annotations

import argparse
import json
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


CASES = {"grazing": case_grazing}


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
