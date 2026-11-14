#!/usr/bin/env python3
"""F15.1 ticket 19: GI at -N of the thin-walled subsurface lobe (DDGI second probe lookup), a DECLARED check.

Scene (bench 2 --material-scenario subsurface-thin-gi-back): three weight-1 thin quads (transmission colour white, warm, teal; black base
colour, metallic 0, roughness 0.5) in front of an 8 m square DIFFUSE wall (albedo 0.8, 0.6 m behind them, facing the camera) lit by a sun
from the camera side (irradiance 3, toward the sun (0.8, 0.3, 0.52)); environment 0.  The FRONT of a quad sees nothing lit (E(N) = 0), the
BACK sees the lit wall.  With the old approximation (the pixel's own GI irradiance at N stands for E(-N)) the quads receive no GI; with the
second lookup they transmit the wall.  Three captures (DDGI + RT, --lighting restir so GI is allowed; 128 frames): `on` (default), `off`
(--thin-gi-back off, the control) and `nogi` (--gi off: everything but GI, so that the sun's own reflection on the quads cancels).

Expectation (the independent part): the GI term of a quad is the radiance of the wall seen through the lobe,
    L = color (1 - E_d(mu)) (1 - 0.16 mu) E(-N) / pi,   E(-N) / pi = L_wall F,   L_wall = albedo Esun cos(theta_sun) / pi,
F = the form factor of the 8 m square at 0.6 m from the quad centre (numerical integral, written here; the shadows of the three quads on
the wall are not subtracted: DDGI cannot resolve them, see form_factor), E_d = clamp(0.04 A + B, 0, 1) from the DFG table (the declared model of the thin environment, ticket 16).  The measured
quantity is the mean of the central region of each quad of (capture - nogi).  Gates (declared BEFORE the first measurement, indicative: DDGI
is a coarse probe grid and no decision-11 threshold exists for GI): `on` within 25 % of L on every quad, and `off` outside that bound (it
must fail the gate: a control that cannot fail proves nothing).  Not covered: the radiometric accuracy of DDGI itself, the Cache/ReSTIR GI
modes (they keep E(N) for E(-N), declared in the support table) and the legacy resolve GI path (no composite).

  uv run --offline --no-project --python 3.13 --with numpy python -I tools/f15_thin_gi_back_check.py --run --out build/f15-check/t19-gi
  ... --evaluate --out DIR     (existing captures, no GPU process)
Exit codes: 0 gates met, 1 a gate failed, 2 usage, 4 GPU busy, 6 capture failed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

SCENARIO = "subsurface-thin-gi-back"
WALL_ALBEDO = 0.8
SUN_IRRADIANCE = 3.0
TOWARD_SUN = (0.8, 0.3, 0.52)
QUAD_HALF = 0.4
WALL_DISTANCE = 0.6
WALL_SIDE = 8.0
QUAD_COLUMNS = (-1.0, 0.0, 1.0)  # x of the quads (spacing 1, three columns)
QUAD_COLOURS = ((1.0, 1.0, 1.0), (1.0, 0.55, 0.25), (0.1, 0.7, 0.6))
ROUGHNESS = 0.5
CAMERA_DISTANCE = 5.5
RESOLUTION = (1280, 720)
FRAMES = 128
GATE_NEW = 0.25      # |measured - L| / L, mean over the region, per quad
GATE_CONTROL = GATE_NEW  # the control must fail the same bound


def form_factor(x, distance=WALL_DISTANCE, side=WALL_SIDE, n=800):
    """Form factor of the wall (a square of side `side`, centred at x = 0) seen from a point of a quad at lateral offset x.  The shadows of
    the three quads on the wall (0.8 m wide, displaced 1 m by the sun) are NOT subtracted: the probe grid (spacing of the order of a metre,
    fitted to an 8 m scene) cannot resolve them, a first run with them subtracted differed from the capture by 35 % on two quads and by
    1 % on the third; this declared check measures the second lookup, not the shadow resolution of DDGI."""
    import numpy as np
    coordinates = (np.arange(n) + 0.5) / n * side - 0.5 * side
    wx, wy = np.meshgrid(coordinates, coordinates)
    r2 = (wx - x) ** 2 + wy ** 2 + distance ** 2
    return float(np.sum(distance * distance / (r2 * r2)) * (side / n) ** 2 / np.pi)


def expected_radiance(index):
    """L per channel of quad `index` (declared thin environment weight applied to the radiance seen from behind)."""
    import numpy as np
    from f15_reference import dfg_lookup
    x = QUAD_COLUMNS[index]
    mu = CAMERA_DISTANCE / np.sqrt(CAMERA_DISTANCE ** 2 + x ** 2)  # the view ray to the quad centre
    a, b = dfg_lookup(np.array([mu]), np.array([ROUGHNESS]))
    e_d = float(np.clip(0.04 * a[0] + b[0], 0.0, 1.0))
    wall_radiance = WALL_ALBEDO * SUN_IRRADIANCE * (TOWARD_SUN[2] / float(np.linalg.norm(TOWARD_SUN))) / np.pi
    return np.asarray(QUAD_COLOURS[index]) * (1.0 - e_d) * (1.0 - 0.16 * mu) * wall_radiance * form_factor(x)


def quad_region(index):
    """Pixel slices of the central region of quad `index` (113.4 px per metre at distance 5.5, quad side 0.8 m: 91 px; ROI +-30 px)."""
    width, height = RESOLUTION
    pixels_per_metre = 0.5 * height / (CAMERA_DISTANCE * 0.5773502691896257)
    cx = int(round(0.5 * width + QUAD_COLUMNS[index] * pixels_per_metre))
    cy = height // 2
    return slice(cy - 30, cy + 31), slice(cx - 30, cx + 31)


def capture_command(binary, folder, mode, seed):
    gi = "off" if mode == "nogi" else "ddgi"
    back = "off" if mode == "off" else "on"
    last = str(FRAMES - 1)
    return [str(binary), "--scene", "procedural", "--bench", "2", "--material-scenario", SCENARIO, "--render-path", "visibility",
            "--geometry-path", "mesh", "--rt", "on", "--shadows", "off", "--lighting", "restir", "--gi", gi, "--reflections", "ssr",
            "--ao", "off", "--lighting-denoise", "custom", "--reflection-samples", "8", "--lighting-seed", str(seed), "--thin-gi-back", back,
            "--post", "--upscaler", "native", "--resolution", f"{RESOLUTION[0]}x{RESOLUTION[1]}", "--frames", str(FRAMES), "--warmup", "0",
            "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--debug-lighting", "1", "--pipeline-sync",
            "--capture-linear", str(folder / "engine.pfm"), "--capture-linear-frame", last, "--capture-linear-signal", "hdr",
            "--report", str(folder / "report.json")]


def evaluate(out):
    import numpy as np
    from f12_reference import read_pfm
    result = {"gates": {"new": GATE_NEW, "control": GATE_CONTROL}, "quads": [], "passed": True}
    images = {mode: read_pfm(out / mode / "engine.pfm") for mode in ("on", "off", "nogi")}
    for i in range(3):
        rows, cols = quad_region(i)
        want = expected_radiance(i)
        baseline = images["nogi"][rows, cols].reshape(-1, 3).mean(axis=0)
        entry = {"column": QUAD_COLUMNS[i], "expected_gi_term": [round(float(v), 5) for v in want],
                 "nogi_pixel": [round(float(v), 5) for v in baseline]}
        for mode in ("on", "off"):
            got = images[mode][rows, cols].reshape(-1, 3).mean(axis=0) - baseline
            entry[mode + "_gi_term"] = [round(float(v), 5) for v in got]
            entry[mode + "_error"] = round(float(np.max(np.abs(got - want) / np.maximum(want, 1e-6))), 4)
        entry["new_ok"] = entry["on_error"] <= GATE_NEW
        entry["control_fails"] = entry["off_error"] > GATE_NEW
        result["passed"] &= bool(entry["new_ok"] and entry["control_fails"])
        result["quads"].append(entry)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result))
    return 0 if result["passed"] else 1


def run(args):
    import f15_check as fc
    out = args.out.resolve()
    binary = args.binary.resolve()
    with fc.gpu_lock(args.gpu_lock):
        for mode in ("on", "off", "nogi"):
            busy = fc.running_gpu_processes()
            if busy:
                raise fc.GpuBusy(f"other GPU jobs are running: {', '.join(busy)}")
            folder = out / mode
            folder.mkdir(parents=True, exist_ok=True)
            status = fc.capture(capture_command(binary, folder, mode, args.seed), folder / "engine.log", args.timeout)
            if not status["passed"]:
                print(f"capture {mode} failed: {status['failures']}")
                return 6
            draws = fc.fallback_draws((folder / "engine.log").read_text(errors="replace"))
            if draws is None or draws > 0:
                print(f"capture {mode} served fallback pipelines ({draws})")
                return 6
            time.sleep(2)
    return evaluate(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=pathlib.Path, default=REPO / "build" / "release" / "phosphor")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--gpu-lock", default="/tmp/phosphor-gpu-verification.lock")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=1)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    args = parser.parse_args(argv)
    if args.evaluate:
        return evaluate(args.out.resolve())
    try:
        return run(args)
    except Exception as error:  # GpuBusy
        print(f"{type(error).__name__}: {error}")
        return 4 if type(error).__name__ == "GpuBusy" else 5


if __name__ == "__main__":
    sys.exit(main())
