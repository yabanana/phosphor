#!/usr/bin/env python3
"""F15.1 ticket 13: transmissive pixels are reactive for MetalFX temporal, in both paths that write the mask.

The mask is the R8Unorm "Reactive mask" resolve output that MetalFX temporal reads (post_processor.cpp).  It is written by
the visibility resolve (shaders/visibility_resolve.metal, plain and lit) and, for the indexed overflow fallback, by the
forward surface fragments (shaders/forward.metal, plain and lit).  `--capture-linear-signal reactive` reads it back (PFM, the
R value of the pixel: 0.75 for a transmissive or cut-out pixel with a valid history, 0 for an opaque one, 1 where the history
is invalid or nothing is drawn).  The overflow path is forced like tools/f13_overflow_emission_check.sh
(`--debug-meshlets 1 --debug-meshlets-corrupt count`: every frame goes through the fallback, the meshlet self-check
reports FAIL and the exit code is 1 by design).

Cases (one process each, serial, API + shader validation, 640x360, mask of frame 5, history valid):
  glass-{lit,legacy}-{visibility,overflow}   transmission-thin-volume (three glass spheres over an emissive checkerboard):
                                    every pixel of the eroded sphere discs is 0.75, and no pixel outside the dilated discs is
  glass-motion-lit                  transmission-roughness-ior with --transmission-motion (orbiting camera, moving panel):
                                    the same, with a moving camera and a dynamic emitter in the scene
  opaque-{lit,legacy}-{visibility,overflow}   NEGATIVE CONTROL: the opaque base-grid-sun spheres (no transmission, no cut-out):
                                    no pixel is 0.75 anywhere, so the same disc predicate FAILS on them (a path that did not
                                    set the bit would look like this) and the glass predicate is shown to discriminate
"Lit" = `--rt on --shadows rt --lighting brute --reflections rt` (forward_surface_lit_fs / visibility_lit_resolve), "legacy" =
`--rt off --lighting legacy --reflections probes` (forward_surface_fs / visibility_resolve).

  uv run --offline --no-project --python 3.13 python -I tools/f15_reactive_check.py --run --out build/f15-reactive
  ... --evaluate --out DIR   (existing captures)     ... --plan --out DIR (commands only)
Exit codes: 0 every expectation met, 1 an expectation failed, 2 usage, 3 GPU busy, 4 capture failed.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import pathlib
import re
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_checked import run_checked  # noqa: E402
from temporal_light_metrics import read_pfm  # noqa: E402

REPO = HERE.parent
EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_BUSY, EXIT_CAPTURE = 0, 1, 2, 3, 4
WIDTH, HEIGHT, FRAMES, CAPTURE_FRAME = 640, 360, 6, 5
FOV_DEGREES, RADIUS = 60.0, 0.4
REACTIVE = 0.75      # shaders: the value of a transmissive (or cut-out) pixel with a valid history
TOLERANCE = 0.01     # R8Unorm stores 191 / 255 = 0.749
ERODED, DILATED = 0.8, 1.2  # disc scale: certainly on the sphere / certainly clear of it (perspective ellipses of off-axis spheres)
GPU_PROCESS = re.compile(r"^(phosphor|soc_bench|f\d+_spike)$")
GPU_ERROR = re.compile(r"GPU timeout|command buffers failed|Shader Validation Error|Metal Validation Error|failed assertion", re.I)


class GpuBusy(RuntimeError):
    pass


class CaptureFailed(RuntimeError):
    pass


# --- declared fixtures (src/testbench/transmission_validation.cpp, material_validation.cpp; both pinned by doctests) ------
def thin_volume():
    """transmission-thin-volume: camera (0, 1, 4.5) -> (0, 1, 0); three spheres on y = 1."""
    return {"target": (0.0, 1.0, 0.0), "distance": 4.5, "spheres": [(-1.2, 1.0, 0.0), (0.0, 1.0, 0.0), (1.2, 1.0, 0.0)]}


def roughness_ior():
    """transmission-roughness-ior: camera (0, 2.2, 5.5), 4 x 3 spheres spaced 1.2 m (rows y = 0.4 + 1.2 r, columns x = (c - 1) 1.2)."""
    return {"target": (0.0, 2.2, 0.0), "distance": 5.5,
            "spheres": [((c - 1.0) * 1.2, 0.4 + 1.2 * r, 0.0) for r in range(4) for c in range(3)]}


def base_grid():
    """base-grid-sun (MaterialValidation): camera (0, 2, 5.5) looking at (0, 2, 0); 5 x 5 spheres 1 m apart, y = row."""
    return {"target": (0.0, 2.0, 0.0), "distance": 5.5, "spheres": [(c - 2.0, float(r), 0.0) for r in range(5) for c in range(5)]}


def disc(fixture, centre, scale, width=WIDTH, height=HEIGHT):
    """(cx, cy, rx, ry) in pixels of the projected sphere, scaled; centre-plane projection of the frozen camera."""
    aspect, tan_half = width / height, math.tan(math.radians(FOV_DEGREES) / 2)
    depth = fixture["distance"] - centre[2]
    ndc_x = (centre[0] - fixture["target"][0]) / (depth * tan_half * aspect)
    ndc_y = (centre[1] - fixture["target"][1]) / (depth * tan_half)
    return ((0.5 + 0.5 * ndc_x) * width, (0.5 - 0.5 * ndc_y) * height,
            scale * RADIUS / (depth * tan_half * aspect) * 0.5 * width, scale * RADIUS / (depth * tan_half) * 0.5 * height)


def inside(d, x, y):
    return ((x + 0.5 - d[0]) / d[2]) ** 2 + ((y + 0.5 - d[1]) / d[3]) ** 2 <= 1.0


def evaluate_mask(image, fixture):
    """Counts of the reactive mask against the sphere discs of `fixture`.

    marked = R within TOLERANCE of 0.75.  `core` pixels (eroded discs) must all be marked, no pixel clear of every dilated disc may be."""
    def cover(scale):
        covered = set()
        for centre in fixture["spheres"]:
            d = disc(fixture, centre, scale, image.width, image.height)
            for y in range(max(0, int(d[1] - d[3]) - 1), min(image.height, int(d[1] + d[3]) + 2)):
                for x in range(max(0, int(d[0] - d[2]) - 1), min(image.width, int(d[0] + d[2]) + 2)):
                    if inside(d, x, y):
                        covered.add(y * image.width + x)
        return covered
    eroded, dilated = cover(ERODED), cover(DILATED)
    core = marked_core = outside_marked = marked = 0
    values = set()
    for index in range(image.width * image.height):
        v = image.rgb[index * 3]
        is_marked = abs(v - REACTIVE) <= TOLERANCE
        marked += is_marked
        values.add(round(v, 2))
        if index in eroded:
            core += 1
            marked_core += is_marked
        elif is_marked and index not in dilated:
            outside_marked += 1
    return {"core_pixels": core, "core_marked": marked_core, "marked_total": marked, "marked_outside_discs": outside_marked,
            "distinct_values": sorted(values)[:8]}


def glass_ok(row):
    """The glass predicate: the whole core is marked, nothing clear of the spheres is."""
    return row["core_pixels"] > 0 and row["core_marked"] == row["core_pixels"] and row["marked_outside_discs"] == 0


def opaque_ok(row):
    """The control: no pixel at all is marked (no transmission, no cut-out in the scene)."""
    return row["marked_total"] == 0 and row["core_pixels"] > 0


# --- cases ---------------------------------------------------------------------------------------------------------------
LIT = ["--rt", "on", "--shadows", "rt", "--lighting", "brute", "--gi", "off", "--reflections", "rt", "--ao", "off",
       "--lighting-denoise", "off"]
LEGACY = ["--rt", "off", "--shadows", "off", "--lighting", "legacy", "--gi", "off", "--reflections", "probes", "--ao", "off",
          "--lighting-denoise", "off"]
OVERFLOW = ["--debug-meshlets", "1", "--debug-meshlets-corrupt", "count"]


def case_table(out, binary):
    def case(name, scenario, mode, path, fixture, role, extra=()):
        folder = out / name
        command = [str(binary), "--scene", "procedural", "--bench", "2", "--material-scenario", scenario,
                   "--render-path", "visibility", "--geometry-path", "mesh", *(LIT if mode == "lit" else LEGACY),
                   "--post", "--upscaler", "native", "--resolution", f"{WIDTH}x{HEIGHT}", "--frames", str(FRAMES), "--warmup", "0",
                   "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing", "--lighting-seed", "1",
                   "--capture-linear", str(folder / "reactive.pfm"), "--capture-linear-frame", str(CAPTURE_FRAME),
                   "--capture-linear-signal", "reactive", "--report", str(folder / "report.json"),
                   *(OVERFLOW if path == "overflow" else []), *extra]
        return {"name": name, "role": role, "fixture": fixture, "path": path, "mode": mode, "command": command,
                "folder": str(folder), "expected_exit": 1 if path == "overflow" else 0}
    cases = []
    for mode in ("lit", "legacy"):
        for path in ("visibility", "overflow"):
            cases.append(case(f"glass-{mode}-{path}", "transmission-thin-volume", mode, path, "thin_volume", "glass"))
    cases.append(case("glass-motion-lit", "transmission-roughness-ior", "lit", "visibility", "roughness_ior", "glass",
                      ["--transmission-motion"]))
    for mode in ("lit", "legacy"):
        for path in ("visibility", "overflow"):
            cases.append(case(f"opaque-{mode}-{path}", "base-grid-sun", mode, path, "base_grid", "opaque-control"))
    return cases


@contextlib.contextmanager
def gpu_lock(path):
    with pathlib.Path(path).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GpuBusy("another cooperating Phosphor GPU verification owns the lock")
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def running_gpu_processes():
    text = subprocess.run(["ps", "-axo", "comm="], capture_output=True, text=True, check=True).stdout
    return sorted({pathlib.Path(line.strip()).name for line in text.splitlines() if GPU_PROCESS.match(pathlib.Path(line.strip()).name)})


def run_cases(cases, timeout):
    import os
    env = dict(os.environ, MTL_DEBUG_LAYER="1", MTL_SHADER_VALIDATION="1", MTL_DEBUG_LAYER_WARNING_MODE="nslog")
    for case in cases:
        folder = pathlib.Path(case["folder"])
        if folder.exists() and any(folder.iterdir()):
            raise CaptureFailed(f"{folder} is not empty: choose a new --out")
    with gpu_lock(pathlib.Path("/tmp/phosphor-gpu-verification.lock")):
        for position, case in enumerate(cases):
            busy = running_gpu_processes()
            if busy:
                raise GpuBusy(f"refusing to start {case['name']}: other GPU jobs are running: {', '.join(busy)}")
            folder = pathlib.Path(case["folder"])
            folder.mkdir(parents=True, exist_ok=True)
            status = run_checked(case["command"], folder / "log.txt", expected=case["expected_exit"], timeout=timeout, env=env)
            if not status["passed"]:
                raise CaptureFailed(f"{case['name']}: capture failed: {status['failures']}")
            if position + 1 < len(cases):
                time.sleep(2)


def functional(case):
    log = (pathlib.Path(case["folder"]) / "log.txt").read_text(errors="replace")
    errors = []
    if not re.search(rf"^EXIT {case['expected_exit']}$", log, re.M):
        errors.append(f"EXIT marker is not {case['expected_exit']}")
    if GPU_ERROR.search(log):
        errors.append("GPU/API/shader validation failure in the log")
    return errors


def evaluate(cases):
    fixtures = {"thin_volume": thin_volume(), "roughness_ior": roughness_ior(), "base_grid": base_grid()}
    results = {}
    for case in cases:
        image = read_pfm(pathlib.Path(case["folder"]) / "reactive.pfm")
        row = evaluate_mask(image, fixtures[case["fixture"]])
        row["glass_predicate"] = glass_ok(row)
        row["role"], row["path"], row["mode"] = case["role"], case["path"], case["mode"]
        if case["role"] == "glass":
            row["passed"] = row["glass_predicate"]
            row["expectation"] = "every pixel of the eroded sphere discs is reactive 0.75 and nothing outside the dilated discs is"
        else:
            # The control passes when it is a faithful negative: no pixel marked, hence the glass predicate FAILS on it.
            row["passed"] = opaque_ok(row) and not row["glass_predicate"]
            row["expectation"] = "no pixel is marked 0.75 in the opaque scene, so the glass predicate fails on it"
        errors = functional(case)
        if errors:
            row["passed"], row["functional_errors"] = False, errors
        results[case["name"]] = row
    return results


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=pathlib.Path, default=REPO / "build/release/phosphor")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-reactive")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--plan", action="store_true")
    parser.add_argument("--timeout", type=float, default=300)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    out, binary = args.out.resolve(), args.binary.resolve()
    cases = case_table(out, binary)
    out.mkdir(parents=True, exist_ok=True)
    if args.plan:
        for case in cases:
            print(" ".join(case["command"]))
        print(f"NOT_EXECUTED: planned {len(cases)} cases")
        return EXIT_OK
    try:
        if args.run:
            if not binary.is_file():
                print("--run requires an existing compiled renderer; this runner never builds", file=sys.stderr)
                return EXIT_USAGE
            run_cases(cases, args.timeout)
        results = evaluate(cases)
    except GpuBusy as error:
        print(f"GPU busy: {error}", file=sys.stderr)
        return EXIT_BUSY
    except CaptureFailed as error:
        print(str(error), file=sys.stderr)
        return EXIT_CAPTURE
    ok = all(row["passed"] for row in results.values())
    status = "ALL EXPECTATIONS MET" if ok else "EXPECTATION FAILED"
    (out / "results.json").write_text(json.dumps({"status": status, "cases": results}, indent=2) + "\n")
    lines = [f"# F15.1 ticket 13 reactive mask: {status}"]
    for name, row in results.items():
        lines.append(f"- {name}: {'PASS' if row['passed'] else 'FAIL'} ({row['role']}) core {row['core_marked']}/{row['core_pixels']} marked, "
                     f"marked outside the discs {row['marked_outside_discs']}, marked total {row['marked_total']}, "
                     f"values {row['distinct_values']}{' ' + str(row.get('functional_errors', '')) if row.get('functional_errors') else ''}")
    print("\n".join(lines))
    print(f"{status} (exit {EXIT_OK if ok else EXIT_FAILED})")
    return EXIT_OK if ok else EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
