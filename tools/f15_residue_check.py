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
  transmissive  C22-D3 follow-up (IORTestGrid "dark ring" count): smooth solid glass spheres (transmission 1, IOR 1.0 .. 2.42) in a white furnace: no pixel of
            the discs, the silhouette ring included, may be darker than the furnace radiance (the glass is energy conserving); the same glass with
            an absorbing medium is the negative control (the gate must fail).  Not a fix: the engine already conserves energy there.
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


# --- transmissive spheres in a white furnace (C22-D3 follow-up: the IORTestGrid "dark ring" count) -----------------------------------------------

# Smooth solid glass (transmission 1, white, ior as the IORTestGrid columns, volume thickness 0.1) in an environment that is the only thing any
# ray can find: no ray is absorbed (attenuation distance infinite, tint 1), so the Fresnel-chosen reflection plus the walked refraction must give
# back exactly the environment at EVERY pixel of the disc, the silhouette ring included (a white furnace: radiance 0.5, the declared rig).  The
# estimator weights are (1 - F) / (1 - pReflect) and F / pReflect, i.e. ~1 per sample: the settled frame is nearly noise free.
IORS = (1.0, 1.33, 1.5, 1.76, 2.42)
GLASS_SPACING = 2.4
FURNACE_DARK_RATIO = 0.70
FURNACE_LOW_RATIO = 0.90
FURNACE_LOW_FRACTION = 0.002  # settled-frame noise: single interior pixels of the stochastic Fresnel choice (measured 9 of 83540 below 0.96)
FURNACE_MEAN_RANGE = (0.97, 1.03)


ABSORBING = {"attenuationColor": [0.5, 0.5, 0.5], "attenuationDistance": 0.2}  # the negative control: a medium that really removes energy


def furnace_scene(folder, glass=True, segments=48, absorbing=False):
    scene = fx.Scene()
    for k, ior in enumerate(IORS):
        extensions = {"KHR_materials_transmission": {"transmissionFactor": 1.0}, "KHR_materials_ior": {"ior": ior},
                      "KHR_materials_volume": {"thicknessFactor": 0.1, **(ABSORBING if absorbing else {})}}
        material = (scene.material(f"glass ior {ior}", (1.0, 1.0, 1.0), 0.0, 0.0, extensions) if glass
                    else scene.material(f"black ior {ior}", (0.0, 0.0, 0.0), 0.0, 1.0))  # the opaque twin only gives the silhouette
        scene.sphere(((k - 0.5 * (len(IORS) - 1)) * GLASS_SPACING, 0.0, 0.0), 1.0, segments, material)
    return scene, scene.write(folder)


def measure_furnace(binary, glass_path, twin_path, out, mode="rt", twin=None):
    twin = twin or fx.read_png_gray(fx.capture(binary, twin_path, out / "twin", {"--reflections": "off"}))
    disc = [(x, y) for y in range(1, twin[1] - 1) for x in range(1, twin[0] - 1) if tuple(twin[3][y][x * twin[2]:x * twin[2] + 3]) != tuple(twin[3][0][0:3])]
    ring = ring_pixels(twin)
    image = fx.read_png_gray(fx.capture(binary, glass_path, out / "glass", {"--reflections": mode, "--gi": "off"}))
    background = fx.luminance(image, 2, 2)
    ratios = {p: fx.luminance(image, *p) / background for p in disc}
    ring_set = set(ring)
    dark_ring = [p for p in ring if ratios[p] < FURNACE_DARK_RATIO]
    dark_disc = [p for p in disc if ratios[p] < FURNACE_DARK_RATIO]
    return {"twin": twin, "disc_pixels": len(disc), "ring_pixels": len(ring), "background": background,
            "mean_ratio": round(sum(ratios.values()) / len(ratios), 4), "ring_mean_ratio": round(sum(ratios[p] for p in ring_set) / len(ring_set), 4),
            "min_ratio": round(min(ratios.values()), 3),
            "low_fraction": round(sum(1 for r in ratios.values() if r < FURNACE_LOW_RATIO) / len(ratios), 5), "dark_ring_pixels": len(dark_ring), "dark_disc_pixels": len(dark_disc), "first": dark_disc[:6]}


def furnace_gate(m):
    return (m["dark_disc_pixels"] == 0 and m["low_fraction"] <= FURNACE_LOW_FRACTION and FURNACE_MEAN_RANGE[0] <= m["mean_ratio"] <= FURNACE_MEAN_RANGE[1]
            and FURNACE_MEAN_RANGE[0] <= m["ring_mean_ratio"] <= FURNACE_MEAN_RANGE[1])


def case_transmissive(binary, control, out):
    _, glass_path = furnace_scene(out / "transmissive" / "scene-glass")
    _, twin_path = furnace_scene(out / "transmissive" / "scene-twin", glass=False)
    fixed = measure_furnace(binary, glass_path, twin_path, out / "transmissive" / "fixed")
    ok = furnace_gate(fixed)
    # Negative control of the fixture itself (same binary): the same glass with a medium that absorbs (Beer-Lambert, attenuation colour 0.5 over
    # 0.2 m: ~1 / 1000 survives a diameter) must read dark, so the gate does fail when energy is lost.  (Screen-space reflections or reflections off
    # are no control: a white diffuse sphere reads exactly the furnace radiance, albedo 1 x environment 0.5.)
    _, absorbing_path = furnace_scene(out / "transmissive" / "scene-absorbing", absorbing=True)
    absorbing_result = measure_furnace(binary, absorbing_path, twin_path, out / "transmissive" / "negative-absorbing", twin=fixed["twin"])
    gate_can_fail = not furnace_gate(absorbing_result)
    for entry in (fixed, absorbing_result):
        entry.pop("twin")  # the decoded image is not a report value
    report = {"fixed": fixed, "negative_control_absorbing": absorbing_result, "negative_control_fails_as_required": gate_can_fail, "gates_hold": ok and gate_can_fail}
    # No defect was found (the IORTestGrid count is a metric artefact, see the doc): there is no build WITHOUT a fix, so --control-binary is not
    # asked to fail here (control_fails None = not applicable); the discriminating control is the absorbing run above.
    return ok and gate_can_fail, True if control else None, report


CASES = {"grazing": case_grazing, "mirror-env": case_mirror_env, "backface": case_backface, "transmissive": case_transmissive}


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
