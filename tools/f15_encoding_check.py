#!/usr/bin/env python3
"""F15.3 (ticket 16, box 2): the glTF encodings of the thin subsurface lobe against the material description file, in IMAGES.

One generated scene (tools/f15_fixture_scene.py), three spheres, rendered under the declared corpus rig:
  encoding A  glTF materials carrying KHR_materials_diffuse_transmission (two spheres) and the draft KHR_materials_scatter with
              KHR_materials_transmission (one sphere);
  encoding B  the same glTF with PLAIN materials plus `--material-overrides` (the description file, schema
              phosphor-material-overrides) holding the subsurface (and transmission) blocks the loader maps the encodings to.
The two captures must be IDENTICAL (0 pixels of the 8-bit PNG differ: the loader maps both encodings to the same MaterialDescription
and the `source` field is provenance only).  Baseline and negative controls:
  repeat   encoding A captured twice: 0 pixels (the capture is deterministic: otherwise an inequality would prove nothing);
  control  encoding B with one sphere's weight changed by 0.02 and with its colour changed by 0.02 in one channel: each must differ
           (the comparison would see a small difference).

Exit codes: 0 equal and every control differs, 1 the encodings differ or the repeat differs, 3 a control did not differ,
4 GPU busy, 5 internal error, 6 capture failed.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_fixture_scene as fx  # noqa: E402

corpus = fx.corpus

if os.environ.get("F15_GPU_LOCK_HELD"):  # the caller holds the shared GPU lock for the whole run (gpulock.py): no nested lock, no polling
    import contextlib
    corpus.gpu_lock = lambda *_a, **_k: contextlib.nullcontext()
    corpus.running_gpu_processes = lambda *_a, **_k: []

# (name, centre x, kind, weight, colour, anisotropy): the values of the three spheres
DIFFUSE = [("dt-orange", -2.4, 0.60, (1.0, 0.50, 0.25)), ("dt-white", 0.0, 0.85, (1.0, 1.0, 1.0))]
SCATTER = ("scatter-teal", 2.4, 0.80, 0.50, (0.90, 0.70, 0.40), 0.30)  # transmission, scatterStrength, multiscatterColor, g


def build_scene(folder, encoded):
    """The scene with the glTF encodings (encoded=True) or plain materials (False); the same geometry and material order."""
    scene = fx.Scene()
    base = (0.6, 0.7, 0.9)
    for name, x, weight, colour in DIFFUSE:
        extensions = {"KHR_materials_diffuse_transmission": {"diffuseTransmissionFactor": weight, "diffuseTransmissionColorFactor": list(colour)}} \
            if encoded else None
        scene.sphere((x, 0.0, 0.0), 1.0, 48, scene.material(name, base, 0.0, 0.5, extensions))
    name, x, transmission, strength, colour, g = SCATTER
    extensions = {"KHR_materials_transmission": {"transmissionFactor": transmission},
                  "KHR_materials_scatter": {"scatterStrengthFactor": strength, "multiscatterColorFactor": list(colour), "scatterAnisotropy": g}} \
        if encoded else None
    scene.sphere((x, 0.0, 0.0), 1.0, 48, scene.material(name, base, 0.0, 0.5, extensions, double_sided=True))
    # a lit back wall so the transmitted light has something to show
    scene.quad([(-5, -1.5, -2.5), (5, -1.5, -2.5), (5, 3.0, -2.5), (-5, 3.0, -2.5)], scene.material("wall", (0.8, 0.8, 0.8), 0.0, 1.0))
    return scene, scene.write(folder)


def overrides(delta_weight=0.0, delta_colour=0.0):
    """The description-file overrides equal to the encodings (what the loader maps them to: weight, colour, mode thin; for the draft scatter
    weight = transmission * scatterStrength and the colour as authored); the deltas perturb the first diffuse sphere (negative controls)."""
    entries = []
    for k, (name, _, weight, colour) in enumerate(DIFFUSE):
        c = list(colour)
        if k == 0:
            c[2] += delta_colour
        entries.append({"name": name, "subsurface": {"mode": "thin", "weight": weight + (delta_weight if k == 0 else 0.0), "color": c}})
    name, _, transmission, strength, colour, g = SCATTER
    entries.append({"name": name, "transmission": {"factor": transmission},
                    "subsurface": {"mode": "thin", "weight": transmission * strength, "color": list(colour), "anisotropy": g}})
    return {"schema": "phosphor-material-overrides", "version": 1, "overrides": entries}


def differing_pixels(a, b):
    """(count of pixels with any channel different, max channel delta) of two images from f15_shadowed_hit_check.read_png_gray."""
    wa, ha, ca, rows_a = a
    wb, hb, cb, rows_b = b
    if (wa, ha, ca) != (wb, hb, cb):
        raise ValueError("image sizes differ")
    count, worst = 0, 0
    for ra, rb in zip(rows_a, rows_b):
        if bytes(ra) == bytes(rb):
            continue
        for x in range(wa):
            d = max(abs(ra[x * ca + c] - rb[x * ca + c]) for c in range(min(ca, 3)))
            if d:
                count += 1
                worst = max(worst, d)
    return count, worst


def run(binary, out):
    out = pathlib.Path(out)
    encoded_scene = build_scene(out / "encoded", True)[1]
    plain_scene = build_scene(out / "plain", False)[1]
    files = {}
    for name, delta in (("overrides", (0.0, 0.0)), ("control-weight", (0.02, 0.0)), ("control-colour", (0.0, 0.02))):
        path = out / f"{name}.json"
        path.write_text(json.dumps(overrides(*delta), indent=1))
        files[name] = path
    shots = {}
    for name, scene, extra in (("a-gltf", encoded_scene, None), ("a-gltf-repeat", encoded_scene, None),
                               ("b-description", plain_scene, ["--material-overrides", str(files["overrides"])]),
                               ("c-weight", plain_scene, ["--material-overrides", str(files["control-weight"])]),
                               ("c-colour", plain_scene, ["--material-overrides", str(files["control-colour"])])):
        shots[name] = fx.read_png_gray(fx.capture(binary, scene, out / name, extra=extra, name="encoding-fixture"))
    result = {"gltf_vs_description": differing_pixels(shots["a-gltf"], shots["b-description"]),
              "repeat": differing_pixels(shots["a-gltf"], shots["a-gltf-repeat"]),
              "control_weight": differing_pixels(shots["b-description"], shots["c-weight"]),
              "control_colour": differing_pixels(shots["b-description"], shots["c-colour"])}
    ok = result["gltf_vs_description"][0] == 0 and result["repeat"][0] == 0
    controls_differ = result["control_weight"][0] > 0 and result["control_colour"][0] > 0
    return ok, controls_differ, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-encoding")
    args = parser.parse_args(argv)
    try:
        ok, controls_differ, result = run(args.binary.resolve(), args.out)
    except corpus.GpuBusy:
        print("GPU busy")
        return 4
    except corpus.CaptureFailed as error:
        print(error)
        return 6
    print(json.dumps(result, indent=2))
    if not ok:
        return 1
    return 0 if controls_differ else 3


if __name__ == "__main__":
    sys.exit(main())
