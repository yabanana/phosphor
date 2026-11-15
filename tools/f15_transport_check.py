#!/usr/bin/env python3
"""F15.1 (tickets 14 and 16) transport-closure fixtures: generated glTF scenes, measured on the linear HDR capture (PFM).

Three checks, each with its negative control (a control must FAIL, otherwise the fixture is redesigned, never the threshold):

  volume    Ticket 14, the interior of a closed volume.  A white wall, a closed glass cube (ior 1.5, Beer-Lambert attenuation, solid) hovering in
            front of it and an emissive quad INSIDE the cube (a sampled emitter: ReSTIR DI, RT visibility).  A wall point beside the cube sees
            the emitter through the cube's side wall, and its shadow ray ENDS inside the medium: the transmittance of the open segment is
            exp(-sigma_t |L - E|) (E = the entry point), times the entry Fresnel factor.  Three captures with the same geometry: A = no
            attenuation, B = attenuation, C = emitter off.  The sun and the environment cancel (D = capture - C), the Fresnel factor cancels
            in B/A, so  sum(D_B) / sum(D_A)  over the ROI (per channel) must equal  sum(D_A T) / sum(D_A)  with the per-pixel analytic
            T = mean over a 5x5 grid of the emitter of exp(-sigma_t |L_s - E_s|), within 5 % per channel (decision 11 transmission bound).
            The engine with the interior clamp (T <= 1: the open segment is lost) gives B/A = 1/(1 - F) instead.
            Controls: `clamp-model` (the expectation of the clamped engine, T = 1) and `no-attenuation` (sigma = 0).
  masked    Ticket 14, a transmissive AND alpha-masked caster.  A thin-walled glass pane (ior 1, tint (0.9, 0.5, 0.3), transmission 1)
            whose base-colour texture has alpha 0 on the left half and 1 on the right half, alphaMode MASK, cutoff 0.5, in front of the wall,
            sun with RT shadows.  The shadow of the left half must be the unshadowed wall (T = 1: a texel below the cutoff does not exist), the
            shadow of the right half the tint (T = tint: a texel above the cutoff accumulates transmittance, it does not block).  Arm O has
            the same pane OPAQUE (ordinary opaque material): the sun is fully blocked, so T_k = (M_k - O_k) / (lit - O_k) (ambient cancels).
            Controls: `opaque-model` (right half T = 0, what the masked-opaque rule gives) and `ignored-model` (T = 1 everywhere).
  thin-backlit  Ticket 16, the back-lit thin receiver: a thin diffuse-transmission quad lit ONLY by a sampled emitter behind it (the corpus sun is on
            the camera side: a sun cannot back-light a receiver the camera sees).  The RT visibility ray of a back-lit thin receiver starts on the light
            side of the surface and must not shadow it: D = capture - capture(emitter off); sum(D, --rt on) / sum(D, --rt off) in [0.95, 1.05].
            Control: an opaque occluder between emitter and receiver must take the ratio below 0.45 (the fixture sees visibility).

Usage: python3 -I tools/f15_transport_check.py {volume|masked|thin-backlit} --binary B [--control-binary B0] --out DIR [--evaluate]
Exit codes: 0 gates hold (and every control fails as it must; a control binary, when given, fails the gate), 1 a gate failed, 3 a control did not
fail, 4 GPU busy, 6 capture failed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import struct
import subprocess
import sys
import time
import zlib

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_corpus as corpus  # noqa: E402  (the runner's GPU guard and the declared capture command)
from temporal_light_metrics import read_pfm  # noqa: E402

WIDTH, HEIGHT = 1024, 768
FOV_Y = math.radians(60.0)
MARGIN = 1.12  # CorpusRig::margin (src/testbench/corpus_rig.h)
TAN_HALF = math.tan(0.5 * FOV_Y)
ASPECT = WIDTH / HEIGHT
BOUND = 0.05  # decision 11 transmission bound (relative, per channel)
FRAMES, WARMUP, ACCUMULATE = 64, 48, 16


# --- scene writer -------------------------------------------------------------------------------------------------

class GltfScene:
    """Indexed quads/triangles with PBR materials, optional UVs and one optional RGBA texture; every primitive is its own node."""

    def __init__(self):
        self.materials, self.primitives, self.extensions, self.images = [], [], set(), []

    def material(self, name, color=(0.8, 0.8, 0.8), alpha=1.0, metallic=0.0, roughness=1.0, extensions=None, double_sided=False,
                 emissive=None, alpha_mode=None, alpha_cutoff=0.5, texture=None):
        pbr = {"baseColorFactor": [*color, alpha], "metallicFactor": metallic, "roughnessFactor": roughness}
        if texture is not None:
            self.images.append(texture)
            pbr["baseColorTexture"] = {"index": len(self.images) - 1}
        entry = {"name": name, "pbrMetallicRoughness": pbr}
        if extensions:
            entry["extensions"] = dict(extensions)
            self.extensions.update(extensions)
        if emissive is not None:
            entry["emissiveFactor"] = list(emissive)
        if alpha_mode:
            entry["alphaMode"] = alpha_mode
            if alpha_mode == "MASK":
                entry["alphaCutoff"] = alpha_cutoff
        if double_sided:
            entry["doubleSided"] = True
        self.materials.append(entry)
        return len(self.materials) - 1

    def quad(self, corners, material, uvs=None):
        a, b, c = corners[:3]
        u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
        n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
        length = math.sqrt(sum(x * x for x in n))
        n = tuple(x / length for x in n)
        self.primitives.append((list(corners), [n] * 4, list(uvs) if uvs else [(0.0, 0.0)] * 4, [0, 1, 2, 0, 2, 3], material))

    def box(self, lo, hi, material):
        """Closed box, outward counter-clockwise faces."""
        x0, y0, z0 = lo
        x1, y1, z1 = hi
        faces = [((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)), ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)),
                 ((x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)), ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)),
                 ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)), ((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0))]
        for face in faces:
            self.quad(face, material)

    def bounds(self):
        points = [p for positions, *_ in self.primitives for p in positions]
        return tuple(min(p[i] for p in points) for i in range(3)), tuple(max(p[i] for p in points) for i in range(3))

    def write(self, folder):
        folder = pathlib.Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        blob, views, accessors, meshes, nodes = bytearray(), [], [], [], []
        for k, (positions, normals, uvs, indices, material) in enumerate(self.primitives):
            def view(data):
                while len(blob) % 4:
                    blob.append(0)
                views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
                blob.extend(data)
                return len(views) - 1
            pos = view(b"".join(struct.pack("<3f", *p) for p in positions))
            nor = view(b"".join(struct.pack("<3f", *n) for n in normals))
            tex = view(b"".join(struct.pack("<2f", *t) for t in uvs))
            idx = view(b"".join(struct.pack("<I", i) for i in indices))
            lo = [min(p[i] for p in positions) for i in range(3)]
            hi = [max(p[i] for p in positions) for i in range(3)]
            base = len(accessors)
            accessors += [{"bufferView": pos, "componentType": 5126, "count": len(positions), "type": "VEC3", "min": lo, "max": hi},
                          {"bufferView": nor, "componentType": 5126, "count": len(positions), "type": "VEC3"},
                          {"bufferView": tex, "componentType": 5126, "count": len(positions), "type": "VEC2"},
                          {"bufferView": idx, "componentType": 5125, "count": len(indices), "type": "SCALAR"}]
            meshes.append({"primitives": [{"attributes": {"POSITION": base, "NORMAL": base + 1, "TEXCOORD_0": base + 2}, "indices": base + 3,
                                           "material": material}]})
            nodes.append({"mesh": k})
        document = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
                    "materials": self.materials, "accessors": accessors, "bufferViews": views, "buffers": [{"uri": "scene.bin", "byteLength": len(blob)}]}
        if self.images:
            document["images"] = [{"uri": f"texture{i}.png"} for i in range(len(self.images))]
            document["textures"] = [{"source": i, "sampler": 0} for i in range(len(self.images))]
            document["samplers"] = [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}]
            for i, png in enumerate(self.images):
                (folder / f"texture{i}.png").write_bytes(png)
        if self.extensions:
            document["extensionsUsed"] = sorted(self.extensions)
        (folder / "scene.bin").write_bytes(bytes(blob))
        path = folder / "scene.gltf"
        path.write_text(json.dumps(document))
        return path


def png_rgba(width, height, pixels):
    """Minimal 8-bit RGBA PNG (stdlib only); pixels = rows of (r, g, b, a)."""
    raw = b"".join(b"\x00" + b"".join(bytes(px) for px in row) for row in pixels)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


# --- the declared corpus rig (yaw 0, pitch 0) -----------------------------------------------------------------------

def frame_rig(bounds_min, bounds_max):
    centre = [0.5 * (a + b) for a, b in zip(bounds_min, bounds_max)]
    half = [0.5 * (b - a) for a, b in zip(bounds_min, bounds_max)]
    distance = 1e-3
    for corner in range(8):
        r = half[0] if corner & 1 else -half[0]
        u = half[1] if corner & 2 else -half[1]
        b = half[2] if corner & 4 else -half[2]
        distance = max(distance, MARGIN * max(abs(r) / (TAN_HALF * ASPECT), abs(u) / TAN_HALF) + b)
    return centre, distance


def wall_point(px, py, centre, distance, z=0.0):
    """World point on the plane z of the pixel centre (px, py) (the camera sits at centre.z + distance, looks along -z)."""
    depth = distance - (z - centre[2])
    ndc_x, ndc_y = (px + 0.5) / WIDTH * 2.0 - 1.0, 1.0 - (py + 0.5) / HEIGHT * 2.0
    return (centre[0] + ndc_x * depth * TAN_HALF * ASPECT, centre[1] + ndc_y * depth * TAN_HALF, z)


def pixel_box(rect, centre, distance, z=0.0):
    """Pixel box (x0, y0, x1, y1) of a world rectangle (x0, y0, x1, y1) in the plane z."""
    def project(x, y):
        depth = distance - (z - centre[2])
        return ((0.5 + 0.5 * (x - centre[0]) / (depth * TAN_HALF * ASPECT)) * WIDTH, (0.5 - 0.5 * (y - centre[1]) / (depth * TAN_HALF)) * HEIGHT)
    x0, y0 = project(rect[0], rect[3])
    x1, y1 = project(rect[2], rect[1])
    return int(math.ceil(x0)), int(math.ceil(y0)), int(math.floor(x1)), int(math.floor(y1))


# --- capture ---------------------------------------------------------------------------------------------------------

_HELD = False


class held_lock:
    """Hold the shared GPU verification lock across a whole check (all its captures and the control-binary captures): a BLOCKING flock (a
    polling lock is starved by the blocking waiters of the other agents), then no other phosphor / soc_bench / f<N>_spike process (the
    orchestrator's gpulock.py rule).  Captures of one check are short (about a minute each)."""

    def __enter__(self):
        import fcntl
        global _HELD
        self.file = open("/tmp/phosphor-gpu-verification.lock", "a+")
        fcntl.flock(self.file, fcntl.LOCK_EX)
        while corpus.running_gpu_processes():
            time.sleep(3)
        _HELD = True
        return self

    def __exit__(self, *exc):
        global _HELD
        _HELD = False
        self.file.close()


def run_locked(command, folder, env_extra=None):
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not _HELD:
        with held_lock():
            return run_locked(command, folder, env_extra)
    env = {k: v for k, v in os.environ.items() if k not in corpus.VALIDATION_ENV}
    env.update(env_extra or {})
    with (folder / "engine.log").open("w") as log:
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def set_flag(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = value
    else:
        command += [flag, value]
    return command


def capture_scene(binary, scene, folder, flags=None, extra=None, accumulate=True):
    """Corpus rig capture (ReSTIR DI, RT sun shadows by default) of a generated glTF; returns the linear PFM path."""
    folder = pathlib.Path(folder)
    corpus.MODELS["transport"] = {"tier": "A", "view": (0, 0), "shows": "generated fixture"}
    corpus.gltf_path = lambda corpus_dir, n: scene
    command = corpus.capture_command(binary, None, "transport", folder)
    set_flag(command, "--frames", str(FRAMES))
    set_flag(command, "--warmup", str(WARMUP))
    command += ["--capture-linear", str(folder / "engine.pfm"), "--capture-linear-frame", str(FRAMES - 1)]
    if accumulate:
        command += ["--capture-linear-accumulate", str(ACCUMULATE), "--capture-linear-signal", "hdr"]
    command += list(extra or [])
    for flag, value in (flags or {}).items():
        set_flag(command, flag, value)
    code = run_locked(command, folder)
    if code != 0 or not (folder / "engine.pfm").is_file():
        raise corpus.CaptureFailed(f"capture failed with exit {code}: {folder / 'engine.log'}")
    return folder / "engine.pfm"


# --- ticket 14: the interior of a closed volume ----------------------------------------------------------------------

CUBE_LO, CUBE_HI = (-0.22, -0.22, 0.13), (0.22, 0.22, 0.57)
EMITTER_CENTRE, EMITTER_HALF = (0.0, 0.0, 0.35), 0.03
ATTENUATION = ((0.6, 0.8, 0.95), 0.3)   # colour, distance (m): sigma_t = -ln(colour) / distance per channel
EMISSION = 40.0                         # KHR_materials_emissive_strength of the emitter (W / m2 / sr scale)
VOLUME_ROIS = {"right": (0.45, -0.12, 0.85, 0.12), "left": (-0.85, -0.12, -0.45, 0.12)}


def volume_scene(attenuated, emitter_on):
    s = GltfScene()
    wall = s.material("wall", color=(0.8, 0.8, 0.8))
    ext = {"KHR_materials_transmission": {"transmissionFactor": 1.0}, "KHR_materials_ior": {"ior": 1.5},
           "KHR_materials_volume": {"thicknessFactor": 0.44}}
    if attenuated:
        ext["KHR_materials_volume"].update({"attenuationColor": list(ATTENUATION[0]), "attenuationDistance": ATTENUATION[1]})
    glass = s.material("glass", color=(1, 1, 1), roughness=0.0, extensions=ext)
    light = s.material("emitter", color=(0, 0, 0), emissive=(1, 1, 1) if emitter_on else (0, 0, 0),
                       extensions={"KHR_materials_emissive_strength": {"emissiveStrength": EMISSION}} if emitter_on else None)
    s.quad([(-1.0, -0.6, 0.0), (1.0, -0.6, 0.0), (1.0, 0.6, 0.0), (-1.0, 0.6, 0.0)], wall)
    s.box(CUBE_LO, CUBE_HI, glass)
    cx, cy, cz = EMITTER_CENTRE
    h = EMITTER_HALF
    s.quad([(cx + h, cy - h, cz), (cx - h, cy - h, cz), (cx - h, cy + h, cz), (cx + h, cy + h, cz)], light)  # faces -z, towards the wall
    return s


def segment_entry(p, q, lo, hi):
    """Parameter s in [0, 1] where the segment p -> q first enters the box (None when it does not)."""
    s0, s1 = 0.0, 1.0
    for axis in range(3):
        d = q[axis] - p[axis]
        if abs(d) < 1e-12:
            if p[axis] < lo[axis] or p[axis] > hi[axis]:
                return None
            continue
        t0, t1 = (lo[axis] - p[axis]) / d, (hi[axis] - p[axis]) / d
        if t0 > t1:
            t0, t1 = t1, t0
        s0, s1 = max(s0, t0), min(s1, t1)
        if s0 > s1:
            return None
    return s0 if s0 > 0.0 else None


def volume_expectation(rois_pixels, centre, distance, d_a):
    """Per ROI: (sum(D_A T) / sum(D_A)) per channel, T the analytic open-segment transmittance, weighted by the measured D_A."""
    sigma = [-math.log(c) / ATTENUATION[1] for c in ATTENUATION[0]]
    cx, cy, cz = EMITTER_CENTRE
    samples = [(cx + (i - 2) * EMITTER_HALF / 2.0, cy + (j - 2) * EMITTER_HALF / 2.0, cz) for i in range(5) for j in range(5)]
    out = {}
    for name, (x0, y0, x1, y1) in rois_pixels.items():
        num, den, used, skipped = [0.0] * 3, [0.0] * 3, 0, 0
        for py in range(y0, y1):
            for px in range(x0, x1):
                point = wall_point(px, py, centre, distance)
                t = [0.0] * 3
                n = 0
                for light in samples:
                    s = segment_entry(point, light, CUBE_LO, CUBE_HI)
                    if s is None:
                        continue
                    d = math.dist(light, tuple(point[k] + s * (light[k] - point[k]) for k in range(3)))
                    for c in range(3):
                        t[c] += math.exp(-sigma[c] * d)
                    n += 1
                if n != len(samples):
                    skipped += 1
                    continue
                weights = d_a(px, py)
                for c in range(3):
                    num[c] += weights[c] * t[c] / n
                    den[c] += weights[c]
                used += 1
        out[name] = {"expected": [num[c] / den[c] for c in range(3)], "pixels": used, "skipped": skipped}
    return out


def measure_volume(pfm):
    """pfm: {'A': path, 'B': path, 'C': path}.  Returns the result dict (gates not yet evaluated)."""
    images = {k: read_pfm(v) for k, v in pfm.items()}
    width, height = images["A"].width, images["A"].height
    bounds = volume_scene(True, True).bounds()
    centre, distance = frame_rig(*bounds)
    boxes = {name: pixel_box(rect, centre, distance) for name, rect in VOLUME_ROIS.items()}

    def d(image, px, py):
        a, c = image.pixel(px, py), images["C"].pixel(px, py)
        return [a[k] - c[k] for k in range(3)]

    result = {"camera_distance": distance, "rois": {}}
    for name, box in boxes.items():
        x0, y0, x1, y1 = box
        sums = {"A": [0.0] * 3, "B": [0.0] * 3}
        for py in range(y0, y1):
            for px in range(x0, x1):
                for arm in ("A", "B"):
                    v = d(images[arm], px, py)
                    for c in range(3):
                        sums[arm][c] += v[c]
        expectation = volume_expectation({name: box}, centre, distance, lambda px, py: d(images["A"], px, py))[name]
        measured = [sums["B"][c] / sums["A"][c] for c in range(3)]
        result["rois"][name] = {"box": box, "d_a": sums["A"], "d_b": sums["B"], "measured_ratio": measured, "expected_ratio": expected_list(expectation),
                                "pixels": expectation["pixels"], "skipped": expectation["skipped"]}
    return result


def expected_list(e):
    return e["expected"]


def relative_errors(measured, expected):
    return [abs(m / e - 1.0) for m, e in zip(measured, expected)]


def gate_volume(result):
    gates, controls = {}, {}
    for name, roi in result["rois"].items():
        err = relative_errors(roi["measured_ratio"], roi["expected_ratio"])
        gates[name] = {"relative_error": err, "pass": max(err[:2]) <= BOUND and err[2] <= BOUND}
        clamp = relative_errors(roi["measured_ratio"], [1.0, 1.0, 1.0])
        controls.setdefault("clamp-model", {})[name] = {"relative_error": clamp, "fails": max(clamp[:2]) > BOUND}
    return gates, controls


def run_volume(args):
    out = pathlib.Path(args.out)
    arms = {"A": volume_scene(False, True), "B": volume_scene(True, True), "C": volume_scene(True, False)}
    pfm = {}
    binary = args.control_binary if args.use_control else args.binary
    base = out / ("control" if args.use_control else "fixed")
    flags = {"--shadows": "off", "--reflections": "off"}
    for arm, scene in arms.items():
        path = scene.write(base / f"scene-{arm}")
        pfm[arm] = args.evaluate and base / arm / "engine.pfm" or capture_scene(binary, path, base / arm, flags)
    result = measure_volume(pfm)
    gates, controls = gate_volume(result)
    result.update({"gates": gates, "controls": controls})
    return result


# --- ticket 14: transmissive + alpha-masked caster -------------------------------------------------------------------

PANE_Z = 0.4
PANE = (-0.5, -0.1, 0.5, 0.1)
TINT = (0.9, 0.5, 0.3)
SUN_OVER_Z = (-0.45 / 0.70, 0.55 / 0.70)  # towardSun.xy / towardSun.z of the rig at yaw 0, pitch 0
MASK_ROIS = {"left": (-0.18, -0.39, 0.18, -0.25), "right": (0.34, -0.39, 0.70, -0.25), "lit": (0.30, 0.25, 0.70, 0.45)}


def masked_scene(arm):
    """arm 'M': masked transmissive pane with the half-transparent texture; arm 'O': the same pane as an ordinary opaque material."""
    s = GltfScene()
    wall = s.material("wall", color=(0.8, 0.8, 0.8))
    texels = [[(255, 255, 255, 0)] * 8 + [(255, 255, 255, 255)] * 8]
    if arm == "M":
        pane = s.material("masked glass", color=TINT, roughness=0.0, double_sided=True, alpha_mode="MASK", alpha_cutoff=0.5,
                          texture=png_rgba(16, 1, texels),
                          extensions={"KHR_materials_transmission": {"transmissionFactor": 1.0}, "KHR_materials_ior": {"ior": 1.0}})
    else:
        pane = s.material("opaque pane", color=(0.5, 0.5, 0.5), double_sided=True)
    s.quad([(-1.0, -0.6, 0.0), (1.0, -0.6, 0.0), (1.0, 0.6, 0.0), (-1.0, 0.6, 0.0)], wall)
    x0, y0, x1, y1 = PANE
    s.quad([(x0, y0, PANE_Z), (x1, y0, PANE_Z), (x1, y1, PANE_Z), (x0, y1, PANE_Z)], pane, uvs=[(0, 1), (1, 1), (1, 0), (0, 0)])
    return s


def roi_mean(image, box):
    x0, y0, x1, y1 = box
    total, n = [0.0] * 3, 0
    for py in range(y0, y1):
        for px in range(x0, x1):
            p = image.pixel(px, py)
            for c in range(3):
                total[c] += p[c]
            n += 1
    return [t / n for t in total]


def measure_masked(pfm):
    images = {k: read_pfm(v) for k, v in pfm.items()}
    centre, distance = frame_rig(*masked_scene("M").bounds())
    boxes = {name: pixel_box(rect, centre, distance) for name, rect in MASK_ROIS.items()}
    means = {arm: {name: roi_mean(images[arm], box) for name, box in boxes.items()} for arm in images}
    lit = means["M"]["lit"]
    result = {"camera_distance": distance, "means": means, "transmittance": {}}
    for name in ("left", "right"):
        o, m = means["O"][name], means["M"][name]
        result["transmittance"][name] = [(m[c] - o[c]) / (lit[c] - o[c]) for c in range(3)]
    return result


def gate_masked(result):
    expected = {"left": [1.0, 1.0, 1.0], "right": list(TINT)}
    gates = {}
    for name, e in expected.items():
        t = result["transmittance"][name]
        err = [abs(t[c] - e[c]) / e[c] for c in range(3)]
        gates[name] = {"measured": t, "expected": e, "relative_error": err, "pass": max(err) <= BOUND}
    controls = {}
    for control, models in (("opaque-model", {"left": [1.0, 1.0, 1.0], "right": [0.0, 0.0, 0.0]}), ("ignored-model", {"left": [1.0] * 3, "right": [1.0] * 3})):
        row = {}
        for name, e in models.items():
            t = result["transmittance"][name]
            err = [abs(t[c] - e[c]) / max(e[c], 0.05) for c in range(3)]
            row[name] = {"relative_error": err, "fails": max(err) > BOUND}
        controls[control] = row
    return gates, controls


def run_masked(args):
    out = pathlib.Path(args.out)
    binary = args.control_binary if args.use_control else args.binary
    base = out / ("control" if args.use_control else "fixed")
    pfm = {}
    for arm in ("M", "O"):
        path = masked_scene(arm).write(base / f"scene-{arm}")
        pfm[arm] = args.evaluate and base / arm / "engine.pfm" or capture_scene(binary, path, base / arm, {"--shadows": "rt", "--reflections": "off"})
    result = measure_masked(pfm)
    gates, controls = gate_masked(result)
    result.update({"gates": gates, "controls": controls})
    return result


# --- ticket 16: the back-lit thin receiver --------------------------------------------------------------------------

THIN_ROI = (-0.15, -0.1, 0.15, 0.1)  # on the receiver quad (plane z = 0)
THIN_GATE = (0.95, 1.05)


def thin_scene(emitter_on, occluder):
    """A thin diffuse-transmission quad (weight 1) facing the camera, a sampled emitter quad BEHIND it facing it, optionally an opaque occluder
    between them.  The camera and the corpus sun are on the receiver's front side: the emitter reaches it only through the thin transmitted lobe."""
    s = GltfScene()
    thin = s.material("thin", color=(0.6, 0.6, 0.6), extensions={"KHR_materials_diffuse_transmission": {"diffuseTransmissionFactor": 1.0}})
    light = s.material("emitter", color=(0, 0, 0), emissive=(1, 1, 1) if emitter_on else (0, 0, 0),
                       extensions={"KHR_materials_emissive_strength": {"emissiveStrength": 20.0}} if emitter_on else None)
    block = s.material("occluder", color=(0.5, 0.5, 0.5), double_sided=True)
    s.quad([(-0.5, -0.3, 0.0), (0.5, -0.3, 0.0), (0.5, 0.3, 0.0), (-0.5, 0.3, 0.0)], thin)               # faces +z (the camera)
    s.quad([(-0.15, -0.15, -0.4), (0.15, -0.15, -0.4), (0.15, 0.15, -0.4), (-0.15, 0.15, -0.4)], light)   # faces +z, towards the receiver
    if occluder:
        s.quad([(-0.4, -0.25, -0.2), (0.4, -0.25, -0.2), (0.4, 0.25, -0.2), (-0.4, 0.25, -0.2)], block)
    return s


def thin_roi_sum(path, box):
    image = read_pfm(path)
    total = [0.0] * 3
    for py in range(box[1], box[3]):
        for px in range(box[0], box[2]):
            p = image.pixel(px, py)
            for c in range(3):
                total[c] += p[c]
    return total


def run_thin_backlit(args):
    out = pathlib.Path(args.out)
    binary = args.control_binary if args.use_control else args.binary
    base = out / ("control" if args.use_control else "fixed")
    arms = {"on": (thin_scene(True, False), {"--rt": "on", "--shadows": "off", "--reflections": "off"}),
            "off": (thin_scene(True, False), {"--rt": "off", "--shadows": "off", "--reflections": "off"}),
            "none": (thin_scene(False, False), {"--rt": "on", "--shadows": "off", "--reflections": "off"}),
            "blocked": (thin_scene(True, True), {"--rt": "on", "--shadows": "off", "--reflections": "off"})}
    pfm = {}
    for name, (scene, flags) in arms.items():
        path = scene.write(base / f"scene-{name}")
        pfm[name] = args.evaluate and base / name / "engine.pfm" or capture_scene(binary, path, base / name, flags)
    centre, distance = frame_rig(*thin_scene(True, True).bounds())
    box = pixel_box(THIN_ROI, centre, distance)
    sums = {name: thin_roi_sum(pfm[name], box) for name in pfm}
    d_on = [sums["on"][c] - sums["none"][c] for c in range(3)]
    d_off = [sums["off"][c] - sums["none"][c] for c in range(3)]
    d_blocked = [sums["blocked"][c] - sums["none"][c] for c in range(3)]
    ratio = [d_on[c] / d_off[c] if d_off[c] > 0 else 0.0 for c in range(3)]
    blocked = [d_blocked[c] / d_off[c] if d_off[c] > 0 else 0.0 for c in range(3)]
    signal = [d_off[c] / sums["none"][c] for c in range(3)]
    result = {"box": box, "pixels": (box[2] - box[0]) * (box[3] - box[1]), "d_on": d_on, "d_off": d_off, "ratio_on_off": ratio,
              "signal_over_unlit": signal, "blocked_over_off": blocked}
    result["gates"] = {"visibility-keeps-the-back-light": {"ratio": ratio, "pass": min(signal) > 0.05 and all(THIN_GATE[0] <= r <= THIN_GATE[1] for r in ratio)}}
    # the control must FAIL the gate: with an opaque occluder the RT visibility removes the light (blocked / off must be far below the gate)
    result["controls"] = {"occluder-between": {"blocked_over_off": blocked, "fails": max(blocked) < THIN_GATE[0] - 0.5}}
    return result


# --- ticket 13: raw transport weights of a rough transmissive sphere -------------------------------------------------

GLASS_ALPHA_ROUGHNESS = 0.2   # alpha 0.04: the refraction tail of the pre-closure estimator reaches weights near 32 (tools/f15_transport_weights.py)
ENV = 0.5                     # CorpusRig::environmentRadiance: every transport sample of an isolated sphere ends in the environment
WEIGHT_BOUND = 1.2            # analytic maximum of the radiance weight (r + t <= 1, 1.1 in the model) with 10 % head room


def sphere(scene, radius, material, slices=96, stacks=48):
    positions, normals, uvs, indices = [], [], [], []
    for j in range(stacks + 1):
        theta = math.pi * j / stacks
        for i in range(slices + 1):
            phi = 2.0 * math.pi * i / slices
            n = (math.sin(theta) * math.cos(phi), math.cos(theta), math.sin(theta) * math.sin(phi))
            positions.append(tuple(radius * c for c in n))
            normals.append(n)
            uvs.append((i / slices, j / stacks))
    for j in range(stacks):
        for i in range(slices):
            a, b = j * (slices + 1) + i, (j + 1) * (slices + 1) + i
            indices += [a, a + 1, b, a + 1, b + 1, b]  # counter-clockwise seen from outside
    scene.primitives.append((positions, normals, uvs, indices, material))


def glass_sphere_scene():
    s = GltfScene()
    glass = s.material("rough glass", color=(1, 1, 1), roughness=GLASS_ALPHA_ROUGHNESS,
                       extensions={"KHR_materials_transmission": {"transmissionFactor": 1.0}, "KHR_materials_ior": {"ior": 2.0},
                                   "KHR_materials_volume": {"thicknessFactor": 1.0}})
    sphere(s, 0.5, glass)
    return s


def run_glass_weights(args):
    """Raw 1-spp reflection transport (`--capture-linear-signal specular`, no denoiser): the pixels whose per-frame radiance exceeds ENV x WEIGHT_BOUND."""
    out = pathlib.Path(args.out)
    binary = args.control_binary if args.use_control else args.binary
    base = out / ("control" if args.use_control else "fixed")
    flags = {"--shadows": "off", "--reflections": "rt", "--lighting-denoise": "off", "--reflection-samples": "1"}
    counts, signal = [], []
    for seed in range(1, 5):
        folder = base / f"seed{seed}"
        path = glass_sphere_scene().write(folder / "scene")
        if args.evaluate:
            pfm = folder / "engine.pfm"
        else:
            pfm = capture_scene(binary, path, folder, dict(flags, **{"--lighting-seed": str(seed)}),
                                extra=["--capture-linear-signal", "specular"], accumulate=False)
        image = read_pfm(pfm)
        over = lit = low = 0
        peak = 0.0
        for y in range(image.height):
            for x in range(image.width):
                p = image.pixel(x, y)
                m = max(p)
                peak = max(peak, m)
                over += m > ENV * WEIGHT_BOUND
                low += m > 0.5 * ENV
                lit += m > 0.05
        counts.append(over)
        signal.append({"lit": lit, "peak": peak, "over_half_env": low})
    result = {"pixels_over_bound": counts, "frames": signal, "bound": ENV * WEIGHT_BOUND}
    result["gates"] = {"no-sample-above-the-analytic-bound": {"counts": counts, "pass": sum(counts) == 0 and min(f["lit"] for f in signal) > 1000}}
    # negative control: the same counter at HALF the environment must see the sphere (the statistic counts), i.e. a bound that low fails
    result["controls"] = {"bound-half-the-environment": {"counts": [f["over_half_env"] for f in signal], "fails": min(f["over_half_env"] for f in signal) > 1000}}
    return result


# --- main ------------------------------------------------------------------------------------------------------------

def verdict(result, kind):
    gates_pass = all(g["pass"] for g in result["gates"].values())
    controls_fail = True
    for control in result["controls"].values():
        rows = control.values() if isinstance(next(iter(control.values())), dict) else [control]
        controls_fail &= any(r["fails"] for r in rows)
    return gates_pass, controls_fail


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("check", choices=["volume", "masked", "thin-backlit", "glass-weights"])
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--control-binary", type=pathlib.Path, help="a build WITHOUT the fix: the gate must fail on it")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--evaluate", action="store_true", help="evaluate existing captures under --out")
    args = parser.parse_args(argv)
    args.use_control = False
    run = {"volume": run_volume, "masked": run_masked, "thin-backlit": run_thin_backlit, "glass-weights": run_glass_weights}[args.check]
    try:
        if True:
            result = run(args)
            gates_pass, controls_fail = verdict(result, args.check)
            status = 0 if gates_pass and controls_fail else 1 if not gates_pass else 3
            if args.control_binary:
                args.use_control = True
                before = run(args)
                result["control_binary"] = before
                before_pass = all(g["pass"] for g in before["gates"].values())
                result["control_binary_fails_gate"] = not before_pass
                if before_pass:
                    status = 3
    except corpus.GpuBusy:
        print("GPU busy")
        return 4
    except corpus.CaptureFailed as error:
        print(error)
        return 6
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"result-{args.check}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
