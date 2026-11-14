#!/usr/bin/env python3
"""F15.3 (ticket 22) Khronos corpus runner: renders the owner-approved glTF-Sample-Assets subset under the DECLARED Phosphor rig.

The corpus (tools/fetch_khronos_corpus.py, manifest tools/khronos_corpus.json, never committed) is untrusted data: it is only loaded as
assets by the renderer, nothing in it is executed.  Without the corpus the runner exits 2 with a message (`--static` still audits whatever
is present; tests skip when it is absent).

The rig (src/testbench/corpus_rig.h, `--corpus-rig YAW,PITCH`): one white sun of 3 W/m^2 from the upper left behind the camera, one constant
isotropic environment of radiance 0.5 (no image-based lighting), exposure 1.0 and the ACES tonemap of the post chain, a 60 degree vertical
field of view framed from the model's world bounds (+12 %), 1024x768, RT sun shadows, ReSTIR direct lighting, RT reflections and glass, the
custom denoiser, ONE settled frame (no accumulation).  The Khronos screenshots use image-based lighting and other tonemaps: a side by side is
qualitative, never a parity claim (docs/research/2026-11-14-f15-khronos-corpus-report.md).

Modes (exactly one):
  --static          audit the glTF JSON of every model (extensions, required extensions, texture transform, images): no GPU
  --run             serial captures under the GPU guard, one process per model; writes <out>/summary.json
  --validate        the same captures under API + shader validation (archive off, nothing else changes), summary-validation.json
  --harvest         one capture per model with archive misses (from <out>/summary.json) with --harvest-pipelines into <out>/harvest/
  --reclassify      CPU only: re-scan the logs of <out>/summary*.json (classification of validation notices, loader messages)
  --archive-update  CPU only: filter (constant 22 [23], --skip-present) and merge the harvests into shaders/pipelines.mtl4-json

Exit codes: 0 ok, 1 a model failed unexpectedly (neither rendered nor refused for a stated reason), 2 usage / corpus missing, 4 GPU busy,
5 internal error, 6 capture failed.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

RESOLUTION = "1024x768"
WARMUP, FRAMES = 48, 48  # warmup frames let transient and cache allocations settle: O7 counts the allocations of the measured frames
GPU_PROCESS = re.compile(r"^(phosphor|soc_bench|f\d+_spike)$")
VALIDATION_ENV = {"MTL_DEBUG_LAYER": "1", "MTL_SHADER_VALIDATION": "1", "MTL_DEBUG_LAYER_WARNING_MODE": "nslog"}

# Declared camera per model: (yaw, pitch) in degrees around the bounds, yaw 0 = from +Z (the glTF front).  Test grids are authored as
# front-facing panels; the showcase models use a three-quarter view that shows their relief; a very wide model gets a wider frame
# (`resolution`, default 1024x768).  Models, tier and feature statement
# (what the model exists to show); the notes (what Phosphor renders) are in tools/f15_corpus_notes.json.
MODELS = {
    "ClearCoatTest": {"tier": "A", "view": (0, 0), "shows": "clearcoat factor, roughness, coat normal, partial coat (labelled grid; base / coated / coat-only columns)"},
    "ClearCoatCarPaint": {"tier": "A", "view": (0, 0), "shows": "clear coat over a normal-mapped car-paint base; KHR_texture_transform required"},
    "SheenTestGrid": {"tier": "A", "view": (0, 0), "shows": "sheenColor x sheenRoughness grid of 16 spheres over a checker; KHR_materials_sheen required"},
    "AnisotropyStrengthTest": {"tier": "A", "view": (0, 0), "shows": "anisotropyStrength sweep"},
    "AnisotropyRotationTest": {"tier": "A", "view": (0, 0), "shows": "anisotropyRotation offsets (README: needs image-based lighting to show the stretching)"},
    "AnisotropyDiscTest": {"tier": "A", "view": (0, 0), "shows": "anisotropy direction texture (disc), strength and rotation"},
    "TransmissionTest": {"tier": "A", "view": (0, 0), "shows": "transmission factor / texture / roughness over a backdrop (thin-walled panes)"},
    "TransmissionRoughnessTest": {"tier": "A", "view": (0, 0), "shows": "roughness x IOR tiles of volume slabs over a pattern"},
    "TransmissionThinwallTestGrid": {"tier": "A", "view": (0, 0), "shows": "the same IOR with and without volume (thin wall vs volumetric)"},
    "AttenuationTest": {"tier": "A", "view": (0, 0), "shows": "attenuationColor x attenuationDistance x thickness"},
    "IORTestGrid": {"tier": "A", "view": (0, 0), "shows": "KHR_materials_ior and specular on transmissive spheres"},
    "SpecularTest": {"tier": "A", "view": (0, 0), "shows": "specularFactor and specularColorFactor, with textures"},
    "IridescenceDielectricSpheres": {"tier": "A", "view": (30, 20), "shows": "iridescence over dielectrics, thickness and film IOR sweep"},
    "IridescenceMetallicSpheres": {"tier": "A", "view": (30, 20), "shows": "iridescence over metals, thickness and film IOR sweep"},
    "IridescenceSuzanne": {"tier": "A", "view": (0, 0), "resolution": "1600x640", "shows": "iridescence + transmission + volume + IOR on one material; KHR_lights_punctual used"},
    "DiffuseTransmissionTest": {"tier": "A", "view": (0, 0), "shows": "diffuse transmission; REQUIRES KHR_materials_unlit and KHR_lights_punctual"},
    "EmissiveStrengthTest": {"tier": "A", "view": (0, 0), "shows": "emissiveStrength above 1"},
    "TextureTransformTest": {"tier": "A", "view": (0, 0), "shows": "KHR_texture_transform offset / rotation / scale on the base colour"},
    "TextureTransformMultiTest": {"tier": "A", "view": (0, 0), "shows": "transform on several textures incl. texCoord override, clearcoat and unlit; README: rows of identical checkmarks"},
    "AnisotropyBarnLamp": {"tier": "B", "view": (20, 8), "shows": "anisotropy + clearcoat on one material, plus transmission/volume and emissive strength"},
    "ToyCar": {"tier": "B", "view": (35, 12), "shows": "clearcoat, sheen, transmission on separate materials, KHR_texture_transform"},
    "SpecularSilkPouf": {"tier": "B", "view": (0, 15), "shows": "specular + sheen on one material (required)"},
    "GlamVelvetSofa": {"tier": "B", "view": (20, 10), "shows": "specular + sheen, KHR_texture_transform required, KHR_materials_variants / lights_punctual used"},
    "ABeautifulGame": {"tier": "chess", "view": (0, 32), "shows": "transmission + volume on a showcase scene (Track Studio case, ticket 20)"},
}


class GpuBusy(RuntimeError):
    """Another GPU job runs or the verification lock is held."""


class CaptureFailed(RuntimeError):
    """The renderer process itself failed in a way no model-level reason explains."""


# --- guard --------------------------------------------------------------------

@contextmanager
def gpu_lock(path):
    import fcntl
    with pathlib.Path(path).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise GpuBusy("another cooperating Phosphor GPU verification owns the lock")
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def running_gpu_processes(ps_output=None):
    """`pid:name` of other GPU jobs (`phosphor`, `soc_bench`, `f<N>_spike`) from `ps -axo pid=,comm=`."""
    if ps_output is None:
        ps_output = subprocess.run(["ps", "-axo", "pid=,comm="], capture_output=True, text=True).stdout
    found = []
    for line in ps_output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0] != str(os.getpid()) and GPU_PROCESS.match(pathlib.Path(parts[1]).name):
            found.append(f"{parts[0]}:{pathlib.Path(parts[1]).name}")
    return found


# --- static audit -------------------------------------------------------------

def gltf_path(corpus, name):
    candidates = sorted((corpus / name / "glTF").glob("*.gltf"))
    return candidates[0] if candidates else None


def audit_model(corpus, name):
    """Extensions and structure of one model straight from its glTF JSON (no renderer)."""
    path = gltf_path(corpus, name)
    if path is None:
        return {"present": False}
    document = json.loads(path.read_text())
    per_material = {}
    texture_transform_uses = 0
    for material in document.get("materials", []):
        for extension, body in material.get("extensions", {}).items():
            per_material[extension] = per_material.get(extension, 0) + 1
            if isinstance(body, dict):
                texture_transform_uses += sum(1 for v in body.values() if isinstance(v, dict) and "KHR_texture_transform" in v.get("extensions", {}))
        for value in material.values():
            if isinstance(value, dict) and "KHR_texture_transform" in value.get("extensions", {}):
                texture_transform_uses += 1
        pbr = material.get("pbrMetallicRoughness", {})
        for value in pbr.values():
            if isinstance(value, dict) and "KHR_texture_transform" in value.get("extensions", {}):
                texture_transform_uses += 1
    return {"present": True, "file": str(path.relative_to(corpus)), "extensions_used": document.get("extensionsUsed", []),
            "extensions_required": document.get("extensionsRequired", []), "material_extensions": per_material,
            "materials": len(document.get("materials", [])), "meshes": len(document.get("meshes", [])), "nodes": len(document.get("nodes", [])),
            "images": [image.get("uri", "<bufferView>").rsplit(".", 1)[-1].lower() for image in document.get("images", [])],
            "texture_transform_uses": texture_transform_uses, "cameras": len(document.get("cameras", [])),
            "animations": len(document.get("animations", [])), "scenes": len(document.get("scenes", []))}


# --- commands -----------------------------------------------------------------

def capture_command(binary, corpus, name, folder, validation=False, harvest=None):
    view = MODELS[name]["view"]
    command = [str(binary), "--bench", "4", "--scene", str(gltf_path(corpus, name)), "--corpus-rig", f"{view[0]},{view[1]}",
               "--render-path", "visibility", "--geometry-path", "mesh", "--rt", "on", "--shadows", "rt", "--lighting", "restir",
               "--gi", "off", "--reflections", "rt", "--ao", "off", "--lighting-denoise", "custom",
               "--post", "--tonemap", "aces", "--upscaler", "native", "--resolution", MODELS[name].get("resolution", RESOLUTION),
               "--warmup", str(WARMUP), "--frames", str(FRAMES), "--fixed-timestep", "--no-ui", "--no-vsync", "--no-gpu-timing",
               # Synchronous pipelines: a cold compiler cache must never let a capture use a fallback variant.
               "--pipeline-sync",
               "--capture", str(folder / "engine.png"), "--report", str(folder / "report.json")]
    if validation:
        command.append("--no-pipeline-archive")  # Metal rejects archive lookups under MTL_SHADER_VALIDATION
    if harvest:
        command += ["--harvest-pipelines", str(harvest)]
    return command


PIPELINES_LINE = re.compile(r"PIPELINES requests (\d+) \| archive hits (\d+), misses (\d+), unavailable (\d+).*?render-thread compiles (\d+).*?"
                            r"fallbacks (\d+) \(draws (\d+)\) \| failures (\d+)")
RIG_LINE = re.compile(r"CORPUS_RIG: (\d+) instances, bounds min \(([-\d. e+]+)\) max \(([-\d. e+]+)\), camera distance ([\d.e+-]+)")
MISS_LINE = re.compile(r"Pipeline archive miss: ([^\n(]+) \(")
VALIDATION_FAILURE = re.compile(r"failed assertion|Shader Validation Error|Metal Validation Error|MTLDebug|validation (error|failure)|GPU timeout|command buffers failed", re.I)
SAFE_LINES = re.compile(r"^\[INFO\]|^BENCH|^SCENE|^MESHLET|^PIPELINES|^RT |^LIGHTING|^EXIT \d+$|^STARTUP|^REFLECTION|^SHADOW|^GI |^DI |Validation Enabled|^$")


KNOWN_NOTICE = re.compile(r"Render Command Encoder Set (Depth Stencil State|Viewport|Render Pipeline State) Validation|^Redundant call to set(DepthStencilState|Viewport|RenderPipelineState)")
LOADER_LINE = re.compile(r"^\[(WARN|ERROR)\][^\n]*(gltf_loader|main\.cpp)")
NSLOG_PREFIX = re.compile(r"^\d{4}-\d\d-\d\d [\d:.]+ phosphor\[[\d:]+\] ")


def scan_log(text):
    """Classify the non-INFO lines of a capture log: validation failures, the KNOWN baseline notices (redundant state calls of the validation
    layer: present with the phase-head binary on every scene, ticket 15 record), loader warnings / refusal errors, and anything else."""
    lines = text.splitlines()
    known, loader, other = {}, [], []
    for line in lines:
        bare = NSLOG_PREFIX.sub("", line)
        if SAFE_LINES.search(line) and not LOADER_LINE.search(line):
            continue
        if KNOWN_NOTICE.search(bare):
            key = re.sub(r"\s+Validation\.?$", "", bare.strip())
            known[key] = known.get(key, 0) + 1
        elif LOADER_LINE.search(line):
            loader.append(re.sub(r"^\[[A-Z]+\] \[[^\]]*\] [^ ]+:\d+: ", "", line))
        else:
            other.append(line)
    return {"other_messages": other[:20], "known_notices": known, "loader_messages": loader[:20],
            "validation_failures": [line.strip() for line in lines if VALIDATION_FAILURE.search(line)][:10]}


def fallback_draws(text):
    found = PIPELINES_LINE.findall(text)
    return int(found[-1][6]) if found else None


def refusal_reason(text):
    """The loader's stated reason when the scene did not load (None when it loaded): the `refusing to load` line, else the scene viewer's
    `Required scene failed to load` line."""
    for marker in ("refusing to load", "Required scene failed to load", "failed to load"):
        for line in text.splitlines():
            if marker in line and "[ERROR]" in line or marker in line and "[WARN" in line:
                return re.sub(r"^\[[A-Z]+\] \[[^\]]*\] [^ ]+:\d+: ", "", line).strip()
    return None


def loader_error_lines(text):
    return [line.split("] ", 2)[-1].strip() for line in text.splitlines() if "[ERROR]" in line or "[WARN" in line][:12]


def summarize_materials(materials):
    """Aggregate the report's per-material support records into (feature -> support -> approximation names)."""
    if not materials:
        return None
    features = {}
    for material in materials.get("materials", []):
        for record in material.get("features", []) + material.get("encodings", []):
            entry = features.setdefault(record["feature"], {"support": record["support"], "approximation": record.get("approximation", ""), "materials": 0, "label": record.get("label", "")})
            entry["materials"] += 1
    warnings = sorted({w for material in materials.get("materials", []) for w in material.get("warnings", [])})
    unknown = sorted({u for material in materials.get("materials", []) for u in material.get("unknown_extensions", [])} | set(materials.get("asset_unknown_extensions", [])))
    return {"count": materials.get("count"), "features": features, "warnings": warnings, "unknown_extensions": unknown,
            "base": materials.get("base"), "extensions_totals": materials.get("features")}


def run_one(binary, corpus, name, folder, timeout, validation=False, harvest=None):
    """One renderer process; returns the model's status record (never raises for a model-level failure)."""
    folder.mkdir(parents=True, exist_ok=True)
    command = capture_command(binary, corpus, name, folder, validation, harvest)
    env = dict(os.environ)
    if validation:
        env.update(VALIDATION_ENV)
    else:
        for key in VALIDATION_ENV:
            env.pop(key, None)
    log = folder / "engine.log"
    started = time.monotonic()
    timed_out = False
    with log.open("w") as output:
        try:
            code = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, env=env).returncode
        except subprocess.TimeoutExpired:
            code, timed_out = None, True
    text = log.read_text(errors="replace")
    markers = re.findall(r"^EXIT (\d+)$", text, re.M)
    exit_marker = int(markers[-1]) if markers else None
    record = {"model": name, "tier": MODELS[name]["tier"], "view": list(MODELS[name]["view"]), "exit_code": code, "exit_marker": exit_marker,
              "resolution": MODELS[name].get("resolution", RESOLUTION), "timed_out": timed_out, "seconds": round(time.monotonic() - started, 1), "validation": validation,
              "command": [c for c in command]}
    pipelines = PIPELINES_LINE.findall(text)
    if pipelines:
        requests, hits, misses, unavailable, render_thread, fallbacks, draws, failures = (int(v) for v in pipelines[-1])
        record["pipelines"] = {"requests": requests, "archive_hits": hits, "archive_misses": misses, "render_thread_compiles": render_thread,
                               "fallbacks": fallbacks, "fallback_draws": draws, "failures": failures}
    record["archive_miss_pipelines"] = sorted(set(MISS_LINE.findall(text)))
    rig = RIG_LINE.search(text)
    if rig:
        record["rig"] = {"instances": int(rig.group(1)), "bounds_min": [float(v) for v in rig.group(2).split()],
                         "bounds_max": [float(v) for v in rig.group(3).split()], "distance": float(rig.group(4))}
    report_path = folder / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        record["gpu_allocations_measured"] = report.get("gpu_allocations")
        record["materials"] = summarize_materials(report.get("materials"))
        record["frames"] = report.get("frames")
        record["scene"] = {k: report.get("scene", {}).get(k) for k in ("instances", "materials", "buckets", "queue_overflow")}
    record.update(scan_log(text))
    record["capture"] = str((folder / "engine.png")) if (folder / "engine.png").is_file() else None
    reason = refusal_reason(text)
    if timed_out:
        record["status"], record["reason"] = "failed", "process timeout"
    elif reason and code != 0:
        record["status"], record["reason"] = "refused", reason
    elif code == 0 and exit_marker == 0 and record["capture"]:
        record["status"], record["reason"] = "rendered", None
    else:
        record["status"], record["reason"] = "failed", f"exit {code}, EXIT marker {exit_marker}; see {log.name}"
    if record["status"] == "rendered":
        problems = []
        if record.get("pipelines", {}).get("fallback_draws", 1) != 0 and not validation:
            problems.append(f"fallback draws {record.get('pipelines', {}).get('fallback_draws')}")
        if record["validation_failures"]:
            problems.append("validation failures in the log")
        if any(line.startswith("[ERROR]") for line in record["other_messages"]):
            problems.append("[ERROR] lines in the log")
        if problems:
            record["status"], record["reason"] = "failed", "; ".join(problems)
    return record


def run_models(args, names, folder_name, validation=False, harvest=False):
    corpus, out, binary = args.corpus.resolve(), args.out.resolve(), args.binary.resolve()
    if not binary.is_file():
        raise SystemExit("--run needs an existing compiled renderer; this runner never builds")
    base = out / folder_name
    for name in names:
        if (base / name).exists() and any((base / name).iterdir()):
            raise SystemExit(f"{base / name} is not empty: choose a new --out")
    records = []
    for position, name in enumerate(names):
        harvest_path = (out / "harvest" / f"{name}.mtl4-json") if harvest else None
        if harvest_path:
            harvest_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 60 * args.wait_busy_minutes
        while True:  # the guard runs before EVERY capture; other cooperating GPU jobs are waited for, never killed
            try:
                with gpu_lock(args.gpu_lock):
                    busy = running_gpu_processes()
                    if busy:
                        raise GpuBusy(f"refusing to start {name}: other GPU jobs are running: {', '.join(busy)}")
                    record = run_one(binary, corpus, name, base / name, args.timeout, validation, harvest_path)
                break
            except GpuBusy:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(15)
        records.append(record)
        extra = f" | misses {record['pipelines']['archive_misses']}" if "pipelines" in record else ""
        print(f"{name:32} {record['status']:8} {record.get('reason') or ''}{extra}", flush=True)
        if position + 1 < len(names):
            time.sleep(2)  # serial and quiet between GPU jobs
    return records


def write_summary(path, records, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((REPO / "tools" / "khronos_corpus.json").read_text())
    binary = args.binary.resolve()
    document = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "corpus_commit": manifest["commit"],
                "rig": {"resolution": RESOLUTION + " (1600x640 for IridescenceSuzanne)", "warmup": WARMUP, "frames": FRAMES, "sun_irradiance": 3.0, "environment_radiance": 0.5,
                        "exposure": 1.0, "tonemap": "aces", "fov_y_degrees": 60, "settled_single_frame": True},
                "binary": str(binary), "binary_sha256": sha256(binary), "models": records}
    path.write_text(json.dumps(document, indent=2) + "\n")


def sha256(path):
    from f15_common import sha256_file
    return sha256_file(path)


# --- archive update (CPU only) ------------------------------------------------

def archive_update(args):
    """Filter every harvest (constants 22 and 23, pipelines the archive lacks) and merge them into shaders/pipelines.mtl4-json."""
    out = args.out.resolve()
    archive = REPO / "shaders" / "pipelines.mtl4-json"
    raws = sorted((out / "harvest").glob("*.mtl4-json"))
    if not raws:
        raise SystemExit(f"no harvests in {out / 'harvest'}")
    kept = []
    for raw in raws:
        normalised = raw.with_suffix(".norm.json")
        text = subprocess.run(["jq", "-S", '.libraries |= map(.path = "@PHOSPHOR_METALLIB@")', str(raw)], capture_output=True, text=True, check=True).stdout
        normalised.write_text(text)
        for constant in args.constant:
            target = raw.with_suffix(f".c{constant}.json")
            subprocess.run([sys.executable, "-I", str(REPO / "tools/filter_pipeline_harvest.py"), str(normalised), "--constant", str(constant),
                            "--skip-present", str(archive), "--output", str(target)], check=True)
            kept.append(str(target))
    merged = out / "harvest" / "merged.json"
    subprocess.run([sys.executable, "-I", str(REPO / "tools/merge_pipeline_harvests.py"), "--output", str(merged), str(archive), *kept], check=True)
    sorted_text = subprocess.run(["jq", "-S", ".", str(merged)], capture_output=True, text=True, check=True).stdout
    archive.write_text(sorted_text)
    print(f"updated {archive}")


def reclassify(args):
    """CPU only: re-scan the logs of an existing summary (after a change of the log classification) and rewrite it."""
    for name in ("summary.json", "summary-validation.json"):
        path = args.out / name
        if not path.is_file():
            continue
        document = json.loads(path.read_text())
        folder = "validation" if "validation" in name else "captures"
        for record in document["models"]:
            log = args.out / folder / record["model"] / "engine.log"
            record.pop("loader_messages", None)
            record.update(scan_log(log.read_text(errors="replace")))
        path.write_text(json.dumps(document, indent=2) + "\n")
        print(f"reclassified {path}")


# --- main ---------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=pathlib.Path, default=REPO / "build/khronos-corpus")
    parser.add_argument("--binary", type=pathlib.Path, default=REPO / "build/release/phosphor")
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "build/f15-corpus")
    parser.add_argument("--only", action="append", default=[], choices=sorted(MODELS))
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("static", "run", "validate", "harvest", "archive-update", "reclassify"):
        mode.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--constant", type=int, action="append", default=None)
    parser.add_argument("--gpu-lock", type=pathlib.Path, default=pathlib.Path("/tmp/phosphor-gpu-verification.lock"))
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--wait-busy-minutes", type=float, default=0, help="wait this long for other GPU jobs before each capture (default 0: exit 4 at once)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.constant is None:
        args.constant = [22, 23]
    names = args.only or list(MODELS)
    try:
        if args.archive_update:
            archive_update(args)
            return 0
        if args.reclassify:
            reclassify(args)
            return 0
        if not args.corpus.is_dir():
            print(f"corpus folder {args.corpus} is absent: fetch it with tools/fetch_khronos_corpus.py (owner-approved set only)", file=sys.stderr)
            return 2
        if args.static:
            audits = {name: audit_model(args.corpus, name) for name in names}
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "static-audit.json").write_text(json.dumps(audits, indent=2) + "\n")
            for name, audit in audits.items():
                print(f"{name:32} {'missing' if not audit['present'] else ','.join(e.replace('KHR_materials_', '') for e in audit['extensions_required'])}")
            return 0
        if args.harvest:
            summary = json.loads((args.out / "summary.json").read_text())
            names = [m["model"] for m in summary["models"] if m.get("pipelines", {}).get("archive_misses", 0) > 0 and (not args.only or m["model"] in args.only)]
            records = run_models(args, names, "harvest-runs", harvest=True)
            return 0 if all(r["status"] in ("rendered", "refused") for r in records) else 6
        folder, summary_name = ("validation", "summary-validation.json") if args.validate else ("captures", "summary.json")
        records = run_models(args, names, folder, validation=args.validate)
        write_summary(args.out.resolve() / summary_name, records, args)
        return 0 if all(r["status"] in ("rendered", "refused") for r in records) else 1
    except GpuBusy as error:
        print(f"GPU busy: {error}", file=sys.stderr)
        return 4
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - exit code 5 is the contract
        traceback.print_exc()
        return 5


if __name__ == "__main__":
    sys.exit(main())
