#!/usr/bin/env python3
"""F15.2 (ticket 21): export the feature-class manifests of the chess rigs and of the Khronos corpus models with the app
(`phosphor --export-class-manifest`), one GPU process per scene, serially under the shared verification lock.

  tools/f15_export_manifests.py --corpus DIR [--out shaders/material_classes] [--only NAME ...]

Writes <out>/chess.json (scenes "chess:chess-backlight|chess-sidelight|chess-shadow", the pinned ABeautifulGame with
fixtures/chess/chess-subsurface.overrides.json) and <out>/khronos-corpus.json (scenes "corpus:<Model>", the models of
tools/f15_corpus.py with their declared rig).  The corpus is owner-approved
read-only data: it is only loaded, never copied or executed.  A model that fails to load or render is listed and skipped.
Exit codes: 0 ok, 3 GPU busy, 6 a run failed.
"""
import argparse
import fcntl
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import f15_corpus as corpus_tool  # noqa: E402
import f15_check as fc  # noqa: E402

LOCK = pathlib.Path("/tmp/phosphor-gpu-verification.lock")
OVERRIDES = REPO / "fixtures/chess/chess-subsurface.overrides.json"
CHESS_SCENARIOS = ["chess-backlight", "chess-sidelight", "chess-shadow"]
COMMON = ["--render-path", "visibility", "--geometry-path", "mesh", "--rt", "off", "--reflections", "off", "--lighting-denoise", "custom", "--post", "--upscaler", "native", "--frames", "2", "--warmup", "0", "--fixed-timestep", "--no-ui", "--no-vsync",
          "--no-gpu-timing", "--pipeline-sync"]


def scene_command(binary, family, name, corpus, frames=2, warmup=0):
    """The app command of a chess rig ("chess", name = rig scenario) or of a corpus model ("corpus", name = model), without any capture
    or export flag.  Shared with tools/f15_permutations.py (harvest, equivalence and budget runs of these scenes)."""
    if family == "chess":
        gltf = corpus_tool.gltf_path(corpus, "ABeautifulGame")
        return [str(binary), "--scene", str(gltf), "--material-overrides", str(OVERRIDES), "--bench", "2", "--material-scenario", name,
                *COMMON[:COMMON.index("--frames")], "--frames", str(frames), "--warmup", str(warmup), *COMMON[COMMON.index("--fixed-timestep"):]]
    command = corpus_tool.capture_command(binary, corpus, name, pathlib.Path("/unused"))  # the corpus runner's own rig and lighting flags
    for flag, value in (("--warmup", str(warmup)), ("--frames", str(frames))):
        command[command.index(flag) + 1] = value
    for flag in ("--capture", "--report"):
        del command[command.index(flag):command.index(flag) + 2]
    return command


def run(command, log):
    busy = fc.running_gpu_processes()
    if busy:
        raise RuntimeError("busy: " + ", ".join(busy))
    with open(log, "wb") as out:
        return subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, timeout=600).returncode


def export_scene(binary, command, tmp):
    target = tmp / "scene.json"
    target.unlink(missing_ok=True)
    code = run(command + ["--export-class-manifest", str(target)], tmp / "scene.log")
    if code != 0 or not target.exists():
        return None
    return json.loads(target.read_text())["scenes"][0]


def write_manifest(path, source, scenes):
    scenes = sorted(scenes, key=lambda s: s["name"])
    path.write_text(json.dumps({"schema": "phosphor.material-class-manifest", "version": 1, "source": source, "scenes": scenes},
                               indent=2, sort_keys=True) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=REPO / "shaders/material_classes")
    parser.add_argument("--binary", type=pathlib.Path, default=REPO / "build/release/phosphor")
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args(argv)
    binary = args.binary.resolve()
    tmp = pathlib.Path("/tmp") / f"f15-export-manifests-{__import__('os').getpid()}"
    tmp.mkdir(parents=True)
    skipped, chess, corpus = [], [], []
    with LOCK.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("GPU busy: the verification lock is held", file=sys.stderr)
            return 3
        for scenario in CHESS_SCENARIOS:
            if args.only and scenario not in args.only:
                continue
            command = scene_command(binary, "chess", scenario, args.corpus)
            scene = export_scene(binary, command, tmp)
            if scene is None:
                print(f"FAILED chess {scenario}", file=sys.stderr)
                return 6
            scene.update(name="chess:" + scenario, family="chess")
            chess.append(scene)
            print(f"chess:{scenario}: classes {[c['mask'] for c in scene['classes']]} union {scene['union']}", flush=True)
        for name, info in corpus_tool.MODELS.items():
            if args.only and name not in args.only:
                continue
            path = corpus_tool.gltf_path(args.corpus, name)
            if path is None:
                skipped.append((name, "not present"))
                continue
            command = scene_command(binary, "corpus", name, args.corpus)
            scene = export_scene(binary, command, tmp)
            if scene is None:
                skipped.append((name, "the app did not render it (see log)"))
                print(f"SKIPPED {name}", flush=True)
                continue
            scene.update(name="corpus:" + name, family="corpus")
            corpus.append(scene)
            print(f"corpus:{name}: classes {[c['mask'] for c in scene['classes']]} union {scene['union']}", flush=True)
    if chess:
        write_manifest(args.out / "chess.json", "phosphor --export-class-manifest (ABeautifulGame, chess rigs, tools/f15_export_manifests.py)", chess)
    if corpus:
        write_manifest(args.out / "khronos-corpus.json", "phosphor --export-class-manifest (Khronos corpus models, declared rigs, tools/f15_export_manifests.py)", corpus)
    print(json.dumps({"chess": len(chess), "corpus": len(corpus), "skipped": skipped}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
