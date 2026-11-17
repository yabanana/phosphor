#!/usr/bin/env python3
"""Wall time of the pipeline archive build (metal-tt), for the F15.2 budget "archive build time +<= 25 %".

  tools/archive_build_time.py [--build-dir build/release] [--runs 3] [--script SCRIPT]

Runs exactly the metal-tt command of the build (taken from `ninja -t commands phosphor_archive`) `--runs` times on the
build's substituted pipelines script (or on --script, a script that still contains the @PHOSPHOR_METALLIB@
placeholder: it is substituted with the build's metallib first) and prints the runs and their median as JSON.  The output
archive goes to a temporary file, never over the build's archive.  A build-time measure: it is valid on a loaded
machine when repeated, and is NOT a render performance figure.
"""
import argparse
import json
import pathlib
import shlex
import statistics
import subprocess
import sys
import tempfile
import time

REPO = pathlib.Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-dir", type=pathlib.Path, default=REPO / "build/release")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--script", type=pathlib.Path, help="pipelines script to translate (default: the build's own)")
    args = parser.parse_args(argv)
    build = args.build_dir.resolve()
    lines = subprocess.run(["ninja", "-C", str(build), "-t", "commands", "phosphor_archive"], capture_output=True, text=True, check=True).stdout.splitlines()
    tt = next(shlex.split(part.strip()) for line in lines for part in line.split("&&") if "metal-tt" in part)
    metallib = build / "shaders/phosphor.metallib"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        script = build / "shaders/pipelines.mtl4-json"
        if args.script:
            script = tmp / "pipelines.mtl4-json"
            script.write_text(args.script.read_text().replace("@PHOSPHOR_METALLIB@", str(metallib)))
        command = [str(script) if a.endswith("pipelines.mtl4-json") else a for a in tt]
        out = tmp / "archive.metallib"
        command[command.index("-o") + 1] = str(out)
        times = []
        for _ in range(args.runs):
            out.unlink(missing_ok=True)
            begin = time.monotonic()
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise SystemExit(f"metal-tt failed: {result.stderr[-600:]}")
            times.append(round(time.monotonic() - begin, 3))
        size = out.stat().st_size
    print(json.dumps({"runs_s": times, "median_s": statistics.median(times), "archive_bytes": size, "script": str(args.script or script)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
