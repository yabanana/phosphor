#!/usr/bin/env python3
"""ctest wrapper: verify the unpacked CC0 terrain sets against tools/terrain_materials.json (F15.4, ticket 23).

The sets are third-party data fetched on demand (tools/fetch_terrain_materials.py, never committed).  The directory comes from
TERRAIN_MATERIALS_DIR or build/terrain-materials; without one the test SKIPS (exit 77, CTest SKIP_RETURN_CODE) instead of failing.
Stdlib only; the fetcher is run read-only (`verify`), nothing in the directory is opened as anything but bytes to hash.
"""
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def main():
    env = os.environ.get("TERRAIN_MATERIALS_DIR")
    directory = pathlib.Path(env) if env else REPO / "build" / "terrain-materials"
    if not (directory / "Grass004").is_dir():
        print(f"terrain materials not fetched at {directory}: skipped (set TERRAIN_MATERIALS_DIR or run tools/fetch_terrain_materials.py fetch)")
        return 77
    return subprocess.run([sys.executable, "-I", str(REPO / "tools" / "fetch_terrain_materials.py"), "verify", str(directory)]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
