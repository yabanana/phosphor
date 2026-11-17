#!/usr/bin/env python3
"""F15.4 (ticket 23): fetch the four owner-approved CC0 terrain texture sets and record them.

Approved scope (owner, 2026-11-14, relayed by the parent session): ambientcg.com only, the 1K JPG zips of Grass004, Rock030,
Ground037 and Snow006 (about 38.9 MB), into build/terrain-materials/ (gitignored), unpacked as DATA only.  Every zip entry is
checked to stay inside its target directory (absolute paths, `..` segments, links and anything but regular files are refused);
nothing downloaded is ever executed.  The manifest records each zip's and each file's sha256 and the CC0-1.0 licence.

  python3 -I tools/fetch_terrain_materials.py fetch build/terrain-materials      # download, unpack, write MANIFEST.json
  python3 -I tools/fetch_terrain_materials.py verify build/terrain-materials     # offline: compare with tools/terrain_materials.json
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import stat
import sys
import urllib.request
import zipfile

SOURCE = "https://ambientcg.com/get?file={}"
ASSETS = ["Grass004", "Rock030", "Ground037", "Snow006"]
VARIANT = "1K-JPG"
LICENSE = {"license": "CC0-1.0", "licenseUrl": "https://creativecommons.org/publicdomain/zero/1.0/legalcode", "owner": "ambientCG (Lennart Demes)"}
COMMITTED = pathlib.Path(__file__).resolve().parent / "terrain_materials.json"
MAX_ZIP_BYTES = 64 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_members(archive: zipfile.ZipFile, target: pathlib.Path):
    """Yield (info, destination) for regular files only; refuse any entry that would leave `target`."""
    root = target.resolve()
    for info in archive.infolist():
        if info.is_dir():
            continue
        mode = (info.external_attr >> 16) & 0xFFFF
        if mode and not stat.S_ISREG(mode):
            raise SystemExit(f"refusing non-regular zip entry {info.filename!r}")
        name = pathlib.PurePosixPath(info.filename)
        if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
            raise SystemExit(f"refusing zip entry {info.filename!r}: escapes the target directory")
        destination = (root / pathlib.Path(*name.parts)).resolve()
        if root not in destination.parents:
            raise SystemExit(f"refusing zip entry {info.filename!r}: escapes the target directory")
        yield info, destination


def fetch(out: pathlib.Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "https://ambientcg.com", "variant": VARIANT, **LICENSE,
                "approved": "owner, 2026-11-14 (four CC0 1K JPG sets, ~38.9 MB)", "assets": {}}
    for asset in ASSETS:
        name = f"{asset}_{VARIANT}.zip"
        request = urllib.request.Request(SOURCE.format(name), headers={"User-Agent": "phosphor-fetch"})
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read(MAX_ZIP_BYTES + 1)
        if len(data) > MAX_ZIP_BYTES:
            raise SystemExit(f"{name}: larger than {MAX_ZIP_BYTES} bytes, refused")
        (out / name).write_bytes(data)
        target = out / asset
        files = []
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for info, destination in safe_members(archive, target):
                content = archive.read(info)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                files.append({"path": destination.relative_to(target.resolve()).as_posix(), "bytes": len(content), "sha256": sha256(content)})
        manifest["assets"][asset] = {"zip": name, "zip_bytes": len(data), "zip_sha256": sha256(data), "files": sorted(files, key=lambda f: f["path"])}
        print(f"{asset}: {len(data) / 1e6:.2f} MB, {len(files)} files", flush=True)
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


def verify(out: pathlib.Path, committed: pathlib.Path = COMMITTED) -> int:
    """Offline check of an unpacked set against the committed manifest (sizes, sha256, no missing or extra file)."""
    expected = json.loads(committed.read_text())
    problems = []
    for asset, record in expected["assets"].items():
        zip_path = out / record["zip"]
        if not zip_path.is_file() or sha256(zip_path.read_bytes()) != record["zip_sha256"]:
            problems.append(f"{record['zip']}: missing or different")
        target = out / asset
        listed = {f["path"]: f for f in record["files"]}
        present = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()} if target.is_dir() else set()
        for path in sorted(present - set(listed)):
            problems.append(f"{asset}/{path}: not in the manifest")
        for path, f in listed.items():
            p = target / path
            if not p.is_file():
                problems.append(f"{asset}/{path}: missing")
            elif p.stat().st_size != f["bytes"] or sha256(p.read_bytes()) != f["sha256"]:
                problems.append(f"{asset}/{path}: size or sha256 differs")
    if expected.get("license") != "CC0-1.0":
        problems.append("manifest licence is not CC0-1.0")
    for problem in problems:
        print(problem, file=sys.stderr)
    print("terrain materials: OK" if not problems else f"terrain materials: {len(problems)} problem(s)")
    return 1 if problems else 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in ("fetch", "verify"):
        print(__doc__, file=sys.stderr)
        return 2
    out = pathlib.Path(argv[1]).resolve()
    return fetch(out) if argv[0] == "fetch" else verify(out)


if __name__ == "__main__":
    raise SystemExit(main())
