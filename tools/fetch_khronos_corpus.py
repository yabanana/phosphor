#!/usr/bin/env python3
"""Download the owner-approved Khronos glTF-Sample-Assets subset (commit pinned, default glTF variant only) into OUT.
`--verify OUT [MANIFEST]` re-checks an existing OUT offline against the committed tools/khronos_corpus.json (sha256, sizes, licence
records, no extra file, no Draco/BasisU/WebP/Meshopt) and exits 1 on any problem.  Download usage: `fetch_khronos_corpus.py OUT`
(`OUT --screenshots` adds the screenshots).

Approved scope (owner, 2026-11-14, relayed by the parent session): KhronosGroup/glTF-Sample-Assets at
8f492390ddc7a2ae221a1369ef560589cf8c70fb, Tier A + Tier B + ABeautifulGame, default `glTF` variant only (no Draco/KTX/WebP),
into build/khronos-corpus/.  Files are untrusted data: this script only writes bytes, never executes anything it downloads.
Per model it also stores the model's own license/metadata text files (README.md, LICENSE.md, metadata.json) and writes
CORPUS.json (source commit, files, sizes, sha256, license fields)."""
import hashlib, json, pathlib, re, subprocess, sys, urllib.request

MANIFEST = pathlib.Path(__file__).resolve().parent / "khronos_corpus.json"
# Extensions that need a decoder Phosphor does not have: a model requiring one is not a valid corpus member (the default glTF variant of the
# approved models uses none of them, which --verify re-checks on every .gltf).
UNSUPPORTED_REQUIRED = {"KHR_draco_mesh_compression", "KHR_texture_basisu", "EXT_texture_webp", "EXT_meshopt_compression"}
UNSUPPORTED_USED = {"KHR_draco_mesh_compression", "KHR_texture_basisu", "EXT_texture_webp", "EXT_meshopt_compression"}

REPO = "KhronosGroup/glTF-Sample-Assets"
COMMIT = "8f492390ddc7a2ae221a1369ef560589cf8c70fb"
TIER_A = ["ClearCoatTest", "ClearCoatCarPaint", "SheenTestGrid", "AnisotropyStrengthTest", "AnisotropyRotationTest",
          "AnisotropyDiscTest", "TransmissionTest", "TransmissionRoughnessTest", "TransmissionThinwallTestGrid", "AttenuationTest",
          "IORTestGrid", "SpecularTest", "IridescenceDielectricSpheres", "IridescenceMetallicSpheres", "IridescenceSuzanne",
          "DiffuseTransmissionTest", "EmissiveStrengthTest", "TextureTransformTest", "TextureTransformMultiTest"]
TIER_B = ["AnisotropyBarnLamp", "ToyCar", "SpecularSilkPouf", "GlamVelvetSofa"]
CHESS = ["ABeautifulGame"]
META_FILES = {"README.md", "LICENSE.md", "metadata.json"}
# Owner approval (2026-11-14, second request): the Khronos reference screenshots of the same models, Models/<Name>/screenshot/,
# only files named screenshot*.jpg / screenshot*.png (no photos, .sbsar or .zip), about 9.8 MB.
SCREENSHOT = re.compile(r"^screenshot.*\.(jpg|png)$")
ALLOWED_SUFFIXES = {".gltf", ".bin", ".png", ".jpg", ".jpeg"}


def gh_list(path):
    out = subprocess.run(["gh", "api", f"repos/{REPO}/contents/{path}?ref={COMMIT}"], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def fetch(url, target):
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return len(data), hashlib.sha256(data).hexdigest()


def walk(path):
    for entry in gh_list(path):
        if entry["type"] == "dir":
            yield from walk(entry["path"])
        else:
            yield entry


def fetch_screenshots(out):
    """Add the approved screenshot images of every model of the existing manifest (sha256 recorded next to the models)."""
    manifest = out / "CORPUS.json"
    corpus = json.loads(manifest.read_text())
    total = 0
    for name, model in corpus["models"].items():
        shots = []
        for entry in gh_list(f"Models/{name}/screenshot"):
            if entry["type"] != "file" or not SCREENSHOT.match(entry["name"]):
                continue
            size, digest = fetch(entry["download_url"], out / name / "screenshot" / entry["name"])
            shots.append({"path": f"screenshot/{entry['name']}", "bytes": size, "sha256": digest})
            total += size
        model["screenshots"] = shots
        print(f"{name:32} {len(shots)} screenshots, {sum(x['bytes'] for x in shots) / 1e6:.2f} MB", flush=True)
    corpus["screenshots_approved"] = "owner, 2026-11-14 (screenshot*.jpg/png of the same models, ~9.8 MB)"
    corpus["screenshot_bytes"] = total
    manifest.write_text(json.dumps(corpus, indent=2) + "\n")
    print(f"screenshots total {total / 1e6:.1f} MB")


def file_digest(path):
    """(size, sha256) of a file, streamed."""
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def verify(out, manifest_path=MANIFEST):
    """Check an existing corpus folder against the committed manifest without any network access.

    Returns the list of problems (empty = the corpus is exactly the manifest): a missing file, a size or sha256 mismatch, a file that the
    manifest does not list (an extra file, also an unlisted model folder), a model without licence records (the three metadata files and a
    `license` field), a file kind outside .gltf/.bin/.png/.jpg/.jpeg under glTF/, or a .gltf that uses a compression or image extension
    (Draco, BasisU, WebP, Meshopt) Phosphor cannot decode."""
    out, problems = pathlib.Path(out), []
    corpus = json.loads(pathlib.Path(manifest_path).read_text())
    if not out.is_dir():
        return [f"{out}: not a directory"]
    listed = {"CORPUS.json"}
    for name, model in sorted(corpus["models"].items()):
        folder = out / name
        if not folder.is_dir():
            problems.append(f"{name}: model folder missing")
            continue
        records = {r["path"] for r in model.get("license_files", [])}
        if not ({"metadata.json", "LICENSE.md"} <= records and "README.md" in records) or not model.get("license"):
            problems.append(f"{name}: licence record incomplete in the manifest")
        entries = model.get("files", []) + model.get("license_files", []) + model.get("screenshots", [])
        for entry in entries:
            relative = pathlib.PurePosixPath(entry["path"])
            if ".." in relative.parts or relative.is_absolute():
                problems.append(f"{name}: unsafe manifest path {entry['path']}")
                continue
            listed.add(f"{name}/{relative}")
            if relative.parts[0] == "glTF" and relative.suffix.lower() not in ALLOWED_SUFFIXES:
                problems.append(f"{name}/{relative}: file kind not allowed in the corpus")
            target = folder / relative
            if not target.is_file():
                problems.append(f"{name}/{relative}: missing")
                continue
            size, digest = file_digest(target)
            if size != entry["bytes"]:
                problems.append(f"{name}/{relative}: size {size} != {entry['bytes']}")
            if digest != entry["sha256"]:
                problems.append(f"{name}/{relative}: sha256 mismatch")
            if relative.suffix.lower() == ".gltf" and target.is_file():
                try:
                    document = json.loads(target.read_text())
                except (ValueError, OSError):
                    problems.append(f"{name}/{relative}: unreadable glTF JSON")
                    continue
                for field, banned in (("extensionsRequired", UNSUPPORTED_REQUIRED), ("extensionsUsed", UNSUPPORTED_USED)):
                    for extension in document.get(field, []):
                        if extension in banned:
                            problems.append(f"{name}/{relative}: {field} lists {extension} (Draco/BasisU/WebP/Meshopt are not part of the corpus)")
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.relative_to(out).as_posix() not in listed:
            problems.append(f"{path.relative_to(out).as_posix()}: extra file not in the manifest")
    present = out / "CORPUS.json"
    if present.is_file():
        try:
            if json.loads(present.read_text()).get("commit") != corpus["commit"]:
                problems.append("CORPUS.json: commit differs from the manifest")
        except ValueError:
            problems.append("CORPUS.json: unreadable")
    return problems


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        if len(sys.argv) < 3:
            raise SystemExit("usage: fetch_khronos_corpus.py --verify OUT [MANIFEST]")
        problems = verify(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else MANIFEST)
        for problem in problems:
            print("FAIL", problem)
        count = len(json.loads(pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else MANIFEST).read_text())["models"])
        print(f"{'verify FAILED: ' + str(len(problems)) + ' problems' if problems else 'verify OK: ' + str(count) + ' models match the manifest'}")
        raise SystemExit(1 if problems else 0)
    out = pathlib.Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2 and sys.argv[2] == "--screenshots":
        return fetch_screenshots(out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = {"source": f"https://github.com/{REPO}", "commit": COMMIT, "variant": "glTF (default; no Draco/KTX/WebP)",
              "approved": "owner, 2026-11-14 (full Tier A + Tier B + ABeautifulGame, ~71 MB)", "models": {}}
    total = 0
    for tier, names in (("A", TIER_A), ("B", TIER_B), ("chess", CHESS)):
        for name in names:
            model = {"tier": tier, "files": [], "license_files": []}
            for entry in gh_list(f"Models/{name}"):
                if entry["type"] == "file" and entry["name"] in META_FILES:
                    size, digest = fetch(entry["download_url"], out / name / entry["name"])
                    model["license_files"].append({"path": entry["name"], "bytes": size, "sha256": digest})
            for entry in walk(f"Models/{name}/glTF"):
                suffix = pathlib.PurePosixPath(entry["name"]).suffix.lower()
                if suffix not in ALLOWED_SUFFIXES:
                    model.setdefault("skipped", []).append(entry["path"])
                    continue
                relative = pathlib.PurePosixPath(entry["path"]).relative_to(f"Models/{name}")
                if ".." in relative.parts:
                    raise SystemExit(f"refusing path {entry['path']}")
                size, digest = fetch(entry["download_url"], out / name / relative)
                model["files"].append({"path": str(relative), "bytes": size, "sha256": digest})
                total += size
            meta = out / name / "metadata.json"
            if meta.is_file():
                try:
                    m = json.loads(meta.read_text())
                    model["license"] = m.get("license") or m.get("legal") or m.get("licenses")
                except (ValueError, OSError):
                    model["license"] = "metadata.json unreadable: see README.md / LICENSE.md"
            corpus["models"][name] = model
            print(f"{tier:5} {name:32} {sum(f['bytes'] for f in model['files']) / 1e6:7.2f} MB, {len(model['files'])} files", flush=True)
    corpus["total_bytes"] = total
    (out / "CORPUS.json").write_text(json.dumps(corpus, indent=2) + "\n")
    print(f"total {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
