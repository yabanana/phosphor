#!/usr/bin/env python3
"""Download the owner-approved Khronos glTF-Sample-Assets subset (commit pinned, default glTF variant only) into OUT.

Approved scope (owner, 2026-11-14, relayed by the parent session): KhronosGroup/glTF-Sample-Assets at
8f492390ddc7a2ae221a1369ef560589cf8c70fb, Tier A + Tier B + ABeautifulGame, default `glTF` variant only (no Draco/KTX/WebP),
into build/khronos-corpus/.  Files are untrusted data: this script only writes bytes, never executes anything it downloads.
Per model it also stores the model's own license/metadata text files (README.md, LICENSE.md, metadata.json) and writes
CORPUS.json (source commit, files, sizes, sha256, license fields)."""
import hashlib, json, pathlib, subprocess, sys, urllib.request

REPO = "KhronosGroup/glTF-Sample-Assets"
COMMIT = "8f492390ddc7a2ae221a1369ef560589cf8c70fb"
TIER_A = ["ClearCoatTest", "ClearCoatCarPaint", "SheenTestGrid", "AnisotropyStrengthTest", "AnisotropyRotationTest",
          "AnisotropyDiscTest", "TransmissionTest", "TransmissionRoughnessTest", "TransmissionThinwallTestGrid", "AttenuationTest",
          "IORTestGrid", "SpecularTest", "IridescenceDielectricSpheres", "IridescenceMetallicSpheres", "IridescenceSuzanne",
          "DiffuseTransmissionTest", "EmissiveStrengthTest", "TextureTransformTest", "TextureTransformMultiTest"]
TIER_B = ["AnisotropyBarnLamp", "ToyCar", "SpecularSilkPouf", "GlamVelvetSofa"]
CHESS = ["ABeautifulGame"]
META_FILES = {"README.md", "LICENSE.md", "metadata.json"}
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


def main():
    out = pathlib.Path(sys.argv[1]).resolve()
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
