#!/usr/bin/env python3
"""Fetch the pinned Sponza fixture, or verify its complete local contents.

Run through mise: mise exec -- python3 tools/fetch_sponza.py [--verify].
The manifest records source, revision, byte sizes and SHA-256 for every file.
Downloaded third-party content remains outside Git; attribution stays local.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'edc7c9e67c639d230715049ee31f9a96a6babbbe'
BASE = f'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/{REVISION}/'
MANIFEST = ROOT / 'assets/manifests/sponza.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(name):
    p = PurePosixPath(urllib.parse.unquote(name))
    if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name:
        raise ValueError(f'Unsafe fixture path: {name}')
    return str(p)


def download(source):
    request = urllib.request.Request(BASE + urllib.parse.quote(source, safe='/'),
                                     headers={'User-Agent': 'Phosphor-fixture-fetch/1'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'assets/sponza')
    args = parser.parse_args()
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        if manifest['revision'] != REVISION:
            raise ValueError('Fixture revision does not match the fetcher')
        records = manifest['files']
    else:
        if args.verify:
            raise FileNotFoundError(MANIFEST)
        data = download('Models/Sponza/glTF/Sponza.gltf')
        model = json.loads(data)
        names = sorted({safe_path(item['uri']) for section in ('buffers', 'images')
                        for item in model.get(section, []) if 'uri' in item})
        records = [{'path': name, 'source': 'Models/Sponza/glTF/' + name}
                   for name in ['Sponza.gltf'] + names]
        records += [{'path': 'SOURCE-README.md', 'source': 'Models/Sponza/README.md'},
                    {'path': 'LICENSE.txt', 'source': 'LICENSES/LicenseRef-CRYENGINE-Agreement.txt'}]
        manifest = {'schema': 1, 'repository': 'https://github.com/KhronosGroup/glTF-Sample-Assets',
                    'revision': REVISION, 'model': 'Sponza',
                    'license': 'LicenseRef-CRYENGINE-Agreement (as declared by upstream)',
                    'attribution': 'Crytek; see upstream model README for credits and licensing notes',
                    'files': records}
    total = 0
    for entry in records:
        path = args.output / safe_path(entry['path'])
        content = path.read_bytes() if path.exists() else None
        valid = content is not None and ('sha256' not in entry or digest(content) == entry['sha256'])
        if not valid:
            if args.verify:
                raise ValueError(f'Missing or damaged fixture: {path}')
            content = download(entry['source'])
        if 'sha256' in entry and digest(content) != entry['sha256']:
            raise ValueError(f'Unexpected source checksum: {entry["source"]}')
        if 'bytes' in entry and len(content) != entry['bytes']:
            raise ValueError(f'Unexpected size: {entry["source"]}')
        if not args.verify and not valid:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + '.download')
            temporary.write_bytes(content)
            temporary.replace(path)
        entry['sha256'] = digest(content)
        entry['bytes'] = len(content)
        total += len(content)
    if not MANIFEST.exists():
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Sponza verified: {len(records)} files, {total} bytes, revision {REVISION}')


if __name__ == '__main__':
    main()
