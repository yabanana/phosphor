# Scene assets

## Pinned Sponza fixture (F7/F8)

Fetch all model buffers, textures and attribution files, then verify their
SHA-256 checksums:

```sh
mise exec -- python3 tools/fetch_sponza.py
mise exec -- python3 tools/fetch_sponza.py --verify
```

The fixture lives in `assets/sponza/` (ignored by Git). Source revision,
individual file sizes and hashes are recorded in
[the manifest](manifests/sponza.json). The Scene Viewer discovers this path.
The initial download is 73 files, 52,690,203 bytes (about 50.3 MiB).

Source: [Khronos glTF Sample Assets — Sponza](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/edc7c9e67c639d230715049ee31f9a96a6babbbe/Models/Sponza).
This is the Crytek Sponza conversion, not the newer Intel Sponza scene.
Upstream declares `LicenseRef-CRYENGINE-Agreement`; do not relabel it CC-BY.
`SOURCE-README.md` retains creator/conversion credits and licensing notes;
`LICENSE.txt` retains the upstream license reference. Asset binaries are
not redistributed by this repository.

Procedural test scenes remain available without this download. Comparisons
must record which scene was loaded; a procedural fallback is not a Sponza
result. The renderer logs the selected source path.

## Other optional scenes

- [Bistro](https://developer.nvidia.com/orca/amazon-lumberyard-bistro)
- [Damaged Helmet](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/DamagedHelmet)

Keep each source's attribution and licensing metadata with its files.
