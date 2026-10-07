# F9.3 measured RT proxy cook

The production geometry and policy live in `renderer/rt_proxy.*` (portable,
CPU-tested). The GPU cook uses the existing `F9-S4` harness so that the selected
mesh levels and their error are measured together, against the full scene.
There is deliberately no ratio-only export path that invents ray-error data.

From the repository root, after building `f9_spike` with `rt_proxy.cpp` in
`phosphor_core`, run **one GPU workload at a time**:

```sh
PHOSPHOR_RT_PROXY_EXPORT_DIR=build/rt-proxy-cook \
  ./build/release/f9_spike --only F9-S4 --runs 3 \
  --out build/rt-proxy-cook-results.json
```

`--quick` cannot export: adoption requires the full three-camera 960x540
Sponza corpus. The output directory is created only after that scene passes
its full-vs-full identity, deliberately bad proxy, single-mesh sabotage and
protected border-locked policy controls. The caller must also check the
harness exit status/report. Only then copy
`build/rt-proxy-cook/sponza.rtproxy.json` to
`assets/manifests/sponza.rtproxy.json` for review/commit. Retain the run report
beside the phase evidence. No generated manifest is committed merely because
this code exists.

The cooker uses the production simplifier for r10b/r25b/r50b, starts coarse,
and promotes meshes with at least 2% of eligible blame (or the worst offender
if none reaches that share). Meshes referenced by any alpha-tested or emissive
material, or unknown material index, are full before policy evaluation.
Thresholds are the measured S4 policy: shadow disagreement <=0.5%, primary
hit/miss or distance error >5cm <=0.2%, distance p95 <=1cm, acne <=0.5%.
These characterize the named **opaque geometric ray corpus**; they are not
conservative bounds on every camera, light, dynamic pose or alpha texture.

The manifest contains schema + meshoptimizer versions, a full vertex/mesh/index
fingerprint, level and actual index fingerprint per mesh, meshoptimizer
relative/object-space error, ray populations, measured errors and corpus.
The exporter compares every regenerated index fingerprint against the buffers
actually traced on the GPU. FNV-1a fingerprints detect content/version drift;
they are not signatures for adversarial assets.

Runtime integration:

```cpp
RtProxyManifest manifest;
std::string diagnostic;
const bool loaded = rtReadProxyManifest(path, manifest, diagnostic);
const auto protectedMeshes = rtProxyProtectedMeshes(scene.getMeshCount(), instances, materials);
const auto geometry = rtBuildProxyGeometry(scene, loaded ? &manifest : nullptr, protectedMeshes);
// Upload geometry.indices; BLAS m uses geometry.meshes[m].vertexOffset,
// indexOffset and indexCount. Vertex positions/UVs stay in the original buffer.
```

No valid manifest means full geometry. A changed scene/cooker/output index
stream or failed/missing evidence invalidates the complete manifest, not just
the first bad mesh. Runtime material changes may additionally promote meshes
to full. Rebuild that selection if assignments change; the manifest's offline
errors describe its original corpus, not newly assigned materials. Proxy
`GPUMeshInfo` records have cleared meshlet ranges and must never replace raster
mesh metadata.
