# F7/F8 rendering contracts and verification

The visibility/HDR path is selected with `--post`; the indexed forward path
remains the default reference. Hardware acceptance is on the available M5 Max.
Forced Apple9 exercises feature selection on that machine; it is not M3/T0
certification. Phase status belongs to [ROADMAP](ROADMAP.md), measurements to
[perf-log](perf-log.md), and experiment decisions to [opt-log](opt-log.md).

## Running the integrated renderer

```sh
mise exec -- python3 tools/fetch_sponza.py
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/release -j 8
./build/release/phosphor --bench 4 --scene assets/sponza/Sponza.gltf \
  --post --upscaler temporal --render-scale 0.75 --auto-exposure --output auto
```

Use `--verify` for a check without downloading. The manifest
and upstream license are described in [assets/README](../assets/README.md).
`--scene procedural` selects the old deterministic fixture explicitly.

`--render-path visibility --material-binning on|off` compares the two material
resolve paths without temporal postprocessing. The measured default resolve is
generic; material binning remains opt-in. `--upscaler native` selects spatial
reconstruction; `--tonemap aces|agx|custom`, `--tone-white`, `--sharpen`,
`--dynamic-resolution --drs-budget MS` and `--output sdr|edr|auto` control F8.
The AgX and ACES options are documented polynomial/rational fits, not exact
Blender OCIO or a complete ACES color-management pipeline.

## GPU scene and surface data

| Signal | Format | Meaning |
|---|---|---|
| Visibility | R32Uint | `(cluster << 7 | triangle) + 1`; zero is background. Cluster names an instance/meshlet candidate in A or B, not just shared geometry. Last cluster is reserved to prevent overflow. |
| HDR | RGBA16Float | Linear scene radiance, opaque alpha 1, pre-exposure 1. Reference captures average this radiance before the display curve. |
| Normal/roughness | RGBA16Float | Signed world-space unit normal in xyz; perceptual roughness in w, clamped to [0.04,1]. |
| Diffuse albedo | RGBA16Float | Base RGB × (1 − metallic). |
| Specular albedo | RGBA16Float | Schlick Fresnel from F0 and N·V, including metal base color. |
| Motion | RG16Float | Current → previous, in input pixels, +X right/+Y down, without raster jitter. Includes camera and object motion. Invalid/new instances produce zero. |
| Depth | Depth32Float | Reverse Z: zero background/far, one near. |
| Reactive | R8Unorm | 1 for invalid history/background; 0.75 for alpha-tested materials; 0 for stable opaque surfaces. |

The compute baseline stores HDR/guides in device textures; it does not claim
that all shading remains in tile memory. The indexed overflow fallback writes
these same signals through raster attachments. Opaque and alpha-tested draws
are separate in each culling phase. Future translucent rendering (F16) follows
both phases; alpha blending is not delivered by F7's alpha-test work.

The material classifier creates bounded lists of 16×16 tiles for four feature
classes (normal mapping/emission), with a generic resolve available. Analytic
homogeneous barycentrics remain defined across the near plane, including a
vertex with clip W = 0. The forward reference retains raster derivatives, so
textured comparisons have a documented numerical/filtering envelope; binning
and tile comparisons against generic resolve remain exact.

## Temporal ownership and presentation

`--upscaler temporal` uses isolated MetalFX worker processes by default.
F8.4 engine lifetime is development-accepted on M5: workers and shared
mappings are reclaimed at resize/shutdown. The stock SDK-only release cycle
still reproduces; `--metalfx-mode direct` is a diagnostic comparison path.
Native remains the product default. See [lifetime, tests and cost](F8_METALFX_LIFETIME.md)
and [the unchanged SDK reduction](../bench/f8_spike/README.md).

Three upload slots are distinct from up to four temporal views. Every view has
its own previous poses, Hi-Z data, exposure and MetalFX scaler. Entity
incarnations prevent a recycled GPU slot from inheriting another object's
motion. The common history registry tracks validity, extent, generation and
last logical reader/writer. Hi-Z stores its jittered matrix; reconstruction
stores an unjittered matrix. Scene changes, camera cuts, resize, input extent
changes and successful shader-generation changes invalidate the relevant history.

The current backend uses an ordered graphics timeline and conservative GPU
retirement. A registry submission stamp is not CPU proof of GPU completion.
Graph dependencies and the MetalContext events/fences order resource reuse;
feedback callbacks are drained separately before context destruction. A GPU
failure is sticky and causes a nonzero process exit. A GPU timeout fails closed
instead of recycling resources that may still be active.

MetalFX executes through an `External` graph pass. The default temporal path
uses a private worker process per view/extent: GPU copies exchange pixels
through a bounded three-slot shared mapping; a socket carries only parameters
and completion. Producer/consumer submissions synchronize with shared events.
The broker runs outside the render thread and starts its timeout only after
submission publication. Unknown completion fails closed; it never makes a
possibly still-written output appear successful. At most eight workers may
be active/retiring; startup retries with native fallback when capacity is full.

In direct diagnostic mode the SDK encodes within the parent command buffer. The graph owns its
boundary barriers/fence; the framework owns its internal resources. The state
import represents that opaque per-view dependency as well as exposure state.
Graph bandwidth estimates cover declared accesses, not undisclosed SDK traffic.
Device allocation readback and GpuMemory resource accounting are reported
separately in report schema 8; `gpu_allocations` counts GpuMemory allocations
in parent and workers. SDK-internal allocations remain opaque. Parent/worker
footprints and shared mappings must not be blindly summed. GPU timing is the
first-to-last graphics submission span, including external/IPC gaps, rather
than the final display commit alone.

The scaler is initialized on PipelineCache utility workers. Startup prewarms it;
resize uses native/spatial reconstruction until the correct scaler is ready.
Dynamic resolution changes the active input rectangle within fixed backing
allocations. Input sizes round up, preserving the maximum 2× ratio for odd
outputs. Halton jitter displaces raster samples in input pixels; motion excludes
that displacement. The temporal mip bias is `min(0, log2(input/output))`.

The SDK exposure input is a 1×1 R16Float texture; adaptation state stays FP32.
Exposure uses a 256-bin log histogram, rejects black/nonfinite samples from the
mean, trims to the 5–95 percentiles and adapts using elapsed time for that view.
Manual exposure and the selected display transform are applied once. EDR uses
RGBA16Float with extended-linear sRGB and the current window screen's observed
headroom. The API's relative headroom is not a photometric measurement in nits.
PNG capture explicitly converts EDR to clipped SDR.

## Optional experiments

`--tile-resolve` evaluates the same material model in a tile kernel reading the
implicit ID imageblock. With frustum culling, raster + tile resolve fuse and IDs
are memoryless. Two-phase Hi-Z can break that fusion. HDR/guides still leave the
tile. This is a minimal on-tile visibility/deferred prototype, not a traditional
large G-buffer implementation.

`--adaptive-shading` handles 2×2 pixels per invocation. Reuse requires identical
current primitive IDs, matching historical incarnation/primitive, low motion,
low previous luminance contrast, high roughness and nonsensitive materials.
Motion is calculated per pixel. Per-view history costs 16 bytes per backing
pixel and an update pass. `--debug-adaptive-no-history` must eliminate reuse.
The two experiments are mutually exclusive and disabled by default; the final
adopt/defer/reject decisions require total-frame measurements, not shader counts.

## Reproducing checks

```sh
ctest --test-dir build --output-on-failure
mise exec -- python3 tools/run_checked.py --self-test
# Create the venv with mise Python, then install tools/quality-requirements.txt.
build/quality-venv/bin/python tools/test_temporal_metric.py
build/quality-venv/bin/python tools/f7_f8_check.py
build/quality-venv/bin/python tools/f7_f8_check.py --quality-only
build/quality-venv/bin/python tools/f7_f8_check.py --context-quality
```

Run one coordinated GPU workload at a time (its internal workers are part of
that workload). Captures/readbacks/validation are correctness
workloads, not performance measurements. `--offscreen` measures rendering
throughput separately from presentation. Keep the binary/metallib hashes,
asset manifest, commands, raw exits, EXIT marker and report together.

`--debug-visibility` checks IDs, finite/ranged guides, analytic diffuse/Fresnel
fixtures, motion and the exact age of previous GPU poses. Exposure histogram
CDFs are checked against CPU samples with a ±0.001% luminance interval only at
bin boundaries. `--debug-motion-corrupt`, `--debug-guide-corrupt`,
`--debug-history-corrupt`, `--debug-exposure-corrupt` and feedback fault injection
must fail for their intended reasons. `--debug-post-curves` compares 576 GPU HDR
ramp/primary-color samples over ACES/AgX/custom and headroom 1/2/8 against an
independent double-precision evaluation; its corruption control must fail.

The temporal suite records spatial PSNR/RMSE, residual flicker, recovery and
old-frame attraction. The v1/v2 attraction heuristics confuse subpixel spatial
coverage with history and remain diagnostics. v3 gates attraction outside the
one-pixel spatial reconstruction support at the declared 75% preset; it does
not identify latency smaller than that support. Per-view histories, image
inspection and delayed-frame negative controls remain necessary. Earlier
failed reports and thresholds are preserved, not relabeled as passes.

Shader AOT and hot reload compile `material_passes.metal` first, grouping the
forward/resolve sources to avoid the measured Metal 27.1 metadata-relocation
failure. `-fpreserve-invariance` accompanies invariant geometry positions.
The archive harvest excludes private MetalFX libraries. See the execution
record for the measured reproducer, deviations and pending final checks:
[F7/F8 execution](plans/F7-F8-EXECUTION.md).
