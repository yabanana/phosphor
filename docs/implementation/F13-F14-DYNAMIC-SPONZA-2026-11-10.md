# F13/F14 dynamic Sponza follow-up, 2026-11-10

This increment was tested on Apple M5 Max with macOS 27.3 build 26C5035r,
Xcode 27.2 and the already cached Metal toolchain 32023.921. The active AOT
archive loaded on this runtime. F13/F14 acceptance and the merge remain gated.
The next phases in the owner's sequence are F15 and F16 after acceptance.

## Independent surface reference

The GPU snapshot now includes the same-frame isotropic sky radiance and solar
angular radius consumed by GI. It previously exported a black sky despite the
nonzero atmospheric GI input. The five filtered-indirect GPU captures at
frames 118, 119, 120, 239 and 240 are byte-identical across the export fix.
For example, the exported sky at frame 118 is RGB (0.281981, 0.523166, 0.882746).

At frame 119, the unfiltered Mitsuba camera saw a one-sided back face in front
of the receiver seen by the raster. Pixel (70,24) differed by 5.09 m. An
independent triangle check with camera back-face rejection found the raster
receiver within 0.25 µm. The new Mitsuba control applies one-sided visibility
and binary transparency to the primary ray, then samples transport with the
native path integrator from the selected pixel-centre ray. Secondary and
shadow paths use the complete scene. The wrapper uses the documented sampling
integrator interface. [Mitsuba API](https://mitsuba.readthedocs.io/en/stable/src/api_reference.html)

The runner verifies the captured frame identity, film and position readback,
then requires every pixel centre in the frozen 32×24 crop to match its GPU
receiver within the existing 0.0001 m position tolerance. The 768 receivers
match at frames 119 and 120. The other crops fail this check with 36, 30 and
29 mismatches at frames 118, 239 and 240 respectively. Each failed crop is
retained. Earlier unvalidated scalar, ray-offset and subpixel trials remain
exploratory artifacts. In particular, zero signal alone cannot admit a crop.

The two matched crops use Mitsuba 3.9.1 scalar RGB CPU, seeds 1234/5678,
unlimited depth minus direct depth 2, and the retained 2% convergence cap.
At 1,024 samples per pixel, seed disagreement is 58.54% at frame 119 and
53.16% at frame 120. The runner publishes an admissible reference image only
after convergence. Dynamic GGX and temporal qualification remain open.

For a matched capture, the CPU runner is invoked as follows.

```sh
python bench/f13_budget/reference.py path/to/snapshot \
  --positions path/to/position.pfm --out build/new-reference-crop \
  --crop 48 24 32 24 --max-spp 1024
```

The position PFM must carry the renderer's same-frame JSON sidecar. Raw
snapshot commands, camera data, seed images and per-run hashes are retained in
`build/f13-f14-reference-20261110-sky` and `build/f13-f14-regression-20261110`.

## Empty local lighting

Sponza has zero sampled local emitters in the captured frames. Directional
lighting is handled independently. The graph reuses its existing zero texture
for the empty local estimator and omits the reservoir, shading and history
passes. Guides still feed GI and reflections. Diagnostic runs execute the
regular path. A transition between empty and populated local domains rebuilds
the graph and invalidates the previous DI history for every view.

Six 384-frame shader/API validation runs compare the old and new binaries at
322×182, with camera motion, moving sun and forced Apple9 capabilities. Every
linear image is bit-identical. A 24-frame scene switch produces zero local
radiance in Sponza, then nonzero local radiance in Many Lights with eight lights.
The two binary outputs match bit for bit across the switch.

The temporal arithmetic experiment also preserved images, but three valid
1080p pairs returned p50 changes of +0.28%, −0.68% and +0.87% (positive means
faster). That candidate was removed. The empty-local-light path was retained
based on the following full preset A/B/A results.

| Pair | Baseline p50 drift | GPU p50 reduction | Candidate GPU p95 | Candidate frame p95 |
|---|---:|---:|---:|---:|
| 1 | 1.32% | 15.30% | 24.8613 ms | 25.8027 ms |
| 2 | 3.12% | 11.85% | 27.3878 ms | 28.1900 ms |
| 3 | 3.80% | 15.86% | 25.6337 ms | 26.5240 ms |

The 1920×1080 workload, ray counts and filters match the previous full Sponza
preset. Each run uses 128 warmup frames and 256 timed frames, with serial GPU
timing, an idle DwarfStar and the CPU oracle paused. All nine runs have zero
steady GPU allocations and zero archive misses or unavailable requests.
The 16.67 ms frame target remains unmet. [Performance log](../perf-log.md)

## Runtime regression results

Nine CTest groups, 33 F7/F8 controls, 13 F13/F14 functional controls, the
independent physical AO fixture and nine 300-frame F14 numerical cases passed.
F9 remains at 56/60. Bench 8 still fails at frame 48, ray 64 and the diffuse
probe at frame 37, ray 162, in both native and forced Apple9 modes.

The original native wide-HDR fixture ran all 96 SDK frames on this runtime.
Steady restored error still reaches 5.17578125% against the 1% cap. The original
lifecycle on the new MetalFX image produced 20 leaks and 12,800 bytes.
Read-only inspection identified UUID `979ED51C-1DF5-3DC5-948F-7BE4505741B3`,
the same class size of 1248 bytes and timing record at offset 1072 with the
existing scalar type. The constructor requests 508 bytes at +1496 and the
loaded allocation is 512 bytes. Neither inspected destructor releases this
field. The exact image was added to the existing private ABI guard.

Protected release passed the 300-frame, four-view, three-extent, two-generation
fixture with 20 actual weak-destruction and reclaim observations and zero
process-exit leaks. The same-binary plain-release control reproduced 20 leaks
and 12,800 bytes. Direct-spawn Guard Malloc loaded, passed the 300-frame fixture
and reclaimed all 20 records. The post-fix HDR run still reaches 5.17578125%.
The final binary also passed the normal Sponza path with the empty estimator
omitted. The timing binaries precede the cold native-lifetime change and use
the identical metallib. [Machine-readable evidence](../results/F13-F14-dynamic-Sponza-M5Max-2026-11-10.json)

## Follow-up on the preliminary gates

The empty local estimator also lets composition bypass its denoiser. Each
frame slot records whether DI was omitted. On an empty domain, the denoiser
releases its graph definition for DI and invalidates that signal's history in
all views. Captures omit the checkpoint for an intentionally absent pass.
Reintroduced local lights schedule the ordinary filter and start fresh history.
The diagnostic path still schedules all requested signals.

Six 384-frame runs retain the 5e-5 normalized image gate. Maximum error is
2.79e-7 with camera motion, 2.61e-7 with moving sun, and zero on forced Apple9.
A 40-frame test switches Sponza → Many Lights → other benches → Sponza → Many
Lights with the clock and local lights frozen. Its captures match bit for bit.
The DI checkpoint is absent on frames 0–3 and 32–35. Frames 4 and 36 both reset
with zero reuse. The following active frames show actual history reuse.

| Replica | Baseline p50 drift | Additional GPU p50 reduction | GPU p95 | Frame p95 |
|---|---:|---:|---:|---:|
| 1 | 1.24% | 5.88% | 23.4450 ms | 24.4736 ms |
| 2 | 1.24% | 3.98% | 23.6283 ms | 24.6300 ms |
| 3 | 0.11% | 4.08% | 23.4732 ms | 24.4534 ms |

The 1080p workload and A/B/A gates stay fixed, with 128 warmup and 256 timed
frames. All nine runs report zero steady GPU allocations and use the accepted
AOT archive. The 16.67 ms gate remains open.

A separate CPU pilot traced the indirect term directly from the verified
frame 119 receivers instead of subtracting two noisy images. At 1,024 spp
its seeds disagree by 51.95%, compared with 58.54% for the original estimator.
It fails the unchanged 2% gate and remains experimental. The first attempt
also hit a traversal bound on a grazing secondary ray. The repeat used the
reference renderer's surface-ray offset for secondary rays and completed.
The production reference runner retains the original estimator. Pilot code
and captures are in `build/f13-f14-secondary-pilot-v2`.

The original 96-frame native HDR control still measures 5.17578125% against
1%. Both F9 failures reproduce on native and forced Apple9. Nine CTest groups
and the physical AO fixture pass. The full preliminary gates still apply
before F15 and F16. [Measurement ledger](../results/F13-F14-dynamic-Sponza-M5Max-2026-11-10.json)
