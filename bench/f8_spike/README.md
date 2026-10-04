# F8 MetalFX lifetime reduction

On the available M5 Max, macOS 27.2 (26B5091g), creating/releasing a Metal 4
MetalFX temporal scaler at 640×360 leaves a live zeroing-weak reference after
all caller-owned objects and autorelease pools are released. `leaks` reports
a cycle between `_M4FXTemporalScalingEffectBBR` and its BBR filter, approximately
0.3 MB in this minimal configuration (the size is configuration-dependent).

The reduction uses only public SDK APIs: no Phosphor renderer, graph, textures,
worker thread, encoding or queue. It is not an expected passing test on the
current SDK. Exit 1 records the reproduced residual; 0 means the weak target
was destroyed, and 77 means the Metal 4 path is unavailable.

```sh
xcrun clang++ -std=c++20 -fno-objc-arc -mmacosx-version-min=26.0 \
  -framework Foundation -framework Metal -framework MetalFX \
  bench/f8_spike/metalfx_lifetime.mm -o /tmp/phosphor-metalfx-lifetime
codesign --force --sign - --entitlements cmake/debuggable.entitlements /tmp/phosphor-metalfx-lifetime
/tmp/phosphor-metalfx-lifetime
leaks --atExit -- /tmp/phosphor-metalfx-lifetime
```

Creation with asynchronous initialization, disabled dynamic content or disabled
reactive masks did not remove the residual in the 640×360 reductions. Some
smaller/early reductions yielded `0 leaks`; that is not a reliable destruction
proof, so the final reduction checks zeroing-weak liveness as well. Main-thread
creation, draining the factory pool before publication, clearing input bindings
and discarding completed command streams did not repair the engine's result.
Those ineffective changes were not adopted as fixes. A further 30-second
run-loop drain on 2026-10-03 still leaves one live weak target (exit 1);
this is not explained by the original two-second wait alone.

Phosphor's native HDR path has `0 leaks` in the same exit-time test.

**Root cause (2026-10-04).** The standard temporal scaler (Metal 3
`_MFXTemporalScalingEffectBBR`, Metal 4 `_M4FXTemporalScalingEffectBBR`) owns
its C++ `BBRNet_Filter` through a `unique_ptr`, and the filter keeps a strong
reference back to the scaler: right after creation the retain count is 2 with
one external owner (spatial: 1). The scaler is never deallocated, so the
filter, which would be destroyed by it, never is either. The engine now
releases that self-reference explicitly and only when it is provably the last
owner ([`metalfx_lifetime`](../../src/platform/metal/metalfx_lifetime.h),
[analysis](../../docs/research/2026-10-04-metalfx-cycle-root-cause.md)). Class and
ivar names were used only to diagnose; the engine relies on the measured
reference count, a weak reference and plain releases, never on private ivars.
The plain reduction below still fails on this runtime: it is the negative
control and must keep failing until the framework itself changes.

`--release-cycle` (matrix, standard temporal) releases through the engine's
code; build with that file and the metal-cpp implementation unit:

```sh
xcrun clang++ -std=c++20 -fno-objc-arc -mmacosx-version-min=26.0 -DPHOSPHOR_RELEASE_CYCLE \
  -Isrc -Ibuild/_deps/metal_cpp-src -framework Foundation -framework Metal -framework MetalFX \
  -framework QuartzCore bench/f8_spike/metalfx_lifetime_matrix.mm \
  src/platform/metal/metalfx_lifetime.cpp src/platform/metal/metal_impl.cpp -o build/matrix-release-cycle
build/matrix-release-cycle --mode temporal4 --count 8 --release-cycle   # FINAL live=0, exit 0
leaks --atExit -- build/matrix-release-cycle --mode temporal3 --count 8 --release-cycle --no-weak
```

The isolated worker experiment (PR #15) remains available as
`--metalfx-mode isolated`; its overhead was rejected by the owner. Rerun this
reduction and the engine lifetime tests when the runtime is updated: the
`METALFX-LIFETIME` exit line reports `released` (plain release sufficed),
`cycle-released` (the self-reference was dropped) and `retained` (leak, FAIL).
No external report has been sent.

## SDK/API comparison and repeated creation

The [2026-10-03 investigation](../../docs/research/2026-10-03-metalfx-lifetime.md)
compares SDK 26.5 and 27.0 on the same macOS 27.2/MetalFX 40.9 runtime.
Changing the build SDK does not replace that framework. Eight standard
640×360 temporal scalers remain alive, with device allocation delta 159.39 MiB.
The earlier approximately 0.3 MB leak-scan figure was not total retained memory.

```sh
mise exec -- python3 bench/f8_spike/run_lifetime_matrix.py \
  --sdk /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  --sdk /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk \
  --out build/metalfx-lifetime-matrix
```

Use installed SDK paths; omitting `--sdk` uses the current xcrun SDK. All tests
run sequentially. Exit 1 is expected on the affected runtime and records a
failed lifetime gate, not a tool success. The JSON records raw process exits,
weak liveness, allocation progression and a separate observer-free leaks scan.
A zero `leaks` count alone is insufficient: conservative scans can miss cycles.
The denoised control releases its scaler but leaks 640 CPU bytes per creation,
so weak liveness alone is also insufficient. Spatial is the passing control.

`metalfx_lifetime_matrix.mm` accepts `--mode temporal4|temporal3|spatial4|denoised4|denoised3`,
`--count 1..16`, bounded width/height, and `--no-dynamic`, `--no-reactive`,
`--async`, `--auto-exposure`, `--reset`, `--no-weak`. Public-option switches apply
to the relevant effect; dynamic content applies only to standard temporal.
`--no-weak` prints `live=-1`: successful execution is not proof of destruction.

Follow-up switches (standard temporal unless noted): `--output-scale 1..3`
(output still bounded to 1920x1080), `--format rgba16|rgba8|rg11`,
`--depth-r32`, and `--gpu-drain` (all modes). The latter executes and checks
a real Metal blit before creation and after each release/pool drain. It tests
deferred driver reclamation; it does not encode the scaler or prove image
quality. An external runner must bound process lifetime because the diagnostic
blit waits for completion. These variants still reproduce the surviving
standard scaler on the current runtime. Spatial with GPU drain is the passing
control; denoised with GPU drain still leaks 640 CPU bytes per creation.

Example after building the matrix source with the flags above:

```sh
./matrix --mode temporal4 --count 8 --gpu-drain
leaks --atExit -- ./matrix --mode temporal4 --count 8 --gpu-drain --no-weak
./matrix --mode temporal4 --count 8 --output-scale 2
./matrix --mode temporal4 --count 8 --format rg11
./matrix --mode spatial4 --count 8 --gpu-drain
```

Running under a different Xcode/SDK on this boot does not test another
runtime. No stable-runtime result is recorded; after the root cause was found
in process, the gate no longer depends on that comparison.

## Rejected denoised alternative

`denoised_encode.mm` verifies real Metal 4 encoding with the denoise bypass
mask set to **one**. Build it using the same frameworks and flags as the minimal
reduction. Run with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`; its optional
`--mismatch` negative control requires API validation and must abort with
`Color texture width mismatch from descriptor` before the denoiser is encoded.
For leak scans, sign it with the debug entitlement shown above.

`denoised_native_spike.patch` is an **unapplied research artifact**, based on
`862032e`, not a supported backend. Apply only in an isolated checkout for
reproduction, build, and enable `PHOSPHOR_DENOISED_SPIKE=1` with
`--post --upscaler temporal --render-scale 1 --no-pipeline-archive`.
The spike intentionally rejects non-native input sizes. The mask disables
denoising according to the SDK but still selects a different reconstruction
algorithm. The experiment did not remove all leaks and did not implement DRS;
it was removed from active renderer code. See the investigation for quality
results and limitations. No phase-completion claim follows from this patch.
