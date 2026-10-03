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

Phosphor's native HDR path has `0 leaks` in the same exit-time test. MetalFX
rendering/quality tests pass, but its lifetime gate remains open as F8.4.
Native is the default; temporal use is explicit. Do not force releases or alter
private framework ivars to hide this cycle. Rerun this reduction and the engine
lifetime tests when the runtime is updated. No external report has been sent.
