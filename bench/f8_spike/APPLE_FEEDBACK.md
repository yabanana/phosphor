# Apple Feedback draft — MetalFX temporal scaler never deallocated

Ready to paste into Feedback Assistant (area: **Developer Technologies & SDKs →
MetalFX**, type: **Incorrect/Unexpected Behavior**). Attach the files listed
at the end. Written 2026-10-04; nothing has been submitted yet.

## Title

MetalFX temporal scalers are never deallocated: BBRNet filter keeps a strong
reference to its owning scaler (retain cycle, ~20 MB GPU memory per instance)

## Description

On macOS 27.2 beta 2 (26B5091g) with MetalFX 40.9, every standard temporal
scaler created with `-[MTLFXTemporalScalerDescriptor newTemporalScalerWithDevice:]`
or `newTemporalScalerWithDevice:compiler:` stays alive after the application
releases its only reference. Its GPU resources are never freed, so every
recreation (output resize, view open/close) permanently grows the device's
allocated size: about 20 MB per scaler at 640×360 and about 165 MB at
1920×1080.

The retain count of a freshly created scaler is already 2 while the caller
holds the only reference (spatial scaler: 1). `leaks` reports a root cycle
`_MFXTemporalScalingEffectBBR._filter (unique_ptr<BBRNet_Filter<MFXDevice3>>)`
→ scaler (Metal 4: `_M4FXTemporalScalingEffectBBR` / `BBRNet_Filter<MFXDevice4>`):
the C++ `BBRNet_Filter` owned through `_filter` appears to hold a strong
reference back to the scaler, so `.cxx_destruct` never runs.

The spatial scaler created in the same program is deallocated normally.
The temporal denoised scaler is deallocated but still loses 640 bytes of CPU
memory per creation.

## Steps to reproduce

1. Build the attached single-file program (ARC):
   `clang -fobjc-arc -framework Foundation -framework Metal -framework MetalFX apple_feedback_repro.m -o metalfx_temporal_leak`
2. Run `./metalfx_temporal_leak`.
3. Optionally run `leaks --atExit -- ./metalfx_temporal_leak`.

## Expected

After the program drops its only reference, each scaler is deallocated, the
weak table is empty and `currentAllocatedSize` returns to its baseline, as
for the spatial scaler.

## Actual

```
Apple M5 Max | Version 27.2 (Build 26B5091g) | MetalFX 40.9
temporal (Metal 4)   class _M4FXTemporalScalingEffectBBR, retain count right after creation: 2 (one owner)
temporal (Metal 4)   8 created and released -> 8 still alive, device allocation +165019648 bytes
temporal (Metal 3)   class _MFXTemporalScalingEffectBBR, retain count right after creation: 2 (one owner)
temporal (Metal 3)   8 created and released -> 8 still alive, device allocation +167673856 bytes
spatial (control)    class _MTL4FXSpatialScalingEffectEFFECT_NAME_V1, retain count right after creation: 1 (one owner)
spatial (control)    8 created and released -> 0 still alive, device allocation +0 bytes
LEAK: released temporal scalers stay alive
```

`leaks --atExit`: `ROOT CYCLE: <_MFXTemporalScalingEffectBBR._filter
(unique_ptr<BBRNet_Filter<MFXDevice3>>)> … _filter --> CYCLE BACK TO …`, one
per created scaler.

Not affected by: SDK 26.5 vs 27.0, Metal 3 vs Metal 4 API, synchronous vs
asynchronous initialization, reactive mask or dynamic input content on/off,
auto exposure, reset, texture formats, output scale, encoding real frames,
GPU idle drains, waiting 30 s, or Swift/ARC instead of manual release.
Under the GPU capture layer the application receives a
`CaptureMTL4FXTemporalScaler` wrapper; the wrapped scaler leaks in the same way.

## Configuration

- Mac17,6, Apple M5 Max, 128 GB
- macOS 27.2 beta 2 (26B5091g), MetalFX.framework 40.9
- Xcode 27.0 (27A266a); also built with the macOS 26.5 SDK
- Stable 27.0.x not tested

## Impact and current workaround

Engines that recreate the scaler on output resize or per view accumulate GPU
memory without bound. Our engine currently releases the scaler's internal
self-reference itself, only on MetalFX 40.9 and only when the retain count
proves it is the last owner. We would like to remove that workaround once
the framework releases the filter's reference (or holds it weakly).

## Attachments

- `apple_feedback_repro.m` (this directory)
- `apple-repro-output.txt`, `apple-repro-leaks.txt` (regenerate with the
  steps above; the leaks log is anonymised by the tool)
