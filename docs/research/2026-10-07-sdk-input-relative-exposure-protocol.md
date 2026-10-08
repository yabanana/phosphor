# Final bounded exposure-basis control

Source-only patch `input-relative-exposure.patch`, SHA256 `cbc4285401e8ffd1a68cb1ab006c5617425f3a3b7e72c8c445f0dfe19ded590a`, above the exact physical-exposure experiment. Three files. `git apply --check` and fourteen Python CPU tests passed. No build/GPU/shared-source edits.

## Why one final unit hypothesis is plausible

The [denoised exposureTexture contract](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/exposuretexture) describes multiplying input color by the R16Float exposure value. [preExposure](https://developer.apple.com/documentation/metalfx/mtlfxtemporaldenoisedscalerbase/preexposure) separately declares existing input premultiplication. Apple's [WWDC25 exposure discussion](https://developer.apple.com/videos/play/wwdc2025/211/) explains the hint in terms of input color multiplied to approximately displayed brightness and says it should not change output brightness.

Those public descriptions do not fully specify the internal ordering/normalization when both controls are present. Therefore exposure relative to the supplied packed color is a plausible remaining interpretation, not an established fix or an SDK defect claim.

One parameter changes from the preceding experiment: E is derived from the actual input texture red, 368640*q=5760, rather than the physical red368640.

- q remains1/64; physical target remains[368640,128,64]; packed color remains[5760,2,1].
- Ideal E=.5/5760=8.680555555555556e-5.
- Requested FP32=8.680555765749887e-5.
- Provided exact R16Float=8.678436279296875e-5, bits0x05b0, hexadecimal Float32 literal0x1.6cp-14f.
- Packed red times provided E=0.4998779296875.
- Auto remainsfalse, strength0, 96 actual native frames, same guides, all thresholds unchanged, output restoration stillSDK/q without E compensation.

Metadata explicitly identifies `manual_exposure_basis=packed-input-color`. Requested ideal, prequantized supplied value and actual texel readback remain distinct. The new case name is `wide-hdr-manual-input-exposure-exact`.

After root build, freeze a fresh directory:

```sh
python3 tools/metalfx_denoise_fixture_check.py --binary build/lighting/phosphor --out build/f13-wide-input-exact-v1 --source-root . --gateway native --frames 96 --resolution 128x96 --denoised-fixture-manual-exposure-control
```

Consume the unchanged manifest with `--run --manifest .../sdk-frozen-manifest.json`. This is the final parameter control: do not follow it with exposure sweeps, color compensation, changed q/targets or relaxed acceptance. All earlier failures remain immutable. A pass here would validate this one fixture/configuration only.

## Production policy proposal after this control

1. Keep physical scene-linear storage and custom denoising Float32 through the opaque residual, lighting composition, temporal/custom filtering and native/spatial post. The separate residual fix addresses real information loss independently of MetalFX quality.
2. Select the custom path **before constructing/encoding the graph** for unverified physical/wide/chromatic native domains. Report requested mode, effective mode and an explicit radiometric-domain reason. Do not call that native SDK execution or complete wide-HDR native acceptance.
3. Native denoising is an explicit opt-in for an accepted numeric profile. That profile must state input/output units, q/exposure basis, tested range and channel ratios, guides, motion/resize/history behavior, device/SDK provenance and quality thresholds. The existing small fixtures are useful evidence but do not establish a continuous universal range or arbitrary scene compatibility.
4. `preExposure==1`, finite input, or values below65504 are insufficient admission criteria: the failed cases can be finite and chromatically wrong. No constant tone/exposure fit or repaired-to-zero input may turn a failed profile into a pass.
5. Runtime pack checks remain necessary but cannot replace profile admission. A CPU readback-only fallback decision happens after a native frame may have been presented. Until a profile and same-frame guard/fallback are verified, choose the available custom Float32 path in advance for unknown scenes.
6. Keep standard F8 temporal's HALF ABI limitation explicit when physical lighting is active; retain verified nonlighting F7/F8 behavior. The denoised SDK path has a separate unresolved radiometric acceptance boundary.

This is a bounded, honest shipping policy, not proof that the SDK is incapable of the domain or that all native integration details are exhausted. Record the unaccepted native domain as open work with exact reproducer and raw evidence; no speculative platform/OS/lifetime workaround is warranted.
