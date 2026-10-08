# Physical opaque emission must survive the primary residual

Source-only patch `residual-float32.patch`; SHA256 `71981340f8dc026453abd81d6f82a8db90ee077d739fc78c6d7b51c12066db40`. The per-file base hashes and root commit are in `patch-provenance.json`. The shared integration source was never edited. No C++ build, shader build or GPU workload was executed by this agent.

## Confirmed source defect

`material_shading.h` adds material emission in Float32. Both the normal visibility resolve and the indexed ICB overflow path write it into `VisibilityRenderer` color output 0, previously always RGBA16Float. The indexed fragment additionally casts to `half4` before storage. Thus an opaque surface emitting [368640,128,64] loses red range before F13/F14 sees it. Creating a Float32 composite later cannot recover that value; the F13 composite can also reject a nonfinite residual and replace it with zero.

The patch selects RGBA32Float only for the F10–F14 lighting residual, consistently in texture creation and the indexed fallback PSO. A separate `ForwardLitSurfaceOutput` uses `float4` color for `forward_surface_lit_fs`; its guide types and buffer-only ICB bindings are unchanged. `forward_surface_fs`, the legacy output struct, and the F7/F8 nonlighting texture formats are unchanged.

## Consumer audit and explicit limit

- Visibility diagnostic readback already allocates 16 bytes per color pixel and decodes the actual RGBA32 format. It continues to read the original resolve attachment, preserving the recently fixed immutable binding contract.
- F13 raw/pre-reflection and final composites already use RGBA32. F14 physical targets already use RGBA32. LinearCapture reads both formats into Float32 without a hidden narrowing conversion.
- Native/spatial post reconstructs with `float4`, including supersampling and bilinear sampling. Its output allocation now selects RGBA32 for **all** lighting paths, including isolated F10/F11/F12, so the restored range is not lost just before tonemapping. Display conversion remains after the tone curve.
- Standard F8 MetalFX temporal has a fixed RGBA16 input/output descriptor (and an RGBA16 worker ABI), and directly binds the source. There is no existing radiometric pre-pack/restore for physical Float32 HDR. Its combination with F13 Float32 was already a format mismatch; silently adding a HALF conversion would lose the recovered values.
- Therefore lighting + `--upscaler temporal` now fails explicitly at CLI validation with an actionable native/denoised message, and the Post API rejects physical-Float32 + standard temporal independently. This is a **real compatibility limitation**, not verified support. F7/F8 without lighting retains the existing temporal path. The F13 denoised path has its own currently unpromoted wide-HDR policy; this patch does not certify it.
- Post also rejects a Float32 input when configured to produce HALF, so an internal caller cannot reintroduce late clipping accidentally.

One independent existing diagnostic caveat remains: `--debug-visibility` computes its exposure histogram from the raw resolve attachment, while post can consume the later F13/F14 composite. With nonzero external light contributions these histograms can differ. This fixture has no external radiance, so they are equal by construction; the patch does not modify that separate checker behavior.

## Actual opaque fixture and independent reference

New `--reflection-scene wide-emission` on bench 6 contains one real opaque plane facing the camera, large enough to cover the whole image. Its emission is [368640,128,64]. Its sole directional light has zero intensity, base albedo is black, and a black occlusion texture zeros the legacy hemisphere contribution. There is no texture emission multiplier, no tone-map reference fitting and no noisy source. This makes each visible pixel's outgoing radiance exactly the declared emission, including through F13 composition when AO is enabled.

`tools/wide_emissive_check.py capture.pfm --report result.json` checks **all pixels and all three channels** against the analytic target. Its fixed error budget is gamma16 for FP32 arithmetic, separately scaled by each target component; a bright red channel cannot hide loss in green/blue. It refuses result overwrite. The three executed Python tests detect old HALF saturation (65504), Inf and black repair, and one bad pixel/channel. The new C++ fixture and CLI tests were authored but not compiled/run here.

Suggested root GPU cases, all at a fresh output path, native post, eight frames, warmup 0, fixed timestep, no UI/vsync/GPU timing:

1. `--shadows csm`: actual Float32 residual without F13 composition.
2. Same, `--render-scale 0.5`: native/spatial reconstruction path.
3. `--ao gtao`: F13 positive composite, same analytic emission.
4. Same plus `--debug-meshlets 1 --debug-meshlets-corrupt count`: actual indexed ICB overflow. Renderer exit1 and the precise expected meshlet checker failure are intentional; lighting/visibility and radiometric oracle must still pass. Reject unrelated validation/crash/failure.
5. F13 case with `--force-family apple9`: capability restriction, not a claim of physical T0 testing.
6. Negative: same fixture with no F10–F14 lighting options uses the deliberately preserved F7/F8 HALF storage. The analytic oracle must reject its captured red. Any unrelated GPU/API failure is not a qualifying negative.

The companion `frozen-gpu-protocol.json` records commands and gates but does not launch processes. Use the existing `run_checked` discipline and preserve every raw log/report/capture. These are new captures, not reinterpretations of the standalone SDK fixture. Native SDK wide-HDR acceptance remains open.
