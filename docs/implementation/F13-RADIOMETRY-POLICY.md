# F13.3 — Native radiometry admission remains unqualified

F13.3 is **partial**, not production-native accepted. The engine contains an actual typed MetalFX denoised gateway and controlled fixtures that encode and read back real SDK frames. That implementation evidence does not certify arbitrary scene-linear/HDR radiometry.

## Evidence and stop decision

The original 96-frame wide-HDR case keeps the physical target [368640,128,64], storage preExposure q=1/64, packed input [5760,2,1], guide inputs, SDK/q reconstruction and its 1% output gate. Unit exposure lost the dominant red. Automatic exposure saturated. Analytic manual exposure recovered red but exposed input texture-write rounding; a representable physical-relative exposure then passed the channel checks but failed chromatic accuracy. The final, exact input-relative exposure0x05b0 also passed channel checks and still failed the original output gate (reported steady-frame worst error about2.64–5.18%). No parameter sweep, output compensation, clipping or gate relaxation follows.

Raw evidence is retained in `build/f13-native-units-v1`, `build/f13-wide-auto-v1`, `build/f13-wide-manual-v1`, `build/f13-wide-manual-exact-v1` and `build/f13-wide-input-exact-v1` in the tester checkout. Source/binary/manifest hashes and actual frame/guide readbacks are in those artifacts. Smaller constant/impulse/channel PASS cases remain scoped evidence, not proof of a continuous universal numeric range. This is not a claim that an SDK defect has been established.

## Production behavior

`MetalfxDenoise::Options` defaults to `RadiometricDomain::UnqualifiedSceneLinear`. `--lighting-denoise metalfx` records the request but selects the existing custom Float32 path before graph construction. The adapter reports `UnqualifiedRadiometricDomain`; it creates no denoised factory request, encodes no native frames, and the engine adds no apparent SDK roughness/pack/encode branch. The fallback reason explicitly names the unqualified radiometry.

Only `MetalfxDenoiseFixture` sets `ControlledFixtureDiagnostic`. This is permission to measure an unqualified profile, not production qualification. There is no production CLI override, no domain inferred from `preExposure==1`, and no admission merely because inputs are finite or fit inside HALF. The fixture remains isolated from normal scene inputs and keeps its own actual SDK evidence and failure status.

The report separates `denoise_requested`, `denoise_effective`, `denoise_fallback`, `denoise_radiometric_domain`, factory requests and encoded native frames. `denoise_native_production_qualified` remains false. Fixture JSON likewise identifies the controlled diagnostic domain and explicitly denies production qualification. A custom fallback must not be reported as native execution.

Physical opaque residuals, F13/F14 composition, custom histories/outputs and native/spatial post retain Float32. Display conversion follows tone mapping. Standard F8 temporal still has a HALF input/output ABI and is explicitly incompatible with physical lighting; the verified F7/F8 nonlighting path remains available. None of these choices bypasses the outstanding native wide-HDR gate.

## Future admission

A later native-production profile needs fresh, bounded evidence for its input/output units, exposure basis, range and channel ratios, guide semantics, motion, cuts, resize and target-device/SDK provenance. Runtime pack checks remain necessary but cannot manufacture such qualification. Unknown content selects custom before presentation; a failure discovered only in CPU readback is too late to protect the already submitted frame. Any future admission requires an explicit reviewed policy change and keeps the same independent references and negative controls.
