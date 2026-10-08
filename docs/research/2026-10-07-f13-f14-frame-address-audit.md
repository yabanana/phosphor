# F13/F14 cached graph address audit

Immutable snapshot `7a7b8d40cdd0fab0c6cfdb7d97e9254b49aa00f2`. No source changes/build/GPU by reviewer.

No remaining confirmed frameUpload address capture bug in reviewed source. Do not patch the safe reflection sample callback merely because add contains a redundant upload.

| File | Refresh / encode lines | Result |
|---|---|---|
| `reflection_passes.cpp` | 201 / 333 | SAFE: callback captures this+sample index, not local address; prepare refreshes every sampleAddress entry each frame. The second upload in add is redundant only. |
| `denoise_passes.cpp` | 57 / 112 | SAFE: prepare calls updateSignal for previously-defined live signals; first add also initializes. Callback indexes current signal address and current view/slot history. |
| `metalfx_denoise.cpp` | 326 / 411 | SAFE: ready-frame prepare refreshes member pack/restore addresses; callbacks read members, not addresses captured at graph construction. |
| `metalfx_denoise_fixture.cpp` | 134 / 145 | SAFE in pinned snapshot: per-frame prepare uploads fixture params; generation/depth/readback callbacks use current member. Later in-progress SDK prewarm merge is outside this pinned review. |
| `atmosphere_passes.cpp` | 211 / 275 | SAFE: current atmosphere/fog/cloud/dummy parameter members refreshed in prepare. Captured input/raw/guide variables are graph refs whose physical textures are rebound by current frame. |
| `volume_diagnostics.cpp` | 67 / 76 | SAFE after c6381a8: current per-slot stamp addresses; foreign-history diagnostic upload happens during current encode, not cached construction. |
| `volume_diagnostics.cpp` | 84 / 92 | SAFE after db6d159: captured Sources contains logical graph refs only; fog buffers and counters are resolved from current Context at encode, avoiding stale frame-slot pointers. |

The raw UploadRing contract permits a slice only through completion of its producing frame. Capturing an address by value in an enduring graph callback would violate that contract; the reviewed production paths instead refresh member/per-slot addresses before each execution. Logical graph refs and stable sample/mip indices may remain captured across cached frames.

No broad refactor or change to sample seeds/frame counts is justified by this audit. Runtime checks should still exercise ring rollover and view/resize changes, but they are validation of these paths rather than evidence of a currently identified stale address.
