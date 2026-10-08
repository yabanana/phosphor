# F13 SDK factory concurrency and lifecycle: partial result

Device: M5 Max, macOS27.2 (26B5091g). This concerns the **denoised** SDK
adapter, not the previously accepted F8 temporal scaler.

`build/f13-sdk-lifecycle-v1` requested four scalers concurrently from the
owned PipelineCache compiler. Three became ready. A fourth worker spun in
Metal's `_MTLMPSGraphInterface::getMPSGraphClassByName`, inside insertion in
its Objective-C class hash table. The initial120s fixture deadline failed;
destruction could not cancel that SDK call. The tester terminated its hung
process. The sample is preserved in
`build/lighting-smoke/f13-sdk-lifecycle-sample.txt`.

The bounded correction serializes SDK scaler **construction** through one
PipelineCache mutex. Ordinary pipeline compilation remains parallel, requests
remain asynchronous, and per-frame GPU encoding takes no new lock. The same
lock covers both SDK factories sharing this compiler. This is a correction to
our use of the shared compiler; the sample alone does not establish a public
SDK thread-safety guarantee or its violation.

`build/f13-sdk-lifecycle-v2` with that change initialized four views in278.607ms
and exited without the worker spin. Build and all six CTest targets passed.
This is **not lifecycle acceptance**: the fixture counted300frames but only73
native encodes. Later resize/reload requests were still pending; there was only
one captured extent. Its initial-only prewarm is insufficient to exercise
each asynchronous replacement before the short unpaced run ends.

The at-exit check also failed: **12 allocations /7680bytes**, each640bytes,
created in `_M4FXTemporalDenoisingScalingEffect initWithDevice:compiler:descriptor:history:`.
All12 factory requests correspond to that count. It is not a proved instance
of the F8 self-reference and does not authorize reusing that lifetime fix,
releasing unknown internal ownership, or treating retirement as destruction.
Raw logs and the failed SDK result remain in the v2 directory.

Next verification: bounded fixture-only readiness after each generation/size
transition, followed by fresh lifecycle and at-exit checks. Production stays
asynchronous. Native production radiometry is independently unqualified on
wide colored HDR; F13.3 remains partial regardless of factory progress.
