# F8.4 — isolated MetalFX lifetime experiment

Current status: **adoption rejected by the owner** (2026-10-03); F8.4 was
resolved in process on 2026-10-04 ([root cause](../research/2026-10-04-metalfx-cycle-root-cause.md)).
This experiment remains available as `--metalfx-mode isolated`.
The measured worker overhead is not accepted as the final temporal path.
See the [follow-up investigation](../research/2026-10-03-metalfx-public-fix-check.md).
Historical technical result on M5: the gates below were executed;
see [final delivery](../F8_METALFX_LIFETIME.md) and its source evidence. PR #15 remains the delivery
branch. The SDK's in-process destruction defect is not claimed fixed.

## Evidence and decision

Independent Swift ARC + main-dispatch drain reproduces eight surviving
standard temporal scalers. The initial subprocess probe uses the same public
Metal 4 scaler and shared file mappings, no private API. Sixteen sequential
worker lifetimes, alternating 1280x720 and 640x360, perform twelve frames each
with 1x/2x input regions. Constant-color output checks and API/shader validation
pass; every child is reaped. Parent physical footprint stabilizes around
1.98 MB. This is not yet a renderer quality/performance or global GPU-memory
reclamation proof. Raw evidence: `build/metalfx-recheck/`.

## Implementation contract

- Same temporal algorithm/settings, one persistent worker per view/extent.
  DRS changes the active input rectangle on the same scaler. Output resize
  retires the old worker after its GPU users finish, then terminates/reaps it.
- Use a private socketpair and unlinked shared-file mapping inherited by
  posix_spawn. No registered service, network listener, private framework
  fields, double releases, device spoofing or unbounded scaler cache.
- Separate per-frame slots in the shared bridge. Copy color/depth/motion/
  reactive/exposure to it on the GPU; the child copies into its private inputs,
  runs MetalFX and copies HDR output back. No CPU pixel conversion.
- All parent/child engine buffers/textures remain owned/accounted by GpuMemory;
  worker startup uses PipelineCache's compile workers. No scaler initialization
  on the renderer thread. Keep native fallback while a resize worker starts.
- An External graph pass may split a submission: signal input-ready after
  producers, wait output-ready before consumers. A broker performs IPC on a
  CPU thread, never the render thread. Every event wait has bounded failure
  handling; a crashed/timed-out worker causes a fail-closed process exit. Unknown GPU
  completion must never be hidden by reusing/clearing the output while the
  child may still be writing it. OS teardown destroys the queued waits; other
  children observe EOF because unrelated descriptors are close-on-exec.
- Resource retirement keeps mappings, events and worker alive through the
  last GPU reader. Reaping runs off the hot path. Shutdown drains jobs and
  reaps all workers. Report worker count, failures, device allocations and
  footprint separately from parent numbers; shared memory cannot be blindly
  summed into a physical-memory total.
- Direct mode stays available only for diagnostics/comparison; native is still
  the product default. Adopt isolation as the temporal path only after tests.

## Gates before adoption

1. IPC/layout bounds and negative protocol controls; worker death/timeout
   must fail closed without a blocked GPU queue or orphan process.
2. Same-scene direct vs isolated captures, including HDR/exposure/motion,
   odd dimensions, 1/2/3 frames in flight, multiple views and DRS.
3. Full F8 temporal quality corpus and controls. No tolerance relaxation.
4. Repeated resize/teardown and long fixed-extent run: bounded live workers,
   mappings/FDs, parent memory and total active-worker memory; all PIDs reaped.
   Distinguish engine lifetime remediation from the still-failing SDK probe.
5. Warm, repeated complete-frame performance at matched resolution/input
   scale, with IPC/copy overhead reported. No claim of improvement from a
   constant-color microbenchmark or captured PNG timings.
6. Relevant graph split/async, GPU validation, unit/CI and native regressions.

If an integration or performance gate fails, retain the evidence and improve
or reject the experiment; do not mark F8.4 complete merely by hiding the SDK
probe or accepting an arbitrary number of surviving instances.

## Verification checkpoint (before CI acceptance)

Implementation now uses a bounded maximum of eight active/retiring child
processes, one scaler lifetime per view. Capacity exhaustion returns to
native rendering while startup retries; compile workers never wait for
future frame retirement. The broker receives jobs only after the producer
submission has been published. Shared mapping ownership extends through
actual buffer destruction. Host/headless-worker allocations are counted
together; SDK-internal allocations remain a separately reported domain.

Passed locally: full F7/F8 functional checks; 480-frame temporal quality
corpus and negative controls; exposure/DRS two-view clips; async/split
validation; malformed protocol and 35-second idle; worker crash/timeout;
parent death with no orphan children; paced output resize (16 children,
all reaped, peak four, mapped bytes zero); parent exit-time leaks zero;
437 portable tests and native F6 regression battery. Direct/isolated Sponza
641x361 at 75% input is pixel-exact. Native/Debug/Release builds pass.

Three rotated 1080p replicas (120 warmup +600 measured): Sponza 2.1667
vs 2.1757 ms; 1024 lights 11.6081 vs 12.4598 ms; two views 1.2656 vs
1.9145 ms. This is complete-frame offscreen throughput; renderer CPU
time excludes worker CPU. There is a measured cost, especially for
multiple views, and extra shared bridge memory. No OPT is activated.

Source CI Linux/macOS passes (run 37132919847). The final 20,000-frame
fixed-extent run also passes with zero new GpuMemory allocations in both
processes and zero residual mappings. Four-view EDR/DRS validation passes.
Publication is tracked by PR #15.
The SDK-only probe remains negative; adoption concerns the engine's
process-owned lifetime, not an assertion that the SDK cycle disappeared.
