# F13 MetalFX writing-agent handoff — NON VERIFIED

Worktree: /Users/danielsan/.codex/worktrees/f13-metalfx-adapter/phosphor.
Branch: codex/f13-metalfx-adapter. Base4b80e36.

Separate backportable F12 reviewer corrections precede F13:

- fc1ea30 — mirrored material side versus unchanged F9 world facing.
- 57a575e — eligible zero-proposal history contributes bounded M.
- b1c5ef2 — radiometric reset preserves probe geometry; unapplied host epoch patch.
- e93e1b6 — full ray-role export/strict unsupported comparison, standalone
  triangle geometry and area-emission MASK.

F13 package:

- 3d5611f — portable channel/extents/request identity contract, GPU layout and
  written independent CPU negatives.
- eb6871a — new metalfx_denoise.h/cpp and denoise_pack.metal, exact active
  packing/depth crop/External graph/per-view resources/lifetime/fallback/readbacks.
- 2d82d5d — complete SDK contract and unapplied PipelineCache gateway artifact.
- Final follow-up avoids allocating packed targets from a late pending future
  during shutdown; the result retires unused instead.

See F13-METALFX-CONTRACT.md for actual local SDK signatures, channel formats/
spaces/units, binding indices, graph order, lifecycle and tester matrix.
This handoff is source-only; no compiler/test/GPU/reference/performance proof.

Root must add source/tests/shader dependencies, pipeline harvest, CLI/report,
raw-versus-custom composition and downstream Post selection. The SDK output is
already output-sized/dejittered and must bypass standard F8 temporal processing.
Root must drainPackChecks after final waitIdle and after resize/switch paths.
Only the tester reconciles F9 and applies/reviews the optional denoised gateway.

PipelineCache is byte-for-byte untouched here. git apply --check of
docs/patches/f13-denoised-factory.patch passed; the patch was NOT applied.
The adapter's callbacks are empty by default, so requested native denoise
reports MissingFactory and returns the caller's custom composite. Older/
unavailable SDK and unsupported devices also have explicit custom reasons.

DENOISED lifetime is not verified. Standard typed shared ownership and deferred
utility retirement only; no TemporalScaler cast, F8 private-cycle workaround,
compiler creation outside PipelineCache, OS change or dependency installation.
No roadmap tick, phase acceptance, merge, push or PR was performed.
