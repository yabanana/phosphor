# F13/F14 full-frame budget follow-up

The owner requires remaining F13 native correctness and the full frame budget
before advancing beyond F14. This is a bounded experiment on measured costs,
not activation or completion of the entire OPT catalog.

## Image equivalence

Requires the Release Metal build, NumPy and the existing licensed Sponza asset.
The default writes a frozen plan only. Use a fresh output directory every time:

```sh
python3 bench/f13_budget/quality.py --binary build/lighting/phosphor \
  --out build/budget-quality --run
```

Runs actual API/shader-validated frames serially under the shared GPU lock.
At noon, sunrise and night it compares the same binary's original atrous and
64-step aerial reference, the selected shared-memory stencil alone, and the
stencil plus adaptive short-distance aerial quadrature. It uses 322x182 to
exercise partial workgroups on Retina, 32 complete Float32 frames per case,
fixed timestep, all full lighting/volume features and dense DDGI16x8x16/64rays.
No resolution, tap, history, reflection-ray or GI-ray budget is lowered.

The stencil has a 5e-5 peak-normalized maximum error cap. The aerial comparison
has a 1% relative cap with denominator floor 0.001 of the reference image peak;
that floor is explicit and **does not apply to the native MetalFX wide-HDR gate**.
A single-pixel/different-frame failure, missing capture or nonfinite input fails.
All commands, diagnostic environment, immutable process status and pixels stay
in the output directory. This is equivalence to the existing renderer, not a
replacement for independent physical F13/F14 references.

`PHOSPHOR_DIAGNOSTIC_DENOISE_REFERENCE=1` selects the original spatial kernel.
`PHOSPHOR_DIAGNOSTIC_AERIAL_REFERENCE=1` selects the original quadrature budget.
Both are diagnostic controls, logged once at startup. Product defaults retain
Float32 and all the original 25 filter taps. The wider stride keeps the original
kernel after the measured SIMD/sublattice alternatives failed to improve it.

## Timing and lifetime

Use the exact full-frame command and artifacts in
`docs/implementation/F13-F14-BUDGET-CLOSURE-2026-10-08.md`. Run GPU workloads
one at a time, without validation, captures or builds; freeze the baseline
binary, shader library and pipeline archive together. Keep paired baseline
drift visible. Do not claim the 60fps gate from an isolated fastest frame.

The native lifetime runner remains `tools/metalfx_denoise_fixture_check.py`.
Its `--gateway native --only 'lifecycle*' --frames 300 --leaks-at-exit --run`
case covers four views, three sizes, pipeline reload and actual SDK frames.
`PHOSPHOR_METALFX_DENOISED_PLAIN_RELEASE=1` disables only the newly identified
POD timing-record retirement for the deliberate leaking negative control.
Native production admission remains disabled while HDR qualification fails.
