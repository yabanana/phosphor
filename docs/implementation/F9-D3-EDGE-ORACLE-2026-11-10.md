# F9 — D3 edge rule in the CPU ray oracle (2026-11-10)

Scope: F9.4 functional correctness on M5 Max (`DEVELOPMENT_ACCEPTED` scope per
`docs/plans/HARDWARE_VALIDATION.md`). This changes the oracle only; no shader,
traversal or tolerance changed. Base: `codex/f13-f14-dynamic-closure` 03d2874.

## Why

Both bench8 cases failed at frame 48, ray 64 (58/60). The exact-arithmetic
analysis of the frozen float32 inputs (one triangle, its instance transform,
ray origin and direction) gives `u = 3.3e-9`: the ray passes 0.056 float32 ULP
from the `u = 0` edge. Perturbing origin, direction or instance translation by
±1 ULP flips the exact classification, so neither "hit" nor "miss" is
determined at input precision. A strict hit/miss oracle is over-specified
there. The owner chose D3: for rays within float32 rounding of an edge, accept
either the hit or the exact result without that triangle.

## Rule

`RtReference::check` admits a GPU miss, or a farther GPU hit on another
triangle, only when both hold:

1. The expected hit point is within the FP32 budget `gamma_8` of an edge of its
   own triangle. The signed edge distances are measured in the triangle's
   plane, and the budget comes from the world-space vertex, origin and
   `t·direction` magnitudes. The budget is also capped by the existing
   distance tolerance.
2. The GPU result matches the exact CPU result recomputed without that
   triangle, under the unchanged distance and barycentric tolerances. The
   recursion cannot exclude a second triangle.

`RtChecker` counts these rays (`edgeExclusions`) and fails a frame when they
exceed `max(1, checked / 1000)`. The engine prints
`edge-excluded N` on every RT check line and
`edge-excluded rays N of M` on the final line. `tools/f9_check.py` parsing is
unchanged.

## Verification

- Unit tests (`tests/test_rt_reference.cpp`, frozen witness):
  - the GPU miss is admitted;
  - the same ray moved 0.005 toward vertex B (1.1 mm inside) is refused;
  - with a large square 5 m behind the triangle, the miss is refused and the
    hit on the square is admitted;
  - the inside ray with the square behind is refused;
  - the checker cap passes at 1 exclusion of 2 rays and fails at 2 of 2.
- Negative control: forcing the edge test to true makes the interior cases fail
  (then restored).
- `ctest`: 9/9 pass.
- F9 matrix: 60 cases, one `tools/f9_check.py --only <case>` per case under
  `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`. The driver is the same as for
  the 58/60 run (`build/f13-f14-f9-refinement/matrix.py` in the development
  worktree), and the Metal toolchain is the one that built the measured
  binary.
  - Result: **60/60**. The protocol records binary sha256 `6d895c10…` and
    shader sha256 `ec8b1ae5…`, with thresholds unchanged.
  - The first attempt passed 38/60 because the new worktree lacked the
    `assets/sponza` link. The 22 affected cases were rerun after the link was
    added.

| Result | Rays checked | Edge-excluded |
|---|---:|---:|
| all 60 case logs | 2,808,814 | 2 |
| native-bench8, frame 48 | 512 | 1 |
| apple9-bench8, frame 48 | 512 | 1 |

The rule fired only on the known witness and on nothing else in the matrix.

## Not covered

This is the scope of the F9 functional runner manifest. The following are not
covered: the TLAS ≤0.5 ms gate and the refit study, A/B/A throughput, proxy
error metrics, the F5–F8 regression batteries and CI, and physical Apple9/M3
certification (`EXTERNAL_VALIDATION_PENDING`).
