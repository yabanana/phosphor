# barrier_spike (F2.3)

Measurement tool for the F2.3 render-graph barrier design.  It answers, on the
real device, which Metal 4 barrier stage pairs are accepted, which ones are
actually needed, and what the cost is.  It is not engine code: it creates
resources directly with `MTL::Device` and compiles its MSL at run time.

Measured on: 2026-09-29, Apple M5 Max (40 GPU cores, Metal 4), macOS 27.2
(build 26B5091g), Debug build, one process per case/variant.

## Usage

```
cmake --build build/spike --target barrier_spike            # see the task's configure line
build/spike/barrier_spike --list                            # all case/variant pairs
build/spike/barrier_spike <case> <variant>                  # one case, one process
build/spike/barrier_spike --reps=20 --iters=3000 --runs=7 <case> <variant>
bench/barrier_spike/run_all.sh [binary] [out_dir] [regex]   # every pair, with and without validation
```

One line is printed:
`SPIKE <case> <variant> result=<ok|WRONG|ERROR> gpu_ms=<x> wrong_reps=<n>/<reps> bad_elems=<n> expected=<hex> got=<hex>`
(`ERROR` = the command buffer reported an error).  Offscreen only; results are
verified on the CPU.  `run_all.sh` runs each pair twice, with
`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER_WARNING_MODE=nslog`
and without them, and prints the Markdown table below.

Variant names: `none`, or `<kind>_<after>_<before>[_<alias|both|none>]` where
kind is `q` = `barrierAfterQueueStages` at the start of the consumer encoder,
`p` = `barrierAfterStages` at the end of the producer encoder, `e` =
`barrierAfterEncoderStages` inside one encoder.  Stages are joined with `+`.
The visibility option defaults to `VisibilityOptionDevice`; `alias` =
`VisibilityOptionResourceAlias`, `both` = both, `none` = `VisibilityOptionNone`.

## Method

Each case encodes 20 independent repetitions (own resources) in one command
buffer.  The producer is slow on purpose (a 3000-iteration dependent ALU loop
per element over 4M elements or a 2048x2048 target, about 9 ms per producer);
its value is `val(i, salt)` (non-zero, exact in float32), unknowable to the
compiler because the loop result is masked with a run-time zero.  Consumers
copy what they read into a shared buffer that the CPU checks against the
expected value; `wrong_reps` counts repetitions with any wrong element.  A
correct-barrier variant that reports WRONG would be a bug in the spike (this
happened once, see git history: mirrored reads in `intra_*` were compared
unmirrored).  Barrier-cost cases chain 1000 tiny dependent dispatches
(`buf[i] += 1`, so a missing barrier loses updates) or 200 render->render
pairs, and take the median GPU time (`MTL4::CommitFeedback`) of 7 runs.

Cases: `compute_to_vertex`, `compute_to_fragment`, `render_to_compute`,
`render_to_render` (Depth32Float store then sample), `blit_to_fragment`,
`intra_render`, `intra_compute`, `alias` (two buffers at offset 0 of a
placement heap), `legality` (API acceptance probes, no data flow),
`cost_queue`, `cost_encoder`, `cost_render`.

Caveats when reading the table:

* `none` variants are races: how many repetitions go wrong varies from run to
  run (e.g. `cost_render none` 1/1400 in one run and 639/1400 in another, with
  identical code).  Only "not zero" is meaningful; a race that happened not to
  fire proves nothing.
* GPU ms comes from the run without validation.
* `cost_*` rows: `none` is faster partly because it overlaps work that has to
  be serialised (and it is wrong), so the difference is barrier plus lost
  overlap, not a pure barrier price.

## Results

Two full runs of `run_all.sh` were made; the wrong-repetition counts of the
correct-barrier variants and all validation messages are identical between
them.  Only the racy `none` counts differ (`blit_to_fragment none`,
`cost_queue none`, `cost_render none`).  Run A:

| case | variant | validation messages | wrong reps (validation) | wrong reps (no validation) | GPU ms |
|---|---|---|---|---|---|
| compute_to_vertex | none | none | 20/20 | 20/20 | 180.939 |
| compute_to_vertex | q_dispatch_vertex | none | 0/20 | 0/20 | 176.439 |
| compute_to_vertex | q_dispatch_fragment | none | 20/20 | 20/20 | 181.414 |
| compute_to_vertex | q_dispatch_vertex+fragment | none | 0/20 | 0/20 | 178.563 |
| compute_to_vertex | p_dispatch_vertex | none | 0/20 | 0/20 | 180.573 |
| compute_to_fragment | none | none | 20/20 | 20/20 | 180.332 |
| compute_to_fragment | q_dispatch_fragment | none | 0/20 | 0/20 | 179.282 |
| compute_to_fragment | q_dispatch_vertex | none | 0/20 | 0/20 | 179.920 |
| compute_to_fragment | q_dispatch_vertex+fragment | none | 0/20 | 0/20 | 179.821 |
| compute_to_fragment | q_dispatch_tile | none | 20/20 | 20/20 | 180.076 |
| compute_to_fragment | p_dispatch_fragment | none | 0/20 | 0/20 | 175.406 |
| compute_to_fragment | q_dispatch_fragment_alias | none | 0/20 | 0/20 | 179.614 |
| compute_to_fragment | q_dispatch_fragment_none | none | 0/20 | 0/20 | 180.861 |
| render_to_compute | none | none | 20/20 | 20/20 | 181.663 |
| render_to_compute | q_fragment_dispatch | none | 0/20 | 0/20 | 179.126 |
| render_to_compute | q_vertex_dispatch | none | 20/20 | 20/20 | 186.631 |
| render_to_compute | p_fragment_dispatch | none | 0/20 | 0/20 | 182.953 |
| render_to_compute | q_tile_dispatch | none | 20/20 | 20/20 | 182.618 |
| render_to_render | none | none | 20/20 | 20/20 | 187.064 |
| render_to_render | q_fragment_fragment | none | 0/20 | 0/20 | 183.604 |
| render_to_render | q_fragment_vertex | none | 0/20 | 0/20 | 183.728 |
| render_to_render | q_fragment_vertex+fragment | none | 0/20 | 0/20 | 184.788 |
| render_to_render | q_fragment_tile | none | 20/20 | 20/20 | 184.493 |
| render_to_render | q_vertex_vertex | none | 0/20 | 0/20 | 184.579 |
| render_to_render | q_vertex_fragment | none | 0/20 | 0/20 | 187.087 |
| render_to_render | p_fragment_vertex | none | 0/20 | 0/20 | 190.097 |
| blit_to_fragment | none | none | 17/20 | 17/20 | 29.732 |
| blit_to_fragment | q_blit_fragment | none | 0/20 | 0/20 | 30.349 |
| blit_to_fragment | q_blit_vertex | none | 0/20 | 0/20 | 31.131 |
| blit_to_fragment | p_blit_fragment | none | 0/20 | 0/20 | 31.232 |
| blit_to_fragment | p_blit_vertex | none | 0/20 | 0/20 | 30.328 |
| intra_render | none_ff | none | 20/20 | 20/20 | 182.529 |
| intra_render | none_fv | none | 20/20 | 20/20 | 183.653 |
| intra_render | none_vf | none | 0/20 | 0/20 | 231.916 |
| intra_render | e_fragment_fragment | crash (rc=134): `-[MTL4DebugCommandEncoder barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:]:213: failed assertion `Command Encoder Barrier Validation / afterEncoderStages must be a valid combination of MTLStages for the current encoder type (MTLStageVertex \| MTLStageObject \| MTLStageMesh).` | - | 20/20 | 182.726 |
| intra_render | e_fragment_vertex | crash (rc=134): `-[MTL4DebugCommandEncoder barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:]:213: failed assertion `Command Encoder Barrier Validation / afterEncoderStages must be a valid combination of MTLStages for the current encoder type (MTLStageVertex \| MTLStageObject \| MTLStageMesh).` | - | 20/20 | 182.674 |
| intra_render | e_vertex_fragment | none | 0/20 | 0/20 | 231.765 |
| intra_render | e_fragment_tile | crash (rc=134): `-[MTL4DebugCommandEncoder barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:]:213: failed assertion `Command Encoder Barrier Validation / afterEncoderStages must be a valid combination of MTLStages for the current encoder type (MTLStageVertex \| MTLStageObject \| MTLStageMesh).` | - | 20/20 | 181.354 |
| intra_render | e_vertex_vertex+fragment | none | 0/20 | 0/20 | 232.648 |
| intra_render | e_vertex_tile | none | 0/20 | 0/20 | 231.112 |
| intra_compute | none | none | 20/20 | 20/20 | 179.752 |
| intra_compute | e_dispatch_dispatch | none | 0/20 | 0/20 | 180.029 |
| alias | none | none | 20/20 | 20/20 | 179.096 |
| alias | q_dispatch_fragment | none | 0/20 | 0/20 | 180.198 |
| alias | q_dispatch_fragment_alias | none | 0/20 | 0/20 | 177.471 |
| alias | q_dispatch_fragment_both | none | 0/20 | 0/20 | 176.930 |
| alias | q_dispatch_vertex | none | 0/20 | 0/20 | 181.402 |
| alias | q_dispatch_vertex_alias | none | 0/20 | 0/20 | 180.062 |
| alias | q_dispatch_vertex_both | none | 0/20 | 0/20 | 180.115 |
| alias | q_dispatch_fragment_none | none | 0/20 | 0/20 | 182.616 |
| alias | q_dispatch_vertex_none | none | 0/20 | 0/20 | 178.053 |
| legality | render_q_dispatch_vertex | none | 0/1 | 0/1 | 0.883 |
| legality | render_q_dispatch_fragment | none | 0/1 | 0/1 | 0.033 |
| legality | render_q_dispatch_tile | none | 0/1 | 0/1 | 0.027 |
| legality | render_q_dispatch_dispatch | none | 0/1 | 0/1 | 0.033 |
| legality | render_q_dispatch_blit | none | 0/1 | 0/1 | 0.027 |
| legality | render_q_dispatch_object | none | 0/1 | 0/1 | 0.032 |
| legality | render_q_dispatch_mesh | none | 0/1 | 0/1 | 0.028 |
| legality | render_q_fragment_tile | none | 0/1 | 0/1 | 0.033 |
| legality | render_q_fragment_fragment | none | 0/1 | 0/1 | 0.026 |
| legality | render_q_vertex_fragment | none | 0/1 | 0/1 | 0.038 |
| legality | render_q_tile_fragment | none | 0/1 | 0/1 | 0.032 |
| legality | render_q_tile_vertex | none | 0/1 | 0/1 | 0.033 |
| legality | render_q_blit_fragment | none | 0/1 | 0/1 | 0.026 |
| legality | render_q_all_all | none | 0/1 | 0/1 | 0.030 |
| legality | render_q_fragment_all | none | 0/1 | 0/1 | 0.030 |
| legality | render_p_fragment_dispatch | none | 0/1 | 0/1 | 0.037 |
| legality | render_p_fragment_fragment | none | 0/1 | 0/1 | 0.034 |
| legality | render_p_vertex_vertex | none | 0/1 | 0/1 | 0.028 |
| legality | render_p_fragment_vertex | none | 0/1 | 0/1 | 0.034 |
| legality | render_p_tile_dispatch | none | 0/1 | 0/1 | 0.028 |
| legality | render_p_fragment_tile | none | 0/1 | 0/1 | 0.033 |
| legality | render_p_fragment_all | none | 0/1 | 0/1 | 0.032 |
| legality | render_e_vertex_vertex | none | 0/1 | 0/1 | 0.035 |
| legality | render_e_vertex_fragment | none | 0/1 | 0/1 | 0.033 |
| legality | render_e_vertex_tile | none | 0/1 | 0/1 | 0.030 |
| legality | render_e_vertex_dispatch | crash (rc=134): `-[MTL4DebugCommandEncoder barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:]:213: failed assertion `Command Encoder Barrier Validation / beforeEncoderStages must be a valid combination of MTLStages for the current encoder type (MTLStageVertex \| MTLStageFragment \| MTLStageTile \| MTLStageObject \| MTLStageMesh).` | - | 0/1 | 0.036 |
| legality | render_e_vertex_blit | crash (rc=134): `-[MTL4DebugCommandEncoder barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:]:213: failed assertion `Command Encoder Barrier Validation / beforeEncoderStages must be a valid combination of MTLStages for the current encoder type (MTLStageVertex \| MTLStageFragment \| MTLStageTile \| MTLStageObject \| MTLStageMesh).` | - | 0/1 | 0.878 |
| legality | render_e_fragment_fragment | crash (rc=134): `-[MTL4DebugCommandEncoder barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:]:213: failed assertion `Command Encoder Barrier Validation / afterEncoderStages must be a valid combination of MTLStages for the current encoder type (MTLStageVertex \| MTLStageObject \| MTLStageMesh).` | - | 0/1 | 0.026 |
| legality | render_e_tile_tile | crash (rc=134): `-[MTL4DebugCommandEncoder barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:]:213: failed assertion `Command Encoder Barrier Validation / afterEncoderStages must be a valid combination of MTLStages for the current encoder type (MTLStageVertex \| MTLStageObject \| MTLStageMesh).` | - | 0/1 | 0.027 |
| legality | render_e_tile_fragment | crash (rc=134): `-[MTL4DebugCommandEncoder barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:]:213: failed assertion `Command Encoder Barrier Validation / afterEncoderStages must be a valid combination of MTLStages for the current encoder type (MTLStageVertex \| MTLStageObject \| MTLStageMesh).` | - | 0/1 | 0.026 |
| legality | render_e_object_vertex | none | 0/1 | 0/1 | 0.028 |
| legality | render_e_mesh_fragment | none | 0/1 | 0/1 | 0.025 |
| legality | compute_q_fragment_dispatch | none | 0/1 | 0/1 | 0.011 |
| legality | compute_q_dispatch_dispatch | none | 0/1 | 0/1 | 0.008 |
| legality | compute_q_blit_dispatch | none | 0/1 | 0/1 | 0.011 |
| legality | compute_q_dispatch_blit | none | 0/1 | 0/1 | 0.011 |
| legality | compute_q_tile_dispatch | none | 0/1 | 0/1 | 0.009 |
| legality | compute_q_vertex_dispatch | none | 0/1 | 0/1 | 0.011 |
| legality | compute_q_dispatch_fragment | none | 0/1 | 0/1 | 0.011 |
| legality | compute_q_dispatch_vertex | none | 0/1 | 0/1 | 0.011 |
| legality | compute_q_all_all | none | 0/1 | 0/1 | 0.012 |
| legality | compute_p_dispatch_dispatch | none | 0/1 | 0/1 | 0.208 |
| legality | compute_p_dispatch_fragment | none | 0/1 | 0/1 | 0.009 |
| legality | compute_p_dispatch_vertex | none | 0/1 | 0/1 | 0.009 |
| legality | compute_p_blit_dispatch | none | 0/1 | 0/1 | 0.009 |
| legality | compute_e_dispatch_dispatch | none | 0/1 | 0/1 | 0.016 |
| legality | compute_e_blit_dispatch | none | 0/1 | 0/1 | 0.012 |
| legality | compute_e_dispatch_blit | none | 0/1 | 0/1 | 0.015 |
| legality | compute_e_blit_blit | none | 0/1 | 0/1 | 0.012 |
| legality | compute_e_dispatch_fragment | crash (rc=134): `-[MTL4DebugCommandEncoder barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:]:213: failed assertion `Command Encoder Barrier Validation / beforeEncoderStages must be a valid combination of MTLStages for the current encoder type (MTLStageDispatch \| MTLStageBlit \| MTLStageAccelerationStructure).` | - | 0/1 | 0.015 |
| legality | compute_e_fragment_dispatch | crash (rc=134): `-[MTL4DebugCommandEncoder barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:]:213: failed assertion `Command Encoder Barrier Validation / afterEncoderStages must be a valid combination of MTLStages for the current encoder type (MTLStageDispatch \| MTLStageBlit \| MTLStageAccelerationStructure).` | - | 0/1 | 0.012 |
| legality | compute_e_dispatch_vertex | crash (rc=134): `-[MTL4DebugCommandEncoder barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:]:213: failed assertion `Command Encoder Barrier Validation / beforeEncoderStages must be a valid combination of MTLStages for the current encoder type (MTLStageDispatch \| MTLStageBlit \| MTLStageAccelerationStructure).` | - | 0/1 | 0.012 |
| cost_queue | none | none | 7/7 | 6/7 | 7.920 |
| cost_queue | q_dispatch_dispatch | none | 0/7 | 0/7 | 13.854 |
| cost_encoder | none | none | 7/7 | 7/7 | 0.435 |
| cost_encoder | e_dispatch_dispatch | none | 0/7 | 0/7 | 1.466 |
| cost_render | none | none | 1099/1400 | 1/1400 | 7.200 |
| cost_render | q_fragment_vertex | none | 0/1400 | 0/1400 | 11.678 |
| cost_render | q_fragment_fragment | none | 0/1400 | 0/1400 | 10.112 |
| cost_render | q_fragment_vertex+fragment | none | 0/1400 | 0/1400 | 13.349 |

## Findings

Validation (with `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`) produced **no
message at all for any queue-level barrier** (`barrierAfterQueueStages`) or
producer-side barrier (`barrierAfterStages`), whatever the stages and encoder
type, including nonsensical ones (`render_q_dispatch_mesh`, `render_q_all_all`,
`compute_q_dispatch_fragment`).  Only `barrierAfterEncoderStages` is
validated, and it aborts (`failed assertion`, rc 134) with exactly:

* render encoder, `afterEncoderStages` containing Fragment or Tile:
  `-[MTL4DebugCommandEncoder barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:]:213: failed assertion `Command Encoder Barrier Validation / afterEncoderStages must be a valid combination of MTLStages for the current encoder type (MTLStageVertex | MTLStageObject | MTLStageMesh).``
* render encoder, `beforeEncoderStages` = Dispatch or Blit: `... beforeEncoderStages must be a valid combination of MTLStages for the current encoder type (MTLStageVertex | MTLStageFragment | MTLStageTile | MTLStageObject | MTLStageMesh).`
* compute encoder, any stage other than Dispatch/Blit/AccelerationStructure on
  either side: `... (MTLStageDispatch | MTLStageBlit | MTLStageAccelerationStructure).`

Without validation those illegal encoder barriers are silently accepted and
have no effect (`intra_render e_fragment_fragment` and `e_fragment_vertex`:
20/20 wrong, same as `none`).

Effect on hazards (no validation, the table's last-but-one column):

* A queue barrier's `before` set matters exactly: `compute_to_vertex` with
  before=Fragment is 20/20 wrong, `compute_to_fragment` with before=Fragment is
  correct.  Fragment on the consumer side of `barrierAfterQueueStages` in a
  render encoder is accepted and effective on this device.
* before=Vertex also protects the fragment reads of the same pass (fragment
  work follows vertex work), so Vertex is the conservative choice.
* Tile as `before` (`q_dispatch_tile`, `q_fragment_tile`) or as `after`
  (`q_tile_dispatch`) is accepted but gives no protection (20/20 wrong).
* `render_to_compute`: after=Fragment is required; after=Vertex or Tile is 20/20 wrong.
  Producer-side `barrierAfterStages(Fragment, Dispatch)` works too.
* `render_to_render` (depth store then sample) works with after=Fragment and
  before=Fragment, Vertex or Vertex|Fragment; after=Vertex also passed here
  (not guaranteed by the API; not relied on).
* Blit: after=Blit with before=Fragment or Vertex works.
* Inside one render encoder: a fragment write consumed by a later draw (in its
  fragment or vertex shader) cannot be fixed by an encoder barrier (illegal
  `afterEncoderStages`, see above): split the pass.  A vertex-stage write is
  correct even with no barrier when read in the fragment stage, and
  `e_vertex_fragment`, `e_vertex_vertex+fragment`, `e_vertex_tile` are legal.
* Inside one compute encoder, `e_dispatch_dispatch` is required (`none` races).
* Aliasing (`alias`): a barrier is required; Device, ResourceAlias, both and
  even `VisibilityOptionNone` all gave 0/20 wrong here, so the option did not
  change the outcome on this device (keep ResourceAlias for correctness on
  other hardware; this run cannot show its effect).

Cost (median of runs, GPU time; see caveats):

* `cost_encoder`: 1000 dispatches, 0.40 ms without, 1.36 ms with an encoder
  barrier between each, about 1 us per barrier.
* `cost_queue`: 1000 encoders, 8.2 ms without, 14.2 ms with a queue barrier
  at the start of each, about 6 us per barrier.
* `cost_render`, 200 render->render pairs, 25-run medians, stable over three
  repetitions: none 7.2 ms, before=Fragment 10.0 ms, before=Vertex 11.9 ms,
  before=Vertex|Fragment 12.0 ms.  Waiting on the fragment side is about 16 %
  cheaper than on the vertex side because the consumer's vertex work can
  overlap the producer's fragment work.
