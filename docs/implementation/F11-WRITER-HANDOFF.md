# F11 writer handoff — NON VERIFIED

Worktree: `/Users/danielsan/.codex/worktrees/f11-light-sampling/phosphor`.
Branch: `codex/f11-light-sampling`; base:
`7997f12713f07895bdf27d7ab27f7947f3da7bd7`.

No builds, configuration, CPU/GPU tests, MSL compilation, renderer, reference
generation, profiler, benchmark or noise generation have run. Source reading
and `git diff --check` are the only checks. No ROADMAP checkbox, push, PR or
merge was performed. No file in the active checkout was modified.

## Commit boundaries

| Commit | Scope |
| --- | --- |
| `4304675` | F11.1/F11.2 initial shared layouts, portable sampling, reservoir, alias/cluster reference |
| `fa6c9f1` | F11.4 original bounded scalar STBN generator, explicit white fallback |
| `455c351` | F11.2 textured-emitter layouts/CPU oracle, pixel motion, invalid-weight diagnostics, brute selector |
| `2caa24f` | F11.1/F10.3 GPU candidate→temporal→spatial→shading, own-PSO IFT contract, full emissive update, cluster fallback |
| subsequent F11 verification-contract commit | written CPU controls, detailed implementation/ABI contract and this handoff |

Pick the sequence from the base; the final range can be recovered with
`git log --oneline 7997f127..codex/f11-light-sampling`. Every commit is source
delivery, explicitly unverified. The first initial GPU layout is extended by
455c351; final GPUSampledLight remains **80 B**, not the abandoned 112 B draft.

## Files and integration

`src/renderer/light_sampling.{h,cpp}`: alias distribution, shape/texture
sampling and full/constant CPU reference, GPU-emissive-transform oracle and
conservative CPU cluster reference. `reservoir.h`: streaming/reuse, finite
guards, history compatibility/reprojection. `stochastic_sampling.{h,cpp}`:
original void-and-cluster same-z/same-xy STBN, versioned seed/config/ranks.
`gpu_types.h`: appended scalar shared layouts/flags only.

`shaders/restir_common.h`: public area/PBR/texture sampling and normalized
reservoir helpers shared with F12. `light_visibility.h`: world geometric
normal W&B offsets, any-hit RT_MASK_SHADOW, alpha LOD0. `restir_di.metal`:
separate candidate/temporal/spatial, unshadowed/RT shade. `light_cluster.metal`:
full-scene GPU emissive update, conservative cluster build and unshadowed/RT
cluster shade with full-light iteration on overflow or forced brute mode.
`tests/test_light_sampling.cpp`: written distribution/reference/negative
controls. `docs/implementation/F11-CONTRACT.md`: exact bindings/equations,
units/history/ownership, assumptions and unexecuted verification contract.

The root owns CMake, engine/Metal wrapper, argument tables, render graph,
scene extraction, material guides, CLI/report and existing shader changes.
Add both new `.cpp` files and the test to CMake; include both new MSL files
and their headers in the shader dependency list. Guard rt_alpha_generic in
rt_common.h with PHOSPHOR_RT_NO_ALPHA_FUNCTION for consumers other than its
linked definition. Source functions are complete, but this writer worktree
does not itself contain root host integration or a reachable F11 frame.

## Contracts needing root attention

- Actual textured receiving guides before resolve: GPUDISurface96 B carries
  depth/world point, GEOMETRIC normal for RT, shaded material normal/albedo,
  roughness/metallic/view direction and instance/material revisions.
- F11 common optional emitter buffers: **12 records, 13 materials, 14
  DITextureHandle table** in every DI kernel and cluster shade kernel.
  Dynamic `light_emissive_update`: params0, records1, instances2, materials3,
  source lights4, current per-slot lights5. It must follow scene transforms
  and precede candidate/cluster reads, with all accesses in the graph.
- Extract FULL emitter geometry, never camera culled/proxy geometry. GPU
  deformation requires metadata re-extraction/current local vertices; the
  implemented updater handles actual GPU instance transforms, not arbitrary
  changes to the local triangle copied into its record. Reset lightRevision
  for movement, emission/alpha changes and membership/domain changes.
- Every RT PSO owns/recreates its own IFT on hot reload. Both BLAS and typed
  TLAS are graph dependencies at Dispatch; respect F9 AS→Dispatch/lifetime.
- Candidate i uses5i..5i+4, temporal48, neighbor i uses49+3i..51+3i. Bounds
  are8 candidates/4 neighbors and64 generated STBN dimensions. Generation
  is LOADING/OFFLINE only. A white fallback must be reported as white.
- Reservoir pad[0] DI_ERROR_* causes invalid output and MUST fail GPU debug
  readback. Light revision/view/epoch and light ID/generation validate reuse.
  Temporal stores temporal output as history; spatial is a separate buffer.
- Cluster count retains TOTAL entries, overflow shades EVERY light. Params
  pad1!=0 forces the `Local brute force` full iteration pass. RT fallback
  tests visibility for each contributing endpoint; non-RT is unshadowed.
- Resolve replaces LOCAL direct contributions only. Sun, ambient, emissive
  and indirect terms remain separate. F13 owns the final denoised DI exit.

## TESTER commands — NOT EXECUTED

After root applies source registration and host integration in its worktree:

```sh
cmake -S . -B build/f11-check -G Ninja
cmake --build build/f11-check
./build/f11-check/tests/phosphor_tests --test-case="F11*"
ctest --test-dir build/f11-check --output-on-failure
```

Run C++/Metal syntax checks and actual MSL compilation; then the root's final
functional runner with one/eight/1024 lights, candidate-only/temporal/spatial,
independent CPU versus forced-brute GPU reference, area/textured-emissive and
MASK fixtures, wall occlusion, light removal, camera cut, resize, two views,
history/generation/PDF/M/NaN corruption controls, hotspot cluster overflow,
IFT hot reload and measured-frame allocation accounting. Freeze tolerances
and preset before comparison; do not infer unbiased GPU images from CPU
unit tests. FFT/correlation/periodicity validation of generated STBN masks
and visual sequences is still missing. Numeric presets are unmeasured.

Apple9 is the feature floor; M5 forced Apple9 does not certify physical M3.
No task or phase acceptance, denoised exit, timing or quality claim is made.
