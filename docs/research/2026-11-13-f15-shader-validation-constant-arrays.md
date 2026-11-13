# MTL_SHADER_VALIDATION and program-scope constant arrays in fragment functions

Date: 2026-11-13 (F15 ticket 03). Device: M5 Max, macOS 27.3. Status: tooling
defect worked around; minimal standalone reproducer not written.

## Observation

After ticket 03 moved the environment specular of the base model from the Karis
fit to the engine's DFG table (`renderer/dfg_table.h`, a program-scope
`constant float DfgTable[8192]` in MSL), `tools/visual_check.sh` (API + shader
validation) showed bench 1's gold torus black except for its highlights, while
the same binary without `MTL_SHADER_VALIDATION` rendered it correctly. No
validation message was printed.

## Evidence (bench 1, `--warmup 30 --frames 1 --fixed-timestep --pipeline-sync --offscreen`)

| Variant | plain vs `MTL_SHADER_VALIDATION=1` |
|---|---|
| Table read from the program-scope array (64×64) | 4,437,203 / 5,760,000 px differ, max delta 155 |
| Same with a 16×16 table (2 KB) | identical failure: size is not the cause |
| Karis fit instead of the table (negative control) | 0 px |
| With / without the pipeline archive (no validation) | 0 px: the archive is not involved |
| Table read through a buffer referenced by `FrameConstants::dfgTable` | 0 px |

Compute kernels are not affected the same way: the deterministic furnace
(`--material-scenario furnace-resolve`, visibility resolve in compute) differs
in 264 of 230,400 pixels by less than 1e-6 under shader validation, the usual
codegen-level difference.

Indices are in range (largest 8191 of 8192) and the inputs are NaN-safe, so the
read itself is valid MSL.

## Decision

Fragment functions read the DFG table through a GPU buffer whose address is in
`FrameConstants::dfgTable` (shared buffer owned by `SceneRenderer`, written once).
Compute kernels without the frame (`reflection_composite`, `reflection_probes`)
keep the program-scope array. One lookup implementation
(`dfgTableLookup(table, nv, roughness, component)`) serves both copies.

Re-check after OS/Xcode updates: run bench 1 with and without
`MTL_SHADER_VALIDATION=1` after temporarily pointing `bsdfEnvSpecular` at
`phosphor::DfgTable`.
