# `bench/soc` — SoC characterisation suite (OPT-0.1)

Microbenchmarks B-01…B-29 of [`docs/APPLE_SOC_PLAYBOOK.md`](../../docs/APPLE_SOC_PLAYBOOK.md)
§0. A measurement tool, **not engine code**: it creates resources with the
device and compiles MSL at run time (both forbidden in the engine). Results go
to `bench/results/<chip>-<os>.json` (schema: `src/diagnostics/soc_results.h`),
the cost model reads them (`src/diagnostics/soc_model.h`, `docs/soc-model.md`).
Method and the spikes behind it: `docs/opt-log.md`, "OPT-0".

```sh
cmake --build build/release --target soc_bench
./build/release/soc_bench --list
./build/release/soc_bench                    # full suite x 3 runs -> bench/results/m5max-macos27.2.json
./build/release/soc_bench --quick --runs 1   # <= 2 minutes, -> ...-quick.json
./build/release/soc_bench --only B-01,B-08 --runs 1 --out /tmp/x.json
./build/release/soc_bench --validate         # quick suite under API + shader validation, 0 messages expected
./build/release/soc_bench --force-family apple9   # Apple9 fallback paths on an Apple10 GPU
./build/release/soc_bench --only B-26 --window    # benchmarks that open a window
./build/release/soc_bench --only B-27 --soak 10   # thermal soak (minutes)
```

Exit status: 0 ok; 1 a benchmark failed or a negative control failed; 2 usage;
3 `--validate` found messages. `SOC_SHADER_DIR` overrides the shader directory
(used for negative controls on a modified copy of the shaders).

## Protocol (every benchmark)

- Time on the GPU with `ComputeTimer` (anchor dispatch + Precise timestamps,
  compute only) or `CommandTimer` (anchor encoder → your encoders → tail
  encoder; subtract `emptySpanMs()` for tiny work).
- Single measured intervals of ~0.1–2 ms; `Context::measure()` (≥15
  repetitions, median; re-warms the GPU after an idle gap); `keepWarm()`
  between groups; the suite runs `warmUp()` before each run and records the
  P-state residency of every benchmark (`gpu.top_state_share`).
- Every kernel stores a result that the CPU checks; inputs are random
  (`randomBuffer`), thread- and chain-dependent (identical or uniform work is
  merged or run once per SIMD-group by the compiler: measured in B-01).
- Every benchmark sets a negative control (`Report::negative`): linearity
  (2× work → 2× time), a variant that must be slower/faster, a physical
  plausibility bound, exact results.
- `ctx.quick()`: the whole suite ≤ 2 minutes. `ctx.apple10()`: false on
  Apple9 or with `--force-family apple9` → take the fallback, say so in the
  notes (`Report::status(Status::Partial, …)` when a metric is missing).
- Output only through `ctx.log()` (`[soc]` prefix): `--validate` counts every
  other line as a validation message.
- Objects only through `Context` (released and removed from the residency set
  at the end of the benchmark; `leaks --atExit` must report 0).

## Files

| File | Content |
|---|---|
| `harness.{h,cpp}` | Context, timers, IOReport GPU state, measure/warm-up, registry, machine description |
| `soc_bench.cpp` | CLI, runs, merge, JSON, `--validate` |
| `bNN_*.cpp` / `.mm` | one benchmark group per file, `SOC_BENCH(...)` |
| `b29_upload.cpp` | B-29 upload storage modes: CPU write/read of write-combined vs cached shared buffers, GPU read of shared vs private (reuses `b08_read` of `shaders/b08_memory.metal`) |
| `shaders/*.metal` | MSL compiled at run time (`harness.metal`: anchor and warm-up kernels) |
