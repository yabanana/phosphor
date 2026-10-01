# `bench/f5_spike` — F5 spikes S2–S7

Measurement tool for the F5 spikes (GPU scene, GPU-driven submission), **not
engine code**: same rules and harness as [`bench/soc`](../soc/README.md)
(objects through `soc::Context`, MSL compiled at run time from
`bench/f5_spike/shaders`, output only through `ctx.log()`, a negative control
per benchmark). Results and decisions: `docs/opt-log.md`, "F5 — Spike".
Spike S1 (CPU path baseline) and S7/F2.5 run in the engine itself.

```sh
cmake --build build/release --target f5_spike
./build/release/f5_spike --list
./build/release/f5_spike --only F5-S2 --runs 3 --out build/f5-s2.json
./build/release/f5_spike --only F5-S3 --validate          # under API + shader validation
./build/release/f5_spike --only F5-S3 --force-family apple9
```

| Id | File | Question |
|---|---|---|
| F5-S2 | `s2_delta.cpp` | delta updates: scatter from a ring into `private` vs direct CPU writes into per-frame `shared` copies vs full reload; cost of the cross-frame wait on a persistent buffer |
| F5-S3 | `s3_draws.cpp` | draw emission: GPU-encoded ICB (ranges per cull class) vs indirect draw per bucket vs direct draws; inherited argument table; ICB under validation across command buffers / queues |
| F5-S4 | `s4_cull.cpp` | instance culling of 1M + stable compaction vs atomics: time, determinism, fast math vs CPU reference |
| F5-S5 | `s5_barriers.cpp` | consumer stage that makes compute-written indirect arguments / ICB visible to draws and dispatches |
| F5-S6 | `s6_hierarchy.cpp` | transform hierarchy: dispatch per level vs walk to root vs CPU; dirty queues with indirect dispatch; bit-exactness |
| F5-S7 | `s7_async.cpp` | F2.6: culling on the second queue beside a raster load |
| F5-K2 | `k2_gpu_scene.cpp` | GPU check of the engine kernels `shaders/gpu_scene.metal` (queue clear, scatter, cull + stable scan, ICB draw build) against `renderer/cull_reference` on layouts up to 4M slots; built-in negative controls |
| F5-K3 | `k3_transforms.cpp` | GPU check of `shaders/transforms.metal` (motion, hierarchy queues with indirect dispatch, F5.4) against `renderer/transform_reference`, bit for bit; overflow and missing-barrier controls |

K2/K3 compile the engine's shaders from source through a small include
expander (the harness has no include paths).  GPU safety (learned the hard
way in F5): every kernel loop has a hard bound, no kernel waits on another
threadgroup, every command buffer stays far below 1 s, and only one GPU
test process runs at a time -- a 60 s job made the WindowServer watchdog
kill the compositor.
