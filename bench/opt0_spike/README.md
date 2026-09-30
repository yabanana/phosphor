# OPT-0 spikes (measurement probes, not engine code)

Results and decisions: `docs/opt-log.md`, section "OPT-0 — Spike di
caratterizzazione". These probes informed the design of the `bench/soc`
suite; they are kept as the record of how each number was obtained. Build
by hand from the repository root (metal-cpp from the CMake build tree):

```sh
M=build/_deps/metal_cpp-src
clang++ -std=c++20 -O2 -I $M bench/opt0_spike/method_spike.cpp -framework Metal -framework Foundation -framework CoreFoundation -lIOReport -o method_spike
clang++ -std=c++20 -O2 bench/opt0_spike/ioreport_probe.cpp -framework CoreFoundation -lIOReport -o ioreport_probe
clang++ -std=c++20 -O2 -I $M bench/opt0_spike/api_probe.cpp -framework Metal -framework Foundation -o api_probe
clang++ -std=c++20 -O2 -fobjc-arc bench/opt0_spike/coreml_probe.mm -framework CoreML -framework Foundation -o coreml_probe
clang++ -std=c++20 -O2 -fobjc-arc bench/opt0_spike/display_probe.mm -framework AppKit -framework Metal -framework QuartzCore -o display_probe
clang++ -std=c++20 -O2 -DACCELERATE_NEW_LAPACK -mcpu=native+sme2 bench/opt0_spike/cpu_probe.cpp -framework Accelerate -o cpu_probe
xcrun metal -std=metal4.0 -O2 -I src -I build/release/generated -S -emit-llvm shaders/forward.metal -o forward.ll
python3 bench/opt0_spike/air_ops.py forward.ll
```

| Probe | Cases |
|---|---|
| `method_spike` | `linear`, `overhead`, `dvfs [s]`, `intermittent`, `cv`, `groups`, `elim`, `chase`, `slc [MiB...]` (`ZEROS=1`: uninitialised private buffer, `AGENT=`: PMP histogram channel) |
| `ioreport_probe` | `list`, `sample [ms]`, `burn [ms]`, `group <group> [subgroup] [ms]` (`RAW=1`, `ALLSTATES=1`) |
| `api_probe` | `gemm [n]` (tensor ops vs simdgroup_matrix), `mtlio [MiB]` |
| `coreml_probe` | `[iterations]` (model built in code, MLComputePlan per layer) |
| `display_probe` | `[seconds] [fps] [preferredFrameLatency]` (opens a window) |
| `cpu_probe` | Accelerate SGEMM, NEON FMA, SME2 compile check |
