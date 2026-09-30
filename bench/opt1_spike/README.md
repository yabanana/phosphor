# OPT-1 spikes (measurement probes, not engine code)

Results and decisions: `docs/opt-log.md`, section "OPT-1 — Spike". The graph
scenarios themselves are engine code (`--graph-scenario`, see
`src/rendergraph/scenario.h`); this directory keeps the offline solver
comparison of spike 2.

`graph_solver.cpp` compares, on the scenario graphs with 1..6 views, the
greedy order of the F2 compiler, an exact dynamic program over downsets (beam
when the state cap is hit), simulated annealing over topological orders and
a MILP solved with HiGHS; every order is judged by the same evaluator (the
real graph compiler with `CompileOptions::order` plus a cost model).

```sh
# Release phosphor_core first: cmake --build build/release --target phosphor_core
git clone --depth 1 --branch v1.15.1 https://github.com/ERGO-Code/HiGHS build/_deps/highs-src
cmake -S build/_deps/highs-src -B build/_deps/highs-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF -DFAST_BUILD=ON -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
cmake --build build/_deps/highs-build
H=build/_deps/highs-src
clang++ -std=c++20 -O2 -DOPT1_WITH_HIGHS -I src -I build/release/generated -I $H/highs -I build/_deps/highs-build \
    bench/opt1_spike/graph_solver.cpp build/release/libphosphor_core.a build/_deps/highs-build/lib/libhighs.a -lz \
    -o graph_solver
# without HiGHS: drop -DOPT1_WITH_HIGHS and the HiGHS include/library arguments
./graph_solver <scenario 0..3> <views 1..6> [dp state cap=200000] [anneal iterations=20000] [milp seconds=60]
```

`MILP_CHECK=1` verifies that the greedy order satisfies every MILP row (a
debugging aid for the formulation).
