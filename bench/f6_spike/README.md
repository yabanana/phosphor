# bench/f6_spike — F6 spikes S1–S4, measurement tool, not engine code

Results live in `bench/results/f6-spike/`. Spikes only inform the choices a phase
plan leaves open (`docs/plans/`); a spike result never replaces the measurement
on the engine's own workloads. Every number below was measured, none is assumed.

## S1 — cook

CPU part of the meshlet cook comparison (F6.1). It measures cook time and
cook statistics only; the GPU frame cost of each option set (mesh-shader
throughput, culling efficiency) is the GPU part of S1/S2 and is **not** measured
here, so this section does not pick a cook.

Command (Release, CPU only, no GPU; the builder's log line goes to stderr):

```bash
cmake --build build-agent-rel --target meshlet_cook
./build-agent-rel/meshlet_cook --runs 5 --json bench/results/f6-spike/s1-cook-m5max-macos27.2.json 2>/dev/null
```

Results JSON: `bench/results/f6-spike/s1-cook-m5max-macos27.2.json` (manifest: git
commit, meshoptimizer 1.3, compiler, build type, date, machine, runs). Source:
`tools/meshlet_cook/meshlet_cook.cpp`. Cook ms is the median / min of 5 runs of
`MeshletBuilder::build` only (single thread, M5 Max, macOS 27.2); stats come from
`MeshletBuilder::stats`.

Corpus (all procedural, deterministic):

| mesh | parameters | triangles | vertices |
|---|---|---|---|
| sphere | `generateSphere(1, 256, 128)` | 65536 | 33153 |
| torus | `generateTorus(1, 0.35, 256, 128)` | 65536 | 33153 |
| icosphere | `generateIcosahedron(1, 6, smooth)` | 81920 | 41161 |
| plane | `generatePlane(100, 100, 512, 512)` | 524288 | 263169 |
| building | cube, 32x32 quads per face, welded | 12288 | 6146 |
| slivers | `generatePlane(2000, 1, 1000, 100)`: 2 x 0.01 cells (edge aspect > 200) | 200000 | 101101 |
| soup | 100k random triangles, seed 12345, edge ~0.03, unwelded | 100000 | 300000 |
| sponza | not available (`assets/README.md` lists the source; no download in this phase) | - | - |

Option sets (max vertices / max triangles): standard (`meshopt_buildMeshlets`,
cone weight 0.5 unless noted) 64/124 (baseline), 64/64, 64/96, 64/128, 96/128,
128/128, 64/124 with cone weight 0.0 and 1.0; spatial (`meshopt_buildMeshletsSpatial`,
fill weight 0.5, min triangles = max/4 rounded up to a multiple of 4) 64/124,
64/96, 64/128; standard 64/124 with `optimize` (`meshopt_optimizeMeshlet`).

Columns: `tri/ml`, `vert/ml` average per meshlet; `fill` = tri/ml / max triangles;
`dup` = meshlet vertex entries / unique mesh vertices; `B/tri` = (meshlets +
bounds + vertex refs + triangle bytes) / triangles; `cone usable` = fraction of
meshlets with a non-zero axis and cutoff < 1; `cone cutoff` = mean cutoff of those
(lower = wider rejection range, 0 = all triangles coplanar); `tri p10/p50/p90` =
nearest-rank quantiles of triangles per meshlet.

### Results

### sphere -- generateSphere(radius 1, 256 slices x 128 stacks)

| option set | cook ms (med/min) | tris | vertices | meshlets | tri/ml | vert/ml | fill | dup | B/tri | cone usable | cone cutoff | tri p10/p50/p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard 64/124 cone0.5 (baseline) | 17.35 / 15.76 | 65536 | 33153 | 703 | 93.2 | 64.0 | 0.752 | 1.356 | 6.43 | 0.983 | 0.088 | 85/96/98 |
| standard 64/64 | 15.25 / 14.53 | 65536 | 33153 | 1025 | 63.9 | 46.8 | 0.999 | 1.447 | 6.93 | 0.988 | 0.075 | 64/64/64 |
| standard 64/96 | 15.89 / 15.59 | 65536 | 33153 | 709 | 92.4 | 63.8 | 0.963 | 1.364 | 6.45 | 0.980 | 0.091 | 85/95/96 |
| standard 64/128 | 15.43 / 15.32 | 65536 | 33153 | 703 | 93.2 | 64.0 | 0.728 | 1.356 | 6.43 | 0.983 | 0.088 | 85/96/98 |
| standard 96/128 | 16.98 / 16.67 | 65536 | 33153 | 516 | 127.0 | 84.7 | 0.992 | 1.318 | 6.17 | 0.984 | 0.109 | 128/128/128 |
| standard 128/128 | 16.94 / 16.80 | 65536 | 33153 | 513 | 127.8 | 84.9 | 0.998 | 1.313 | 6.16 | 0.986 | 0.107 | 128/128/128 |
| standard 64/124 cone0.0 | 15.71 / 15.49 | 65536 | 33153 | 703 | 93.2 | 63.9 | 0.752 | 1.356 | 6.43 | 0.983 | 0.089 | 85/96/98 |
| standard 64/124 cone1.0 | 15.18 / 14.77 | 65536 | 33153 | 694 | 94.4 | 64.0 | 0.762 | 1.339 | 6.39 | 0.968 | 0.093 | 90/95/98 |
| spatial 64/124 | 8.77 / 8.75 | 65536 | 33153 | 1079 | 60.7 | 45.9 | 0.490 | 1.495 | 7.08 | 0.990 | 0.074 | 45/60/77 |
| spatial 64/96 | 8.62 / 8.48 | 65536 | 33153 | 1164 | 56.3 | 43.1 | 0.586 | 1.512 | 7.20 | 0.990 | 0.070 | 34/54/96 |
| spatial 64/128 | 8.69 / 8.64 | 65536 | 33153 | 1041 | 63.0 | 47.2 | 0.492 | 1.483 | 7.02 | 0.988 | 0.075 | 47/63/80 |
| standard 64/124 optimize | 17.62 / 17.40 | 65536 | 33153 | 703 | 93.2 | 64.0 | 0.752 | 1.356 | 6.43 | 0.983 | 0.088 | 85/96/98 |

### torus -- generateTorus(major 1, minor 0.35, 256 x 128)

| option set | cook ms (med/min) | tris | vertices | meshlets | tri/ml | vert/ml | fill | dup | B/tri | cone usable | cone cutoff | tri p10/p50/p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard 64/124 cone0.5 (baseline) | 15.40 / 15.10 | 65536 | 33153 | 684 | 95.8 | 63.9 | 0.773 | 1.319 | 6.34 | 1.000 | 0.217 | 94/96/98 |
| standard 64/64 | 15.31 / 15.05 | 65536 | 33153 | 1024 | 64.0 | 45.9 | 1.000 | 1.417 | 6.87 | 1.000 | 0.178 | 64/64/64 |
| standard 64/96 | 16.27 / 15.66 | 65536 | 33153 | 691 | 94.8 | 63.7 | 0.988 | 1.328 | 6.36 | 1.000 | 0.217 | 93/96/96 |
| standard 64/128 | 15.77 / 15.65 | 65536 | 33153 | 684 | 95.8 | 64.0 | 0.749 | 1.320 | 6.34 | 1.000 | 0.216 | 94/96/98 |
| standard 96/128 | 17.07 / 16.84 | 65536 | 33153 | 513 | 127.8 | 83.2 | 0.998 | 1.287 | 6.10 | 1.000 | 0.255 | 128/128/128 |
| standard 128/128 | 16.76 / 16.39 | 65536 | 33153 | 512 | 128.0 | 83.2 | 1.000 | 1.285 | 6.10 | 1.000 | 0.255 | 128/128/128 |
| standard 64/124 cone0.0 | 15.33 / 15.22 | 65536 | 33153 | 685 | 95.7 | 64.0 | 0.772 | 1.322 | 6.34 | 1.000 | 0.222 | 94/96/98 |
| standard 64/124 cone1.0 | 14.94 / 14.69 | 65536 | 33153 | 741 | 88.4 | 64.0 | 0.713 | 1.430 | 6.62 | 1.000 | 0.129 | 76/92/96 |
| spatial 64/124 | 8.48 / 8.24 | 65536 | 33153 | 1064 | 61.6 | 44.4 | 0.497 | 1.426 | 6.93 | 1.000 | 0.162 | 46/61/77 |
| spatial 64/96 | 8.38 / 8.29 | 65536 | 33153 | 1083 | 60.5 | 43.5 | 0.630 | 1.421 | 6.93 | 1.000 | 0.158 | 37/54/96 |
| spatial 64/128 | 8.67 / 8.60 | 65536 | 33153 | 1024 | 64.0 | 45.9 | 0.500 | 1.417 | 6.87 | 1.000 | 0.166 | 48/64/80 |
| standard 64/124 optimize | 17.20 / 17.14 | 65536 | 33153 | 684 | 95.8 | 63.9 | 0.773 | 1.319 | 6.34 | 1.000 | 0.217 | 94/96/98 |

### icosphere -- generateIcosahedron(radius 1, subdivisions 6, smooth normals)

| option set | cook ms (med/min) | tris | vertices | meshlets | tri/ml | vert/ml | fill | dup | B/tri | cone usable | cone cutoff | tri p10/p50/p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard 64/124 cone0.5 (baseline) | 20.08 / 19.76 | 81920 | 41161 | 841 | 97.4 | 64.0 | 0.786 | 1.307 | 6.28 | 1.000 | 0.082 | 96/98/99 |
| standard 64/64 | 19.15 / 18.98 | 81920 | 41161 | 1280 | 64.0 | 45.2 | 1.000 | 1.406 | 6.83 | 1.000 | 0.067 | 64/64/64 |
| standard 64/96 | 20.01 / 19.99 | 81920 | 41161 | 860 | 95.3 | 63.3 | 0.992 | 1.322 | 6.33 | 1.000 | 0.083 | 94/96/96 |
| standard 64/128 | 19.95 / 19.92 | 81920 | 41161 | 841 | 97.4 | 64.0 | 0.761 | 1.307 | 6.28 | 1.000 | 0.082 | 96/98/99 |
| standard 96/128 | 21.95 / 21.71 | 81920 | 41161 | 641 | 127.8 | 82.2 | 0.998 | 1.281 | 6.07 | 1.000 | 0.099 | 128/128/128 |
| standard 128/128 | 22.21 / 21.92 | 81920 | 41161 | 640 | 128.0 | 82.4 | 1.000 | 1.281 | 6.08 | 1.000 | 0.099 | 128/128/128 |
| standard 64/124 cone0.0 | 20.39 / 20.20 | 81920 | 41161 | 842 | 97.3 | 63.9 | 0.785 | 1.308 | 6.29 | 1.000 | 0.082 | 96/98/99 |
| standard 64/124 cone1.0 | 21.00 / 20.73 | 81920 | 41161 | 842 | 97.3 | 64.0 | 0.785 | 1.309 | 6.29 | 1.000 | 0.082 | 96/98/99 |
| spatial 64/124 | 11.18 / 11.08 | 81920 | 41161 | 1324 | 61.9 | 43.9 | 0.499 | 1.411 | 6.87 | 1.000 | 0.069 | 48/62/76 |
| spatial 64/96 | 10.68 / 10.27 | 81920 | 41161 | 1155 | 70.9 | 49.2 | 0.739 | 1.379 | 6.67 | 1.000 | 0.072 | 40/63/96 |
| spatial 64/128 | 10.90 / 10.74 | 81920 | 41161 | 1284 | 63.8 | 45.0 | 0.498 | 1.405 | 6.83 | 1.000 | 0.070 | 49/63/79 |
| standard 64/124 optimize | 21.79 / 21.50 | 81920 | 41161 | 841 | 97.4 | 64.0 | 0.786 | 1.307 | 6.28 | 1.000 | 0.082 | 96/98/99 |

### plane -- generatePlane(100 x 100, 512 x 512 quads)

| option set | cook ms (med/min) | tris | vertices | meshlets | tri/ml | vert/ml | fill | dup | B/tri | cone usable | cone cutoff | tri p10/p50/p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard 64/124 cone0.5 (baseline) | 118.85 / 117.01 | 524288 | 263169 | 5437 | 96.4 | 64.0 | 0.778 | 1.322 | 6.32 | 1.000 | 0.000 | 94/97/98 |
| standard 64/64 | 112.92 / 110.78 | 524288 | 263169 | 8192 | 64.0 | 45.5 | 1.000 | 1.416 | 6.84 | 1.000 | 0.000 | 64/64/64 |
| standard 64/96 | 111.49 / 110.72 | 524288 | 263169 | 5511 | 95.1 | 63.8 | 0.991 | 1.336 | 6.36 | 1.000 | 0.000 | 93/96/96 |
| standard 64/128 | 118.93 / 118.16 | 524288 | 263169 | 5435 | 96.5 | 64.0 | 0.754 | 1.322 | 6.32 | 1.000 | 0.000 | 95/97/98 |
| standard 96/128 | 132.16 / 130.03 | 524288 | 263169 | 4097 | 128.0 | 83.0 | 1.000 | 1.292 | 6.09 | 1.000 | 0.000 | 128/128/128 |
| standard 128/128 | 133.74 / 131.80 | 524288 | 263169 | 4096 | 128.0 | 83.0 | 1.000 | 1.292 | 6.09 | 1.000 | 0.000 | 128/128/128 |
| standard 64/124 cone0.0 | 119.94 / 117.48 | 524288 | 263169 | 5435 | 96.5 | 64.0 | 0.778 | 1.322 | 6.32 | 1.000 | 0.000 | 95/97/98 |
| standard 64/124 cone1.0 | 104.29 / 103.50 | 524288 | 263169 | 5497 | 95.4 | 64.0 | 0.769 | 1.337 | 6.35 | 1.000 | 0.000 | 94/95/97 |
| spatial 64/124 | 80.92 / 80.77 | 524288 | 263169 | 8476 | 61.9 | 61.2 | 0.499 | 1.971 | 7.99 | 1.000 | 0.000 | 62/62/62 |
| spatial 64/96 | 84.58 / 82.26 | 524288 | 263169 | 10088 | 52.0 | 44.6 | 0.541 | 1.709 | 7.66 | 1.000 | 0.000 | 32/48/64 |
| spatial 64/128 | 97.55 / 97.30 | 524288 | 263169 | 12288 | 42.7 | 44.7 | 0.333 | 2.086 | 8.69 | 1.000 | 0.000 | 24/42/62 |
| standard 64/124 optimize | 126.29 / 126.05 | 524288 | 263169 | 5437 | 96.4 | 64.0 | 0.778 | 1.322 | 6.32 | 1.000 | 0.000 | 94/97/98 |

### building -- subdivided cube, 32x32 quads per face, welded

| option set | cook ms (med/min) | tris | vertices | meshlets | tri/ml | vert/ml | fill | dup | B/tri | cone usable | cone cutoff | tri p10/p50/p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard 64/124 cone0.5 (baseline) | 2.49 / 2.42 | 12288 | 6146 | 129 | 95.3 | 63.7 | 0.768 | 1.337 | 6.35 | 1.000 | 0.206 | 91/97/98 |
| standard 64/64 | 2.50 / 2.34 | 12288 | 6146 | 192 | 64.0 | 45.8 | 1.000 | 1.432 | 6.86 | 1.000 | 0.157 | 64/64/64 |
| standard 64/96 | 2.69 / 2.52 | 12288 | 6146 | 130 | 94.5 | 63.7 | 0.985 | 1.348 | 6.37 | 1.000 | 0.183 | 89/96/96 |
| standard 64/128 | 2.58 / 2.49 | 12288 | 6146 | 129 | 95.3 | 63.6 | 0.744 | 1.335 | 6.34 | 1.000 | 0.210 | 93/97/98 |
| standard 96/128 | 2.91 / 2.86 | 12288 | 6146 | 97 | 126.7 | 82.9 | 0.990 | 1.309 | 6.12 | 1.000 | 0.221 | 128/128/128 |
| standard 128/128 | 2.97 / 2.81 | 12288 | 6146 | 96 | 128.0 | 83.8 | 1.000 | 1.308 | 6.12 | 1.000 | 0.223 | 128/128/128 |
| standard 64/124 cone0.0 | 2.64 / 2.54 | 12288 | 6146 | 128 | 96.0 | 63.8 | 0.774 | 1.328 | 6.32 | 1.000 | 0.342 | 95/97/98 |
| standard 64/124 cone1.0 | 2.26 / 2.12 | 12288 | 6146 | 129 | 95.3 | 63.8 | 0.768 | 1.339 | 6.35 | 1.000 | 0.120 | 93/96/98 |
| spatial 64/124 | 1.27 / 1.25 | 12288 | 6146 | 200 | 61.4 | 44.3 | 0.495 | 1.441 | 6.92 | 1.000 | 0.133 | 46/60/76 |
| spatial 64/96 | 1.21 / 1.21 | 12288 | 6146 | 206 | 59.7 | 45.8 | 0.621 | 1.534 | 7.14 | 1.000 | 0.055 | 32/64/96 |
| spatial 64/128 | 1.25 / 1.24 | 12288 | 6146 | 208 | 59.1 | 45.2 | 0.462 | 1.531 | 7.15 | 1.000 | 0.114 | 32/64/76 |
| standard 64/124 optimize | 2.80 / 2.75 | 12288 | 6146 | 129 | 95.3 | 63.7 | 0.768 | 1.337 | 6.35 | 1.000 | 0.206 | 91/97/98 |

### slivers -- generatePlane(2000 x 1, 1000 x 100 quads): 2 x 0.01 cells, 200k triangles, edge aspect > 200

| option set | cook ms (med/min) | tris | vertices | meshlets | tri/ml | vert/ml | fill | dup | B/tri | cone usable | cone cutoff | tri p10/p50/p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard 64/124 cone0.5 (baseline) | 35.10 / 34.94 | 200000 | 101101 | 3000 | 66.7 | 64.0 | 0.538 | 1.899 | 7.80 | 1.000 | 0.000 | 62/62/76 |
| standard 64/64 | 36.50 / 36.29 | 200000 | 101101 | 3163 | 63.2 | 59.5 | 0.988 | 1.861 | 7.78 | 1.000 | 0.000 | 62/64/64 |
| standard 64/96 | 35.21 / 35.12 | 200000 | 101101 | 3000 | 66.7 | 64.0 | 0.694 | 1.899 | 7.80 | 1.000 | 0.000 | 62/62/76 |
| standard 64/128 | 35.35 / 35.30 | 200000 | 101101 | 3000 | 66.7 | 64.0 | 0.521 | 1.899 | 7.80 | 1.000 | 0.000 | 62/62/76 |
| standard 96/128 | 44.18 / 43.88 | 200000 | 101101 | 2000 | 100.0 | 96.0 | 0.781 | 1.898 | 7.48 | 1.000 | 0.000 | 94/95/106 |
| standard 128/128 | 52.67 / 50.31 | 200000 | 101101 | 1564 | 127.9 | 109.4 | 0.999 | 1.693 | 6.92 | 1.000 | 0.000 | 128/128/128 |
| standard 64/124 cone0.0 | 35.34 / 35.22 | 200000 | 101101 | 3000 | 66.7 | 64.0 | 0.538 | 1.899 | 7.80 | 1.000 | 0.000 | 62/62/76 |
| standard 64/124 cone1.0 | 36.94 / 36.87 | 200000 | 101101 | 2101 | 95.2 | 64.0 | 0.768 | 1.330 | 6.36 | 1.000 | 0.000 | 94/95/97 |
| spatial 64/124 | 26.21 / 26.15 | 200000 | 101101 | 3232 | 61.9 | 62.3 | 0.499 | 1.990 | 8.06 | 1.000 | 0.000 | 62/62/62 |
| spatial 64/96 | 24.51 / 24.41 | 200000 | 101101 | 2860 | 69.9 | 52.6 | 0.728 | 1.489 | 6.93 | 1.000 | 0.000 | 48/64/96 |
| spatial 64/128 | 25.86 / 25.76 | 200000 | 101101 | 3156 | 63.4 | 48.3 | 0.495 | 1.507 | 7.06 | 1.000 | 0.000 | 48/64/80 |
| standard 64/124 optimize | 37.47 / 37.18 | 200000 | 101101 | 3000 | 66.7 | 64.0 | 0.538 | 1.899 | 7.80 | 1.000 | 0.000 | 62/62/76 |

### soup -- 100000 random triangles, seed 12345, centres uniform in [-1,1]^3, edge ~0.03, no shared vertices

| option set | cook ms (med/min) | tris | vertices | meshlets | tri/ml | vert/ml | fill | dup | B/tri | cone usable | cone cutoff | tri p10/p50/p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| standard 64/124 cone0.5 (baseline) | 70.55 / 63.59 | 100000 | 300000 | 4762 | 21.0 | 63.0 | 0.169 | 1.000 | 18.05 | 0.000 | 0.000 | 21/21/21 |
| standard 64/64 | 64.28 / 61.49 | 100000 | 300000 | 4762 | 21.0 | 63.0 | 0.328 | 1.000 | 18.05 | 0.000 | 0.000 | 21/21/21 |
| standard 64/96 | 60.43 / 59.65 | 100000 | 300000 | 4762 | 21.0 | 63.0 | 0.219 | 1.000 | 18.05 | 0.000 | 0.000 | 21/21/21 |
| standard 64/128 | 66.64 / 61.85 | 100000 | 300000 | 4762 | 21.0 | 63.0 | 0.164 | 1.000 | 18.05 | 0.000 | 0.000 | 21/21/21 |
| standard 96/128 | 63.15 / 62.96 | 100000 | 300000 | 3125 | 32.0 | 96.0 | 0.250 | 1.000 | 17.00 | 0.000 | 0.000 | 32/32/32 |
| standard 128/128 | 67.00 / 65.95 | 100000 | 300000 | 2381 | 42.0 | 126.0 | 0.328 | 1.000 | 16.52 | 0.000 | 0.000 | 42/42/42 |
| standard 64/124 cone0.0 | 60.91 / 59.17 | 100000 | 300000 | 4762 | 21.0 | 63.0 | 0.169 | 1.000 | 18.05 | 0.000 | 0.000 | 21/21/21 |
| standard 64/124 cone1.0 | 59.27 / 59.03 | 100000 | 300000 | 4762 | 21.0 | 63.0 | 0.169 | 1.000 | 18.05 | 0.000 | 0.000 | 21/21/21 |
| spatial 64/124 | 17.52 / 17.37 | 100000 | 300000 | 4839 | 20.7 | 62.0 | 0.167 | 1.000 | 18.10 | 0.014 | 0.010 | 21/21/21 |
| spatial 64/96 | 18.50 / 18.12 | 100000 | 300000 | 4839 | 20.7 | 62.0 | 0.215 | 1.000 | 18.10 | 0.006 | 0.030 | 21/21/21 |
| spatial 64/128 | 17.97 / 17.53 | 100000 | 300000 | 4839 | 20.7 | 62.0 | 0.161 | 1.000 | 18.10 | 0.017 | 0.028 | 21/21/21 |
| standard 64/124 optimize | 63.05 / 61.01 | 100000 | 300000 | 4762 | 21.0 | 63.0 | 0.169 | 1.000 | 18.05 | 0.000 | 0.000 | 21/21/21 |

sponza: not available (assets/README.md lists the source; no download in this phase)

### Observations (facts only, from the tables above)

- Bytes per triangle: on the smooth meshes (sphere, torus, icosphere, plane,
  building) the lowest value is reached by the largest meshlets: 6.16 (sphere,
  128/128), 6.10 (torus, 96/128 and 128/128), 6.07 (icosphere, 96/128), 6.09
  (plane, 96/128 and 128/128), 6.12 (building, 96/128 and 128/128). The baseline
  64/124 costs 6.28–6.43 there; spatial costs 6.67–8.69; 64/64 costs 6.83–6.93. On
  slivers the lowest is standard 64/124 with cone weight 1.0 (6.36; baseline 7.80,
  128/128 6.92, spatial 64/96 6.93). On the soup every option is 16.5–18.1
  (128/128 lowest, 16.52).
- Fill: the baseline 64/124 fills 0.75–0.79 of the triangle capacity on the smooth
  meshes (the 64-vertex limit binds, ~93–97 triangles per meshlet), 0.54 on
  slivers and 0.17 on the soup (21 triangles per meshlet: 3 vertices each, 63 of 64
  used). 64/64, 96/128 and 128/128 reach 0.98–1.00 on the smooth meshes. Spatial
  64/124 averages 0.49–0.50 (about 61 triangles per meshlet) on every smooth mesh;
  spatial 64/96 reaches 0.54–0.74.
- Vertex duplication: standard 64/124 gives 1.31–1.36 on smooth meshes and 1.90 on
  slivers (3000 meshlets, p50 62 triangles); standard cone 1.0 brings slivers to
  1.33 with 2101 meshlets. Spatial gives 1.41–1.53 on the sphere/torus/icosphere/
  building, 1.97 (64/124) and 2.09 (64/128) on the plane, and 1.49–1.99 on slivers.
- Cone: the usable fraction is 0.97–1.00 on every connected mesh and 0.00 on the
  soup for the standard builder (0.006–0.017 for spatial). Mean cutoff on the
  torus: 0.217 (cone 0.5), 0.222 (0.0), 0.129 (1.0, with 741 meshlets instead of
  684); on the building 0.342 (0.0), 0.206 (0.5), 0.120 (1.0); on the sphere 0.088
  (64/124) and 0.107 (128/128). The plane and slivers have cutoff 0.000 (all
  triangles coplanar). On the icosphere the cone weight changed neither meshlet
  count (841–842) nor cutoff.
- Cook time (median of 5, single thread): spatial is 0.26–0.82x of standard at the
  same limits (sphere 8.7 vs 17.4 ms, plane 81–98 vs 119 ms, soup 17.5–18.5 vs
  60–70 ms). 96/128 and 128/128 cost 1.0–1.5x the baseline (plane 132/134 vs 119
  ms, slivers 44/53 vs 35 ms). `optimize` adds 2–12% (none visible on the soup) and leaves every statistic
  unchanged. Standard 64/128 gives the baseline's meshlet count on every mesh (within 2
  meshlets): the vertex limit, not the triangle limit, binds.
- Cook time of the 12 option sets differs by run-to-run noise of the order of
  1 ms on the 15 ms cooks (min and median columns); differences below ~5% are not
  claimed.

Not decided here: which option set the engine uses. Triangle/vertex limits trade
mesh-shader launch count, fill, per-meshlet cull cost and bytes, and only the GPU
part of S1/S2 (frame cost on real engine workloads, plus Hi-Z/cone culling
efficiency) can rank them. The decision is taken after those measurements and is
recorded in `docs/opt-log.md`.
