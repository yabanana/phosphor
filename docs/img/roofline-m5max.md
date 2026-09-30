# Predetto vs misurato — Apple M5 Max (Torus Demo, PBR Material Grid, Stress Test (100K), Scene Viewer (glTF), Many Lights (1024), Cornell Box (GI), Culling Visualization)

Il predetto è un **limite inferiore** (roofline, nessun overhead): rapporto = misurato p50 / predetto.

| Unità | Misurato ms | Predetto ms | Limite | AI (F/B) | Rapporto | Stima ms (non un limite) |
|---|---:|---:|---|---:|---:|---:|
| Torus Demo: Forward | 1.02 | 0.0402 | dram | 15.4 | 25.3 | 0.117 |
| PBR Material Grid: Forward | 0.124 | 0.0402 | dram | 2.12 | 3.08 | 0.05 |
| Stress Test (100K): Forward | 1.86 | 0.055 | alu | 33.5 | 33.9 | 0.12 |
| Scene Viewer (glTF): Forward | 0.203 | 0.0402 | dram | 11.9 | 5.04 | 0.0905 |
| Many Lights (1024): Forward | 52.6 | 0.0402 | dram | 16.5 | 1.31e+03 | 99.9 |
| Cornell Box (GI): Forward | 0.209 | 0.0402 | dram | 8.22 | 5.2 | 0.0676 |
| Culling Visualization: Forward | 0.861 | 0.0402 | dram | 20.8 | 21.4 | 0.153 |
