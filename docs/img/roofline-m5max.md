# Predetto vs misurato — Apple M5 Max (Torus Demo, PBR Material Grid, Stress Test (100K), Scene Viewer (glTF), Many Lights (1024), Cornell Box (GI), Culling Visualization)

Il predetto è un **limite inferiore** (roofline, nessun overhead): rapporto = misurato p50 / predetto.

| Unità | Misurato ms | Predetto ms | Limite | AI (F/B) | Rapporto | Stima ms (non un limite) |
|---|---:|---:|---|---:|---:|---:|
| Torus Demo: Forward | 1.02 | 0.0402 | dram | 15.4 | 25.4 | 0.117 |
| PBR Material Grid: Forward | 0.209 | 0.0402 | dram | 2.12 | 5.19 | 0.05 |
| Stress Test (100K): Forward | 1.89 | 0.055 | alu | 33.5 | 34.4 | 0.12 |
| Scene Viewer (glTF): Forward | 0.269 | 0.0402 | dram | 11.9 | 6.69 | 0.0905 |
| Many Lights (1024): Forward | 52.5 | 0.0402 | dram | 16.5 | 1.3e+03 | 99.9 |
| Cornell Box (GI): Forward | 0.307 | 0.0402 | dram | 8.22 | 7.62 | 0.0676 |
| Culling Visualization: Forward | 0.699 | 0.0402 | dram | 20.8 | 17.4 | 0.153 |
