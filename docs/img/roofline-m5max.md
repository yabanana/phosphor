# Predetto vs misurato — Apple M5 Max (Torus Demo, PBR Material Grid, Stress Test (100K), Scene Viewer (glTF), Many Lights (1024), Cornell Box (GI), Culling Visualization)

Il predetto è un **limite inferiore** (roofline, nessun overhead): rapporto = misurato p50 / predetto.

| Unità | Misurato ms | Predetto ms | Limite | AI (F/B) | Rapporto | Stima ms (non un limite) |
|---|---:|---:|---|---:|---:|---:|
| Torus Demo: Forward | 0.342 | 0.0405 | dram | 15.4 | 8.43 | 0.117 |
| PBR Material Grid: Forward | 0.218 | 0.0405 | dram | 2.12 | 5.39 | 0.05 |
| Stress Test (100K): Forward | 1.88 | 0.055 | alu | 33.5 | 34.2 | 0.12 |
| Scene Viewer (glTF): Forward | 0.807 | 0.0405 | dram | 11.9 | 19.9 | 0.0904 |
| Many Lights (1024): Forward | 52.4 | 0.0405 | dram | 16.5 | 1.29e+03 | 99.8 |
| Cornell Box (GI): Forward | 0.624 | 0.0405 | dram | 8.22 | 15.4 | 0.0676 |
| Culling Visualization: Forward | 0.63 | 0.0405 | dram | 20.8 | 15.6 | 0.153 |
