# Phosphor — Perf log

Registro delle misure di prestazione, in ordine cronologico (le più recenti in
fondo a ogni sezione). Ogni voce riporta commit, macchina, condizioni e metodo:
un numero senza contesto non vale. Le ottimizzazioni sperimentali vanno in
[`opt-log.md`](opt-log.md); qui restano le baseline e le misure di chiusura fase.

## Metodo

- Build **Release** senza validazione Metal:
  `cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release`.
- `tools/bench_all.sh build/release 600 3`: per ogni testbench 3 esecuzioni da
  120 frame di warm-up + 600 misurati, `--no-vsync --no-ui`; la riga riporta
  l'esecuzione con il frame time medio mediano. I JSON completi finiscono in
  `build/release/bench-results/`.
- Finestra di default 1600×900 punti = **3200×1800 pixel** sul display Retina,
  finestra visibile e in primo piano (macOS rallenta le finestre coperte).
- Colonne (vedi `src/diagnostics/bench_report.h`):
  - **Frame ms**: tempo tra due frame consecutivi; FPS = 1000 / media.
  - **CPU ms**: tempo CPU per produrre il frame (eventi, simulazione,
    estrazione, encoding, submit), **escluse** le attese in `beginFrame()`.
  - **GPU ms**: intervallo inizio→fine del command buffer sulla GPU (commit
    feedback MTL4). Se la GPU sovrappone frame consecutivi l'intervallo include
    la sovrapposizione: è un **limite superiore** del costo GPU del frame, non il
    tempo di occupazione. I timestamp per pass arrivano con F4.1.
  - **Wait ms**: CPU bloccata in `beginFrame()` (slot di frame libero +
    `nextDrawable()`).
  - Tra parentesi il p99 (nearest-rank).

---

## F0 — baseline (F0.8)

**2026-09-28** · branch `phase/f0-close` · M5 Max (T2) · Mac17,6, GPU 40 core,
128 GB · macOS 27.2 (26B5091g) · Xcode 27.0, Metal 32023.921 · display
interno 3456×2234 Retina a 120 Hz · **a batteria (39%), powermode 2 = High
Power** · Release, `--no-vsync --no-ui`, 3×600 frame.

| # | Bench | Resolution | FPS | Frame ms (p99) | CPU ms (p99) | GPU ms (p99) | Wait ms (p99) |
|---|---|---|---|---|---|---|---|
| 1 | Torus Demo | 3200x1800 | 345 | 2.898 (11.764) | 0.115 (0.209) | 1.337 (3.026) | 2.774 (11.599) |
| 2 | PBR Material Grid | 3200x1800 | 418 | 2.393 (10.333) | 0.115 (0.226) | 0.812 (2.024) | 2.265 (10.154) |
| 3 | Stress Test (100K) | 3200x1800 | 238 | 4.195 (9.392) | 2.985 (3.24) | 3.044 (3.917) | 1.21 (6.307) |
| 4 | Scene Viewer (glTF) | 3200x1800 | 365 | 2.739 (10.516) | 0.112 (0.214) | 1.308 (2.633) | 2.627 (10.279) |
| 5 | Many Lights (1024) | 3200x1800 | 17 | 58.897 (127.24) | 0.107 (0.145) | 108.735 (171.948) | 58.703 (127.123) |
| 6 | Cornell Box (GI) | 3200x1800 | 434 | 2.305 (9.829) | 0.106 (0.201) | 0.829 (2.083) | 2.199 (9.667) |
| 7 | Culling Visualization | 3200x1800 | 316 | 3.166 (11.201) | 0.91 (1.306) | 1.987 (5.3) | 2.247 (10.479) |

**T0 (M3/M4/M5 base)**: da misurare, nessun dispositivo T0 disponibile in
questa sessione (regola O12). Stesso comando su T0 e nuova tabella qui sotto.

### Ripetizione su alimentazione di rete (F0.8)

**2026-09-28 (notte)** · stesso codice `phase/f0-close` · **alimentazione di
rete**, powermode High Power · ambiente non limitato · Release,
`--no-vsync --no-ui`, 3×600 frame.

| # | Bench | Resolution | FPS | Frame ms (p99) | CPU ms (p99) | GPU ms (p99) | Wait ms (p99) |
|---|---|---|---|---|---|---|---|
| 1 | Torus Demo | 3200x1800 | 415 | 2.41 (17.845) | 0.097 (0.177) | 0.942 (1.558) | 2.312 (17.679) |
| 2 | PBR Material Grid | 3200x1800 | 469 | 2.13 (16.021) | 0.106 (0.183) | 0.756 (1.114) | 2.023 (15.885) |
| 3 | Stress Test (100K) | 3200x1800 | 281 | 3.562 (17.148) | 2.757 (3.031) | 2.152 (2.883) | 0.785 (14.355) |
| 4 | Scene Viewer (glTF) | 3200x1800 | 474 | 2.11 (16.314) | 0.093 (0.175) | 0.872 (1.361) | 2.017 (16.116) |
| 5 | Many Lights (1024) | 3200x1800 | 19 | 51.319 (108.206) | 0.101 (0.205) | 91.543 (147.554) | 51.21 (108.102) |
| 6 | Cornell Box (GI) | 3200x1800 | 471 | 2.123 (17.261) | 0.093 (0.161) | 0.651 (0.92) | 2.033 (17.099) |
| 7 | Culling Visualization | 3200x1800 | 423 | 2.362 (14.523) | 0.861 (1.143) | 1.245 (1.944) | 1.502 (13.638) |

Rispetto alla misura a batteria: CPU ms quasi uguali, GPU ms più bassi
sullo Stress Test (2,15 contro 3,04 ms); le altre differenze stanno nella
variabilità di presentazione descritta sotto. **Questa è la baseline M5 Max
di riferimento per F0.**

### Osservazioni

- **Scene leggere (1, 2, 4, 6, 7) limitate dalla presentazione, non dall'engine.**
  CPU ≈ 0,1 ms e GPU ≈ 1 ms, ma il frame dura 2,3–3,2 ms: il resto è attesa del
  drawable (Wait ≈ Frame). Il p99 del frame (~10–11 ms) coincide con il p99 di
  Wait: è `nextDrawable()` che ogni tanto aspetta circa un refresh a 120 Hz,
  cioè il compositor di macOS in modalità finestra. In queste scene l'FPS senza
  vsync misura quindi il sistema di presentazione, e varia molto tra
  esecuzioni: una prima esecuzione identica ha dato 417 fps sul Torus e 437 sul
  PBR Grid, contro 345 e 418 qui. Per confrontare il costo dell'engine su
  queste scene usare CPU ms e GPU ms, non l'FPS.
- **Stress Test (100K)**: l'unico limitato dalla CPU. 3,0 ms per estrarre e
  copiare 100.000 istanze a ogni frame (ECS → `GPUInstance`, ordinamento per
  mesh, memcpy). È il costo che F5 (GPU scene persistente, aggiornamenti delta)
  deve eliminare.
- **Many Lights (1024)**: limitato dalla GPU, 17 fps. Il forward di F0 valuta
  tutte le 1.024 luci per ogni pixel (5,76 M pixel × 1.024 luci). La GPU ms
  (109 ms) supera il frame time (59 ms) perché la GPU esegue due frame in
  parallelo: non c'è dipendenza tra frame consecutivi (drawable diversi,
  nessuna barriera). Il costo reale per frame è ≈ il frame time. Obiettivo di
  F11 (ReSTIR DI / cluster di luci).
- **Culling Visualization**: 10.001 istanze, CPU 0,9 ms (estrazione) e GPU 2 ms
  senza alcun culling: base di confronto per F5/F6.
- **DVFS**: sul Torus con vsync attivo (120 fps) la GPU ms media sale da 1,15 a
  1,67 ms: con poco carico la GPU abbassa la frequenza, quindi un pass costa
  di più in millisecondi. I confronti di GPU ms vanno fatti nelle stesse
  condizioni di vsync e di carico.
- **Da rifare su alimentazione di rete**: questa baseline è stata presa a
  batteria (High Power attivo). Ripetere `tools/bench_all.sh` collegati alla
  rete e aggiungere la tabella sotto questa.

### Condizioni che hanno cambiato le misure durante F0.7

- Con `MTL_SHADER_VALIDATION=1` Many Lights scende a ~3 fps (GPU ~450 ms):
  la validazione degli shader non va mai usata per misurare.
- Build Debug (`-O0`, che attiva da sola la validazione API): lo Stress Test
  passa da 2,9 a 32,9 ms di CPU per frame (30 fps invece di 239). Per il perf
  log usare sempre Release.
