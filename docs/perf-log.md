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

---

## F1 — memoria, heap e residency

**2026-09-28 (sera)** · branch `phase/f1` · stessa macchina · **alimentazione
di rete** · Release, `--no-vsync --no-ui`, 3×600 frame.

**Attenzione: l'ambiente è cambiato rispetto alla baseline F0.** Nel
pomeriggio il sistema ha iniziato a limitare la presentazione senza vsync a
~80 fps (12,5 ms per frame), con CPU/GPU ms più alti a parità di codice
(probabile riduzione delle frequenze). Il codice di F0 ricompilato e misurato
subito prima di F1 dà gli stessi numeri: **il confronto valido è quello
consecutivo qui sotto, non con la tabella della baseline.**

| # | Bench | F0 CPU ms (p99) | F1 CPU ms (p99) | F0 GPU ms (p99) | F1 GPU ms (p99) |
|---|---|---|---|---|---|
| 1 | Torus Demo | 0.25 (0.356) | 0.243 (0.351) | 1.141 (1.617) | 1.142 (1.717) |
| 2 | PBR Material Grid | 0.269 (0.377) | 0.277 (0.377) | 0.816 (1.575) | 0.815 (1.277) |
| 3 | Stress Test (100K) | 4.362 (4.66) | 4.297 (4.652) | 5.979 (8.311) | 5.585 (8.4) |
| 4 | Scene Viewer (glTF) | 0.245 (0.347) | 0.247 (0.357) | 0.876 (1.018) | 0.877 (1.139) |
| 5 | Many Lights (1024) | 0.107 (0.203) | 0.115 (0.238) | 103.454 (164.216) | 102.711 (160.943) |
| 6 | Cornell Box (GI) | 0.241 (0.356) | 0.244 (0.353) | 0.654 (0.748) | 0.649 (0.723) |
| 7 | Culling Visualization | 1.17 (1.359) | 1.152 (1.334) | 3.516 (6.057) | 3.553 (6.056) |

Nessuna regressione: le differenze stanno nel rumore tra esecuzioni.

### Criteri di uscita di F1

- **Allocazioni GPU nel frame**: 0 su tutti i 7 bench (1.200 frame misurati;
  contatore di `GpuMemory`, colonna `gpu_allocations` dei JSON).
- **Heap CPU**: `malloc_zone_statistics` (tutte le zone) tra inizio e fine
  della misura: +1.000…+2.100 blocchi (~50–120 KB) **indipendentemente dal
  numero di frame** (600 → 4.800 frame: stessa crescita), con e senza UI.
  È una fluttuazione limitata di cache di sistema/driver, non una perdita
  per frame. L'unica allocazione per frame nota è `MTL4CommitOptions` (un
  oggetto riusato smette di consegnare il feedback: misurato 0 su 600 frame).
- **Allocazioni CPU nel frame (strumenti Apple)**. Serve un binario
  debuggable: le build ora sono firmate ad hoc con `get-task-allow`
  (`PHOSPHOR_DEBUGGABLE`, predefinito ON); prima `xctrace`/`leaks` non
  potevano agganciarsi e una traccia Allocations lanciata con `sudo` era
  vuota (12 KB di dati).
  - `malloc_history -allByCount` a 20 s e 80 s di Stress Test (Release,
    ~4.800 frame nel mezzo), differenza per stack: **frame loop
    (`Engine::frame`) −3 allocazioni vive**, processo intero −165. I ±45
    tra `beginFrame` e `submitFrame` sono oggetti dei frame in volo.
  - `leaks --atExit` dopo 1.200 frame: **0 leak**.
  - Una prima misura con `heap` aveva mostrato +26.548 blocchi in 45 s:
    erano CoreSVG/CoreUI, le icone del menu Finestra caricate in modo
    lazy da AppKit dopo `makeKeyAndOrderFront` (costo di sistema una
    tantum, stack verificati con `malloc_history`), non codice Phosphor.
  - Traccia Instruments Allocations completa (30 s, 126 MB) registrata
    senza `sudo`; `xctrace export` non espone la tabella delle
    allocazioni, quindi si ispeziona aprendo il file in Instruments.
- **F1.6 stress**: `--memory-stress 10000` e `30000` con validazione API:
  la memoria del device torna esattamente alla baseline (192,81 MiB) e i
  contatori per categoria sono ripristinati; picco di 9 heap e 704,81 MiB
  con heap trattenuti **identici** a 10k e 30k cicli (il pool è limitato dal
  picco di dati vivi, non dal numero di operazioni).
- **F1.5 pressione di memoria reale**: con l'app in esecuzione (Stress Test),
  `sudo memory_pressure -S -l warn|critical|normal` → l'app riceve ogni
  livello e reagisce tra due frame ("warning" → "back to normal",
  "critical" → "back to normal"). Trim di 0 MiB perché l'unico heap del bench
  è in uso; il rilascio effettivo degli heap vuoti è coperto da `--memory-stress`.
- **Budget** (M5 Max): working set interrogato 107,5 GiB, budget engine
  80,6 GiB; tier rilevato "T2 Max + T3 Neural". Lo Stress Test usa fino a
  18 MiB di upload per frame (48,6 dei 64 MiB dell'anello in volo).
