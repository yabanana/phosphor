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

### F1 — chiusura definitiva (audit severo dei residui)

**2026-09-28 (notte)** · `phase/f1` · alimentazione di rete · Release,
`--no-vsync --no-ui`, 3×600 frame. L'ambiente è tornato non limitato
(come la baseline F0 del mattino), quindi questa tabella è confrontabile con
quella della baseline.

| # | Bench | Resolution | FPS | Frame ms (p99) | CPU ms (p99) | GPU ms (p99) | Wait ms (p99) |
|---|---|---|---|---|---|---|---|
| 1 | Torus Demo | 3200x1800 | 461 | 2.168 (16.191) | 0.098 (0.168) | 0.878 (1.609) | 2.071 (16.065) |
| 2 | PBR Material Grid | 3200x1800 | 470 | 2.129 (15.394) | 0.113 (0.185) | 0.8 (0.989) | 2.015 (15.18) |
| 3 | Stress Test (100K) | 3200x1800 | 266 | 3.758 (18.072) | 2.853 (3.119) | 2.181 (3.031) | 0.881 (15.245) |
| 4 | Scene Viewer (glTF) | 3200x1800 | 478 | 2.094 (14.15) | 0.085 (0.198) | 0.873 (1.327) | 2.008 (13.962) |
| 5 | Many Lights (1024) | 3200x1800 | 18 | 54.625 (116.693) | 0.102 (0.209) | 101.323 (160.207) | 54.515 (116.515) |
| 6 | Cornell Box (GI) | 3200x1800 | 478 | 2.09 (16.731) | 0.095 (0.162) | 0.625 (0.777) | 1.993 (16.583) |
| 7 | Culling Visualization | 3200x1800 | 417 | 2.4 (16.766) | 0.889 (1.161) | 1.235 (2.278) | 1.51 (15.653) |

Rispetto alla baseline F0 (a batteria): nessuna regressione; Stress Test
CPU 2,85 ms (F0: 2,99), GPU 2,18 ms (F0: 3,04).

Residui trovati dall'audit e chiusi:
- **F1.1 heap transitorio** (la casella era stata spuntata con l'heap
  rimandato a F2.2): `TransientHeap` + `--transient-test` → buffer e
  texture sovrapposti con barriera `ResourceAlias` letti correttamente,
  sovrapposizione reale della memoria verificata. PASS, validazione a zero.
- **Dimensioni dei pool dal budget** (erano costanti "per tier in F1.4"):
  su M5 Max anello frame 128 MiB, staging 256 MiB, pagine heap 128 MiB;
  16 MiB sulle macchine piccole (test in `test_memory_budget.cpp`).
- **Heap di riserva purgeable** (previsto dal piano per F1.5): lo heap vuoto
  tenuto dopo un trim è `Volatile`, torna `NonVolatile` prima del riuso.
- **Fallimenti intermittenti del visual check**: benchmark e catture
  reagivano a tastiera/mouse (la finestra prende il focus all'avvio):
  riprodotto con `--inject-input` (filtro spento → PSNR 9–11 dB), risolto
  ignorando l'input in modalità benchmark; `visual_check.sh` ora inietta
  sempre input. 5 visual check consecutivi puliti.
- **Heap CPU**: +3.200…+3.600 blocchi indipendentemente da 600 o 6.000 frame
  e dall'input iniettato: fluttuazione limitata, nessuna crescita per frame.
- Instruments: la traccia Allocations si registra senza `sudo` ma non è
  esportabile da riga di comando; il criterio è verificato con
  `malloc_history` (−3 allocazioni vive nel frame loop, vedi sopra).

---

## F2 — render graph

**2026-09-29 (mattina)** · branch `phase/f2` (commit `e4116dd`; i commit
successivi non toccano il percorso misurato) · M5 Max, **alimentazione di
rete** · Release, `--no-vsync --no-ui`, `tools/bench_all.sh build/release`
(3×600 frame). `main` (`e17d684`) ricompilato in un worktree e misurato
**subito prima**, stessa sessione: il confronto valido è tra queste due
tabelle.

`main` (prima di F2):

| # | Bench | Resolution | FPS | Frame ms (p99) | CPU ms (p99) | GPU ms (p99) | Wait ms (p99) |
|---|---|---|---|---|---|---|---|
| 1 | Torus Demo | 3200x1800 | 434 | 2.306 (18.359) | 0.089 (0.18) | 0.72 (1.357) | 2.217 (18.21) |
| 2 | PBR Material Grid | 3200x1800 | 413 | 2.421 (18.352) | 0.106 (0.199) | 0.8 (1.105) | 2.314 (18.233) |
| 3 | Stress Test (100K) | 3200x1800 | 236 | 4.237 (20.024) | 2.874 (3.423) | 2.249 (3.22) | 1.368 (17.039) |
| 4 | Scene Viewer (glTF) | 3200x1800 | 437 | 2.288 (16.771) | 0.083 (0.169) | 0.858 (1.316) | 2.205 (16.63) |
| 5 | Many Lights (1024) | 3200x1800 | 19 | 52.26 (112.878) | 0.105 (0.194) | 95.699 (156.72) | 52.268 (112.713) |
| 6 | Cornell Box (GI) | 3200x1800 | 415 | 2.407 (16.797) | 0.109 (0.233) | 0.652 (0.767) | 2.297 (16.564) |
| 7 | Culling Visualization | 3200x1800 | 381 | 2.623 (16.234) | 0.845 (1.15) | 1.024 (3.287) | 1.802 (16.015) |

`phase/f2` (frame eseguito dal render graph, back-face culling attivo):

| # | Bench | Resolution | FPS | Frame ms (p99) | CPU ms (p99) | GPU ms (p99) | Wait ms (p99) |
|---|---|---|---|---|---|---|---|
| 1 | Torus Demo | 3200x1800 | 391 | 2.559 (17.552) | 0.101 (0.187) | 0.969 (1.585) | 2.457 (17.42) |
| 2 | PBR Material Grid | 3200x1800 | 449 | 2.226 (16.979) | 0.109 (0.205) | 0.574 (1.055) | 2.085 (15.675) |
| 3 | Stress Test (100K) | 3200x1800 | 313 | 3.198 (18.107) | 1.912 (2.349) | 1.805 (2.65) | 1.287 (16.119) |
| 4 | Scene Viewer (glTF) | 3200x1800 | 432 | 2.313 (16.85) | 0.093 (0.182) | 0.83 (1.356) | 2.22 (16.674) |
| 5 | Many Lights (1024) | 3200x1800 | 19 | 51.472 (116.585) | 0.108 (0.198) | 93.844 (151.917) | 51.336 (116.388) |
| 6 | Cornell Box (GI) | 3200x1800 | 437 | 2.286 (14.737) | 0.094 (0.162) | 0.627 (0.87) | 2.191 (14.581) |
| 7 | Culling Visualization | 3200x1800 | 360 | 2.78 (17.365) | 0.722 (0.968) | 1.059 (2.979) | 2.066 (16.83) |

Gli FPS sono dominati dall'attesa del drawable (Wait ≈ Frame) e non dicono
nulla sul costo del frame: si confrontano CPU e GPU ms.

- **Stress Test**: CPU −33% (2,87 → 1,91 ms), GPU −20% (2,25 → 1,81 ms).
  CPU: l'ordinamento delle istanze ora usa chiavi a 64 bit precalcolate
  (mesh, classe di culling, indice) invece di `stable_sort` su struct da 80
  byte; la prima versione di F2, con il lookup del materiale nel comparatore,
  era **+0,5 ms** (3,375 ms, misurato e corretto). GPU: back-face culling.
- **PBR Grid** GPU 0,80 → 0,57 ms, **Culling Viz** CPU 0,85 → 0,72 ms.
- **Torus Demo** GPU 0,72 → 0,97 ms in questo run: rumore. Run alternati
  main/F2 (3 coppie, 600 frame): main 0,883 / 0,967 / 0,897 ms, F2 0,936 /
  0,941 / 0,896 ms. Idem per Scene Viewer (main 0,824–0,875, F2 0,808–0,848)
  e Cornell (main 0,637–0,654, F2 0,623–0,644).
- **Allocazioni GPU nei frame misurati: 0** in tutti i 21 report. Heap CPU
  +2.479…+3.442 blocchi, stesso intervallo di `main` (+1.869…+3.583):
  fluttuazione limitata già documentata in F1, non crescita per frame.
- Il grafo si compila **una volta**: 1 compilazione in 300 frame con
  `--switch-every 20` (15 cambi bench); con `--resize-every 25` una per
  cambio di dimensione del drawable (8 in 200 frame).
- Traffico DRAM stimato dal grafo (O1, `--dump-graph`): 21,97 MiB/frame a
  3200×1800 = solo lo store del drawable (depth memoryless, overlay fuso).

Flag di debug (Debug + validazione, bench 1, solo verifica funzionale, non
prestazioni): `--debug-async-compute` porta la CPU a ~72 ms/frame perché la
sonda ricalcola ogni frame sulla CPU il riferimento esatto (64K elementi ×
128 iterazioni, build Debug); con più submission per frame il GPU ms misura
solo l'ultima submission grafica. Il guadagno reale di F2.5/F2.6 si misura
in F5/F8, come previsto dal piano.

### F2 — residui chiusi dopo il merge (audit, 2026-09-29)

Branch `phase/f2-residues`. Correzioni di sincronizzazione tra frame
(`ImportPerFrame` e barriera di primo uso per le risorse importate
persistenti; il frame grafico N attende il lavoro async del frame N-1), queue
sync nel dump, documentazione. Il frame normale non cambia: 0 punti di
barriera (1 con `--capture`, invariato), quindi le tabelle sopra restano
valide. Verifiche nuove:

- **O7 lato CPU** (Release, bench 1): heap +3.608 blocchi dopo 600 frame e
  +3.651 dopo 6.000; con `--debug-graph-transients --debug-split-encoding`
  +2.083 / +2.265; con `--debug-async-compute` −200 / −633. Nessuna crescita
  per frame; allocazioni GPU 0 in tutti i casi.
- `leaks --atExit` con tutti i flag, `--switch-every 30 --resize-every 40`:
  **0 leak**.
- `--debug-split-encoding`: i 4 chunk di ogni frame girano sul main thread e
  sui 3 worker `phosphor-worker-0…2` (strumentazione temporanea, rimossa).
- Frame con UI + split + async ispezionato: overlay corretto nel command
  buffer di coda.
- Con `--debug-async-compute` la verifica esatta della sonda costa 27,6 ms di
  CPU per frame in Release (72 ms in Debug): costo della verifica, non del
  frame; PASS su 3.120 frame.

## F3 — pipeline: compilazione asincrona e archivi AOT

**2026-09-29** · branch `phase/f3` · M5 Max, macOS 27.2, Xcode 27.0 ·
Release, `--no-vsync --no-ui` salvo dove indicato. Strumento: `FrameTrace`
(`--frame-trace`, riga `SWITCH`): per ogni cambio bench le fasi
(waitIdle, setup, upload, GC, richieste pipeline, compilazione sul render
thread) e i 10 frame successivi confrontati con la soglia del bench di
arrivo, max(1,5 × p99, p99 + 0,5 ms) sui frame stazionari. Il frame del
cambio (caricamento sincrono, lavoro di F22) è riportato a parte; il
caricamento iniziale non è un cambio.

### Nessun hitch al cambio testbench

`--frames 1050 --switch-every 70` (14 cambi, tutti i bench due volte).

| Configurazione | Run | Hitch | Compilazione sul render thread | Chiamate al compilatore | Note |
|---|---|---|---|---|---|
| Archivio (default Release) | 3 | 0 / 0 / 0 | 0 ms | 0 (4 hit) | frame post-cambio peggiore 0,60–0,81 × soglia |
| Senza archivio, cache shader OS calda | 3 | 0 / 0 / 0 | 0 ms | 4 (~2 ms, dalla cache OS) | |
| Senza archivio, compilazioni **a freddo** (`--pipeline-salt` unico) | 6 | 1 / 1 / 0 / 0 / 0 / 0 | 0 ms | 4 (70–91 ms, max 48–69 ms) | i 2 frame segnalati non coincidono con eventi pipeline (vedi sotto) |
| **Controllo negativo** `--pipeline-sync`, a freddo | 2 | 1 / 1 | 29,1 ms / 0,5 ms | 4 | il cambio che richiede una variante la compila sul render thread → hitch |

Fasi del cambio (archivio, media/max ms): totale 18–25 / 55–110, di cui
waitIdle 10–17 / 50–110 (Many Lights: ~95 ms di GPU per frame in volo), setup
6–8 / 17–34, upload 0,8–1,4 / 1,7–3,0, GC 0,03 / 0,06, richieste pipeline
0,00 / 0,02. La compilazione non compare mai nel frame.

I due "hitch" a freddo sono frame con flag 0 (nessuno swap, nessun
fallback): 1,1 ms su Culling Viz (+3…+5 dopo il cambio) e 3,75 ms su PBR (+9,
dopo lo swap al frame +4). Il primo passaggio a PBR, dove la variante v1
compila a freddo in parallelo, ha CPU massima ≤ 0,16 ms in 5 run su 6
(0,26–0,34 ms con l'archivio). Picchi isolati di 5–46 ms compaiono anche in
stato stazionario, lontano dai cambi, **con l'archivio** e su `main` (28–46 ms
su Many Lights/Cornell; un 137 ms su Many Lights senza cambi non riprodotto in
16 coppie alternate `main`/F3, max ≤ 2,5 ms): rumore di sistema che a volte
cade nella finestra di 10 frame. Nota di metodo: in zsh `$RANDOM` nel primo
elemento di una pipeline gira in una subshell e ripete lo stesso valore; i
sali vanno passati espliciti, altrimenti la "compilazione a freddo" viene
servita dalla cache dell'OS.

### Avvio a freddo

`--bench 1 --frames 5`, 3 run per riga; cache shader dell'OS svuotata
spostando da parte `$(getconf DARWIN_USER_CACHE_DIR)com.apple.metal` e
ripristinandola.

| Configurazione | Primo frame dopo il lancio | Pipeline di avvio pronte | Chiamate al compilatore |
|---|---|---|---|
| Archivio, cache OS calda | 138–151 ms | 0,0 ms | **0** (2 hit) |
| Archivio, cache OS vuota | 147–155 ms | 5,5 ms | **0** (2 hit) |
| Senza archivio, cache OS calda | 136–147 ms | 0,0 ms | 2 (1,7–2,3 ms) |
| Senza archivio, cache OS vuota | 166–178 ms | 27–30 ms | 2 (79–85 ms) |

Run completo con archivio, UI, `--switch-every 20` e tutti i flag di debug
F2: 12 richieste, **12 hit, 0 compilazioni**. I primi due frame costano
16–28 ms di CPU in ogni configurazione: è `processEvents` (SDL/Cocoa alla
comparsa della finestra), non il rendering (`frame()` < 0,6 ms, misurato con
strumentazione temporanea).

### O7 e leak

- **Allocazioni GPU nei frame misurati: 0** in tutti i report (bench singoli,
  anche con i flag di debug; con `--switch-every` le allocazioni sono quelle
  dei cambi bench, 99 su `main` come su F3).
- **Heap CPU, 600 contro 6000 frame** (bench 1): senza flag +1974 / +1548
  blocchi; `--debug-graph-transients --debug-split-encoding` F3 +1057 / −43 e
  +1196 / +1174 (`main` +1080 / +3, +2054 / −47); `--debug-async-compute` F3
  +491 / −384, −393 (`main` +6 / −610, −477 / −594). Oscillazioni di qualche
  migliaio di blocchi in entrambe le build.
- **Crescita trovata solo sui run lunghi** (`--debug-async-compute`, 12000
  frame): F3 +35.509 blocchi, `main` **+69.240**: difetto preesistente da F0.
  Due istantanee `malloc_history -allByCount` a 4 minuti di distanza
  (`MallocStackLogging=lite`) mettono tutta la crescita in `processEvents` →
  SDL `Cocoa_PumpEvents` → `-[NSApplication nextEventMatchingMask:…]`: AppKit
  mette oggetti in autorelease e solo `frame()` aveva un pool. Con un pool
  attorno al pompaggio degli eventi: **+6.746 / +8.629** blocchi a 12000
  frame. Il residuo (~0,08 blocchi/frame nelle istantanee) sono continuazioni
  `libdispatch` allocate dentro IOGPU/QuartzCore/FramePacing (commit, attese
  sugli eventi, metriche) e le IOSurface del pool dei drawable (3
  allocazioni da 23 MB ricreate da `CAMetalLayer`): nessuna allocazione del
  motore. Resta da verificare che si stabilizzi su run ancora più lunghi.
- `leaks --atExit` con tutti i flag, `--switch-every 30 --resize-every 40`:
  **0 leak**.

### `bench_all` contro `main`

`main` (`9ed8448`) ricompilato in Release in un worktree temporaneo e
misurato subito prima di F3 (`4d61dd3`), stessa sessione, `tools/bench_all.sh
build/release 600 3`, alimentazione di rete.

`main`:

| # | Bench | Resolution | FPS | Frame ms (p99) | CPU ms (p99) | GPU ms (p99) | Wait ms (p99) |
|---|---|---|---|---|---|---|---|
| 1 | Torus Demo | 3200x1800 | 339 | 2.947 (10.998) | 0.079 (0.176) | 0.896 (3.564) | 2.867 (10.913) |
| 2 | PBR Material Grid | 3200x1800 | 349 | 2.865 (12.316) | 0.123 (0.223) | 0.56 (2.741) | 2.734 (12.138) |
| 3 | Stress Test (100K) | 3200x1800 | 199 | 5.017 (14.713) | 2.482 (4.128) | 2.857 (4.699) | 2.541 (12.565) |
| 4 | Scene Viewer (glTF) | 3200x1800 | 349 | 2.864 (10.968) | 0.099 (0.173) | 0.814 (2.68) | 2.77 (10.844) |
| 5 | Many Lights (1024) | 3200x1800 | 16 | 63.774 (142.23) | 0.112 (0.194) | 113.79 (171.081) | 63.453 (142.08) |
| 6 | Cornell Box (GI) | 3200x1800 | 327 | 3.054 (12.005) | 0.102 (0.234) | 1.047 (3.351) | 2.952 (11.893) |
| 7 | Culling Visualization | 3200x1800 | 307 | 3.26 (12.548) | 0.554 (1.042) | 1.842 (4.461) | 2.705 (12.081) |

`phase/f3`:

| # | Bench | Resolution | FPS | Frame ms (p99) | CPU ms (p99) | GPU ms (p99) | Wait ms (p99) |
|---|---|---|---|---|---|---|---|
| 1 | Torus Demo | 3200x1800 | 298 | 3.358 (12.291) | 0.112 (0.219) | 2.165 (5.879) | 3.237 (12.075) |
| 2 | PBR Material Grid | 3200x1800 | 314 | 3.184 (10.788) | 0.137 (0.234) | 1.702 (4.407) | 3.035 (10.568) |
| 3 | Stress Test (100K) | 3200x1800 | 218 | 4.583 (12.942) | 2.125 (3.492) | 3.59 (6.504) | 2.458 (10.718) |
| 4 | Scene Viewer (glTF) | 3200x1800 | 329 | 3.042 (11.838) | 0.109 (0.238) | 1.417 (4.255) | 2.911 (11.566) |
| 5 | Many Lights (1024) | 3200x1800 | 17 | 59.021 (133.326) | 0.126 (0.365) | 106.435 (158.967) | 58.894 (133.162) |
| 6 | Cornell Box (GI) | 3200x1800 | 240 | 4.166 (14.962) | 0.119 (0.221) | 1.248 (3.525) | 4.047 (14.836) |
| 7 | Culling Visualization | 3200x1800 | 226 | 4.416 (15.727) | 0.781 (1.111) | 2.915 (5.379) | 3.635 (14.776) |

Il pomeriggio il sistema era più rumoroso della mattina (`main` Torus GPU
0,90 ms contro 0,72 nella tabella F2) e le differenze GPU della tabella
(Torus 0,90 → 2,17 ms, Culling 1,84 → 2,92 ms) non si riproducono: GPU ms
senza vsync include la sovrapposizione dei frame in coda (`main` stesso
oscilla 1,38–2,03 ms su Torus in run consecutivi). Verifiche alternate:

- **GPU con vsync** (un frame per volta, 480 frame, 3 run per build,
  gpu ms media): Torus `main` 1,72–1,95 / F3 1,70–1,78; PBR 0,95–1,00 /
  0,65–0,94; Stress 4,53–4,55 / 4,09–4,43; Culling 4,09–4,34 / 4,13–4,18;
  Scene Viewer (0,20–1,11 / 0,82–2,92) e Cornell (1,53–2,61 / 1,10–2,00)
  rumorosi in entrambi i sensi. Nessuna regressione; le varianti
  specializzate non costano GPU in più.
- **CPU senza vsync** (4 coppie alternate): Culling `main` 0,677–0,719 /
  F3 0,682–0,704; Torus 0,116–0,128 / 0,101–0,134; Cornell 0,121–0,131 /
  0,114–0,128. Lo 0,554 → 0,781 della tabella era rumore.
- Stress Test CPU 2,48 → 2,13 ms nella tabella; con vsync 4,49–4,59 /
  4,55–4,64 (a clock bassi): invariato.
- Allocazioni GPU nei frame misurati: 0 in tutti i 42 report.
