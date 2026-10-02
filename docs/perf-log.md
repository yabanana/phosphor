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
    tempo di occupazione. Dal F4.1 il report ha i tempi per pass
    (`passes`, `gpu_pass_sum_ms`: contributo esclusivo di ogni unità alla
    timeline della coda).
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

### F3 — residui chiusi dopo il merge (2026-09-29, branch `phase/f3-residues`)

- **Tutte le varianti controllate sui pixel** (`tools/variant_check.sh
  build/release`): 7 bench × 42 varianti forzate (`--force-variant`) contro
  la pipeline generica nella stessa modalità debug (`--debug-mode`): 252
  varianti compatibili entro 1 livello (al massimo 34 pixel diversi), 42
  incompatibili (modalità illuminata senza un tipo di luce che la scena usa)
  tutte diverse (controllo negativo). Cornell ha ora il pannello emissivo
  sotto il soffitto (il materiale 3 era "emissivo" solo nel commento): i
  bench usano 3 varianti (solo direzionale; solo puntiformi; puntiformi +
  emissivo); riferimento del bench 6 cambiato esattamente dei 12.590 pixel
  del pannello (anche nel percorso generico).
- **Heap CPU su run lunghi** (Release, bench 1): senza UI +11.791 / +9.647 /
  +9.996 blocchi a 6.000 / 30.000 / 60.000 frame; con UI +3.588 / +3.513;
  con `--debug-graph-transients --debug-split-encoding` +1.957 / +1.150;
  con `--debug-async-compute` +6.746 / +8.629 a 12.000 frame e +6.002 a
  24.000.
  Nessuna crescita con la durata: è il riempimento iniziale delle cache di
  sistema (continuazioni `libdispatch`, pool dei drawable), poi costante.
- Il test `pipelines script` fallisce se `shaders/pipelines.mtl4-json` non
  copre più tutte le varianti o le pipeline del motore (va rigenerato con
  `tools/harvest_pipelines.sh`); controllo negativo verificato.

## F4 — Osservabilità (chiusura, 2026-09-29)

M5 Max (Mac17,6), macOS 27.2, alimentazione di rete, Release, `--no-ui`.
Metodo e scoperte in [`opt-log.md`](opt-log.md) (voci F4).

### Baseline di fine fase (storico per commit)

`tools/perf_record.sh build/release 600 3` sul commit `606133f`; righe in
`docs/perf-history.csv` / `docs/perf-history-passes.csv`, tabelle da
`tools/perf_table.py latest`. "GPU pass-sum" = somma dei tempi esclusivi
delle unità temporizzate (F4.1); "GPU ms" = span del command buffer (limite
superiore: include la sovrapposizione dei frame, vedi Many Lights).

**2026-09-29T21:15:08Z** · commit `606133f` · Mac17,6 · Apple M5 Max · macOS 27.2 · AC · 600 frames x 3 runs, --no-ui

vsync on (mean ± sd across runs (CV%); p99 of the median run in parentheses)

| # | Bench | Resolution | Frame ms [p99] | CPU ms [p99] | GPU ms [p99] | GPU pass-sum ms | GPU allocs |
|---|---|---|---|---|---|---|---|
| 1 | Torus Demo | 3200x1800 | 8.347 ± 0.014 (0.2%) [10.306] | 0.36 ± 0.037 (10.2%) [0.588] | 2.18 ± 0.04 (1.8%) [2.692] | 2.167 ± 0.04 (1.9%) | 0 |
| 2 | PBR Material Grid | 3200x1800 | 8.371 ± 0.022 (0.3%) [11.316] | 0.368 ± 0.098 (26.6%) [0.687] | 1.189 ± 0.178 (15.0%) [2.309] | 1.177 ± 0.177 (15.0%) | 0 |
| 3 | Stress Test (100K) | 3200x1800 | 8.334 ± 0.002 (0.0%) [10.567] | 4.724 ± 0.012 (0.3%) [5.874] | 5.098 ± 0.054 (1.1%) [6.052] | 5.037 ± 0.071 (1.4%) | 0 |
| 4 | Scene Viewer (glTF) | 3200x1800 | 8.356 ± 0.041 (0.5%) [9.745] | 0.388 ± 0.076 (19.6%) [0.523] | 1.702 ± 0.081 (4.7%) [2.664] | 1.69 ± 0.078 (4.6%) | 0 |
| 5 | Many Lights (1024) | 3200x1800 | 57.324 ± 0.974 (1.7%) [120.744] | 0.15 ± 0.047 (31.0%) [0.186] | 104.954 ± 2.446 (2.3%) [165.982] | 63.353 ± 1.709 (2.7%) | 0 |
| 6 | Cornell Box (GI) | 3200x1800 | 8.333 ± 0 (0.0%) [9.72] | 0.236 ± 0.01 (4.3%) [0.495] | 0.824 ± 0.014 (1.7%) [1.34] | 0.807 ± 0.011 (1.3%) | 0 |
| 7 | Culling Visualization | 3200x1800 | 8.333 ± 0 (0.0%) [9.71] | 0.926 ± 0.04 (4.3%) [1.415] | 2.964 ± 0.144 (4.9%) [3.979] | 2.953 ± 0.143 (4.9%) | 0 |

vsync off (mean ± sd across runs (CV%); p99 of the median run in parentheses)

| # | Bench | Resolution | Frame ms [p99] | CPU ms [p99] | GPU ms [p99] | GPU pass-sum ms | GPU allocs |
|---|---|---|---|---|---|---|---|
| 1 | Torus Demo | 3200x1800 | 3.89 ± 0.244 (6.3%) [11.502] | 0.25 ± 0.006 (2.4%) [0.487] | 2.402 ± 0.397 (16.5%) [3.626] | 1.483 ± 0.23 (15.5%) | 0 |
| 2 | PBR Material Grid | 3200x1800 | 4.122 ± 0.147 (3.6%) [12.033] | 0.304 ± 0.04 (13.3%) [0.544] | 1.495 ± 0.053 (3.6%) [2.674] | 1.014 ± 0.03 (2.9%) | 0 |
| 3 | Stress Test (100K) | 3200x1800 | 4.212 ± 0.067 (1.6%) [10.759] | 2.394 ± 0.04 (1.7%) [4.134] | 2.879 ± 0.461 (16.0%) [5.759] | 2.33 ± 0.235 (10.1%) | 0 |
| 4 | Scene Viewer (glTF) | 3200x1800 | 4.181 ± 0.024 (0.6%) [9.674] | 0.178 ± 0.028 (15.7%) [0.416] | 2.115 ± 0.087 (4.1%) [3.11] | 1.375 ± 0.04 (2.9%) | 0 |
| 5 | Many Lights (1024) | 3200x1800 | 56.45 ± 0.306 (0.5%) [120.176] | 0.116 ± 0.002 (1.3%) [0.173] | 103.677 ± 0.74 (0.7%) [166.619] | 67.51 ± 0.42 (0.6%) | 0 |
| 6 | Cornell Box (GI) | 3200x1800 | 2.582 ± 0.104 (4.0%) [10.725] | 0.106 ± 0.002 (2.3%) [0.21] | 0.572 ± 0.015 (2.7%) [1.962] | 0.399 ± 0.008 (2.0%) | 0 |
| 7 | Culling Visualization | 3200x1800 | 3.127 ± 0.232 (7.4%) [11.699] | 0.772 ± 0.075 (9.7%) [1.145] | 1.824 ± 0.416 (22.8%) [5.24] | 1.357 ± 0.251 (18.5%) | 0 |


### Profilazione spenta contro `main`

`main` (`791fb36`) in un worktree Release accanto a `phase/f4`; per ogni bench
3 ripetizioni alternate main / F4 (timing acceso, default) / F4
`--no-gpu-timing`; CPU senza vsync (600 frame), GPU con vsync (600 frame;
Many Lights 120). Intervallo min–max (mediana).

| Bench | Config | Metric | main | F4 timing on | F4 timing off |
|---|---|---|---|---|---|
| 1 | novsync | CPU ms | 0.047–0.231 (0.111) | 0.115–0.202 (0.124) | 0.095–0.195 (0.103) |
| 1 | vsync | GPU ms | 1.560–1.684 (1.613) | 1.506–1.727 (1.659) | 1.554–1.587 (1.566) |
| 2 | novsync | CPU ms | 0.145–0.303 (0.299) | 0.305–0.329 (0.326) | 0.292–0.308 (0.296) |
| 2 | vsync | GPU ms | 0.574–0.588 (0.575) | 0.623–0.630 (0.624) | 0.576–0.820 (0.590) |
| 3 | novsync | CPU ms | 4.867–4.907 (4.896) | 4.439–4.867 (4.864) | 4.770–4.903 (4.883) |
| 3 | vsync | GPU ms | 5.227–5.338 (5.321) | 5.315–5.662 (5.383) | 5.390–5.423 (5.393) |
| 4 | novsync | CPU ms | 0.141–0.198 (0.194) | 0.149–0.243 (0.231) | 0.136–0.283 (0.270) |
| 4 | vsync | GPU ms | 1.161–1.465 (1.397) | 1.140–1.333 (1.142) | 1.081–1.195 (1.084) |
| 5 | novsync | CPU ms | 0.114–0.140 (0.128) | 0.126–0.632 (0.132) | 0.096–0.124 (0.110) |
| 5 | vsync | GPU ms | 92.613–96.337 (94.897) | 94.155–102.437 (94.667) | 93.845–105.994 (93.929) |
| 6 | novsync | CPU ms | 0.127–0.164 (0.159) | 0.157–0.171 (0.168) | 0.119–0.163 (0.162) |
| 6 | vsync | GPU ms | 0.875–1.292 (1.077) | 1.207–1.322 (1.270) | 1.107–1.335 (1.194) |
| 7 | novsync | CPU ms | 0.495–0.751 (0.687) | 0.695–0.823 (0.794) | 0.712–0.775 (0.755) |
| 7 | vsync | GPU ms | 3.303–3.567 (3.464) | 3.431–3.458 (3.455) | 3.380–3.441 (3.395) |

- **Timing spento** = `main` entro il rumore. Righe sospette ricontrollate
  con 6 coppie alternate `main` / F4 spento (CPU senza vsync): Culling
  0,659–1,03 / 0,602–0,964 (F4 più basso in 5 coppie su 6), Scene Viewer
  0,165–0,216 / 0,151–0,229.
- **Timing acceso** (default): +0,01–0,03 ms di CPU e fino a +0,05 ms di
  GPU sulle scene leggere (PBR 0,624 contro 0,575–0,590): encoder anchor con
  barriera, un timestamp per unità, risoluzione CPU. Non misurabile su
  Stress Test e Many Lights.
- Allocazioni GPU nei frame misurati: **0** in tutti i 126 run.
- Pixel: `tools/visual_check.sh` 0 pixel diversi e 0 messaggi con timing
  acceso, spento, unfused, seriale, tutti i flag F2, costo noto.

### Costi dichiarati con la profilazione accesa

| Strumento | Costo misurato |
|---|---|
| Timestamp per pass (default) | CPU +0,01–0,03 ms, GPU ≤ +0,05 ms (scene leggere) |
| Tracy (`PHOSPHOR_TRACY=ON`, on demand) | Stress Test CPU 2,21–2,30 ms senza client, 2,21–2,26 collegato, contro 2,26–2,37: non misurabile |
| Layer di cattura (`--gpu-capture`, archivio disattivato) | CPU invariata (2,21–2,29 contro 2,25–2,31), GPU ≤ +0,1 ms |
| Cattura di un frame | 58–80 ms; Stress Test 559 MB di documento |
| Overlay (per pass, `--no-vsync --gpu-timing-serial`, p50) | Stress Test: overdraw 0,70 + composizione 0,08; luci 0,74 + 0,08; costo per tile 0,72 + 1,20 + 0,05 + 0,09 ms. Many Lights: overdraw 0,13 + 0,09; luci 21,9 + 0,11 (ripete il ciclo sulle 1024 luci); costo per tile 21,6 + 0,07 + 0,05 + 0,09 ms |

### Controllo negativo e coerenza

- Pass di costo noto (`--debug-gpu-cost N --no-vsync --gpu-timing-serial`,
  p50, 2 run): 2,92 / 6,11 / 11,81 ms per 4000 / 8000 / 16000 iterazioni
  (lineare, ~0,74 ms per 1000), forward 0,216–0,218 ms. Meccanismo rotto
  apposta: il forward ingloba il pass noto (9,5 / 7,5 / 14,7 ms).
- Somma delle unità contro GPU ms del command buffer (vsync): Torus 2,513 /
  2,522, Stress Test 4,463 / 4,510; Many Lights seriale 54,316 / 54,324.
- Tracy: media delle zone GPU = report (4,9541 ms, 300 frame).

### O7 e leak

- Heap CPU (bench 1, timing acceso): +2635 / −286 / +1455 blocchi e
  +178 / +71 / +523 KB a 600 / 6000 / 30000 frame (i buffer di misura, 12 B
  per frame; `main` +2495 blocchi, +138 KB a 30000). Prima della correzione
  del timestamp di commit: +1,72 MB a 30000 (driver, vedi opt-log).
- Tracy on demand senza client: footprint 955 → 957 MB in 50 s (build
  normale 949 → 951).
- `leaks --atExit` con tutti i flag, cambio bench, resize e overlay: **0
  leak**. Con `--gpu-capture`: 10 leak (784 B) nel framework di cattura di
  Apple anche senza catture, +2 per cattura.

## OPT-0 — Caratterizzazione del SoC (chiusura, 2026-09-30)

OPT-0 aggiunge la suite `bench/soc`, il modello di costo e il roofline; l'unica
modifica al motore è il campo `work` per pass nel report (schema v3, dati CPU
calcolati a fine misura). Verifica "motore invariato":

- `ctest` verde (macOS e Linux in container: build, test, `soc_model`);
  `metal_syntax_check` verde in un container x86_64 (su Linux arm64 fallisce
  il controllo d'architettura di metal-cpp, preesistente); `visual_check`
  0 pixel diversi e 0 messaggi sui 7 bench.
- `bench_all --stats` A/B/A (Release, `main` = 8d0855f in un worktree,
  600 frame × 3 run, `--no-vsync --no-ui`, a batteria):

| # | Bench | Frame ms main / branch / main | CPU ms main / branch / main | GPU ms main / branch / main |
|---|---|---|---|---|
| 1 | Torus Demo | 10.09 / 11.56 / 11.31 | 0.209 / 0.285 / 0.296 | 0.976 / 0.972 / 0.950 |
| 2 | PBR Material Grid | 11.03 / 11.25 / 11.00 | 0.247 / 0.283 / 0.316 | 0.484 / 0.506 / 0.498 |
| 3 | Stress Test (100K) | 11.46 / 11.39 / 11.47 | 4.928 / 4.426 / 5.041 | 5.252 / 5.416 / 5.177 |
| 4 | Scene Viewer (glTF) | 11.46 / 11.44 / 11.33 | 0.228 / 0.282 / 0.291 | 0.756 / 0.777 / 0.772 |
| 5 | Many Lights (1024) | 54.45 / 54.28 / 53.81 | 0.127 / 0.197 / 0.119 | 94.439 / 93.996 / 92.834 |
| 6 | Cornell Box (GI) | 11.33 / 11.36 / 11.37 | 0.289 / 0.285 / 0.295 | 0.587 / 0.584 / 0.600 |
| 7 | Culling Visualization | 11.68 / 11.63 / 11.59 | 0.930 / 0.883 / 0.914 | 2.285 / 2.120 / 2.181 |

  L'unica riga sospetta (CPU di Many Lights 0,197 ms sul branch contro
  0,127/0,119) non si riproduce in 4 run singoli alternati da 600 frame:
  main 0,134–0,152 ms, branch 0,135–0,164 ms. Frame e GPU entro il rumore su
  tutti i bench. Il frame a ~11 ms anche senza vsync è lo stesso su `main`
  (attesa del drawable).

Numeri del SoC, modello e roofline: [`soc-model.md`](soc-model.md),
[`img/roofline-m5max.md`](img/roofline-m5max.md); scoperte e soglie in
[`opt-log.md`](opt-log.md) (sezioni OPT-0).

## OPT-1 — Memoria, grafo e banda (chiusura, 2026-10-01)

OPT-1 si misura su **scenari di grafo** (`--graph-scenario N`: deferred,
forward+, catena di post, compute asincrono; pass sintetici di lavoro e byte
noti, immagini deterministiche) perché il grafo del motore ha 1–2 pass.
Piani offline in `shaders/graph-plans.json` scelti **per misura**
(`tools/graph_opt --top` + `tools/graph_select.py`); uscita con
`tools/graph_scenarios.sh` (M5 Max, Release, alimentazione, macchina quieta,
`--no-vsync --no-ui`, 3 giri a ordine ruotato × 600 frame; `off` = compilatore
di fine F4, `greedy` = politiche OPT-1 senza piano, `plan` = piano adottato):

| Scenario | Modo | Frame GPU p50 ms (CV) | Δ frame | DRAM MiB (Δ) | Heap MiB (Δ) | Render pass | Memoryless | Barriere | Immagine | Validazione |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 deferred | off | 2,700 (0,3%) | — | 609,8 | 167,0 | 8 | 0 | 36 | rif. | 0 |
| | greedy | 2,699 (0,0%) | −0,0% | 609,8 (0) | 167,0 (0) | 8 | 0 | 36 | 0 px | 0 |
| | **plan** | **2,473** (0,1%) | **−8,4%** | 609,8 (0) | 167,0 (0) | 8 | 0 | 41 | 0 px | 0 |
| 1 forward+ | off | 4,005 (0,0%) | — | 502,3 | 106,1 | 7 | 0 | 23 | rif. | 0 |
| | greedy | 4,006 (0,1%) | +0,0% | 502,3 (0) | 106,1 (0) | 7 | 0 | 23 | 0 px | 0 |
| | **plan** | **3,865** (0,4%) | **−3,5%** | 502,3 (0) | 106,1 (0) | 7 | 0 | 26 | 0 px | 0 |
| 2 post | off | 3,048 (0,1%) | — | 568,3 | 93,3 | 4 | 0 | 70 | rif. | 0 |
| | greedy | 3,049 (0,0%) | +0,0% | 568,3 (0) | 93,3 (0) | 4 | 0 | 70 | 0 px | 0 |
| | **plan** | 3,033 (0,1%) | −0,5% | **547,2 (−3,7%)** | **85,9 (−8,0%)** | 4 | 0 | 67 | 0 px | 0 |
| 3 async | off | 3,456 (0,0%) | — | 538,8 | 161,4 | 8 | 0 | 12 | rif. | 0 |
| | greedy | 3,453 (0,1%) | −0,1% | 538,8 (0) | 161,4 (0) | 8 | 0 | 12 | 0 px | 0 |
| | **plan** | **3,377** (0,2%) | **−2,3%** | **454,4 (−15,7%)** | **117,5 (−27,2%)** | 7 | 3 | 11 | 0 px | 0 |

(DRAM stimata dal grafo con la cattura del drawable inclusa; heap = memoria
di picco dei transitori.) Attribuzione (scenario 0/1, stessi ordini dei
piani): solo l'ordine con le barriere di fine F4 −3,5% / −2,1%; con le
barriere minime OPT-1.4 −8,4% / −3,6%; le barriere minime con l'ordine
greedy 0%. Obiettivo ROADMAP (−25% byte, −20% picco) raggiunto solo per il
picco dello scenario 3; motivi in [`opt-log.md`](opt-log.md) ("OPT-1 —
Risultati"). Criterio di adozione (≥ −3% frame) soddisfatto sugli scenari 0
e 1 su M5 Max; **T0 non disponibile** (O12).

**Motore invariato** (default `--graph-opt off`):
- `ctest` verde; `visual_check` 0 pixel e 0 messaggi sui 7 bench con i flag
  F2 (nessuno, `--debug-graph-transients`, `--debug-split-encoding`,
  `--debug-async-compute`, `--gpu-timing-unfused`, i tre debug insieme) e con
  `--graph-opt greedy`/`plan` (anche insieme ai tre debug);
  `--switch-every 20 --resize-every 45` con UI sotto validazione in `off` e
  `greedy`: 0 messaggi, 7 compilazioni; scenari con piano + UI + resize ogni
  30 frame: 0 messaggi, piano applicato a ogni compilazione.
- `bench_all --stats` A/B/A (Release, `main` = 98bace4 in un worktree,
  600 frame × 3 run, `--no-vsync --no-ui`, alimentazione):

| # | Bench | Frame ms main / branch / main | CPU ms main / branch / main | GPU ms main / branch / main |
|---|---|---|---|---|
| 1 | Torus Demo | 2,37 / 2,48 / 2,33 | 0,104 / 0,140 / 0,116 | 0,776 / 0,968 / 0,964 |
| 2 | PBR Material Grid | 2,37 / 2,40 / 2,42 | 0,141 / 0,134 / 0,142 | 0,549 / 0,554 / 0,558 |
| 3 | Stress Test (100K) | 3,23 / 3,14 / 3,35 | 1,966 / 1,960 / 1,994 | 1,794 / 1,793 / 1,800 |
| 4 | Scene Viewer (glTF) | 2,36 / 2,42 / 2,31 | 0,115 / 0,112 / 0,112 | 0,743 / 0,830 / 0,757 |
| 5 | Many Lights (1024) | 54,92 / 50,79 / 50,64 | 0,117 / 0,117 / 0,114 | 100,13 / 93,13 / 92,05 |
| 6 | Cornell Box (GI) | 2,37 / 2,50 / 2,38 | 0,115 / 0,114 / 0,113 | 0,635 / 0,644 / 0,626 |
| 7 | Culling Visualization | 2,80 / 2,85 / 2,67 | 0,819 / 0,842 / 0,837 | 1,162 / 1,315 / 1,212 |

  Righe sospette (Torus CPU, Culling GPU) ripetute con 5 coppie di run
  singoli alternati: Culling GPU mediana 1,218 main / 1,195 branch, Torus
  0,999 / 1,019, span 1,151/1,159 e 1,014/1,015, CPU entro ±2%: rumore (le
  medie tra run hanno CV 4–14%; pass < 1 ms senza clock saturi, OPT-0).
- O7: `GPU allocations 0` nei frame misurati (testbench e scenari);
  heap CPU a 600/6000 frame identico a `main` (bench 1: main +2701/+3619
  blocchi, branch +2686/+3615); negli scenari i byte crescono con unità ×
  frame (buffer dei campioni della misura). `leaks --atExit` 0 con i tre
  flag di debug + `greedy`, con gli scenari 0 (3 viste), 2 e 3 con piano.

## F5 — GPU scene persistente e submission guidata dalla GPU (chiusura, 2026-10-01)

M5 Max, macOS 27.2, Release, 3200×1800, alimentazione, `caffeinate -d`.
**Condizioni**: durante le misure finali la macchina non era quieta (altri
processi dell'utente: Codex Computer Use, un emulatore Android, WindowServer
al 50–60% di CPU): i tempi sono più rumorosi delle fasi precedenti (frame
dei bench piccoli ~3,2–4,2 ms contro ~2,4 ms in OPT-1) e i p99 sono un limite
superiore. **T0 non disponibile** (O12).

### Bench 8 "1M Instances (dynamic)", `--gpu-driven on` (default)

1M istanze in moto (moto procedurale GPU, 1% aggiornato dalla CPU ogni frame
= 10.000 record, 100K satelliti a profondità ≤ 3), 693.677 visibili dopo il
frustum culling, 24 bucket. 600 frame + 120 di riscaldamento, 3 run:

| Misura | run 1 / 2 / 3 | CV |
|---|---|---|
| Frame p50 (p99), `--no-vsync` | 5,94 (15,10) / 5,85 (14,14) / 5,99 (15,01) ms | 1,2% (3,6%) |
| CPU p50 (p99) | 2,03 (3,65) / 2,06 (3,72) / 2,01 (3,77) ms | 1,4% (1,6%) |
| GPU p50 (p99), span del command buffer | 7,55 (11,23) / 7,62 (11,47) / 7,36 (11,07) ms | 1,8% (1,8%) |
| Somma dei pass GPU p50 (p99) | 5,53 (7,95) / 5,54 (7,83) / 5,56 (7,26) ms | 0,3% (4,8%) |
| Con vsync (120 Hz) | 120,0 fps, frame p99 9,48 ms, max < 10,7 ms | — |

Comandi CPU della scena 66 per frame (min = max); upload 0,96 MB/frame
(10.000 record × 96 B + costanti e luci); 0 allocazioni GPU; fasi CPU p50:
simulazione 0,35 ms (il bench aggiorna il suo 1%), sync dello store 1,44 ms,
prepare 0,02, encoding del grafo 0,07, submit 0,04. Pass GPU (frame
serializzati, p50 / p99): Scene update 0,005 / 0,034 ms, Scene transforms
0,40 / 0,76, Instance cull 0,24 / 0,58, Draw build 0,009 / 0,079, Forward
4,66 / 5,86.

Percorso CPU di prima (spike S1, `main`, 1M cubi): CPU 17,7 ms a scena ferma,
30,2 ms con tutte le istanze in moto, 152,6 MiB caricati per frame.

### O8: comandi CPU costanti

Bench 8, comandi CPU della scena per frame (min = max in ogni run):

| N istanze | K = 8 (24 bucket) | K = 64 (192) | K = 1024 (1.858–3.072) |
|---|---|---|---|
| on, 10K / 100K / 1M | 90 / 90 / 90 | 90 / 90 / 90 | 90 / 90 / 90 |
| off, 10K / 100K / 1M | 97 / 97 / 97 | 265 / 265 / 265 | 1.931 / 3.139 / 3.145 |

(Misurato prima di codificare solo i livelli della gerarchia presenti: oggi
on = 66 nel bench 8, 38 sui bench senza gerarchia, sempre indipendente da N.)
Controllo negativo: `off` cresce con i bucket.

### Byte di delta proporzionali ai cambi

Bench 8, `--dynamic-cpu P` (on): 0% → 1,4 KiB/frame (costanti e luci; sync
0,003 ms, CPU del frame 0,10 ms), 0,1% → 1.000 record, +94 KiB; 1% → 10.000,
+937 KiB; 10% → 100.000, +9,2 MiB (96 B per record); 100% → copie intere
(1M istanze + nodi + moto, 113 MiB). Bench 3, 4, 7 (statici): 1,1 KiB/frame,
0 record. `--churn 1000` (1.000 spawn + 1.000 despawn per frame): 14.010
record, 1,3 MiB/frame, 0 cambi di struttura, comandi CPU costanti.

### Motore invariato (bench 1–7)

`bench_all --stats` A/B/A (600 frame × 3 run, `--no-vsync --no-ui`; frame ms
p50; CPU e GPU ms p50; macchina non quieta):

| # | Bench | Frame main / off / on / main | CPU main / off / on / main | GPU main / off / on / main |
|---|---|---|---|---|
| 1 | Torus Demo | 4,19 / 3,85 / 3,69 / 5,84 | 0,158 / 0,184 / 0,169 / 0,155 | 1,59 / 1,49 / 1,52 / 1,42 |
| 2 | PBR Material Grid | 3,24 / 3,23 / 3,56 / 2,81 | 0,146 / 0,172 / 0,252 / 0,132 | 0,87 / 0,95 / 1,03 / 0,50 |
| 3 | Stress Test (100K) | 4,22 / 4,25 / 4,35 / 4,24 | **1,955 / 0,178 / 0,263 / 1,963** | 2,96 / 4,32 / 3,30 / 3,98 |
| 4 | Scene Viewer (glTF) | 3,36 / 3,92 / 4,03 / 3,26 | 0,125 / 0,195 / 0,210 / 0,064 | 1,10 / 1,39 / 1,39 / 0,78 |
| 5 | Many Lights (1024) | 56,3 / 54,9 / 59,4 / 54,8 | 0,128 / 0,180 / 0,332 / 0,131 | 103,1 / 104,6 / 109,1 / 103,7 |
| 6 | Cornell Box (GI) | 3,51 / 3,63 / 5,12 / 4,15 | 0,152 / 0,195 / 0,169 / 0,143 | 1,11 / 1,22 / 1,91 / 1,18 |
| 7 | Culling Visualization | 4,49 / 4,26 / 4,24 / 4,18 | **0,829 / 0,194 / 0,260 / 0,727** | 2,09 / 2,15 / 3,63 / 2,63 |

- CPU: Stress Test −87/−91%, Culling Viz −73/−65%; i bench piccoli +0,02–0,12
  ms (passi della scena e store). GPU senza vsync: span sovrapposti, rumore
  ±50% tra le due run di `main`.
- GPU per pass (frame serializzati, 2 run alternate, prima delle ultime
  ottimizzazioni): costo fisso dei pass della scena ~0,07 ms (off) / 0,15 ms
  (on) per frame, poi ridotto a ~0,02 / 0,08 ms (passi della gerarchia
  solo se presenti, nessun reset dell'ICB); il Forward dello Stress Test
  sembrava +0,9 ms (2,8 contro 1,83 ms) ma è un artefatto DVFS: con 2 ms di
  lavoro CPU per frame, come `main`, scende a 1,77 ms; con vsync main e F5
  danno 4,9–5,3 ms entrambe (`opt-log.md`, "F5 — Scoperte dell'integrazione").
- O7: 0 allocazioni GPU nei frame misurati su tutti i bench, anche con
  `--churn 1000`/`10000`; heap CPU del bench 8 con churn piatto con
  `--no-gpu-timing` (+13,5 KB a 600 frame, +11,2 KB a 6000); con i timestamp
  lo storage delle misure per pass cresce ~30 B/frame (anche su `main`,
  ~9 B/frame). `leaks --atExit` 0 (switch di tutti gli 8 bench, on con
  `--debug-graph-transients --debug-async-compute`, off con
  `--debug-split-encoding`).
- Correttezza: `visual_check` (API + shader validation) off e on, senza flag
  e con `--debug-graph-transients`, `--debug-split-encoding`,
  `--debug-async-compute` e tutti e tre: 8/8 bench, 0 pixel, 0 messaggi
  (riferimenti F5 dal modo off; bench 2–7 = `main`, bench 1 differisce di 9
  pixel ±1 per il flag ICB della pipeline, provato); 10 run identiche per
  bench; `--debug-gpu-scene` PASS su 8 bench off/on e con soglie di culling,
  ogni controllo negativo FAIL; `--switch-every 20 --resize-every 45` con UI
  sotto validazione: 0 messaggi; `archive_check` (riferimenti Release),
  `variant_check` (284 varianti compatibili, 52 incompatibili), 
  `hot_reload_check` verdi; `hitch_check` intermittente anche su `main` per
  i picchi del pompaggio eventi SDL/Cocoa (macchina non quieta, vedi
  opt-log).

## F6 — Mesh shader e culling a due fasi (2026-10-02, branch `phase/f6`)

### Preset congelato del gate (dichiarato prima della misura)

Preset `culling-viz-f6-1920x1080`, fissato prima delle misure del gate e non
modificato dopo:

- scena: bench 7 `--culling-script` (edifici suddivisi 12×12 per faccia,
  1728 triangoli, 10 000 edifici + piano + muro + sfera; loop di 20 s con
  muro che scompare/ricompare, 3 tagli di camera, salita, pan 360°, sfera a
  150 unità/s, churn 20 edifici ogni 0,25 s nella stessa cella); seed fissi
  nel codice (42 per le altezze, xorshift `0x9E3779B97F4A7C15` per il churn);
- risoluzione interna = output 1920×1080 (`--resolution 1920x1080`, drawable
  verificato all'avvio), scala 1, niente DRS né interpolazione;
- `--geometry-path mesh --meshlet-cull two-phase --hiz-path auto` (compute
  SIMD-group, scelta S3), cook `standard-64v124t`, `--fixed-timestep`,
  `--no-ui`, `--warmup 120 --frames 1200` (un loop intero), vsync on (60 fps
  reali: il display è a 120 Hz, frame presentati realmente renderizzati);
- 3 repliche, macchina quieta (un solo processo GPU), alimentazione AC.
  Riportati anche: vsync off (costi CPU/GPU), fallback `--force-family
  apple9` (stesso preset), backend `--hiz-path sampler` (Apple10) e il
  percorso indexed F5 sullo stesso preset come riferimento.

### Gate: risultati (preset congelato, M5 Max, 3 repliche per configurazione)

Stato della macchina dichiarato: **non quieta** (emulatore Android dalle
12:17, sessione Codex Computer Use con registrazione dello schermo per
tutta la giornata, carico crescente nel pomeriggio). Tre serie, tutte con
il preset sopra; fps = frame realmente renderizzati e presentati.

| Serie (ora, codice) | Configurazione | fps | frame p95 ms | p99 | CPU ms | GPU pass sum ms |
|---|---|---|---|---|---|---|
| 1 (14:05, 0734cdc) | mesh two-phase, compute | 120,0 ×3 | 9,07 / 9,24 / 9,90 | 9,38–10,99 | 0,35–0,39 | 2,55–3,21 |
| 1 | `--force-family apple9` | 120,0 ×3 | 9,09 / 9,23 / 9,68 | 9,39–11,14 | 0,35–0,37 | 2,74–3,10 |
| 1 | `--hiz-path sampler` | 119,9–120,0 | 9,10 / 9,13 / 9,55 | 9,41–10,58 | 0,35–0,37 | 3,03–3,19 |
| 1 | indexed F5 (riferimento) | 119,9–120,0 | 9,01 / 9,08 / 10,03 | 9,26–11,19 | 0,28–0,32 | 3,84–4,68 |
| 1 | mesh, vsync off | 255–297 | 10,5–11,8 | 12,7–15,1 | 0,24–0,26 | 1,12–1,35 |
| 2 (`f6_check --perf`, 1d727ec) | nativo / apple9 / sampler | – | 10,48–12,46 (9 run) | – | – | – |
| 3 (14:50, HEAD f3505b3) | mesh two-phase, compute | 118,7 / 117,6 / 86,4 | 12,41 / 13,07 / 16,03 | 14,0–16,9 | 0,27–0,29 | 1,66–2,25 |
| 3 | `--force-family apple9` | 119,1 / 117,3 / 86,9 | 12,07 / 13,11 / 16,28 | 13,7–16,9 | 0,27–0,28 | 1,70–2,06 |
| 3 | `--hiz-path sampler` | 118,8 / 115,9 / 86,8 | 11,12 / 13,53 / 16,52 | 14,0–17,2 | 0,28 | 1,71–2,20 |
| 3 | indexed F5 (riferimento) | 119,1 / 112,4 / 88,1 | 12,08 / 13,62 / 16,18 | 13,8–16,9 | 0,23 | 3,29–3,78 |
| 3 | mesh, vsync off | 199–212 | 13,9–14,4 | 16,5–18,2 | 0,20 | 1,33–1,43 |

- **Gate locale superato** in tutte le esecuzioni mesh (27 su 27 con p95 ≤
  16,67 ms), nativo, fallback Apple9 e sampler Apple10; 0 allocazioni GPU nei
  frame misurati, blocchi dell'heap CPU piatti, 0 overflow. Target T0 fisico:
  `EXTERNAL_VALIDATION_PENDING`; il fallback Apple9 è stato eseguito sul M5
  (percorso software, non una misura M3).
- Nella serie 3 la terza replica di **tutte** le configurazioni, indexed
  compreso, scende a ~87 fps con p95 ~16,0–16,5 ms mentre CPU (0,28 ms) e GPU
  (1,7 ms) del motore restano bassi: il limite è la presentazione con il
  compositore carico (WindowServer al 76% per la registrazione dello schermo
  di un altro agente). Il margine pulito è stato rimisurato in serata
  (serie 4 sotto).

#### Serie 4 (21:18, HEAD 0c097c4, macchina più libera)

Emulatore Android chiuso; restavano l'app ChatGPT/Codex (renderer al
~125% + ~100% di CPU, non chiudibili) e la GPU a riposo con picchi
intermittenti al ~30% (campioni `ioreg` `Device Utilization %` prima della
misura); alimentazione AC, `powermode 2`, nessun avviso termico.  Stesso
preset, 3 repliche per configurazione in ordine ruotato:

| Configurazione | fps | frame p95 ms | p99 | CPU ms | GPU pass sum ms | alloc |
|---|---|---|---|---|---|---|
| mesh two-phase, compute | 120,0 ×3 | 8,51 / 8,52 / 8,52 | 8,62–8,64 | 0,25–0,29 | 2,44–2,52 | 0 |
| `--force-family apple9` | 120,0 ×3 | 8,51 / 8,52 / 8,53 | 8,62–8,64 | 0,25–0,28 | 2,36–2,53 | 0 |
| `--hiz-path sampler` | 120,0 / 120,0 / 119,8 | 8,49 / 8,50 / 8,52 | 8,61–8,63 | 0,25–0,27 | 2,46–2,68 | 0 |
| indexed F5 (riferimento) | 120,0 ×3 | 8,49 / 8,49 / 8,50 | 8,54–8,58 | 0,21–0,23 | 4,74–4,88 | 0 |
| mesh, vsync off | 80,0 ×3 | 16,49–16,53 | 16,63–16,69 | 0,27–0,28 | 2,48–2,56 | 0 |

- **Gate superato con margine pulito**: 120 fps reali (il display a 120 Hz
  presenta ogni frame), p95 8,49–8,53 ms in tutte le 12 esecuzioni con vsync,
  0 overflow, 0 allocazioni GPU.  Le serie 1–3 restano valide come esito del
  gate, la 4 è quella che misura il margine.
- **Vsync off limitato dalla presentazione**: 80,0 fps fissi con attesa del
  drawable 12,2 ms in media mentre CPU (0,27 ms) e GPU (2,5 ms) sono minimi;
  il controllo indexed senza vsync dà lo stesso 80,0 fps (attesa 12,3 ms), e
  il bench 3 senza vsync 88–90 fps: è il compositore in questa finestra
  (al mattino 255–297 fps), non il motore.  Con la presentazione limitata la
  GPU scala le frequenze (DVFS): i tempi GPU senza vsync di questa serata
  non sono confrontabili con quelli del mattino (vedi S2 in `opt-log.md`).
- Costo del motore sul preset (serie 1, vsync off): GPU pass sum 1,12–1,35 ms
  contro 2,07 ms dell'indexed sulla stessa scena senza vsync (S2: −39%), CPU
  0,24–0,26 ms per frame; comandi CPU costanti (109 per frame, O8).
- Spike S1–S4, scelte e misure di S2 per scena: `opt-log.md`, "F6 — Spike".
