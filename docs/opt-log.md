# Phosphor — Opt log

Registro degli spike di ottimizzazione e delle misure che hanno deciso una
scelta, riusciti o no (ROADMAP, "Fasi OPT: metodo comune", punto 6). Ogni
voce: ipotesi, metodo, misure, decisione. Le baseline e le chiusure di fase
sono in [`perf-log.md`](perf-log.md).

---

## F2.3 — Legalità e costo delle barriere Metal 4 (spike)

**2026-09-29** · Apple M5 Max (Apple10, 40 core GPU), macOS 27.2 (26B5091g),
build Debug · strumento: `bench/barrier_spike` (tabella completa di 110 righe,
metodo e messaggi esatti in [`bench/barrier_spike/README.md`](../bench/barrier_spike/README.md)).

**Ipotesi** (S-TBDR-5, lettura delle Feature Set Tables): le barriere con
fragment/tile sul lato "after"/consumatore non sono supportate da Apple3 ad
Apple10; il render graph dovrebbe promuovere il consumatore fragment agli
stadi geometrici.

**Metodo**: un processo per coppia caso/variante, con e senza
`MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`; 20 ripetizioni in gara per
variante, produttore da ~9 ms; il consumatore copia ciò che legge e la CPU
confronta con il valore atteso. Due run completi, verdetti identici.

**Risultati**
- `barrierAfterQueueStages(…, Fragment)` su un render encoder è **legale ed
  efficace** (compute→fragment, render→render con depth campionata,
  blit→fragment, alias: 0/20 errate). L'ipotesi è smentita su questo
  dispositivo.
- Attendere in Fragment invece che in Vertex costa **~16% in meno** (200
  coppie render→render: 10,0 contro 11,9 ms).
- `before = Fragment` non protegge le letture nel vertex shader (20/20
  errate): gli stadi del consumatore vanno dichiarati esatti.
- **Tile** è accettato ovunque senza messaggi ma **non sincronizza nulla**
  (20/20 errate sia come produttore sia come consumatore).
- Dentro un render encoder `barrierAfterEncoderStages` accetta come produttore
  solo Vertex/Object/Mesh: Fragment o Tile → *abort* della validazione
  ("afterEncoderStages must be a valid combination … (MTLStageVertex |
  MTLStageObject | MTLStageMesh)"); senza validazione vengono ignorate.
  Dentro un compute encoder solo Dispatch/Blit/AccelerationStructure.
- Memoria aliasata in un heap placement: senza barriera 20/20 errate; con
  barriera di coda corretta 0/20 con `Device`, `ResourceAlias`, entrambe e
  perfino `None`: l'opzione di visibilità non è osservabile qui.
- Costi (GPU, mediane): barriera d'encoder ~1 µs, barriera di coda ~6 µs.

**Decisione** (`defaultBarrierRules()`, tabella in `barrier_plan.h`): stadi
del consumatore esatti, Fragment incluso; Tile promosso (before → stadi
geometrici, after → Fragment); barriere d'encoder illegali = errore di
compilazione del grafo (la fusione non le genera); `ResourceAlias` mantenuto
per contratto API. Da rivalutare su T0 (Apple9) quando disponibile (O12) e
per i flussi di dati dei tile shader, non misurati.

---

## F2.2 — Aliasing dei transitori verificato sulla GPU

**2026-09-29** · `--debug-graph-transients`: catena sintetica (compute →
compute → compute → raster → compute) su transitori R32Uint, controllo
esatto di ogni valore sulla CPU.

- Piano greedy: heap 528.384 byte contro 794.624 senza aliasing (−34%),
  4 risorse aliasate (A↔C, B↔D).
- PASS su 310 frame sotto validazione, anche con `--switch-every 20` e UI.
- Controlli negativi: senza barriere → FAIL (tutti i valori errati, **la
  validazione non segnala nulla**); C piazzata sopra D (sovrapposizione di
  intervalli vivi) → FAIL intermittente (gara reale); togliendo solo il flag
  di aliasing → ancora PASS (le barriere RAW/WAR ordinano già gli accessi;
  coerente con lo spike).

---

## F2.5 — Render pass sospeso/ripreso tra command buffer

**2026-09-29** · `--debug-split-encoding` (forward in 4 chunk su 4 thread,
5+1 command buffer per frame, un solo commit).

- Misurato: un altro encoder (la copia di `--capture` con la sua barriera di
  coda) nello **stesso** command buffer dopo il pass ripreso fa fallire
  l'intero commit (`MTL4CommandQueueErrorDomain` error 1, frame persi, poi
  timeout); la validazione non lo segnala prima del commit. Decisione: gli
  encoder successivi vanno in un command buffer nuovo.
- Immagini identiche (0 pixel) sui 7 bench, validazione a zero. Il guadagno
  di CPU si misura in F5 (oggi il forward costa ~0,1 ms di encoding).

---

## F2.6 — Async compute su una seconda coda

**2026-09-29** · `--debug-async-compute`: seed (grafica) → riduzione (coda
async) → consumo (grafica), con la riduzione sovrapposta al forward.

- Metal 4 sincronizza le code solo tra commit (`wait`/`signalEvent` di coda;
  i fence valgono in una sola coda): il frame diventa una lista di
  submission tagliate ai punti di sync; un evento timeline per coda (i
  valori restano monotòni).
- Bug trovato dalla validazione: la prima submission grafica partiva dal
  buffer 1 invece che dallo 0 e il command buffer async veniva committato due
  volte.
- PASS esatto su 300 frame con tutti i flag e `--switch-every 20`; controllo
  negativo senza le attese tra code → FAIL (~33.000 valori errati su 130
  frame).
- Guadagno da misurare in F5/F8 con carichi reali.

---

## F2.7 — Ordinamento delle istanze per classe di culling

**2026-09-29** · Stress Test (100K istanze), Release, run alternati con `main`.

- Prima versione: `stable_sort` con la classe di culling calcolata nel
  comparatore (lookup del materiale) → CPU 2,87 → 3,38 ms (**+17%**,
  regressione trovata dal confronto con `main`).
- Chiave a 64 bit (mesh | classe | indice originale) calcolata una volta,
  `std::sort` sulle chiavi e permutazione in vettori riusati: **1,89 ms**
  (−34% rispetto a `main`), ordine stabile (immagini identiche), nessuna
  allocazione a regime.

---

## F3.4 — `metal-tt`, `MTL4Archive` e runner CI (spike)

**2026-09-29** · M5 Max, macOS 27.2, Xcode 27.0 (toolchain Metal 27.1); runner
GitHub `macos-26` (Xcode 26.6, toolchain 17.6). Tool usa-e-getta fuori dal
repo (harvest con `MTL4PipelineDataSetSerializer`, lookup e compilazione
cronometrati) e un branch CI temporaneo, poi eliminato.

Ipotesi: descrittori raccolti a runtime → `.mtl4-json` → `metal-tt` in CI →
archivio caricato dal Mac con lookup quasi gratuiti; miss gestibili.

| Misura | Risultato |
|---|---|
| Harvest (`CaptureDescriptors`) → `serializeAsPipelinesScript` | JSON di 1,8 KB: librerie, function descriptor, pipeline descriptor |
| `metal-tt` locale | 15,8 s per tutte le arch (13 famiglie Apple + AMD/Intel vuote), ~0,1 s con `-arch applegpu_g17s` (M5 Max, da `xcrun metal-arch`) |
| Lookup nell'archivio | HIT 0,05–0,4 ms; compilazione a freddo della stessa pipeline 40–53 ms (seconda compilazione ~1 ms: cache shader dell'OS) |
| `metal-tt` sul runner `macos-26` | funziona (31 s, archivio 1,1 MB) |
| Archivio del runner aperto su macOS 27.2 | **rifiutato all'apertura**: "deployment target for architecture applegpu_g17s … is not compatible with current OS" |
| Archivio con la sola arch `applegpu_g16s` | rifiutato all'apertura: "unable to find applegpu_g17s slice" |
| Descrittore non raccolto (RGBA16F, `debug_reduce`) | miss per singola pipeline con errore esplicito ("Failed to find fragment function … with key") |
| Archivio vecchio + metallib modificato (costante del tonemap) | miss solo delle pipeline con funzioni cambiate (`forward_fs`), hit delle altre (`debug_fill`) |
| Etichette-hash del JSON sostituite con `L0…Ln` | `metal-tt` e lookup invariati: le etichette sono solo riferimenti interni, il JSON non dipende dal corpo degli shader |
| Alternativa senza `metal-tt`: serializer `CaptureBinaries` + `serializeAsArchiveAndFlushToURL` | funziona (118 KB, HIT); con `CaptureDescriptors \| CaptureBinaries` la serializzazione fallisce senza errore |
| `maximumConcurrentCompilationTaskCount` | 18 su M5 Max, 2 sul runner (GPU paravirtuale senza Metal 4) |
| Lookup con `MTL_SHADER_VALIDATION=1` | **ogni lookup fallisce**: "MTL4Archive instances are not compatible with Metal shader validation" |

Decisioni:

- L'archivio usato dal Mac è costruito dal build locale (`phosphor_archive`,
  `metal-tt` dello stesso OS, arch nativa); il CI costruisce l'archivio per
  tutte le arch come verifica di coerenza JSON/metallib e lo pubblica come
  artefatto per macOS 26. Sul Mac quell'archivio è il caso reale "OS
  diverso" di F3.5 (`tools/archive_check.sh`, scenario a).
- Il `.mtl4-json` è committato (`shaders/pipelines.mtl4-json`, path della
  libreria → segnaposto): resta valido finché non cambiano nomi, costanti o
  stato delle pipeline.
- Con la shader validation attiva l'archivio è dichiarato "non disponibile"
  (motivo nel log) invece di produrre un miss per pipeline; le verifiche
  dell'archivio girano con la sola API validation.
