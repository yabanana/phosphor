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

---

## F3.2/F3.3 — Pipeline flessibili e varianti specializzate: esattezza e validazione

**2026-09-29** · M5 Max, macOS 27.2, build Debug, `tools/visual_check.sh`
(7 bench, 3200×1800, API + shader validation).

Ipotesi: le varianti con function constant (O11) e le pipeline flessibili di
Metal 4 (stato di uscita `Unspecialized`, poi
`newRenderPipelineStateBySpecialization`) rendono gli stessi pixel della
pipeline generica, che a sua volta è identica al forward di F2.

| Pipeline che disegna il forward | Pixel diversi dai riferimenti F2 (su 5,76 M) | Messaggi |
|---|---|---|
| Generica a stato completo (`--debug-pipeline-fallback`) | **0** su tutti i bench | 0 |
| Variante specializzata (normale) | 9 / 3 / 14 / 13 / 30 / 10 / 9, delta max 1 | 0 |
| Flessibile specializzata (`--debug-flexible-pipelines`) | 15.848 – 301.286, delta max 1 | ~80 per bench |
| Variante con sale (`--pipeline-salt`) | 15.847 – 301.297, delta max 1 | 0 |

- Le varianti tolgono solo rami morti per la scena, ma il compilatore genera
  codice diverso (contrazioni FMA, scheduling): ≤1 LSB in qualche decina di
  pixel. Nel generico, un ramo o una select attorno all'emissivo cambiava
  1 LSB: l'agente F3.3 ha lasciato l'`add` incondizionato (fattore 0 esatto) e
  il ramo del sale in testa alla funzione. Riferimenti rigenerati con le
  varianti (`build/reference`); quelli F2 restano il riferimento del percorso
  generico.
- **Validazione e pipeline flessibili**: ogni specializzazione produce
  "blend state set to disabled, but blending substate set to Unspecialized.
  Blending substate is ignored." per ogni attachment con blend disabilitato,
  qualunque configurazione della base (6 configurazioni provate in un
  programma minimo: sottostati espliciti, tutti gli attachment
  `Unspecialized`, blend concreto nella base…). La compilazione a stato
  completo non avvisa mai.
- Una funzione con function constant non si può usare non specializzata
  ("Use newFunctionWithName:constantValues:"): il generico passa da un
  `SpecializedFunctionDescriptor` con valori vuoti.

Decisione: il fallback di una variante nuova è la pipeline **generica a stato
completo** (già pronta, bit-identica a F2, 0 compilazioni); la
specializzazione flessibile resta disponibile solo quando non esiste una
generica pronta per quello stato di uscita (utile da F8, con più formati di
uscita) e si ri-misura con `--debug-flexible-pipelines`. Il sale serve solo
alle misure di compilazione a freddo.

---

## F3.4 — `metal-tt` e debug info degli shader

**2026-09-29** · toolchain Metal 27.1. Con le function constant di F3.3,
`metal-tt` fallisce ("applegpu-nt: error: cannot find private metadata at
offset …") se il metallib è compilato con `-gline-tables-only` **oppure**
`-frecord-sources` (provati separatamente); senza entrambi l'archivio si
costruisce (1,7 MB per `applegpu_g17s`). Prima delle function constant lo
stesso metallib Debug funzionava. Decisione: `PHOSPHOR_SHADER_DEBUG_INFO`
(ON in Debug, per il debugging in Xcode) salta l'archivio con un messaggio;
Release e CI (shader ricompilati senza debug info) lo costruiscono. Con la
shader validation attiva l'archivio è comunque inutilizzabile (voce F3.4
precedente).

---

## F3.1 — QoS dei thread di compilazione: tempesta di 42 compilazioni

**2026-09-29** · M5 Max, Release, Stress Test (100K istanze, CPU ~2–3 ms
per frame), `--no-vsync --no-ui --debug-compile-storm`: al frame misurato 60
il forward richiede tutte le 42 varianti con un sale nuovo (compilazioni a
freddo, verificato: 540–610 ms di compilazione in totale, max 24–29 ms per
pipeline). 18 thread di compilazione (`maximumConcurrentCompilationTaskCount`);
3 run per QoS, alternati, sali espliciti.

Ipotesi (WWDC25-254): con i thread di compilazione a QoS più bassa del
render thread l'hitch sparisce; a QoS uguale il render thread soffre.

| QoS dei thread | CPU render thread, frame 60–69 (media / max) | Frame ms, frame 60–69 (media / max) | Swap completati entro il frame |
|---|---|---|---|
| utility (default) | 2,01 / 2,15 · 2,20 / 2,39 · 1,92 / 2,21 | 9,9 – 12,5 / 16,2 – 22,5 | 64–66 |
| user-interactive (controllo) | 2,34 / 2,77 · 2,34 / 2,53 · 2,28 / 2,60 | 11,7 – 12,5 / 17,2 – 17,7 | 64 |

- La QoS è quella attesa: il worker 0 legge `qos_class_self()` = 0x11
  (utility) e lo stampa nel log.
- Nessun hitch in nessuna delle due configurazioni: le 42 compilazioni
  girano nel servizio di compilazione su 18 thread e finiscono in ~5 frame.
  Con utility la CPU del render thread nei frame della tempesta è più bassa
  in 3 run su 3 (media 2,04 contro 2,32 ms, −12%), differenza piccola ma
  coerente.
- Su un M5 Max (18 core) la QoS pesa poco; la verifica sul pavimento T0 (O12,
  meno core) resta da fare quando ci sarà l'hardware.

Decisione: utility come default (come da playbook), `--compile-qos
interactive` resta come controllo negativo.

---

## F4 — Spike di osservabilità: timestamp MTL4, Tracy, contatori, cattura

**2026-09-29** · M5 Max, macOS 27.2, Xcode 27.0, Release · strumento:
`bench/timestamp_spike` (offscreen, kernel e fragment di costo noto: ciclo
LCG di `iters` iterazioni su 1M thread o su un target 2048² con blending
additivo, così nessun frammento viene scartato). Mediane di 7 run dopo un
riscaldamento di 20 command buffer; "feedback" = `GPUEndTime −
GPUStartTime` del commit.

### 1. Timestamp per pass con `MTL4::CounterHeap`

- **Dominio temporale**: 24 MHz (41,667 ns/tick), gli stessi tick di
  `mach_absolute_time` e di `CNTVCT_EL0`; `sampleTimestamps` restituisce
  invece nanosecondi (per CPU e GPU). La correlazione CPU/GPU è gratuita.
  `sizeOfCounterHeapEntry` = 8; `resolveCounterRange` (CPU) e
  `resolveCounterHeap` (GPU → buffer) danno gli stessi valori.
- **Il timestamp di inizio dentro un encoder è inaffidabile**: scritto in
  ritardo quando il lavoro è lungo (dispatch da 12,3 ms: inizio encoder 10,7
  ms dopo il timestamp di command buffer che lo precede; 8000 iterazioni:
  intervallo "encoder" 1,6–2,6 ms contro 6,1 ms reali). Il timestamp di
  **fine** encoder è corretto (≤ 0,07 ms prima del timestamp di command
  buffer successivo). I timestamp di command buffer
  (`writeTimestampIntoHeap`) coincidono col feedback (12,28 contro 12,28 ms).
- **Linearità** (controllo negativo) con timestamp di command buffer:
  0 / 500 / 1000 / 2000 / 4000 / 8000 / 16000 iterazioni → 0,013 / 0,36 /
  0,72 / 1,41 / 2,95 / 6,07 / 12,28 ms: lineare, rapporto con il feedback
  0,98–1,00.
- **Dentro un render encoder** (due draw di costo 100/400 e 400/100 =
  due "pass fusi"): tutti i timestamp Fragment cadono nello stesso istante,
  a fine pass (TBDR: i frammenti di tutti i pass del gruppo si alternano per
  tile); `Precise` non cambia nulla e non spezza l'encoder. Con Vertex il
  timestamp segna la fine della geometria (0,012 ms). **Il tempo per pass
  dentro un gruppo fuso non è misurabile con i timestamp.**
- **Limite di scritture**: in un render encoder, e in un compute encoder
  `Relaxed`, vengono scritti al massimo **4 timestamp** (tutti con lo stesso
  valore); gli altri restano a 0, senza alcun messaggio della validazione.
  In un compute encoder `Precise` ogni timestamp dopo un dispatch è
  distinto (64 su 64) e segue il costo del dispatch (1000/3000/6000 →
  0,70/2,17/4,60 ms, somma = feedback), purché l'inizio sia la fine del
  precedente e non un timestamp d'inizio encoder.
- **Sequenza di encoder** compute(2000) → render(100) → compute(4000) →
  render(300) con barriere: timestamp di command buffer tra gli encoder
  oppure un timestamp di fine per encoder (Relaxed o Precise) danno
  1,40/0,37/3,02/0,93 ms e la somma coincide col feedback (6,19 contro
  6,20); con i costi invertiti gli intervalli si invertono.
- **Costo**: 65 timestamp in un encoder da 64 draw: GPU 0,849 contro 0,850
  ms senza (non misurabile); CPU di codifica +0,15 µs circa per timestamp.
- **Suspend/resume (F2.5)**: tre chunk di un render pass su tre command
  buffer: tutti i timestamp dei chunk cadono a fine pass; si misura il
  gruppo intero.
- **Coda async (F2.6)**: stesso dominio temporale sulle due code (inizi a
  ±0,01 ms quando partono insieme); gli intervalli sono tempi di parete e
  includono la contesa (compute da 2,96 ms da solo → 3,4–5,3 ms in parallelo
  al render).

**Decisione**: per ogni command buffer un timestamp di command buffer
all'inizio, per ogni encoder un timestamp di fine (render: `afterStage
Fragment`, `Relaxed`); nei compute encoder con più pass un timestamp
`Precise` dopo ogni pass. Tempo di un encoder/pass = sua fine − fine
precedente (o inizio del command buffer). Un gruppo di render fuso ha un
solo tempo, attribuito al gruppo: i pass membri sono marcati "fused" e la
ripartizione viene da Metal System Trace (punto 3).

### 2. Tracy con command buffer MTL4

- Il backend Metal di Tracy 0.14.1 (`TracyMetal.hmm`) usa
  `MTLCounterSampleBuffer` e `sampleBufferAttachments` di Metal 3: gli
  encoder MTL4 non hanno né l'uno né l'altro (header SDK): **non
  utilizzabile**.
- Zone GPU manuali con l'API C (`___tracy_emit_gpu_new_context`,
  `…_zone_begin_serial`/`…_end_serial`, `…_gpu_time_serial`): periodo
  41,667 ns, `gpuTime` di riferimento = `mach_absolute_time()` (stesso
  contatore di `CNTVCT_EL0` usato da Tracy per la CPU, quindi nessuna
  calibrazione). Verifica headless con `tracy-capture` e `tracy-csvexport -g`
  compilati dai sorgenti (CMake, senza sudo): 200 frame × 4 zone, medie
  1,5600 / 0,4329 / 3,1352 / 1,0593 ms, **identiche** a quelle calcolate dai
  timestamp nello stesso processo.
- Insidia: se `tracy-capture` parte prima che il client ascolti sulla porta
  8086, `connect` "riesce" su macOS e la cattura fallisce con "disconnected
  during the initial connection handshake"; il client va avviato prima (o
  la cattura ritentata).

### 3. Contatori hardware (F4.4)

- API pubblica: `device->counterSets()` contiene solo `timestamp`
  (`GPUTimestamp`); campionamento Metal 3 solo ai confini di stadio; MTL4
  espone solo `CounterHeapTypeTimestamp`. Occupancy, banda, compressione,
  stalli: **non disponibili via API**.
- `xcrun xctrace record --template 'Metal System Trace' --instrument 'Metal
  GPU Counters' --launch -- ./phosphor …` funziona headless (8 s → 260 MB).
  Il set di contatori registrato contiene solo "RT Unit Active": il set si
  sceglie nella GUI; copie del template con `counterprofile` 0–4 e con
  l'ID acceleratore di questo Mac non cambiano nulla. **Contatori di
  limiter/banda non ottenibili da CLI.**
- Utilizzabili dal trace: `metal-gpu-intervals` (intervalli per encoder con
  l'etichetta dell'encoder, es. "Forward + ImGui overlay", per canale
  Vertex/Fragment/Compute) e `metal-shader-profiler-intervals` ("Shader
  Timeline": durata e percentuale del kick per shader, es. `forward_fs`
  98,8% e `imgui_fs` 0,2% dello stesso encoder fuso). Mappando le funzioni
  shader ai pass si ottiene la ripartizione dei gruppi fusi che i timestamp
  non danno. Anche `gpu-performance-state-intervals` (stato di clock).

**Decisione**: F4.4 parziale e dichiarato: script headless che registra
Metal System Trace, esporta le tabelle e produce per pass le durate
Vertex/Fragment/Compute e la quota per shader; nessun contatore hardware.

### 4. Cattura `.gputrace` da eseguibile CLI (F4.3)

- Senza variabili: `supportsDestination(GPUTraceDocument)` = 0,
  `startCapture` → "Capture layer is not inserted".
- Con `MTL_CAPTURE_ENABLED=1` (anche impostata con `setenv` nel processo
  **prima** di creare il device): cattura di un commit della coda MTL4 in un
  `.gputrace` da 37 MB (buffer e texture inclusi), 58–80 ms.
- Il layer di cattura inserito non cambia i tempi GPU del kernel noto (2000
  iterazioni 1,51 contro 1,51 ms; 4000: 2,88 contro 2,88). Il costo CPU nel
  motore va misurato.

**Decisione**: la cattura si abilita solo con le opzioni di cattura
(`setenv` prima del device), così senza opzioni il motore resta identico;
una cattura "sopra soglia" arma il frame successivo (non si cattura a
posteriori).

---

## F4.1–F4.3 — Scoperte dell'integrazione nel motore

**2026-09-29** · M5 Max, Release, `--frames 300`, report JSON v2.

**Semantica dei tempi per pass.** Il tempo di un'unità (gruppo di render o
pass compute) è il suo contributo **esclusivo** alla timeline della coda:
fine − fine precedente (o inizio del commit). Le unità di un frame sommano
allo span della coda: Torus con vsync 2,446 ms contro 2,465 del command
buffer, Stress Test 4,095 contro 4,177.

- **Sovrapposizione dentro il frame**: un pass compute senza dipendenze dal
  forward gira in parallelo; il forward risultava ~0 ms (finiva prima del
  compute). Due pass indipendenti si dividono il tempo: chi finisce dopo
  prende la sovrapposizione. Il pass di costo noto è quindi una dipendenza
  del forward (lettura dichiarata nel vertex stage).
- **Sovrapposizione tra frame** (senza vsync): il timestamp d'inizio commit
  del frame N+1 viene scritto mentre il GPU finisce il frame N (pass noto da
  1000 iterazioni misurato 2,06 ms contro 0,72). Correzione: l'inizio commit
  è limitato alla fine dell'ultima unità del frame precedente sulla stessa
  coda. Restano 3–8 frame su 300 con un'unità invalida (fine prima
  dell'inizio limitato) senza vsync.
- **DVFS**: con vsync il GPU abbassa i clock per riempire il frame: il pass
  noto costa ~5 ms con 1000, 2000 e 4000 iterazioni; con carico basso anche
  in modalità seriale il forward passa da ~0,3 ms (clock alti) a 1,47 ms. I
  timestamp misurano il tempo reale al clock corrente: i confronti per pass
  vanno fatti a clock saturi (senza vsync, `--gpu-timing-serial`, carico
  alto) e dichiarati.
- **Controllo negativo** (`--debug-gpu-cost N --no-vsync
  --gpu-timing-serial`, 3 run, p50): N = 4000 / 6000 / 8000 / 12000 / 16000 →
  pass noto 3,47 / 4,95 / 6,68 / 9,75 / 13,13 ms (~0,8 ms per 1000
  iterazioni, lineare entro ~5%, come il kernel isolato dello spike),
  forward 0,23–0,36 ms costante. Un run su 15 (N=12000) tutto lento di
  2,5–7× mentre altri processi usavano il GPU (agenti in parallelo).
  Meccanismo rotto apposta (ogni unità parte dall'inizio del commit): il
  forward diventa 9,47 / 7,52 / 14,68 ms e cresce con N → il controllo
  distingue.
- **`invalidateCounterRange` non è ordinata con il GPU**: invalidare il range
  di uno slot mentre gli altri slot sono in volo azzera anche scritture del
  frame successivo in quello slot (1 frame su 3 letto a 0; con i range
  spostati di 1 un frame diverso a 0). Nessuna invalidazione per frame: una
  voce non scritta conserva il valore di 3 frame prima ed è scartata perché
  più vecchia del primo inizio commit del frame.

**Tracy.** Con i contesti GPU creati sulla coda per-thread
(`___tracy_emit_gpu_new_context`) e le zone su quella seriale, `tracy-capture`
andava in segfault (accesso a 0x8 in `Worker::Exec`) quando le zone erano in
coda prima della connessione: contesti creati con le varianti `_serial`,
`tools/tracy_check.sh` passa e la media GPU di Tracy coincide col report
(4,9659 contro 4,9659 ms).

**Cattura `.gputrace` (F4.3).**
- SDL crea un device Metal con la `CAMetalLayer` della finestra:
  `MTL_CAPTURE_ENABLED` va impostata prima di `SDL_Init` (dopo non ha
  effetto).
- Un device Metal 4 non può essere l'oggetto della cattura ("Capturing Metal
  4 Device is not supported"); la coda MTL4 grafica sì (il lavoro della coda
  async non entra nel documento).
- Con un `MTL4Archive` caricato ogni cattura fallisce con lo stesso
  messaggio: con le opzioni di cattura l'archivio è disattivato (log).
- Stress Test, frame 100: documento da 559 MB (heap, buffer, texture,
  drawable). Soglia `--gpu-capture-over`: il frame lento è noto 3 frame dopo,
  si cattura il successivo.
