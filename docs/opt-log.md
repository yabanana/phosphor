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

---

## F4 — Scoperte della verifica finale

**2026-09-29** · M5 Max, Release, macchina scarica salvo dove indicato.

- **`writeTimestampIntoHeap` cresce senza limite nel driver.** Due istantanee
  `malloc_history -allBySize` a un minuto di distanza: ogni chiamata aggiunge
  un elemento a un `std::vector<ProgressMarker>` dentro l'oggetto command
  buffer AGX, mai svuotato quando il command buffer (uno per slot) viene
  riusato: +442 KB al minuto, heap CPU +1,72 MB a 30.000 frame con blocchi
  invariati. La stessa chiamata apre un contesto compute. Sostituita da un
  encoder compute con un dispatch di 1 thread (`timestamp_anchor`) e un
  timestamp: a 30.000 frame +523 KB (i buffer di misura), come senza timing
  più quei buffer.
- **Un encoder compute con il solo timestamp viene scartato** dal driver: il
  timestamp non viene mai scritto (tutte le unità invalide). Serve un
  dispatch.
- **L'anchor ha bisogno di una barriera produttore**: senza, il dispatch di 1
  thread veniva schedulato dietro al lavoro fragment e l'inizio commit cadeva
  dopo la fine dell'unità. Con `barrierAfterStages(Dispatch → tutti)`: bench
  1 e 3 200/200 frame validi anche senza vsync.
- **Frame che si sovrappongono (Many Lights, ~60 ms di GPU)**: in ~10% dei
  frame il render pass del frame N finisce prima di quello del frame N−1 (i
  due girano insieme): il tempo esclusivo non è definito e il frame è
  scartato (il report ne dà il conteggio, `frames`). Con
  `--gpu-timing-serial`: 200/200 e somma = command buffer (54,316 contro
  54,324 ms).
- **Controllo negativo a macchina scarica** (`--no-vsync --gpu-timing-serial`,
  p50, 2 run): pass noto 2,92 / 6,11 / 11,81 ms per 4k / 8k / 16k iterazioni
  (~0,74 ms per 1000, lineare), forward 0,216–0,218 ms, run concordi entro
  il 2%.
- **Shader Timeline di xctrace nei gruppi fusi non dà il costo per pass**:
  ImGui risulta il 25% del tempo shader dell'encoder fuso su Many Lights, ma
  0,144 ms su 62 (0,2%) misurato da solo (`--gpu-timing-unfused`).
- **Race in `PipelineCache::waitAllFinal` (F3)**: un job pubblica la sua
  completion prima che il worker smetta di contarlo come in corso; se il
  controllo `outstanding()` cadeva in mezzo, l'attesa della completion
  successiva era infinita (un run di cattura bloccato 25 minuti, tutti i
  thread di compilazione inattivi). Controllo negativo: una pausa di 50 ms
  in quella finestra blocca il codice vecchio 3 volte su 3 catturando il
  frame 1 (variante in volo), mai quello nuovo, che attende lo stato idle
  della coda.
- **`hot_reload_check`**: `cp` + `perl -pi` erano due scritture; se il poll
  del watcher (250 ms) cadeva in mezzo le ricariche erano 2 invece di 1.
  Ora una rinomina.
- **`hitch_check`**: con `--warmup 0` il cambio al frame 20 cadeva nella
  comparsa della finestra, dove il pompaggio eventi SDL/Cocoa costa fino a
  ~15 ms: 1 hitch per run su `main` e su F4 (3/3 ciascuno); Tracy mette il
  tempo nella zona `Events`. Con 60 frame di warm-up: 0 hitch, 3/3 su
  entrambe.
- **Tracy senza client** (modalità normale): la coda degli eventi cresce
  (~22 MB in 50 s di footprint, fuori dalle statistiche malloc perché Tracy
  usa il proprio allocatore). Con `TRACY_ON_DEMAND`: +2 MB come senza Tracy;
  i contesti GPU sono rimandati da Tracy alla connessione.
- **Leak del layer di cattura**: con `--gpu-capture` `leaks --atExit` trova
  10 leak (784 B) anche senza catture (ciclo di retain in
  `GTMTLCaptureServiceXPCDispatcher`) e 2 per ogni cattura (`CFString` dei
  nomi dei file del documento): tutti nel framework di Apple; senza layer 0
  leak.
- **xctrace `--launch`** (riportato dall'agente F4.4, non riprodotto dal
  coordinatore, le cui tracce dello spike partivano da `~/Documents`): un
  processo lanciato da `~/Documents` restava bloccato in `open()` di dyld;
  `tools/gpu_trace.sh` copia l'app in `$TMPDIR`. `--target-stdout` elimina
  le righe dello Shader Timeline.

---

## OPT-0 — Spike di caratterizzazione: metodo, energia, API, intensità aritmetica, SLC

**2026-09-30** · Apple M5 Max (Apple10, 40 core GPU, CPU 6+12), macOS 27.2
(26B5091g), Xcode 27.0 (metal 32023.921), **a batteria** (72→61%, nessun
avviso termico da `pmset -g therm`) · strumenti in
[`bench/opt0_spike/`](../bench/opt0_spike/) (compilati a mano, comandi nel
[README](../bench/opt0_spike/README.md)). Sul Mac giravano altri client GPU
(WindowServer ~45% CPU, `replayd`, Chrome, app Codex): condizione reale della
macchina, non eliminabile senza chiudere le app del proprietario.

### 1. Metodo di misura (`method_spike`)

Kernel noto: 4 catene FMA FP32 indipendenti per thread su 1M thread, risultato
scritto e confrontato con la CPU (6 campioni, 0 errori). Tempo = timestamp
`Precise` dopo il dispatch − timestamp dopo un dispatch "anchor" di 1 thread
(stesso encoder, barriera Dispatch→Dispatch).

- **Timestamp vs feedback**: il feedback del commit (`GPUEndTime −
  GPUStartTime`) supera i timestamp di un offset costante di **1,5 µs** (1×…
  8×): i due metodi coincidono; la suite usa i timestamp (più misure per
  commit, nessun costo di commit incluso).
- **Controllo negativo lineare** (30 ripetizioni per punto, P13 al 99–100%):
  1× / 2× / 4× / 8× lavoro → 1,0005 / 2,0228 / 4,1266 / 8,3441 ms, rapporti
  1 / 2,02 / 4,13 / 8,34: lineare entro il 4%. Lo scarto cresce con la durata
  del dispatch (vedi CV).
- **Commit quasi vuoto** (200 commit, clock caldi): solo anchor → GPU
  **1,6 µs** (p10 1,5, p90 1,9); dispatch minimo (256 thread) → feedback
  3,4 µs, timestamp 1,9 µs; latenza CPU commit → evento visto dalla CPU
  **156 µs** p50 (p10 125, p90 289): base di B-28.
- **Guardia anti-eliminazione**: stesso ciclo con risultato non scritto →
  **0,0057 ms** contro 1,97 ms (il compilatore cancella il ciclo); risultato
  mascherato con uno zero letto a runtime (`& p.zero`) → 1,76–1,86 ms (tenuto,
  ma 6–10% più veloce della versione che scrive il float: il codegen cambia;
  la suite scrive e verifica i risultati, la maschera solo dove la verifica è
  impossibile).
- **DVFS**: con carico continuo il GPU va allo stato più alto (P13) entro
  0,25 s dalla ripartenza dopo 3 s di idle; la tabella `voltage-states9` di
  `pmgr` (ioreg, senza root) dà 13 P-state da 338 a **1620 MHz** (P13), ~50 W
  di GPU (IOReport) sul kernel FMA. Con pause tra dispatch da ~0,5 ms:
  pausa 0 / 2 / 8 / 16 / 33 ms → 0,485 / 0,486 / 0,616 / **2,32** / 2,33 ms
  (a 16 ms il GPU scende a P3; tempo ×4,8). La suite tiene il carico continuo
  tra i gruppi di misura e registra la residenza dei P-state per ogni
  benchmark.
- **Verifica con xctrace** (Metal System Trace sul caso "pause"):
  `gpu-performance-state-intervals` dà solo tre livelli (Minimum / Medium /
  Maximum): Maximum nei primi 2 s (carico continuo e pause di 2 ms),
  Minimum con le pause di 16–33 ms, coerente con IOReport. IOReport è più
  fine (P-state esatto) ed è in-process: la suite usa IOReport, xctrace resta
  il controllo esterno.
- **Varianza**: a clock massimo il CV del singolo dispatch è 3–20%, con code
  lunghe (min/mediana 0,996 per dispatch da 0,05 ms, 0,906 da 5 ms: i
  dispatch lunghi vengono interrotti più spesso dagli altri client GPU, il
  compositor a 120 Hz). **Il CV delle mediane** di 8 gruppi, con carico
  continuo tra i gruppi: dispatch 0,1 ms × 5 ripetizioni → 0,04%; 0,5 ms × 15
  → 0,05%; 1,7 ms × 5 → 2,7%, × 45 → 0,5%. Il CV dei minimi è ≤ 0,08% in
  tutti i casi.

**Decisione (protocollo della suite)**: timestamp Precise con anchor;
dispatch da 0,1–0,5 ms (mai oltre ~2 ms); riscaldamento continuo fino a P-state
massimo ≥ 95% del tempo attivo (IOReport); gruppi di ≥ 15 ripetizioni,
statistica = mediana (min e p10/p90 registrati); 3 run → CV tra run per
metrica; ogni kernel scrive un risultato verificato sulla CPU; dati di
input casuali (vedi punto 5); ogni benchmark ha un controllo negativo.

### 2. Energia e clock senza sudo (`ioreport_probe`)

- `libIOReport` (nell'SDK come `libIOReport.tbd`, API privata) si usa senza
  root: 6121 canali. **GPU Energy** (gruppo "Energy Model", nJ) si aggiorna
  in tempo reale (~2 volte al secondo): 1,0–1,6 W a riposo, 28–29 W con il
  kernel LCG dello spike F4, ~50–53 W con il kernel FMA; finestre < 0,5 s
  danno 0 o il doppio: misurare su ≥ 1 s.
- **GPU Performance States** ("GPU Stats"): residenza per P-state
  (OFF, P1…P15; la tabella ioreg ne definisce 13), in tempo reale.
- I canali CPU/DRAM/ANE dell'Energy Model (`CPU Energy`, `DRAM0`, `ANE0`,
  mJ) **esistono ma sono aggiornati raramente** (fermi per decine di secondi;
  +5,4 kJ di `CPU Energy` in ~25 min). Polling ogni secondo per 10 minuti:
  aggiornamenti a 85 s e 405 s (**~5 minuti** tra due aggiornamenti, +209 J
  = 0,65 W medi con la macchina quasi a riposo): utilizzabili solo come
  media su finestre di molti minuti (soak), non per benchmark da secondi.
- PMP "DCS BW" (istogrammi di banda DRAM per agente, 32 GB/s per bucket):
  `AMCC RD` segue il traffico ma **sottostima** (media 346 GB/s mentre il GPU
  legge a 540 GB/s da DRAM); `AGX RD` resta fermo: solo indicazione
  qualitativa, non usato come misura.
- Termica: `NSProcessInfo.thermalState` (metal-cpp `NS::ProcessInfo`),
  `pmset -g therm` (nessun avviso registrato durante gli spike).

**Decisione**: B-27 misura watt GPU (IOReport), P-state e frequenza (IOReport
+ tabella ioreg), stato termico, e la potenza CPU/DRAM/ANE solo come media
nel soak (finestre ≥ 10 min, periodo ~5 min); senza
alimentazione di rete e senza soak su più Mac il resto di B-27 è parziale e
dichiarato.

### 3. API sul chip

- **Neural Accelerator** (`api_probe gemm`, tensor ops di
  MetalPerformancePrimitives con tensori `tensor_inline` costruiti da
  puntatori device, `matmul2d` 64×32, 4 SIMD-group; 4096³, input interi
  piccoli → risultato esatto, 64 campioni verificati, 0 errori): FP16→FP32
  **32,2 TFLOPS**, BF16→FP32 32,2, INT8→INT32 **53,8 TOPS**; GEMM FP16 con
  `simdgroup_matrix` sugli ALU 14,3 TFLOPS. I tipi accettati dal header
  (70 combinazioni) includono FP8 e4m3/e5m2, FP4 e2m1, INT4/INT2; `char`
  come INT8 non compila (serve `int8_t`/`int32_t`). Fonte esterna ([R7]/
  Creative Strategies nel playbook: 19,9 TFLOPS) sotto la nostra misura.
- **Core ML / Neural Engine** senza modelli esterni (`coreml_probe`): modello
  NeuralNetwork scritto in codice (protobuf codificato a mano, 4 × conv 3×3
  256→256 + ReLU su 256×64×64, 19,3 GFLOP), compilato in 33 ms;
  `MLComputePlan` conferma il dispositivo per layer (ANE con
  `cpuAndNeuralEngine`/`all`). Predizione p50: CPU 12,4 ms (1,56 TFLOPS),
  CPU+GPU 1,38 ms (14,0), **ANE 1,26 ms (15,4 TFLOPS efficaci)**.
- **MTLIO + compressione** (`api_probe mtlio`, 256 MiB di rampe a 16 bit con
  rumore, `MTLIOCreateCompressionContext`, chunk 64 KiB, risultato
  confrontato byte per byte): rapporto / lettura fredda (copia scritta con
  `F_NOCACHE`) / calda: lz4 1,45 / 1,62 / 2,33 GB/s; lzfse 2,59 / 1,08 / 1,32;
  zlib 2,89 / 0,50 / 0,53; lzma 4,53 / 0,10 / 0,10; lzbitmap 2,45 / 2,46 /
  2,83 GB/s. Con dati incomprimibili (rapporto 1,00, chunk salvati crudi)
  ~15 GB/s dalla page cache: il limite è la decompressione.
- **CPU** (`cpu_probe`): `sysctl` riporta FEAT_SME/SME2/SME2p1, vettore
  streaming 64 B; il compilatore accetta codice SME2 con
  `-mcpu=native+sme2` (`__arm_locally_streaming`, `svcntsb()` = 64).
  SGEMM Accelerate 1,69 TFLOPS FP32 (4096²), NEON FMA 76,6 GFLOPS su un core.
- **`CAMetalDisplayLink`** (`display_probe`, Objective-C++: metal-cpp non lo
  espone): finestra 1280×800 a 120 Hz: intervallo di callback 8,332 ms
  (sd 0,46), presentazione = target ±3 µs, intervallo di presentazione
  8,333 ms, 1 frame non presentato su 351 (l'ultimo); a 60 Hz 16,667 ms.
  Anticipo callback → presentazione **41,6 ms** (5 refresh) sia con
  `preferredFrameLatency` 1 sia 2 (finestra composta): da approfondire in
  B-26. La latenza input→fotoni richiede un sensore: non misurabile qui.

### 4. Intensità aritmetica dei pass (IR AIR)

`xcrun metal -S -emit-llvm` sugli shader del motore + conteggio per blocco
di base (`air_ops.py`, `bench/opt0_spike/air_ops.py`): `forward_fs` 38 blocchi,
427 flop statici (fadd/fmul/fma/dot/mix per lane), 15 trascendentali, 10
divisioni, 5 campionamenti; il ciclo sulle luci (back edge rilevato) costa
**166 flop + 9 trascendentali + 7 divisioni per luce**; `forward_vs` 82 flop.
Con il numero di invocazioni (frammenti ≥ pixel coperti per l'HSR, vertici,
thread) e i byte DRAM del grafo si colloca ogni pass nel roofline.
Approssimazione da dichiarare: l'AIR non è l'ISA AGX (il backend fonde,
espande divisioni e trascendentali); i byte del grafo contano solo gli
attachment.

### 5. SLC dalla curva latenza/banda (`method_spike chase|slc`)

- **Latenza** (1 thread, ciclo casuale di linee da 128 B, mediana di 5):
  ~31 ns fino a 128 KiB, 97–150 ns tra 256 KiB e 1 MiB, 250–320 ns a
  2–4 MiB, **~330–345 ns piatti da 4 a 64 MiB**, poi 356 / 381 / 455–490 ns
  a 128 / 256 / 512–2048 MiB (DRAM + TLB con pagine da 16 KiB).
- **Banda di lettura** (327.680 thread, float4 coalescenti, ~4 GiB letti per
  misura, dati casuali): ~9 TB/s fino a 24–40 MiB, discesa tra 48 e 72 MiB
  (2,1–6,6), ~2 TB/s a 96–128 MiB, **~975 GB/s piatti tra 192 e 384 MiB**,
  757–800 a 512 MiB, **540 GB/s a 1–2 GiB** (DRAM; picco dichiarato M5 Max
  460/614 GB/s). Forma riproducibile su 3 run, livelli del plateau variabili
  fino al 25% tra run (7,1 contro 9,4 TB/s a 24 MiB).
- **Trappola dei dati**: con un buffer privato non inizializzato (zeri) la
  banda on-chip sale del ~25% (9,1–9,5 TB/s): input casuali obbligatori.
- La guardia "base della passata dipende dalla precedente" non cambia i
  numeri: nessuno scambio di cicli da parte del compilatore.
- Interpretazione: la banda sopra la DRAM fino a 384 MiB non si spiega con un
  gradino netto; è compatibile con una cache di ultimo livello resistente al
  thrashing (frazione di hit ≈ C/WS): con quel modello C ≈ 100–130 MiB. È
  **un'ipotesi da confermare** con B-08 (curva fine × 3 run, fit), non un
  valore.

**Decisioni per il piano**: B-08 riporta curve complete (latenza e banda),
stime dei ginocchi e fit dichiarati come stime; dati casuali; IOReport PMP
solo qualitativo.

---

## OPT-0 — Suite `bench/soc`: scoperte della misura e dei controlli negativi

**2026-09-30** · M5 Max, macOS 27.2, a batteria, schermo tenuto acceso
(`caffeinate -d`) · `tools/soc_bench_all.sh`: 28 benchmark × 3 run, tutti
i controlli negativi superati, `--validate` 0 messaggi e 0 benchmark falliti,
run con i percorsi Apple9 (`--force-family apple9`) verde, `leaks` 0.
Risultati: [`bench/results/m5max-macos27.2.json`](../bench/results/m5max-macos27.2.json),
modello e confronto con fonti esterne: [`docs/soc-model.md`](soc-model.md).
Ogni controllo negativo è stato rotto apposta almeno una volta (copia degli
shader via `SOC_SHADER_DIR` o modifica temporanea, `git diff` pulito dopo) e
ha fallito.

### Bug di metodo trovati dai controlli (e corretti)

- **Catene uniformi o identiche** (B-01): con un seme intero uguale per tutti
  i thread (`seed * int(0.0625)`) il calcolo era eseguito una volta per
  SIMD-group e le 8 catene fuse: 8.054–13.957 op/core/clk (impossibile con 128
  ALU). Semi per thread e per catena + limite di plausibilità 512 op/core/clk.
  Stessa trappola per le catene ADD intere: `w - (v + w)` si semplifica anche
  senza fast-math → coppie `v += w; w ^= v`, float in `MathModeSafe`.
- **Costo bimodale dei render pass** (B-14, B-19): un piccolo render pass tra
  due encoder compute costa **~6–15 µs oppure ~60 µs**; le mediane di varianti
  diverse cadevano in modalità diverse (RGBA8 1080p: "store" 0,015 ms <
  "dontCare" 0,055 ms; B-19: dipendenza forzata = 0,87 × (A+B)). Con i minimi
  per giro e ordine ruotato: dipendenza forzata = **1,000 × (A+B)** in 4 run
  su 4. Anche i dispatch vuoti sono bimodali (0,12 o 2,6 µs, B-17).
- **Presentazione senza dipendenza dal lavoro** (B-26): il carico GPU e il
  command buffer del drawable non avevano dipendenza, quindi il piccolo pass di
  presentazione girava accanto al carico e 9 ms di lavoro a 120 Hz davano 0%
  di frame in ritardo. Con un `MTLEvent` carico → presentazione: 9 ms → 80–97%
  in ritardo a 106–111 Hz (= 1000/9). La cadenza si calcola con la media degli
  intervalli (la mediana nasconde i frame raddoppiati).
- **Controllo vuoto** (B-23): senza una configurazione `cpuAndNeuralEngine`
  il controllo "MLComputePlan mette tutto su ANE" passava senza controllare
  nulla; ora fallisce.
- **Schermo spento durante la batteria**: B-26 senza frame presentati, B-27
  con GPU idle a 0,02–0,06 W. La batteria gira sotto `caffeinate -d`; B-26
  rileva `CGDisplayIsAsleep` e l'occlusione della finestra e si dichiara non
  misurabile.
- **Latenza DRAM e fabric** (B-08): a working set ≥ 64 MiB la latenza variava
  475–1224 ns tra run (4N/N 3,9–8,8×), anche con il GPU al P-state massimo e
  lo schermo acceso. IOReport "GPU Stats / AFR Performance States": durante il
  chase a thread singolo il fabric GPU scende da un P-state medio ~12 a
  **~4–5 su 13** (il governor segue la banda richiesta, minima). Catene
  lunghe (30 ms) peggiorano (770–880 ns: il tempo per passo cresce con la
  durata). Scelta: pendenza su span da 3 ms, AFR registrato per punto,
  controllo di linearità sul set L1 (stabile, CV 0,9%; un ciclo limitato dà
  1,2×), metrica `latency_dram.page_local` (un miss di pagina ogni 128 passi).
  Risultato: latency_dram 516/523/919 ns nei 3 run a batteria, 593/599/610 nei 3
  run finali all'alimentazione (page-local 471–527); la causa è dichiarata.
- **Validazione**: sotto `MTL_DEBUG_LAYER`/`MTL_SHADER_VALIDATION` i tempi
  cambiano in modo disuniforme (latenza L1 ×4); `--validate` ora conta i
  messaggi e i fallimenti di correttezza, i controlli di tempo sono elencati
  ma non applicati. Controllo negativo: una riga estranea nell'output del
  figlio → exit 3.
- **`--quick`**: troppe poche ripetizioni per i controlli di tempo (B-08,
  B-12, B-14 fallivano a volte): il run Apple9 della batteria usa la qualità
  piena.

### OPT-0.6: soglie critiche misurate

- **Partial render (B-12)**: la prima versione (fino a 16,7 M triangoli, un
  target RGBA8) non vedeva nessun ginocchio, ma 16 varyings float4 per 16,7 M
  triangoli non possono stare in nessun parameter buffer: i partial render
  c'erano, invisibili perché il flush di 16 MiB costa poco. Con una seconda
  curva a flush costoso (16 varyings, 4 attachment RGBA32F = 256 MiB salvati
  per flush) e il campionamento esteso a 2^26 (ginocchio = primo N oltre 1,3 ×
  il minimo e che resta sopra): **16 varyings → 23,7 M triangoli** per pass
  (3/3 run, entrambe le curve; salto 4,6 → 7,7 ns/triangolo con le attachment
  pesanti, 4,1 → 5,8 con RGBA8), **8 → 47 M** (3/3), **4 → 67 M** (2/3, 33,5 M
  in un run), **0 varyings → nessuno netto fino a 67 M** (rialzo marginale
  ~1,3× a 47–67 M). La soglia scala con la dimensione dei vertici in uscita e
  il costo cresce con i byte delle attachment: la firma di un parameter
  buffer che va in overflow.
- **Thrashing dei registri (B-04)**: throughput stabile fino a 120 valori FP32
  vivi (6,9–7,2 Top/s), crollo a **128** (1,88; 0,46 a 256);
  stessa soglia senza carichi (spill del compilatore). Un array indicizzato
  dinamicamente (stack) costa 62× la versione a registri.
- **Imageblock massimo (B-15)**: tile 32×32 → 24 B/pixel espliciti (32 B
  riportati, 32 KiB di tile memory = `maxThreadgroupMemoryLength`); 32×16 e
  16×16 → 56 B/pixel (64 B per campione: il limite per campione); tile 16×8,
  8×8 rifiutate dalla validazione del descrittore anche senza layer. **Oltre
  il limite la pipeline si crea e il pass termina senza errori, ma il tile
  kernel non gira** (sentinella intatta): va controllato a priori.

### Risultati salienti (dettaglio e CV in `soc-model.md`)

- ALU: FP32 FMA 15,15 TFLOPS (116,9 FMA/core/clk a 1620 MHz); FP16 raggiunge
  **1,85× FP32 solo con 32 catene indipendenti** (1,15× con 8: serve più ILP);
  INT32 mul 4,06 Top/s (~62% dell'add+xor). Trascendenti fast 8,3 Top/s.
- Memoria: DRAM lettura 572 GB/s (93% dei 614 dichiarati), scrittura 486,
  copia 535; on-chip ~8,6 TB/s fino a ~32 MiB; SLC stimata dal fit 71 MiB
  (modello, residuo 12%). CPU e GPU condividono lo stesso budget: 6 thread
  memcpy portano il GPU da 573 a 335 GB/s, totale ~553 GB/s (B-09).
- Threadgroup memory 6,2 TB/s a stride 1, 1,2 TB/s a stride 32 (conflitti di
  banco), 3,5 a stride 33; latenza 33 ns (B-05). `simd_shuffle` con lane
  dinamica risulta 0,37× l'emulazione in threadgroup memory (quad_shuffle
  1,9×, sum 5,7× a favore dell'intrinseco): da riverificare prima di usarlo.
- Atomici device su un indirizzo 1,6 Gop/s contro 204 Gop/s su indirizzi per
  thread (137×); threadgroup 104 Gop/s su un indirizzo. 64 bit: solo
  `atomic_max/min` (senza valore restituito) su Apple9+.
- Geometria: ~7,9–9,5 Gtri/s sia vertex sia mesh shader (limite raster),
  front-end senza raster ~22 Gtri/s; culling nell'object shader 1,8–1,9×.
  Dispatch vuoto 0,13 µs (0,82 con barriera), draw via ICB 0,043 µs, barriera
  d'encoder 0,70 µs; barriere di coda ~0 tra compute, ~10,5 µs tra render
  pass (B-18).
- Sovrapposizione: due pass render indipendenti 0,84 della somma, compute su
  seconda coda 0,98 (ALU) / 0,78 (banda) (B-19).
- RT: 8,8 Grays/s coerenti, 5,4 incoerenti (`intersector`); `intersection_query`
  ~1,9× più lento; BLAS 1M triangoli in 4,7 ms, refit 7,6× più veloce,
  compaction 0,48.
- Neural Accelerator: GEMM FP16/BF16 58 TFLOPS, INT8 114 TOPS, INT4 92, FP8
  62 (MSL 4.1); `simdgroup_matrix` 15; MLP fuso 1,53× la versione simdgroup.
- ANE (Core ML, modello in codice): 15,7 TFLOPS efficaci, 1,24 ms per 19,3
  GFLOP; GPU via Core ML 13,9, CPU 1,5; con il GPU carico la latenza ANE
  sale di 2,0–2,9× (run diversi), il GPU non rallenta.
- CPU: NEON 106 GFLOPS per core, SGEMM Accelerate 2,6 TFLOPS, SME2 diretto
  compilabile; risveglio di un thread user-interactive 1,1 µs.
- MTLIO: SSD freddo 3,0–3,4 GB/s (non compresso), cache 62 GB/s; decompressione
  lz4 2,5, lzbitmap 2,9, lzfse 1,4, zlib 0,53, lzma 0,10 GB/s.
- Display: jitter di presentazione ~0,04 µs a 120 Hz; anticipo callback →
  presentazione **41,6 ms in finestra e a schermo intero, 16,3 ms in una
  finestra borderless** grande quanto lo schermo.
- Energia (B-27): GPU ~63 W sul carico FMA (8 pJ per FMA), idle 0,03 W;
  carico CPU su tutti i core non rallenta il GPU (1,00).
- Commit: 3,4 µs di GPU, 168 µs CPU commit → evento (spin 141, listener 200).
- Roofline (OPT-0.4, `tools/soc_roofline.sh`): limite inferiore ≤ misurato
  per i 7 pass (7/7); il limite è dominato dalla scrittura del drawable in
  DRAM (0,040 ms) perché il lavoro per luce di `forward_fs` sta dietro un
  `continue` dipendente dai dati (cammino minimo 0) e il modello non include
  costi fissi né raster. I tempi misurati dei pass sotto ~1 ms non saturano i
  clock nemmeno con `--no-vsync --gpu-timing-serial` (Torus 1,02 → 0,34 ms,
  Scene Viewer 0,20 → 0,81 tra due sessioni); Many Lights (52,4–52,6 ms) è
  stabile. Controllo di coerenza: 52,5 ms × 15,1 TFLOPS / (4,88 M pixel ×
  1024 luci) ≤ 158 flop per luce realmente eseguiti.
- **Soak B-27 (10 min, a batteria 32 → 12%)**: prestazioni GPU invariate
  (0,984 ms per dispatch, 1620 MHz), 68,5 → 67,0 W, termica "fair" dopo
  2,5 min. Nelle fasi dello stesso run (eseguite prima del soak, batteria
  al ~32%) il controllo di B-27 è fallito: GPU a 649 MHz sotto carico misto (2,03× più
  lento; 1,00× nei run con batteria al 58–85%) e "GPU Energy" con delta 0
  su una finestra di 4 s. Causa: limitazione di potenza a batteria scarica
  (il run è stato scartato; le misure vanno fatte a batteria carica o
  all'alimentazione, sempre registrata in `machine.power_source` /
  `battery_percent`).

---

## OPT-1 — Spike: scenari di grafo, solver, compressione, rematerializzazione, code, upload

**2026-09-30** · Apple M5 Max (Apple10, 40 core GPU), macOS 27.2 (26B5091g),
all'alimentazione, schermo acceso (`caffeinate -d`), build Release,
`--no-vsync`; GPU al P-state massimo (P13) per l'83% del tempo del run e
spento nel resto (IOReport `GPUPH`, 39 W medi sullo scenario 0).

Il grafo del frame del motore ha 1–2 pass (Forward + ImGui): OPT-1 non si
può misurare lì. Gli spike usano **scenari di grafo realistici** eseguiti
dal vero `MetalGraphExecutor` (`--graph-scenario N`,
`src/rendergraph/scenario.h`, `shaders/scenario.metal`): pass sintetici
con lavoro noto (catene LCG a 4 vie, un IMAD per passo e catena) e byte
noti, valori interi deterministici (multipli di 1/255, profondità
quantizzate) così che **qualunque ordine, fusione, aliasing, coda o
rematerializzazione corretti diano la stessa immagine bit per bit**.
Scenari: 0 deferred (4 cascate d'ombra, G-buffer, SSAO, lighting con
framebuffer fetch, bloom 5+4, TAA, tonemap, UI), 1 forward+ (3 cascate,
depth prepass, light grid, forward opaco con solo depth test, cielo,
trasparenti, bloom, TAA), 2 catena di post (DoF, motion blur, bloom 6+5,
lens flare, esposizione, tonemap, grana, FXAA, UI), 3 compute asincrono
(4 cascate, G-buffer + lighting fondibili, aggiornamento GI limitato dalla
banda e simulazione di particelle limitata dall'ALU sulla seconda coda).
Risoluzione interna 2560×1440, drawable 3200×1800.

Correttezza del banco di prova (Debug, `MTL_DEBUG_LAYER` +
`MTL_SHADER_VALIDATION`): 0 messaggi sui 4 scenari; immagini identiche
(0 pixel) tra due run, tra fuso e non fuso (`--gpu-timing-unfused`), con
rematerializzazione e con ordini diversi. Controllo negativo: seme del
ricalcolo sbagliato di 1 → 5.759.907 pixel su 5.760.000 diversi.

### 1. Byte DRAM stimati contro tempo misurato (spike 1)

Metodo: 4 scenari × 3 run × 600 frame (+120 di riscaldamento), tempo per
unità temporizzata (F4.1: gruppo di render fuso o pass compute) = minimo
per unità, mediana dei 3 run; byte = `estimatePassBandwidth` (O1: una
copia per lettura/scrittura, attachment secondo load/store del gruppo);
limite = byte / 569 GB/s (B-08). Controllo: `--graph-scenario-wide` porta
gli intermedi RGBA16Float a RGBA32Float (stesso lavoro, byte ×2 dove li
usano).

| Scenario | Unità | Frame (p50, CV 3 run) | DRAM stimata | Limite byte/banda | Wide: byte / frame |
|---|---:|---:|---:|---:|---:|
| 0 deferred | 20 | 2,663 ms (0,06%) | 565,9 MiB | 1,043 ms | ×1,38 / ×1,059 |
| 1 forward+ | 14 | 3,965 ms (0,03%) | 458,3 MiB | 0,845 ms | ×1,47 / ×1,038 |
| 2 post | 25 | 2,989 ms (0,02%) | 524,3 MiB | 0,966 ms | ×1,62 / ×1,068 |
| 3 async | 10 | 3,430 ms (0,02%) | 494,8 MiB | 0,912 ms | ×1,40 / ×1,176 |

- **Tempo ≥ byte/banda in 69 unità su 69** (nessuna stima dei byte
  "impossibile"). Unità vicine al limite (≤1,3×: Bloom down 1–2, DoF
  downsample, UI): con 2× byte → **1,64–1,99× tempo** (Bloom down 1:
  1,84–1,99); unità limitate da ALU o latenza (Lighting 2,75× il limite,
  TAA 1,6×, G-buffer 2,6×): 0,93–1,08×. Il pass "GI probe update" (3,05×
  il limite) sale di 1,73×: limitato in parte dalla banda.
- Il frame è poco sensibile ai byte su M5 Max: +38–62% di byte → +3,8–17,6%
  di tempo. La riduzione dei byte vale tempo solo nei pass limitati dalla
  banda; negli altri vale energia (non misurabile qui, B-27) e banda lasciata
  alla CPU (B-09).
- **Costo fisso di un compute pass dipendente minuscolo ~7 µs** (Bloom
  down 4–6, up 3–5, MB neighbor max: 0,007–0,008 ms per 0,03–0,5 MiB),
  molto sopra la barriera di B-18 (≤1 µs) e il dispatch vuoto (0,13 µs):
  svuotamento e riempimento del GPU a ogni dipendenza. Peso per il modello.
- Un'iterazione LCG (4 catene) costa 4 IMAD: "Particle simulation" (512² ×
  4096 passi) 1,110 ms contro 1,06 ms predetti a 4,06 Top/s (B-01 mul
  intera): il modello conta IMAD al ritmo della mul.
- Pass sintetici "una thread per tile" (Light grid, MB tile max, 16×16
  letture per thread) sono limitati dalla latenza (0,39–0,41 ms per 14 MiB):
  non rappresentano un'implementazione reale ma non dipendono dall'ordine.

### 2. Solver dell'ordine: greedy, DP esatto, annealing, MILP HiGHS (spike 2)

Strumento: [`bench/opt1_spike/graph_solver.cpp`](../bench/opt1_spike/graph_solver.cpp)
(portabile, solo CPU). Ogni ordine è giudicato dallo **stesso valutatore**:
il compilatore reale (`CompileOptions::order`: fusione, load/store,
memoryless, aliasing con un sizer finto allineato a 64 KiB, barriere) più
un modello T = Σ unità max(byte/569 GB/s, ALU/4,06 Top/s IMAD + triangoli/
8 G/s) + 11 µs per render pass + 7 µs per pass compute (spike 1),
J = T + 0,1 ms/GiB × heap. Metodi: **greedy** = Kahn per dichiarazione (F2);
**DP** sui downset (stato = pass eseguiti + gruppo di render aperto, fusione
identica al compilatore; costo = load/store dei gruppi chiusi + fissi + picco
di byte vivi), esatto sotto 100–200 mila stati per livello, altrimenti beam;
**annealing** sugli ordini topologici con il valutatore reale (5–20 mila
valutazioni, dal greedy); **MILP** con HiGHS 1.15.1 (FetchContent, 1 thread,
limite 120 s): assegnazione x[pass][posizione], vincoli di precedenza,
risparmio delle coppie fondibili adiacenti, picco ≥ Σ byte vivi a ogni
posizione (linearizzazione esatta dei vivi con x cumulati).

| Grafo (pass vivi) | greedy J | DP | annealing | MILP | Nota |
|---|---:|---|---|---|---|
| 1 vista: deferred 21, forward+ 17, post 26 | ottimo | = greedy, ≤0,001 s, esatto | = greedy | = greedy, 0,02–0,06 s, ottimo | la struttura impone l'ordine |
| async 11 | 3,155 | **−17,1% byte, −26,6% heap**, esatto 0,000 s | idem 0,02 s | idem 0,01 s | fusione G-buffer+Lighting |
| async ×2 (21) | 6,144 | −17,4% / −23,1%, esatto 0,13 s | idem 0,25 s | idem 0,33 s | |
| deferred ×2 (41) | 4,411 | heap −3,9%, esatto 0,04 s | idem 0,70 s | idem, **limite 120 s** (gap 0,6%) | |
| post ×2 (51) | 3,918 | = greedy, esatto 0,001 s | = | = , limite (gap 3%) | |
| async ×3 (31) | 9,133 | −17,6% / −20,4%, beam 5,3 s | −17,6% / −13,6% 0,6 s | −17,6% / −19,5%, ottimo 7,5 s | |
| deferred ×3 (61) | 6,533 | beam: heap **+32,9%** | **heap −4,8%** 1,9 s | limite, gap 10%, heap +35% | |
| post ×3 (76) | 5,796 | = greedy, esatto 0,045 s | = | limite, heap +20% | |
| async ×4 (41) | 12,125 | beam: heap +3,4% | heap −12,2% 1,1 s | **heap −17,5%, ottimo 57,6 s** | |
| deferred ×4 (81) | 8,657 | beam: heap +65% | = greedy 4,3 s | limite, +3,8% byte, +65% heap | |
| post ×4 (101) | 7,677 | = greedy, esatto 2,8 s (2,1 M stati) | = 6,1 s | limite, +81% heap | |
| async ×6 (61) | 18,155 | beam: −17,7% byte, heap +46% | −17,7% / heap +12%, 2,5 s | limite, heap +21% | la fusione allunga le vite dei compute |
| deferred ×6 (121), post ×6 (151) | 12,952 / 11,487 | beam: heap +41..+118% | = greedy 12–21 s | **nessuna soluzione** in 120 s (80–123 mila righe) | |

- Il **greedy è già ottimo sull'ordine** quando la catena di dati impone la
  sequenza (scenari 0–2): lì l'ottimizzatore non guadagna byte né picco.
- Il **DP esatto** è il più veloce finché gli stati restano pochi (grafi
  "stretti" fino a 101 pass in 2,8 s); con viste indipendenti in parallelo
  gli stati esplodono e il beam per J sbaglia il picco (il criterio "J
  minimo per stato" non è esatto sul termine di massimo: serve il fronte di
  Pareto costo/picco).
- L'**annealing** sul valutatore reale non peggiora mai il greedy (parte da
  lì) e trova i migliori heap nei grafi grandi, in 1–21 s.
- **HiGHS**: ottimo e più bravo sul picco nei grafi medi (async ×4: −17,5%
  contro −12,2%) ma 57 s; da 41–51 pass in su va al limite con soluzioni
  peggiori del greedy, oltre 120 pass non ne trova. Il modello lineare non
  vede fissi, barriere e sovrapposizione.
- **Decisione**: ottimizzatore portabile con **DP esatto (fronte di Pareto
  costo/picco, limite di stati) + annealing sul valutatore reale**, il
  greedy sempre candidato; **HiGHS non adottato** (dipendenza pesante per
  un modello più povero e tempi non limitati); il codice MILP resta nello
  spike come riferimento.
- Un trade-off vero: in async ×6 l'ordine che fonde riduce i byte del 17,7%
  ma allunga le vite dei risultati compute (heap +12–46%): il piano deve
  dichiarare cosa sceglie.

### 3. Compressione lossless con heap placement e alias (spike 3, B-11 esteso)

B-11 esteso (`bench/soc/b11_compression.cpp`): la stessa texture RGBA8
4096² creata con `device->newTexture` (base), in un **heap placement**
privato non tracciato (come `TransientHeap`, a 4 unità di allineamento) e
**aliasata**: nello stesso heap e offset una RGBA16Float scritta con dati
casuali, barriera `Device|ResourceAlias`, poi la RGBA8. Casi optimized
(`allowGPUOptimizedContents`), plain e **pfview** (optimized +
`TextureUsagePixelFormatView`, documentato come disattivante: controllo
negativo, provato fallito rendendolo uguale a optimized → 1,34–1,37 contro
la soglia). Rimisura a macchina quieta, 3 run.

| Percorso | Guadagno (media geometrica celle comprimibili, optimized/plain) | pfview/plain |
|---|---:|---:|
| device | 1,32–1,36 | 1,00–1,01 |
| heap | 1,31–1,37 | 0,98–1,01 |
| heap + alias RGBA16F → RGBA8 | 1,35–1,37 | 0,98–0,99 |

- **Heap placement e alias tra formati diversi mantengono la compressione**
  (stesso vantaggio del percorso device in scrittura compute, render target
  e lettura).
- `heapTextureSizeAndAlign` RGBA8 4096²: optimized 67.633.152 B allineamento
  2048, plain 67.108.864 B allineamento 128: **+1/128 di metadati e
  allineamento 2048 B** per le texture compresse; `PixelFormatView` toglie
  entrambi (compressione disattivata). RGBA16F: 134.742.016 contro
  134.217.728.
- Vincoli per OPT-1.3: nessun vincolo di formato per la compressione sugli
  alias (coppia misurata RGBA16F→RGBA8); niente `PixelFormatView` sulle
  risorse del grafo; offset allineati a 2048 B per le compresse (già dati da
  `heapTextureSizeAndAlign`). Non misurati: altre coppie di formati, alias
  parziali, costo una tantum del primo uso dopo l'alias.

### 4. Rematerializzazione nella tile contro store + load (spike 4, OPT-1.2)

Metodo: segnali "economici" (velocity dal G-buffer / dalla scena, CoC del
DoF) che dipendono solo dal pixel e dalla sua profondità: con
`--graph-remat` il produttore non li scrive e ogni consumatore li ricalcola
dalla profondità (che legge comunque) con N passi ALU
(`--graph-remat-cost`, 16 = riproiezione tipica). Immagini identiche (0
pixel) con e senza; 3 giri a ordine ruotato, GPU libera (un primo giro con
un altro benchmark sul GPU aveva CV 13–47%: scartato).

| Scenario, segnale | Costo | DRAM | Frame p50 (CV) |
|---|---:|---:|---:|
| 0, nessuno | — | 565,9 MiB | 2,664 ms (0,05%) |
| 0, Velocity (letto 1×1 da TAA) | 16 | 537,7 (−5,0%) | 2,704 (0,03%) = **+1,5%** |
| 0, Velocity | 64 | 537,7 | 2,860 = +7,4% |
| 0, Velocity | 256 | 537,7 | 3,534 = +32,7% |
| 2, nessuno | — | 524,3 | 2,988 (0,02%) |
| 2, Velocity (letto 16×16 dal tile max) | 16 | 510,2 (−2,7%) | ~~3,381 = +13,1%~~ → **+0,05%** (rimisura) |
| 2, CoC (letto 1×1 e 2×2) | 16 | 517,3 (−1,3%) | ~~3,060 = +2,4%~~ → **−0,49%** |
| 2, entrambi | 16 | 503,2 (−4,0%) | ~~3,438 = +15,0%~~ → **−0,57%** |

**Correzione (2026-10-01)**: le righe barrate dello scenario 2 erano falsate
da un bug del banco di prova trovato dalla selezione per misura
(`tools/graph_select.py`, immagini diverse di 5,76 M pixel): quando il
consumatore non leggeva già la profondità, la rematerializzazione gliela
aggiungeva come input ordinario, che entrava nell'hash (immagine sbagliata)
e, nel tile max, costava 256 letture in più per thread. Ora è un input
"sorgente" non sommato al valore (commit `50d2f81`); immagini identiche (0
pixel) per ogni scelta in tutti gli scenari. Rimisura (3 giri ruotati, CV ≤
0,06%): scenario 2 Velocity 3,0512 contro 3,0498 ms (+0,05%), CoC 3,0350
(−0,49%), entrambi 3,0323 (−0,57%); scenario 0 Velocity 2,7436 contro
2,7015 (+1,56%, confermato). Tra sessioni la stessa configurazione si sposta
dell'1–2% (2,66 → 2,70 ms): si confrontano solo varianti misurate a giri
alternati nella stessa sessione.

Su M5 Max **ricalcolare non conviene quando il consumatore è limitato da ALU
o latenza** (TAA dello scenario 0: +1,5% con 16 passi, +7% con 64, +33% con
256); dove il consumatore è vicino alla banda o legge il segnale una volta
(scenario 2) il saldo è nullo o lievemente positivo e i byte scendono del
2,7–4%. Il pareggio dipende dal consumatore: la rematerializzazione è una
scelta del piano per segnale e per scenario, decisa dalla misura
(OPT-1.2), non una regola.

### 5. Riordino, sovrapposizione e seconda coda (spike 5)

Metodo: `--graph-order` impone un ordine (validato: ogni pass vivo una
volta, dipendenze rispettate); 3 giri a ordine ruotato, frame p50.

Scenario 3 (async; il greedy mette G-buffer, GI, particelle, Lighting: il
compute tra G-buffer e Lighting impedisce la fusione):

| Ordine | Coda del compute | Frame p50 (CV) | DRAM stimata | Note |
|---|---|---:|---:|---|
| A greedy | async | **3,431 ms** (0,03%) | 494,8 MiB | G-buffer si sovrappone al compute async |
| B compute prima, G-buffer+Lighting fusi | async | 3,660 (0,03%) = +6,7% | 410,5 (−17%) | 3 memoryless, heap −27%; l'attesa della coda async sale all'inizio del gruppo fuso: G-buffer non si sovrappone più |
| C ombre, compute, G-buffer+Lighting | async | 3,659 (0,03%) = +6,7% | 410,5 | idem |
| A greedy | grafica | 3,927 (0,03%) = +14,5% | 494,8 | |
| B | grafica | **3,428 (0,04%)** | 410,5 | le 4 cascate d'ombra si sovrappongono al compute lungo (particelle 1,1 ms, ALU): nessuna barriera tra loro |
| C | grafica | 3,860 (0,02%) = +12,5% | 410,5 | stessa fusione, ma il gruppo fuso aspetta il compute: niente sovrapposizione |

Immagini identiche (0 pixel) tra A e B. Conclusioni: (1) fusione e
sovrapposizione tra code sono in conflitto (le attese si agganciano
all'inizio del gruppo): il modello di costo deve valutarle insieme; (2) su
una coda, lavoro indipendente di tipo diverso (raster di geometria dopo un
compute ALU) si sovrappone quasi del tutto se nessuna barriera lo separa
(−0,43 ms su 0,4 ms di ombre).

Scenari 0 e 1 (ombre spostate dopo SSAO / dopo la light grid): +0,5% e +0,4%,
nessuna sovrapposizione. Causa (dal dump del grafo): **ogni primo uso di un
transitorio piazzato ha una barriera di coda dopo tutti gli stadi che
toccano la sua memoria** (frame precedente compreso, più gli alias): con
alias misti raster/compute diventa `fragment|dispatch → fragment` e aspetta
anche il compute appena precedente, indipendente. Dove la barriera resta
`fragment → fragment` (scenario 3 B) il compute precedente continua. Le
barriere di aliasing e di protezione tra frame sono il limite della
sovrapposizione: materia di OPT-1.4 (stadi minimi) e OPT-1.3 (alias per
classe di stadi) e un termine del modello (OPT-1.8).

### 6. Anelli di upload e storage mode (B-29, OPT-1.10)

Nuovo benchmark B-29 `upload.storage_mode` (`bench/soc/b29_upload.cpp`):
scrittura CPU in buffer `shared` WriteCombined contro DefaultCache (memcpy
1/4/8 thread, store da 16/64 byte, store sparsi), lettura CPU, lettura GPU
di `shared`/`private`/`shared` WC da 1 MiB a 1 GiB. Controlli negativi
provati falliti: WC reso DefaultCache → rapporto di lettura 1,13 e
`cpuCacheMode()` diverso → fail; 3 passate al posto di 4 → linearità 1,39 →
fail.

- Scrittura CPU: **WC = cached** (memcpy 0,98–1,01× da 1 MiB in su; store
  16 B 33,5–34 GB/s in entrambi; store 64 B 127–133); unica differenza gli
  store sparsi da 16 B a 64 KiB–1 MiB (WC 22–24 contro 31–35 GB/s).
- Lettura CPU da WC **20× più lenta** (7,0 contro 108–141 GB/s).
- Lettura GPU: **`private` = `shared`** (rapporto 0,999–1,004 da 1 MiB a
  1 GiB; 571–573 GB/s a 1 GiB, come B-08/B-09); WC letto dal GPU uguale.
- Il motore usa già `shared` + WC per gli anelli (`upload_ring.cpp`,
  `metal_texture_manager.cpp`) e `private` per ciò che il GPU legge
  (geometria, texture, transitori): OPT-1.10 è soddisfatto; WC resta (non
  costa in scrittura, il motore non rilegge gli anelli).

### 7. SLC tra produttore e consumatore adiacenti (OPT-1.7, primo sguardo)

Scenario 0, Bloom down 1 (legge l'HDR da 28 MiB scritto da Lighting) subito
dopo Lighting contro dopo TAA (che legge 112 MiB): **0,0707 contro 0,0763 ms
(−7%)**, frame 2,663 contro 2,637 ms (l'altro ordine è lo 0,9% più veloce
per effetti di sovrapposizione): una parte dell'HDR si legge dalla SLC se il
consumatore è adiacente; l'effetto sul frame è piccolo.

### Trappola di misura nuova

Con un processo CPU al 100% su un core (il solver dello spike 2) il frame
degli scenari diventa **bimodale tra run identici** (2,67 oppure 3,6–3,9
ms, CV 1–4%) pur con GPU a P13 per il 77% del tempo; a macchina quieta torna
2,662 ms (CV 0,02%). Anche un benchmark `soc_bench` concorrente (agente)
porta il CV al 13–47%. Tutte le misure di OPT-1 si fanno a macchina quieta,
in serie.

---

## OPT-1 — Risultati: ottimizzatore del grafo, piani misurati, barriere e aliasing

**2026-10-01** · M5 Max, macOS 27.2, Release, alimentazione, macchina quieta
(nessun agente né solver in esecuzione) · tabelle complete in
[`perf-log.md`](perf-log.md) ("OPT-1 — chiusura").

### Cosa è stato costruito

- **Ottimizzatore portabile** (`src/rendergraph/optimizer/`): piano
  (`GraphPlan`: famiglia, chiave strutturale FNV-1a, ordine per nome, scelte
  di costruzione remat/async, politiche di alias e barriere, metriche),
  modello di costo (`GraphCostModel`: unità = max(byte/banda, ALU +
  geometria) + fisso, simulazione delle due code con sovrapposizione e
  condivisione), ricerca (DP esatta sui downset con fronte di Pareto, beam
  oltre il limite di stati, annealing sul valutatore reale; il greedy è sempre
  candidato). Test: sullo scenario 3 l'ottimizzatore raggiunge l'ottimo dei
  5040 ordini topologici (controllo negativo: senza annealing e con beam di 1
  stato si ferma a J 2,892 contro 2,807 e il test fallisce).
- **Motore**: `--graph-opt off|greedy|plan`, `--graph-plan`; il piano si
  applica solo se la chiave del grafo costruito coincide (la dimensione del
  drawable non conta), altrimenti ripiego registrato; report schema 4 con
  l'oggetto `graph`.
- **OPT-1.3** `AliasPolicy::Coloring` (mai peggio del greedy, limite = max
  byte vivi) e `ColoringStageClass`; **OPT-1.4** `BarrierPolicy::Minimal`;
  **OPT-1.5/1.6/1.7** `graph_lint`, `graph_budget` (byte per pass, quota del
  budget per tier, working set contro la SLC stimata, coppie di riuso),
  dump esteso; **strumenti** `tools/graph_opt` (piani offline, `--top`),
  `tools/graph_select.py` (adozione per misura), `tools/graph_scenarios.sh` +
  `tools/scenario_table.py` (uscita).

### Il modello ordina, la misura decide

Contro le 6 varianti misurate dello spike 5 (scenario 3) il modello rispetta
l'ordinamento misurato (async: A < B = C; una coda: B < C < A) e la stima dei
risparmi è del giusto ordine (B-sync contro A-sync 0,56 ms predetti, 0,50
misurati), ma i frame assoluti sono sottostimati (0,62–0,91×: la latenza dei
pass non è modellata) e i pass limitati da latenza/ALU ingannano la scelta
della rematerializzazione e della sovrapposizione. Con il solo modello il
piano preferito dello scenario 3 (niente coda async, niente fusione, ombre
intercalate) misurava +7%. Procedura adottata: `graph_opt --top 6` con J
pesato sul tempo e con J pesato su memoria e byte (`--gamma 2 --beta 0.5`),
poi `graph_select.py`: `off` + ogni candidato distinto, 3 giri ruotati ×
600 frame, immagine identica a `off` obbligatoria; adottato il più veloce se
batte `off` oltre l'1%, altrimenti quello che riduce di più byte + heap
senza perdere oltre l'1%, altrimenti il piano di base.

| Scenario | Candidati | Adottato | Misura contro off (selezione) |
|---|---:|---|---|
| 0 deferred | 7 | ordine DP+annealing, Coloring + Minimal | −6,7% frame |
| 1 forward+ | 8 | ordine DP+annealing, Coloring + Minimal | −3,6% frame |
| 2 post | 7 | Velocity e CoC rematerializzate, Coloring + Minimal | −0,8% frame (rumore), −3,7% byte, −8,0% heap |
| 3 async | 12 | G-buffer + Lighting fusi, GI sulla coda async, particelle sulla grafica | −2,3% frame, −15,7% byte, −27,2% heap |

La selezione ha trovato un **bug del banco di prova**: con la
rematerializzazione nello scenario 2 tutte le immagini differivano (5,76 M
pixel); la profondità aggiunta al consumatore come sorgente del ricalcolo
entrava nell'hash del valore. Corretto (`50d2f81`, input "sorgente"),
test aggiunto, spike 4 corretto sopra.

### Uscita e attribuzione

`tools/graph_scenarios.sh` (perf-log): piano contro `off` −8,4% / −3,5% /
−0,5% / −2,3% di frame sugli scenari 0–3; 0 pixel e 0 messaggi di
validazione per tutti i modi; `greedy` (politiche OPT-1 con l'ordine greedy)
neutro (±0,1%). Attribuzione sugli scenari 0 e 1 con l'ordine dei piani:
solo ordine (barriere di fine F4) −3,5% / −2,1%; ordine + barriere minime
−8,4% / −3,6%; barriere minime con l'ordine greedy 0%. Il guadagno viene
dalla **sovrapposizione** (OPT-1.8): l'ordine mette le cascate d'ombra
accanto a compute indipendenti (SSAO, light grid) e le barriere minime non le
fanno più aspettare il compute.

### Obiettivo della ROADMAP (−25% byte, −20% picco rispetto alla fine di F4)

Raggiunto solo per il picco dello scenario 3 (−27,2%; byte −15,7%). Scarto
spiegato per scenario:
- **0 deferred, 1 forward+**: il grafo impone già l'ordine dei dati (SSAO e
  light grid compute tra i pass raster): nessun ordine permette nuove fusioni
  o memoryless (DP esatto e annealing concordano, spike 2); l'aliasing greedy
  è già al limite inferiore (heap = max byte vivi, OPT-1.3); la
  rematerializzazione della velocity toglierebbe il 5% dei byte ma costa
  +1,5% di frame (TAA limitato dall'ALU) e la selezione la scarta. Guadagno
  ottenuto in tempo, non in byte.
- **2 post**: unica leva sui byte la rematerializzazione (−3,7% byte, −8%
  heap, tempo neutro): adottata.
- **3 async**: la leva è l'ordine (compute prima del G-buffer → fusione e 3
  memoryless): −15,7% byte, −27,2% heap, −2,3% frame; il resto dei byte è
  lavoro reale dei pass (G-buffer letto da SSAO/lighting, ombre campionate,
  atlante GI).
- I byte che restano sono quelli richiesti dagli algoritmi così come sono
  scritti; ridurli oltre richiede cambiare gli algoritmi (pass di profondità
  per SSAO prima di un G-buffer fuso con il lighting, V-buffer: F5/OPT-4),
  non l'ordine o la memoria del grafo.

### Barriere minime: cosa è dimostrato

`BarrierPolicy::Minimal` restringe la barriera di primo uso di una memoria
agli stadi degli accessi massimali (ordinati dopo gli altri da barriere
realmente presenti nel piano); dimostrazione per transitività nel codice;
test: Minimal ⊆ Conservative su 300 grafi casuali (1676 barriere ristrette su
5093) e sugli scenari 1–3 viste; 0 pixel e 0 messaggi su tutti gli scenari
anche con `--no-vsync` e senza fusione. **Limite dichiarato**: il controllo
negativo sul GPU non è riuscito a rendere visibile una corsa nemmeno
togliendo del tutto le barriere di primo uso (le barriere di dipendenza
bastano in questi carichi; togliendo quelle l'immagine cambia di 5,76 M
pixel): la sicurezza di Minimal poggia sulla dimostrazione e sui test, non
sulle immagini. Per questo il default resta `--graph-opt off`; i piani
misurati che la usano sono opzionali (`--graph-opt plan`).

### Aliasing, lint, budget

- Coloring = greedy su tutti gli scenari 1–3 viste (il greedy è già al limite
  inferiore); su 400 insiemi casuali di intervalli Coloring è minore in 55 e
  raggiunge sempre il limite. `ColoringStageClass` costa 11–19 MB sugli
  scenari 0–2 (il prezzo di non mischiare raster e compute).
- Lint (OPT-1.6): nessun errore; note per ogni transitorio salvato (perché
  campionato dopo: ombre, profondità, velocity, HDR, LDR), G-buffer degli
  scenari 0/3 che attraversa due gruppi per via del compute in mezzo, store
  conservativo della profondità solo letta (forward+).
- Budget (OPT-1.5, stime del grafo): 458–566 MiB per frame = 5–6% del budget
  a 60 fps su M5 Max (569 GB/s, misurato), 19–23% su M5 base (153,6 GB/s,
  **esterno**), 29–36% su M3 base (**esterno**). Senza contatori hardware
  (F4.4) i byte restano stime; la verifica indiretta è lo spike 1.
- SLC (OPT-1.7, stima 71,4 MiB): sopra la stima TAA (112,5 MiB), DoF
  composite (77,3), GI probe update (136); effetto misurato dell'adiacenza
  produttore→consumatore: −7% sul pass (spike 7).

### Più viste (solo predizione)

`graph_opt --views 2,3` (modello pesato sul tempo): scenario 3 −17,4/−17,6%
byte, −28,6/−11,8% heap; scenari 0–2 il modello sceglie ordini con heap
+6…+43% per un frame predetto migliore. Non misurati né adottati: il modello
non è affidabile da solo (sopra), i piani per più viste vanno scelti con
`graph_select.py --views N` quando serviranno.

### Bug trovati durante la fase

- Storia TAA: una sola coppia di texture per tutte le viste (perdita e viste
  che condividevano la storia) → una coppia per vista (`7ecb750`).
- Rematerializzazione: profondità sorgente sommata al valore (`50d2f81`).
- Piano con un pass che diventa culled (CoC rematerializzata): l'ordine
  forzato lo rifiutava → i pass non pianificati in coda sono solo quelli vivi
  (`b49a716`, anche `--graph-order`).
- Misure: un processo al 100% su un core o un benchmark concorrente rendono
  il frame bimodale; le misure sono rifatte a macchina quieta.

---

## F5 — Spike: percorso CPU, delta, draw guidati dalla GPU, culling, barriere, gerarchia

**2026-10-01** · Apple M5 Max (Apple10, 40 core GPU), macOS 27.2, all'alimentazione
(batteria 100%), schermo acceso (`caffeinate -d`), macchina quieta (nessun
agente, nessun altro benchmark), build Release. S1 e S7a girano nel motore;
S2–S7 nello strumento [`bench/f5_spike`](../bench/f5_spike/README.md)
(harness `bench/soc`: ≥ 15 ripetizioni per metrica, mediana, 3 run, CV tra
run; risultati in `bench/results/f5-spike/m5max-macos27.2.json`). Ogni
spike ha un controllo negativo integrato che verifica la propria capacità
di fallire (prova con il meccanismo rotto a mano annotata nel commit
dell'agente); `f5_spike --validate` (API + shader validation): 6 benchmark,
0 falliti, 0 messaggi. Il codice degli spike S2–S6 è stato scritto da
sotto-agenti senza misure; tutte le misure qui sotto sono state prese dopo.

### 1. Baseline del percorso CPU a 10K/100K/1M istanze (S1)

Metodo: Stress Test resa parametrica solo per lo spike
(`PHOSPHOR_SPIKE_INSTANCES`, `_DYNAMIC` = ogni istanza si muove ogni frame
scorrendo l'array denso dell'ECS, `_MESH=cube`, `_MESHES=K` = K copie della
mesh → K batch), riga `SPIKE-S1` con i ms CPU per fase del frame, i byte
scritti nell'anello di upload e i comandi CPU del forward (draw + stato +
binding). Release, 3200×1800, `--no-vsync --no-ui`, 300 frame + 60 di
riscaldamento, un run per riga (spike, non uscita).

| N | mesh (tri) | moto | CPU frame p50 (p99) | sim | extract + sort | prepare (upload) | graph encode | submit | upload/frame | Forward GPU p50 (p99) | comandi CPU |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10K | sfera (96) | no | 0,83 (1,07) | 0 | 0,51 | 0,10 | 0,09 | 0,06 | 1,53 MiB | 1,49 (1,91) | 10 |
| 10K | sfera | sì | 1,03 (1,54) | 0,37 | 0,42 | 0,08 | 0,07 | 0,05 | 1,53 MiB | 0,75 (1,63) | 10 |
| 100K | sfera | no | 1,95 (3,44) | 0 | 1,64 | 0,36 | 0,04 | 0,03 | 15,26 MiB | 1,74 (6,56) | 10 |
| 100K | sfera | sì | 3,02 (3,37) | 1,24 | 1,45 | 0,31 | 0,02 | 0,03 | 15,26 MiB | 1,77 (3,50) | 10 |
| 1M | cubo (12) | no | 17,67 (17,92) | 0 | 14,39 | 3,17 | 0,04 | 0,04 | 152,6 MiB | 6,39 (9,77) | 10 |
| 1M | cubo | sì | 30,15 (30,74) | 12,48 | 14,38 | 3,19 | 0,04 | 0,04 | 152,6 MiB | 5,59 (9,77) | 10 |
| 1M | sfera (96) | no | 17,87 (18,07) | 0 | 14,55 | 3,21 | 0,04 | 0,04 | 152,6 MiB | 13,16 (31,18) | 10 |

- Il costo CPU del percorso attuale è **O(N) per frame anche a scena
  ferma**: extract + sort 14,4 ms e copia di 152,6 MiB (istanze + un
  materiale per entità) 3,2 ms a 1M; con 1M istanze in moto la simulazione
  CPU (matrice TRS per istanza) aggiunge 12,5 ms → 30 ms di CPU, 33 fps.
- I comandi CPU **non** crescono con le istanze (un batch istanziato, 10
  comandi) ma con i **batch** (mesh × classe di culling): 10.009 comandi con
  10.000 mesh (riga S7). Il controllo negativo di O8 va quindi espresso sui
  bucket (bench e scene con più mesh), non sul numero di istanze.
- Raster: 1M cubi tutti disegnati (12 M triangoli, nessun culling) 5,6–6,4
  ms di Forward; 1M sfere (96 M triangoli) 13,2 ms p50 e 31 ms p99, oltre
  il ginocchio del partial render di B-12 (23,7 M triangoli con 16
  varyings): il bench 1M deve usare mesh da ≤ 12–20 triangoli e affidarsi
  al culling.
- Tempi GPU senza `--gpu-timing-serial` (DVFS e sovrapposizione dei
  frame): indicativi, servono solo a dimensionare il bench.

### 7a. Encoding parallelo F2.5 sul percorso CPU (S7, residuo di F2)

100K cubi in 10.000 mesh (10.000 batch, un draw ciascuno), statici, 3 run
alternati per modo: encoding del Forward (fase `graph`) **0,50–0,53 ms →
0,24–0,26 ms** con `--debug-split-encoding` (4 thread), CPU del frame
3,75–4,00 → 3,55–3,67 ms (−6%). Con un solo batch (S1) l'encoding costa
0,02–0,09 ms e F2.5 non ha nulla da dividere. Nel percorso GPU-driven la
CPU codifica un numero costante di comandi: F2.5 resta utile solo al
percorso CPU (`--gpu-driven off`) e ai pass futuri con molti draw CPU.
(Il contatore dei comandi dello spike non è thread-safe: con lo split
sottoconta, ignorato.)

### 2. Aggiornamenti delta e attesa tra frame (S2)

1M record da 80 B (layout `GPUInstance`), K slot distinti cambiati per frame.
(a) **scatter**: record delta da 96 B (slot + istanza) scritti dalla CPU
nella regione del frame di un anello condiviso, kernel di scatter in un
buffer `private` persistente; (b) **copie dirette**: 3 copie `shared`
persistenti, ogni frame scrive i propri cambi e riapplica quelli dei due
frame precedenti; (c) **ricarico completo** (oggi).

| Cambiati | (a) CPU scrittura | (a) GPU scatter | (a) byte | (b) CPU scrittura | (b) byte |
|---:|---:|---:|---:|---:|---:|
| 0,1% (1.049) | 0,002 ms | 0,003 ms | 0,10 MB | 0,030 ms | 0,25 MB |
| 1% | 0,027 ms | 0,008 ms | 1,0 MB | 0,384 ms | 2,5 MB |
| 10% | 0,768 ms | 0,100 ms | 10,1 MB | 4,57 ms | 25,2 MB |
| 100% | 10,3 ms | 1,34 ms | 100,7 MB | 48,8 ms | 251,7 MB |
| (c) completo | 1,21 ms (memcpy 80 MiB) | — | 83,9 MB | | |

- Lettura GPU di 1M record (80 MiB) da `private` 0,145 ms, da copia
  `shared` 0,143, dalla regione dell'anello 0,145: identiche (conferma B-29
  in questo schema di accesso, ~580 GB/s).
- **Attesa tra frame**: 3 frame in volo GPU-bound (~4,8 ms/frame: scatter →
  cull → render pass che legge le istanze nel vertex shader). Con la
  barriera di coda `Dispatch|Vertex → Dispatch` all'inizio del frame (ciò
  che il grafo emette per un buffer persistente scritto dal grafo)
  4,848 ms/frame, senza barriera 4,843, copie per frame 4,839: **sovrapposizione
  persa 0,009 ms (0,2%)**. La variante senza barriera è rimasta esatta:
  la corsa non si è manifestata, quindi non è una prova di sicurezza.
- Le copie dirette costano 14× lo scatter in CPU (store sparsi da 80 B
  ×3 in memoria non in cache) e 3× la memoria.
- Esattezza: dopo ogni strategia memoria GPU = specchio CPU byte per byte;
  controlli integrati: un record omesso → trovato esattamente quello slot;
  riapplicazione n−2 saltata → 10.278 slot diversi (attesi 10.278).

**Decisione D1**: scatter dall'anello del frame in buffer `private`
persistenti (la "scrittura diretta" della ROADMAP è la scrittura dei record
delta dalla CPU nella memoria unificata, senza staging); oltre ~1/8 di slot
cambiati la CPU copia l'intero specchio (memcpy, 1,2 ms a 1M) e la GPU lo
copia nel persistente. Attesa tra frame trascurabile: nessuna copia per
frame.

### 3. Emissione dei draw (S3)

100.000 istanze in B bucket (mesh da 2–12 triangoli, classe di culling
b % 3), lista visibile compattata, target 1024² R32Uint + profondità
reverse-Z. Tempo GPU del render pass (span meno pass vuoto), CPU di
encoding, comandi CPU:

| B | ICB `reset` | ICB 0 istanze | ICB compattato | ICB Apple10 per comando | indiretti per bucket | diretti |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0,165 ms · 3 cmd | 0,166 · 3 | 0,169 · 7 | 0,166 · 1 | 0,160 · 3 | 0,154 · 3 |
| 100 | 0,237 · 7 | 0,242 · 7 | 0,245 · 7 | 0,239 · 1 | 0,228 · 104 | 0,236 · 94 |
| 1.000 | 0,281 · 7 | 0,287 · 7 | 0,285 · 7 | 0,286 · 1 | 0,230 · 1.004 | 0,224 · 904 |
| 10.000 | 0,735 · 7 | 0,764 · 7 | 0,748 · 7 | 0,839 · 1 | 0,824 · 10.004 | 0,761 · 9.004 |
| 10.000, 90% vuoti | 0,320 | 0,378 | 0,097 | 0,505 | 0,491 | 0,039 |

- CPU di encoding a 10.000 bucket: ICB 0,019 ms (7 comandi, costante con
  B), indiretti 0,44 ms, diretti 0,41 ms. Kernel di codifica dell'ICB
  0,008–0,012 ms, reset sulla timeline GPU (`resetCommandsInBuffer`)
  0,0003 ms.
- Comandi vuoti eseguiti: ~31 ns l'uno con `reset()`, ~37 ns come draw a 0
  istanze; il compattato elimina il costo ma usa un range indiretto (vedi
  S5 e validazione). Lo stato di culling per comando di Apple10 (compila:
  `render_command::set_cull_mode/set_front_facing_winding`, immagine
  identica) costa +14% a 10.000 bucket e +58% con 90% vuoti: su Apple10 non
  conviene, i range per classe valgono su entrambe le famiglie.
- ICB `private`, argument table dell'encoder ereditata, `instance_id`
  comprende `base_instance`: tutte le varianti danno la stessa immagine e
  gli stessi flag di copertura; controllo integrato: un bucket tolto →
  rilevato a ogni B.
- **Validazione tra command buffer**: senza layer e con il solo API layer
  disegnano esattamente v1 (codifica ed esecuzione nello stesso command
  buffer), v2 (codifica in A, esecuzione in B, un solo commit, come F2.5),
  v3 (codifica sulla seconda coda, evento, esecuzione sulla principale,
  come F2.6), v4 (stesso ICB ricodificato per 12 frame, 3 in volo). Con
  **shader validation**: v1–v3 disegnano; v4 con draw a 0 istanze **non
  disegna il primo frame**; un ICB codificato in un commit precedente ed
  eseguito con range indiretto non disegna (il B-17 osservava lo stesso
  con un ICB codificato in un command buffer precedente). Nessun messaggio
  in tutti i casi: un controllo che non disegna si vede solo dai pixel.

**Decisione D2**: ICB codificato dalla GPU, un comando per bucket, ordine
(classe, mesh, slot), tre range fissi per classe di culling con lo stato
impostato dalla CPU (Apple9 e Apple10), bucket vuoti con `reset()`,
reset dell'ICB sulla timeline GPU, codifica ed esecuzione nello stesso
frame. Nessun range indiretto (D3). I controlli sotto shader validation
sono a pixel (`visual_check` confronta l'ultimo frame), mai "0 messaggi".

### 4. Culling di 1M istanze e compattazione (S4)

1.048.576 istanze in 8 bucket contigui (dimensioni dispari: i SIMD-group
attraversano i confini), rotazioni e scale non uniformi, 5% specchiate;
camera e proiezione come `Camera::updateMatrices` (reverse-Z infinita,
near 0,05), 3200×1800; sfera mondo, 5 piani, distanza, dimensione
proiettata; 33.904 visibili.

| Variante | Tempo GPU | Determinismo (10 run) | Discrepanze col riferimento CPU |
|---|---:|---|---|
| (a) atomici per bucket (aggregati per SIMD-group) | 0,137 ms | 9/9 run in ordine diverso | 0 (fast e safe) |
| (b) reduce-then-scan stabile (3 dispatch: flag 0,146 + scan 0,002 + scrittura 0,010) | 0,158 ms | 0/9: identico byte per byte | 0 (fast e safe) |

- Riferimento CPU con le stesse formule e lo stesso ordine delle
  operazioni (`FP_CONTRACT OFF`): 0 discrepanze fuori dalla banda (1e-4
  relativo), 0 dentro (30 istanze nella banda), sia con la libreria
  fast-math sia con `MathModeSafe`; tempi identici nei due modi.
- La scansione a passata singola con look-back non è stata implementata:
  il progresso tra threadgroup non è garantito sulle GPU Apple.
- Controllo integrato: un piano con il segno invertito → 40.764 slot
  diversi.
- **Trovato**: `Camera::getFrustumPlanes()` usa l'estrazione OpenGL; con il
  reverse-Z infinito il piano 4 (riga3 + riga2) non è il near (accetta i
  punti dietro la camera): il near è riga3 − riga2 e non esiste un far.
  Nessun chiamante oggi; da correggere in F5.

**Decisione D4**: compattazione stabile (+0,02 ms rispetto agli atomici a
1M) → liste per bucket contigue nell'ordine degli slot, immagini
deterministiche e identiche al percorso CPU.

### 5. Stadio del consumatore di argomenti indiretti e ICB (S5)

Produttore compute di ~6 ms che scrive gli argomenti alla fine; 20
ripetizioni per variante, "sbagliata" = il consumatore ha visto gli
argomenti iniziali (stale); stesse conclusioni con buffer/ICB `shared` e
`private`, senza validazione e sotto validazione (0 messaggi, le corse non
spariscono).

| Consumatore | nessuna | q Dispatch→Vertex | q Dispatch→Fragment | q Dispatch→Object\|Mesh | barriera del produttore Dispatch→Vertex |
|---|---:|---:|---:|---:|---:|
| `drawIndexedPrimitives` indiretto | 20/20 | 0/20 | 20/20 | 20/20 | 0/20 |
| ICB codificato dalla GPU | 20/20 | 0/20 | 20/20 | 20/20 | 0/20 |
| ICB con range indiretto | 20/20 | 0/20 | 20/20 | 20/20 | 0/20 |

| Consumatore | nessuna | encoder Dispatch→Dispatch | coda Dispatch→Dispatch (encoder successivo) |
|---|---:|---:|---:|
| `dispatchThreadgroups` indiretto | 20/20 (stesso encoder), 14–20/20 (successivo) | 0/20 | 0/20 |
| `dispatchThreads` indiretto | 20/20, 14–20/20 | 0/20 | 0/20 |

- Gli argomenti e i comandi dell'ICB sono letti **allo stadio Vertex**:
  Fragment e Object|Mesh sono accettati dall'API ma non sincronizzano
  (come Tile in F2.3). Anche senza produttore lento il draw indiretto
  senza barriera sbaglia (2/20): gli argomenti sono letti molto presto.
- **Shader validation e range indiretto**: ogni
  `executeCommandsInBuffer(icb, indirectRangeBuffer)` fa abortire il layer
  (`NSInvalidArgumentException` in `MTL4GPUDebugCommandQueue
  _decodeReportLogState`, uscita 134), anche con range e ICB scritti dalla
  CPU; con il solo API layer funziona.

**Decisione D5**: nel grafo un pass raster che consuma argomenti indiretti
o un ICB dichiara `Usage::IndirectArgs` allo stadio **Vertex**; un
dispatch indiretto allo stadio Dispatch. Nessun range indiretto (D3).

### 6. Gerarchia di transform (S6)

1.048.576 nodi in ordine di livello, profondità D, locali TRS costruite con
glm come `TransformComponent::updateMatrix` (30% scale non uniformi, 5%
specchiate).

| D | GPU per livello | GPU risalita alla radice | CPU 1 thread | CPU 12 thread |
|---:|---:|---:|---:|---:|
| 1 | 0,246 ms | 0,247 | 2,36 | 0,52 |
| 4 | 0,270 | 0,307 | 6,78 | 1,04 |
| 8 | 0,255 | 0,341 | 5,07 | 1,10 |

Propagazione dei soli sporchi (code GPU per livello con append aggregato
per SIMD-group, kernel "args" a un thread, `dispatchThreadgroups`
indiretto, tutti i livelli codificati in anticipo senza readback),
rapporto col ricalcolo completo:

| Radici sporche | D=1 | D=4 | D=8 |
|---:|---:|---:|---:|
| 0,1% | 0,008 ms (0,03×) | 0,031 (0,11×) | 0,064 (0,25×) |
| 1% | 0,025 (0,10×) | 0,047 (0,18×) | 0,077 (0,30×) |
| 10% | 0,114 (0,46×) | 0,276 (1,02×) | 0,339 (1,33×) |

- **Bit-esattezza**: il prodotto `mat4` di glm su questa build **non** è
  contratto in FMA (200.000/200.000 identici al prodotto non contratto). In
  MSL il prodotto scritto nello stesso ordine viene contratto dal
  compilatore anche con `MathModeSafe` (69,9% di float identici a D=4,
  60,2% a D=8); con **`#pragma METAL fp contract(off)`** nella funzione è
  **identico bit per bit a glm** a D = 1, 4, 8, con la libreria fast-math e
  con quella safe (0 ulp). La risalita alla radice è bit-identica ai
  livelli con la stessa associazione.
- Controllo integrato: un figlio non accodato → discrepanza rilevata.

**Decisione D6**: matrici mondo dei figli calcolate sulla GPU per livelli
(prodotto con `fp contract(off)`: identico alla CPU, nessun pixel cambia),
radici scritte dalla CPU con i delta; code di nodi sporchi + dispatch
indiretti quando le radici sporche sono poche (≤ ~5%), ricalcolo completo
per livelli altrimenti (bench 8: tutto in moto).

### 7b. Culling sulla seconda coda accanto al raster (S7, residuo F2.6)

Raster di ~3 ms (3200×1800, ALU nel fragment) sulla coda principale, cull
stabile di 1M (3 dispatch) sulla seconda coda, span da timestamp GPU di
entrambe le code: raster da solo 2,995 ms, cull da solo 0,228, **in serie
3,221, concorrenti 3,059** (−0,16 ms: 70% del cull nascosto), stessa coda
senza barriera 3,082. Risultato del cull e pixel esatti anche in
concorrenza. Il controllo (concorrente tra max(parti) e serie) è temporale:
15/15 run passano a macchina quieta (`--force-family apple9`), 4/8
falliscono con il motore che gira in parallelo (span distorti; esattezza
intatta): spiega il fallimento intermittente visto da un agente su
macchina condivisa.

**Decisione D7**: F2.6 dà ~0,16 ms nel caso favorevole ma quasi lo stesso
si ottiene sulla coda principale senza barriera tra pass indipendenti: il
percorso GPU-driven resta sulla coda grafica (default), la seconda coda
resta per `--debug-async-compute`. F2.5: utile solo al percorso CPU con
molti batch (§7a).

---

## F5 — Scoperte dell'integrazione nel motore

**2026-10-01** · M5 Max, macOS 27.2, Release salvo dove indicato. Ogni
scoperta è stata riprodotta e spiegata prima di decidere.

1. **Il flag ICB della pipeline cambia la codegen.**
   `setSupportIndirectCommandBuffers` sulle pipeline del forward cambia 9
   pixel di ±1 LSB nel bench 1 (Torus) rispetto a `main`; la stessa build
   senza il flag dà 0 pixel (prova). Off e on usano la stessa pipeline
   (immagini identiche tra i due modi); i riferimenti di `visual_check`
   sono stati rigenerati dalla build F5 in modo off (`build/reference-f5`),
   bench 2–7 identici a `main`. Il flag non costa tempo misurabile
   (Forward Stress Test 2,85/2,83 contro 2,88/2,89 ms senza flag).
2. **`executeCommandsInBuffer` in un encoder render ripreso** in un altro
   command buffer (F2.5) manda la GPU in errore e recovery a ogni frame
   ("Discarded (victim of GPU error/recovery)", anche senza validazione;
   0 errori con gli stessi pass e nessuna esecuzione di ICB): con
   `--gpu-driven on` il forward non viene diviso in chunk (codifica ~7
   comandi).
3. **Nessuna barriera ordina i pezzi di un render pass ripreso** in altri
   command buffer dopo il lavoro precedente dello stesso commit: con i pass
   compute della scena prima del forward, `--debug-split-encoding` in modo
   off dava immagini intermittenti (bench 6: 8.862 pixel, la cattura è
   identica al frame **precedente**, quindi letture prima della scrittura;
   bench 8: 1,8–2,2 M pixel). Inefficaci: le barriere di coda del gruppo
   sull'head e su ogni pezzo, una barriera produttore `barrierAfterStages`
   alla fine dell'encoder compute, la profondità non memoryless; esatto con
   1 solo chunk. Soluzione nell'executor: un pass diviso apre un nuovo
   commit che aspetta un evento segnalato dopo tutto ciò che lo precede, e
   il primo commit del frame successivo aspetta l'ultimo di quello prima
   (il rischio inverso). Matrice `visual_check` off/on × flag F2: 0 pixel,
   0 messaggi. Era un bug latente di F2.5: prima di F5 nessun pass compute
   precedeva il forward.
4. **Validazione e pixel**: catture fatte senza layer contro riferimenti
   fatti con i layer differiscono di 9 pixel nel bench 6 anche su `main`:
   si confronta sempre nello stesso modo (lo fa `visual_check`).
5. **Coda 0 statica**: le radici in moto con figli (~70K nel bench 8)
   riscritte dalla CPU a ogni frame costavano 264 KiB/frame e 1 ms di sync
   senza alcun cambio CPU; ora sono una coda GPU persistente (ricaricata
   solo se cambia) espansa da un dispatch dedicato: 1,4 KiB/frame, sync
   0,003 ms.
6. **Swap-and-pop e churn**: spostare l'ultima istanza del bucket in ogni
   buco cambiava la struttura in 120 frame su 300 con `--churn 1000`
   (lista del moto e CSR ricaricate, ~5 MB/frame, comandi CPU 92–97):
   le rimozioni lasciano buchi riutilizzati dalle aggiunte e l'ECS ricicla
   gli `EntityID`: 0 cambi di struttura, comandi costanti, byte
   proporzionali.
7. **Heap CPU**: con i timestamp GPU attivi lo storage delle misure per pass
   (allocato dopo il primo campione dell'heap) cresce di ~30 B/frame (anche
   su `main`, ~9 B/frame con 1 unità); con `--no-gpu-timing` l'heap del
   bench 8 con churn è piatto (+13,5 KB a 600 frame, +11,2 KB a 6000).
8. **Trappola DVFS nelle misure seriali**: con `--gpu-timing-serial` il
   Forward dello Stress Test sembrava +0,9 ms rispetto a `main` (2,8 contro
   1,83 ms) sia in off sia in on; aggiungendo a F5 2 ms di lavoro CPU per
   frame (quanto ne spendeva `main` per l'estrazione) scende a 1,77 ms. La
   GPU riduce i clock quando la CPU lascia pause più brevi tra un frame e
   l'altro: confrontare build con costo CPU diverso richiede di pareggiare
   il tempo CPU (o vsync, dove sono uguali: 4,9–5,3 ms entrambe).
9. **Costo fisso dei pass della scena** (seriale, bench 1): Scene update
   0,013 ms, Scene transforms 0,064 → **0,009 ms** codificando solo i
   livelli fino alla profondità della scena (comandi costanti al variare
   delle istanze, non della profondità), Instance cull 0,026, Draw build
   0,046 → **0,033 ms** senza `resetCommandsInBuffer` (il kernel riscrive
   ogni comando a ogni frame, `reset()` per vuoti e sentinelle).
10. **Sicurezza GPU**: alle 14:31 WindowServer è stato terminato dal
    watchdog (40 s senza risposta) mentre due verifiche GPU dei sotto-agenti
    aspettavano in `waitIdle`: il controllo negativo "senza barriere" di
    F5-K3 usava voci di coda spazzatura come indici (job da ~60 s). Regola
    adottata: cicli dei kernel sempre limitati, nessuna attesa tra
    threadgroup, command buffer ≪ 1 s, mai due processi GPU insieme.

## F6 — Spike: cook dei meshlet, mesh/object shader, Hi-Z, barriere (S1–S4)

**2026-10-02** · Apple M5 Max (Apple10, 40 core GPU), 128 GB, macOS 27.2
(26B5091g), SDK 27.0, toolchain Metal 32023.921, alimentazione AC, build
Release. Piano: [F6](plans/F6.md), pacchetto B. **Stato della macchina:
non quieta.** Dalle 12:17 girava un emulatore Android (`qemu-system-aarch64`,
progetto PitWall) e per tutta la giornata una sessione Codex Computer Use
(`SkyComputerUseService`, `replayd` che registra lo schermo, WindowServer
fino al 66% di CPU): carichi di altri, non fermabili da questa sessione.
Tutte le misure GPU di S1/S2 e le esecuzioni formali di S3/S4 sono dopo le
12:17; dalle ~14:20 il carico è aumentato (bench 8 two-phase da 5,0 a
~10 ms a parità di codice). Per questo: confronti solo **interlacciati**
(A/B/A, ordine ruotato) all'interno della stessa finestra; nessun confronto
tra serie prese in momenti diversi; le scelte sotto il rumore restano sulla
baseline. Strumenti: `bench/f6_spike` (target `f6_spike`, harness
`bench/soc`, ≥15 ripetizioni per metrica, mediana, controllo negativo
integrato; README con il protocollo), `tools/meshlet_cook` (S1 CPU) e il
motore stesso (S1 GPU, S2: `--report` schema 6, tempi per pass validi).

### S1 — cook dei meshlet (64/124 baseline, 64/64/96/128, spatial)

CPU (`meshlet_cook --runs 5`, commit e0bf32a, risultati
`bench/results/f6-spike/s1-cook-m5max-macos27.2.json`; corpus procedurale,
Sponza non disponibile in `assets/`, dichiarato): byte/triangolo minimi
6,07–6,16 con 96/128 e 128/128, baseline 64/124 6,28–6,43, spatial
6,67–8,69, 64/64 6,83–6,93; riempimento della baseline 0,75–0,79 su mesh
lisce (spatial ~0,50); duplicazione vertici 1,31–1,36; cono utilizzabile
97–100% sulle mesh connesse (0% sulla zuppa casuale); spatial 0,26–0,82× il
tempo di cook dello standard. Con 64 vertici, 124 e 128 triangoli danno gli
stessi meshlet (il limite dei vertici prevale).

GPU (motore, `--geometry-path mesh --meshlet-cull two-phase`, 3 repliche
ruotate, codice di cc91f31, GPU pass sum ms, deviazione standard):

| Scena | 64/124 | 64/64 | 64/96 | 96/128 | 128/128 | spatial 64/124 |
|---|---|---|---|---|---|---|
| bench 1 (3200×1800) | 1,210 (0,038) | 1,106 (0,090) | 1,106 (0,037) | 1,230 (0,114) | 1,210 (0,113) | 1,160 (0,041) |
| bench 3 | 1,718 (0,062) | 1,727 (0,085) | 1,692 (0,036) | 1,852 (0,078) | 1,640 (0,097) | 1,824 (0,086) |
| Culling Viz script 1080p | 1,243 (0,044) | 1,230 (0,087) | 1,401 (0,038) | 1,421 (0,106) | 1,372 (0,030) | 1,396 (0,065) |

Fallback `--force-family apple9`, Culling Viz script (interlacciato con il
nativo, macchina più carica, 14:33–14:35): 64/124 1,481 contro 1,457
nativo; 64/64 1,496/1,435; 64/96 1,562/1,437; 128/128 1,485/1,401;
spatial 1,368/1,331 (differenze nativo/Apple9 entro il rumore: il percorso
Apple9 differisce solo per il backend Hi-Z, che con `auto` è comunque
compute).

**Decisione:** nessuna variante vince in modo ripetibile ≥3% su tutto il
corpus (64/64 −9% sul bench 1 ma +0,5% sul bench 3 e il doppio dei
candidati; meshlet più grandi +10% sulla scena di occlusione). Resta la
baseline **standard 64 vertici / 124 triangoli, cone weight 0,5**; le
opzioni restano nella CLI (`--meshlet-builder`, `--meshlet-max-*`).

### S2 — indexed contro mesh contro object+mesh

Motore, 3 serie ×2 (6 repliche, ordine ruotato), `--no-vsync --no-ui
--fixed-timestep`, 600 frame + 120, codice di cc91f31, GPU pass sum ms:

| Scena | indexed F5 | mesh pass-through | frustum + cono | two-phase |
|---|---|---|---|---|
| bench 1 (toro, 3200×1800) | 0,819 | 0,841 | 0,839 | 1,097 |
| bench 3 (stress) | 1,643 | 1,681 | 1,625 | 1,858 |
| bench 7 classico | 1,216 | 1,362 | 1,194 | 1,566 |
| Culling Viz script 1080p | 2,066 | 2,134 | 1,940 | **1,270** |
| bench 8 (1M cubi) | 5,137 | 7,384 | 7,356 | 5,006 |

Variante mesh-only senza object stage (`--meshlet-object off`, 3 repliche
interlacciate con il pass-through object+mesh): uguale su bench 1/7, +7%
su bench 3 e sulla scena scriptata, +10% sul bench 8. **Decisione:**
object+mesh (necessario comunque per il culling); il pass-through non batte
l'indexed (B-16: limite raster), costa +43% con meshlet da 12 triangoli
(bench 8); il guadagno arriva dal culling quando c'è occlusione (scena di
occlusione −39%), mentre il two-phase costa +13…34% su scene senza
occlusione (piramidi ~0,2–0,28 ms a 3200×1800). **Il default del motore
resta `--geometry-path indexed`** (riferimento F5, nessuna regressione); il
percorso mesh two-phase è il preset di Culling Viz e la base di F7.
Scelte per scena: candidato OPT, non attivato.

Rimisura serale (21:23, HEAD 0c097c4, 3 repliche ruotate, stesso script,
`build/f6-s2g`): non utilizzabile per i numeri assoluti. Il compositore
limitava la presentazione senza vsync a ~80–90 fps (attesa del drawable
~11–12 ms, anche con l'indexed), la GPU restava inattiva gran parte del
frame e scalava le frequenze: bench 3 indexed 5,03 ms (dev. std. 0,85)
contro 1,64 al mattino, deviazioni fino a 1,2 ms. L'ordine qualitativo
conferma le decisioni: scena di occlusione two-phase 3,20 contro indexed
4,75 (−33%); two-phase più caro su bench 1 e bench 7 classico; bench 8
two-phase ≈ indexed (5,93 / 5,95); pass-through mesh mai migliore in modo
ripetibile (bench 3 entro la deviazione standard). Restano i valori del
mattino (frequenze sature, interlacciati).

Scoperta: una griglia mesh **1D** di ~700K threadgroup (mesh-only, bench 8)
disegna solo una parte della scena senza alcun messaggio di validazione;
la variante usa una griglia 2D (x ≤ 32768). La griglia object 1D del
percorso principale è stata provata fino a 86 600 threadgroup (4M istanze,
2,77M candidati: tutti disegnati, immagine identica all'indexed).

### S3 — piramide Hi-Z: compute per livello, SIMD-group, sampler min

`f6_spike --only F6-S3 --runs 3`, kernel del motore compilati da sorgente,
risultati `bench/results/f6-spike/m5max-macos27.2-s3.json` (e
`-s3-apple9paths.json` con `--force-family apple9`): i tre backend sono
**bit-exact** con la piramide CPU su 7 dimensioni (1×1, 1×17, 17×1, 63×65,
1920×1080, 1919×1081, 3200×1800) × 5 pattern (casuale, 1% di buchi a zero,
tutto zero, occluder 0,8, gradiente con un buco); il buco sopravvive a ogni
livello; il controllo negativo (riduzione puntuale) produce migliaia di
texel diversi. Catena intera, mediana di 3 run: 1920×1080 compute
0,046 ms, **SIMD 0,031**, sampler 0,042; 3200×1800 0,081 / **0,065** /
0,073. `--validate`: 0 messaggi (dopo aver tolto un `setComputePipelineState`
ridondante dello spike). Livello 0 a potenze di due ≥ ⌈dim/2⌉: ogni
riduzione è esatta 2×2, i bordi NPOT diventano padding neutro (1).
**Decisione:** backend compute **SIMD-group** come default e fallback Apple9
(`auto`), sampler min Apple10 disponibile con `--hiz-path sampler` (più lento
del SIMD su M5, equivalente bit per bit, rifiutato con l'Apple9 effettivo).
Nel frame le due piramidi costano 0,15–0,28 ms (S2), sotto il 3% del frame
del preset: nessun ulteriore lavoro di ottimizzazione giustificato.

### S4 — ordinamenti del percorso mesh

`f6_spike --only F6-S4 --runs 3` (metodo di F5-S5: produttore lento ~6 ms,
l'ultimo threadgroup scrive), risultati
`bench/results/f6-spike/m5max-macos27.2-s4.json`, 20 ripetizioni per
variante, identico nelle 3 run:

| Consumatore ← produttore | none | Vertex | Object | Mesh | Object+Mesh | Vertex+Object+Mesh | Fragment |
|---|---|---|---|---|---|---|---|
| argomenti indiretti di `drawMeshThreadgroups` ← compute | 20/20 errati | **20/20 errati** | 0/20 | 0/20 | 0/20 | 0/20 | – |
| letture dell'object shader ← compute | 20/20 | **20/20** | 0/20 | 0/20 | 0/20 | 0/20 | – |
| compute ← scritture dell'object shader (barriera after) | 20/20 | **20/20** | 0/20 | 0/20 | 0/20 | – | 0/20 |

**Opposto a F5:** per i draw indexed gli argomenti indiretti si ordinano allo
stadio Vertex e Object|Mesh non sincronizzano (F5-S5); per i draw **mesh**
Vertex non ordina nulla e serve Object/Mesh. Il grafo dichiara quindi gli
argomenti e le liste mesh a `StageObject | StageMesh`, le scritture dei flag B
a `StageObject`, mentre il fallback ICB dello stesso pass resta a
`StageVertex`. `--validate`: 0 messaggi. Le dipendenze depth (raster) →
Hi-Z (compute) seguono F2.3 (Fragment → Dispatch); la storia tra frame è un
import persistente del grafo (la scrittura di Hi-Z final del frame n precede
la lettura della fase A del frame n+1).

## F6 — Scoperte dell'integrazione nel motore

- **Bug F5 trovato dalla revisione di conservatività:** il raggio della
  sfera mondo di F5 (`cullWorldSphere`) usava la colonna più lunga, che
  non limita lo stiramento con shear (figlio ruotato sotto un genitore a
  scala non uniforme: colonna più lunga 1,414, stiramento 1,618).
  Sostituito con il limite di Gershgorin sulla matrice di Gram (esatto per
  rotazione × scala), test con il caso di shear; immagini dei bench 1–8
  invariate.
- **Cono in spazio mesh:** il primo test (solo similitudini) non scartava
  quasi nulla sugli edifici a scala non uniforme (1 meshlet). La faccia è
  invariante sotto ogni affine invertibile a meno di sign(det), che la
  classe di cull compensa: portando la camera in spazio mesh (M⁻¹·cam) il
  test è esatto per scala non uniforme, shear e specchio; sulla scena
  scriptata scarta il 57% dei candidati (primitive 1,14M → 0,61M) a
  immagine identica; test di proprietà su 5 tipi di matrice (>1000 scarti
  ciascuno, nessun triangolo visibile scartato).
- **Pareggi di profondità dipendenti dall'ordine:** il percorso mesh
  disegna in ordine (classe, slot, meshlet), l'indexed in (classe, mesh,
  slot). Bench 1–7 identici al pixel; bench 8 differisce di 1 pixel in tutti
  i modi mesh, 0 pixel con una sola mesh (stesso ordine): un pareggio di
  depth tra istanze intersecanti, non una superficie persa. Lo scenario
  scriptato non crea sovrapposizioni (il churn riusa la cella).
- **Encoder vuoti e timestamp:** i pass di readback del self-check, vuoti
  nei frame normali, venivano scartati da Metal e lasciavano il pass
  successivo (Forward) senza timestamp valido (frames 0, tempo attribuito
  altrove): ora entrano nel grafo solo con `--debug-meshlets`.
- **Gate dell'ICB F5 come fallback di overflow:** il `Draw build` scrive le
  draw solo quando i candidati superano la capacità; con overflow forzato
  ogni frame (controllo `count`) l'immagine è identica al riferimento
  indexed, 0 messaggi di validazione. Il self-check F5 ora conosce il gate.
- **Culling per triangolo nel mesh shader** (`--meshlet-triangle-cull on`):
  tiene 1796/4098 triangoli sul bench 1 a immagine identica (controllo
  negativo: segno invertito → 1,56M pixel diversi); A/B interlacciato
  inconclusivo con il carico esterno (−2% / +3%): resta spento di default.
  Un confronto iniziale tra serie prese in momenti diversi sembrava dare
  +18% sul bench 8: era il carico della macchina, non il codice.
- **Hitch check:** segnalazioni intermittenti allo stesso tasso su indexed e
  mesh (2/6 esecuzioni ciascuno; in serata 1/6 indexed, 0/6 mesh,
  interlacciati: un frame da 5,2 ms dopo uno switch), singoli frame CPU da ~0,6 ms in bench con
  soglia p99 ~0,1 + 0,5 ms; preesistente (nota F5). Le liste meshlet ora si
  ridimensionano allo switch (nessuna allocazione o rilascio differito nei
  frame successivi).
- **Uscita non nulla intermittente** (1 su 164 esecuzioni dello script S2,
  report completo, nessun crash report, nessun percorso di uscita del motore
  applicabile): non riprodotta in 104 esecuzioni mirate; aggiunta la riga
  `EXIT <code>` stampata da `main` per distinguere una futura occorrenza da
  un segnale. Rimisura serale: 88 esecuzioni (60 S2, 16 gate, 12
  `hitch_check`), tutte `EXIT 0` con stato della shell 0. In totale 1 su
  ~356 esecuzioni dal primo caso. **Aperta, non spiegata.**
- **O7:** 0 allocazioni GPU nei frame misurati (anche con churn); blocchi
  dell'heap CPU piatti; i byte crescono solo con i timestamp per pass
  (storage delle misure, comportamento F5 noto: ~48 B/frame con i pass in
  più del percorso mesh); con `--no-gpu-timing` −1,7 KB a 600 frame e
  −2,2 KB a 6000. `leaks --atExit` 0 con self-check e churn.


## F7.3/F7.4/F7.5 — esperimenti sul frame F8, senza attivare OPT

**2026-10-03**, richiesta esplicita del proprietario. Codice `3231364`,
runner `0fc5fa3`; condizioni e 39 prove nel [perf-log](perf-log.md).
Ipotesi: ridurre traffico o shading con tile/bins/2×2. Controllo: stesso
material model, scena, camera, formato e ricostruzione; costo di classificazione,
storia e fallback incluso. Soglia di adozione: beneficio completo ripetibile
(riferimento ≥3%) o un vincolo di memoria risolto, a qualità verificata.

- **F7.3:** binning esatto contro generic, ma +4,34% sul frame F7 Sponza;
  con F8 il risultato non è stabile e Many Lights non mostra un guadagno
  ripetibile. **Default generic**, binning disponibile con flag esplicito.
- **F7.4:** imageblock ID e tile kernel con stessi materiali, frustum in
  entrambe le varianti. 2,1569→2,1535 ms (−0,16%, intervalli sovrapposti),
  transient/device allocation −8.486.912 byte (8,09 MiB). Immagini esatte
  contro compute, clip temporali passano. **Rimanda**: il frame non guadagna
  abbastanza e non c'è un vincolo di memoria risolto. Non è un G-buffer
  tradizionale; HDR/guide restano in texture. Two-phase può interrompere la
  fusione. T0 e contatori di banda fisica non disponibili.
- **F7.5:** identità e luminanza precedenti, stesso triangolo 2×2, roughness
  alta, materiali non sensibili e moto <0,25 px; nessun atteso fra threadgroup.
  Cornell riusa 279867/492102 shading (56,87%) nell'ultimo frame ma costa
  2,0829→2,0890 ms (+0,29%). Sponza 2,1612→2,2097 ms (+2,24%), zero
  pixel eleggibili nell'ultimo frame. Storia aggiuntiva 33.177.600 byte
  (31,64 MiB) per vista al backing 1080p, anche quando il pool già riservato
  assorbe l'allocazione senza cambiare il totale del device. **Non adotta**
  nel preset corrente. Disabilitare la storia elimina il riuso; clip e
  poison dei dati temporali verificano il controllo.

Le tecniche restano prototipi opt-in e candidate per carichi futuri; i
risultati non autorizzano scheduler/optimizer generali. Le stime del grafo
sono somme di accessi dichiarati, non byte DRAM misurati né conteggi delle
operazioni interne MetalFX. [Consegna completa](F7_F8_HANDOFF.md).

## F9 — Spike: BLAS, TLAS dalla GPU scene, traversal, proxy, alpha RT (S0–S5)

**2026-10-05** · Apple M5 Max (Apple10, 40 core GPU), 128 GB, macOS 27.2
(26B5091g), SDK 27.0, alimentazione AC, build Release, commit `12e3cea`
(albero pulito). Piano: [F9](plans/F9.md). **Macchina quieta** (nessun
agente o benchmark in corso; restano WindowServer e lo sfondo animato di
sistema), `caffeinate -d`, 3 run, ≥15 ripetizioni per metrica, mediana;
CV fra le run ≤2% per tutte le metriche citate salvo dove indicato.
Strumento: `bench/f9_spike` (target `f9_spike`, harness `bench/soc`,
README con il protocollo), risultati
`bench/results/f9-spike/m5max-macos27.2.json` e `…-s5-apple9.json`.
Riferimento CPU comune: test raggio/triangolo **watertight** (Woop et al.)
in doppia precisione con un BVH esatto; ogni hit GPU è confrontato (t entro
2e-4 relativo, il triangolo nominato deve contenere il punto entro 1e-5
baricentrico). Il primo riferimento con Möller-Trumbore perdeva raggi
esattamente sugli spigoli condivisi che la GPU (watertight) colpisce: per
questo il riferimento è watertight. Corpus: procedurali (5 mesh) e Sponza
(103 mesh, 262.267 triangoli, una BLAS per mesh che legge in place il
layout `GPUVertex`/indici del motore). Tutti gli spike passano anche con
`--validate` (API + shader validation, 0 messaggi) e S5 con
`--force-family apple9`. Ogni controllo negativo è stato rotto
temporaneamente e visto fallire.

### S1 — ciclo di vita delle BLAS (F9-S1, S1b, S1c, S1d, S1e)

- **Allocazione.** `heapAccelerationStructureSizeAndAlign` = dimensione
  dell'AS arrotondata, allineamento **1 KiB** per tutte le 108 mesh. Sponza:
  24.616.576 B di AS; allocazioni del device +25.198.592 B standalone,
  +25.411.584 con un heap per AS, **+24.674.304 impacchettate in un heap di
  piazzamento** (offset allineati a 1 KiB): tutte tracciano con 0 raggi
  errati. Le sotto-allocazioni non entrano nel residency set: basta l'heap.
- **Build delle 103 BLAS.** Un encoder con scratch condiviso e barriera
  AS→AS fra le build: **22,8 ms**; scratch disgiunti per build senza
  barriere: **2,60 ms** (8,8×; scratch 7,42 MB contro 1,26 MB). Nessun
  vincolo di allineamento dello scratch trovato (offset a 1 B validi). Flag
  d'uso: PreferFastBuild −1,6% byte, Refit +0,08%; `refitScratchBufferSize`
  = 0 sempre.
- **Compaction asincrona** (dimensione scritta in un command buffer, copia
  in uno successivo): Sponza **0,552** (13,59 MB), per mesh 0,495–0,946;
  copia di tutte le 103 BLAS 0,56 ms; hit bit-identici prima e dopo.
  `writeCompactedAccelerationStructureSize` scrive **8 byte**.
- **Refit su deformazione** (kernel che scrive le posizioni nel buffer
  `GPUVertex`, barriera Dispatch→AS, refit in place o fuori posto, AS→Dispatch,
  traccia): 0 raggi errati. Refit 3,4–5,2× più veloce del rebuild (mesh
  Sponza 27.796 triangoli: 0,13 contro 0,53 ms). Dopo una deformazione
  grande il traversal dell'AS rifittata costa **1,33×** quello della
  ricostruita (1,61 contro 1,21 ms per 8,4 M raggi): il refit degrada.
- **Ordinamenti** (produttore lento: piano 1M triangoli, due stati, 20
  ripetizioni): build/refit → traccia è ordinato solo da
  `barrierAfterEncoderStages(AccelerationStructure, Dispatch)` o da una
  barriera di coda AS→Dispatch sull'encoder successivo (0/20 fallimenti);
  senza barriera, con Dispatch→Dispatch o con due encoder senza barriera
  **20/20 ripetizioni hanno tutti i 262.144 raggi stantii**. Scrittura dei
  vertici → build: nessuna variante senza barriera ha mai fallito (anche
  con produttore rallentato): **inconcludente**, la barriera Dispatch→AS
  resta per contratto API.
- **Crash del layer di shader validation (S1e).** Rilasciare un heap di
  piazzamento (con AS **o con semplici buffer**) e poi riusare lo stesso
  oggetto command buffer fa crashare il commit successivo in MetalTools
  (`HeapUsageTable::processHeapEntry` da `MTL4GPUDebugCommandBuffer
  preCommit`); con command buffer nuovi 0 crash su 8 configurazioni, senza
  validation nessun crash. È il difetto già aggirato dal motore in
  `MetalContext::refreshCommandBuffer` (misurato su 27.1), ancora presente
  su 27.2: le AS del motore devono uscire dalla residency attraverso
  `MetalContext::evict` come gli altri heap.

### S2 — TLAS scritta in compute dalla GPU scene (F9-S2)

`GPUInstance` (80 B) → descrittori indiretti (**72 B** impacchettati) da un
kernel; 4 BLAS procedurali, 1K/10K/100K istanze animate, ~10% specchiate,
5% di slot non validi. Strategie: **A** un descrittore per slot (slot non
valido: mask 0), **B** compattazione atomica con conteggio GPU e
`IndirectInstanceAccelerationStructureDescriptor` (nessuna lettura CPU),
**Bs** compattazione stabile. 0 raggi errati dopo build, 1 e 5 frame di
refit, delete/reuse di slot riciclati (nuova mesh e generazione lette via
`user_instance_id`).

| 100K istanze (ms) | A | B | Bs |
|---|---:|---:|---:|
| scrittura descrittori | 0,013 | 0,014 | 0,021 |
| build usage None | 1,723 | 1,644 | 1,677 |
| build PreferFastBuild | 1,135 | 1,061 | 1,087 |
| refit | 0,139 | 0,137 | 0,135 |
| **descrittori + barriera + refit** | **0,151** | 0,149 | 0,156 |

1K/10K (A): build 0,41/0,52 ms, refit 0,064/0,067 ms. TLAS 100K 22,8 MB.
**Il target 0,5 ms per 100K istanze dinamiche è raggiunto solo dal refit**
(0,15 ms); nessun rebuild ci sta (≥1,06 ms). Il refit resta corretto anche
dopo delete/reuse (A) e con l'ordine dei descrittori che cambia (B).
**Specchiate:** `triangle_front_facing` ignora il segno del determinante;
con l'opzione `TriangleFrontFacingWindingCounterClockwise` impostata
**solo sulle istanze `INSTANCE_FLAG_MIRRORED`** 0 discordanze su entrambe le
classi (3460 normali, 436 specchiate); controlli negativi: 4x3 trasposta
1980/2048 errati, slot cancellati con mask 0xFF 440 hit su istanze morte,
regola specchiata ignorata 436/436 discordanze.

### S3 — traversal e costo per raggio (F9-S3)

Sponza, 1920×1080 (2.073.600 raggi primari), raggi secondari dai punti
primari (il loro tempo esclude i primari); costo ammortizzato in ns per
raggio (GPU intera):

| ns/raggio | primario | ombra | AO (0,5 m) | diffuso |
|---|---:|---:|---:|---:|
| `intersector` closest | 0,239 | 0,282 | 0,180 | 0,293 |
| `intersector` any-hit | — | 0,259 | 0,174 | — |
| `intersection_query` closest | 0,321 | 0,377 | 0,308 | 0,434 |
| `intersection_query` any | — | 0,382 | 0,316 | — |

`intersection_query` costa 1,34–1,71×; any-hit −8% sulle ombre; senza
intersection function `assume_geometry_type`, istanze non opache e
`force_opacity` non cambiano nulla. Un passo d'ombra a 1080p ≈ 0,54 ms.
0 discordanze fra varianti (≈4 M raggi) e 0 errori contro la CPU.
**Auto-intersezione:** origine sul punto 1.970.672 auto-hit e 8350 acne;
tmin 1e-4 1747 auto-hit; tmin 1e-3 274 auto-hit e 50 fughe di luce;
offset Wächter-Binder in spazio oggetto 0 auto-hit d'ombra ma auto-hit
diffusi su un'istanza scalata 0,008; **offset Wächter-Binder in spazio
mondo: 0 auto-hit, accordo 99,989% con l'ombra esatta, 0 acne**, 22 fughe
su raggi radenti (n·l 0,005–0,25).

### S4 — geometria proxy (F9-S4)

Proxy solo-indici (meshoptimizer) sopra il buffer di vertici originale:
UV/materiale condivisi per costruzione, BLAS proxy = stessi vertici, nuovi
indici. Errore misurato contro la geometria piena (3 camere 960×540, 1,55 M
primari, 705K ricevitori d'ombra):

| Livello | triangoli | primari errati | dt95 | ombra discorde | acne |
|---|---:|---:|---:|---:|---:|
| ratio 0,5 senza bordi | 51% | 4,8% | 4 cm | 25,4% | 1,65% |
| ratio 0,5 bordi bloccati | 51,7% | 4,1% | 4,2 cm | 0,57% | 0,98% |
| errore 1e-3 | 48,4% | 0,52% | 1,1 cm | 0,89% | 1,31% |
| sloppy 0,01 (controllo) | 3,5% | 62,4% | 5,1 m | 25,4% | 3,5% |
| **politica adattiva, bordi bloccati** | **55,0%** | **0,10%** | **0,07 cm** | **0,05%** | **0,15%** |

Un rapporto fisso non è sicuro: una sola mesh di 32 triangoli (estensione
minima 3,25 m) collassa e produce il 25% di ombre false. La politica che
parte grossolana (ratio 0,1 con bordi bloccati) e promuove solo le mesh
colpevoli finché ombra ≤0,5%, primari ≤0,2%, dt95 ≤1 cm, acne ≤0,5% converge
in 7–14 iterazioni e dimezza circa triangoli e byte delle BLAS (Sponza non
ha mesh emissive). Controlli: pieno contro pieno esattamente 0, sloppy 0,01
oltre soglia, una mesh sabotata è individuata.

### S5 — alpha test con intersection function (F9-S5)

3 materiali MASK di Sponza (34.940 triangoli) e una scena sintetica con
buchi noti. **A**: una funzione generica all'offset 0, geometria mascherata
non opaca, il resto opaco (portabile); **B**: slot per materiale e funzioni
specializzate (indicizzazione hardware M5); **C**: tutto opaco (controllo).
A, A con primitive data, B: **0 raggi errati** su 777.600 (Sponza) e
300.000 (sintetica) contro la regola del raster in doppia precisione; C
discorda sul 41,6% della sintetica. Nessuna chiamata a funzione su
geometria opaca (nessun occlusore opaco può lasciare passare luce).

| ns/raggio Sponza | A | B | C (opaco) |
|---|---:|---:|---:|
| primario | 0,287 | 0,289 | 0,285 |
| ombra | 0,322 | 0,318 | 0,316 |

Con il 5,6% di raggi primari che invocano la funzione il costo è entro
l'1%; B **non** guadagna su A sul M5. Apple9 (`--force-family apple9`,
solo A): 0,289/0,318 ns, 0 errori. LOD della texture (nessuna derivata in
RT): sui raggi che toccano superfici mascherate LOD 0 contro un LOD "da
raster" (derivate dei pixel vicini, aniso 8) cambia esito nel 16,6%, il cono
di raggio nel 12,9%; per le ombre LOD 0 e cono differiscono solo nello
0,02% dei raggi.

### Decisioni per l'integrazione (F9.1–F9.5)

1. **BLAS** in heap di piazzamento attraverso `GpuMemory` (allineamento
   1 KiB, una voce di residency per heap, rilascio con evict ⇒ ricostruzione
   dei command buffer); build batch con scratch disgiunti e senza barriere
   fra build indipendenti; compaction asincrona (dimensione 8 B letta al
   completamento, copia in un frame successivo, vecchia AS rilasciata dopo
   l'ultimo lettore); refit per deformazioni con rebuild quando la topologia
   cambia o il refit degrada.
2. **TLAS per frame: strategia A** (un descrittore per slot, mask 0 per gli
   slot non validi, `userID` = slot, opzione CCW sulle specchiate), scritta
   da un kernel dopo `Scene transforms`, **refit ogni frame**, rebuild
   quando cambia la capacità o l'insieme di BLAS e a cadenza configurabile
   contro il degrado. B/Bs restano disponibili (stesse prestazioni) ma A
   conserva l'identità slot = `instance_id` senza compattazione.
3. **Barriere**: AS→Dispatch fra build/refit e traccia (obbligatoria),
   Dispatch→AS fra scrittura dei descrittori/vertici e build: stadi esatti
   dichiarati nel grafo.
4. **Traversal**: solo `intersector`; any-hit per le ombre; origine dei
   raggi secondari con offset Wächter-Binder in spazio mondo.
5. **Proxy**: semplificazione solo-indici con politica adattiva e errore
   dichiarato per mesh; senza misura la mesh resta piena.
6. **Alpha RT**: strategia A (una funzione generica, geometria opaca per i
   materiali opachi) su Apple9 e Apple10: B non guadagna sul M5; LOD 0 per le
   ombre, LOD da cono per i raggi primari.
7. **F9.6 (ray binning)**: nessuno spike ne misura un guadagno (raggi
   coerenti 0,24–0,28 ns contro diffusi 0,29 ns): non attivato.
