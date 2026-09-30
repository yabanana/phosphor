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
