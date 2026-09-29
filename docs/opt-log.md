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
