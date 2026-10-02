# Phosphor — Roadmap implementativa 2026–2027 (e oltre)

Piano operativo per portare Phosphor dalle fondamenta Metal 4 (F0, completata)
a un **engine AAA nativo per Apple Silicon, bleeding edge e ottimizzato fino
all'ultimo byte di banda**, con editor, runtime/ECS, contenuti della comunità,
2D e UI di gioco completi. Il renderer attivo è Metal sulle piattaforme Apple.

Riferimenti:
- motivazioni tecniche → [`reports/Engine AAA nativo per Apple Silicon.md`](../reports/Engine%20AAA%20nativo%20per%20Apple%20Silicon.md) e `research_notes/`
- come spremere ogni componente del SoC → [`APPLE_SOC_PLAYBOOK.md`](APPLE_SOC_PLAYBOOK.md)
- bibliografia delle fasi OPT (`[Rn]`) → [`RESEARCH_REFERENCES.md`](RESEARCH_REFERENCES.md)
- priorità e maturità dei piani → [`plans/SEQUENCING.md`](plans/SEQUENCING.md)
- piattaforma completa e riuso di Bevy → [`plans/PRODUCT_PLATFORM.md`](plans/PRODUCT_PLATFORM.md)
- piani dettagliati anticipati per le fasi F e OPT → [`plans/README.md`](plans/README.md)
- ricerca CPU/GPU/ANE, memoria e kernel (2026-10-01) → [`research/2026-10-01-apple-soc.md`](research/2026-10-01-apple-soc.md)
- regole per le sessioni AI → [`CLAUDE.md`](../CLAUDE.md)

---

## Filosofia

1. **Nativo, non tradotto.** Ogni sistema nasce attorno a TBDR, memoria
   unificata, Metal 4 e acceleratori neurali. Nessuna astrazione "da PC".
2. **Prima il rendering, poi il collo di bottiglia.** Baseline corretta e
   semplice, misura sul motore, esperimento mirato e decisione. Il profiler
   arriva in F4; una nuova infrastruttura deve giustificare la sua complessità.
3. **Meno raggi, più ricostruzione.** Budget di ray tracing bassi, compensati
   da cache di radianza, riuso temporale e ricostruzione ML: è la direzione
   dell'hardware Apple.
4. **Tutto guidato dalla GPU.** La CPU descrive il mondo; la GPU decide cosa
   disegnare, a che LOD e con che qualità.
5. **Pavimento progettato, hardware dichiarato.** Apple9 (M3) resta il minimo
   previsto; sviluppo e accettazione sul M5 Max disponibile, fallback verificati
   qui e certificazione T0 fisica separata quando il dispositivo sarà disponibile.
6. **Spike di ricerca espliciti.** Le idee non provate diventano un prototipo
   a tempo con criterio "adotta / rimanda / scarta", mai un ramo infinito.
7. **Spremere il SoC dove serve al frame.** Le OPT raccolgono opportunità
   algoritmiche per ALU, registri, tile, memoria, RT, CPU, ANE e I/O.
   Dopo ogni era si scelgono gli interventi sostenuti dalle misure; il catalogo
   non è una lista da implementare interamente.

---

## Come usare questo documento

- Fasi in **ordine di dipendenza**; le fasi marcate *parallela* possono
  procedere accanto alla linea principale perché toccano sottosistemi separati.
- Etichette di priorità:
  - **[CORE]** indispensabile per avere un engine funzionante
  - **[AAA]** necessaria per qualità da gioco AAA
  - **[EDGE]** opportunità di ricerca, attivata per evidenza; con fallback se adottata
  - **[OPT]** catalogo di ottimizzazioni su carichi reali, da cui scegliere un ambito limitato
  - **[CANDIDATO]** task non attivo per default: serve un problema misurato prima dello spike; non blocca il percorso principale
  - **[BASELINE]** raccolta di evidenza sul frame corrente; usare gli strumenti esistenti prima di costruirne altri
- Ogni task ha un ID (`F7.3`) da citare in commit e PR; la casella si spunta
  solo quando il task è **verificato sul dispositivo disponibile per l’ambito
  dichiarato**, non solo scritto. Le misure multi-device esplicite restano
  parziali finché eseguite: [politica hardware](plans/HARDWARE_VALIDATION.md).
- **Definizione di "ambito operativo chiuso"**: task funzionali e candidati
  selezionati verificati, criteri di uscita misurati e annotati in
  `docs/perf-log.md`, `ctest` e CI verdi, zero errori di validazione Metal sui
  testbench, test visivi aggiornati, README aggiornato. Riportare gli ID
  consegnati e quelli rimasti candidati: questi ultimi restano non spuntati.
  La selezione non rende facoltative correttezza e verifiche della funzione adottata.
- I budget in millisecondi sono **ipotesi di progetto**, da ricalibrare con le
  misure reali alla fine di F8.

**Requisiti di prodotto (decisione del proprietario, 2026-10-01):** editor,
ECS/runtime, contenuti, 2D, UI ed ecosistema sono capacità da consegnare, con
Bevy come riferimento funzionale e prima scelta di riuso dove praticabile.
La regola del collo di bottiglia riguarda nuova complessità di ottimizzazione:
non rende facoltative queste funzionalità. F39–F41 estendono la numerazione
senza rinumerare i task esistenti; l'ordine dipende dai contratti, non dal numero.
Compatibilità per versione, adapter e gap devono essere verificati: un ECS
ispirato a Bevy non abilita automaticamente i plugin Rust o quelli legati a wgpu.

### Tier hardware

| Tier | Chip | Ruolo |
|---|---|---|
| **T0 Base** | M3/M4/M5 base, A18 Pro, iPad M3+ | Target di prodotto; certificazione fisica pendente, non blocca lo sviluppo sul M5 Max |
| **T1 Pro** | M3/M4/M5 Pro | Qualità alta |
| **T2 Max** | M3/M4/M5 Max | Qualità ultra; macchina di sviluppo (M5 Max 128 GB) |
| **T3 Neurale** | M5 Pro/Max/Ultra (Apple10) | Funzioni neurali, path tracing |

**Politica hardware, decisione 2026-10-02:** abbiamo soltanto M5 Max 128 GB.
`DEVELOPMENT_ACCEPTED` su questa macchina permette di integrare e proseguire;
`HARDWARE_CERTIFIED` si dichiara solo per configurazioni fisiche provate.
Apple9 forzato esercita fallback sul M5, non emula un M3. I gate T0 e le
misure multi-device ancora mancanti restano nel
[registro delle verifiche esterne](plans/HARDWARE_VALIDATION.md), senza bloccare
le fasi successive. I target di qualità e le prove di correttezza locali restano
obbligatori. Questa regola vale per tutte le uscite F/OPT che citano altri tier.

### Budget frame di riferimento (ipotesi, 60 fps)

| Pass | T0 1080p (upscaled) | T2 1440p→4K |
|---|---|---|
| GPU scene + culling + visibility buffer | 2,0 ms | 1,5 ms |
| Material resolve | 2,0 ms | 1,5 ms |
| Ombre | 1,5 ms | 1,5 ms |
| Luci dirette (ReSTIR DI) | 1,0 ms | 2,0 ms |
| GI + riflessioni | 2,0 ms | 3,5 ms |
| Atmosfera, volumetrici, acqua | 1,0 ms | 1,5 ms |
| Trasparenze, particelle | 1,0 ms | 1,0 ms |
| Post + MetalFX | 2,0 ms | 2,0 ms |
| Margine | 0,5–3,5 ms | 1,5 ms |
| **Totale** | **≤ 16,6 ms** | **≤ 16,6 ms** |

---

## Binario trasversale: ultra-ottimizzazione (regole O1–O12)

Non è una fase: sono vincoli di progettazione e verifica sul lavoro introdotto.
Applicarli ai percorsi effettivamente usati, senza costruire in anticipo una
infrastruttura generale per ogni voce; resta vincolante la correttezza.

| Regola | Contenuto | Come si verifica |
|---|---|---|
| **O1 Byte per pixel** | Ogni pass dichiara quanta banda DRAM legge/scrive; il totale per frame sta nel budget del tier (≈ 10 GB/frame a 60 fps su M5 Max, ¼ su T0) | Contatori di banda in Xcode, colonna "bytes" nel profiler |
| **O2 Tile memory first** | Intermedi `memoryless`, store `.dontCare`, pass fusi; fragment barrier vietate | Nessun load/store inutile nel report del render graph |
| **O3 Half ovunque possibile** | `half` per colore, normali, BRDF; `float` solo dove la precisione lo richiede (posizioni, profondità) | Revisione shader + test visivi invariati |
| **O4 Registri sotto controllo** | Controllare registri vivi e private memory degli shader caldi | Causa degli stalli e tempo del pass/frame; nessuna soglia universale di occupancy |
| **O5 SIMD-group** | Riduzioni, prefix sum e compattazioni con `simd_*`/`quad_*`, non con atomici in threadgroup memory | Microbenchmark in `bench/` |
| **O6 Compressione GPU** | Texture popolate via blit (mai `replaceRegion`), formati compressibili, private storage | Contatori di compressione (Xcode 26.4+) |
| **O7 Zero allocazioni per frame** | CPU: arena per frame; GPU: anelli e heap; nessun `new` nel ciclo caldo | Instruments Allocations piatto durante il gameplay |
| **O8 Zero draw call dalla CPU** | Dopo F5 la CPU non codifica draw per oggetto: ICB e dispatch indiretti costruiti dalla GPU | Conteggio comandi CPU per frame costante al variare della scena |
| **O9 Stutter zero** | Nessuna compilazione, allocazione di heap o I/O bloccante nel frame | Frame pacing a 2 bucket nel Performance HUD |
| **O10 Energia** | Modalità "efficienza" per batteria: fps cap, tier ridotto, QoS adeguata ai job | Energia/potenza misurata; nessun placement fisso dei core presunto |
| **O11 Specializzazione** | Function constant per eliminare rami statici; varianti generate, non `if` a runtime | Numero di varianti e tempo di compilazione tracciati |
| **O12 Ambito hardware verificabile** | Gate sviluppo sul M5 Max e fallback pertinenti; target T0 conservato, certificazione fisica separata e non bloccante | Chip fisico/percorso/preset nel report; T0 `EXTERNAL_VALIDATION_PENDING` finché manca, mai dedotto dal Max |

---

## Fasi OPT: metodo comune

Dopo ogni era si confronta il frame reale con i budget. Le **OPT** sono
cataloghi di interventi: si attivano soltanto i task pertinenti a un limite
osservato. Se il budget è adeguato, si prosegue col rendering successivo.
Il [playbook](APPLE_SOC_PLAYBOOK.md) e i paper aiutano a scegliere esperimenti;
non impongono di implementare ogni tecnica o tenere occupata ogni unità.

**Protocollo di ogni fase OPT**

1. **Baseline reale**: scena e clip del motore con manifest, tempi e qualità; strumenti F4 già disponibili, misure T0/T2 e limiti dichiarati.
2. **Diagnosi**: scegliere un collo di bottiglia prioritario e stimare il beneficio sul frame; provare prima la correzione locale o l'euristica semplice.
3. **Selezione**: solo se serve, scegliere un task `[CANDIDATO]`, consultare fonti/playbook pertinenti e registrare ipotesi, costo e criterio di abbandono nel log esistente.
4. **Spike limitato**: al massimo un esperimento architetturale attivo, flag/fallback, stop anticipato se non emerge valore; massimo 1–2 settimane.
5. **Adozione**: confronto end-to-end a qualità comparabile oltre il rumore, p99/latency/memoria/energia entro budget. Nuova infrastruttura: riferimento ≥3% sul frame target o risoluzione di un limite concreto; −10% su un pass isolato non basta. Per modifiche locali il pass deve essere prioritario nel carico reale.
6. **Decisione**: adotta, rimanda o scarta, con dati e costo di manutenzione in `docs/opt-log.md`; il catalogo non selezionato resta non spuntato e non blocca la fase seguente.
7. **Verifica**: chiudere tutte le verifiche pertinenti ai task scelti. Le altre voci "Spremitura del SoC" restano opportunità da riesaminare se il carico cambia.

Le percentuali obiettivo delle fasi OPT sono **ipotesi di partenza**, da
ricalibrare con le misure di OPT-0. Le direzioni marcate **(idea Phosphor)**
sono combinazioni nostre non trovate in letteratura: vanno trattate come
ipotesi da validare con uno spike, non come tecniche già provate.

**Integrazione di ricerca del 2026-10-01.** Gli esperimenti H1–H12 del
documento di ricerca hanno task nelle fasi aperte. F5 è ora integrata in main
(2026-10-01); il relativo piano conserva contratti e verifiche per audit.
Dal 2026-10-02 tutti i 58 piani sono dettagliati in anticipo, con decisioni
empiriche condizionate agli spike, senza attivare automaticamente i candidati. Le fasi completate conservano le evidenze e gli eventuali
residui; i loro piani servono per audit e regressioni. Gli ID restano stabili.

Per una specializzazione destinata ai chip recenti, dichiarare prima dello
spike il tier target e il beneficio atteso: la soglia di adozione si misura
su quel tier, mantenendo il fallback e verificando O12 per i percorsi T0.
Nessun risultato sul Max vale come verifica sul Base. L'obiettivo è qualità
e prestazioni sostenute entro budget, con latenza e memoria sotto controllo;
occupancy e utilizzo di tutte le unità sono diagnostiche, non obiettivi da
massimizzare indipendentemente. I [piani](plans/METHOD.md) distinguono API
pubbliche, sorgenti OS, risultati di paper e ipotesi da misurare.

---

## Panoramica

| # | Fase | Tag | Era |
|---|---|---|---|
| F0 | Fondamenta Metal 4 ✅ (manca F0.8 su T0) | CORE | I · Fondamenta |
| F1 | Memoria, heap e residency ✅ | CORE | I |
| F2 | Render graph e sincronizzazione automatica ✅ | CORE | I |
| F3 | Pipeline: compilazione asincrona e archivi AOT | CORE | I |
| F4 | Osservabilità: profiler, contatori, cattura, perf log | CORE | I |
| OPT-0 | Caratterizzazione del SoC (`bench/`, modello di costo) ✅ (manca OPT-0.2 su T0) | OPT | I |
| OPT-1 | Memoria, grafo e banda come problema di ottimizzazione ✅ (OPT-1.5/1.7 parziali, T0 non misurato) | OPT | I |
| F5 | GPU scene persistente e submission guidata dalla GPU ✅ | CORE | II · Geometria |
| F6 | Mesh shader e culling a due fasi ✅ (T0/M3 fisico pendente) | CORE | II |
| F7 | Visibility buffer e shading ibrido TBDR | CORE | II |
| F8 | HDR, EDR, esposizione e MetalFX temporal | CORE | II |
| OPT-2 | Shader, pipeline e occupancy (spostata dopo F8) | OPT | II |
| OPT-3 | Geometria, culling e dati di vertice | OPT | II |
| OPT-4 | Shading, banda e ricostruzione | OPT | II |
| F9 | Infrastruttura ray tracing | CORE | III · Luce |
| F10 | Ombre ibride (CSM, RT, contact) | AAA | III |
| F11 | Migliaia di luci: ReSTIR DI, luci ad area ed emissive | AAA | III |
| F12 | Illuminazione globale: DDGI → radiance cache | AAA | III |
| F13 | Riflessioni, AO e denoiser MetalFX | AAA | III |
| F14 | Cielo, atmosfera, nuvole volumetriche, meteo | AAA | III |
| OPT-5 | Budget di raggi e campionamento | OPT | III |
| OPT-6 | Illuminazione globale, cache e ammortamento | OPT | III |
| F15 | Materiali avanzati e generazione delle varianti shader | AAA | IV · Mondo |
| F16 | Trasparenze on-tile, particelle GPU, VFX | AAA | IV |
| F17 | Post-processing cinematografico | AAA | IV |
| F18 | Geometria virtualizzata (cluster LOD + raster software) | AAA | IV |
| F19 | Terreno, vegetazione e mondo procedurale GPU | AAA | IV |
| F20 | Acqua, oceano e fluidi | EDGE | IV |
| F21 | Asset pipeline e cooker | CORE | IV |
| F22 | Streaming: MTLIO, sparse, virtual texturing | CORE | IV |
| OPT-7 | Geometria virtualizzata e mondo | OPT | IV |
| OPT-8 | Streaming, texture e materiali | OPT | IV |
| F23 | CPU ultra-ottimizzata: job system, QoS, NEON, SME | CORE | V · Simulazione (parallela) |
| F24 | Fisica, cloth e distruzione (Jolt + GPU) | AAA | V (parallela) |
| F25 | Animazione, skinning GPU, capelli a filamenti | AAA | V (parallela) |
| F26 | Audio di gioco; spazializzazione e acustica avanzata selezionabili | CORE | V (parallela) |
| F27 | Runtime di gioco: ECS, input, scripting, rete | CORE | V (parallela) |
| OPT-9 | CPU, thread e memoria unificata | OPT | V |
| OPT-10 | Simulazione | OPT | V |
| — | *Checkpoint WWDC 2027* | — | — |
| F28 | Scalabilità: preset, DRS, rate map, frame interpolation, termica | CORE | VI · Frontiera |
| F29 | Autotuning per dispositivo | EDGE | VI |
| F30 | Rendering neurale I: tensori, NTC, radiance cache neurale | EDGE | VI |
| F31 | Rendering neurale II: reti addestrate in casa con MLX | EDGE | VI |
| F32 | Path tracing in tempo reale e riferimento | EDGE | VI |
| F33 | Gaussian splatting ibrido | EDGE | VI |
| OPT-11 | Rendering neurale | OPT | VI |
| OPT-12 | Autotuning, path tracing, splatting | OPT | VI |
| OPT-13 | Scalabilità ed energia | OPT | VI |
| F34 | Editor e strumenti di contenuto | CORE | VII · Prodotto |
| F35 | QA automatizzata: test visivi, perf bot, replay | CORE | VII |
| F36 | Piattaforme: iPad, iPhone Pro, visionOS | AAA | VII |
| F37 | Vertical slice | CORE | VII |
| F38 | Distribuzione e live ops | CORE | VII |
| F39 | 2D completo, riferimento Bevy | CORE | VII · Piattaforma |
| F40 | UI di gioco completa, riferimento Bevy | CORE | VII · Piattaforma |
| F41 | Ecosistema Bevy e SDK di estensione | CORE | VII · Piattaforma |
| OPT-14 | QA delle prestazioni e ottimizzazione continua | OPT | VII |
| OPT-15 | Avvio, dimensioni e distribuzione | OPT | VII |

### Sequenza guidata dalle evidenze

| Tappa | Consegna principale | Decisione successiva |
|---|---|---|
| Ora | F5 integrata, F6 consegnata (branch `phase/f6`), poi baseline F7–F8 | Frame reale con visibilità, materiali e ricostruzione, misurato e verificato |
| Dopo F8 | Baseline del corpus OPT-4.16 con strumenti esistenti | Selezionare un problema di OPT-2/3/4 se rilevante; altrimenti proseguire F9–F13 |
| Dopo F13 | Ombre, GI, riflessi e denoise integrati; F14 se utile al corpus | Nuove misure; valutare soltanto le ottimizzazioni rese necessarie dal frame con luce |
| Sviluppo successivo | Materiali/mondo/streaming e runtime richiesti dalla slice | Attivare un piano dettagliato quando il suo consumatore è concreto |
| Piattaforma di prodotto | F27 integrazione ECS → nucleo F41/F21 → F39/F40 → editor F34 | Riuso e parità funzionale verificati; implementazione progressiva senza avviare optimizer generali |
| Frontiera | F29–F33 e ricerca SoC avanzata | Attivazione per trigger misurato; nessuna dipendenza automatica della slice |

Il percorso operativo è **F7→F8 dopo F5 integrata e F6 consegnata, poi F9→F13**, con una baseline corretta e
semplice a ogni passaggio. Cooker, runtime e strumenti minimi si introducono
quando necessari ai contenuti correnti. Le OPT non sono una barriera obbligatoria
fra ere; F37 può usare fallback e preset misurati senza aspettare ogni idea
neurale, compilatore generale o laboratorio kernel.

Il calendario precedente non costituisce un impegno: le date si stimano
sull'orizzonte vicino dopo le misure di consegna, senza una promessa "SOTA
entro fine 2027" derivata dalla sola lista di feature. L'ambizione rimane;
la scelta delle tecniche dipende dal risultato sul motore.

La [politica dei piani](plans/SEQUENCING.md) definisce i due checkpoint,
i trigger dei candidati e la maturità diversa dei 58 documenti.

---

# Era I — Fondamenta

## F0 — Fondamenta Metal 4 ✅ (manca solo la misura F0.8 su un dispositivo T0) [CORE]

- [x] F0.1 Codice Vulkan archiviato in `legacy/vulkan/`
- [x] F0.2 `phosphor_core` portabile + test doctest
- [x] F0.3 `MetalContext`: coda MTL4, 3 frame in volo, `MTLSharedEvent`, residency set, rilascio differito
- [x] F0.4 Texture bindless (tabella di `MTLResourceID`), mipmap via blit
- [x] F0.5 Pass forward PBR, depth memoryless, ImGui (coda Metal 3; da F0.7 renderer MTL4 nello stesso pass della scena)
- [x] F0.6 CI Linux (test + `metal_syntax_check`) e macOS (app + shader)
- [x] F0.7 Zero errori con `MTL_DEBUG_LAYER=1` sui testbench 1–7 (API + shader validation, anche con cambio bench `--switch-every`; verificato il 2026-09-28)
- [ ] F0.8 Baseline FPS/GPU ms di ogni testbench su M5 Max (e T0) in `docs/perf-log.md` — M5 Max misurato il 2026-09-28 (anche su alimentazione di rete, tabella di riferimento in `perf-log.md`); **T0 da misurare: serve un Mac M3/M4/M5 base, non disponibile**

## F1 — Memoria, heap e residency ✅ [CORE]

**Obiettivo**: controllo totale della memoria GPU, zero allocazioni nel frame (O7).

- [x] F1.1 Heap `MTLHeapTypePlacement` per risorse transitorie; sub-allocatore TLSF per buffer persistenti (heap placement + TLSF per tutte le risorse private; `TransientHeap` con aliasing verificato da `--transient-test`, usato dal grafo in F2.2)
- [x] F1.2 Anelli di upload per frame con suballocazione lineare (sostituiscono `UploadBuffer` di F0)
- [x] F1.3 Residency: set statico + set per-streaming, aggiornamento incrementale e batch di `commit()`
- [x] F1.4 Budget di memoria per tier, statistiche live (per heap, per categoria) in ImGui
- [x] F1.5 Gestione della pressione di memoria di sistema (notifiche macOS → eviction) — notifiche reali warn/critical/normal ricevute (`sudo memory_pressure -S`, 2026-09-28)
- [x] F1.6 Test di stress: 10.000 creazioni/distruzioni di risorse senza crescita di memoria (`--memory-stress`)

**Uscita**: Instruments Allocations piatto in gameplay; nessuna allocazione Metal nel frame.

## F2 — Render graph e sincronizzazione automatica ✅ [CORE]

**Obiettivo**: aggiungere decine di pass senza bug di sincronizzazione (Metal 4 non traccia gli hazard).

- [x] F2.1 Render graph portabile (in `phosphor_core`, testato su Linux): pass, risorse virtuali, read/write per stage, culling dei pass inutili, ordinamento topologico (accessi versionati RAW/WAR/WAW, risorse importate, Kahn stabile, intervalli di vita; `src/rendergraph/`)
- [x] F2.2 Aliasing delle risorse transitorie sugli heap di F1 (analisi degli intervalli di vita) — greedy via `ResourceSizer`; verificato sulla GPU con `--debug-graph-transients` (controllo esatto, heap −34%)
- [x] F2.3 Barrier builder Metal 4: coppie di stage con `barrierAfterEncoderStages` / `barrierAfterQueueStages`; divieto di fragment/tile nel lato "after" — regole misurate dallo spike `bench/barrier_spike` (2026-09-29): Fragment sul lato consumatore **è** legale ed efficace su M5 Max, Tile è inefficace, fragment/tile come produttore dentro un render encoder è illegale (tabella in `barrier_plan.h`, `opt-log.md`)
- [x] F2.4 Fusione automatica dei pass TBDR compatibili; load/store e `memoryless` dedotti dal grafo (O2)
- [x] F2.5 Encoding parallelo: più thread codificano pass diversi (allocatori per thread), render pass sospesi/ripresi tra command buffer — infrastruttura verificata con `--debug-split-encoding`; guadagno da misurare in F5
- [x] F2.6 Async compute: pass compute indipendenti su una seconda coda MTL4 con eventi — verificato con `--debug-async-compute` (kernel sintetici, controllo esatto); guadagno da misurare in F5/F8
- [x] F2.7 Migrazione del forward di F0 sul grafo; backface culling verificato e riattivato (istanze specchiate: cull front; materiali glTF `doubleSided`: niente culling)
- [x] F2.8 Dump del grafo (Graphviz) con banda stimata per risorsa (O1) — `--dump-graph FILE`

**Uscita**: stessa immagine di F0; unit test su ordinamento, aliasing e barriere attese. Verificata il 2026-09-29: 0 pixel diversi sui 7 bench anche con i flag di debug, validazione a zero, 0 allocazioni GPU nei frame, nessuna regressione rispetto a `main` (`perf-log.md`). Due riferimenti rigenerati con motivazione: Cornell box (pareti orientate male, corrette nei contenuti) e 3 pixel di Stress Test (facce posteriori che vincevano il depth test).

## F3 — Pipeline: compilazione asincrona e archivi AOT ✅ [CORE]

**Obiettivo**: stutter di compilazione zero (O9) fin dall'inizio, non alla fine.

- [x] F3.1 `MTL4Compiler` su thread dedicati con QoS inferiore al render thread, `maximumConcurrentCompilationTaskCount` worker — `PipelineCache` + `CompileQueue`: 18 thread su M5 Max, QoS utility letta a runtime (0x11); spike "tempesta" di 42 compilazioni a freddo: nessun hitch, render thread −12% di CPU rispetto a QoS interactive (`opt-log.md`); verifica su T0 (O12) da fare
- [x] F3.2 Pipeline "flessibili" subito, specializzate in background; chiave di cache hashata — chiave FNV-1a 64 deterministica, registro portabile con swap solo a inizio frame (grafo in cache mai ricompilato); il fallback immediato di una variante è la pipeline generica a stato completo (bit-identica a F2); la specializzazione flessibile Metal 4 è usata solo senza una generica pronta, perché avvisa in validazione e cambia fino a 300k pixel di 1 LSB (misurato, `--debug-flexible-pipelines`)
- [x] F3.3 Function constant per le varianti (O11); tabella delle varianti generata — `shaders/variants.def` → `tools/variant_gen` → tabelle C++/MSL; 42 varianti del forward (tipi di luce, emissivo, debug mode); le varianti differiscono dalla generica di ≤1 LSB in 3–30 pixel per bench (riferimenti rigenerati); i bench usano 2 varianti, le altre sono coperte da harvest e test
- [x] F3.4 Raccolta automatica dei descrittori in `.mtl4-json` durante i test, `metal-tt` in CI → `MTL4Archive` — harvest con `MTL4PipelineDataSetSerializer` (`tools/harvest_pipelines.sh`, JSON committato), archivio costruito da CMake con `metal-tt` (Release; niente archivio con la debug info degli shader, che `metal-tt` non traduce); il CI costruisce l'archivio per tutte le arch come artefatto, valido solo per macOS 26 (rifiutato su 27, misurato)
- [x] F3.5 Fallback obbligatorio su miss dell'archivio (OS o GPU diversi), con tasso di miss registrato — riga `PIPELINES`, report JSON, pannello Pipelines; `tools/archive_check.sh`: 7 scenari con archivi reali (OS diverso = artefatto del CI, arch diversa, parziale, vecchio, assente) PASS con controllo negativo; con `MTL_SHADER_VALIDATION` l'archivio è inutilizzabile (Metal) e viene dichiarato non disponibile
- [x] F3.6 Hot reload degli shader in Debug (ricompilazione del `.metallib` e sostituzione delle pipeline) — `ShaderReloader` (watcher su thread utility, stessi flag di CMake) + generazioni atomiche nel registro; `tools/hot_reload_check.sh`: auto-test con tutti i flag F2, controllo negativo, watcher reale con errore di sintassi e modifica valida

**Uscita**: nessun hitch al cambio testbench; avvio a freddo con archivio senza compilazioni. Verificata il 2026-09-29 (`perf-log.md`, sezione F3): 0 compilazioni sul render thread dopo l'avvio e 0 hitch con l'archivio (controllo negativo `--pipeline-sync` rilevato); avvio a freddo con archivio a 0 chiamate al compilatore anche con la cache shader dell'OS vuota (12 hit su 12 con tutti i bench e i flag); `visual_check` a 0 pixel e 0 messaggi con tutti i flag F2, `--switch-every`/`--resize-every`, `--pipeline-sync`, `--no-pipeline-archive`; 0 allocazioni GPU nei frame; nessuna regressione rispetto a `main`. Trovato e corretto un accumulo di memoria CPU preesistente da F0 (eventi AppKit senza autorelease pool).

## F4 — Osservabilità [CORE]

**Obiettivo**: ogni decisione di ottimizzazione si basa su numeri.

- [x] F4.1 `MTL4CounterHeap` con timestamp per pass, media/max su 60 frame, pannello Pass Timings — timestamp di fine unità + inizio commit (encoder "anchor"), unità = gruppo di render fuso o pass compute; tempo esclusivo sulla timeline della coda; pannello Pass Timings (media/max 60 frame, DRAM stimata), report JSON v2 `passes`; controllo negativo `--debug-gpu-cost` lineare (0,74 ms/1000 iterazioni) col forward fermo; attribuzione dei pass fusi con `--gpu-timing-unfused`, frame senza sovrapposizione con `--gpu-timing-serial` (`opt-log.md`, `perf-log.md`)
- [x] F4.2 Tracy: zone CPU per sistema, zone GPU per pass, allocazioni — opzione `PHOSPHOR_TRACY` (Tracy 0.14.1, on demand); zone GPU manuali dai timestamp MTL4 (il backend Metal di Tracy non supporta MTL4), pool di allocazione CPU e GPU per categoria; `tools/tracy_check.sh`: zone GPU = report, costo CPU non misurabile
- [x] F4.3 Cattura GPU da tasto (`MTLCaptureManager`), cattura automatica sui frame sopra soglia — `--gpu-capture` (F12), `--gpu-capture-frame N`, `--gpu-capture-over MS` (cattura il frame successivo a quello lento: non si cattura a posteriori); coda MTL4 grafica come oggetto (Metal 4 non cattura il device), archivio disattivato durante le catture
- [ ] F4.4 Contatori hardware (occupancy, banda, compressione, stalli) esportati per pass quando disponibili — **parziale**: su M5 Max / macOS 27.2 né l'API (MTL4 ha solo timestamp, counter set legacy = solo `timestamp`) né `xctrace` da riga di comando (solo "RT Unit Active") espongono contatori hardware; `tools/gpu_trace.sh` esporta per encoder gli intervalli Vertex/Fragment/Compute, lo Shader Timeline (non affidabile come costo per pass nei gruppi fusi, misurato) e lo stato di clock. Da riprendere quando Apple esporrà i contatori
- [x] F4.5 `docs/perf-log.md` + script che registra le misure dei testbench per commit — `tools/perf_record.sh` → `docs/perf-history*.csv` (baseline di fine F4 registrata), `tools/perf_table.py latest|compare|passes`
- [x] F4.6 Modalità benchmark da riga di comando (`--bench N --frames 600 --report out.json`) — report v2 con tempi per pass; `tools/bench_all.sh --stats --passes --vsync` con deviazione standard e CV tra run
- [x] F4.7 Sovrapposizioni di debug generiche (heatmap di costo per tile, overdraw, contatori) — `--overlay overdraw|lights|tilecost|timings` e selettore UI; overdraw = frammenti rasterizzati (l'HSR non è visibile), costo per tile = Σ overdraw × (1 + luci) come approssimazione dichiarata; costi per pass nel perf-log

**Uscita**: ogni pass ha tempo GPU e banda visibili; benchmark ripetibili da CLI.

---

## OPT-0 — Caratterizzazione del SoC ✅ (manca OPT-0.2 su un T0) [OPT]

**Obiettivo**: conoscere i chip su cui giriamo meglio di qualunque documento
pubblico. Senza questi numeri tutte le fasi OPT successive tirano a indovinare.

**Spremitura del SoC**
- [x] OPT-0.1 Suite `bench/` con i microbenchmark B-01…B-28 del playbook, eseguibile da CLI, risultati in `bench/results/<chip>-<os>.json` — `bench/soc` (`soc_bench`: `--list/--only/--quick/--runs/--out/--validate/--force-family apple9/--window/--soak`), 28 benchmark, ognuno con un controllo negativo di cui è stato provato il fallimento; `tools/soc_bench_all.sh` (× 3 run, validazione 0 messaggi, percorsi Apple9, `leaks` 0); spike e scoperte in `opt-log.md` (OPT-0)
- [ ] OPT-0.2 Esecuzione su M5 Max e su almeno un T0 (M3/M4 base); poi su ogni Mac disponibile — **M5 Max misurato** (`bench/results/m5max-macos27.2.json`, × 3 run, all'alimentazione; anche con i soli percorsi Apple9); **T0 non disponibile**: resta da fare su un Mac M3/M4/M5 base
- [x] OPT-0.3 **Modello di costo** Phosphor: tabella per chip con throughput ALU FP16/FP32, banda per livello, dimensione stimata di SLC, costo di barriere, dispatch, load/store, raggi/s, GEMM per tile — `SocCostModel` portabile (`src/diagnostics/soc_model.h`, doctest), CLI `soc_model`, [`soc-model.md`](soc-model.md) generato (CV tra run e cause, confronto con fonti esterne e scarti spiegati); SLC = stima da un fit, dichiarata
- [x] OPT-0.4 Grafici roofline per chip (tetto di banda e di calcolo) su cui collocare ogni pass — [`img/roofline-m5max.svg`](img/roofline-m5max.svg) con i 7 testbench (report v3 con `work` per pass, conteggi AIR del cammino minimo della variante usata, copertura dei pixel dalle catture): predetto ≤ misurato per ogni pass, scarti tabulati; `tools/soc_roofline.sh` lo rifà
- [x] OPT-0.5 Verifica delle voci **(ipotesi)** del playbook: correggere il playbook con i dati misurati — ogni riga "Misura" con valore e benchmark (o "non misurabile qui" e perché), §16 con la colonna misurata su M5 Max; correzioni (mul intera 1,6× l'add, FP16 2× solo con 32 catene, decompressione MTLIO sulla CPU, latenza ANE ×2–2,9 con GPU carico, 19,9 TFLOPS di terzi = GEMM FP16)
- [x] OPT-0.6 Soglie critiche misurate: partial render (B-12), punto di thrashing dei registri (B-04), dimensione massima dell'imageblock (B-15) — partial render a ~23,7 M triangoli per pass con 16 varyings float4 (~47 M con 8, ~67 M con 4); thrashing a 128 valori FP32 vivi (stabile fino a 120); imageblock 24 B/pixel con tile 32×32, 56 B con 32×16 e 16×16 (oltre il limite il tile kernel non gira, senza errori)

**Uscita**: modello di costo pubblicato in `docs/soc-model.md` e usato come
input da render graph (OPT-1) e autotuning (F29). Verificata il 2026-09-30 su
M5 Max: `tools/soc_bench_all.sh` PASS (28 benchmark × 3 run, tutti i
controlli negativi, `--validate` 0 messaggi, percorsi Apple9, `leaks` 0);
motore invariato (`ctest`, `metal_syntax_check`, `visual_check` 0 pixel,
`bench_all` A/B/A contro `main`, `perf-log.md`). L'uso del modello da parte
del render graph è di OPT-1; OPT-0.2 attende un T0.

## OPT-1 — Memoria, grafo e banda come problema di ottimizzazione ✅ (OPT-1.5 e OPT-1.7 parziali; T0 non misurato) [OPT]

**Obiettivo (ipotesi)**: −25% di byte DRAM per frame e −20% di memoria di
picco rispetto alla fine di F4, su T0 e T2.

**Letture**: [R1] [R2] [R3] [R4]; playbook S-TBDR-*, S-MEM-*, S-SYNC-*.

**Direzioni di ricerca**
- [x] OPT-1.1 **Render graph risolto come problema di ottimizzazione**: ordinamento dei pass, aliasing e fusione dei pass TBDR formulati come programma lineare intero misto (MILP), risolto offline per ogni preset di qualità; a runtime si carica il piano ottimo invece di usare euristiche greedy [R1][R2]. Stesso approccio che Checkmate usa per i tensori [R3] e che [R12] usa per le triangle strip — ottimizzatore portabile `src/rendergraph/optimizer/` (spike 2: il MILP con HiGHS è ottimo fino a ~41 pass ma va al limite oltre e non trova soluzioni sopra 120 → adottati DP esatto sui downset + annealing sul compilatore reale, mai peggio del greedy; ottimo esatto verificato su 5040 ordini); piani offline per famiglia e chiave strutturale in `shaders/graph-plans.json` (`tools/graph_opt`), scelti **per misura** (`tools/graph_select.py`, immagini identiche obbligatorie), caricati con `--graph-opt plan`; scenari misurati: −8,4% / −3,5% / −0,5% / −2,3% di frame, 0 pixel, 0 messaggi (perf-log)
- [x] OPT-1.2 **Rematerializzazione TBDR** (idea Phosphor, ispirata a [R3]): per ogni risorsa il grafo sceglie se scriverla in DRAM o **ricalcolarla nella tile** quando serve, minimizzando i byte (O1). Il visibility buffer è un caso particolare di questa idea: generalizzarla a tutti i segnali economici — scelta di costruzione per segnale (velocity, CoC) valutata dall'ottimizzatore e decisa dalla misura, ricalcolo esatto (0 pixel); TAA limitato dall'ALU: +1,5% (scartata), catena di post: −3,7% byte, −8% heap a tempo neutro (adottata)
- [x] OPT-1.3 Aliasing ottimo con colorazione di grafi degli intervalli di vita, vincolata ai formati che preservano la compressione lossless (S-TEX-2) — `AliasPolicy::Coloring` (mai peggio del greedy, raggiunge il limite inferiore su 400/400 casi casuali) e `ColoringStageClass`; B-11 esteso: heap e alias tra formati diversi mantengono la compressione, `PixelFormatView` la toglie (nessun vincolo di formato necessario); sugli scenari il greedy era già al limite inferiore
- [x] OPT-1.4 Barriere minime: raggruppamento, scelta intra-encoder vs coda sulla base dei costi misurati (B-18) come pesi del solver di OPT-1.1 — `BarrierPolicy::Minimal` (stadi degli accessi massimali, Minimal ⊆ Conservative su 300 grafi casuali e sugli scenari), pesi misurati nel modello (7 µs compute dipendente, 11 µs render pass); con l'ordine dei piani vale −5% di frame sullo scenario 0; limite dichiarato: nessuna corsa resa visibile dal controllo negativo sul GPU (default `off`)

**Spremitura del SoC**
- [ ] OPT-1.5 Mappa dei byte DRAM per pass (contatori) e verifica del budget per tier (S-MEM-1) — **parziale**: byte per pass stimati dal grafo nel report/dump e budget per tier (`graph_budget`: 5–6% del budget a 60 fps su M5 Max misurato, 19–23% su M5 base e 29–36% su M3 base da specifiche esterne); verifica indiretta tempo ≥ byte/banda in 69/69 unità (spike 1); **niente contatori hardware headless** (F4.4) e niente T0 fisico
- [x] OPT-1.6 Ogni intermedio candidato a `memoryless` verificato; ogni `.store` giustificato (S-TBDR-4) — `graph_lint` (`LintMode`, attivo con `--graph-opt greedy|plan`): store ingiustificato = errore, ogni transitorio non memoryless con il motivo, store conservativi segnalati; 0 errori sugli scenari
- [ ] OPT-1.7 Working set dei pass compute dimensionati per restare nella SLC misurata (S-MEM-2) — **parziale**: `graph_budget` segnala i compute oltre la SLC stimata (71,4 MiB, stima da fit) e le coppie produttore→consumatore riusabili; adiacenza misurata −7% sul pass (spike 7); il ridimensionamento dei working set riguarda i compute reali del motore (F5+), oggi assenti
- [x] OPT-1.8 Sovrapposizione tra pass misurata e massimizzata riordinando geometria e fragment (S-TBDR-6) — modello a due code con sovrapposizione (ordinamento corretto delle varianti misurate) e piani che mettono le ombre accanto ai compute indipendenti: −8,4% (deferred) e −3,5% (forward+) misurati
- [x] OPT-1.9 Seconda coda MTL4 per compute asincrono dove B-19 mostra guadagno (S-SYNC-2) — coda per pass idoneo scelta dal piano: scenario async, GI sulla seconda coda e particelle sulla grafica insieme alla fusione G-buffer+Lighting (−2,3% frame, −15,7% byte); la fusione anticipa le attese tra code (spike 5)
- [x] OPT-1.10 Anelli di upload in `shared` write-combined; tutti i dati letti spesso in `private` (S-MEM-3) — nuovo B-29: WC = cached in scrittura CPU, lettura CPU da WC 20× più lenta, `private` = `shared` in lettura GPU; il motore usa già `shared`+WC per gli anelli e `private` per geometria, texture e transitori (audit)

**Uscita**: obiettivo di banda raggiunto o scostamento spiegato in `docs/opt-log.md`.
Verificata il 2026-10-01 su M5 Max: obiettivo (−25% byte, −20% picco) raggiunto
solo per il picco dello scenario async (−27,2%, byte −15,7%); negli altri
scenari l'ordine è imposto dai dati e i byte restano quelli degli algoritmi
(scarto spiegato in opt-log, "OPT-1 — Risultati"); guadagni di frame
−8,4% / −3,5% / −0,5% / −2,3% sugli scenari misurati; motore invariato
(perf-log). OPT-1.5 e OPT-1.7 restano parziali (niente contatori hardware,
nessun compute reale da ridimensionare); T0 non disponibile (O12).

---

# Era II — Geometria GPU-driven

## F5 — GPU scene persistente e submission guidata dalla GPU ✅ [CORE]

**Obiettivo**: la CPU non codifica più draw per oggetto (O8).

- [x] F5.1 GPU scene persistente: istanze, materiali, mesh in buffer GPU con **aggiornamenti delta** (solo gli oggetti cambiati), memoria unificata con scrittura diretta — `SceneStore` (slot stabili per bucket (classe di culling, mesh), buchi riutilizzati, materiali persistenti per entità), tracciamento delle modifiche nell'ECS in O(cambiati) e riciclo degli id, record delta scritti dalla CPU nell'anello del frame e applicati da `scene_scatter` in buffer `private` persistenti (copia intera oltre 1/8 cambiato, spike S2); byte per frame proporzionali ai cambi (96 B per record, 1,1 KiB/frame sulle scene ferme), `--debug-gpu-scene` esatto byte per byte
- [x] F5.2 Gerarchia di transform aggiornata in compute — figli e moto procedurale calcolati sulla GPU con prodotto `fp contract(off)` bit-identico a glm (spike S6), verificati bit per bit dal self-check; radici in moto con figli espanse da una coda persistente
- [x] F5.3 **Indirect command buffer costruiti dalla GPU**: un dispatch di culling scrive i comandi di draw — `scene_draw_build` scrive un comando per bucket, tre range fissi per classe di culling (Apple9/Apple10), `--gpu-driven off|on` (default on): stessa immagine (0 pixel sugli 8 bench con ogni flag F2), comandi CPU costanti al variare delle istanze (66 nel bench 8 da 10K a 1M; off cresce con i bucket)
- [x] F5.4 Catena di dispatch indiretti come sostituto dei work graph (Metal non li ha): code GPU a produttore/consumatore tra pass — `renderer/gpu_queue.h` (append aggregati per SIMD-group, overflow contato, argomenti scritti da un kernel a un thread, `dispatchThreadgroups` indiretto); usi reali: code dei nodi sporchi della gerarchia per livello e coda persistente delle radici in moto; verifica GPU F5-K3 con conteggi dipendenti dai dati
- [x] F5.5 Instance culling in compute (frustum + distanza + dimensione su schermo) — `renderer/cull_math.h` condiviso C++/MSL, piani del reverse-Z infinito (corretto `Camera::getFrustumPlanes`), compattazione stabile reduce-then-scan (deterministica, spike S4); lista visibile = riferimento CPU (0 differenze anche nella banda), `--cull-distance`/`--cull-min-pixels` (spenti di default), controllo negativo a pixel
- [x] F5.6 Testbench "1M istanze" dinamiche — bench 8 "1M Instances (dynamic)": 1M istanze in moto (moto GPU, 1% aggiornato dalla CPU, satelliti a profondità 3, `--churn`, `--instances`, `--scene-meshes`); 60 fps su M5 Max (dettagli e CV nel perf-log); T0 non disponibile (O12)

**Uscita**: Stress Test con conteggio di comandi CPU costante; 1M istanze in movimento a 60 fps su T2.
Raggiunta su M5 Max (T2): comandi CPU costanti al variare delle istanze e
dei bucket in `--gpu-driven on`; bench 8 a 1M istanze in moto con p99 di
frame, CPU e GPU sotto 16,6 ms e 120 fps con vsync (`perf-log.md`, "F5").

## F6 — Mesh shader e culling a due fasi ✅ [CORE]

- [x] F6.1 Meshlet ritarati per Apple: confronto 64/96/128 triangoli e baseline 64 vertici/124 triangoli, `meshopt_buildMeshletsSpatial`, limiti verificati; misure sul M5 Max nativo e con fallback Apple9, certificazione M3 fisica differita secondo O12 — `MeshletBuildOptions` (standard/spatial, massimi validati contro gli assert di meshoptimizer v1.3 e le uscite del mesh shader), spike S1 CPU (`tools/meshlet_cook`, 7 mesh × 12 opzioni) e GPU nel frame (nativo e `--force-family apple9`): nessuna variante ≥3% migliore su tutto il corpus, resta 64/124; ricostruzione, bounds e coni testati; M3 fisico `EXTERNAL_VALIDATION_PENDING`
- [x] F6.2 Object shader: culling per meshlet (frustum, cono di normali, area su schermo, occlusione Hi-Z) — `renderer/meshlet_cull_math.h` condiviso C++/MSL: sfera con limite spettrale (Gershgorin, corregge anche il raggio F5 con shear), frustum, cono esatto in spazio mesh per ogni affine invertibile, footprint Hi-Z conservativo (near plane, NPOT, margini), area su schermo approssimata (`--meshlet-min-pixels`, spenta nel preset esatto); compattazione nel payload con prefix sum SIMD; ogni decisione verificata contro il riferimento CPU (`--debug-meshlets`)
- [x] F6.3 Mesh shader con output minimi dichiarati e primitive scartate omesse (guida Apple) — `meshlet_mesh`: attributi del forward e conteggio esatto delle primitive (nessuna primitiva di riempimento, i meshlet scartati non lanciano threadgroup); scarto per triangolo con compattazione stabile disponibile (`--meshlet-triangle-cull on`, immagine identica) ma spento di default (A/B inconclusivo sul M5, `opt-log.md`); pixel-identico all'indexed sui bench 1–7 (bench 8: 1 pixel di pareggio di depth dipendente dall'ordine, provato), varianti forward tutte verificate (`variant_check`)
- [x] F6.4 Piramide Hi-Z in compute con riduzione SIMD-group (O5); percorso sampler min/max su Apple10 — `HiZBuilder`/`shaders/hiz.metal`: livello 0 a potenze di due, compute SIMD-group (default, Apple9) e sampler min (Apple10, `--hiz-path sampler`, rifiutato con l'Apple9 effettivo), entrambi bit-exact con la piramide CPU su NPOT/buchi (spike S3); storia per vista con reset su resize/switch/taglio
- [x] F6.5 Culling a due fasi: frame precedente → nuovo test dei rifiutati sulla profondità corrente — fase A con la storia (solo un suggerimento), Hi-Z della fase A, flag→scan→scatter dei rifiutati, fase B contro la profondità corrente, Hi-Z finale come storia; Culling Viz scriptata (occluder che scompare, tagli, oggetto veloce, spawn/delete/riuso): two-phase = indexed al pixel sui frame degli eventi, nessuna superficie persa nel self-check
- [x] F6.6 Compattazione dei meshlet visibili con prefix sum SIMD-group, draw mesh indirette (Apple9) — candidati in tre regioni di classe con scan stabile a 3 canali, argomenti indiretti per classe e fase (6 draw, comandi CPU costanti), capacità = limite superiore strutturale; overflow mai troncato: il `Draw build` F5 disegna il frame (provato con overflow forzato, immagine identica)
- [x] F6.7 Debug view: meshlet colorati, rifiutati per fase, Hi-Z — `--debug-view meshlets|cull|hiz`, contatori per fase e motivo nel report (schema 6) e nel pannello, `--debug-meshlets` con controlli negativi id/depth/count, `tools/f6_check.sh`; Culling Viz 1080p a 60 fps reali sul M5 Max (p95 ≤ 16,67 ms in 27/27 esecuzioni mesh, nativo, Apple9, sampler; macchina non quieta, `perf-log.md` "F6")

**Uscita di sviluppo**: Culling Viz a 60 fps reali sul M5 Max (preset fisso 1080p, p95 frame ≤16,67 ms), percorso nativo e fallback Apple9 verificati; nessuna superficie visibile persa. Target 60 fps T0 conservato come certificazione fisica esterna pendente, non blocca F7. Dettagli in [F6](plans/F6.md) e [politica hardware](plans/HARDWARE_VALIDATION.md).
Raggiunta su M5 Max (`DEVELOPMENT_ACCEPTED`): Culling Viz scriptata 1920×1080
a 120 fps presentati con p95 9,1–10,2 ms nella serie meno carica e ≤ 16,5 ms
anche con il compositore saturato da un altro agente, percorso nativo,
fallback Apple9 e sampler Apple10; nessuna superficie persa (self-check,
eventi al pixel). Certificazione T0/M3 fisica `EXTERNAL_VALIDATION_PENDING`.

## F7 — Visibility buffer e shading ibrido TBDR [CORE]

**Obiettivo**: il cuore del renderer, progettato per la tile memory.

- [ ] F7.1 Visibility buffer R32Uint con codifica **25 bit cluster + 7 bit triangolo**
- [ ] F7.2 Material resolve in compute: baricentriche analitiche, derivate per il mip, fetch attributi dalla GPU scene
- [ ] F7.3 **Binning per materiale** in compute (tile classification): uno shader specializzato per classe di materiale invece di un uber-shader (O4, O11)
- [ ] F7.4 **[CANDIDATO]** **Spike [EDGE]**: deferred "on-tile" con tile shader e imageblock contro V-buffer + resolve in compute; misurare banda e tempo su T0 e T2; adottare un ibrido per tier se conviene; rivalutare dopo F8, senza bloccare la baseline F7
- [ ] F7.5 **[CANDIDATO]** **Spike [EDGE]**: shading a frequenza variabile software nel resolve (2×2 dove il contrasto è basso), guidato dalla luminanza del frame precedente; rivalutare dopo F8, senza bloccare la baseline F7
- [ ] F7.6 Canali per il denoiser MetalFX emessi da subito: normali con segno, albedo diffusa, albedo speculare con Fresnel, roughness, motion vector, depth
- [ ] F7.7 Alpha test in fase raster separata; ordine opachi → alpha-test → traslucidi

**Uscita della baseline**: V-buffer + resolve corretti su Sponza, confronto con forward di F0 per immagine, tempo e memoria; scarti documentati per il checkpoint dopo F8. Il vantaggio prestazionale resta un obiettivo misurato; F7.4/F7.5 si selezionano dopo F8 e non ne bloccano la baseline.

## F8 — HDR, EDR, esposizione e MetalFX temporal [CORE]

- [ ] F8.1 Target RGBA16F nella tile, istogramma di luminanza in compute con SIMD-group, esposizione automatica
- [ ] F8.2 Tonemapping configurabile (AgX, ACES, curva custom) e uscita **EDR** su display XDR con calibrazione della luminanza massima
- [ ] F8.3 Jitter Halton sub-pixel, motion vector per oggetti e camera
- [ ] F8.4 MetalFX temporal upscaler con risoluzione dinamica e reactive mask (scala max 2x); valutare subrectangle, motion vector diretti e distortion field se esposti dall'SDK/device [R111], mantenendo il percorso base; queste estensioni si attivano solo se necessarie al carico corrente
- [ ] F8.5 Sharpening adattivo e mip bias corretto per la risoluzione di render
- [ ] F8.6 **Risorse temporali esplicite nel render graph**: storia per vista, versioni fra frame, inizializzazione e invalidazione su camera cut, resize e cambio di risoluzione; distinguere storia temporale e risorse per frame slot, dichiarare ultimo lettore e sincronizzazione prima del riuso; iniziare con risorse per vista e riuso conservativo, senza richiedere aliasing temporale generale di OPT-4.14
- [ ] F8.7 **Suite di qualità in movimento**: clip deterministiche con disocclusioni, camera rapida, dettagli sub-pixel, oggetti animati e cambi di esposizione/risoluzione; riferimento ad alta qualità e soglie dichiarate per ghosting, flicker, perdita di dettaglio e tempo di recupero della storia. Estendere la suite quando arrivano illuminazione (F13), trasparenze (F16) e vegetazione (F19), senza attendere F35

**Uscita**: stabilità temporale senza ghosting visibile sui testbench in movimento; storia isolata per vista e riuso corretto con più frame in volo, camera cut e resize; clip e soglie di F8.7 registrate; misure base per ricalibrare i budget.

---

## OPT-2 — Shader, pipeline e occupancy [OPT]

*Spostata dopo F8 il 2026-10-01 (decisione del proprietario): alla fine
dell'era I l'unico shader caldo era `forward_fs`; le direzioni di OPT-2
(shader LOD, specializzazione da profilo, ILP, sweep dei threadgroup,
`half`) rendono sui kernel e sugli shader che F5–F8 aggiungono (culling,
compattazione, Hi-Z, material resolve, post). Gli ID restano invariati.*

**Obiettivo (ipotesi)**: −15% di tempo GPU totale a parità di immagine;
occupancy, registri e stalli usati per trovare il punto migliore per shader,
senza imporre il 90% quando aumenta la contesa [R91].

**Letture**: [R5] [R6] [R7]; playbook S-ALU-*, S-OCC-*, S-SIMD-*.

**Direzioni di ricerca**
- [ ] OPT-2.1 **[CANDIDATO]** **Shader LOD automatico**: varianti semplificate degli shader generate con tecniche di semplificazione automatica [R5][R6] e usate dove l'errore non si vede (oggetti lontani, riflessioni, GI, tier bassi)
- [ ] OPT-2.2 **[CANDIDATO]** **Specializzazione guidata dal profilo**: registrare durante i test quali combinazioni di feature compaiono davvero e generare varianti (function constant) solo per quelle (O11)
- [ ] OPT-2.3 **[CANDIDATO]** **Roofline automatica**: strumento che da counter heap e contatori calcola intensità aritmetica e collo di bottiglia per pass e lo mostra in ImGui [R4]
- [ ] OPT-2.4 **[CANDIDATO]** Riscrittura ILP-friendly dei kernel più caldi (più catene indipendenti, niente `float4` che maschera dipendenze) [R7]

**Spremitura del SoC**
- [ ] OPT-2.5 **[CANDIDATO]** Censimento dei registri vivi per riga (Xcode 26.4+) per ogni shader caldo; riduzione dei picchi (S-OCC-1)
- [ ] OPT-2.6 **[CANDIDATO]** Tabella occupancy target + causa di throttling per shader, con correzione mirata (S-OCC-2)
- [ ] OPT-2.7 **[CANDIDATO]** Conversione sistematica a `half` con suffisso `h`, verificata dai test visivi (S-ALU-3)
- [ ] OPT-2.8 **[CANDIDATO]** Strength reduction: niente div/mod interi nei cicli caldi, trascendentali `half`/`fast::` dove accettabile (S-ALU-4)
- [ ] OPT-2.9 **[CANDIDATO]** Sweep delle dimensioni di threadgroup per ogni kernel e per chip, risultati salvati per l'autotuning (S-OCC-3)
- [ ] OPT-2.10 **[CANDIDATO]** Compattazioni e riduzioni riscritte con intrinsics SIMD-group (S-SIMD-1)
- [ ] OPT-2.11 **[CANDIDATO]** Tempo di compilazione e numero di varianti misurati; pruning delle varianti mai usate

## OPT-3 — Geometria, culling e dati di vertice [OPT]

**Obiettivo (ipotesi)**: −30% di tempo nel culling + raster del visibility
buffer e −40% di byte di geometria letti per frame rispetto alla fine di F8.

**Letture**: [R9] [R10] [R11] [R12] [R13] [R14] [R22]; playbook S-GEO-*, S-TBDR-3.

**Direzioni di ricerca**
- [ ] OPT-3.1 **[CANDIDATO]** **Meshlet compressi decompressi nel mesh shader**: triangle strip generalizzate ottime [R12] e formato denso stile DGF [R13], confrontati con il codec di meshoptimizer; meno byte per triangolo letti dalla DRAM
- [ ] OPT-3.2 **[CANDIDATO]** **Quantizzazione aggressiva**: posizioni a 16 bit relative al cluster, normali e tangenti ottaedriche [R14], UV `half`; attributi letti solo nel resolve
- [ ] OPT-3.3 **[CANDIDATO]** **Occlusion culling ibrido CPU+GPU su memoria unificata** (idea Phosphor): rasterizzazione software degli occluder sui P-core liberi con NEON [R10], risultato letto dalla GPU senza copie per scartare istanze prima del Hi-Z
- [ ] OPT-3.4 **[CANDIDATO]** Strategie di generazione dei meshlet confrontate sul nostro hardware [R11]; coni di normali più stretti
- [ ] OPT-3.5 **[CANDIDATO]** **Tessellation adattiva in compute** [R22] per superfici lisce invece di geometria densa precalcolata
- [ ] OPT-3.6 **[CANDIDATO]** Culling incrementale: su camera ferma o quasi ferma, riuso della lista visibile e test solo dei cluster cambiati [R9]

**Spremitura del SoC**
- [ ] OPT-3.7 **[CANDIDATO]** Dimensione dei meshlet e massimi dichiarati calibrati per chip con B-16 (S-GEO-1)
- [ ] OPT-3.8 **[CANDIDATO]** Soglia del parameter buffer (B-12) mai superata nei testbench; attributi nel pass di raster ridotti al minimo (S-TBDR-3)
- [ ] OPT-3.9 **[CANDIDATO]** Percorso Apple10: Hi-Z con riduzione min/max nel sampler, ICB estesi, valori per-vertex non interpolati (S-GEO-4, S-GEO-5)
- [ ] OPT-3.10 **[CANDIDATO]** Soglie di LOD diverse per M3/M4 e M5 (geometria 2x su M5, S-GEO-2)
- [ ] OPT-3.11 **[CANDIDATO]** Compattazione dei meshlet visibili con prefix sum SIMD-group, nessun atomico globale per thread (S-SIMD-1, S-SIMD-4)

## OPT-4 — Shading, banda e ricostruzione [OPT]

**Obiettivo (ipotesi)**: −25% di tempo nel material resolve e nel post, a
qualità percepita invariata.

**Letture**: [R15] [R16] [R17] [R18] [R19] [R20] [R21]; playbook S-TEX-*, S-TBDR-*, S-SIMD-2.

**Direzioni di ricerca**
- [ ] OPT-4.1 **[CANDIDATO]** **Shading disaccoppiato / texel shading** [R16][R17] per superfici costose: ombreggiare in spazio texture a frequenza ridotta e riusare tra frame
- [ ] OPT-4.2 **[CANDIDATO]** **Variable rate shading software sul visibility buffer** [R18], con la frequenza decisa da un predittore dell'errore visivo appreso [R21]; su Apple anche con rasterization rate map per i pass raster
- [ ] OPT-4.3 **[CANDIDATO]** **Catene di mip in un solo dispatch** (stile SPD [R20]) per Hi-Z, bloom, esposizione: meno pass, meno banda
- [ ] OPT-4.4 **[CANDIDATO]** Resolve per tile classificate con uno shader specializzato per classe di materiale e salto dei rami con `simd_all/any` [R15]
- [ ] OPT-4.5 **[CANDIDATO]** **Spike deferred on-tile vs V-buffer** ripetuto con i dati di OPT-0: per ogni tier scegliere la combinazione migliore (anche ibrida: V-buffer per la geometria, lighting on-tile); valutare sul frame completo banda, pressione sulla memoria on-chip, occupancy, partial render e sovrapposizione persa dalle fusioni, con contatori quando disponibili e misure dichiarate negli altri casi
- [ ] OPT-4.6 **[CANDIDATO]** Ricostruzione temporale: rapporto qualità/costo della risoluzione interna con MetalFX per ogni tier [R19]

**Spremitura del SoC**
- [ ] OPT-4.7 **[CANDIDATO]** Formati intermedi ridotti (R11G11B10F, RGB9E5, `half`) dove i test visivi lo consentono (S-TEX-1)
- [ ] OPT-4.8 **[CANDIDATO]** Output compute scritti a blocchi interi per la compressione universale di M5; contatore "write inefficiency" a zero (S-TEX-2)
- [ ] OPT-4.9 **[CANDIDATO]** Compressione disattivata sulle texture ad accesso sparso dopo misura del Compression Ratio (S-TEX-2)
- [ ] OPT-4.10 **[CANDIDATO]** Mip bias corretto per MetalFX; mip più bassi per effetti a bassa frequenza (S-TEX-3)
- [ ] OPT-4.11 **[CANDIDATO]** MSAA 4x memoryless valutato per UI e vegetazione in alpha-to-coverage (S-TBDR-8)
- [ ] OPT-4.12 **[CANDIDATO]** Uscita EDR con headroom interrogato e tonemapping adattato al display (S-DISP-2)
- [ ] OPT-4.13 **[CANDIDATO]** **Accessi a sottorisorse nel render graph**: mip, layer e aspect delle texture, intervalli di byte dei buffer; dipendenze e barriere per intervalli sovrapposti, mantenendo allocazione e vincoli di aliasing della risorsa fisica intera; test di accessi disgiunti, parzialmente sovrapposti e di lettura/scrittura fra code
- [ ] OPT-4.14 **[CANDIDATO]** **Lifetimes precise fra code e frame**: sostituire la vita conservativa di tutto il frame per le risorse async con un modello di ordinamento parziale derivato da dipendenze ed eventi; aliasing solo dopo tutti gli ultimi accessi ordinati, riuso fra frame/slot coerente con F8.6. Confronto con il piano conservativo, test dell'ordinamento e stress di riuso con più frame in volo; ridurre le attese solo dove la correttezza è dimostrata
- [ ] OPT-4.15 **[CANDIDATO]** **Piani ottimizzati sul frame reale F5–F8**: applicare ricerca e selezione misurata di OPT-1 a culling, Hi-Z, visibility, resolve e ricostruzione; scegliere ordine, fusioni, rematerializzazione e code sui dispositivi disponibili. Chiave strutturale più ambito di validità misurato (chip/famiglia, OS, preset, risoluzione e classe di carico); invalidazione e ripiego conservativo registrati. Controllare equivalenza per i riordini esatti e soglie F8.7 per le varianti approssimate; estendere i piani a ogni nuova era senza assumere che il modello di costo basti
- [ ] OPT-4.16 **[BASELINE]** **Baseline di scene rappresentative**: oltre ai microbenchmark e agli scenari sintetici, percorsi di camera ripetibili in scene reali dense con materiali eterogenei e oggetti in movimento; estenderli con luci dinamiche, trasparenze, vegetazione e streaming quando disponibili. Registrare tempo del frame e per pass, p50/p95/p99, memoria di picco, banda (misurata o stimata esplicitamente), energia e latenza; confronto a parità di qualità e in regime termico stabile, su T0 e T2 (O12), riusando il corpus in F29/F35
- [ ] OPT-4.17 **[CANDIDATO]** **Compilazione algebrica del frame [EDGE]**: spike offline con e-graph/equality saturation [R100] e rematerializzazione [R101], regole esatte con precondizioni e varianti approssimate separate; estrazione Pareto sotto vincoli di tile/heap/dipendenze, confronto con DP/annealing OPT-1, cap di ricerca e adozione misurata (ricerca H4)

---

# Era III — Luce

## F9 — Infrastruttura ray tracing [CORE]

- [ ] F9.1 BLAS per mesh con build, compaction e refit nel `MTL4ComputeCommandEncoder`
- [ ] F9.2 TLAS per frame con istanze filtrate dalla GPU scene; build guidate da indirizzo (Apple9)
- [ ] F9.3 **Geometria proxy per l'RT** (LOD semplificati) separata dal raster: niente RT contro geometria a piena densità
- [ ] F9.4 Libreria di traversal con `intersector` (non `intersection_query`, che disabilita il reorder hardware)
- [ ] F9.5 Intersection function buffer per alpha test nell'RT; su M5 indicizzazione hardware
- [ ] F9.6 **[CANDIDATO]** **Spike [EDGE]**: ordinamento software dei raggi per coerenza (binning per direzione/materiale) come sostituto dello shader execution reordering assente in Metal

**Uscita**: TLAS da 100K istanze aggiornato entro 0,5 ms su T2; costo per raggio misurato per tier.

## F10 — Ombre ibride [AAA]

- [ ] F10.1 CSM a 4 cascate renderizzate con mesh shader, stabilizzate, PCSS (fallback T0)
- [ ] F10.2 Ombre RT del sole con penombra fisica (1 raggio + denoise)
- [ ] F10.3 Ombre RT per luci locali integrate con ReSTIR
- [ ] F10.4 Contact shadow in screen space per i dettagli fini
- [ ] F10.5 Cache delle ombre statiche (aggiornamento solo delle regioni cambiate)

## F11 — Migliaia di luci [AAA]

- [ ] F11.1 ReSTIR DI (port da `legacy/vulkan/shaders/lighting/`): candidati, riuso temporale e spaziale, shading
- [ ] F11.2 Luci ad area (rettangoli, dischi, tubi) e mesh emissive campionate
- [ ] F11.3 Light BVH / alias table per il campionamento dei candidati
- [ ] F11.4 Rumore blu spazio-temporale (STBN) per tutte le decisioni stocastiche
- [ ] F11.5 Percorso T0: cluster di luci 3D + ReSTIR ridotto

**Uscita**: Many Lights (1.024 luci) stabile dopo il denoiser su T1; T0 ≥ 30 fps.

## F12 — Illuminazione globale [AAA]

- [ ] F12.1 DDGI in compute (port da `legacy/vulkan/shaders/gi/` con `intersector`): trace, atlanti, classificazione e riallocazione delle sonde
- [ ] F12.2 **Radiance cache a hash spaziale** (stile SHaRC) come cache unica per GI e riflessioni ruvide
- [ ] F12.3 ReSTIR GI su T2+ sopra la radiance cache
- [ ] F12.4 Superfici emissive che illuminano la scena tramite la cache
- [ ] F12.5 Confronto con riferimento (F32.1 quando disponibile, prima renderer offline esterno)

**Uscita**: Cornell Box fisicamente plausibile; Sponza con sole dinamico senza leak; budget GI rispettato.

## F13 — Riflessioni, AO e denoiser [AAA]

- [ ] F13.1 Riflessioni RT per roughness bassa, fallback su radiance cache e SSR (T0)
- [ ] F13.2 RTAO (T1+) / GTAO (T0)
- [ ] F13.3 **MetalFX denoised upscaler** (Apple9) collegato ai canali di F7.6
- [ ] F13.4 Denoiser custom leggero per T0 (temporal + atrous) quando il denoiser MetalFX non è usato
- [ ] F13.5 Probe di riflessione statiche filtrate per T0
- [ ] F13.6 **Qualità temporale di luce e denoise**: estendere F8.7 con luci ed emissive in movimento, variazioni di GI, riflessi a diverse roughness e superfici appena disoccluse; verificare motion vector, rejection/invalidation della storia e recupero dopo camera cut, con confronto al riferimento di F12.5 e soglie per tier

## F14 — Cielo, atmosfera, nuvole, meteo [AAA]

- [ ] F14.1 Atmosfera fisica (Hillaire): LUT di trasmittanza, multi-scattering, sky-view
- [ ] F14.2 Nebbia volumetrica froxel con scattering di luci e ombre, integrata con la GI
- [ ] F14.3 **Nuvole volumetriche** raymarched (approccio stile Nubis) con ricostruzione temporale a bassa risoluzione
- [ ] F14.4 Ciclo giorno/notte, luna e stelle
- [ ] F14.5 **[CANDIDATO]** **[EDGE]** Meteo dinamico: pioggia (particelle + superfici bagnate), neve con accumulo, fulmini come luci ReSTIR

---

## OPT-5 — Budget di raggi e campionamento [OPT]

**Obiettivo (ipotesi)**: stessa qualità dopo il denoiser con −40% di raggi
per frame rispetto alla fine di F14.

**Letture**: [R23]–[R36] [R42] [R43]; playbook S-RT-*, S-NA-4.

**Direzioni di ricerca**
- [ ] OPT-5.1 **[CANDIDATO]** **ReSTIR architettato per la produzione** [R23]: reservoir compatti in `half`, accessi coerenti; scelta dei vicini guidata dalla compatibilità [R28]; mappe di shift GRIS [R24] e ReSTIR condizionale [R25]
- [ ] OPT-5.2 **[CANDIDATO]** **Reservoir splatting** [R27] e Area ReSTIR [R26] per un riuso temporale più robusto a costo minore (anche antialiasing e depth of field "gratis")
- [ ] OPT-5.3 **[CANDIDATO]** **Variable Rate Ray Tracing** [R32]: raggi per pixel decisi dinamicamente da varianza, disocclusione e contenuto
- [ ] OPT-5.4 **[CANDIDATO]** **Campionamento delle luci più intelligente**: albero di luci con Spherical Gaussian [R29], adaptive tree splitting [R30], stochastic lightcuts [R31]: candidati migliori, meno raggi d'ombra
- [ ] OPT-5.5 **[CANDIDATO]** **Coerenza senza SER**: ordinamento software dei raggi per direzione/origine prima del trace [R33], misurato contro il reorder hardware di M3+
- [ ] OPT-5.6 **[CANDIDATO]** **Qualità della TLAS**: re-braiding [R34] e unione offline delle istanze statiche piccole; BVH compatte a nodi fusi [R35] per le strutture software (proxy, audio, splat); tecniche per geometria animata massiva [R36]
- [ ] OPT-5.7 **[CANDIDATO]** **Rumore adattato al filtro**: blue noise spazio-temporale [R42] e FAST [R43] per tutte le decisioni stocastiche; stesso numero di campioni, meno rumore residuo

**Spremitura del SoC**
- [ ] OPT-5.8 **[CANDIDATO]** `intersector` in tutti i kernel caldi, zero `intersection_query`; payload minimi; intersection function brevi (S-RT-1)
- [ ] OPT-5.9 **[CANDIDATO]** Percorso M5: molte istanze piccole (istanze HW, allineamento 1 KB); percorso M3/M4: BLAS unite (S-RT-2)
- [ ] OPT-5.10 **[CANDIDATO]** Build/refit/compaction ammortizzati su più frame secondo B-21 (S-RT-4)
- [ ] OPT-5.11 **[CANDIDATO]** Raggi/s coerenti e incoerenti per chip nel modello di costo; budget di raggi per tier derivato dai numeri (S-RT-3)

## OPT-6 — Illuminazione globale, cache e ammortamento [OPT]

**Obiettivo (ipotesi)**: GI + riflessioni + atmosfera −35% di tempo a qualità
invariata; frame time piatto (nessun picco da aggiornamenti ammortizzati).

**Letture**: [R37]–[R41] [R44]–[R48]; playbook S-MEM-2, S-SYNC-2, S-ALU-3.

**Direzioni di ricerca**
- [ ] OPT-6.1 **[CANDIDATO]** **Cache di radianza a due livelli** [R37] e hash spaziale jittered [R38], confrontate con cache sulle superfici [R48]
- [ ] OPT-6.2 **[CANDIDATO]** **Spike Radiance Cascades / Split Radiance Cascades** [R39]: costo costante indipendente dalla complessità della scena, probe sparse in hashmap
- [ ] OPT-6.3 **[CANDIDATO]** **Cache ORCA** [R40] per accelerare il path tracing (T3) senza dipendere dalla storia temporale
- [ ] OPT-6.4 **[CANDIDATO]** DDGI di produzione [R41]: classificazione e riallocazione delle sonde, aggiornamento guidato dalla varianza invece che a rotazione fissa
- [ ] OPT-6.5 **[CANDIDATO]** **Scheduler dei lavori ammortizzati** (idea Phosphor): GI, ombre statiche, LUT di atmosfera, BVH, streaming aggiornati a frequenze diverse da uno scheduler che riempie il budget residuo di ogni frame, per un frame time piatto
- [ ] OPT-6.6 **[CANDIDATO]** Volumetrici, nuvole ed effetti a bassa risoluzione con ricostruzione temporale [R45][R46][R47]
- [ ] OPT-6.7 **[CANDIDATO]** Denoiser: SVGF [R44] e varianti per T0, confronto con il denoiser MetalFX su costo e qualità

**Spremitura del SoC**
- [ ] OPT-6.8 **[CANDIDATO]** Atlanti di sonde, cache e reservoir in `half` o formati compatti; dimensioni tarate per restare nella SLC (S-ALU-3, S-MEM-2)
- [ ] OPT-6.9 **[CANDIDATO]** GI e build di BVH sulla seconda coda compute, sovrapposti al raster (S-SYNC-2), solo se migliorano il frame completo: misurare contesa di banda/cache, attese e memoria di picco con le lifetimes di OPT-4.14; aggiornare e riselezionare i piani reali di OPT-4.15 dopo F9–F14
- [ ] OPT-6.10 **[CANDIDATO]** Depth bounds test su M5 per volumi di luce e nebbia (S-GEO-5)
- [ ] OPT-6.11 **[CANDIDATO]** Compressione disattivata sugli atlanti ad accesso sparso se il Compression Ratio lo indica (S-TEX-2)
- [ ] OPT-6.12 **[CANDIDATO]** **Cache temporali con budget d'errore [EDGE]**: sonde, ombre e shading con età massima, invalidazione e priorità per riduzione attesa dell'errore per unità di tempo; confronto con aggiornamento fisso/per varianza, camera cut e luce improvvisa; preservare i requisiti statistici dei reservoir ReSTIR (H8, F8.7/F13.6)

---

# Era IV — Mondo

## F15 — Materiali avanzati e varianti shader [AAA]

- [ ] F15.1 Modello esteso: clearcoat, sheen, anisotropia, trasmissione, iridescenza, subsurface (estensioni glTF `KHR_materials_*`)
- [ ] F15.2 Descrizione dati dei materiali → generatore di permutazioni MSL offline (O11), niente uber-shader
- [ ] F15.3 Testbench materiali confrontato con i modelli di riferimento Khronos
- [ ] F15.4 Layered material (blend di strati) per terreni e superfici complesse

## F16 — Trasparenze on-tile, particelle, VFX [AAA]

- [ ] F16.1 **OIT on-tile** con imageblock + raster order group (strumenti TBDR che il desktop non ha)
- [ ] F16.2 Vetro e liquidi con rifrazione approssimata e assorbimento
- [ ] F16.3 Particelle GPU: simulazione in compute, rendering con mesh shader, collisione con la depth, luce dalla radiance cache
- [ ] F16.4 Decal deferred sul visibility buffer
- [ ] F16.5 **[CANDIDATO]** **[EDGE]** Fumo e fuoco volumetrici simulati (griglia sparse in compute)

## F17 — Post-processing cinematografico [AAA]

- [ ] F17.1 Bloom fisico basato sull'energia, lens dirt
- [ ] F17.2 Depth of field con bokeh fisico
- [ ] F17.3 Motion blur per oggetto e camera
- [ ] F17.4 Color grading con LUT 3D, film grain, aberrazione cromatica, vignetta
- [ ] F17.5 Pass fusi in un unico compute dove possibile (O1)

**Uscita**: post completo entro 2 ms su T0 incluso MetalFX.

## F18 — Geometria virtualizzata [AAA]

- [ ] F18.1 DAG di cluster LOD offline con `clusterlod.h` (meshoptimizer ≥ 1.3)
- [ ] F18.2 Selezione LOD per livelli con dispatch indiretti (**no persistent thread**: Metal non garantisce il forward progress)
- [ ] F18.3 **Raster software** dei micro-triangoli in compute con atomici a 64 bit (Apple9), fuso con il raster hardware nel V-buffer; misurare l'impatto sull'HSR del TBDR
- [ ] F18.4 Compressione dei cluster (codec meshlet di meshoptimizer)
- [ ] F18.5 Ombre compatibili (CSM e RT contro proxy)
- [ ] F18.6 Testbench "Mega Geometry" (≥ 100M triangoli)

**Uscita**: Mega Geometry 60 fps su T2, ≥ 30 su T0; niente popping (errore < 1 pixel).

## F19 — Terreno, vegetazione, mondo procedurale GPU [AAA]

- [ ] F19.1 Terreno a clipmap con heightfield virtuale in streaming
- [ ] F19.2 Texturing del terreno con virtual texture e materiali a strati
- [ ] F19.3 Piazzamento procedurale di vegetazione e rocce in compute (regole + densità)
- [ ] F19.4 Vegetazione animata dal vento, impostor ottaedrici per la distanza
- [ ] F19.5 **[CANDIDATO]** **[EDGE]** Generazione procedurale di interi biomi sulla GPU a runtime, deterministica per seed

## F20 — Acqua, oceano e fluidi [EDGE]

- [ ] F20.1 **[CANDIDATO]** Oceano FFT in compute (cascate multiple), schiuma, interazione con gli oggetti
- [ ] F20.2 **[CANDIDATO]** Caustiche in ray tracing
- [ ] F20.3 **[CANDIDATO]** Rendering subacqueo con volumetrici
- [ ] F20.4 **[CANDIDATO]** Fluidi locali (FLIP/SPH in compute) per fiumi, cascate, schizzi

## F21 — Asset pipeline e cooker [CORE]

**Requisito di prodotto:** contenuti dei workflow e delle comunità, attraverso
importer riusati o conversioni esplicite. Copertura estendibile e dichiarata,
con test per feature del formato, senza perdita silenziosa di contenuto.

- [ ] F21.1 `tools/cooker`: glTF e **OpenUSD** → formato Phosphor, build incrementale per hash
- [ ] F21.2 Texture ASTC (astcenc) e BC7, mip precalcolati, supercompressione per la distribuzione
- [ ] F21.3 Mesh: cluster, DAG LOD, proxy RT e meshlet compressi precalcolati
- [ ] F21.4 Formato pacchetto mappabile in memoria, allineato ai codec MTLIO
- [ ] F21.5 Cooker parallelo sui core del Mac (job system di F23)
- [ ] F21.6 Registro importer/converter con interfaccia estendibile, identificatori stabili, dipendenze e diagnostica; riusare loader Bevy e librerie upstream dove i contratti sono compatibili [R116], evitando parser duplicati
- [ ] F21.7 Matrice di formati della comunità: 3D/DCC (glTF/GLB, USD, Blender e conversioni FBX/OBJ), raster/HDR/EXR/container GPU, audio e font; per ogni feature indicare diretto, conversione, subset o gap, con fixture e roundtrip/resa confrontati
- [ ] F21.8 Workflow 2D: spritesheet/atlanti, animazioni Aseprite, Tiled/LDtk e font; coordinate, layer, animazioni, collisioni e metadati preservati tramite adapter o conversioni [R121]
- [ ] F21.9 Asset/scene Bevy per versione e tipi registrati: mapping di handle, materiali, scheletri, morph e animazioni al runtime e al cooker; reimport e hot reload con identità preservate, provenienza e attribuzioni nel manifest

## F22 — Streaming [CORE]

- [ ] F22.1 **Prototipo** MTLIO + risorse placement sparse per chiarire la sincronizzazione con le code MTL4 (lacuna aperta nel report)
- [ ] F22.2 Streaming a priorità dalla camera, budget per tier, eviction LRU
- [ ] F22.3 Virtual texturing con texture sparse (pagine 16/64 KB) e feedback buffer
- [ ] F22.4 Streaming delle pagine di cluster di F18
- [ ] F22.5 Streaming predittivo basato su velocità e direzione della camera
- [ ] F22.6 Contratto di ownership I/O→CPU→GPU: generazioni delle pagine, completamento delle letture/decompressioni, pubblicazione atomica della tabella, retirement dopo l'ultimo lettore e backpressure; traccia dei byte copiati/convertiti oltre ai byte letti (H3)

**Uscita**: volo in una scena più grande della RAM su T0 (16 GB) senza hitch > 33 ms.

---

## OPT-7 — Geometria virtualizzata e mondo [OPT]

**Obiettivo (ipotesi)**: −30% di tempo per la geometria virtualizzata e il
terreno, −30% di memoria residente dei cluster.

**Letture**: [R49]–[R58]; playbook S-GEO-*, S-SIMD-4, S-TBDR-3, S-TBDR-7.

**Direzioni di ricerca**
- [ ] OPT-7.1 **[CANDIDATO]** **Soglia raster software/hardware** misurata per chip [R51][R52]; raster software dedicato per fili e capelli [R53]
- [ ] OPT-7.2 **[CANDIDATO]** DAG di cluster con **errore percettivo** (non solo geometrico) e build parallelo veloce [R50]; streaming ordinato per errore [R49]
- [ ] OPT-7.3 **[CANDIDATO]** **Terreno con concurrent binary tree** usato come pool di memoria [R54] e tessellation adattiva invece di clipmap fisse
- [ ] OPT-7.4 **[CANDIDATO]** **Vegetazione massiva in ray tracing** con le tecniche di [R55]; impostor per la distanza
- [ ] OPT-7.5 **[CANDIDATO]** **Acqua con Water Surface Wavelets** [R56] dove serve interazione locale, FFT solo per l'oceano aperto
- [ ] OPT-7.6 **[CANDIDATO]** **Trasparenze nella tile**: MLAB con raster order group [R57] per il vetro, OIT a momenti [R58] per le particelle

**Spremitura del SoC**
- [ ] OPT-7.7 **[CANDIDATO]** Raster software con `atomic_max` a 64 bit (Apple9) e atomici gerarchici; contesa misurata con B-07 (S-SIMD-4)
- [ ] OPT-7.8 **[CANDIDATO]** Raster software solo dove non rompe l'HSR del TBDR (misura B-13) (S-TBDR-2)
- [ ] OPT-7.9 **[CANDIDATO]** OIT e decal interamente in tile memory con ROG, nessun atomico in device memory (S-TBDR-7)
- [ ] OPT-7.10 **[CANDIDATO]** Soglie di LOD per generazione (M5 con geometria 2x) (S-GEO-2)

## OPT-8 — Streaming, texture e materiali [OPT]

**Obiettivo (ipotesi)**: −40% di dimensione su disco, −30% di I/O per
secondo di gioco, zero hitch da streaming su T0 con 16 GB.

**Letture**: [R59]–[R63]; playbook S-IO-*, S-TEX-*, S-MEM-*.

**Direzioni di ricerca**
- [ ] OPT-8.1 **[CANDIDATO]** **Texture supercompresse decodificate dalla GPU** [R59] e compressione neurale a blocchi [R60]: meno disco e I/O senza cambiare gli shader
- [ ] OPT-8.2 **[CANDIDATO]** **Virtual texture adattiva** [R61] con feedback a bassa risoluzione e decompressione in compute con SIMD-group
- [ ] OPT-8.3 **[CANDIDATO]** **Prefiltraggio delle normali e specular antialiasing** [R62]: meno aliasing speculare → meno bisogno di supersampling e di risoluzione interna alta
- [ ] OPT-8.4 **[CANDIDATO]** **Rappresentazioni per dispositivo generate offline** (come SLIM [R63]): il cooker produce varianti di asset per T0…T3, non un solo asset scalato a runtime
- [ ] OPT-8.5 **[CANDIDATO]** **Streaming predittivo** (idea Phosphor): previsione della traiettoria della camera (modello piccolo su ANE) per anticipare le richieste

**Spremitura del SoC**
- [ ] OPT-8.6 **[CANDIDATO]** Codec MTLIO scelto per tipo di dato in base a B-25; richieste grandi e allineate (S-IO-2)
- [ ] OPT-8.7 **[CANDIDATO]** Budget di streaming per tier e per velocità del disco rilevata (S-IO-1)
- [ ] OPT-8.8 **[CANDIDATO]** Pagine sparse da 16 o 64 KB scelte per tipo di risorsa; costo di mapping misurato (S-TEX-4)
- [ ] OPT-8.9 **[CANDIDATO]** Residency set aggiornati in modo incrementale e in batch (S-MEM-5)
- [ ] OPT-8.10 **[CANDIDATO]** Streaming in QoS utility, con concorrenza e granularità misurate per non disturbare render e simulazione; verificare il placement osservato senza assumere pinning sugli E-core (S-CPU-1, R87)
- [ ] OPT-8.11 **[CANDIDATO]** **Scelta congiunta di compressione e layout [EDGE]**: costo end-to-end da disco al campione shader per ASTC/BC, decode-on-load e decode-on-sample neurale quando F30 è disponibile; includere packing, copie, cache miss e filtraggio; asset classici come baseline (H3/H9, R103)

---

# Era V — Simulazione e runtime (linea parallela)

## F23 — CPU ultra-ottimizzata [CORE]

- [ ] F23.1 Job system (enkiTS come baseline) con classi QoS per urgenza, pool persistenti e granularità misurata; verificare scheduling e migrazioni con Instruments, senza assumere un mapping rigido QoS→tipo di core [R87–R88]
- [ ] F23.2 Arena per frame, allocatori lineari, niente `new` nel ciclo caldo (O7)
- [ ] F23.3 Layout data-oriented (SoA) per ECS, transform, culling CPU residuo
- [ ] F23.4 NEON esplicito nei cicli caldi; Accelerate/BNNS come baseline matriciale e spike SME custom sulle CPU che lo espongono, con verifica feature/ABI e fallback; distinguere SME dall'AMX proprietario e dall'ANE [R95–R96]
- [ ] F23.5 Pipelining CPU/GPU: simulazione del frame N+1 mentre la GPU disegna il frame N
- [ ] F23.6 Latenza input→fotoni misurata e minimizzata (`CAMetalDisplayLink`, present timing)
- [ ] F23.7 **[CANDIDATO]** **Scheduler Phosphor a dipendenze e deadline [EDGE]**: contratto del job con implementazioni CPU/GPU/ANE ammesse, costo stimato, working set ed età massima; replay offline e runtime a continuazioni confrontato con enkiTS/GCD, critical-path scheduling, aging e stealing per batch (H1/H11, R98–R99); integrare gli acceleratori progressivamente quando disponibili
- [ ] F23.8 **Ownership e layout per memoria unificata**: descrittori produttore/consumatore e generazioni, SoA/AoSoA, false sharing, packing e conversioni misurati; riuso senza copia solo con lifetime, formato e sincronizzazione compatibili (H3)
- [ ] F23.9 **[CANDIDATO]** **Laboratorio macOS/XNU [EDGE]**: mappa delle API pubbliche pthread/Mach/QoS/workgroup disponibili, studio Clutch/Edge a revisione fissata, trace e simulatore delle politiche; dossier di fattibilità per eventuale prototipo kernel solo dopo un limite OS riproducibile. Specificare accessi richiesti, compatibilità driver/distribuzione e costo di mantenimento; nessuna sostituzione dello scheduler macOS presunta (H11, R88–R89/R97)

## F24 — Fisica, cloth e distruzione [AAA]

- [ ] F24.1 Jolt Physics: corpi rigidi, character controller, raycast, trigger, debug draw
- [ ] F24.2 Verificare e fissare una release Jolt con backend compute Metal [R107], censire solver e feature GPU realmente disponibili (inclusi capelli); integrare i carichi supportati e mantenere Jolt CPU per rigid-body non coperti, senza assumere il port GPU dell'intero motore fisico
- [ ] F24.3 Cloth XPBD in compute con collisioni sulla depth e sulle capsule
- [ ] F24.4 Distruzione con frammenti precalcolati e simulazione GPU dei detriti

## F25 — Animazione e personaggi [AAA]

- [ ] F25.1 ozz-animation + ACL (compressione), blend tree, IK
- [ ] F25.2 Skinning in compute con output diretto ai buffer della GPU scene; BLAS refit per i personaggi
- [ ] F25.3 **Capelli a filamenti** renderizzati con mesh shader e simulati in compute
- [ ] F25.4 **[CANDIDATO]** **[EDGE]** Motion matching, poi versione appresa (learned motion matching con rete piccola in tensori Metal)
- [ ] F25.5 Rendering della pelle (subsurface), occhi, denti

## F26 — Audio di gioco e acustica avanzata [CORE]

- [ ] F26.1 Audio di gioco completo: playback, mixer, streaming, volume e lifecycle; valutare riuso Bevy/audio community e adapter Apple (AVAudioEngine/PHASE) mantenendo un solo proprietario del backend e del clock
- [ ] F26.2 **[CANDIDATO]** **Acustica in ray tracing** sulla stessa BVH del renderer: occlusione, riverbero e propagazione calcolati sulla GPU
- [ ] F26.3 **[CANDIDATO]** Audio spaziale personalizzato (AirPods, head tracking)

## F27 — Runtime di gioco ed ECS con riuso Bevy [CORE]

- [ ] F27.1 Scelta ECS con priorità al riuso diretto di `bevy_ecs` e `bevy_app`/reflection/moduli necessari, affiancati al renderer C++ Metal; prova di integrazione contro ECS corrente. Alternativa C++ deliberatamente ispirata a Bevy, anche su EnTT/flecs, solo con motivazione e costo di compatibilità espliciti; non promettere plugin Rust compatibili con un clone [R112–R113]
- [ ] F27.2 Input: GameController (controller, haptics), mappatura azioni rimappabile, tastiera/mouse/trackpad
- [ ] F27.3 Scripting Luau con binding all'ECS e hot reload
- [ ] F27.4 Serializzazione di scene e salvataggi, iCloud opzionale
- [ ] F27.5 Rete (GameNetworkingSockets) per multiplayer, opzionale
- [ ] F27.6 Determinismo della simulazione (prerequisito per replay e test in F35)
- [ ] F27.7 Prototipo del confine Rust/C++: snapshot e delta in batch verso GPU scene F5, entity handle con generazioni, ownership/retirement, errori e shutdown; un solo mondo autorevole, clock/event loop e gestione dei pool espliciti, senza chiamate FFI per ogni componente nel ciclo caldo
- [ ] F27.8 Parità dei contratti ECS necessari: query/filtri, risorse, change detection, comandi differiti, schedule/order, stati, eventi/messaggi/observer, gerarchie e reflection/serializzazione; riusare la semantica upstream e verificarla con fixture prima dell'editor
- [ ] F27.9 Host modulare Bevy con versione/feature fissate: asset, scene, time/input/transform/animazione e servizi richiesti dai plugin; censire e adattare le dipendenze transitive dal renderer, evitando due window loop o due scheduler in competizione
- [ ] F27.10 Prova end-to-end con plugin di logica, asset e input, entity lifecycle e extraction del frame; misurare correttezza, costo del bridge e manutenzione, poi registrare la scelta di integrazione prima della migrazione del runtime

---

## OPT-9 — CPU, thread e memoria unificata [OPT]

**Obiettivo (ipotesi)**: tempo CPU del frame −30%, latenza input→fotoni −20%.

**Letture**: [R64] [R74]; playbook S-CPU-*, S-ANE-*, S-MEM-4, S-SYNC-4.

**Direzioni di ricerca**
- [ ] OPT-9.1 **[CANDIDATO]** Job system a fiber [R64] confrontato con enkiTS sulla topologia reale (super/performance/efficiency core)
- [ ] OPT-9.2 **[CANDIDATO]** **Bilanciamento adattivo CPU↔GPU** (idea Phosphor): lavori leggeri (culling di luci, selezione LOD, animazione) spostati tra CPU e GPU sulla base del costo completo di lancio, layout, copie necessarie, sincronizzazione e contesa; isteresi e baseline fissa (H1/H3)
- [ ] OPT-9.3 **[CANDIDATO]** **Neural Engine per le reti fuori dal frame** (animazione appresa, audio, IA, previsione dello streaming) per lasciare liberi GPU e CPU
- [ ] OPT-9.4 **[CANDIDATO]** **Extrapolazione del frame** [R74] come alternativa a bassa latenza all'interpolazione MetalFX

**Spremitura del SoC**
- [ ] OPT-9.5 **[CANDIDATO]** Classi QoS verificate con Instruments per ogni thread; nessuno spin-wait (S-CPU-1, S-PWR-3)
- [ ] OPT-9.6 **[CANDIDATO]** Cicli caldi in NEON (SoA); operazioni su matrici in batch via Accelerate/SME (S-CPU-2, S-CPU-3)
- [ ] OPT-9.7 **[CANDIDATO]** Contesa di banda CPU/GPU misurata (B-09) e job CPU pesanti pianificati fuori dalle finestre critiche della GPU (S-MEM-4)
- [ ] OPT-9.8 **[CANDIDATO]** Frame in volo scelti dalla latenza misurata (B-28) (S-SYNC-4)
- [ ] OPT-9.9 **[CANDIDATO]** **Matrice di interferenza del SoC**: estendere B-09/B-23/B-24 con coppie/triplette CPU NEON/SME, GPU raster/compute, ANE e I/O; sweep del working set, granularità e concorrenza, tempi p50/p95/p99 e potenza; modello per chip/OS con incertezza (H2)
- [ ] OPT-9.10 **[CANDIDATO]** **Scheduling a budget di banda e deadline [EDGE]**: ammissione dei job memory-bound, earliest-finish con interferenza, aging dei lavori differibili; confronto con FIFO/work stealing e serializzazione sul corpus reale, inclusi costo dello scheduler e starvation (H1/H2)
- [ ] OPT-9.11 **[CANDIDATO]** **Compilatore di layout [EDGE]**: scegliere SoA/AoSoA, packing e buffer condivisi/doppi per classe di dati; confronto produzione→consumo e verifica numerica, retirement ed eviction sotto pressione (H3)
- [ ] OPT-9.12 **[CANDIDATO]** **Kernel SME generati per shape reali [EDGE]**: NEON vs Accelerate/BNNS vs SME vs GPU, con packing e transizioni ABI inclusi; varianti offline solo dove il guadagno end-to-end è misurato (H7, R95–R96/R109)
- [ ] OPT-9.13 **[CANDIDATO]** **Esperimento scheduler/kernel condizionale [EDGE]**: implementare e misurare le politiche di F23.9 nel simulatore e nel runtime; se il dossier dimostra fattibilità e vantaggio residuo, progettare e prototipare in un laboratorio OS separato, confrontando con macOS standard. Esito ammesso: adozione user-space, esperimento kernel o esclusione motivata; niente dipendenza kernel implicita per il prodotto (H11)

## OPT-10 — Simulazione [OPT]

**Obiettivo (ipotesi)**: stessa qualità di simulazione con −40% di tempo, o
10x oggetti simulati a parità di tempo.

**Letture**: [R65]–[R68]; playbook S-SIMD-*, S-NA-*, S-ANE-*.

**Direzioni di ricerca**
- [ ] OPT-10.1 **[CANDIDATO]** **Vertex Block Descent** [R65] come solver GPU unico per cloth, corpi morbidi e particelle: più parallelo di XPBD e stabile con poche iterazioni
- [ ] OPT-10.2 **[CANDIDATO]** **Small steps** [R66]: più substep con meno iterazioni
- [ ] OPT-10.3 **[CANDIDATO]** **Learned motion matching** [R67] compresso per ANE o per i Neural Accelerator
- [ ] OPT-10.4 **[CANDIDATO]** **Acustica ibrida**: codifica parametrica precomputata [R68] + ray tracing a runtime solo per le parti dinamiche
- [ ] OPT-10.5 **[CANDIDATO]** Animazione di folle con decompressione in compute e skinning nel mesh shader

**Spremitura del SoC**
- [ ] OPT-10.6 **[CANDIDATO]** Solver GPU con riduzioni SIMD-group e partizionamento in threadgroup memory (S-SIMD-*)
- [ ] OPT-10.7 **[CANDIDATO]** Reti di animazione dimensionate per le tile ≥ 32×32 dei Neural Accelerator o spostate su ANE secondo B-22/B-23 (S-NA-1, S-ANE-1)
- [ ] OPT-10.8 **[CANDIDATO]** Simulazione con dati SoA e NEON, QoS coerente con la deadline e scheduling verificato tramite trace, senza presumere assegnazione fissa ai P-core (S-CPU-2, R87)

---

## Checkpoint WWDC 2027 (giugno 2027)

- [ ] Sessioni Metal, MetalFX, Game Porting Toolkit, release note di OS 27
- [ ] Aggiornare metal-cpp; nuove famiglie GPU (M6?), deprecazioni
- [ ] Valutare nuove funzioni (estensioni RT, eventuale equivalente dei work graph, nuovi tipi di tensori) e riconciliare i piani dettagliati F28–F41 quando attivati, mantenendo i requisiti funzionali del prodotto
- [ ] Aggiornare report e roadmap

---

# Era VI — Frontiera

## F28 — Scalabilità e presentazione [CORE]

- [ ] F28.1 Rilevamento famiglia GPU, core, memoria → tier; preset per modello di Mac ("For This Mac"), override utente
- [ ] F28.2 Risoluzione dinamica guidata dai tempi GPU
- [ ] F28.3 **Rasterization rate map** (shading a frequenza variabile nativo di Metal) per bordi, motion blur e foveazione
- [ ] F28.4 MetalFX frame interpolation con present thread dedicato; ingresso ≥ 30 fps
- [ ] F28.5 ProMotion 120 Hz, VRR, frame pacing a 2 bucket
- [ ] F28.6 Termica ed energia: riduzione progressiva della qualità, modalità batteria (O10), Game Mode
- [ ] F28.7 Matrice di compatibilità verificata su T0–T3 reali

## F29 — Autotuning per dispositivo [EDGE]

**Idea**: un engine che si ottimizza da solo su ogni Mac.

- [ ] F29.1 **[CANDIDATO]** Parametri esposti: dimensioni dei threadgroup, tile, numero di sonde, dimensione dei meshlet, varianti di shader e piani del render graph (OPT-4.15); percorsi consentiti dalle capacità Metal rilevate, poi selezione per chip misurato, senza assumere un vincitore universale fra on-tile, compute e ibrido
- [ ] F29.2 **[CANDIDATO]** Benchmark automatico al primo avvio (o in background) che cerca la configurazione migliore per il dispositivo, sul corpus reale di OPT-4.16 e con soglie di qualità temporale F8.7/F13.6; scelta sul frame completo, non sulla sola occupancy o sul throughput di un kernel
- [ ] F29.3 **[CANDIDATO]** Risultati salvati per modello di chip e versione di OS, condivisibili tramite telemetria opzionale; includere versione del renderer/shader, capacità, preset, risoluzione, classe di carico e regime energetico, con invalidazione e ripiego quando il profilo non è applicabile
- [ ] F29.4 **[CANDIDATO]** Uso di Claude Code in locale come "perf engineer" automatico: cattura → analisi dei contatori → proposta di patch → misura
- [ ] F29.5 **[CANDIDATO]** **Selezione sotto budget**: mantenere configurazioni non dominate per qualità, tempo del frame, memoria, energia e latenza, selezionate secondo preset e condizioni di F28; cambi con isteresi e senza compilazione/I/O bloccante nel frame. Fino a F29 usare i piani offline misurati di OPT-4.15

## F30 — Rendering neurale I [EDGE]

- [ ] F30.1 **[CANDIDATO]** Infrastruttura: `MTLTensor`, TensorOps / Metal Performance Primitives negli shader, encoder ML per reti Core ML; microbenchmark reali (GEMM ≥ 32×32 per il picco), formati quantizzati con scale factor dove supportati [R111]; inventario dei limiti di dtype/shape e confronto sulle shape del motore
- [ ] F30.2 **[CANDIDATO]** **Radiance cache neurale** addestrata online negli shader (evoluzione di F12.2)
- [ ] F30.3 **[CANDIDATO]** **Compressione neurale delle texture**: "on load" (transcodifica ad ASTC/BC, tutti i tier) poi "on sample" su T3
- [ ] F30.4 **[CANDIDATO]** Upscaler MetalFX neurale (WWDC26) su M5 Pro/Max
- [ ] F30.5 **[CANDIDATO]** Ogni funzione con fallback, interruttore nei preset e guadagno misurato
- [ ] F30.6 **[CANDIDATO]** **Runtime Core ML asincrono per ANE**: adapter distinto dal ML Metal, modelli precompilati, piano stimato più riscontro runtime del dispositivo, snapshot versionati, output con età/confidenza e fallback senza attesa del frame; misurare conversioni, copie e contesa col renderer (H6, R93–R94)
- [ ] F30.7 **[CANDIDATO]** **Laboratorio ANE diretto [EDGE]**: riprodurre kernel/modelli piccoli dai preprint [R105–R106] in un eseguibile separato; API private, requisiti e dipendenza da OS dichiarati, confronto con Core ML/Metal e correttezza numerica; risultati di ricerca senza rendere il percorso privato requisito del prodotto (H12)

## F31 — Rendering neurale II: reti addestrate in casa [EDGE]

**Idea**: il Mac di sviluppo da 128 GB diventa la workstation di training con **MLX**.

- [ ] F31.1 **[CANDIDATO]** Pipeline di dati: l'engine esporta coppie (input rumoroso, riferimento path traced da F32.1)
- [ ] F31.2 **[CANDIDATO]** Denoiser e/o upscaler proprietari addestrati con MLX, esportati in tensori Metal
- [ ] F31.3 **[CANDIDATO]** **Materiali neurali** per materiali stratificati complessi
- [ ] F31.4 **[CANDIDATO]** **LOD e impostor neurali** per oggetti lontani
- [ ] F31.5 **[CANDIDATO]** Esposizione e tonemapping appresi dal giudizio estetico (dataset curato)

## F32 — Path tracing in tempo reale [EDGE]

- [ ] F32.1 Path tracer di riferimento (accumulo progressivo) per validare GI, riflessioni e materiali
- [ ] F32.2 **[CANDIDATO]** ReSTIR PT in tempo reale con terminazione nella radiance cache
- [ ] F32.3 **[CANDIDATO]** Denoise MetalFX a 1 spp o denoiser di F31.2
- [ ] F32.4 **[CANDIDATO]** Modalità foto con accumulo ad alta qualità

**Uscita**: modalità PT a 30 fps su M5 Max con upscaling + frame interpolation.

## F33 — Gaussian splatting ibrido [EDGE]

- [ ] F33.1 **[CANDIDATO]** Rendering di 3D Gaussian splatting ordinato in compute con TBDR
- [ ] F33.2 **[CANDIDATO]** Integrazione nel visibility buffer e nella depth: splat e mesh nella stessa scena
- [ ] F33.3 **[CANDIDATO]** Illuminazione degli splat dalla radiance cache (rilluminazione approssimata)
- [ ] F33.4 **[CANDIDATO]** Pipeline di import da cattura fotogrammetrica (anche da iPhone)

---

## OPT-11 — Rendering neurale [OPT]

**Obiettivo (ipotesi)**: ogni rete nel frame sopra il 50% di utilizzo dei
Neural Accelerator e sotto il suo budget in ms; qualità pari o superiore al
fallback non neurale.

**Letture**: [R69]–[R73] [R75] [R76]; playbook S-NA-*, S-ANE-*.

**Direzioni di ricerca**
- [ ] OPT-11.1 **[CANDIDATO]** **MLP fully fused** in threadgroup memory con cooperative tensor, ispirati a [R69][R70]: strati consecutivi senza passare dalla DRAM
- [ ] OPT-11.2 **[CANDIDATO]** **Quantizzazione** INT8/FP8 (quando l'OS lo consente) per tutte le reti del frame, con fallback FP16
- [ ] OPT-11.3 **[CANDIDATO]** **Upscaler ibrido rete + soluzioni in forma chiusa**, sulla linea del PSSR 2026 [R75]: la rete fa solo ciò che le regole analitiche non sanno fare
- [ ] OPT-11.4 **[CANDIDATO]** Compressione neurale delle texture ad accesso casuale [R71] vs a blocchi [R60]; materiali neurali [R72]
- [ ] OPT-11.5 **[CANDIDATO]** **Supersampling neurale proprietario** [R73] addestrato con MLX sui dati di Phosphor (F31)

**Spremitura del SoC**
- [ ] OPT-11.6 **[CANDIDATO]** Tile di GEMM ≥ 32×32, traversal Morton/Hilbert dei threadgroup, barriere ogni poche iterazioni K (S-NA-1, S-NA-3)
- [ ] OPT-11.7 **[CANDIDATO]** Utilizzo dei Neural Accelerator letto in Metal System Trace per ogni rete (S-NA-1)
- [ ] OPT-11.8 **[CANDIDATO]** Tipi di dato per versione di OS (BF16, INT8/INT4, FP8) con fallback (S-NA-2)
- [ ] OPT-11.9 **[CANDIDATO]** Budget ML nel frame e nel SoC: scegliere GPU tensor, CPU o ANE solo fra implementazioni compatibili e misurate, includendo deadline, conversioni e contesa; ridurre o differire il lavoro se nessun percorso rispetta il budget (S-NA-4, S-ANE-1)
- [ ] OPT-11.10 **[CANDIDATO]** ANE con output anticipati e layout ottimizzati: confrontare predittori analitici, Core ML, GPU tensor e risultati del laboratorio F30.7; latenza completa a GPU carica, scadenza/confidenza e costo energetico (H6/H12)
- [ ] OPT-11.11 **[CANDIDATO]** Compressione neurale e piccoli MLP specializzati per materiale: valutazione congiunta con OPT-8.11, cache dei latenti, raggruppamento dei campioni e qualità temporale; accettare solo il punto Pareto misurato (H9)

## OPT-12 — Autotuning, path tracing, splatting [OPT]

**Obiettivo (ipotesi)**: −15% di tempo del frame su ogni chip grazie ai
parametri trovati dall'autotuning; path tracing T3 a 30 fps con metà dei
campioni.

**Letture**: [R77]–[R83] [R40] [R27]; playbook sezione 16 (differenze per generazione).

**Direzioni di ricerca**
- [ ] OPT-12.1 **[CANDIDATO]** **Autotuning come ricerca**: esplorazione dello spazio dei parametri (tile, threadgroup, meshlet, raggi, varianti) con tecniche da compilatori [R77][R78] e ottimizzazione bayesiana [R79], risultati per chip e per versione di OS
- [ ] OPT-12.2 **[CANDIDATO]** **Path guiding in tempo reale** [R80] per il path tracing: meno campioni a parità di rumore
- [ ] OPT-12.3 **[CANDIDATO]** Path tracing con ORCA [R40] e reservoir splatting [R27]
- [ ] OPT-12.4 **[CANDIDATO]** **Splatting senza ordinamento** [R81], adatto a TBDR e iPad; ordinamento stabile [R82] per la qualità; splat nel ray tracing [R83]

**Spremitura del SoC**
- [ ] OPT-12.5 **[CANDIDATO]** L'autotuning parte dal modello di costo di OPT-0 per ridurre lo spazio di ricerca
- [ ] OPT-12.6 **[CANDIDATO]** Parametri separati per famiglia (percorsi di codice) e per chip misurato (valori numerici) (sezione 16)
- [ ] OPT-12.7 **[CANDIDATO]** Splat blending nella tile con imageblock (S-TBDR-1, S-TBDR-7)
- [ ] OPT-12.8 **[CANDIDATO]** Estendere il compilatore algebrico di OPT-4.17 alle varianti CPU/SME/GPU/ANE semanticamente compatibili; candidati offline e piano ibrido con certificato di dipendenze/ownership, nessuna equivalenza approssimata implicita (H4)
- [ ] OPT-12.9 **[CANDIDATO]** **Autotuning robusto [EDGE]**: ricerca bayesiana vincolata [R104] con misure rumorose, scene holdout, budget di esperimenti e verifica del guadagno su chip diversi; evitare sovra-adattamento al testbench (H10)
- [ ] OPT-12.10 **[CANDIDATO]** Confronto piani statici, selezione per classe di scena e controllo a orizzonte breve; costo delle decisioni, isteresi, invalidazione di profili e storia, ripiego su piano misurato quando il modello deriva (H10)

## OPT-13 — Scalabilità ed energia [OPT]

**Obiettivo (ipotesi)**: +25% di autonomia in modalità batteria a qualità
"media"; nessun throttling percepibile dopo 30 minuti su MacBook Air.

**Letture**: playbook S-PWR-*, S-DISP-*.

**Direzioni di ricerca**
- [ ] OPT-13.1 **[CANDIDATO]** **Race-to-idle vs frequenza costante**: quale strategia consuma meno per frame a 60 fps su ciascun chip
- [ ] OPT-13.2 **[CANDIDATO]** **Qualità guidata dall'energia** (idea Phosphor): il preset si adatta ai watt disponibili, non solo ai millisecondi
- [ ] OPT-13.3 **[CANDIDATO]** Frame cap intelligente: fps adattati al contenuto (menu, scene statiche) e al display (ProMotion)

**Spremitura del SoC**
- [ ] OPT-13.4 **[CANDIDATO]** Misure `powermetrics` per ogni preset e ogni chip disponibile (B-27) (S-PWR-1)
- [ ] OPT-13.5 **[CANDIDATO]** Preset basati sul regime termico, non sui primi secondi (S-PWR-2)
- [ ] OPT-13.6 **[CANDIDATO]** Frame pacing a 2 bucket con `CAMetalDisplayLink` (S-DISP-1)
- [ ] OPT-13.7 **[CANDIDATO]** Video in gioco tramite media engine senza copie (S-DISP-3), da verificare
- [ ] OPT-13.8 **[CANDIDATO]** **Controllo predittivo del budget [EDGE]**: stimare pressione termica/banda dalle osservazioni disponibili e selezionare qualità, concorrenza e lavori differibili; confronto con isteresi semplice, soak prolungato e transitori alimentazione/batteria, senza presumere controllo diretto di clock o core (H2/H10)

---

# Era VII — Prodotto

## F34 — Editor e strumenti [CORE]

- [ ] F34.1 Editor di produzione per scene 2D/3D: viewport, outliner, inspector riflessivo, gizmo, selezione e undo/redo; valutare riuso di editor/inspector Bevy e community prima di riscriverli. ImGui/SwiftUI è una scelta di shell, non il limite funzionale dell'editor
- [ ] F34.2 Editor di materiali e grafo VFX
- [ ] F34.3 Browser degli asset collegato al cooker, reimport automatico, hot reload di tutto
- [ ] F34.4 Strumenti di illuminazione: posizionamento sonde, anteprima dei tier
- [ ] F34.5 Authoring 2D e UI: tilemap/sprite, gerarchie di layout, stili e preview di dimensioni/DPI/input; usare gli stessi componenti e rendering di F39/F40 della build finale
- [ ] F34.6 Project manager, template, play/pause/step, separazione mondo editor/gioco, salvataggio e recovery; scene/prefab riusabili, duplicazione e reimport senza rompere riferimenti
- [ ] F34.7 Estensioni dell'editor tramite reflection e contratto plugin F41; integrare inspector, asset tooling e strumenti community compatibili con diagnostica dei gap [R119–R120]
- [ ] F34.8 Workflow completo verificato: importare contenuti community, comporre scena 2D/3D e menu/HUD, modificare materiali/animazioni, provare e produrre una build; editor essenziale intermedio distinto dalla completezza richiesta

## F35 — QA automatizzata [CORE]

- [ ] F35.1 Rendering deterministico dei testbench, **test visivi** (FLIP/PSNR) su runner macOS self-hosted; automatizzare le clip F8.7/F13.6 e la copertura di trasparenze, vegetazione deformata e materiali animati, con metriche temporali e verifica dei camera cut, oltre alle immagini statiche
- [ ] F35.2 **Perf bot**: tempi per pass per commit su T0 e T2, allarme sopra soglia; includere corpus OPT-4.16, distribuzioni del frame time, memoria di picco e prove sostenute, con chip/OS/preset/risoluzione e rumore sperimentale registrati
- [ ] F35.3 Replay deterministico del gameplay (da F27.6) per bug e prestazioni
- [ ] F35.4 Fuzzing del cooker e dei loader, sanitizer in CI
- [ ] F35.5 Validazione shader (`MTL_SHADER_VALIDATION`) in una suite notturna
- [ ] F35.6 Provenienza riproducibile delle misure e delle decisioni: commit, asset hash, chip/OS/SDK, piano, seed, configurazione, stato termico, dati grezzi e controlli negativi; distinguere contatori, stime e risultati esterni, con test di invalidazione dei profili
- [ ] F35.7 Suite di parità/versione con esempi Bevy e fixture community: ECS, asset, 2D, UI/interazione/accessibilità e plugin; confronto comportamentale e visivo su Metal, installazione in progetto pulito e regressioni degli upgrade

## F36 — Piattaforme Apple [AAA]

- [ ] F36.1 iPad M3+: input touch, preset T0, termica
- [ ] F36.2 iPhone Pro (A17 Pro+, Apple9): budget di memoria e termici mobili
- [ ] F36.3 **[CANDIDATO]** **[EDGE] visionOS**: rendering stereo con Compositor Services, rendering foveato con rasterization rate map, reprojection

## F37 — Vertical slice [CORE]

- [ ] F37.1 Livello di 10–15 minuti AAA-like (area aperta + interni + meteo + personaggi)
- [ ] F37.2 Tutti i budget rispettati su T0–T3 con i preset automatici
- [ ] F37.3 Nessun crash in 1 ora di gioco, frame pacing a 2 bucket
- [ ] F37.4 Prova della piattaforma completa: piccolo gioco 2D con menu/HUD, input e audio, più scena 3D editabile; asset e plugin del corpus F41, authoring F34, build senza editor e copertura F39/F40 verificata. Una demo grafica sola non chiude questo requisito

## F38 — Distribuzione e live ops [CORE]

- [ ] F38.1 App bundle firmata, hardened runtime, notarizzazione
- [ ] F38.2 Build Steam e Mac App Store (sandbox)
- [ ] F38.3 Telemetria opzionale (prestazioni per modello, crash), aggiornamenti incrementali dei contenuti
- [ ] F38.4 Documentazione dell'engine per chi crea contenuti
- [ ] F38.5 SDK di estensione, esempi/template 2D/3D/UI e guida di migrazione/import; versioni Bevy/plugin supportate, dipendenze, attribuzioni e matrice dei gap distribuite con il prodotto

## F39 — 2D completo con riferimento Bevy [CORE]

**Obiettivo:** parità funzionale del 2D rispetto alla versione Bevy fissata e
ai workflow community selezionati come copertura; renderer Metal Phosphor.

- [ ] F39.1 Matrice di parità da API/esempi Bevy [R115/R118], con versione, caso dimostrativo e prova per ogni comportamento; riuso di componenti/logica separabili prima di implementazioni equivalenti
- [ ] F39.2 Sprite, texture atlas, animazione spritesheet, flip/anchor/tint e slicing; batching e invalidazione asset nel grafo Metal, con blend/spazi colore corretti
- [ ] F39.3 Camere ortografiche, viewport, layer, ordinamento Z/trasparenza, pixel snapping e scaling; render-to-texture e composizione 2D/3D senza secondo renderer
- [ ] F39.4 Mesh e materiali 2D, effetti e shader personalizzati tramite adapter/port Metal; documentare i limiti dei materiali Bevy legati a wgpu/WGSL, con errori espliciti
- [ ] F39.5 Testo nel mondo, font fallback, shaping e atlas dei glifi condivisibili con F40; riuso delle librerie upstream e verifiche DPI/Unicode
- [ ] F39.6 Tilemap e livelli Tiled/LDtk/Aseprite attraverso F21/F41: layer, chunk, collisioni, animazioni e riferimenti ECS coerenti; runtime e preview editor equivalenti
- [ ] F39.7 Picking/interazione e integrazione con input e plugin di fisica 2D; hit test trasformati, coordinate camera e lifecycle delle entità verificati
- [ ] F39.8 Corpus di esempi Bevy/community, scene miste 2D/3D e gioco dimostrativo; parità visiva/comportamentale, resize e stress del batching, build finale senza editor

**Uscita:** tutte le righe del perimetro di parità dichiarato verificate;
gap registrati come non completati, non sostituiti dalla sola somiglianza API.

## F40 — UI di gioco completa con riferimento Bevy [CORE]

**Obiettivo:** UI distribuita col gioco, distinta dal debug ImGui; riuso di
layout, testo, input e accessibilità di Bevy dove praticabile [R114/R117].

- [ ] F40.1 Matrice di copertura/versione: esempi e semantica Bevy più requisiti di prodotto espliciti; distinguere parità upstream da capacità da completare oltre quella release
- [ ] F40.2 Layout reattivo flex/grid, misura, ancoraggio, unità/scaling, stili e gerarchie; invalidazione incrementale e casi annidati confrontati al riferimento
- [ ] F40.3 Adapter di rendering UI Metal per geometria, testo, immagini/atlanti, clipping, bordi/ombre e compositing; layout/logica riusati senza dipendenza implicita dal renderer Bevy
- [ ] F40.4 Pipeline testo con shaping, fallback font, wrapping, rich text, allineamento e DPI; localizzazione e lingue RTL verificate con contenuti reali
- [ ] F40.5 Editing testo completo: selezione, cursore, navigazione, clipboard, undo e IME/composizione; input non latino e integrazione piattaforma oltre una semplice casella ASCII
- [ ] F40.6 Focus, eventi e navigazione tastiera/controller/mouse/touch, scroll, drag/drop e modal; hit testing con trasformazioni/clipping e cattura dell'input senza interferire col gameplay
- [ ] F40.7 Widget e stato riusabili: pulsanti, toggle, slider, liste/menu, text field, tooltip e pannelli; binding ai dati, transizioni/animazioni e temi, con estensione tramite plugin
- [ ] F40.8 Accessibilità semantica, navigazione assistita e integrazione con servizi Apple; scala testo, contrasto e gestione del focus nelle viste dinamiche
- [ ] F40.9 Authoring/preview nell'editor F34 e uso runtime autonomo; stessa scena UI e stesso comportamento in play mode e build distribuita
- [ ] F40.10 Suite golden layout/immagini e test di interazione, IME/RTL, DPI/resize, controller e accessibilità; menu/HUD completi nel gioco F37.4 e budget misurati

**Uscita:** copertura funzionale verificata nella build del gioco. Un overlay
debug o pochi widget di esempio non soddisfano la completezza richiesta.

## F41 — Ecosistema Bevy e SDK di estensione [CORE]

**Obiettivo:** massimizzare il riuso dell'ecosistema con compatibilità reale
e dichiarata; integrazione preferita a riscrittura, per versione e piattaforma.

- [ ] F41.1 Inventario estendibile da Bevy Assets [R118] e progetti upstream: categorie logica, asset, input, fisica, animazione, UI/editor, 2D/tilemap e rendering; versione, dipendenze e test, senza dichiarare universalmente compatibile il catalogo
- [ ] F41.2 Contratto host/SDK dopo F27.1/F27.7: App/Plugin lifecycle, registrazione tipi/risorse/sistemi e versionamento; crate Rust ricompilate con dipendenze fissate, bridge pubblico C++/scripting e policy di upgrade
- [ ] F41.3 Riutilizzo diretto dei plugin indipendenti dal renderer e adapter dei servizi asset/input/scene/animazione; corpus rappresentativo eseguito con readback, non solo compilato
- [ ] F41.4 Adapter/port per UI/egui/inspector, picking/gizmo, tilemap e plugin grafici dipendenti da `bevy_render`/wgpu; tradurre contratti e materiali dove possibile, classificare le incompatibilità invece di emulare automaticamente l'intera API [R117/R119–R121]
- [ ] F41.5 Matrice verificabile diretto/adapter/port/non supportato per versione: codice d'esempio, comportamento atteso, costi del bridge e risultato; nessuna equivalenza fra API simile e compatibilità plugin
- [ ] F41.6 Test su progetto pulito, aggiornamenti delle dipendenze e regressioni con F35.7; scenari di asset mancanti, disabilitazione plugin e lifecycle completo
- [ ] F41.7 Distribuire template, guide, esempi e punti di estensione con F38.5; contributi di importer/plugin aggiungono fixture alla matrice, obiettivo di copertura ampia mantenuto senza promessa di compatibilità binaria universale

**Uscita:** corpus/versioni dichiarati funzionanti su Metal con prove end-to-end;
ogni gap resta visibile. Il nucleo F41.1/F41.2 precede l'authoring completo,
la certificazione UI/2D segue F39/F40 senza dipendenze circolari.

### Ipotesi remota: backend Vulkan

Possibile estensione futura, fuori da scope, calendario e implementazione
attivi. Per ora nessun port Vulkan, ripresa del legacy o RHI generale per
anticiparlo; conservare il confine dati/backend già esistente. Un eventuale
backend futuro non rende automaticamente compatibili i plugin Bevy.

---

## OPT-14 — QA delle prestazioni e ottimizzazione continua [OPT]

**Obiettivo**: nessuna regressione di prestazioni o qualità arriva su `main`
senza essere vista; l'ottimizzazione diventa un processo continuo.

**Letture**: [R84] [R85].

**Direzioni di ricerca**
- [ ] OPT-14.1 **[CANDIDATO]** **FLIP** [R84] per il confronto delle immagini di ogni ottimizzazione, con soglie per tier, affiancato alle metriche temporali di F8.7/F13.6: ghosting, flicker, disocclusioni e recupero della storia; adozione solo se supera entrambe le verifiche
- [ ] OPT-14.2 **[CANDIDATO]** **Rilevamento statistico delle regressioni** (change point detection [R85]) sui dati dei perf bot invece di soglie fisse
- [ ] OPT-14.3 **[CANDIDATO]** **Perf engineer automatico**: ciclo con Claude Code in locale (cattura → contatori → ipotesi → patch → misura → PR) sui pass che superano il budget
- [ ] OPT-14.4 **[CANDIDATO]** Telemetria opzionale delle prestazioni per modello di Mac per aggiornare preset e autotuning

**Spremitura del SoC**
- [ ] OPT-14.5 **[CANDIDATO]** Suite `bench/` rieseguita automaticamente a ogni nuova versione di macOS e su ogni nuovo chip
- [ ] OPT-14.6 **[CANDIDATO]** Playbook aggiornato a ogni WWDC e a ogni nuovo chip (M6?)

## OPT-15 — Avvio, dimensioni e distribuzione [OPT]

**Obiettivo (ipotesi)**: avvio a freddo sotto 5 s fino al menu, patch di
contenuto −60% rispetto al download completo dei pacchetti cambiati.

**Letture**: [R86]; playbook S-IO-*, S-MEM-*.

**Direzioni di ricerca**
- [ ] OPT-15.1 **[CANDIDATO]** **Patch minime** con chunking basato sul contenuto [R86]
- [ ] OPT-15.2 **[CANDIDATO]** **Prefetch registrato**: la traccia di accesso ai file dei primi minuti di gioco guida l'ordine dei dati nei pacchetti e il prefetch all'avvio
- [ ] OPT-15.3 **[CANDIDATO]** Archivi di pipeline per famiglia GPU e versione di OS, scaricati con gli aggiornamenti
- [ ] OPT-15.4 **[CANDIDATO]** Valutazione di un target a 8 GB con streaming più aggressivo e asset T0 dedicati

**Spremitura del SoC**
- [ ] OPT-15.5 **[CANDIDATO]** Pacchetti mappati in memoria e allineati alle pagine (S-IO-2)
- [ ] OPT-15.6 **[CANDIDATO]** Decompressione all'avvio distribuita su tutti i core con QoS corrette (S-CPU-1)

---

## Ambizione di lungo periodo e capacità candidate

Le tecniche avanzate seguenti descrivono la direzione del motore e rimangono
selezionabili. Editor, ECS, contenuti, 2D/UI ed ecosistema sono invece requisiti
funzionali del prodotto completo, con consegna progressiva. Le date e i vantaggi
prestazionali restano da dimostrare sui carichi reali.

| Area | Direzione da valutare |
|---|---|
| Geometria | GPU scene persistente, submission GPU-driven, cluster LOD virtualizzati, V-buffer, raster ibrido HW/SW |
| Luce | ReSTIR DI con migliaia di luci, ombre RT, GI a radiance cache (neurale su T3), ReSTIR GI, riflessioni RT |
| Ricostruzione | MetalFX temporal/denoised/neurale, frame interpolation, rasterization rate map |
| Path tracing | ReSTIR PT in tempo reale su M5 Max/Ultra |
| Neurale | Radiance cache e compressione delle texture neurali; reti proprietarie addestrate con MLX |
| Streaming | MTLIO + risorse sparse, virtual texturing, archivi di pipeline AOT |
| Efficienza | Banda per pixel budgettata, zero allocazioni e zero stutter per costruzione, autotuning per dispositivo |
| Piattaforma | Preset per ogni Mac Apple9+, EDR, ProMotion, Game Mode, iPad |
| Sviluppo del gioco | ECS/runtime con priorità al riuso Bevy, editor di produzione e SDK di estensione |
| 2D e UI | Copertura completa dichiarata e verificata: sprite/tilemap, testo, layout/widget, input e accessibilità |
| Contenuti ed ecosistema | Import/conversioni community e plugin riusati o adattati, con versioni e gap espliciti |

**Limiti di Metal rispetto ai motori PC** (dal report): niente shader execution
reordering generale, opacity micromap, cluster acceleration structure, TLAS
partizionati, work graph. La roadmap li aggira (proxy per l'RT, ordinamento
software dei raggi, radiance cache, catene di dispatch indiretti, ricostruzione
ML) invece di emularli.
