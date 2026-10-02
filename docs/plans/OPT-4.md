# OPT-4 — Shading, banda e ricostruzione

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Piano dettagliato anticipato; implementazione e misure ancora da eseguire. **Attivazione:** `opportunita`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

OPT-4.16 è baseline anticipata e non richiede il catalogo. Tutte le trasformazioni esatte conservano immagine e ownership; approssimazioni hanno preset/soglie temporali distinti. Grafo Conservative è fallback.

## Prerequisiti e confini

Baseline richieste: [F8](F8.md), [OPT-1](OPT-1.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.
Integrazione successiva con [OPT-2](OPT-2.md): Use selected optimizations only when measured and adopted; the baseline does not require completing this OPT catalog.
Integrazione successiva con [OPT-3](OPT-3.md): Use selected optimizations only when measured and adopted; the baseline does not require completing this OPT catalog.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `src/rendergraph/`.
- `src/rendergraph/optimizer/`.
- `shaders/material_resolve.metal (F7)`.
- `src/platform/metal/metalfx_temporal.* (F8)`.
- `tools/graph_select.py`.
- `tools/frame_corpus/ (nuovo solo estensione necessaria)`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

.16 dopo F8 → .15 con solver esistente se necessario; selezionare solo task dominante. .13/.14/.17 richiedono limite reale e non bloccano history F8.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

Frame reale con view/extent/materiali eterogenei; fusioni e async si giudicano sul frame intero. E-graph solo offline dopo fallimento della soluzione circoscritta.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-4.1 — Texel shading

**Ambito:** **[CANDIDATO]** **Shading disaccoppiato / texel shading** [R16][R17] per superfici costose: ombreggiare in spazio texture a frequenza ridotta e riusare tra frame

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Cache shading in spazio texture con coverage, età e invalidazione materiale/luce/view-dependent.
- **Verifica/accettazione:** Moto/speculari/disocclusioni, costo atlas+refresh e memoria; resolve full-rate fallback.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.2 — Software VRS

**Ambito:** **[CANDIDATO]** **Variable rate shading software sul visibility buffer** [R18], con la frequenza decisa da un predittore dell'errore visivo appreso [R21]; su Apple anche con rasterization rate map per i pass raster

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Classificare blocchi per ID/normal/contrast/motion, ricostruire soltanto regioni compatibili; predittore semplice prima della rete.
- **Verifica/accettazione:** Bordi/subpixel/storia invalida e tempo completo; rete richiede ulteriore beneficio misurato.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.3 — Mip single dispatch

**Ambito:** **[CANDIDATO]** **Catene di mip in un solo dispatch** (stile SPD [R20]) per Hi-Z, bloom, esposizione: meno pass, meno banda

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Confrontare più dispatch con algoritmo bounded senza spin fra gruppi; usare dispatch extra se richiesto per forward progress.
- **Verifica/accettazione:** NPOT/1xN/depth zero, min CPU e bloom energia; savings meno sync/banda misurati.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.4 — Material tiles

**Ambito:** **[CANDIDATO]** Resolve per tile classificate con uno shader specializzato per classe di materiale e salto dei rami con `simd_all/any` [R15]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Classificare code per materiale e varianti, conservando mapping pixel univoco e fallback generic.
- **Verifica/accettazione:** Tile mista/empty, divergence e overflow; classification+resolve contro baseline.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.5 — On-tile confronto

**Ambito:** **[CANDIDATO]** **Spike deferred on-tile vs V-buffer** ripetuto con i dati di OPT-0: per ogni tier scegliere la combinazione migliore (anche ibrida: V-buffer per la geometria, lighting on-tile); valutare sul frame completo banda, pressione sulla memoria on-chip, occupancy, partial render e sovrapposizione persa dalle fusioni, con contatori quando disponibili e misure dichiarate negli altri casi

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Stesso frame, allegati e qualità, misurare pressure tile/spill/partial render e overlap perso.
- **Verifica/accettazione:** Varianti V-buffer/on-tile/ibrida corrette; scegliere per workload/device, non per principio.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.6 — Scala MetalFX

**Ambito:** **[CANDIDATO]** Ricostruzione temporale: rapporto qualità/costo della risoluzione interna con MetalFX per ogni tier [R19]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Sweep risoluzione interna con motion/exposure identici e qualità temporale vincolata.
- **Verifica/accettazione:** Pareto qualità/frame/latency, ROI difficili e recovery; preset distinto per compromessi.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.7 — Formati compatti

**Ambito:** **[CANDIDATO]** Formati intermedi ridotti (R11G11B10F, RGB9E5, `half`) dove i test visivi lo consentono (S-TEX-1)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Analizzare range/segno prima di sostituire RGBA16F; encoded storage separato da math precision.
- **Verifica/accettazione:** HDR/signed normals/negative values e overflow, errori localizzati e costo bandwidth.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.8 — Compute compression

**Ambito:** **[CANDIDATO]** Output compute scritti a blocchi interi per la compressione universale di M5; contatore "write inefficiency" a zero (S-TEX-2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Scritture coerenti a blocchi dove SDK/layout consente, evitando formati/view che rompono compressione.
- **Verifica/accettazione:** Pattern pieno/sparso e frame reale; counter unavailable dichiarato, target write inefficiency non presunto misurabile.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.9 — Sparse access compression

**Ambito:** **[CANDIDATO]** Compressione disattivata sulle texture ad accesso sparso dopo misura del Compression Ratio (S-TEX-2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** A/B flag/formato applicabile senza cambiare contenuto, su texture realmente campionate sparse.
- **Verifica/accettazione:** Cache/tempo/byte osservabili e qualità invariata; disabilitazione solo se API e dati la sostengono.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.10 — Mip bias

**Ambito:** **[CANDIDATO]** Mip bias corretto per MetalFX; mip più bassi per effetti a bassa frequenza (S-TEX-3)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Politica per segnale/scala, clamp e derivative correctness; nessuna riduzione su detail sensibile senza preset.
- **Verifica/accettazione:** Aliasing speculare e testo/UI, scene in movimento; confronto F8 baseline.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.11 — MSAA memoryless

**Ambito:** **[CANDIDATO]** MSAA 4x memoryless valutato per UI e vegetazione in alpha-to-coverage (S-TBDR-8)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Percorso 4x con resolve e alpha-to-coverage espliciti, budget tile verificato.
- **Verifica/accettazione:** UI/foliage edge, transparencies e costo resolve; single-sample fallback.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.12 — EDR adaptation

**Ambito:** **[CANDIDATO]** Uscita EDR con headroom interrogato e tonemapping adattato al display (S-DISP-2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Usare F8 output adapter, aggiornare curve da headroom corrente con smoothing.
- **Verifica/accettazione:** Display/headroom change, SDR fallback e clipping; niente seconda pipeline colore concorrente.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.13 — Subresource graph

**Ambito:** **[CANDIDATO]** **Accessi a sottorisorse nel render graph**: mip, layer e aspect delle texture, intervalli di byte dei buffer; dipendenze e barriere per intervalli sovrapposti, mantenendo allocazione e vincoli di aliasing della risorsa fisica intera; test di accessi disgiunti, parzialmente sovrapposti e di lettura/scrittura fra code

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Range mip/layer/aspect/byte, overlap normalizzato per dipendenze; allocazione/alias restano su risorsa intera.
- **Verifica/accettazione:** Disjoint/partial overlap, RAW/WAR/WAW cross-queue, random interval oracle; resource-wide fallback.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.14 — Cross-queue lifetime

**Ambito:** **[CANDIDATO]** **Lifetimes precise fra code e frame**: sostituire la vita conservativa di tutto il frame per le risorse async con un modello di ordinamento parziale derivato da dipendenze ed eventi; aliasing solo dopo tutti gli ultimi accessi ordinati, riuso fra frame/slot coerente con F8.6. Confronto con il piano conservativo, test dell'ordinamento e stress di riuso con più frame in volo; ridurre le attese solo dove la correttezza è dimostrata

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Happens-before da eventi e dipendenze; alias solo se ogni ultimo reader/writer precede il nuovo uso.
- **Verifica/accettazione:** Interleaving model checker piccolo, frame slots e poison; confronto Conservative, mai dedurre ordine dal wall clock.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.15 — Real frame plans

**Ambito:** **[CANDIDATO]** **Piani ottimizzati sul frame reale F5–F8**: applicare ricerca e selezione misurata di OPT-1 a culling, Hi-Z, visibility, resolve e ricostruzione; scegliere ordine, fusioni, rematerializzazione e code sui dispositivi disponibili. Chiave strutturale più ambito di validità misurato (chip/famiglia, OS, preset, risoluzione e classe di carico); invalidazione e ripiego conservativo registrati. Controllare equivalenza per i riordini esatti e soglie F8.7 per le varianti approssimate; estendere i piani a ogni nuova era senza assumere che il modello di costo basti

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Estendere selezione OPT-1 con applicability chip/OS/preset/extent/workload e structural key.
- **Verifica/accettazione:** Off/greedy/plan pixel+temporal, stale profile fallback e gain end-to-end; cost model da solo non decide.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.16 — Corpus baseline

**Ambito:** **[BASELINE]** **Baseline di scene rappresentative**: oltre ai microbenchmark e agli scenari sintetici, percorsi di camera ripetibili in scene reali dense con materiali eterogenei e oggetti in movimento; estenderli con luci dinamiche, trasparenze, vegetazione e streaming quando disponibili. Registrare tempo del frame e per pass, p50/p95/p99, memoria di picco, banda (misurata o stimata esplicitamente), energia e latenza; confronto a parità di qualità e in regime termico stabile, su T0 e T2 (O12), riusando il corpus in F29/F35

**Stato di pianificazione:** Da implementare o completare; stato dettagliato nella roadmap.

- **Implementazione:** Congelare asset hash/seed/camera, scene densa/materiali/dinamica e runner esistente.
- **Verifica/accettazione:** Report frame/pass p50/p95/p99/peak/energy se disponibile, raw e limiti T0; estendere con ogni era.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-4.17 — E-graph

**Ambito:** **[CANDIDATO]** **Compilazione algebrica del frame [EDGE]**: spike offline con e-graph/equality saturation [R100] e rematerializzazione [R101], regole esatte con precondizioni e varianti approssimate separate; estrazione Pareto sotto vincoli di tile/heap/dipendenze, confronto con DP/annealing OPT-1, cap di ricerca e adozione misurata (ricerca H4)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** IR pure transformations con precondizioni, exact rules separate da approximations; search/extraction offline bounded.
- **Verifica/accettazione:** Small exhaustive cases, proof ownership/dependencies, DP/annealing baseline e frame holdout; stop se costo/manutenzione superano beneficio.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Esplosione di ricerca, falsa equivalenza FP e barriera troppo stretta. Ripiego solver OPT-1 e piano conservativo; prototipo e-graph esclusivamente offline.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R90–R91, R100–R101; H4/H5. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
