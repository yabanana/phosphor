# OPT-3 — Geometria, culling e dati di vertice

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Piano dettagliato anticipato; implementazione e misure ancora da eseguire. **Attivazione:** `opportunita`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

Geometria esatta di F6–F8 come riferimento; codec/LOD/area sono approssimazioni con bounds conservativi. Separare guadagno fetch da costo decode e da qualità sacrificata.

## Prerequisiti e confini

Baseline richieste: [F8](F8.md), [OPT-0](OPT-0.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `src/renderer/meshlet_builder.*`.
- `src/renderer/meshlet_cull_reference.* (F6)`.
- `shaders/meshlet.metal (F6)`.
- `tools/cooker/ (F21)`.
- `bench/f6_spike/ (F6)`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

Profilo → .4/.7 baseline dimensioni → una fra .1/.2/.6/.11; .3/.5 nuove infrastrutture solo dopo trigger; .9/.10 specializzazioni misurate.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

Corpus denso/skinny/foliage e camera dinamica; bytes per tri, object+mesh+cull+raster+resolve nel tempo completo.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-3.1 — Codec meshlet

**Ambito:** **[CANDIDATO]** **Meshlet compressi decompressi nel mesh shader**: triangle strip generalizzate ottime [R12] e formato denso stile DGF [R13], confrontati con il codec di meshoptimizer; meno byte per triangolo letti dalla DRAM

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Confrontare meshoptimizer con strip/DGF-like su stesso ordine, implementare decoder bounded e schema versionato.
- **Verifica/accettazione:** Roundtrip o errore quantizzato dichiarato, bounds, byte/tri e decode+fetch+frame; asset classico fallback.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-3.2 — Quantizzazione

**Ambito:** **[CANDIDATO]** **Quantizzazione aggressiva**: posizioni a 16 bit relative al cluster, normali e tangenti ottaedriche [R14], UV `half`; attributi letti solo nel resolve

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Posizione cluster-relative, oct normal/tangent e UV half con scale/range espliciti.
- **Verifica/accettazione:** Seam, tangent handedness e extreme UV, clip speculari; bounds allargati per errore massimo.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-3.3 — CPU occlusion

**Ambito:** **[CANDIDATO]** **Occlusion culling ibrido CPU+GPU su memoria unificata** (idea Phosphor): rasterizzazione software degli occluder sui P-core liberi con NEON [R10], risultato letto dalla GPU senza copie per scartare istanze prima del Hi-Z

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Raster software NEON su occluder selezionati, pubblicazione snapshot con generation/completion senza bloccare GPU.
- **Verifica/accettazione:** Visibilità conservativa, near-plane e motion; costo CPU+sync+banda vs Hi-Z GPU, scarto se contesa annulla beneficio.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-3.4 — Builder strategies

**Ambito:** **[CANDIDATO]** Strategie di generazione dei meshlet confrontate sul nostro hardware [R11]; coni di normali più stretti

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Spatial vs greedy/topological, coneWeight e vertex reuse con cook deterministico.
- **Verifica/accettazione:** Stesso contenuto, cone conservativo, build time e frame con vari tassi occlusione.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-3.5 — Tessellation compute

**Ambito:** **[CANDIDATO]** **Tessellation adattiva in compute** [R22] per superfici lisce invece di geometria densa precalcolata

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Livelli da errore proiettato e edge agreement, output bounded e crack-free.
- **Verifica/accettazione:** Edge condivisi, degenerate patch e buffer pieno; confronto geometria cooked e costo produzione+render.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-3.6 — Culling incrementale

**Ambito:** **[CANDIDATO]** Culling incrementale: su camera ferma o quasi ferma, riuso della lista visibile e test solo dei cluster cambiati [R9]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Dirty sets per camera/instance/occluder e validity stamp; invalidazione globale sul cut.
- **Verifica/accettazione:** Movimento piccolo/grande e cambio occluder; reference full cull con zero falsi negativi.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-3.7 — Meshlet limits

**Ambito:** **[CANDIDATO]** Dimensione dei meshlet e massimi dichiarati calibrati per chip con B-16 (S-GEO-1)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Sweep dimensioni/output massimi per chip reale, non family astratta; conservare variant baseline.
- **Verifica/accettazione:** Occupazione e payload legali, S1 corpus e default con vantaggio robusto.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-3.8 — Parameter buffer

**Ambito:** **[CANDIDATO]** Soglia del parameter buffer (B-12) mai superata nei testbench; attributi nel pass di raster ridotti al minimo (S-TBDR-3)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Ridurre attributi/raster output e chunking solo se partial render limita il frame.
- **Verifica/accettazione:** Soglia workload specifica, ordine draw e immagine invariati; costi split inclusi.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-3.9 — Apple10 paths

**Ambito:** **[CANDIDATO]** Percorso Apple10: Hi-Z con riduzione min/max nel sampler, ICB estesi, valori per-vertex non interpolati (S-GEO-4, S-GEO-5)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Sampler reduction/ICB state/per-vertex gated dalle capacità, fallback Apple9 sempre esercitabile.
- **Verifica/accettazione:** Override effettivo su M5 e confronto reference; 0 errori non equivalgono a prova hardware M3.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-3.10 — LOD generation

**Ambito:** **[CANDIDATO]** Soglie di LOD diverse per M3/M4 e M5 (geometria 2x su M5, S-GEO-2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Profili qualità/costo per chip misurato, partire da errore visivo e non da throughput teorico 2x.
- **Verifica/accettazione:** Clip e frame sul M5; profili M3/M4 non certificati finché mancano misure.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-3.11 — Prefix SIMD

**Ambito:** **[CANDIDATO]** Compattazione dei meshlet visibili con prefix sum SIMD-group, nessun atomico globale per thread (S-SIMD-1, S-SIMD-4)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Riutilizzare scan stabile F5/F6, aggregare per SIMD/group con fasi bounded.
- **Verifica/accettazione:** Conteggi/liste contro CPU, overflow e casi tutti visibili; costo intera catena e ordine deterministico.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Quantizzazione aumenta errori al bordo; decode può annullare la banda risparmiata. Ripiego codec semplice, LOD conservativo e culling GPU invariato.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R9–R14/R22 da verificare, R91, R108. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
