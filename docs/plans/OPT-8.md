# OPT-8 — Streaming, texture e materiali

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Piano dettagliato anticipato; implementazione e misure ancora da eseguire. **Attivazione:** `opportunita`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

Ottimizzare catena disco→decode→upload→sample. Compressione che sposta costo sul frame o distrugge filtering non è automaticamente vantaggio. Residency e table publication restano transazionali.

## Prerequisiti e confini

Baseline richieste: [F15](F15.md), [F21](F21.md), [F22](F22.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.
Integrazione successiva con [F30](F30.md): Varianti NTC/ANE successive; codec/predictor analitici prima.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `tools/cooker/ (F21)`.
- `src/assets/ (F21/F22)`.
- `src/platform/metal/io_streamer.* (F22)`.
- `src/platform/metal/sparse_pages.* (F22)`.
- `src/ml/ (F30 se selezionata)`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

Baseline .6/.7/.8/.9/.10 → scegliere .1/.2/.3/.4; .5 dopo predittore analitico, .11 dopo consumer neurale.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

Trace volo/teleport con budget logico limitato e dispositivi realmente disponibili; disk cold/warm distinti, niente pressione memoria distruttiva sul sistema.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-8.1 — Supercompression

**Ambito:** **[CANDIDATO]** **Texture supercompresse decodificate dalla GPU** [R59] e compressione neurale a blocchi [R60]: meno disco e I/O senza cambiare gli shader

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Codec GPU block decode con payload versionato, transcode finale compatibile texture classiche.
- **Verifica/accettazione:** File corrotto, qualità/alpha/mip e disk→sample latency; ASTC/BC baseline e fallback.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-8.2 — Adaptive VT

**Ambito:** **[CANDIDATO]** **Virtual texture adattiva** [R61] con feedback a bassa risoluzione e decompressione in compute con SIMD-group

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Feedback coarse→richieste conservative, page granularity da API e decompressione bounded.
- **Verifica/accettazione:** Bordi/anisotropy/teleport, feedback pieno e parent mip corretto; byte risparmiati vs lavoro extra.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-8.3 — Normal filtering

**Ambito:** **[CANDIDATO]** **Prefiltraggio delle normali e specular antialiasing** [R62]: meno aliasing speculare → meno bisogno di supersampling e di risoluzione interna alta

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Prefilter variance e specular AA con roughness compensation coerente col BRDF.
- **Verifica/accettazione:** Grazing/specular subpixel in movimento, energia e dettaglio; costo/qualità vs supersampling.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-8.4 — Asset per tier

**Ambito:** **[CANDIDATO]** **Rappresentazioni per dispositivo generate offline** (come SLIM [R63]): il cooker produce varianti di asset per T0…T3, non un solo asset scalato a runtime

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Cooker produce variant manifest compatibili con stesso GUID e materiali; selezione offline per budget.
- **Verifica/accettazione:** Switch preset e asset fallback, disco/memoria/qualità; profili altri device restano ipotesi.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-8.5 — ANE prediction

**Ambito:** **[CANDIDATO]** **Streaming predittivo** (idea Phosphor): previsione della traiettoria della camera (modello piccolo su ANE) per anticipare le richieste

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Modello piccolo camera predictor asincrono con max age, confronto velocità/direzione semplice.
- **Verifica/accettazione:** Inversione/teleport e output tardivo, hit rate/byte inutili/p99; non aspettare inferenza.
- **Dipendenze specifiche:** F30.6.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-8.6 — Codec I/O

**Ambito:** **[CANDIDATO]** Codec MTLIO scelto per tipo di dato in base a B-25; richieste grandi e allineate (S-IO-2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Sweep per tipo dato e dimensione chunk, align query SDK, batch richieste.
- **Verifica/accettazione:** CPU decode e MTLIO end-to-end, warm/cold file cache e errori dati.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-8.7 — Streaming budget

**Ambito:** **[CANDIDATO]** Budget di streaming per tier e per velocità del disco rilevata (S-IO-1)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Admission per memoria/IO latency osservata con quota urgente/prefetch e backpressure.
- **Verifica/accettazione:** SSD lento simulato/queue burst e resident cap; no render-thread wait.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-8.8 — Sparse pages

**Ambito:** **[CANDIDATO]** Pagine sparse da 16 o 64 KB scelte per tipo di risorsa; costo di mapping misurato (S-TEX-4)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Valutare solo page sizes realmente esposte, costi mapping/tail e fragmentation.
- **Verifica/accettazione:** Map/unmap/reuse concorrente e texture mips piccole; scelta per tipo risorsa.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-8.9 — Residency batches

**Ambito:** **[CANDIDATO]** Residency set aggiornati in modo incrementale e in batch (S-MEM-5)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Delta set aggregati prima del consumer, limitare commit superflui e mantenere generation.
- **Verifica/accettazione:** Pagina cancellata in volo, frame concurrent e readback; tempi CPU inclusi.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-8.10 — QoS streaming

**Ambito:** **[CANDIDATO]** Streaming in QoS utility, con concorrenza e granularità misurate per non disturbare render e simulazione; verificare il placement osservato senza assumere pinning sugli E-core (S-CPU-1, R87)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Utility pool bounded e job chunking con deadline, coordinato con render/simulation.
- **Verifica/accettazione:** Trace migration/QoS e p99 sotto I/O+GPU; nessuna promessa E-core pinning.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-8.11 — Codec/layout congiunti

**Ambito:** **[CANDIDATO]** **Scelta congiunta di compressione e layout [EDGE]**: costo end-to-end da disco al campione shader per ASTC/BC, decode-on-load e decode-on-sample neurale quando F30 è disponibile; includere packing, copie, cache miss e filtraggio; asset classici come baseline (H3/H9, R103)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Confrontare classico, decode-on-load e on-sample per asset class con packing/filter/cache nel costo.
- **Verifica/accettazione:** Pareto qualità/memoria/IO/frame e temporalità; soluzione neurale solo dove vince realmente.
- **Dipendenze specifiche:** F30.3. Confronto formati classici prima; parte neurale dopo F30.3.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Compressione salva disco ma peggiora rendering; predictor spreca I/O. Ripiego codec classico per asset e richieste demand-first, con budget speculativo limitato.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R93–R94, R103; H3/H6/H9; R59–R63 da verificare. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
