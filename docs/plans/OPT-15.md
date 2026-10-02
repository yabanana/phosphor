# OPT-15 — Avvio, dimensioni e distribuzione

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Piano dettagliato anticipato; implementazione e misure ancora da eseguire. **Attivazione:** `opportunita`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

Avvio misurato da lancio a menu interattivo, cold/warm distinti. Pacchetti verificati per hash/schema e aggiornamenti atomici; tempo decompilazione/decompressione e lavoro CPU sono inclusi.

## Prerequisiti e confini

Baseline richieste: [F3](F3.md), [F21](F21.md), [F23](F23.md), [F38](F38.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `tools/package/ (F38)`.
- `tools/cooker/ (F21)`.
- `src/assets/ (F22)`.
- `src/platform/metal/pipeline_cache.*`.
- `cmake/PipelineArchive.cmake`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

Baseline startup trace → .2/.3/.5/.6 → .1 se aggiornamenti pesano; .4 richiede un target memoria reale futuro.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

Trace file iniziale e archivi miss; cold start ripetibile senza azioni distruttive sulla cache di sistema. Patch full-download è fallback.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-15.1 — Content chunking

**Ambito:** **[CANDIDATO]** **Patch minime** con chunking basato sul contenuto [R86]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Chunk content-defined e manifest hash, patch assembly atomico con download dei soli chunk mancanti.
- **Verifica/accettazione:** Corruzione/interruzione/version mismatch e rollback; byte/CPU/disk write contro full package.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-15.2 — Prefetch trace

**Ambito:** **[CANDIDATO]** **Prefetch registrato**: la traccia di accesso ai file dei primi minuti di gioco guida l'ordine dei dati nei pacchetti e il prefetch all'avvio

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Registrare access pattern avvio, riordinare chunk e prefetch bounded separato dall’urgente.
- **Verifica/accettazione:** Scene/start differenti holdout, warm/cold, I/O inutile e time-to-interactive.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-15.3 — Pipeline delivery

**Ambito:** **[CANDIDATO]** Archivi di pipeline per famiglia GPU e versione di OS, scaricati con gli aggiornamenti

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Bundle archivi per GPU/OS/metallib hash con verifica applicabilità e fallback compiler F3.
- **Verifica/accettazione:** Miss/corrupt/OS upgrade, cold startup e nessun hitch sul render thread.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-15.4 — 8 GB target

**Ambito:** **[CANDIDATO]** Valutazione di un target a 8 GB con streaming più aggressivo e asset T0 dedicati

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Preset asset/budget limitato e streaming aggressivo con eviction; stress logico prima del device reale.
- **Verifica/accettazione:** Peak resident e hitch su M5 limitato sono preliminari; 8 GB certificato solo con macchina fisica.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-15.5 — Mmap packages

**Ambito:** **[CANDIDATO]** Pacchetti mappati in memoria e allineati alle pagine (S-IO-2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Query page/allignment, mapping read-only e bounded spans, evitare page fault critici con prefetch.
- **Verifica/accettazione:** File troncato, offset overflow e working set; mmap non è automaticamente GPU zero-copy.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-15.6 — Parallel decode

**Ambito:** **[CANDIDATO]** Decompressione all'avvio distribuita su tutti i core con QoS corrette (S-CPU-1)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Pool condiviso/QoS utility, chunk sizes e concurrency sweep con memory cap.
- **Verifica/accettazione:** Startup+render contention, p99 e peak RAM; max core count non è necessariamente optimum.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Prefetch accelera menu ma causa pressione subito dopo. Ripiego demand-first e package baseline, patch completa come fallback verificato.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R86 da verificare, R90. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
