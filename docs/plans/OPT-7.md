# OPT-7 — Geometria virtualizzata e mondo

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Piano dettagliato anticipato; implementazione e misure ancora da eseguire. **Attivazione:** `opportunita`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

World baseline F18/F19 e renderer trasparenze corretti; task acqua richiedono F20 selezionata. LOD, raster e codec scelti a errore visivo vincolato, non da throughput isolato.

## Prerequisiti e confini

Baseline richieste: [F18](F18.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.
Integrazione successiva con [F16](F16.md): Required by the corresponding implementation package only; does not block the initial baseline or unrelated candidates.
Integrazione successiva con [F19](F19.md): Required by the corresponding implementation package only; does not block the initial baseline or unrelated candidates.
Integrazione successiva con [F20](F20.md): Required by the corresponding implementation package only; does not block the initial baseline or unrelated candidates.
Integrazione successiva con [F22](F22.md): Required by the corresponding implementation package only; does not block the initial baseline or unrelated candidates.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `tools/cooker/cluster_lod.* (F18)`.
- `shaders/software_raster.metal (F18)`.
- `shaders/terrain.metal (F19)`.
- `shaders/oit.metal (F16)`.
- `shaders/water.metal (F20)`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

Profilo → .1/.7/.8 stessa sperimentazione raster; .2/.10 LOD; .3/.4/.5/.6/.9 solo consumer pertinenti.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

Intersezioni HW/SW, parameter buffer e HSR confrontati su scena completa; immagini di riferimento includono alpha/motion/shadow.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-7.1 — Raster threshold

**Ambito:** **[CANDIDATO]** **Soglia raster software/hardware** misurata per chip [R51][R52]; raster software dedicato per fili e capelli [R53]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Sweep dimensione proiettata/shape per HW/SW, profilo per shader/chip; capelli separati dal triangle workload.
- **Verifica/accettazione:** Merge/depth/ID esatti e costo sorting+raster; threshold con isteresi e fallback HW.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-7.2 — LOD percettivo

**Ambito:** **[CANDIDATO]** DAG di cluster con **errore percettivo** (non solo geometrico) e build parallelo veloce [R50]; streaming ordinato per errore [R49]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Error estimator da silhouette/normal/materiale, DAG monotono e cook parallelo.
- **Verifica/accettazione:** Holdout camera/luce e seams, qualità <1 px ove contratto lo richiede; build time/memoria inclusi.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-7.3 — Concurrent tree

**Ambito:** **[CANDIDATO]** **Terreno con concurrent binary tree** usato come pool di memoria [R54] e tessellation adattiva invece di clipmap fisse

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Pool CBT bounded con update split/merge e versioni, confronto clipmap semplice.
- **Verifica/accettazione:** Crack-free, alloc/free concur e overflow; mantenere clipmap se guadagno insufficiente.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-7.4 — Vegetazione RT

**Ambito:** **[CANDIDATO]** **Vegetazione massiva in ray tracing** con le tecniche di [R55]; impostor per la distanza

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Proxy/cluster/impostor con alpha semantics e threshold distanza; AS rebuild bounded.
- **Verifica/accettazione:** Shadow/reflection su vento e LOD change, thin geometry e tempo build+trace.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-7.5 — Wavelets

**Ambito:** **[CANDIDATO]** **Acqua con Water Surface Wavelets** [R56] dove serve interazione locale, FFT solo per l'oceano aperto

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Onde locali wavelet con dominio/bordi e compositing con oceano FFT.
- **Verifica/accettazione:** Interazione/energia/riflessioni e costo sim+render; FFT/Gerstner fallback.
- **Dipendenze specifiche:** F20.1.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-7.6 — MLAB/momenti

**Ambito:** **[CANDIDATO]** **Trasparenze nella tile**: MLAB con raster order group [R57] per il vetro, OIT a momenti [R58] per le particelle

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Varianti OIT con capacity/error espliciti e blending ordinato compatibile ROG.
- **Verifica/accettazione:** Layer sovrapposti, alpha alta e overflow; confronto sorted reference e frame completo.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-7.7 — Atomici raster

**Ambito:** **[CANDIDATO]** Raster software con `atomic_max` a 64 bit (Apple9) e atomici gerarchici; contesa misurata con B-07 (S-SIMD-4)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Packing depth+tie ID total-order reverse-Z e aggregazione locale prima di atomic max legale.
- **Verifica/accettazione:** Equal depth, NaN sanitizzato, contesa worst-case e bit semantics contro reference.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-7.8 — HSR

**Ambito:** **[CANDIDATO]** Raster software solo dove non rompe l'HSR del TBDR (misura B-13) (S-TBDR-2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Separare casi dove SW costringe load/store o perde hidden-surface removal; segmentare solo se giustificato.
- **Verifica/accettazione:** Full frame sotto alta occlusione e mixed geometry; nessuna deduzione dalla sola velocità compute.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-7.9 — Tile-only decals/OIT

**Ambito:** **[CANDIDATO]** OIT e decal interamente in tile memory con ROG, nessun atomico in device memory (S-TBDR-7)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Fondere soltanto consumer legali con imageblock/ROG, dichiarare memoria e limiti.
- **Verifica/accettazione:** Validation, overflow tile e immagini; store evitati reali vs overlap perso.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-7.10 — LOD per generazione

**Ambito:** **[CANDIDATO]** Soglie di LOD per generazione (M5 con geometria 2x) (S-GEO-2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Parametri da profili reali per chip, fallback errore geometrico conservativo.
- **Verifica/accettazione:** Clip no popping e performance M5; altri chip non dichiarati tarati senza misure.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Specializzare troppo per scena o chip. Ripiego ai percorsi della fase F e soglia statica conservativa; conservare failure case nel corpus.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R49–R58 da verificare, R91. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
