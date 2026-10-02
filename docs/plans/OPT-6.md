# OPT-6 — Illuminazione globale, cache e ammortamento

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Piano dettagliato anticipato; implementazione e misure ancora da eseguire. **Attivazione:** `opportunita`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

Cache e update hanno errore/età misurabili e invalidazione semantica. Scheduler baseline round-robin bounded precede policy generale; nessuna storia stale per nascondere lavoro.

## Prerequisiti e confini

Baseline richieste: [F14](F14.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.
Integrazione successiva con [OPT-4](OPT-4.md): Use selected optimizations only when measured and adopted; the baseline does not require completing this OPT catalog.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `src/renderer/radiance_cache.* (F12)`.
- `src/renderer/probe_grid.* (F12)`.
- `shaders/ddgi.metal (F12)`.
- `src/renderer/deferred_updates.* (nuovo candidato)`.
- `src/rendergraph/optimizer/`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

Profilo GI/cache/volumi → .4/.8 baseline → .1/.6/.7; .5/.12 soltanto con picchi rilevanti, .9 dopo prova contesa.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

Camera cut, apertura porta e luce improvvisa obbligatori; confronto medesima qualità temporale, non solo screenshot convergente.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-6.1 — Cache gerarchica

**Ambito:** **[CANDIDATO]** **Cache di radianza a due livelli** [R37] e hash spaziale jittered [R38], confrontate con cache sulle superfici [R48]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Due livelli/hash jittered/surface cache in varianti distinte con lookup bounded e invalidazione.
- **Verifica/accettazione:** Collisioni, thin wall, cache piena e refresh lag; costo memoria+lookup+update vs F12.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-6.2 — Radiance cascades

**Ambito:** **[CANDIDATO]** **Spike Radiance Cascades / Split Radiance Cascades** [R39]: costo costante indipendente dalla complessità della scena, probe sparse in hashmap

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Prototipo sparse con intervalli/merge e budget definito, confronto DDGI stessa scena.
- **Verifica/accettazione:** Leak, risoluzione angolare e scene fuori ipotesi; nessuna assunzione costo costante universale.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-6.3 — ORCA

**Ambito:** **[CANDIDATO]** **Cache ORCA** [R40] per accelerare il path tracing (T3) senza dipendere dalla storia temporale

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Cache opzionale per PT con accesso/versione/terminazione espliciti e reference unbiased.
- **Verifica/accettazione:** Luce/geometry change e camera nuova; costo totale e bias misurato, fallback PT base.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-6.4 — Probe scheduling

**Ambito:** **[CANDIDATO]** DDGI di produzione [R41]: classificazione e riallocazione delle sonde, aggiornamento guidato dalla varianza invece che a rotazione fissa

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Priorità variance/age e classificazione/relocation bounded; minimo aggiornamento garantito.
- **Verifica/accettazione:** Probe in parete e zona mai vista, age cap e confronto round-robin.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-6.5 — Update scheduler

**Ambito:** **[CANDIDATO]** **Scheduler dei lavori ammortizzati** (idea Phosphor): GI, ombre statiche, LUT di atmosfera, BVH, streaming aggiornati a frequenze diverse da uno scheduler che riempie il budget residuo di ogni frame, per un frame time piatto

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Job deferred con deadline/costo/errore, quota frame e aging; iniziare con lista prioritaria semplice.
- **Verifica/accettazione:** Burst di invalidazioni e starvation, p99 budget e overhead; adozione generale solo se serve.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-6.6 — Volumetric low-res

**Ambito:** **[CANDIDATO]** Volumetrici, nuvole ed effetti a bassa risoluzione con ricostruzione temporale [R45][R46][R47]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Riproiezione depth-aware, confidence e max history age per fog/cloud.
- **Verifica/accettazione:** Camera rapida, silhouette e changing light; ghosting vs full-rate reference.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-6.7 — Denoiser compare

**Ambito:** **[CANDIDATO]** Denoiser: SVGF [R44] e varianti per T0, confronto con il denoiser MetalFX su costo e qualità

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** SVGF/atrous/MetalFX su medesimi noisy inputs e canali, configurazioni esplicite.
- **Verifica/accettazione:** Qualità temporale e latency+memoria; baseline più semplice se pari.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-6.8 — Cache packing

**Ambito:** **[CANDIDATO]** Atlanti di sonde, cache e reservoir in `half` o formati compatti; dimensioni tarate per restare nella SLC (S-ALU-3, S-MEM-2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** FP16/compact con range per radiance/moments, layout locality e chunk size.
- **Verifica/accettazione:** HDR outlier, precisione variance e leak; non garantire fit SLC da capacity stimata.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-6.9 — Async GI/AS

**Ambito:** **[CANDIDATO]** GI e build di BVH sulla seconda coda compute, sovrapposti al raster (S-SYNC-2), solo se migliorano il frame completo: misurare contesa di banda/cache, attese e memoria di picco con le lifetimes di OPT-4.14; aggiornare e riselezionare i piani reali di OPT-4.15 dopo F9–F14

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Placement di pass indipendenti nel grafo, contesa e retirement dichiarati; life conservative consentita.
- **Verifica/accettazione:** Una/due code frame A/B, bandwidth/heap/p99 e reselect plan; no dipendenza obbligatoria da OPT-4.14.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-6.10 — Depth bounds

**Ambito:** **[CANDIDATO]** Depth bounds test su M5 per volumi di luce e nebbia (S-GEO-5)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Feature gate Apple10, limiti conservativi per volumi e fallback shader reject.
- **Verifica/accettazione:** Camera dentro volume, reverse-Z e invalid bounds; immagine identica e frame gain.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-6.11 — Compression cache

**Ambito:** **[CANDIDATO]** Compressione disattivata sugli atlanti ad accesso sparso se il Compression Ratio lo indica (S-TEX-2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** A/B attivazione applicabile sugli atlanti sparsi, mantenendo storage e semantic invarianti.
- **Verifica/accettazione:** Ratio se disponibile, lookup/frame time e bytes stimati distinti; fallback driver default.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-6.12 — Error-budget history

**Ambito:** **[CANDIDATO]** **Cache temporali con budget d'errore [EDGE]**: sonde, ombre e shading con età massima, invalidazione e priorità per riduzione attesa dell'errore per unità di tempo; confronto con aggiornamento fisso/per varianza, camera cut e luce improvvisa; preservare i requisiti statistici dei reservoir ReSTIR (H8, F8.7/F13.6)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Stato age/confidence/error e priorità expected-error-reduction/cost, reset per eventi semantici.
- **Verifica/accettazione:** Luce impulsiva/cut, max age e fairness; reservoir conserva condizioni estimator, non cache generica indiscriminata.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Bilanciamento medio può lasciare regioni stale. Ripiego aging con limite duro, coda unica e update uniformi; preservare correttezza statistica ReSTIR.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R102; H8; R37–R48 da verificare. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
