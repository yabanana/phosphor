# OPT-11 — Rendering neurale

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Piano dettagliato anticipato; implementazione e misure ancora da eseguire. **Attivazione:** `opportunita`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

Reti candidate hanno fallback numerico e funzionale. Utilization è metrica quando osservabile, mai gate inventato; adozione per qualità e costo end-to-end, includendo training, layout e contesa.

## Prerequisiti e confini

Baseline richieste: [F30](F30.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.
Integrazione successiva con [F31](F31.md): Required by the corresponding implementation package only; does not block the initial baseline or unrelated candidates.
Integrazione successiva con [F32](F32.md): Required by the corresponding implementation package only; does not block the initial baseline or unrelated candidates.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `src/platform/metal/tensor_runtime.* (F30)`.
- `src/platform/apple/coreml_runtime.mm (F30)`.
- `shaders/neural/ (F30)`.
- `tools/training/ (F31)`.
- `bench/soc/`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

Consumer F30/F31 selezionato → .8/.7 diagnosi → .1/.2/.6 → .9 scheduling; .3/.4/.5/.10/.11 solo consumer richiesto.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

Operatore piccolo reference e rete reale con shapes/layout del motore; prove a GPU carica e MetalFX attivo se nel preset.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-11.1 — Fused MLP

**Ambito:** **[CANDIDATO]** **MLP fully fused** in threadgroup memory con cooperative tensor, ispirati a [R69][R70]: strati consecutivi senza passare dalla DRAM

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Strati consecutivi in cooperative tensor/threadgroup con limiti memoria/registri e fallback multipass.
- **Verifica/accettazione:** Parità numerica, shapes/tails e barrier legality; frame+packing contro tensor baseline.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-11.2 — Quantizzazione

**Ambito:** **[CANDIDATO]** **Quantizzazione** INT8/FP8 (quando l'OS lo consente) per tutte le reti del frame, con fallback FP16

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Calibrazione representative e scale per tensore/canale, dtype realmente supportato e FP16 fallback.
- **Verifica/accettazione:** Outlier, saturazione, qualità temporale e bandwidth; disponibilità verificata su SDK/device.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-11.3 — Hybrid upscaler

**Ambito:** **[CANDIDATO]** **Upscaler ibrido rete + soluzioni in forma chiusa**, sulla linea del PSSR 2026 [R75]: la rete fa solo ciò che le regole analitiche non sanno fare

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Separare reprojection/rejection analitiche da residual network, senza attribuire paper non verificati al prodotto.
- **Verifica/accettazione:** Ablation rete/regole, disocclusioni e input latency; confronto MetalFX alla stessa qualità.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-11.4 — NTC access

**Ambito:** **[CANDIDATO]** Compressione neurale delle texture ad accesso casuale [R71] vs a blocchi [R60]; materiali neurali [R72]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Codec random/block e material network con cache latenti bounded, mip/filter espliciti.
- **Verifica/accettazione:** Accesso sparso, anisotropy e detail in motion; IO→sample costo completo.
- **Dipendenze specifiche:** F30.3.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-11.5 — Supersampling

**Ambito:** **[CANDIDATO]** **Supersampling neurale proprietario** [R73] addestrato con MLX sui dati di Phosphor (F31)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Training dati F31 con target reference e export runtime F30, seed/split congelati.
- **Verifica/accettazione:** Holdout per scena, temporal artifacts e costo per risoluzione; baseline classica mantiene supporto.
- **Dipendenze specifiche:** F31.2.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-11.6 — GEMM traversal

**Ambito:** **[CANDIDATO]** Tile di GEMM ≥ 32×32, traversal Morton/Hilbert dei threadgroup, barriere ogni poche iterazioni K (S-NA-1, S-NA-3)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Sweep tile legali e Morton/Hilbert dove locality giustifica, K blocking senza sync ridondanti.
- **Verifica/accettazione:** Correctness tails, working set, barrier overhead e frame gain, non solo TFLOPS.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-11.7 — Accelerator trace

**Ambito:** **[CANDIDATO]** Utilizzo dei Neural Accelerator letto in Metal System Trace per ogni rete (S-NA-1)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Raccogliere counter/trace disponibili con provenance e distinguere GPU neural da ANE.
- **Verifica/accettazione:** Unavailable resta tale; utilization target originario è ipotesi diagnostica, non numero da inventare.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-11.8 — Dtype matrix

**Ambito:** **[CANDIDATO]** Tipi di dato per versione di OS (BF16, INT8/INT4, FP8) con fallback (S-NA-2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Tabella OS/device/operator per BF16/INT8/INT4/FP8, converter e fallback precompilati.
- **Verifica/accettazione:** Capability mask e model load rifiutano combinazioni illegali prima del dispatch.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-11.9 — ML budget

**Ambito:** **[CANDIDATO]** Budget ML nel frame e nel SoC: scegliere GPU tensor, CPU o ANE solo fra implementazioni compatibili e misurate, includendo deadline, conversioni e contesa; ridurre o differire il lavoro se nessun percorso rispetta il budget (S-NA-4, S-ANE-1)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Admission tra implementazioni valide con deadline, copies/interference, quota lavoro e skip-safe fallback.
- **Verifica/accettazione:** Bursts e thermal drift, output tardivo e p99; scheduler semplice prima di generale.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-11.10 — ANE anticipation

**Ambito:** **[CANDIDATO]** ANE con output anticipati e layout ottimizzati: confrontare predittori analitici, Core ML, GPU tensor e risultati del laboratorio F30.7; latenza completa a GPU carica, scadenza/confidenza e costo energetico (H6/H12)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Predict ahead da snapshot versionato con TTL/confidence e baseline analitica; confronto Core ML e lab separati.
- **Verifica/accettazione:** Camera/scene change, inferenza stale e GPU busy; device realmente osservato e energia se disponibile.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-11.11 — Material MLP

**Ambito:** **[CANDIDATO]** Compressione neurale e piccoli MLP specializzati per materiale: valutazione congiunta con OPT-8.11, cache dei latenti, raggruppamento dei campioni e qualità temporale; accettare solo il punto Pareto misurato (H9)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Raggruppare campioni per materiale/latenti, cache bounded e pesi versionati.
- **Verifica/accettazione:** Materiali rari, cache thrash e temporalità; Pareto con OPT-8.11 senza obbligo reciproco per baseline.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Quantizzazione salva banda ma rovina temporalità; più tensor utilization può rallentare raster. Ripiego FP16/decoder classico e timing differito.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R92–R94, R103, R105–R106; H6/H9/H12. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
