# OPT-12 — Autotuning, path tracing, splatting

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Piano dettagliato anticipato; implementazione e misure ancora da eseguire. **Attivazione:** `opportunita`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

Ricerca bounded e holdout impediscono overfit. Piani statici restano fallback, e trasformazioni eterogenee richiedono equivalenza e ownership esplicite. Ogni ramo ha consumer e selezione indipendenti.

## Prerequisiti e confini

Baseline richieste: [F29](F29.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.
Integrazione successiva con [OPT-4](OPT-4.md): Use selected optimizations only when measured and adopted; the baseline does not require completing this OPT catalog.
Integrazione successiva con [F30](F30.md): Required by the corresponding implementation package only; does not block the initial baseline or unrelated candidates.
Integrazione successiva con [F31](F31.md): Required by the corresponding implementation package only; does not block the initial baseline or unrelated candidates.
Integrazione successiva con [F32](F32.md): Required by the corresponding implementation package only; does not block the initial baseline or unrelated candidates.
Integrazione successiva con [F33](F33.md): Required by the corresponding implementation package only; does not block the initial baseline or unrelated candidates.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `tools/autotune/ (F29)`.
- `src/renderer/tuning_profile.* (F29)`.
- `shaders/path_trace.metal (F32)`.
- `shaders/splat_render.metal (F33)`.
- `src/rendergraph/optimizer/`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

.5/.6 → .1/.9 → .10; .2/.3 solo PT, .4/.7 solo splat; .8 dopo e-graph realmente utile.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

Random/grid baseline contro Bayesian a pari budget, rumore e chip disponibili; equivalenza exact prima della qualità approssimata.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-12.1 — Search algorithms

**Ambito:** **[CANDIDATO]** **Autotuning come ricerca**: esplorazione dello spazio dei parametri (tile, threadgroup, meshlet, raggi, varianti) con tecniche da compilatori [R77][R78] e ottimizzazione bayesiana [R79], risultati per chip e per versione di OS

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Spazio constraint-aware con budget tentativi, prune dei candidati illegali e baseline inclusa.
- **Verifica/accettazione:** Confronto random/grid/Bayesian su stesso budget, holdout e costo ricerca ammortizzato.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-12.2 — Path guiding

**Ambito:** **[CANDIDATO]** **Path guiding in tempo reale** [R80] per il path tracing: meno campioni a parità di rumore

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Distribuzione direzionale appresa dai campioni con PDF e MIS, aggiornamento bounded.
- **Verifica/accettazione:** Scena caustiche/diffuse e cambio luce, convergenza contro reference e training overhead.
- **Dipendenze specifiche:** F32.1.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-12.3 — PT cache/reservoir

**Ambito:** **[CANDIDATO]** Path tracing con ORCA [R40] e reservoir splatting [R27]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Varianti ORCA e splatting con estimator esplicito e invalidazione; non cumulare tecniche prima delle ablation.
- **Verifica/accettazione:** Bias/variance, reuse temporale e costo totale; fallback path tracer reference.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-12.4 — Splat alternatives

**Ambito:** **[CANDIDATO]** **Splatting senza ordinamento** [R81], adatto a TBDR e iPad; ordinamento stabile [R82] per la qualità; splat nel ray tracing [R83]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Sorted stabile come reference, approssimazioni order-independent e RT come varianti separate.
- **Verifica/accettazione:** Alpha/intersections e motion, costo binning/sort/trace; qualità per target dichiarata.
- **Dipendenze specifiche:** F33.1.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-12.5 — Cost priors

**Ambito:** **[CANDIDATO]** L'autotuning parte dal modello di costo di OPT-0 per ridurre lo spazio di ricerca

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Modello OPT-0 riduce candidati ma non esclude baseline o sostituisce la misura.
- **Verifica/accettazione:** Errore modello/uncertainty e casi fuori dominio; winner rivalutato sul frame.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-12.6 — Family vs chip

**Ambito:** **[CANDIDATO]** Parametri separati per famiglia (percorsi di codice) e per chip misurato (valori numerici) (sezione 16)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Feature path da capability, parametri numerici da misure modello/OS/workload.
- **Verifica/accettazione:** Override family non cambia hardware label; profili sconosciuti non ereditano tuning arbitrario.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-12.7 — Splat tile

**Ambito:** **[CANDIDATO]** Splat blending nella tile con imageblock (S-TBDR-1, S-TBDR-7)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Imageblock/ROG con capacity e compositing bounded, fallback sorted buffers.
- **Verifica/accettazione:** Overflow, tile edges e image equivalence/errore, costo spill misurato.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-12.8 — Heterogeneous compiler

**Ambito:** **[CANDIDATO]** Estendere il compilatore algebrico di OPT-4.17 alle varianti CPU/SME/GPU/ANE semanticamente compatibili; candidati offline e piano ibrido con certificato di dipendenze/ownership, nessuna equivalenza approssimata implicita (H4)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** IR variante CPU/SME/GPU/ANE con semantica e costi conversione; certificato DAG/ownership per piano.
- **Verifica/accettazione:** Piccole equivalenze, error bounds e lifetime check; offline bounded, fallback piano statico.
- **Dipendenze specifiche:** OPT-4.17, F23.7, F30.6.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-12.9 — Robust tuning

**Ambito:** **[CANDIDATO]** **Autotuning robusto [EDGE]**: ricerca bayesiana vincolata [R104] con misure rumorose, scene holdout, budget di esperimenti e verifica del guadagno su chip diversi; evitare sovra-adattamento al testbench (H10)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Bayesian constrained con repliche adattive, validation holdout e limite esperimenti.
- **Verifica/accettazione:** Overfit/noise controls, cross-device solo se device disponibile; una macchina non prova generalizzazione.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-12.10 — Dynamic policy

**Ambito:** **[CANDIDATO]** Confronto piani statici, selezione per classe di scena e controllo a orizzonte breve; costo delle decisioni, isteresi, invalidazione di profili e storia, ripiego su piano misurato quando il modello deriva (H10)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Classi scena e isteresi prima di predictive horizon; profilo invalidato su drift/capability change.
- **Verifica/accettazione:** Transition cost, history reset, deadline e overhead; ripiego statico quando stima incerta.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Sovra-adattamento e solver costoso. Ripiego piano offline migliore verificato, flag indipendenti per PT/splat e IR senza riscritture approssimate implicite.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R100, R104; H4/H10; R77–R83 da verificare. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
