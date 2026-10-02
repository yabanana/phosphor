# OPT-9 — CPU, thread e memoria unificata

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Piano dettagliato anticipato; implementazione e misure ancora da eseguire. **Attivazione:** `opportunita`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

Pool semplice e snapshot corretti sono baseline. Scheduler eterogeneo ammette solo implementazioni semanticamente equivalenti; stime includono interference e conversioni. Kernel non è una dipendenza del runtime distribuibile.

## Prerequisiti e confini

Baseline richieste: [F23](F23.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.
Integrazione successiva con [F30](F30.md): Contesa e scheduler ANE solo dopo F30.6; misure CPU/GPU precedenti.
Integrazione successiva con [OPT-4](OPT-4.md): Use selected optimizations only when measured and adopted; the baseline does not require completing this OPT catalog.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `src/core/worker_pool.*`.
- `src/runtime/jobs.* (F23)`.
- `src/runtime/data_exchange.* (F23)`.
- `bench/cpu_jobs/ (F23)`.
- `bench/soc/`.
- `bench/os_lab/ (F23)`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

.5/.6/.7/.8 diagnosi → .9 matrice → .1/.2/.10 se trigger → .11/.12 specifiche; .13 dopo dossier F23.9. .3/.4 consumer separati.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

DAG replay più runtime conteso, falsi sharing e p99; matrici pairwise poi triplette solo rappresentative, incluse inferenze MetalFX se in uso.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-9.1 — Fibers

**Ambito:** **[CANDIDATO]** Job system a fiber [R64] confrontato con enkiTS sulla topologia reale (super/performance/efficiency core)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Continuazioni e wakeup bounded contro enkiTS/GCD, niente fiber che occupa worker aspettando GPU.
- **Verifica/accettazione:** Catene lunghe, fan-out, cancel e starvation; overhead/frame/deadline vs pool semplice.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-9.2 — CPU/GPU placement

**Ambito:** **[CANDIDATO]** **Bilanciamento adattivo CPU↔GPU** (idea Phosphor): lavori leggeri (culling di luci, selezione LOD, animazione) spostati tra CPU e GPU sulla base del costo completo di lancio, layout, copie necessarie, sincronizzazione e contesa; isteresi e baseline fissa (H1/H3)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Implementazioni equivalenti con costi packing/dispatch/sync, hysteresis e classi statiche baseline.
- **Verifica/accettazione:** Carico che cambia, output versionato e numerico; decision overhead e contesa inclusi.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-9.3 — ANE fuori frame

**Ambito:** **[CANDIDATO]** **Neural Engine per le reti fuori dal frame** (animazione appresa, audio, IA, previsione dello streaming) per lasciare liberi GPU e CPU

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Scegliere consumer con tolleranza al ritardo, Core ML adapter e fallback analitico.
- **Verifica/accettazione:** Deadline miss/stale output, GPU+IO busy e dispositivo inferenza osservato o non verificato.
- **Dipendenze specifiche:** F30.6.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-9.4 — Extrapolation

**Ambito:** **[CANDIDATO]** **Extrapolazione del frame** [R74] come alternativa a bassa latenza all'interpolazione MetalFX

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Prototipo con motion/depth e hole handling, present pipeline separata dal tick.
- **Verifica/accettazione:** Input latency e disocclusioni, errore visivo rispetto a vero frame; fallback render normale.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-9.5 — QoS/waits

**Ambito:** **[CANDIDATO]** Classi QoS verificate con Instruments per ogni thread; nessuno spin-wait (S-CPU-1, S-PWR-3)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Inventario thread/urgenza, notifiche invece di spin e budget pool unico.
- **Verifica/accettazione:** Trace attese/wakeup, priorità inversione e idle power; nessuna assunzione fixed core.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-9.6 — NEON/SME

**Ambito:** **[CANDIDATO]** Cicli caldi in NEON (SoA); operazioni su matrici in batch via Accelerate/SME (S-CPU-2, S-CPU-3)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Hot loops reali SoA, SIMD tail e shape batching; Accelerate baseline per matrici.
- **Verifica/accettazione:** Numeric reference, packing incluso e workload piccolo/grande; feature gate e ABI correctness.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-9.7 — Bandwidth windows

**Ambito:** **[CANDIDATO]** Contesa di banda CPU/GPU misurata (B-09) e job CPU pesanti pianificati fuori dalle finestre critiche della GPU (S-MEM-4)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Misurare interferenza e spostare solo job differibili con deadline/age espliciti.
- **Verifica/accettazione:** p99 CPU/GPU e starvation, stessa quantità lavoro per confronto; serial baseline inclusa.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-9.8 — Frames in flight

**Ambito:** **[CANDIDATO]** Frame in volo scelti dalla latenza misurata (B-28) (S-SYNC-4)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Sweep 1/2/3 con stesso workload/display e input trace, capacità ring coerenti.
- **Verifica/accettazione:** Throughput e latency distribuiti; present time proxy distinto da misura input-to-photon.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-9.9 — Interference matrix

**Ambito:** **[CANDIDATO]** **Matrice di interferenza del SoC**: estendere B-09/B-23/B-24 con coppie/triplette CPU NEON/SME, GPU raster/compute, ANE e I/O; sweep del working set, granularità e concorrenza, tempi p50/p95/p99 e potenza; modello per chip/OS con incertezza (H2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Pair/triple CPU/GPU/ANE/IO per working set e regime, protocollo replicato e incertezza.
- **Verifica/accettazione:** Previsione holdout, thermal/DVFS annotati e timeout bounded; matrice non verità universale.
- **Dipendenze specifiche:** F30.6. La matrice CPU/GPU/SME può partire subito; parte ANE e chiusura completa dopo F30.6.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-9.10 — Bandwidth scheduler

**Ambito:** **[CANDIDATO]** **Scheduling a budget di banda e deadline [EDGE]**: ammissione dei job memory-bound, earliest-finish con interferenza, aging dei lavori differibili; confronto con FIFO/work stealing e serializzazione sul corpus reale, inclusi costo dello scheduler e starvation (H1/H2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Admission cost+deadline, earliest finish stimato e aging, prima lista prioritaria statica.
- **Verifica/accettazione:** DAG causale, overload, starvation e overhead; frame benefit misurato prima di adozione.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-9.11 — Layout compiler

**Ambito:** **[CANDIDATO]** **Compilatore di layout [EDGE]**: scegliere SoA/AoSoA, packing e buffer condivisi/doppi per classe di dati; confronto produzione→consumo e verifica numerica, retirement ed eviction sotto pressione (H3)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Varianti SoA/AoSoA/packed generate offline con schema ownership; runtime sceglie layout stabile.
- **Verifica/accettazione:** Roundtrip/FP error, stale generation, eviction e producer→consumer completo; manual layout baseline.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-9.12 — SME codegen

**Ambito:** **[CANDIDATO]** **Kernel SME generati per shape reali [EDGE]**: NEON vs Accelerate/BNNS vs SME vs GPU, con packing e transizioni ABI inclusi; varianti offline solo dove il guadagno end-to-end è misurato (H7, R95–R96/R109)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Generare poche shape reali con dispatcher e versioni ISA/ABI, packing amortizzato misurato.
- **Verifica/accettazione:** NEON/Accelerate/GPU reference, tails e batch piccoli; niente uso presunto dell’AMX privato.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-9.13 — OS experiment

**Ambito:** **[CANDIDATO]** **Esperimento scheduler/kernel condizionale [EDGE]**: implementare e misurare le politiche di F23.9 nel simulatore e nel runtime; se il dossier dimostra fattibilità e vantaggio residuo, progettare e prototipare in un laboratorio OS separato, confrontando con macOS standard. Esito ammesso: adozione user-space, esperimento kernel o esclusione motivata; niente dipendenza kernel implicita per il prodotto (H11)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Simulatore e runtime user-space prima, eventuale prototipo kernel isolato dal progetto distribuito.
- **Verifica/accettazione:** Limite pubblico riproducibile, accessi/fattibilità dimostrati e confronto stock; esclusione motivata è esito valido.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Contesa non stazionaria e modello sovra-adattato. Ripiego policy statica per classe, layout stabile e pool semplice; risultati del kernel di laboratorio non certificano prestazioni del prodotto stock.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R87–R99, R109; H1/H2/H3/H6/H7/H11. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
