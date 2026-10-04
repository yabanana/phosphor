# OPT-2 — Shader, pipeline e occupancy

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Piano dettagliato anticipato; implementazione e misure ancora da eseguire. **Attivazione:** `opportunita`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

Baseline F8 congelata, shader prioritario dal frame reale. Specializzazione e precisione cambiano solo dove contratto/qualità lo consentono; occupancy è diagnostica, non obiettivo indipendente.

## Prerequisiti e confini

Baseline richieste: [F13](F13.md) (frame con luce reale), [OPT-0](OPT-0.md) e OPT-2.0 (equivalenza generica/varianti su contenuti texturizzati, derivate analitiche nel forward). Spostata dopo F13 il 2026-10-04: sul frame F8 i nostri shader pesano ~0,5 ms a 1080p, MetalFX il 69–83% dei frame temporali e Many Lights è il ciclo luci di F11 ([revisione](../research/2026-10-04-post-f8-review.md)). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `shaders/`.
- `shaders/variants.def`.
- `src/pipeline/`.
- `src/diagnostics/`.
- `tools/variant_gen/`.
- `bench/shader_tuning/ (nuovo solo se selezionato)`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

Baseline → .3/.5/.6 diagnosi → scegliere una fra .4/.7/.8/.9/.10 → .2/.11; .1 richiede errore visivo esplicito.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

Una trasformazione per volta, confronto generica/variante e costi register/spill se osservabili. Instrumentation assente dichiarata, non sostituita da occupancy inventata.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-2.1 — Shader LOD

**Ambito:** **[CANDIDATO]** **Shader LOD automatico**: varianti semplificate degli shader generate con tecniche di semplificazione automatica [R5][R6] e usate dove l'errore non si vede (oggetti lontani, riflessioni, GI, tier bassi)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Semplificazione offline con errore per materiale/distanza, feature key e fallback shader completo.
- **Verifica/accettazione:** Speculari e silhouette in movimento, qualità holdout e costo frame; non chiamare equivalenti varianti approssimate.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-2.2 — Profile specialization

**Ambito:** **[CANDIDATO]** **Specializzazione guidata dal profilo**: registrare durante i test quali combinazioni di feature compaiono davvero e generare varianti (function constant) solo per quelle (O11)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Raccogliere frequenza combinazioni, selezionare top set e compilare AOT; generic miss sempre valido.
- **Verifica/accettazione:** Corpus non visto, cold cache, miss e crescita archivio; nessuna compilation sul thread render.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-2.3 — Roofline

**Ambito:** **[CANDIDATO]** **Roofline automatica**: strumento che da counter heap e contatori calcola intensità aritmetica e collo di bottiglia per pass e lo mostra in ImGui [R4]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Riutilizzare soc_model/work/byte report e aggiungere provenance per counter o stima.
- **Verifica/accettazione:** Known kernels e unità, limiti inferiori plausibili; no conclusione certa se mancano counter.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-2.4 — ILP

**Ambito:** **[CANDIDATO]** Riscrittura ILP-friendly dei kernel più caldi (più catene indipendenti, niente `float4` che maschera dipendenze) [R7]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Spezzare dipendenze reali e riordinare istruzioni indipendenti senza gonfiare working set.
- **Verifica/accettazione:** Reference numerica e shader compiler output, frame A/B con register pressure inclusa.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-2.5 — Register audit

**Ambito:** **[CANDIDATO]** Censimento dei registri vivi per riga (Xcode 26.4+) per ogni shader caldo; riduzione dei picchi (S-OCC-1)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Individuare variabili vive/temporanei sullo shader caldo con strumenti disponibili; shorten lifetime prima di riscrivere.
- **Verifica/accettazione:** Differenza register/private memory quando osservabile e tempo pass/frame; dati mancanti etichettati.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-2.6 — Occupancy

**Ambito:** **[CANDIDATO]** Tabella occupancy target + causa di throttling per shader, con correzione mirata (S-OCC-2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Correlare resident groups, stalls, bandwidth e launch geometry per shader/workload.
- **Verifica/accettazione:** Sweep mostra punto migliore ripetibile; nessun target percentuale universale adottato.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-2.7 — Half

**Ambito:** **[CANDIDATO]** Conversione sistematica a `half` con suffisso `h`, verificata dai test visivi (S-ALU-3)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Convertire solo range numerici sicuri, mantenendo depth/position e accumuli sensibili in float.
- **Verifica/accettazione:** Estremi/HDR/roughness, NaN e clip temporali; errore prima della soglia di accettazione.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-2.8 — Strength reduction

**Ambito:** **[CANDIDATO]** Strength reduction: niente div/mod interi nei cicli caldi, trascendentali `half`/`fast::` dove accettabile (S-ALU-4)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Invarianti fuori loop, reciprocal/bitmask soltanto con dominio dimostrato; fast math opt-in per pass.
- **Verifica/accettazione:** Zero/negativi/overflow e precisione nei casi limite; vantaggio nel workload corrente.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-2.9 — Threadgroup sweep

**Ambito:** **[CANDIDATO]** Sweep delle dimensioni di threadgroup per ogni kernel e per chip, risultati salvati per l'autotuning (S-OCC-3)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Varianti legali da limiti pipeline, dimensioni e tail reali; cache per shader/device/OS.
- **Verifica/accettazione:** Input non multipli, immagini/readback e p99; winner del microbench riconfermato nel frame.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-2.10 — SIMD compaction

**Ambito:** **[CANDIDATO]** Compattazioni e riduzioni riscritte con intrinsics SIMD-group (S-SIMD-1)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Ballot/reduce/scan per gruppi con fasi globali separate e ordine dichiarato.
- **Verifica/accettazione:** N=0/1/nonmultiplo/capacity, reference scan e overflow; nessuna ipotesi forward progress inter-group.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-2.11 — Variant pruning

**Ambito:** **[CANDIDATO]** Tempo di compilazione e numero di varianti misurati; pruning delle varianti mai usate

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Inventario usate/unused e budget build/archive, preservando generic fallback e scenari non osservati.
- **Verifica/accettazione:** Cold startup e replay holdout, log miss e riduzione manutenzione senza perdita funzionale.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Ottimizzazioni FP possono alterare stabilità; meno registri può aggiungere calcolo. Ripiego della singola variante e matrice di precisione per canale.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R5–R7 da verificare, R91–R92. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
