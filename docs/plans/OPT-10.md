# OPT-10 — Simulazione

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Piano dettagliato anticipato; implementazione e misure ancora da eseguire. **Attivazione:** `opportunita`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

Equivalenza di simulazione include stabilità/errore e latenza, non solo fps. Fixed-step e snapshot restano comuni; nuove reti/solver non sostituiscono tutti i sottosistemi per principio.

## Prerequisiti e confini

Baseline richieste: [F24](F24.md), [F25](F25.md), [F26](F26.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.
Integrazione successiva con [F30](F30.md): Motion matching su ANE quando disponibile.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `src/physics/ (F24)`.
- `src/animation/ (F25)`.
- `src/audio/ (F26)`.
- `shaders/cloth.metal (F24)`.
- `src/ml/ (F30 se selezionata)`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

Profilo → .8/.6 → .1 o .2 per solver; .3/.7 per animazione, .4 audio e .5 folla solo sui relativi carichi.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

XPBD/Jolt/animazione CPU come reference operativo, scene quantitative di stretch/energy/contacts; training separato da benchmark.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-10.1 — VBD

**Ambito:** **[CANDIDATO]** **Vertex Block Descent** [R65] come solver GPU unico per cloth, corpi morbidi e particelle: più parallelo di XPBD e stabile con poche iterazioni

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Solver vertex blocks con partizionamento, vincoli e schedule bounded; reference CPU su piccola mesh.
- **Verifica/accettazione:** Residual/energia/stabilità vs XPBD a stesso costo, collisioni e timestep; adottare per classi dove migliora.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-10.2 — Small steps

**Ambito:** **[CANDIDATO]** **Small steps** [R66]: più substep con meno iterazioni

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Sweep substep/iterazioni con fixed budget e collision detection coerente.
- **Verifica/accettazione:** Stretch, tunneling e stiff constraints; includere extra collision/broadphase nel costo.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-10.3 — Learned matching

**Ambito:** **[CANDIDATO]** **Learned motion matching** [R67] compresso per ANE o per i Neural Accelerator

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Rete distillata da database baseline, normalization e output confidenza per fallback.
- **Verifica/accettazione:** Holdout movimenti, foot sliding e transizioni, inferenza+packing vs nearest-neighbor.
- **Dipendenze specifiche:** F30.6. Baseline CPU/Metal prima; percorso ANE dopo F30.6.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-10.4 — Acustica ibrida

**Ambito:** **[CANDIDATO]** **Acustica ibrida**: codifica parametrica precomputata [R68] + ray tracing a runtime solo per le parti dinamiche

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Parametri baked statici più contributi dinamici RT, blending con limite età.
- **Verifica/accettazione:** Porta aperta/chiusa e scena mutata, audio continuity e GPU late fallback.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-10.5 — Folle

**Ambito:** **[CANDIDATO]** Animazione di folle con decompressione in compute e skinning nel mesh shader

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Batch animazioni condivise, decode GPU e skinning mesh solo se riduce traffico completo.
- **Verifica/accettazione:** Pose/motion/bounds reference, LOD e crowd heterogenea; costo RT refit incluso.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-10.6 — GPU solver SIMD

**Ambito:** **[CANDIDATO]** Solver GPU con riduzioni SIMD-group e partizionamento in threadgroup memory (S-SIMD-*)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Color/partition vincoli e riduzioni locali senza race, fasi globali con dispatch separati.
- **Verifica/accettazione:** Layout collisioni, determinismo nel perimetro, overflow e residuo numerico; no threadgroup wait.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-10.7 — Reti shape

**Ambito:** **[CANDIDATO]** Reti di animazione dimensionate per le tile ≥ 32×32 dei Neural Accelerator o spostate su ANE secondo B-22/B-23 (S-NA-1, S-ANE-1)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Batch e padding solo se ammortizzati, scegliere tensor/ANE dalla latency reale.
- **Verifica/accettazione:** Shape piccole/grandi, output stale e potenza; utilization non criterio unico.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-10.8 — CPU sim

**Ambito:** **[CANDIDATO]** Simulazione con dati SoA e NEON, QoS coerente con la deadline e scheduling verificato tramite trace, senza presumere assegnazione fissa ai P-core (S-CPU-2, R87)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** SoA/NEON e QoS per deadline, pool unico e snapshot immutabili.
- **Verifica/accettazione:** p99 sotto renderer carico, race tests e convergenza; no pinning presunto.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Cambiare solver può alterare gameplay. Ripiego Jolt/XPBD e matching baseline, con nuova variante limitata inizialmente agli effetti cosmetici.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R110, R93–R96; R65–R68 da verificare. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
