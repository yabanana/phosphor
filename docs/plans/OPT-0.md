# OPT-0 — Caratterizzazione del SoC  (manca OPT-0.2 su un T0)

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Audit dettagliato della fase consegnata e dei residui. **Attivazione:** `audit`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

Modello di costo empirico per chip/OS; il benchmark non è il renderer e non certifica automaticamente il frame. Input casuali e controlli impediscono constant folding o falsi throughput.

## Prerequisiti e confini

Baseline richieste: [F4](F4.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `bench/soc/`.
- `bench/results/`.
- `src/diagnostics/soc_model.*`.
- `tools/soc_model.py`.
- `tools/soc_roofline.sh`.
- `docs/soc-model.md`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

.1 → .2 → .3/.4 → .5/.6; riesecuzione mirata se SDK/OS o kernel cambiano.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

Conservare dati storici e provenance. Nuovi campioni non sovrascrivono risultati di un OS diverso; Apple9 forzato resta sul medesimo M5.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-0.1 — Suite SoC

**Ambito:** Suite `bench/` con i microbenchmark B-01…B-28 del playbook, eseguibile da CLI, risultati in `bench/results/<chip>-<os>.json`

**Stato di pianificazione:** Audit; task già spuntato nella roadmap.

- **Implementazione:** Preservare B-01…B-29, seed, warmup e controlli negativi; carichi sotto watchdog e un processo GPU alla volta.
- **Verifica/accettazione:** Ogni benchmark produce status e unità; negative control fallisce, validation e leaks pertinenti.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-0.2 — Copertura hardware

**Ambito:** Esecuzione su M5 Max e su almeno un T0 (M3/M4 base); poi su ogni Mac disponibile

**Stato di pianificazione:** Da implementare o completare; stato dettagliato nella roadmap.

- **Implementazione:** Eseguire native/Apple9 forzato sul device disponibile, tenendo separata la riga del chip fisico.
- **Verifica/accettazione:** OPT-0.2 rimane parziale per T0; nessun valore di throughput M3 dedotto da override.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-0.3 — Cost model

**Ambito:** **Modello di costo** Phosphor: tabella per chip con throughput ALU FP16/FP32, banda per livello, dimensione stimata di SLC, costo di barriere, dispatch, load/store, raggi/s, GEMM per tile

**Stato di pianificazione:** Audit; task già spuntato nella roadmap.

- **Implementazione:** Fit per regime e incertezza, fallback conservativo per campioni mancanti; formato versionato.
- **Verifica/accettazione:** Fixture, unità, outlier e chip sconosciuto; classifica dei candidati verificata poi sul frame.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-0.4 — Roofline

**Ambito:** Grafici roofline per chip (tetto di banda e di calcolo) su cui collocare ogni pass

**Stato di pianificazione:** Audit; task già spuntato nella roadmap.

- **Implementazione:** Collegare operazioni/byte stimati ai pass reali e al tetto misurato del dispositivo.
- **Verifica/accettazione:** Costi predetti non presentati come tempi osservati; spiegare scarti e sovrapposizioni.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-0.5 — Playbook

**Ambito:** Verifica delle voci **(ipotesi)** del playbook: correggere il playbook con i dati misurati

**Stato di pianificazione:** Audit; task già spuntato nella roadmap.

- **Implementazione:** Ogni affermazione hardware riporta fonte e perimetro del test; ipotesi falsificate restano nel log.
- **Verifica/accettazione:** Audit delle voci Misura contro JSON e protocollo; nessun numero senza origine.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-0.6 — Soglie

**Ambito:** Soglie critiche misurate: partial render (B-12), punto di thrashing dei registri (B-04), dimensione massima dell'imageblock (B-15)

**Stato di pianificazione:** Audit; task già spuntato nella roadmap.

- **Implementazione:** Conservare sweep di registri, imageblock e parameter buffer per workload preciso.
- **Verifica/accettazione:** Controllare failure silenziose tramite readback; soglie non trasferite ad altri shader/chip senza misura.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Requisito di uscita dalla roadmap (eventuali verifiche su dispositivi assenti seguono il registro hardware):

**Uscita**: modello di costo pubblicato in `docs/soc-model.md` e usato come

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Compiler folding, thermal drift e dati comprimibili falsano il tetto. Ripiego: microbenchmark più semplice con risultato CPU noto e fit etichettato come ipotesi.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R91–R96, R108. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
