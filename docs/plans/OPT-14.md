# OPT-14 — QA delle prestazioni e ottimizzazione continua

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Piano dettagliato anticipato; implementazione e misure ancora da eseguire. **Attivazione:** `opportunita`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

Automazione propone e misura sullo SHA esatto; controlli di qualità e baseline non sono modificati per far passare la patch. Nuovo OS/chip crea coorte separata, non reset silenzioso delle regressioni.

## Prerequisiti e confini

Baseline richieste: [F35](F35.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `tools/qa/ (F35)`.
- `tools/perf_table.py`.
- `docs/perf-log.md`.
- `docs/opt-log.md`.
- `docs/APPLE_SOC_PLAYBOOK.md`.
- `.github/workflows/ci.yml`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

.1/.2 sopra F35 → .5/.6 manutenzione; .3 workflow isolato, .4 telemetria solo opt-in.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

Dataset sintetico con regressioni note e falsi positivi misurati prima di sostituire soglie semplici; nessun auto-merge introdotto implicitamente.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-14.1 — FLIP temporale

**Ambito:** **[CANDIDATO]** **FLIP** [R84] per il confronto delle immagini di ogni ottimizzazione, con soglie per tier, affiancato alle metriche temporali di F8.7/F13.6: ghosting, flicker, disocclusioni e recupero della storia; adozione solo se supera entrambe le verifiche

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Integrare metrica a versione fissata con ROI e metriche ghosting/flicker/recovery.
- **Verifica/accettazione:** Corruzioni di colore/history note devono fallire; soglie legate a preset e display transform.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-14.2 — Change point

**Ambito:** **[CANDIDATO]** **Rilevamento statistico delle regressioni** (change point detection [R85]) sui dati dei perf bot invece di soglie fisse

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Modello per coorte device/OS/preset con rumorosità e history minima, confronto soglie fisse.
- **Verifica/accettazione:** Regressioni/shift termico simulati, precision/recall e false alarm; fallback soglia semplice.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-14.3 — Perf engineer

**Ambito:** **[CANDIDATO]** **Perf engineer automatico**: ciclo con Claude Code in locale (cattura → contatori → ipotesi → patch → misura → PR) sui pass che superano il budget

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Capture→hypothesis→isolated patch→tests→A/B→PR concreta, con log di fallimenti e rollback.
- **Verifica/accettazione:** SHA immutabile, nessuna modifica ai gate per auto-pass; merge/pubblicazione non automatici.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-14.4 — Fleet metrics

**Ambito:** **[CANDIDATO]** Telemetria opzionale delle prestazioni per modello di Mac per aggiornare preset e autotuning

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Raccolta opt-in minimale, aggregati per chip/OS e profilo applicability, schema versionato.
- **Verifica/accettazione:** Opt-out/offline/delete, input malformed e nessun tuning trasferito senza verifica locale.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-14.5 — OS/chip rerun

**Ambito:** **[CANDIDATO]** Suite `bench/` rieseguita automaticamente a ogni nuova versione di macOS e su ogni nuovo chip

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Matrix runner disponibili con suite pertinente, baseline precedente conservata e manifest nuovi.
- **Verifica/accettazione:** Stesso corpus, shader/SDK drift dichiarato e device unavailable non segnato verde.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-14.6 — Research refresh

**Ambito:** **[CANDIDATO]** Playbook aggiornato a ogni WWDC e a ogni nuovo chip (M6?)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Rivedere fonti primarie/SDK, distinguere announce/available/measured e collegare alle ipotesi coinvolte.
- **Verifica/accettazione:** Nessun task attivato automaticamente da WWDC; aggiornare solo contratti invalidati o esperimenti utili.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Perf bot impara il rumore o adotta regressioni locali. Ripiego controllo A/B/A manualmente riproducibile e profilo precedente, modello predittivo sempre distinguibile dai dati.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R104; R84–R85 da verificare. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
