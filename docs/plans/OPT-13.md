# OPT-13 — Scalabilità ed energia

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Piano dettagliato anticipato; implementazione e misure ancora da eseguire. **Attivazione:** `opportunita`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

Energia, qualità e latency misurate in regime sostenuto. macOS governa clock e scheduling; il motore controlla carico/cadenza/QoS. Disponibilità batterie/sensori determina quali claim sono verificabili.

## Prerequisiti e confini

Baseline richieste: [F28](F28.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.
Integrazione successiva con [OPT-9](OPT-9.md): Matrice completa di interferenza quando disponibile.
Integrazione successiva con [F29](F29.md): Required by the corresponding implementation package only; does not block the initial baseline or unrelated candidates.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `src/renderer/quality_settings.* (F28)`.
- `src/diagnostics/thermal_state.* (F28)`.
- `src/platform/apple/display_link.mm (F28)`.
- `src/platform/apple/video_decode.mm (nuovo candidato)`.
- `bench/soc/b27*`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

.4/.5 baseline → .1/.3/.6 → .2; .7 consumer video, .8 solo limite non risolto da isteresi.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

Preset identico, alimentazione/display/termica registrati, durata adeguata e misure non privilegiate ove disponibili; assenza watt non diventa stima presentata come misura.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-13.1 — Race-to-idle

**Ambito:** **[CANDIDATO]** **Race-to-idle vs frequenza costante**: quale strategia consuma meno per frame a 60 fps su ciascun chip

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Confrontare burst+idle e pacing uniforme a stesso output/qualità; nessun tentativo di fissare clock privati.
- **Verifica/accettazione:** Joule/frame e p99/latency se misurabili, soak e DVFS annotati; non inferire autonomia da gpu-ms.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-13.2 — Energy quality

**Ambito:** **[CANDIDATO]** **Qualità guidata dall'energia** (idea Phosphor): il preset si adatta ai watt disponibili, non solo ai millisecondi

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Controller da segnali disponibili con qualità minima e isteresi, fallback preset statico.
- **Verifica/accettazione:** Transitori e scene difficili, budget qualità e flapping; vantaggio misurato rispetto a DRS semplice.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-13.3 — Adaptive cap

**Ambito:** **[CANDIDATO]** Frame cap intelligente: fps adattati al contenuto (menu, scene statiche) e al display (ProMotion)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Scene/menu state e display refresh determinano fps cap, input risveglia entro limite.
- **Verifica/accettazione:** UI animata e input improvviso, idle power e latency; nessun blocco gameplay non previsto.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-13.4 — Power measures

**Ambito:** **[CANDIDATO]** Misure `powermetrics` per ogni preset e ogni chip disponibile (B-27) (S-PWR-1)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Protocollo con powermetrics quando accessibile o strumenti pubblici disponibili, unità/provenance.
- **Verifica/accettazione:** Dati mancanti segnati unavailable; dispositivo senza batteria non certifica +autonomia.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-13.5 — Sustained presets

**Ambito:** **[CANDIDATO]** Preset basati sul regime termico, non sui primi secondi (S-PWR-2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Soak a workload fisso e transitori, scegliere qualità per plateau termico.
- **Verifica/accettazione:** Nessun target derivato dal solo cold run; Air/iPhone non verificati sul desktop Max.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-13.6 — Display pacing

**Ambito:** **[CANDIDATO]** Frame pacing a 2 bucket con `CAMetalDisplayLink` (S-DISP-1)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** CAMetalDisplayLink e queue depth controllati, periodi coerenti col refresh effettivo.
- **Verifica/accettazione:** Bucket e dropped presents su display disponibile, occlusion/minimize/refresh change.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-13.7 — Media engine

**Ambito:** **[CANDIDATO]** Video in gioco tramite media engine senza copie (S-DISP-3), da verificare

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Decode via framework supportato e CVPixelBuffer/IOSurface import con lifetime esplicito.
- **Verifica/accettazione:** Color/range HDR, frame reorder, resize e copies contate; zero-copy soltanto se percorso provato.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-13.8 — Predictive controller

**Ambito:** **[CANDIDATO]** **Controllo predittivo del budget [EDGE]**: stimare pressione termica/banda dalle osservazioni disponibili e selezionare qualità, concorrenza e lavori differibili; confronto con isteresi semplice, soak prolungato e transitori alimentazione/batteria, senza presumere controllo diretto di clock o core (H2/H10)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Modello breve orizzonte su osservazioni termiche/banda, costi/uncertainty e rate limit.
- **Verifica/accettazione:** Holdout workload, drift e overload; p99/qualità/energia vs isteresi semplice con overhead incluso.
- **Dipendenze specifiche:** F28.6, F29.5.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Feedback termico lento e controllo oscillante. Ripiego cap/preset statico con hysteresis; non tenere attivo hardware per il solo utilizzo.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R87, R91; H2/H10. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
