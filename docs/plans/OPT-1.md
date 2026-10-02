# OPT-1 — Memoria, grafo e banda come problema di ottimizzazione  (OPT-1.5 e OPT-1.7 parziali; T0 non misurato)

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Audit dettagliato della fase consegnata e dei residui. **Attivazione:** `audit`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

Il piano è ottimizzazione offline, applicata solo con chiave compatibile e validata per misura. DP/annealing già adottati prevalgono sull’ipotesi MILP originaria. Off/Conservative rimane il riferimento.

## Prerequisiti e confini

Baseline richieste: [F2](F2.md), [F4](F4.md), [OPT-0](OPT-0.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `src/rendergraph/optimizer/`.
- `src/rendergraph/aliasing.*`.
- `src/rendergraph/graph_lint.*`.
- `src/rendergraph/graph_budget.*`.
- `tools/graph_opt/`.
- `tools/graph_select.py`.
- `shaders/graph-plans.json`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

.1 → .2/.3/.4 → .6/.8/.9/.10; .5/.7 sono residui da riaprire su carichi reali.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

I risultati sintetici storici non bastano per adottare nuovi piani F6–F8; applicazione al frame reale tramite OPT-4.15 soltanto dopo baseline.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-1.1 — Ricerca offline

**Ambito:** **Render graph risolto come problema di ottimizzazione**: ordinamento dei pass, aliasing e fusione dei pass TBDR formulati come programma lineare intero misto (MILP), risolto offline per ogni preset di qualità; a runtime si carica il piano ottimo invece di usare euristiche greedy [R1][R2]. Stesso approccio che Checkmate usa per i tensori [R3] e che [R12] usa per le triangle strip

**Stato di pianificazione:** Audit; task già spuntato nella roadmap.

- **Implementazione:** Audit DP/annealing contro compilatore reale e lower bound; invalidazione dei nomi/accessi dopo modifiche.
- **Verifica/accettazione:** Enumerazione piccole istanze, mai peggio del greedy, timeout bounded e fallback piano assente.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-1.2 — Rematerializzazione

**Ambito:** **Rematerializzazione TBDR** (idea Phosphor, ispirata a [R3]): per ogni risorsa il grafo sceglie se scriverla in DRAM o **ricalcolarla nella tile** quando serve, minimizzando i byte (O1). Il visibility buffer è un caso particolare di questa idea: generalizzarla a tutti i segnali economici

**Stato di pianificazione:** Audit; task già spuntato nella roadmap.

- **Implementazione:** Duplicare solo funzioni pure con input e precisione equivalenti; conteggiare lavoro e byte evitati.
- **Verifica/accettazione:** Immagine esatta per velocity/CoC, costo completo e nessuna side effect duplicata.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-1.3 — Colorazione lifetime

**Ambito:** Aliasing ottimo con colorazione di grafi degli intervalli di vita, vincolata ai formati che preservano la compressione lossless (S-TEX-2)

**Stato di pianificazione:** Audit; task già spuntato nella roadmap.

- **Implementazione:** Conservare conflitti e requisiti fisici degli heap; algoritmo deve rispettare allineamenti e ordinamento.
- **Verifica/accettazione:** Grafi casuali, limiti inferiori noti, alias poison e no regressione memoria vs greedy.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-1.4 — Barriere minime

**Ambito:** Barriere minime: raggruppamento, scelta intra-encoder vs coda sulla base dei costi misurati (B-18) come pesi del solver di OPT-1.1

**Stato di pianificazione:** Audit; task già spuntato nella roadmap.

- **Implementazione:** Proof del modello e tabella stage, confronto strutturale con Conservative; default non si cambia per assenza di glitch.
- **Verifica/accettazione:** Minimal sottoinsieme legale, race-model e test dipendenze; limite GPU negative control dichiarato.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-1.5 — Byte per pass

**Ambito:** Mappa dei byte DRAM per pass (contatori) e verifica del budget per tier (S-MEM-1)

**Stato di pianificazione:** Da implementare o completare; stato dettagliato nella roadmap.

- **Implementazione:** Aggiornare stime del grafo con accessi reali; contatori solo se disponibili e attribuibili.
- **Verifica/accettazione:** F4.4 mancante lascia la porzione contatori aperta; T0 non inferito da specifiche.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-1.6 — Store e memoryless

**Ambito:** Ogni intermedio candidato a `memoryless` verificato; ogni `.store` giustificato (S-TBDR-4)

**Stato di pianificazione:** Audit; task già spuntato nella roadmap.

- **Implementazione:** Lint motiva ogni store e ogni transitorio persistito; correzioni non rompono history/readback.
- **Verifica/accettazione:** Errori intenzionali rilevati, nessuno store necessario rimosso e dump coerente.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-1.7 — Working set compute

**Ambito:** Working set dei pass compute dimensionati per restare nella SLC misurata (S-MEM-2)

**Stato di pianificazione:** Da implementare o completare; stato dettagliato nella roadmap.

- **Implementazione:** Selezionare compute reale F5+, sweep chunking e adiacenza produttore/consumatore.
- **Verifica/accettazione:** Misurare frame e code, non dichiarare SLC garantita; residuo chiuso soltanto sui carichi effettivamente analizzati.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-1.8 — Overlap

**Ambito:** Sovrapposizione tra pass misurata e massimizzata riordinando geometria e fragment (S-TBDR-6)

**Stato di pianificazione:** Audit; task già spuntato nella roadmap.

- **Implementazione:** Ordinamenti legali confrontati a immagine identica; durata sovrapposta distinta da timeline esclusiva.
- **Verifica/accettazione:** Trace e tempi completi, scene holdout, nessun guadagno dedotto dalla sola somma dei pass.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-1.9 — Seconda coda

**Ambito:** Seconda coda MTL4 per compute asincrono dove B-19 mostra guadagno (S-SYNC-2)

**Stato di pianificazione:** Audit; task già spuntato nella roadmap.

- **Implementazione:** Capacità/eventi/lifetime del grafo governano placement; nessuna coda aggiunta se contesa annulla beneficio.
- **Verifica/accettazione:** Una/due code A/B/A, peak heap e p99; fallback una coda immediato.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-1.10 — Storage

**Ambito:** Anelli di upload in `shared` write-combined; tutti i dati letti spesso in `private` (S-MEM-3)

**Stato di pianificazione:** Audit; task già spuntato nella roadmap.

- **Implementazione:** Upload write-combined solo in scrittura CPU; letture CPU da buffer cached/readback dedicato.
- **Verifica/accettazione:** Byte/copie e tempo produzione-consumo; nessuna equivalenza fra UMA e sincronizzazione gratuita.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Requisito di uscita dalla roadmap (eventuali verifiche su dispositivi assenti seguono il registro hardware):

**Uscita**: obiettivo di banda raggiunto o scostamento spiegato in `docs/opt-log.md`.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Cost model ottimista per latenza e overlap. Ripiego: off/conservativo e piano misurato, rigenerazione se cambia chiave o ambito di validità.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R90, R101. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
