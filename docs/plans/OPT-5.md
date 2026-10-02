# OPT-5 — Budget di raggi e campionamento

Aggiornato: 2026-10-02. [Indice](README.md) · [Roadmap](../ROADMAP.md) · [Metodo](METHOD.md) · [Hardware](HARDWARE_VALIDATION.md).

**Uso:** Piano dettagliato anticipato; implementazione e misure ancora da eseguire. **Attivazione:** `opportunita`.

La roadmap resta l’autorità delle caselle. Questo piano specifica il lavoro task per task; non dichiara risultati futuri. I candidati restano selezionabili e non bloccano la baseline. Le scelte dipendenti da misure seguono lo spike e il ripiego qui descritti, senza richiedere una nuova autorizzazione di routine.

## Architettura e decisioni fissate

Budget raggi è mezzo: qualità dopo filtro e tempo completo sono metriche. Estimator esplicita PDF, target, normalizzazione e bias; riordini mantengono mapping sample→pixel.

## Prerequisiti e confini

Baseline richieste: [F13](F13.md), [F14](F14.md). Sono contratti utilizzabili, non la chiusura di interi cataloghi o di certificazioni hardware esterne.

## File e ownership

Percorsi relativi alla radice del repository; `nuovo` indica una destinazione proposta. I percorsi marcati con una fase precedente sono artefatti pianificati da quella fase: vanno riusati, non duplicati. Adeguare i nomi al codice al kickoff mantenendo il contratto.

- `src/renderer/light_sampling.* (F11)`.
- `shaders/restir_di.metal (F11)`.
- `shaders/rt_common.h (F9)`.
- `src/platform/metal/acceleration_structures.* (F9)`.
- `bench/rt_sampling/ (nuovo)`.

Logica/dati e riferimenti CPU rimangono portabili. Creazione risorse e residency passano da GpuMemory, pipeline da PipelineCache, accessi GPU dal render graph. Per Rust/Bevy si attraversa soltanto il bridge versionato di F27. Il proprietario della fase integra i pacchetti nell’ordine sotto; eventuale delega ha confini per file e non consente misure GPU concorrenti.

## Ordine dei pacchetti

Profilo trace/build/filter → .11 → selezionare .3/.4/.5/.10; cambi reservoir .1/.2 richiedono reference statistico.

Ogni pacchetto è un commit revisionabile con ID della roadmap: contratto e riferimento → implementazione semplice → integrazione → prove. Non committare test falliti come fase chiusa; non cambiare i riferimenti per assorbire regressioni.

## Spike e scelte condizionali

Scene analitiche e reference convergente, poi corpus F13 con disocclusione. A/B stesso budget qualità o raggi con tempi completi.

Prima di eseguire fissare manifest, controllo e soglie secondo [METHOD](METHOD.md). Con un dato incerto si mantiene la baseline; si amplia la misura soltanto se il risultato cambia una decisione. I candidati seguono [SEQUENCING](SEQUENCING.md), con stop anticipato e massimo 1–2 settimane per un esperimento architetturale.

## Implementazione e accettazione per task

### OPT-5.1 — Reservoir production

**Ambito:** **[CANDIDATO]** **ReSTIR architettato per la produzione** [R23]: reservoir compatti in `half`, accessi coerenti; scelta dei vicini guidata dalla compatibilità [R28]; mappe di shift GRIS [R24] e ReSTIR condizionale [R25]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Packing dopo analisi dinamica dei pesi, compatibility e shift GRIS/conditional come varianti separate.
- **Verifica/accettazione:** Media/variance e overflow pesi, luce dominante e riuso; full precision estimator fallback.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-5.2 — Reservoir splat/area

**Ambito:** **[CANDIDATO]** **Reservoir splatting** [R27] e Area ReSTIR [R26] per un riuso temporale più robusto a costo minore (anche antialiasing e depth of field "gratis")

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Implementare mapping e normalizzazione del riuso area, includendo coverage e conflitti.
- **Verifica/accettazione:** Anti-alias/DoF e bias contro reference; non dichiarare effetti gratis senza costo aggiuntivo misurato.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-5.3 — Variable rays

**Ambito:** **[CANDIDATO]** **Variable Rate Ray Tracing** [R32]: raggi per pixel decisi dinamicamente da varianza, disocclusione e contenuto

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Budget da variance/disocclusion con minimo campioni e quote bounded, starvation age.
- **Verifica/accettazione:** Luce improvvisa e regioni mai campionate; qualità/tempo vs uniform, count raggi totale.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-5.4 — Light tree

**Ambito:** **[CANDIDATO]** **Campionamento delle luci più intelligente**: albero di luci con Spherical Gaussian [R29], adaptive tree splitting [R30], stochastic lightcuts [R31]: candidati migliori, meno raggi d'ombra

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Alias baseline contro gerarchia SG/splitting con PDF calcolabile e aggiornamenti incrementali.
- **Verifica/accettazione:** Istogramma/PDF, luci moving/zero-energy e costo build+sampling+shadow.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-5.5 — Ray sort

**Ambito:** **[CANDIDATO]** **Coerenza senza SER**: ordinamento software dei raggi per direzione/origine prima del trace [R33], misurato contro il reorder hardware di M3+

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Radix/binning per chiavi bounded, scatter inverse e payload minimo.
- **Verifica/accettazione:** Hit equivalenti, incoherent/coherent scene; sort+trace+scatter contro hardware traversal diretto.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-5.6 — BVH quality

**Ambito:** **[CANDIDATO]** **Qualità della TLAS**: re-braiding [R34] e unione offline delle istanze statiche piccole; BVH compatte a nodi fusi [R35] per le strutture software (proxy, audio, splat); tecniche per geometria animata massiva [R36]

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Aggregare geometria statica/proxy dove legale; software BVH ottimizzabile, layout AS Metal opaco non riscrivibile.
- **Verifica/accettazione:** Trace/build/update e instancing memory; nessuna promessa re-braiding interno API non esposte.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-5.7 — Noise filter

**Ambito:** **[CANDIDATO]** **Rumore adattato al filtro**: blue noise spazio-temporale [R42] e FAST [R43] per tutte le decisioni stocastiche; stesso numero di campioni, meno rumore residuo

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Sequenze STBN/FAST coerenti con dimensioni del campione e filtro effettivo.
- **Verifica/accettazione:** Temporal spectra/recovery e same-spp quality; evitare correlazioni che falsano estimator.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-5.8 — Traversal audit

**Ambito:** **[CANDIDATO]** `intersector` in tutti i kernel caldi, zero `intersection_query`; payload minimi; intersection function brevi (S-RT-1)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Shorten payload/intersection function e usare intersector sui percorsi supportati.
- **Verifica/accettazione:** Alpha test corretto e hit reference; contenere branching senza perdere materiali.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-5.9 — Instance strategy

**Ambito:** **[CANDIDATO]** Percorso M5: molte istanze piccole (istanze HW, allineamento 1 KB); percorso M3/M4: BLAS unite (S-RT-2)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Confrontare BLAS aggregate e piccole istanze sul M5, rispettando allineamenti interrogati.
- **Verifica/accettazione:** Memory/build/trace per scena; M3 resta non misurato, nessun default dedotto da nome famiglia.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-5.10 — Amortize AS

**Ambito:** **[CANDIDATO]** Build/refit/compaction ammortizzati su più frame secondo B-21 (S-RT-4)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Queue rebuild/refit/compact con revisioni, max age e quota lavoro; vecchia AS sicura fino a completion.
- **Verifica/accettazione:** Deformazione improvvisa e budget saturo, shadow correctness e p99.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

### OPT-5.11 — Ray cost model

**Ambito:** **[CANDIDATO]** Raggi/s coerenti e incoerenti per chip nel modello di costo; budget di raggi per tier derivato dai numeri (S-RT-3)

**Stato di pianificazione:** Candidato non attivato.

- **Implementazione:** Campionare coh/incoh e payload reali per chip/OS, incertezza e budget per segnale.
- **Verifica/accettazione:** Previsione confrontata frame reale, dati mancanti fallback conservativo e T0 esterno.
- **Evidenza da conservare:** commit e artefatti del pacchetto, comando/configurazione della prova e risultato nel log pertinente; un esito non eseguito rimane dichiarato tale.

## Verifica integrata e uscita

Applicare la [matrice di verifica per tipo di modifica](METHOD.md#matrice-di-verifica) all’ambito effettivamente toccato. Unit test verificano logica portabile, readback e golden il percorso GPU, clip temporali storia e qualità. Runner e flag ancora da creare nel piano non sono comandi già disponibili.

Sul dispositivo disponibile: modalità nativa e fallback pertinenti, budget/preset espliciti, warmup/steady-state e almeno tre repliche quando si misura una prestazione. Nessun numero nuovo è stato misurato per scrivere questo piano. `DEVELOPMENT_ACCEPTED` e `HARDWARE_CERTIFIED` sono esiti distinti; T0 fisico mancante non impedisce la fase successiva e non viene marcato verificato.

## Rischi, ripiego e revisione

In caso di errore di correttezza fermare l’adozione, ridurre al caso riproducibile e mantenere la baseline precedente. In caso di guadagno sotto rumore o costo eccessivo, registrare rimanda/scarta per il candidato; non tickare il task come implementato. Un requisito funzionale rimane da completare anche se una tecnica candidata viene scartata.

Meno raggi può aumentare bias e costo di selezione. Ripiego sampling uniforme e AS native; parametri sensibili restano FP32 se half non passa.

Al kickoff verificare i percorsi, le firme dell’SDK fissato, le release delle librerie e le evidenze della fase precedente. Aggiornare solo questo piano e i consumer attivi se un contratto cambia: il dettaglio anticipato è revisionabile, non una promessa sui risultati degli spike.

## Fonti e consegna

Fonti di partenza: R102, R91; R23–R43 da verificare. Vedi [bibliografia](../RESEARCH_REFERENCES.md) e [dossier H1–H12](../research/2026-10-01-apple-soc.md).

Consegna: riepilogo per ID, scelte dopo gli spike, comandi e risultati, regressioni risolte, gap/residui e prossimo pacchetto. Log prestazioni in [perf-log](../perf-log.md), esperimenti in [opt-log](../opt-log.md). Nessun risultato dedotto da un titolo di paper o da una feature annunciata.
