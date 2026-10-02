# Audit della pianificazione — 2026-10-02

Base del renderer: `main`, merge F5 `7a7ac54`. Ambito di questa consegna:
documentazione di roadmap, ricerca e piani; nessuna modifica al codice del
renderer e nessun nuovo risultato GPU/SoC dichiarato.

## Controlli eseguiti

- 58 piani: F0–F41 e OPT-0–OPT-15.
- 414 ID della roadmap presenti una volta ciascuno nelle sezioni operative
  dei rispettivi piani, con implementazione, verifica ed evidenza richiesta.
- 52 task già spuntati nel checkout iniziale preservati nel testo e nello
  stato; nessuna casella implementativa modificata da questa pianificazione.
- 184 task candidati coerenti fra roadmap e `dependencies.json`; nessun
  candidato selezionato automaticamente.
- Grafi delle dipendenze di fase e di task, incluse quelle condizionali,
  verificati aciclici; riferimenti a ID e fasi risolti.
- Collegamenti Markdown locali e ancore controllati nei documenti interessati.
- Politiche, indice, roadmap e CLAUDE.md allineati a F5 integrata, piani
  dettagliati anticipati e accettazione di sviluppo sul M5 disponibile.
- Whitespace e diff controllati con `git diff --check` e ispezione dei file
  nuovi, prima della pubblicazione.

## Verifiche di contenuto

Il piano F6 è stato riconciliato con `meshlet_builder.*`, `gpu_types.h`,
`PipelineDesc`, capacità del `MetalContext`, parser delle opzioni e regole
F5 su ICB/barriere/split encoding. Gli header metal-cpp fissati espongono il
descrittore mesh Metal 4; non è stata compilata una nuova pipeline F6.

La CLI Apple9 dell'app è dichiarata **da implementare**, mentre quella di
`soc_bench` è già presente. Le due non vengono confuse. Hardware fisico e
capacità effettive hanno campi distinti nel contratto del futuro report.

I gate T0 storici restano pendenti. La decisione nuova permette la chiusura
dell'ambito di sviluppo sul M5 con le prove pertinenti; non converte
fallback, budget di memoria simulati o build hosted in certificazioni M3,
Base, mobile o XR. I residui tecnici sul dispositivo disponibile rimangono tali.

## Limiti dell'audit

Questo controllo dimostra copertura e coerenza dei piani, non correttezza
di implementazioni future, vincitori degli spike o prestazioni. Le API e
release dei sottosistemi lontani sono da riconfermare al kickoff. Le fasi
consegnate usano i piani come audit; quelle future come istruzioni di lavoro
revisionabili. La CI del commit pubblicato rimane una verifica distinta.
