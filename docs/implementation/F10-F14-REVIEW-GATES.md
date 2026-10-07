# F10–F14 — registro dell'aggregatore

Base accettata: F9, PR #18, `main` `d5ec450`. Il proprietario richiede stop
dopo F14 integrata. La consegna dello scrittore `4b80e36` era esplicitamente
NON VERIFICATA; nessuna fase F10+ è accettata da questo registro.

L'integrazione usa `codex/lighting-integration`; la chat Sviluppo Phosphor
scrive F13/F14 e i fix fisici in commit separati. Un solo processo GPU di
prova alla volta. File grezzi locali: `build/lighting-smoke`, `build/f10-*`,
`build/f11-*`, `build/f12-*` nel worktree dell'aggregatore.

| ID | Riscontro | Stato/evidenza |
|---|---|---|
| I-01 | Errori C++/MSL della consegna: keyword vertex, parentesi, address space, include e costanti | Corretti in `2afbf20`; build e 561 test CPU iniziali passati |
| I-02 | Receiver alpha con mip bias diverso dal resolve finale | Codice corretto in `2afbf20`; fixture DRS+overflow ad alta frequenza ancora da eseguire |
| I-03 | Fragment lit con texture top-level incompatibile ICB | Corretto in `bad600b` con handle buffer10; primo smoke CSM API/shader validation PASS |
| I-04 | Shadow dispatch globale meshlet×slot, lavoro cartesiano | Corretto in `3afcffd`, per bucket.used inclusi holes; 4 test CPU/1011 assertion, 5 casi GPU CSM/cache/negativi passati |
| I-05 | Guide half non rinormalizzate nel reuse GI, target non coerente | `8e533bf`; 2192 errori reservoir al primo frame prima del fix, zero dopo; DDGI/cache/restir e negativi GPU passati nei microcase |
| R-11-01 | Area emissiva cambia con scala/shear senza invalidare il dominio DI | Fix scrittore `dfe8cdb`+`d147149` integrati; CPU verdi, verifica GPU/moto energetica pendente |
| R-11-02 | Nonfinite azzerati prima del checker | Fix dello scrittore pendente; serve errore registrato prima del sanitizing |
| R-11-03 | Reservoir nero valido perde M nel reuse DI | Fix pendente; Bernoulli statico deve dare media0,5, non0,625; verificare anche filtri precedenti al merge |
| R-11-04 | STBN ripete i medesimi campioni ogni16frame | Fix pendente; per convergenza usare seed indipendenti, non contare cicli ripetuti come nuovi campioni |
| R-12-01 | Sole/direzionali non invalidano la GI | `3f9ffd9`+`ab64225` integrati; CPU verdi; verifica sole dinamico e spegnimento pendente |
| R-12-02 | Facing world confuso con front materiale sulle specchiate | `5eb70f0` integrato; test CPU e microcase GPU passati; fixture specchiata dedicata pendente |
| R-12-03 | Reservoir nero valido perde M nel reuse GI | Fix pendente; stesso controesempio Bernoulli indipendente |
| R-12-04 | Reset luce azzera relocation e ne impedisce il progresso | Fix pendente; separare radiometria e stato spaziale, test embedded probe con luce mobile |
| R-12-05 | Mitsuba reference lascia emettere i buchi MASK | Fix/model gate pendente; mai accettare un oracle con emissione differente |
| R-12-06 | Reference altera visible/castsShadows e omette triangoli luminosi standalone | Fix/model gate pendente; verifica ruoli dei raggi e deduplicazione solo delle mesh emissive |
| R-12-07 | UV emissive estratte da stream CPU non aggiornate con vertici GPU | Contratto API da correggere/delimitare; fuori dalla CLI di deformazione attuale |

## Prove indipendenti ancora necessarie

I checker GPU dimostrano consistenza e funzionamento dei controlli negativi;
non dimostrano da soli energia corretta, convergenza, qualità temporale,
penombra fisica o assenza di leak di luce. I confronti indipendenti restano
gate obbligatori. Il runner marca il bias negativo `QUALITY_CONTROL_PENDING`
finché l'immagine non viene confrontata con un controllo e soglie congelate.

Mitsuba **3.9.1 scalar_rgb**, Dr.Jit1.5.0 e NumPy2.5.3 sono installati soltanto
in `build/reference-venv` del tester. Uno smoke CPU 4×4 con ambiente costante
ha restituito esattamente RGB(0,25;0,5;1); non è una verifica del modello
Phosphor. Il warning di inizializzazione LLVM non riguarda il percorso scalar
verificato; nessun rendering CUDA/Metal è stato avviato da questo riferimento.

Prima dell'accettazione: fixture/soglie reference congelate, immagini e clip,
Apple9 effettivo, camere/view/resize/DRS, qualità energia/statistica, lifecycle,
archivio/hot reload, zero allocazioni stabili, misure×3 e regressioni F9/F7/F8.
F13/F14 rimangono codice in scrittura finché non arrivano e superano gli stessi
confini di integrazione. M3 fisico resta validazione esterna pendente.
