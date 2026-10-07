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
| R-11-02 | Nonfinite azzerati prima del checker | Fix integrato; `build/f11-numeric-overflow`: 16/16 controlli negativi rilevati prima del sanitizing, exit1 previsto |
| R-11-03 | Reservoir nero valido perde M nel reuse DI | Fix counted-zero e guard numerici integrati; suite CPU verde, energia fresh/spatial passa; ulteriore expiry bias corretto in `970d7ce`, rerun ensemble pendente |
| R-11-04 | STBN ripete i medesimi campioni ogni16frame | Rotazione per blocco integrata. Raw S1 FAIL correlazione conservato; consumed S2 80/80 gate PASS e 2097152 valori CPU bit-exact. Vedi report STBN, nessuna prova di qualità immagine dedotta |
| R-12-01 | Sole/direzionali non invalidano la GI | `3f9ffd9`+`ab64225` integrati; CPU verdi; verifica sole dinamico e spegnimento pendente |
| R-12-02 | Facing world confuso con front materiale sulle specchiate | `5eb70f0` integrato; test CPU e microcase GPU passati; fixture specchiata dedicata pendente |
| R-12-03 | Reservoir nero valido perde M nel reuse GI | Fix integrato e CPU verde; expiry della catena corretto insieme a DI in `970d7ce` |
| R-12-04 | Reset luce azzera relocation e ne impedisce il progresso | Epoch radiometrica/spaziale separate nel codice integrato; test embedded probe con luce mobile ancora pendente |
| R-12-05 | Mitsuba reference lascia emettere i buchi MASK | Adapter MASK scritto e test contrattuali verdi; audit runtime Mitsuba in corso, sorgente non ancora accettata come oracle |
| R-12-06 | Reference altera visible/castsShadows e omette triangoli luminosi standalone | Export e strict model gate integrati, test CPU verdi; verifica indipendente geometria/materiali/raggi in corso |
| R-12-07 | UV emissive estratte da stream CPU non aggiornate con vertici GPU | Lettura corrente di vertici/UV GPU integrata; verifica API deformazione e texture ancora pendente |

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

## Integrazione successiva (evidenze locali, non accettazione)

- `4b3303d`: merge della consegna F13/F14 `2918f78` con correzioni F10–F12 preservate; `886ed00` abilita il gateway SDK tipizzato e corregge compilazione C++/MSL. Build Release e CTest passano.
- `970d7ce`: scadenza della history indipendente dall'endpoint selezionato, DI e GI. Enumerazione razionale: vecchia regola 2.0014878388 contro integrale2; nuova2 esatto. L'ensemble precedente è conservato come FAIL (22 confronti); non si sono cambiate soglie.
- Energia F11 fresh point/area/emissive PASS su32seed; point1 esatto e negativo PDF×2 energia0.5 rilevato dal gate indipendente. Tutti i processi e checker dell'ensemble precedente passavano: non erano prova dell'assenza di bias.
- F10 oracle v1: FAIL penombre CSM/RT, bias negativo e flicker PASS. `9d49e21` elimina clip della media Bernoulli; `838f446` usa bounds caster completi senza aggiungere reach500m spurio. Rerun medesimo protocollo richiesto.
- F12 Cornell128×72:512frame per DDGI/cache/restir, tutti checker PASS, snapshot esatto e catture grezze. Oracle CPU indipendente in costruzione/verifica; nessun giudizio di qualità GI ancora.
- F13 smoke: rawRT/SSR/AO raggiungono i checker; custom fallisce il contatore di composizione. `463f2f4` corregge fixture texture slots e lettura AO scalare. `11ee326` rende diagnostiche le cause dei checker; non le nasconde.

La consegna F13/F14 elenca ancora readback numerici e controlli negativi mancanti. Lo scrittore li completa in commit incrementali mentre l'aggregatore misura. Il branch d'integrazione non è `main`; nessuna fase F10–F14 è dichiarata chiusa.
