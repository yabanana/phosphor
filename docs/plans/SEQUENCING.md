# Priorità e maturità dei piani

Decisione del proprietario, 2026-10-01: **prima implementazione corretta e
semplice, poi collo di bottiglia, esperimento e decisione**. Questo documento
regola l'attivazione del lavoro descritto nella roadmap e nei 58 piani.

**Estensione di prodotto:** editor, ECS/runtime, contenuti community, 2D,
UI di gioco ed ecosistema sono requisiti funzionali secondo
[PRODUCT_PLATFORM](PRODUCT_PLATFORM.md). Riusare Bevy/librerie esistenti è
la prima opzione; i piani sono dettagliati in anticipo e l’implementazione resta progressiva. La soglia di beneficio
prestazionale si applica alle ottimizzazioni aggiuntive, non all'esistenza di
queste capacità. Il prototipo di integrazione ECS verifica compatibilità e
costo prima della migrazione, senza richiedere un guadagno del 3% sul frame.

## Percorso corrente

| Orizzonte | Lavoro | Cosa deve produrre |
|---|---|---|
| Checkpoint corrente | F5/F6 integrate; F7/F8 DEVELOPMENT_ACCEPTED M5; F8.4 risolta in-process | Sosta prima delle OPT richiesta dal proprietario |
| Prossimo | F9–F13; F14 da scegliere in base alla scena | Illuminazione integrata e qualità temporale; dettagli di progetto rivisti dopo F8 |
| Successivo | Altre F, compresi runtime e mondo | Piani dettagliati da riconciliare quando il consumatore esiste |
| Prodotto richiesto | F21/F27/F34, F39–F41 e audio base F26.1 | Riuso, 2D/UI/editor e compatibilità verificata; baseline incrementale, copertura finale obbligatoria |
| Opportunità | OPT aperte, task `[CANDIDATO]`, fasi `[EDGE]` | Nessuna implementazione automatica: attivazione per un problema concreto e misurato |
| Audit | F0–F8, OPT-0/OPT-1 | Conservare evidenze; riprendere soltanto residui e verifiche pertinenti alle modifiche |

F5/F6 sono integrate; F7/F8 hanno accettazione di sviluppo M5. F8.4, riaperta
per il costo dei worker isolati (PR #15), è risolta in-process: il motore
rilascia l'autoriferimento interno dello scaler MetalFX
([causa e verifiche](../research/2026-10-04-metalfx-cycle-root-cause.md),
[costo storico dell'isolamento](../F8_METALFX_LIFETIME.md)). Gli esperimenti F7.4/F7.5
richiesti dal proprietario sono stati eseguiti e non adottati nel preset corrente:
[misure e decisioni](../F7_F8_HANDOFF.md). Non bloccano la baseline F7.
Nessuna OPT o fase F9+ viene avviata da questo checkpoint. La storia
F8.6 parte da risorse per vista e sincronizzazione conservativa; aliasing
temporale generale e scheduling eterogeneo non servono a renderla corretta.

## Due momenti di decisione

**Dopo F8 — primo frame reale.** Usare il runner e i report già disponibili:
almeno una scena reale con materiali eterogenei, una scena densa/dinamica e
clip con disocclusioni, camera cut e resize. Salvare configurazione, frame
time, costi dei pass, memoria e qualità; usare il M5 disponibile secondo
[HARDWARE_VALIDATION](HARDWARE_VALIDATION.md), con T0 esterno pendente.
La baseline del corpus è OPT-4.16, anticipabile senza completare OPT-2/3/4.
Scegliere al massimo un collo di bottiglia prioritario: correzione locale,
esperimento mirato oppure proseguimento verso la luce se il budget è adeguato.
Esito del 2026-10-04 ([revisione](../research/2026-10-04-post-f8-review.md)):
baseline M5 registrata; il tempo GPU sta in MetalFX e nel ciclo luci, quindi
si prosegue verso la luce (F9–F13) e OPT-2 si rivaluta dopo F13.

**Dopo F13 — frame con luce reale.** Ripetere sul frame integrato con ombre,
GI, riflessi e denoising; registrare contesa e aggiornamenti temporali.
Questo rende valutabili le proposte di architettura generale. F14 può
arricchire il corpus quando rilevante. Aver raggiunto questa tappa non avvia
automaticamente scheduler, e-graph, SME o ANE.

Correttezza, crash, regressioni e lavoro necessario a un consumatore corrente
si risolvono quando emergono. Non occorre aspettare queste tappe per correggere
un bug o fare una modifica locale chiaramente motivata.

**Linea di piattaforma dopo F8:** verificare F27.1/F27.7 (host Bevy e bridge),
poi nucleo F41.1/F41.2 e importer F21, quindi runtime F39/F40 e authoring F34.
Luce F9–F13 e piattaforma possono avanzare sui contratti condivisi; conservare i
contratti della GPU scene F5. F41 completa i test dei plugin grafici dopo UI/2D:
non richiedere la certificazione dell'intero ecosistema prima del primo sprite.

## Come una complessità si guadagna il posto

Prima di sviluppare **nuova infrastruttura di ottimizzazione**, registrare
una breve nota nel `docs/opt-log.md` esistente:

1. Scena/manifest e costo sul frame: qual è il limite osservato e quanto può
   migliorare il risultato finale? Un microbenchmark da solo non basta.
2. Soluzione semplice confrontata e ragione per cui non basta; task candidato,
   ipotesi, limite di tempo e criterio di abbandono.
3. Risultato end-to-end a qualità comparabile, costo di runtime/memoria e
   manutenzione; decisione **adotta**, **rimanda** o **scarta**.

Per un nuovo sistema, richiedere un guadagno ripetibile sul frame (riferimento
iniziale ≥3% sul tier target), oppure la soluzione di un limite concreto di
memoria, energia o latenza entro budget concordati nel preset. Un −10% su un
kernel isolato non giustifica un compilatore o scheduler generale. Le modifiche
locali possono essere adottate sul pass prioritario se il vantaggio si vede
nel carico reale e il costo di manutenzione resta proporzionato.

Un solo esperimento architetturale attivo alla volta; niente benchmark GPU
concorrenti. Lo spike resta limitato a 1–2 settimane al massimo, con stop
anticipato se la fattibilità o il beneficio non emergono. «Rimanda/scarta»
è un esito valido: non richiede una riscrittura alternativa per forza.

## Quando riaprire le idee avanzate

| Candidato | Evidenza necessaria prima dello spike |
|---|---|
| E-graph / compilatore generale | Frame reale con più trasformazioni utili, solver semplice insufficiente e potenziale end-to-end quantificato |
| Scheduler proprio / predittivo | Trace di job e dipendenze che mostrano ritardi rilevanti dopo aver corretto granularità, pool e attese |
| Lifetimes più precise / compilatore di layout | Picco memoria, copie o sincronizzazioni realmente limitanti; intervento circoscritto provato prima |
| SME custom | Batch reali caldi e limite misurato di NEON/Accelerate, includendo packing |
| ANE / modelli predittivi | Lavoro concreto compatibile, baseline analitica e vantaggio completo includendo contesa e risultati tardivi |
| Kernel / ANE privato | Limite riproducibile del percorso pubblico, beneficio residuo significativo e fattibilità di piattaforma documentata |

Questi sono trigger di ricerca, non nuove feature obbligatorie né prerequisiti
delle normali fasi F.

## Gestione dei 58 piani

**Aggiornamento richiesto dal proprietario il 2026-10-02:** tutti i piani F e
OPT contengono dettaglio anticipato: file, contratti, pacchetti, spike e verifica
per ogni task. Questo sostituisce la scelta precedente di mantenere soltanto
brevi bozze lontane. Rimane distinta l'attivazione: F6–F8 sono operative,
F9–F14 sono successive, le altre specifiche si riconciliano al kickoff.
Gli OPT restano cataloghi di interventi selezionabili.
Per i requisiti di piattaforma, «pianificato» riguarda tempi e soluzione,
non la facoltà di omettere editor/2D/UI dal prodotto completo. I primi
incrementi possono avere copertura parziale, da dichiarare esplicitamente.
I dettagli lontani sono già descritti, ma restano revisionabili. Nomi dei file
nuovi e scelte di libreria sono proposte; il vincitore di uno spike si fissa
soltanto dopo la misura. Aggiornare il piano al kickoff o al cambio di contratto,
non a ogni refactor locale delle altre fasi.

L’attivazione di un piano anticipato richiede di rileggere codice e misure attuali,
definire scope/baseline, verificare dipendenze e scegliere i task candidati
eventualmente necessari. È una decisione tecnica registrata, non un nuovo
passaggio di autorizzazione. Cambiare una fase non impone di riscrivere tutte
i piani lontani: aggiornare soltanto i consumatori già attivi.

Gli ID e le caselle restano nella roadmap. I candidati non selezionati
rimangono `[ ]` e non bloccano la chiusura dell'**ambito operativo**; riportare
sempre quali task sono stati consegnati e quali restano nel catalogo. Non
spuntare una tecnica perché è stata scartata, né dichiararla implementata.
I residui storici T0/contatori mantengono il loro stato e la loro evidenza.
Il M5 Max è il gate di sviluppo disponibile; T0 fisico e altri dispositivi
sono certificazioni separate non bloccanti per proseguire, secondo
[HARDWARE_VALIDATION](HARDWARE_VALIDATION.md).

Vulkan resta una nota di possibilità remota, senza task attivo o modifica
del renderer per anticiparlo. Gli ID F39–F41 non indicano che queste capacità
debbano essere sviluppate dopo la distribuzione F38.
