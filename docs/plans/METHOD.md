# Metodo dei piani dettagliati F e OPT

Data: 2026-10-02. La [roadmap](../ROADMAP.md) è l'autorità per scope e stato;
la [politica degli orizzonti](SEQUENCING.md) decide quali piani guidano
l'esecuzione e quali rimangono specifiche anticipate non ancora attive. La ricerca è nel
[dossier SoC](../research/2026-10-01-apple-soc.md). Un piano non è evidenza di
implementazione. Tutte le firme e disponibilità di API citate vanno verificate
contro l'SDK effettivo prima del codice; i nomi di nuovi componenti sono proposte.

## Contratto comune

Tutti i 58 piani specificano prerequisiti, dati/invarianti, file, ordine dei
pacchetti, attività e verifiche per ogni task. I dettagli lontani sono
proposte revisionabili: il kickoff riconcilia codice, SDK e misure senza
riscrivere il piano da zero. Le scelte empiriche hanno uno spike, una regola
di decisione e un fallback; non hanno risultati numerici inventati. Il primo
percorso deve produrre un'immagine o risultato corretto con gli strumenti
esistenti. Una variante complessa entra solo dopo la selezione motivata di
un candidato secondo SEQUENCING.md.

Editor, ECS, contenuti, 2D, UI ed ecosistema definiti in
[PRODUCT_PLATFORM](PRODUCT_PLATFORM.md) sono requisiti funzionali: la loro
implementazione e parità non dipendono da un guadagno prestazionale. Riuso e
prototipi di integrazione riducono il lavoro duplicato; le ottimizzazioni
aggiuntive restano soggette alla prova del collo di bottiglia. Registrare
release Bevy, feature, plugin e fixture prima di dichiarare compatibilità.

Per fasi già completate, usare il piano come audit/regressione e chiudere
solo i residui aperti. Non riscrivere un sottosistema per soddisfare una nuova
formulazione del piano. **F5 è integrata in main** (merge `7a7ac54`); il suo
piano serve per audit e regressioni. La revisione dei piani del 2026-10-02
non esegue nuovi benchmark né modifica il renderer. Lavoro documentale e
risultati sperimentali restano distinti.

Le dipendenze riportate sono funzionali, non un calendario: «integrazione
successiva» permette una baseline iniziale senza aspettare una fase lontana.
La numerazione F/OPT non è una sequenza totale (es. cooker minimo prima di F18,
riferimento offline per F12 prima del path tracer F32, osservabilità presto).

## Formato degli esperimenti

Prima della misura fissare in un manifest: commit, eventuale patch locale,
asset hash, seed, chip, memoria, OS/build, SDK/compiler, schema del piano,
preset, risoluzione interna/output, frame in volo, warmup, campioni e opzioni.
Registrare alimentazione, stato termico, carichi concorrenti e metodo di timing.
Per i confronti prestazionali usare macchina quieta: esecuzioni A/B/A o ordine
ruotato, almeno tre repliche e durata adeguata al segnale. Separare cold start,
steady state e soak; estendere il campione solo se rumore o failure lo richiedono.

Produrre frame time p50/p95/p99 (campioni sufficienti per le code), tempi CPU/GPU
e per unità temporizzata, picco memoria, byte letti/scritti, energia per frame
quando misurabile e latenza. Frame interpolati e frame realmente renderizzati
vanno contati separatamente. Segnare esplicitamente ciò che è stimato, non
disponibile o misurato con perturbazione. I contatori di Xcode possono richiedere
una cattura GUI; non sostituirli con un numero inventato.

Ogni ottimizzazione ha un **controllo negativo**: disabilitare il culling deve
cambiare il carico, omettere un update deve far fallire il readback, introdurre
una storia errata deve essere visto dalla suite. Per sincronizzazione e
concorrenza servono inoltre invarianti e test del modello; l'assenza di artefatti
GPU da sola non dimostra la correttezza.

## Criteri di adozione

- Correttezza: confronto esatto dove semanticamente equivalente; tolleranze
  numeriche dichiarate prima della prova per FP, RT e trasformazioni approssimate.
- Qualità: immagini di riferimento più clip temporali con regioni difficili;
  FLIP/PSNR non sostituiscono ghosting, flicker e tempo di recupero.
- Prestazioni: applicare il criterio della roadmap sul tier dichiarato;
  nuova infrastruttura richiede un beneficio end-to-end (riferimento ≥3% sul
  frame) o la soluzione di un limite concreto di memoria/energia/latenza,
  oltre il rumore misurato. Il −10% su un pass isolato non basta a giustificare
  un sistema generale. Modifiche locali restano proporzionate al problema.
- Nessuna regressione materiale di p99, latenza, memoria o energia nel preset
  considerato; un compromesso di qualità richiede un preset distinto e misurato.
- Accettazione di sviluppo sul M5 Max disponibile e fallback pertinenti;
  certificazione fisica T0 separata e pendente senza bloccare le fasi.
  Applicare [HARDWARE_VALIDATION](HARDWARE_VALIDATION.md), senza
  generalizzazione dall’override Apple9 a prestazioni/supporto fisico M3.
- Costo di tuning, compilazione, packing e dispatch incluso. Vietato attribuire
  al motore un guadagno osservato soltanto in un microbenchmark isolato.

## Consegna di una fase

1. Rileggere i task funzionali, i candidati selezionati, i criteri di uscita
   e i residui/TODO collegati; indicare esplicitamente l'ambito consegnato.
2. Eseguire `ctest` per logica portabile; per modifiche host Metal anche
   `metal_syntax_check` sul percorso supportato; su Mac compilare MSL e app.
3. Eseguire API/shader validation, readback, visual check e benchmark pertinenti
   alla modifica; non farli concorrere con altri esperimenti GPU.
4. Registrare manifest/dati grezzi, comandi esatti, immagini, decisione e limiti
   nei log. Aggiornare roadmap soltanto per task realmente verificati;
   candidati rimandati/scartati restano non spuntati. Non omettere verifiche
   di una funzione adottata classificandole come ricerca.
5. Lasciare flag/fallback e spiegare il ripristino della baseline. Non nascondere
   failure intermittenti nella media né dichiarare finita una fase per il solo build.

## Laboratori e distribuzione

User-space pubblico è il percorso ordinario. Studio XNU, API private ANE e
kernel sperimentale hanno manifest e binari separati, con requisito di accesso,
revisione e riproducibilità. Prima di sviluppare un componente privilegiato,
il dossier deve mostrare una limitazione concreta e una via tecnica praticabile.
La roadmap ammette la ricerca radicale e anche l'esito «non conveniente/non
distribuibile», senza rendere il laboratorio una dipendenza del renderer.

## Manutenzione

Lo stato di completamento vive soltanto nella roadmap. Tenere allineati i
piani operativi; aggiornare una specifica anticipata quando viene promossa
o cambia un suo contratto, senza rincorrere ogni refactor nei piani lontani. I riferimenti
R87–R111 sono stati consultati in questa
ricerca; R1–R86 vanno verificati quando usati, prima di importare codice o
assumere che una feature sia disponibile sull'SDK scelto.


## Matrice di verifica

I comandi seguenti sono strumenti **esistenti**; usare la configurazione di
build coerente con i riferimenti. Non rieseguire l'intera caratterizzazione
SoC per una modifica documentale o per un pass non coinvolto.

| Modifica | Prova minima pertinente | Evidenza e criterio |
|---|---|---|
| Logica/layout portabile | `cmake --build build`, `ctest --test-dir build --output-on-failure`; Linux in CI | Test indipendenti del contratto, edge case e fixture; ABI static assertions per layout GPU |
| Host Metal/MSL | Build app+shader macOS; `metal_syntax_check` nel job Linux | Compilazione non sostituisce esecuzione sul M5 |
| Raster/compute/sync | `tools/visual_check.sh build/release <reference-dir>`, readback specifico e API/shader validation | Riferimenti presi dalla baseline prima del cambio; esatto se semanticamente equivalente, tolleranze motivate prima del test |
| Pipeline/varianti | `tools/harvest_pipelines.sh`, `tools/archive_check.sh`, `tools/variant_check.sh`, `tools/hitch_check.sh` secondo il cambiamento | Nuovi descrittori nell'archivio, miss/fallback e cold start; verificare gli argomenti negli script |
| Scene GPU | Bench 8 e `--debug-gpu-scene N`, corruption control F5 | Conteggi/delta/transform/indiretti corretti; nessun errore intermittente ignorato |
| History/approssimazioni | Clip F8.7/F13.6, prima da implementare quando la fase parte | Ghosting/flicker/disocclusion/recovery oltre alle golden statiche |
| Ottimizzazione | `tools/bench_all.sh --stats --passes`, report per scena e A/B/A | Almeno tre repliche; stessi asset/preset/risoluzione/build mode, dati raw e rumore |
| CPU/ownership | Test DAG/lifecycle, sanitizers pertinenti, trace e allocation accounting | Nessuna race, deadlock, starvation o riuso prima dell'ultimo consumer |
| Asset/runtime/editor | Fixture di formato, roundtrip, lifecycle e progetto pulito | Il workflow finisce nella build standalone, non nel solo editor |
| Port ad altro hardware | Build target più runtime fisico | Se il device manca: verifica esterna pendente, nessun claim certificato |
| Soli piani/documenti | ID, link, dipendenze acicliche, caselle storiche e `git diff --check` | Nessuna necessità di benchmark GPU; CI controlla comunque il commit pubblicato |

Le soglie di confronto esatto non si applicano ciecamente a percorsi FP diversi:
congelare tolleranze, ROI e motivo prima del confronto, e conservare confronto
con riferimento indipendente. Mai decidere la soglia dopo aver visto una
regressione per farla passare. Un test di culling deve rilevare una superficie
persa, anche se occupa pochi pixel.

## Kickoff e decisioni senza falsa certezza

1. Fissare SHA di partenza e rileggere consegna precedente, task e piano.
2. Confermare i file esistenti/proposti e le API nell'SDK fissato; per librerie
   nuove verificare fonte primaria, versione, feature e condizioni di riuso.
3. Scrivere il contratto minimo e il riferimento; eseguire soltanto gli spike
   indicati che cambiano una decisione. I test GPU restano seriali.
4. Registrare nel piano/log variante scelta, dati, fallback e eventuali
   scostamenti. Proseguire autonomamente nel perimetro autorizzato: il piano
   non introduce un'approvazione in plan mode per ogni misura.
5. Integrare i pacchetti, eseguire verifica pertinente e residue audit; riferire
   `DEVELOPMENT_ACCEPTED` separatamente dai dispositivi non certificati.

Un piano anticipato è completo come istruzione di lavoro quando definisce
anche **come decidere** una scelta empirica. Non può contenere in anticipo
il vincitore di uno spike mai eseguito. Se l'API/ipotesi è impraticabile,
registrare la prova, mantenere la baseline ed aggiornare il pacchetto coinvolto.
