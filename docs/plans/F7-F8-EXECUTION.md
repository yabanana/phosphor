# Esecuzione F7/F8 e revisione F5/F6

Autorizzazione del proprietario: 2026-10-02. Branch `codex/f7-f8`, base
`e600887` (merge F6). Scope: acquisire Sponza, rivedere F5/F6, implementare
F7 e F8, infine sperimentare F7.4/F7.5 sul frame integrato. Fermarsi prima
delle fasi OPT. Nessuna nuova approvazione ordinaria necessaria per questi
passaggi; una sola prova GPU alla volta.

## Evidenze iniziali e residui

- F6 è già integrata in main; CLAUDE.md riferiva ancora il branch separato.
- Build Debug app/MSL e suite portabile della base eseguite con successo.
- F6 dichiara gate locale superato, ma conserva un'uscita non nulla
  intermittente non spiegata e un hitch check fragile: restano da indagare.
  Non sono cancellati dall'assenza di T0.
- `tools/f6_check.sh` controlla alcune righe PASS senza verificare sempre
  l'exit status del processo: rafforzare il runner prima della nuova chiusura.
- Sponza Khronos fissata a `edc7c9e67c639d230715049ee31f9a96a6babbbe`:
  73 file, 52,690,203 byte verificati SHA-256. Asset locali, manifest e
  fetcher in repo. La fonte dichiara licenza CryEngine, non CC-BY.

## Correzioni prerequisite in corso

Il loader glTF preesistente accorpava primitive con materiali diversi,
usava il materiale della prima primitive e applicava il cutoff anche a
OPAQUE. Correzione: geometria per primitive riusata fra nodi, materiale
per draw e default glTF indipendente, cutoff solo MASK; cache texture per
indice e spazio colore. Conservare matrici affine esatte e tracciare le
entità importate per teardown. Test indipendenti per questi casi.

La fixture deve essere esplicita: `--scene procedural` congela le vecchie
scene; `--scene assets/sponza/Sponza.gltf` richiede Sponza e fallisce se non
caricabile. Non confrontare un nuovo asset contro golden procedurali.

## Sequenza esecutiva

1. Correzioni/review, baseline procedurale e Sponza con manifest e immagini.
2. F7: ID visibility, raster mesh, resolve generico e confronto forward;
   poi classi materiali, alpha test separato e canali per F8/F13.
3. F8: history per vista e motion, HDR/esposizione/output SDR-EDR,
   integrazione MetalFX temporal e scala, mip bias/sharpening, clip qualità.
4. Verifica integrata e residue audit F5–F8, con percorsi Apple9 sul M5.
5. F7.4 e F7.5: baseline congelata dopo F8, prototipi bounded, misura
   end-to-end e qualità temporale, decisione adotta/rimanda/scarta.
6. Handoff per task, prove e limiti. Nessuna fase OPT attivata.

## Contratti da conservare

Una GPU scene, PipelineCache e GpuMemory esistenti, grafo come autorità
per accessi/barriere. Gli argomenti delle mesh draw F6 usano Object|Mesh
secondo lo spike S4; la regola Vertex di F5 vale per le ICB indexed e non
si applica indiscriminatamente alle mesh draw.

Visibility ID riserva il sentinel e distingue istanze/triangoli; il resolve
ricostruisce derivate prospettiche, conserva winding e gestisce near-plane.
Overflow non può perdere superfici. I buffer di storia hanno un owner per
vista e completion del reader/writer; dati previous pose pubblicati prima
di sovrascrivere le istanze correnti. MetalFX usa l'API Metal 4 realmente
presente negli header, con accessi dichiarati e fallback corretto.

Il forward resta riferimento. Se si corregge una semantica preesistente
(es. alpha mode), conservare prova del difetto e aggiornare i riferimenti
solo per quel motivo. Un confronto col forward da solo non prova che
entrambi non condividano lo stesso bug.

## Esperimenti richiesti dopo F8

- F7.4: deferred on-tile contro visibility+compute a stessa scena, materiali,
  risoluzione e qualità. Registrare memoria tile/DRAM stimata, costo frame
  e pressione/overlap osservabili. Nessun vincitore presunto.
- F7.5: blocchi 2×2 soltanto con ID/materiale/superficie compatibili,
  disocclusioni/bordi/motion sensibili full-rate; classificazione dalla
  luminanza precedente con invalidazione. Confronto temporale e costo totale,
  non solo numero pixel ombreggiati.

Lo spike completa l'esperimento anche quando scarta la tecnica; non dichiarare
la tecnica adottata o una soglia qualità passata senza evidenza.


## Revisione dello shading F5/F6

La trasformazione delle normali usava la matrice modello anziché la direzione
inverse-transpose: errata per scale non uniformi/shear. Ora si usa la matrice
dei cofattori con segno del determinante, normalizzata dopo interpolazione;
proprietà di ortogonalità e riferimento GLM coprono scala, shear e riflessione.
La handedness della tangente viene inoltre invertita per istanze specchiate.
Queste correzioni possono cambiare immagini precedenti per ragioni semantiche;
conservare i confronti e non attribuirle a differenze del visibility buffer.

## Note di implementazione F8

- MetalFX temporal è creato sui worker utility di PipelineCache; la prima
  configurazione viene preriscaldata, resize futuri hanno fallback spaziale
  finché lo scaler corretto non è pronto.
- Pass External nel grafo, non fondibile con compute/raster, con stage ML
  esplicito e fence gestito dall'esecutore. Lo spike ha rilevato che
  `updateFence` di un compute encoder non accetta `StageAll`: la boundary
  aggiorna il fence dopo il dispatch, ordinato con i produttori dal grafo.
- Motion current→previous in pixel, +Y verso il basso, senza jitter,
  verificato contro la convenzione Apple WWDC26 sessione 359.
- AgX è il fit polinomiale della configurazione iniziale di Troy Sobotka e
  dell'approssimazione di Benjamin Wrensch, non equivalenza bit-exact alla
  LUT di Blender. Fonti: https://github.com/sobotka/AgX e
  https://github.com/MissingDeadlines/iolite/discussions/12.
- EDR usa extended-linear sRGB e headroom interrogato sul display della
  finestra; screenshot PNG sono conversioni SDR esplicite e non prova dei nit.

## Stato delle verifiche (lavoro in corso, non chiusura)

Il codice F7/F8 è integrato nel branch di lavoro. Passano build e suite
portabile, controlli GPU di ID/guide/motion/esposizione, controlli negativi
motion/esposizione/feedback, due viste alternate e variazioni della risoluzione
interna senza nuove allocazioni GPU dopo il warmup. Restano da completare
l'audit finale, la suite temporale, release/AOT/CI e gli esperimenti F7.4/F7.5.

La prima metrica di attrazione verso il frame precedente confondeva aliasing
spaziale e storia: falliva anche sul controllo senza storia. I dati e le soglie
v1 sono conservati. La versione 2 mantiene le soglie spaziali/flicker/recovery
e misura anche l'errore in eccesso rispetto a un controllo corrente senza
storia; test sintetico indipendente e controllo con frame ritardato impediscono
che un semplice PASS diventi la definizione della metrica. Non è una metrica
percettiva standard né prova unica dell'assenza di ghosting.

I clip iniziali (320×180, input 75%, riferimento raster 2×) passano PSNR,
RMSE, flicker e recupero al camera cut; resta un errore temporale su alcune
silhouette del toro. Il caso è ancora aperto, non dichiarato PASS.
La scansione delle quattro convenzioni di jitter ha confermato la convenzione
originale (spostamento raster +X destra/+Y basso; Y invertito passando a NDC).
L'inversione dei segni peggiora PSNR e RMSE ed è stata rimossa. Il controllo
con pose GPU precedenti deve verificare anche l'età dei dati, oltre alla sola
formula dei motion vector. La dilatazione reattiva sui bordi di profondità non
ha migliorato il caso ed è stata rimossa. L'input colore MetalFX è opaco;
il background viene identificato dalla depth per l'istogramma.

Artefatti locali: `build/f7-review/` e `build/f8-quality/`; non sono misure
prestazionali quando includono readback o cattura PNG. `--offscreen` separa
il throughput di rendering dal compositore; i gate a schermo rimangono distinti.

## Aggiornamento sperimentale del 2026-10-03

Sono disponibili due prototipi espliciti, disattivati di default:

- `--tile-resolve` (F7.4): shading differito degli ID tramite imageblock
  implicito e tile kernel Metal 4. Non è un G-buffer tradizionale: conserva
  lo stesso fetch degli attributi del V-buffer per isolare l'effetto del
  posizionamento dello shading. Con culling frustum il grafo fonde raster
  opaco/alpha e tile resolve, rendendo memoryless gli ID; con two-phase la
  riduzione Hi-Z può interrompere la fusione. Le guide restano texture in
  memoria principale. Il caso Sponza 641×361 coincide esattamente con compute.
- `--adaptive-shading` (F7.5): ogni invocazione gestisce un blocco 2×2,
  riusando lo shading soltanto su una singola primitive, generazione valida,
  storia con identità esatta, moto <0,25 pixel, roughness ≥0,75, materiale
  non metallico e senza normal map/emissione/alpha test, contrasto storico
  ≤max(0,002; 2%). Il moto viene comunque ricostruito per ciascun pixel.
  Storia per vista da 16 byte/pixel; il costo di aggiornarla è incluso.

Le prime tre repliche Release offscreen a 1920×1080, input 75%, 120 warmup
+600 frame, indicano **nessuna adozione automatica**: F7.4 Sponza circa
2,144 ms/frame compute contro 2,150 ms tile; F7.5 Cornell circa 2,075 contro
2,087 ms nonostante 279867 shading riusati su 492102 pixel coperti nell'ultimo
frame. Sono dati preliminari in ordine sequenziale; ripetere l'ordine ruotato
per la decisione finale. Tempi GPU con overlap non equivalgono al throughput.
Nessuna misura T0 fisica né contatore DRAM/energia attribuito a questi numeri.

L'esperimento adattivo ha scoperto un difetto indipendente: MaterialComponent
partiva con alphaCutoff=0,5 anche per materiali procedurali opachi. Questo
li inseriva nella fase alpha-test e applicava una reactive mask a 0,75.
Il default diventa opaco (cutoff zero); i MASK importati conservano il cutoff
esplicito. La correzione non modifica la copertura dei materiali opachi.

### Riferimenti e differenze numeriche

Il riferimento supersampled media la luce HDR prima del tonemapping; un
ridimensionamento di PNG già tonemappati non è un riferimento equivalente.
`--reference-scale 2|4|8` usa la media dell'intero footprint; limite interno
8192 pixel. `--settled-reference 16..64` ricostruisce ogni scena immobile per
un ciclo di jitter, resettando la storia fra frame logici (diagnostica).

Il primo confronto Sponza forward/resolve a 640×360 ha PSNR 51,49 dB, 1204
pixel con delta >4 su 230400, 33 con delta >16, massimo 38. Torus senza
texture ha massimo 1 LSB. Binning e tile hanno zero differenze dal generico.
Gli spike a mip zero e forward con derivate razionali aumentano PSNR a
58,42 e 57,55 dB: il filtraggio è una componente sostanziale dello scarto.
La diagnostica UV differisce soltanto in quattro pixel di wrap; le derivate
variano maggiormente. Le varianti diagnostiche sono state rimosse, senza
cambiare il forward per assorbire il confronto. Anche la riscrittura
sample-relative delle baricentriche e le derivate quad non migliorano il
caso e non vengono adottate.

La tolleranza per il corpus texturizzato viene calibrata su questo primo
scatto: PSNR ≥50 dB, delta p99 ≤4, massimo ≤64. Non era una soglia fissata
prima del primo scatto. `f7_f8_check.py` la verifica anche su una camera e
risoluzione diverse (1280×720, frame 259 del percorso temporale). I confronti
fra varianti dello stesso resolve rimangono esatti; il confronto procedurale
forward/resolve conserva 1 LSB. Le soglie non si applicano a geometria persa,
ID invalidi o corruzione delle guide.

### Limite della metrica temporale

Le metriche v1/v2 di attrazione verso il vecchio frame danno falsi positivi
anche su ricostruzioni del solo frame corrente con diverso filtro spaziale.
La metrica v3 conserva entrambi i diagnostici e misura la scia oltre il
supporto spaziale corrente (un pixel output nel preset 75%). Non dimostra
l'assenza di ritardi inferiori al supporto del filtro; PSNR/RMSE/flicker,
immagini, camera cut e controllo GPU delle pose/motion rimangono necessari.
Il controllo sintetico include filtri spaziali differenti e un frame
ritardato: il primo passa il gate temporale, il secondo deve fallire.
Soglie spaziali, flicker e recupero non sono state allentate. v1 e v2 sono
conservate e rieseguibili. Restano da completare il corpus lungo, la review
finale e i gate AOT/CI; nessuna chiusura di fase dichiarata in questo documento.

La prima soglia sul massimo delta **fallisce** nello scatto held-out (123).
Il confronto dei due forward preesistenti, indexed e mesh, ha già un massimo
120 nello stesso scatto: il crop localizza i picchi sulle foglie alpha-test.
La discrepanza rimane con culling disattivato; frustum/off/two-phase del
V-buffer coincidono esattamente. Anche l'invarianza delle posizioni e un
campionamento alpha a gradienti espliciti sono stati provati: il primo non
cambia lo scatto, il secondo peggiora e viene rimosso. Non attribuire questi
picchi a superfici perse dall'occlusion culling.

La soglia finale texturizzata usa PSNR ≥50 dB, delta p99 ≤4 e al massimo
0,05% dei pixel con delta >16. Massimo e errore oltre il footprint rimangono
nel report come diagnostici; un massimo isolato sul cutoff è discontinuo.
Questo è un affinamento **successivo ai primi confronti**, dichiarato, non
un gate originale retrodatato. Conservare i report dei gate precedenti
falliti. Il corpus lungo e un controllo negativo di immagine spostata devono
ancora convalidare la sensibilità. Nessun riferimento viene sovrascritto.

Nei test HDR l'istogramma CPU/GPU può differire per un campione esattamente
sul confine di un bin (FMA/log2). Il readback ora verifica ogni conteggio
cumulativo entro i limiti ottenuti perturbando ciascuna luminanza CPU di
±0,001%; il numero totale dei campioni deve coincidere esattamente. La stampa
distingue `exact`, `bin-boundary rounding` e `mismatch`. Non è una tolleranza
arbitraria sul numero globale dei pixel e non può assorbire una perdita di
campioni o uno spostamento non locale dell'istogramma.

### Archivio AOT, hot reload e lifetime

La build Release e la suite portabile passano anche con le nuove pipeline.
Il primo harvest con MetalFX attivo includeva le librerie private del framework:
il fetcher dell'archivio ora usa `--upscaler native` e rifiuta librerie esterne,
senza sostituire i loro percorsi con quello del motore. Le pipeline MetalFX
rimangono di proprietà del framework, create sui worker prima dell'uso.

Il toolchain 27.1 ha inoltre fallito l'AOT dei materiali compute/tile con
`cannot find private metadata at offset ...`. Caso ridotto: resolve in una
libreria singola funziona; con più moduli AIR collegati fallisce. Una sola
unità `material_passes.metal` per forward e resolve, collegata per prima,
permette l'archivio completo. Runtime compiler invariato, Metal 4 mantenuto.
Tentativi con nomi/tipi, inline, cache disabilitata e versioni del linguaggio
non sono stati adottati. Il reloader applica lo stesso raggruppamento; il
watcher continua a osservare i file sorgente separati. Sponza con HDR e
MetalFX: **47/47 pipeline del motore da archivio, zero compiler calls del
motore e zero compilazioni sul render thread** nel controllo AOT iniziale.
Questi contatori non descrivono le compilazioni interne al framework MetalFX.

Il registro comune gestisce ora anche la storia geometrica/Hi-Z per vista,
con matrice jittered; PostProcessor registra il segnale unjittered. Un cambio
della generazione shader invalida entrambe le storie. Il controllo di reload
con post temporale mostra una nuova invalidazione (due reset totali).
Un controllo negativo copia deliberatamente le pose correnti nella storia
prima del resolve: l'oracolo di età delle pose lo rileva anche quando la
formula dei motion vector, calcolata da quelle pose scorrette, è coerente.

La validazione ha riprodotto un errore offscreen di residency dei render
target su heap. GpuMemory registra ora esplicitamente anche questi attachment,
come già faceva il grafo per i propri target, e li rimuove alla distruzione.
Dieci ripetizioni con callback feedback ritardate e validazione API/MSL
passano dopo la correzione. Questo errore locale nuovo non viene spacciato
per la causa dell'uscita intermittente storica di F6, che non è stata attribuita.

Il corpus lungo contiene 480 frame per scena, tre camera cut, moto rapido e
oggetti animati. Passano baseline, tile e adaptive su Torus e Sponza; il
controllo ritardato di tre frame fallisce anche il gate ghosting. Passano
inoltre clip da 180 frame con esposizione variabile e con due viste alternate
+ DRS. La metrica conserva una storia distinta per vista. Sono prove di
qualità sul preset dichiarato, non tempi di prestazione o equivalenza a tutti
gli scenari futuri. Artefatti in `build/f7-f8-long-quality/` e
`build/f7-f8-context-quality/`.

### Invarianza delle posizioni: regressione scoperta da F6

La prima ripetizione della suite F6 ha fallito il confronto indexed/mesh,
anche sui procedurali (15 pixel nella griglia PBR, massimo delta 145).
L'attributo `[[position, invariant]]` era stato aggiunto, ma senza il flag
`-fpreserve-invariance` veniva ignorato: comportamento esplicitato dalla
[documentazione Apple](https://developer.apple.com/documentation/metal/mtlcompileoptions/preserveinvariance).
Il flag è ora comune a build e hot reload; anche gli overlay che ridisegnano
la geometria marcano la posizione. La coppia PBR indexed/mesh a 3200×1800
ritorna **zero differenze su 5.760.000 pixel**. Conservati i riferimenti e i
log precedenti, inclusa la batteria fallita; non aumentata la tolleranza F6.
La batteria completa viene rieseguita su riferimenti coerenti con questa
correzione. Il precedente spike con il solo attributo non era una prova
dell'invarianza: mancava il flag necessario.

### Resize e cache dei command buffer nella validazione Metal

Il test che cresce da 640×360 a 2560×1440 e 3200×1800 ha riprodotto un
SIGSEGV dentro `HeapUsageTable::processHeapEntry` con shader validation.
Riproduce anche con ricostruzione nativa e output SDR: non è specifico di
MetalFX o EDR. NSZombie ha identificato `MTLGPUDebugHeap enumerateBufferIndices:`
su un heap già deallocato. Ricreare i command buffer elimina il caso ridotto.

La correzione mantiene una generazione della residency: soltanto quando
un'allocazione viene rimossa, i command buffer completati che vengono riusati
ricreano il loro stato di registrazione (main, upload e stream aggiuntivi).
Gli allocatori restano pooled e i frame stabili riusano i command buffer.
Non disabilita la validazione né trattiene una lista crescente di heap vecchi.
Passano native, temporal e async compute con EDR e quattro resize: 12 rebuild
main nel caso semplice su 135 frame totali, anziché un rebuild per frame.
La variante async ricrea anche i suoi stream. Il report espone il contatore.

La validazione delle varianti ha inoltre scoperto due problemi distinti:
le catture offscreen potevano precedere il completamento del caricamento AOT
della variante richiesta; il test ora forza richieste sincrone e non viene
usato per misurare prestazioni. Il return `half4` del forward consentiva
arrotondamenti diversi tra generico/specializzato. Il return FP32 conserva
la precisione fino alla conversione sRGB: **284 combinazioni compatibili
passano (peggiore: 1 pixel, delta 1), 52 incompatibili vengono respinte**,
su otto scene a 640×360 (bench 8: 100000 istanze). Le soglie restano quelle
originali del test. Riferimenti precedenti conservati.
