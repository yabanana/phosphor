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
