# F8.4 — causa del ciclo MetalFX e rilascio in-process

M5 Max 128 GB, macOS 27.2 beta 2 (`26B5091g`), MetalFX **40.9**, Xcode 27.0.
Base: `a2b2b96`; correzione: `fe622ac` e successivi del branch
`phase/f8.4-metalfx-cycle-release`.

**Esito:** il difetto è un riferimento forte dello scaler verso se stesso,
tenuto dal suo filtro interno. Il motore lo rilascia in-process, senza
worker, senza patch al framework e senza scritture su ivar private. Lo scaler
temporale standard (stesso algoritmo, DRS, reactive mask, più viste) torna a
essere distrutto a ogni ricreazione.

## Causa

Diagnosi con un programma separato (introspezione del runtime Objective-C e
scansione conservativa della memoria, mai nel motore):

- lo scaler Metal 4 è `_M4FXTemporalScalingEffectBBR` (Metal 3:
  `_MFXTemporalScalingEffectBBR`); possiede `_filter`, un
  `std::unique_ptr<BBRNet_Filter<MFXDevice4>>`;
- l'oggetto C++ `BBRNet_Filter` contiene, all'offset 8, il puntatore allo
  scaler; quel riferimento è **forte**: subito dopo la creazione, con il pool
  già svuotato e un solo proprietario esterno, il retain count è **2**
  (Metal 3 e Metal 4). Il controllo spaziale vale **1**;
- scaler → `unique_ptr` → filtro → scaler: il conteggio non arriva mai a zero,
  quindi `.cxx_destruct`, che distruggerebbe il filtro, non viene mai eseguito.
  Il filtro trattiene le sue risorse GPU (circa 20 MiB a 640×360, 165 MiB a
  1080p per istanza nel riproduttore).

Questo spiega tutte le osservazioni precedenti: SDK, API Metal 3/4, ARC Swift,
attese, drain GPU e opzioni pubbliche non potevano cambiare un riferimento
interno al framework. I nomi di classe e ivar sono serviti solo alla diagnosi.

## Rimedio

[`metalfx_lifetime`](../../src/platform/metal/metalfx_lifetime.h):

1. `adoptTemporalScaler`, sul thread di creazione dopo il drain del pool:
   riferimenti interni = `retainCount − 1`.
2. `releaseTemporalScaler`, quando la GPU non usa più lo scaler: weak
   reference, `release` del chiamante; se l'oggetto sopravvive, i riferimenti
   interni registrati erano **esattamente 1** e ora il conteggio è
   esattamente quello (sonda + autoriferimento), rilascia l'autoriferimento.
   La weak reference deve poi risultare nulla.
3. Ogni altra forma riceve solo il rilascio ordinario ed è contata:
   `retained` (perdita, `METALFX-LIFETIME … | FAIL`, exit 1) o `unknown`.
   Se Apple corregge il framework, i riferimenti interni diventano 0 e il
   percorso degenera in un `release` normale senza modifiche al codice.

Durante la deallocazione il filtro rilascia il proprio riferimento a un
oggetto già in deallocazione: il runtime Objective-C lo ignora (oggetti con
isa non-pointer). Verificato con `NSZombieEnabled`, Guard Malloc +
`MallocScribble`, API/shader validation e `leaks`.

API usate: `retainCount`/`release` di `NSObject` e le funzioni weak dell'ABI
ARC (`objc_initWeak`, `objc_loadWeakRetained`, `objc_destroyWeak`). Nessuna
API privata, swizzling, scrittura di ivar o selezione per nome di classe.

Nel motore lo scaler sostituito viene ritirato con `MetalContext::deferCall`
dopo i frame in volo e distrutto su un worker utility di `PipelineCache`: sul
render thread la deallocazione attendeva fino a ~150 ms l'inizializzazione
MetalFX in corso su altri thread (lock interno; da solo il dealloc costa 1–2 ms
anche a 3200×1800). Il percorso in-process diventa quello di
`--upscaler temporal`; i worker isolati restano opt-in
(`--metalfx-mode isolated`).

## Riproduttore minimo (solo API pubbliche)

[`metalfx_lifetime_matrix.mm`](../../bench/f8_spike/metalfx_lifetime_matrix.mm),
8 creazioni/rilasci, 640×360 salvo indicazione:

| Variante | Vivi | Delta allocazioni device | `leaks` |
|---|---:|---:|---:|
| temporal4, rilascio semplice (controllo negativo) | 8/8 | 167.133.184 B | ciclo |
| temporal3, rilascio semplice (controllo negativo) | 8/8 | 167.133.184 B | ciclo |
| temporal4 `--release-cycle` | 0/8 | 1.720.320 B | 0 |
| temporal3 `--release-cycle` | 0/8 | 1.720.320 B | 0 |
| temporal4 `--release-cycle --gpu-drain`, 1920×1080 | 0/8 | 1.327.104 B | — |

Esperimento con encoding MTL4 reale (3 frame per scaler, residency set,
attesa sull'evento): 60 cicli → 0 vivi, delta oscillante 1,3–2,3 MB senza
tendenza; 1080p ×30 → 0 vivi, 9,6 MB stabili. Senza rilascio del ciclo
4 scaler con encoding: 4 vivi, +83 MB.

## Motore

Resize ogni 30 frame, due viste, 240 frame, output 2560×1440/3200×1800,
pacing diagnostico 64 ms (test di lifetime, non di prestazioni; questa
esecuzione precede lo spostamento del dealloc sul worker, che non cambia
l'esito lifetime):

| | Direct senza rimedio (PR #15) | Isolated (oggi) | Direct con rilascio del ciclo (oggi) |
|---|---:|---:|---:|
| Footprint fisico a fine misura | 7.440.994.504 B, solo parent, in crescita | 2.946 MiB parent + worker | 2.352 MiB |
| Picco del footprint campionato | — | 4.244 MiB | 3.324 MiB |
| Frame temporali | 225/240 | 226/240 | 233/240 |
| Scaler distrutti | 0 | 16 worker recuperati | 16/16 (`cycle-released`) |

Footprint fisico da `proc_pid_rusage` ogni 100 ms
(`tools/metalfx_lifetime_check.py`). Il direct oscilla con le due dimensioni
di output (2.351–2.935 MiB) senza crescita. `leaks --atExit` del renderer: **0 leak**, footprint finale
295,5 MB. API + shader validation (Debug, resize ogni 20 frame, due viste):
12/12 scaler distrutti, `EXIT 0`.

Stallo al ritiro, stesso scenario: con il rilascio sul render thread frame
max **232 ms** e attesa max **167 ms**; con il ritiro sul worker frame max
**85 ms** (64 ms di pacing) e attesa max **0,28 ms**.

Soak con il codice finale (`tools/metalfx_lifetime_check.py --direct-only --frames 1200`): 40
resize × 2 viste = **80 scaler creati e 80 distrutti** (`cycle-released`),
1.161/1.200 frame temporali, `EXIT 0`. Footprint fisico per quartile della
corsa, mediana/massimo: 2.443/3.486, 2.693/3.327, 2.698/3.328,
2.698/3.328 MiB: nessuna crescita dopo il primo quarto.

## Qualità, DRS e più viste

Sul percorso in-process, senza cambiare soglie o riferimenti:

- `tools/f7_f8_check.py --build build` (Debug, API + shader validation):
  **26 PASS**, confronti immagine a 0 pixel diversi;
- `--build build/release --quality-only`: **16 PASS** (clip da 480 frame,
  b1/b4, temporal/adaptive/tile); metriche come l'accettazione PR #14:
  b4 identica a 6 decimali, b1 PSNR sRGB 34,8535 → 34,8519 dB, flicker e
  ghosting di supporto invariati;
- `--context-quality --quality-only`: **24 PASS**, incluse esposizione e
  DRS con due viste;
- in tutti i run temporali delle tre suite ogni scaler è `cycle-released`,
  `retained 0`.

## Prestazioni

`tools/metalfx_isolation_bench.py`, Release, 1920×1080, input 75%, 120 warmup
+ 600 frame, tre repliche in ordine ruotato, offscreen, senza vsync né
validation. Mediana dei tempi medi; macchina non completamente quieta (app
ChatGPT/Codex e Claude aperte, nessun altro carico GPU).

| Carico | In-process | Isolated | In-process vs isolated | Footprint in-process / isolated totale |
|---|---:|---:|---:|---:|
| Sponza | 2,1369 ms (p95 2,1858) | 2,1328 ms (p95 2,2196) | +0,19% (rumore) | 1.930 / 2.213 MiB |
| 1.024 luci | 11,2585 ms (p95 11,4514) | 12,0505 ms (p95 12,3099) | −6,6% | 1.347 / 1.635 MiB |
| Sponza, due viste alternate | 1,2297 ms (p95 1,2748) | 1,8625 ms (p95 1,9374) | −34,0% | 2.101 / 2.696 MiB |

Il rilascio del ciclo non aggiunge lavoro per frame: avviene solo alla
distruzione dello scaler. I tempi in-process coincidono con il direct
storico di PR #15 (2,1503 / 12,0212 / 1,2743 ms, altra sessione) entro la
variabilità tra sessioni.

## Regressione nativa: riferimenti F6 superati da una correzione F7

`tools/f6_check.sh build build/release --quick` falliva con 3 FAIL nel
percorso mesh (visual check two-phase 3/8/154 pixel; overflow forzato
bench 7: 9; bench 8: 190.241, delta massimo 170), con conteggi identici su
`main` `cc38443`. La stessa batteria sul merge F6 `e600887` passa.
Bisezione con la sonda di overflow: `d9cfacf` 0 pixel, `70a64d8` (F7: shading
condiviso, normali con cofattori, segno della tangente per istanze
specchiate) 186.041 pixel. Sull'HEAD overflow e indexed coincidono (0 pixel)
e mesh contro indexed differisce di 0/1 pixel: il percorso mesh è coerente,
è cambiata l'immagine del percorso base.

Attribuzione su bench 8 (5% di istanze specchiate), annullando
temporaneamente le righe dello shader indexed: la sola correzione della
tangente spiega 185.854 pixel; restano 4.579 pixel, di cui 4.448 con delta 1
(riordino aritmetico) e 131 con delta maggiore, non attribuiti singolarmente.
Gli altri bench differiscono solo per delta 1–2 (bench 5: 221.237 pixel a
delta 1). È una correzione voluta e coerente con la convenzione glTF del
progetto; F7 non aveva aggiornato i riferimenti F6.

Riferimenti rigenerati con gli strumenti F6 (`tools/visual_check.sh build
build/reference-f6base --update`; `EXTRA_ARGS="--geometry-path mesh
--meshlet-cull off"` per `build/reference-f6mesh`), conservando i precedenti
in `build/reference-f6*-pre-f7`. Batteria quick: **40 ok, 0 FAIL**.

## Ritiro al resize: attesa della dimensione stabile

Con ridimensionamento continuo ogni dimensione intermedia richiedeva un
nuovo scaler, superato prima dell'uso. Ora lo scaler in-process viene
richiesto dopo `--metalfx-resize-settle N` frame con output invariato
(default 4; 0 = comportamento precedente; la prima dimensione resta
immediata e i worker isolati non cambiano). Uno scaler sostituito passa
sempre dal ritiro differito.

| Resize ogni 2 frame, due viste, 400 frame (3 ripetizioni) | settle 0 | settle 4 |
|---|---:|---:|
| Scaler creati | 78–83 | 2 |
| Picco footprint | 4,96–5,11 GB | 2,53 GB |
| Durata | 0,95–1,05 s | 0,68–0,72 s |

Costo: con resize ogni 30 frame i frame temporali passano da 233 a 205 su
240 (lo scaler arriva 4 frame più tardi; ~33 ms a 120 fps). DRS non è
coinvolta: cambia l'input, non l'output. Controllo lifetime e suite F7/F8
funzionale (26/26) rieseguiti con il default.

## Revisione avversariale

Dopo la prima consegna il proprietario ha chiesto di cercare impatti non
considerati. Punto di partenza: con il difetto nessuna app distrugge questi
scaler, quindi `~BBRNet_Filter` è un percorso che il framework non esercita.
Ogni ipotesi è un test con controllo negativo e positivo (sorgenti di prova
in scratchpad, non nel repository).

| Ipotesi | Prova | Esito |
|---|---|---|
| Distruggere uno scaler altera un altro scaler vivo (stato condiviso) | Output dello scaler vivo, hash bit a bit per frame, contro un controllo indisturbato; distruzione di uno scaler uguale, di uno diverso, o creazione/distruzione continua su un altro thread; anche sotto API + shader validation | 0 frame diversi; determinismo 0/24; controllo positivo (un pixel di input cambiato) 12/24 diversi |
| Il dealloc concorrente blocca l'encoding del render thread | Tempo di `encodeToCommandBuffer:` con distruzioni parallele | max 0,80–0,86 ms contro 0,21–0,53 ms; nessuno stallo |
| Il secondo `release` dipende dall'isa ottimizzata | `OBJC_DISABLE_NONPOINTER_ISA=YES`, zombie, Guard Malloc + scribble | nessun errore, 0 vivi |
| La coda utility ritarda il recupero della memoria | Latenza fra ritiro e distruzione, anche senza archivio pipeline | 12–44 µs (18 thread) |
| Un layer Metal avvolge lo scaler | Classe e conteggio sotto cattura, validation, shader validation, HUD | **Solo la cattura GPU avvolge** (`CaptureMTL4FXTemporalScaler`): il rilascio distruggeva l'involucro e il controllo dichiarava PASS mentre lo scaler interno perdeva ~20 MB per ricreazione |
| Un riferimento transitorio imita la firma | Riproduttore ripetuto 25 volte | 11/100 casi con 2 riferimenti: artefatto dell'inserimento in `NSHashTable` prima della misura; con l'ordine corretto 100/100 rilasciati |

Ridimensionamento continuo (resize ogni 2 frame, due viste, output fino a
3200×1800, come il trascinamento del bordo): 400 frame → 82 scaler creati e
82 distrutti; 800 frame → 162 e 162. Render thread: frame max 28,5 ms,
attesa max 9,0 ms. **Picco del footprint 5,15 GB e 5,12 GB**: limitato e
indipendente dalla durata, ma alto. Per vista convivono uno scaler in
creazione, uno attivo e uno in ritiro; solo 25 frame su 400 sono temporali,
perché ogni scaler è superato prima di diventare utile. Non è un effetto del
rimedio (senza rimedio ogni scaler resterebbe vivo), ma è un costo reale
della ricreazione: la mitigazione naturale è attendere che la dimensione
sia stabile prima di crearne uno nuovo, restando nel fallback nativo durante
il trascinamento: implementata, vedi sopra.

Correzioni conseguenti (codice):

- sotto cattura (`MTLCaptureManager supportsDestination:` pubblico) i
  rilasci sono contati come `wrapped` e il risultato è **UNVERIFIED**, mai
  PASS; avviso esplicito: durante la cattura ogni ricreazione perde memoria;
- il secondo `release` è limitato alle versioni di MetalFX in cui difetto e
  rimedio sono verificati (`40.9`), oltre al controllo sul singolo oggetto.
  Su una versione nuova: rilascio semplice; se il difetto persiste il leak è
  visibile (`retained`, exit 1) e si rivalida, invece di rischiare un
  rilascio in eccesso nel caso "framework corretto + riferimento
  transitorio". `PHOSPHOR_METALFX_UNVERIFIED=1` forza quel percorso:
  controllo negativo con 8/8 trattenuti e `FAIL`;
- il riproduttore misura prima di inserire lo scaler nel monitor weak.

## Limiti

- Il rimedio è attivo solo su MetalFX 40.9. Ogni aggiornamento di macOS
  richiede di rieseguire riproduttore e controllo lifetime: con una versione
  nuova il motore torna al rilascio semplice finché non è verificata.
- Con la cattura GPU attiva lo scaler interno non è raggiungibile: ogni
  ricreazione durante una sessione di cattura perde memoria fino all'uscita.
- Il secondo `release` richiede che l'oggetto sopravviva con un solo
  proprietario oltre alla sonda e che alla creazione ce ne fosse
  esattamente uno: ogni altra forma dà un leak visibile, mai un crash.
- Non è stata inviata alcuna segnalazione ad Apple; la bozza pronta con
  riproduttore ARC è in [`APPLE_FEEDBACK.md`](../../bench/f8_spike/APPLE_FEEDBACK.md).
- Il confronto con macOS 27.0.1 stabile non è più necessario al gate e non è
  stato eseguito.

Dati compatti: [risultati](../results/MetalFX-in-process-M5Max-2026-10-04.json).
Grezzi locali: `build/f84-cycle/`.
