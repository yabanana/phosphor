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

## Regressione nativa

`tools/f6_check.sh build build/release --quick`: 37 controlli ok e 3 FAIL nel
percorso mesh (visual check two-phase: 3/8/154 pixel; overflow forzato
bench 7: 9 pixel; bench 8: 190.241 pixel, delta massimo 170). Lo stesso
comando su `main` `cc38443`, con gli stessi riferimenti F6 del 2026-10-02,
dà **conteggi identici**: residuo deterministico preesistente, non
attribuibile a questo lavoro e non spiegato qui. Resta aperto e separato
(log: `build/f84-cycle/f6-check-quick*.out`).

## Limiti

- La firma (un riferimento interno esatto) è misurata su questo runtime. Un
  runtime con forma diversa produce una perdita **visibile** (`retained`,
  exit 1), mai un rilascio in eccesso: il secondo `release` richiede che
  l'oggetto sopravviva con un solo proprietario oltre alla sonda.
- Rischio residuo: un runtime futuro in cui l'unico riferimento interno a
  creazione sia transitorio e ancora vivo al rilascio. Il riproduttore e il
  controllo lifetime vanno rieseguiti a ogni aggiornamento di macOS.
- Non è stata inviata alcuna segnalazione ad Apple.
- Il confronto con macOS 27.0.1 stabile non è più necessario al gate e non è
  stato eseguito.

Dati compatti: [risultati](../results/MetalFX-in-process-M5Max-2026-10-04.json).
Grezzi locali: `build/f84-cycle/`.
