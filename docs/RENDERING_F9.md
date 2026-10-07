# F9 — Infrastruttura ray tracing

Stato al 2026-10-07: implementazione e verifica locale completate sul M5;
la PR #18 governa CI e merge. La [roadmap](ROADMAP.md) conserva lo stato
ufficiale. F9 costruisce e verifica le risorse RT; le ombre e l'illuminazione
arrivano con F10–F13. Il default resta `--rt off`.

## Contratti

- BLAS per mesh, indici RT separati dal raster, build batch con scratch
  disgiunti e allineamenti interrogati al device. Compaction differita, refit
  dei vertici e rebuild esplicito. Le AS passano da GpuMemory e residency.
- TLAS per slot del frame ring: un descriptor per slot della GPU scene,
  `userID=slot`, mask zero per slot non valido. Le istanze fuori dalla camera
  restano disponibili per gli altri raggi se le loro mask lo consentono.
- Il registry portabile governa versioni, query di compaction e ritiro delle
  vecchie AS: completamento del frame e sostituzione di tutti gli snapshot
  TLAS sono entrambi necessari. Una richiesta di nuova geometria rende
  obsoleta una query precedente anche prima di pubblicare la nuova versione.
- Risorse AS tipizzate nel grafo; dipendenze AS→Dispatch sia dal TLAS sia
  dai BLAS modificabili. Nessun cast di AS a buffer. Il timestamp iniziale
  include lo stadio acceleration structure.
- Traversal con `intersector`: closest per primarie/AO/diffuse, any-hit per
  ombre. Offset W&B sul punto e normale geometrici in spazio mondo.
  Il culling delle primarie segue il winding nello spazio oggetto; il facing
  restituito al chiamante è geometrico nello spazio mondo, anche su specchiate.
- Alpha IFT generica: LOD0 per ombre e LOD dal cono per primarie. Le istanze
  opache saltano la funzione alpha. La IFT appartiene al PSO risolto e al
  frame slot; hot reload la ricrea soltanto su slot completati.

`updateVertices` conserva indici e numero dei vertici. Il consumer deve
aggiornare anche bounds/normali del raster quando necessario. Il diagnostico
F9 usa la vista RT. Topologia diversa è supportata tramite `loadScene` e
cambio scena; F9 non dichiara aggiornamenti incrementali di indici/topologia
nel backend. Il registry CPU tiene già le revisioni di topologia.

## Diagnostica

```sh
./build/release/phosphor --rt on --debug-rt 1 --bench 1 --scene procedural
./build/release/phosphor --rt on --debug-view rt --debug-mode 1
./build/release/phosphor --rt on --rt-probe shadow --debug-rt 1
./build/release/phosphor --rt on --render-path visibility --debug-rt 1
```

`--debug-rt N` controlla ogni N frame sulla CPU; GPU trace/readback rimangono
attivi ogni frame per mantenere stabile il grafo. Senza probe o vista espliciti
usa fino a 512 primarie stratificate sull'immagine. La vista RT e i probe
usano l'immagine intera; la CPU ne verifica fino a 512 campioni. Il confronto
GPU col V-buffer include invece tutti i pixel.

I probe shadow/AO/diffuse nascono da un trace primario RT di setup, il cui
costo è separato dal trace secondario. `probe_rays` conta i raggi realmente
tracciati dal probe finale; gli slot senza ricevitore sono inattivi. Il report
riporta il probe effettivo, quindi `primary` anche quando attivato dalla sola
diagnostica. I contatori alpha comprendono anche il setup primario.

I controlli negativi `--debug-rt-corrupt transform|mask|blas` devono uscire
con `EXIT 1`. Il riferimento CPU usa BVH a due livelli, intersezione watertight
double, alpha su mip GPU letti esattamente, maschere, generazioni e facing.
Any-hit verifica un occlusore valido, senza pretendere il più vicino.

Il confronto baricentrico verifica dominio e residuo di ricostruzione in
spazio mondo, con budget aritmetico FP32 limitato dalla tolleranza di distanza
S0 (`2e-4` relativa). Una differenza parametrica fissa di `1e-5` non distingue
gli errori da normali amplificazioni su triangoli visti di taglio. I test
negativi includono corruzioni laterali, distanza, grande traslazione e
baricentriche fuori dominio; nessuna assunzione sulle operazioni interne Metal.

```sh
mise exec -- python3 tools/f9_check.py --out build/f9-functional --quick
mise exec -- python3 tools/f9_check.py --out build/f9-lifecycle --only '*flight-*' --only '*dynamic-churn' --only '*switch-resize' --only '*periodic-rebuild'
mise exec -- python3 tools/f9_check.py --out build/f9-transitions --only '*proxy-transition-*'
mise exec -- python3 tools/f9_check.py --out build/f9-deformation --only '*deform*'
tools/hot_reload_check.sh build --rt-only
```

Le prove di transizione modificano realmente materiale/assegnazione o
provocano un upload completo: una mesh inizialmente semplificata deve diventare
Full e comparire correttamente nel readback GPU. Il test pretende anche tutti
i raggi misurati e le operazioni dei due caricamenti. La deformazione richiede
refit, ricostruzioni aggiuntive e compaction effettivi; un run che non le
esegue non passa. Il caso `deform-inflight` controlla ogni 17 frame per
esercitare il frame ring senza attendere la GPU a ogni frame.

## Proxy

`--rt-proxy manifest --rt-proxy-manifest assets/manifests/sponza.rtproxy.json`
attiva il manifest validato per geometria, versione meshoptimizer e indici.
Livelli solo-indici con bordi bloccati; MASK, emissive e materiali sconosciuti
restano Full. Un cambio materiale può promuovere una mesh a Full prima di
iniziare il frame. Manifest assente/incompatibile conserva la geometria piena.

Il manifest Sponza contiene 147.509 dei 262.267 triangoli (56,24%). Nel corpus
S4 dichiarato: errore primarie 0,09279%, ombre 0,04282%, dt95 0,07280 cm,
acne 0,13341%. Sono tre camere 960×540, ricevitori sulla geometria piena,
sole e offset specificati nel manifest. Non è una garanzia generale per nuove
scene, animazioni, distribuzioni di luce o gli offset del consumer F10.
Default proxy off; il consumer deve verificare il proprio errore.

## Pipeline e lifetime

La raccolta `tools/harvest_pipelines.sh` include tutti i kernel RT e il link
statico `rt_trace_rays → rt_alpha_generic`, oltre a present SDR/HDR. `metal-tt`
traduce il dataset. I test controllano anche la presenza del link effettivo.

Su M5 Max/macOS 27.2 (`26B5091g`) il caricamento diretto da `MTL4Archive` di
questa pipeline statica lasciava un `_MTLFunctionInternal` alpha con due
oggetti dipendenti: 736 byte in `leaks --atExit`. Il descriptor non
specializzato non risolveva. Il caricamento tramite `MTL4Compiler` con
`CompilerTaskOptions.lookupArchives` ha invece zero leak, anche con trace e
cambi scena. È un percorso pubblico: il compiler può consultare l'archivio,
senza esporre se lo abbia usato. Perciò il report conta una chiamata Compiler,
non un hit dedotto. Gli altri PSO mantengono il caricamento diretto; nessuna
compilazione avviene sul render thread nei preset verificati.
[API Apple](https://developer.apple.com/documentation/metal/mtl4compilertaskoptions).

L'hot reload è verificato su Sponza MASK: confronto CPU corretto prima e dopo,
poi una copia temporanea della funzione alpha viene alterata. Tutti e tre
gli slot devono eseguire la nuova funzione e produrre il fallimento previsto.
Timeout, segnali o errori Metal non valgono come controllo negativo riuscito.

## Interpretazione delle misure

Report schema 9: byte e AS vive descrivono la scena corrente; `blas_build_ms`
è il tempo GPU dell'ultimo caricamento, separato dalle query di dimensione.
Contatori build/refit/compaction sono cumulativi attraverso reload e cambi
scena. Raggi e tempi del report Engine riguardano i frame misurati. La coda
dei readback viene raccolta prima di distruggere le risorse di una scena.

I tempi delle unità sono contributi esclusivi lungo la catena della coda,
non contatori hardware di tempo attivo. Dopo ricompilazioni del grafo, le
unità del piano precedente non confluiscono nelle statistiche del piano
nuovo: deformazione/resize sono prove di correttezza e lifecycle. Le misure
prestazionali si eseguono a piano stabile e senza diagnostica perturbante:

```sh
mise exec -- python3 tools/f9_bench.py --suite tlas --out build/f9-perf-tlas
mise exec -- python3 tools/f9_bench.py --suite probes --out build/f9-perf-probes
mise exec -- python3 tools/f9_bench.py --suite baseline --baseline-app /path/to/baseline/phosphor --baseline-ref COMMIT --out build/f9-perf-baseline
```

Il runner conserva condizioni, hash asset/binari, ordine e report di tre
repliche; il confronto con baseline è A/B/A. La decisione finale su intervallo
di rebuild è `0` (solo cambi strutturali): 64 frame peggiora il frame medio,
256 frame non guadagna oltre il rumore ed entrambi aumentano il p99.

Il campionamento finale delle unità AS usa una barriera AS→Dispatch e un
dispatch di un thread prima del timestamp preciso. Sul M5/27.2 il timestamp
nudo attribuiva soltanto 0,3 µs al TLAS e spostava il suo costo nelle unità
successive. La prima serie è conservata come non valida per l'attribuzione AS.
L'anchor si applica solo agli encoder Compute con lavoro AS dichiarato;
External/MetalFX e traversal Dispatch ne sono esclusi. Non esiste con timing
disabilitato. La prova con rebuild ogni frame misura 1,585 ms invece del
refit a circa 0,22 ms, rendendo visibile il lavoro effettivamente eseguito.

Tre repliche 1080p, 120 warmup +600 frame: aggiornamento TLAS 100K, mediana
delle medie, **0,2213 ms native / 0,2189 ms Apple9 forzato**, contro il gate
medio di 0,5 ms. Native p95 delle tre run: 0,257–0,340 ms; p99 0,451–0,784 ms,
picco 2,153 ms. Non è una garanzia di deadline su ogni frame del desktop.

Sponza 1080p, costo ammortizzato del probe finale in ns/raggio:

| Percorso | Primary | Shadow | AO | Diffuse |
|---|---:|---:|---:|---:|
| Native M5 | 0,2923 | 0,3268 | 0,2220 | 0,5868 |
| Apple9 forzato sul M5 | 0,2930 | 0,3254 | 0,2272 | 0,5876 |

Tre run per caso, CV delle medie 0,5–2,4%. I secondari escludono il setup
primario dal tempo; in questa vista tutti i pixel hanno un ricevitore.
A/B/A lungo contro `main` `2ccdf6b`, 600+3000 frame ×3: variazione mediana
del frame −0,16% forward, +1,22% temporal, −0,05% instances; variazione p99
mediana −1,12%, +2,54%, +0,41%. Sono entro il criterio materiale del 3%.
Le app desktop sono rimaste aperte; alcune repliche hanno interferenza e
picchi maggiori. La prima serie breve rumorosa e tutte le repliche lunghe
sono conservate, senza eliminare frame o scegliere soltanto i risultati migliori.
[Dati](results/F9-M5Max-2026-10-07.json).

## Verifiche già eseguite e limiti

- 36 casi funzionali, 12 lifecycle, 8 transizioni proxy e 4 deformazioni
  passati sul M5, includendo il percorso Apple9 forzato.
- F9-K1 native/Apple9: zero errori API/shader validation.
- Visual default e RT on: zero pixel diversi dalla baseline su otto scene.
  F6 quick, F7/F8 funzionale/qualità/contesto, archive, hitch e MetalFX lifetime
  passati. Scenario archive di OS diverso richiede l'artefatto CI esterno.
- Accordo V-buffer a 640×360: Torus 99,99304%, Sponza 99,99609%, identici
  native/Apple9. Nel corpus osservato nessuna discordanza opaque interna;
  le discrepanze riguardano bordi/coverage/pareggi. Tutti i pixel sono inclusi.
  Il runner verifica completezza dei dati: l'accettazione delle discrepanze
  resta una revisione distinta, non un PASS automatico del solo conteggio.
- Varianti complete: mesh 284 compatibili/52 negativi PASS (peggiore 35 pixel,
  delta 1); indexed conserva i 28 fallimenti noti su Sponza debug 1/2, con
  esattamente 48.081/69.903 pixel e delta 49/52 della baseline OPT-2.0.
  L'archivio CI macOS 26 è stato verificato anche come fallback da OS diverso.
- Hot reload Release con archivio attivo e funzione alpha avvelenata: PASS;
  l'archivio consultato dal Compiler non nasconde la nuova funzione.
- CI e merge restano registrati nella PR #18, separati dall'accettazione locale.
- Unico dispositivo fisico M5 Max 128 GB. Apple9 forzato non certifica M3/T0.
  F9.6 ray binning resta candidato non attivato.
