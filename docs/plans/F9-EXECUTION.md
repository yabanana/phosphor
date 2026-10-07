# F9 — Infrastruttura ray tracing: piano di integrazione (dopo gli spike)

Piano approvato recuperato dalla sessione Claude; esecuzione ripresa il 2026-10-07.
Gli agenti disponibili in questa sessione mantengono la divisione per file e la
serializzazione della GPU. Nessuna nuova fase OPT/F10+ viene attivata.

## Contesto

F9 porta nel motore l'infrastruttura RT che F10–F13 (ombre RT, ReSTIR, GI,
riflessi) consumeranno: BLAS per mesh/proxy, TLAS per frame dalla GPU scene,
libreria di traversal, alpha test RT. Gli spike S0–S5 sono fatti, misurati x3
a macchina quieta e registrati (`docs/opt-log.md` "F9 — Spike", commit
`3e66849`, branch `phase/f9`, worktree degli agenti già rimossi). Le loro
decisioni sono il contratto di questo piano:

| Spike | Misura (M5 Max, 27.2) | Decisione |
|---|---|---|
| S1 BLAS | heap align 1 KiB; 103 BLAS Sponza 22,8 ms con scratch condiviso+barriere, **2,6 ms con scratch disgiunti**; compaction 0,552; refit 3–5× più veloce ma traversal 1,33× dopo grande deformazione; **AS→Dispatch obbligatoria** (senza: 20/20 frame tutti i raggi stantii) | BLAS in heap di piazzamento GpuMemory, build batch senza barriere, compaction asincrona, refit + rebuild |
| S1e | crash shader-validation riusando un command buffer dopo il rilascio di un heap (anche buffer semplici) | difetto noto già aggirato da `MetalContext::refreshCommandBuffer`: le AS escono dalla residency via `evict` |
| S2 TLAS | 100K: rebuild ≥1,06 ms, **descrittori+refit 0,15 ms**; lo spike storico usava CCW sulle specchiate; integrazione corretta: culling object-space e facing world-space | strategia A (descrittore per slot, mask 0 se non valido, userID = slot), refit per frame, rebuild a cambi di struttura |
| S3 traversal | intersector 0,18–0,29 ns/raggio; query 1,34–1,71×; any-hit −8% ombre; W&B world 0 auto-hit | solo `intersector`, any-hit per ombre, offset W&B in spazio mondo |
| S4 proxy | rapporto fisso: 25% ombre false; **politica adattiva: 55% triangoli, ombra 0,05%, dt95 0,07 cm** | proxy solo-indici, livello per mesh cotto offline con errore dichiarato; senza manifest la mesh resta piena |
| S5 alpha | A (funzione generica) 0 errori, costo ≤1%; B (slot per materiale, HW M5) nessun guadagno | strategia A su Apple9 e Apple10; LOD 0 ombre, LOD da cono primari |

F9.6 (ray binning) resta **candidato non attivato** (nessuna misura di guadagno).

Il proxy è usato solo con manifest misurato valido; MASK/emissive e fallback restano Full. Una nuova assegnazione protetta richiede il reload prima di `beginFrame`, anche quando SceneStore produce upload completi anziché delta. Topologia diversa passa per `loadScene`/bench switch; il refit e il forced rebuild a frame caldo conservano indici e range di vertici, senza anticipare streaming/cooker F21–F22.

## Architettura

Default `--rt off`: nessun consumatore prima di F10, il frame resta identico
(A/B/A contro OPT-4.16). Con `--rt on`:

**Portabile (`phosphor_core`, testato su Linux)**
- `src/renderer/gpu_types.h`: `GPURtInstanceDesc` (72 B, = `MTLIndirectAccelerationStructureInstanceDescriptor`, static_assert), `GPURtMesh` (id BLAS 2×u32, offset/contatori geometria RT e livello proxy; opacità per istanza/materiale, non per mesh), `GPURtParams`, `GPURtCounters`, `GPURtRay`/`GPURtHit` (payload piccolo: t, slot, primitive, bary, front), `GPURtProbeParams`; costanti `RT_MASK_*`.
- `src/renderer/rt_scene.{h,cpp}` (nuovo): registro mesh/proxy→BLAS (stato: Pending→Built→CompactionQueued→Compacted, versione, byte, retire con frame dell'ultimo lettore), pianificazione per frame (budget di build/compaction/refit, rebuild su topologia), politica TLAS (rebuild su capacità/insieme BLAS cambiati o ogni N frame; refit altrimenti). Nessun tipo Metal.
- `src/renderer/rt_proxy.{h,cpp}` (nuovo): livelli solo-indici con meshoptimizer (bordi bloccati), lettura/scrittura del manifest `assets/manifests/<scena>.rtproxy.json` (livello e errore dichiarato per mesh), logica della politica adattiva di S4 (promozione per colpa).
- `src/renderer/rt_reference.{h,cpp}` (nuovo): riferimento CPU a due livelli (watertight double, BVH per mesh, istanze) portato da `bench/f9_spike/f9_common` per `--debug-rt`.
- Il caricamento in `AccelerationStructures::loadScene` usa `rtBuildProxyGeometry`: genera gli indici RT senza alterare la geometria raster e promuove a Full le mesh protette. Non serve un secondo hook nel mesh processing.

**Metal (`src/platform/metal`)**
- `GpuMemory::newAccelerationStructure(size, category, label)` + `release`: heap di piazzamento (`heapAccelerationStructureSizeAndAlign` interrogato a runtime: 1 KiB misurato su M5; altri dispositivi da verificare), `MemoryCategory::RayTracing` (budget, report, test del budget), rilascio differito con `evict` (⇒ ricostruzione command buffer, S1e). Scratch di build in buffer temporanei di caricamento; scratch TLAS persistente.
- `acceleration_structures.{h,cpp}` (nuovo, `AccelerationStructures`): BLAS per mesh/proxy che leggono in place i buffer `GPUVertex`/indici di `SceneRenderer` (vertex range all'offset della mesh, stride 48), build batch con scratch disgiunti al caricamento (bloccante come gli upload di geometria), dimensioni compatte (8 B) lette al completamento, copia compatta in frame successivi, refit/rebuild su richiesta del registro, tabella mesh→BLAS (frame ring), TLAS + buffer descrittori persistenti dimensionati solo ai cambi di struttura (O7: 0 allocazioni nei frame misurati).
- Pass del grafo (dichiarati in `Engine::declareFrameGraph`, solo con `--rt on`):
  - `RT instances` (Compute): legge `Scene data` a Dispatch dopo `Scene transforms`, scrive `RT descriptors` (import persistente) a Dispatch: kernel strategia A (mask 0 slot non validi, CCW sulle specchiate, userID = slot).
  - `RT TLAS` (Compute): legge descrittori e BLAS ad AccelerationStructure, scrive `RT TLAS` (import persistente) ad AccelerationStructure (refit in place o build).
  - `RT BLAS maintenance` (Compute, AS) **solo quando ha lavoro** (la chiave del grafo include il flag; un pass vuoto non deve stare nel grafo).
  - Consumatori: leggono `RT TLAS` a Dispatch ⇒ il grafo emette la barriera AS→Dispatch (stadio già mappato in `metal_graph_executor.cpp:91`).
- Pipeline via `PipelineCache`: `PipelineDesc` esteso con funzioni linkate staticamente (chiave, harvest, test della chiave); tabella delle intersection function creata dalla pipeline risolta (una funzione alpha generica, `setBuffer` per materiali/texture/vertici/istanze), ricreata quando la pipeline cambia (hot reload).

**Shader**
- `shaders/rt_common.h` (nuovo): wrapper `intersector` closest/shadow-any (payload piccolo), `rtOffsetRay` W&B in spazio mondo, funzione alpha generica (regola del raster, LOD esplicito: 0 ombre, cono primari).
- `shaders/rt_scene.metal` (nuovo): descrittori TLAS, vista di debug, raggi di controllo, sonda costo per raggio, accordo col V-buffer.

**Debug, misura, report**
- `--rt off|on`, `--rt-tlas-rebuild-every N` (default da misura del degrado in motore), `--rt-proxy off|manifest`.
- `--debug-view rt` (anche fuori dal percorso mesh): raggi primari dalla camera, colore per slot/normale/hit.
- `--debug-rt N`: ogni N frame raggi primari campionati dalla camera per default; con `--rt-probe` esplicito si controllano anche shadow/AO/diffuse. Raggi e hit sono letti indietro e confrontati con `rt_reference` con tolleranze geometriche dichiarate e ambiguità alpha conteggiata separatamente (matrici mondo copiate nello stesso frame); `--debug-rt-corrupt transform|mask|blas` deve fallire (exit 1). Con `--visibility` anche accordo pixel per pixel col V-buffer (slot + profondità, % riportata).
- `--rt-probe primary|shadow|ao|diffuse`: primari dalla camera, secondari dal setup RT primario (normale geometrica e W&B), un raggio per pixel valido. Il costo ns/raggio riguarda il pass di traversal selezionato; generazione e setup primario sono unità separate. Depth/V-buffer è il confronto indipendente, non il generatore dei secondari F9.
- Report schema 9 `rt`: BLAS (numero, byte, compatti, scratch, build ms al caricamento), TLAS (istanze, capacità, byte, scratch, refit/rebuild), proxy (mesh, % triangoli, errore dichiarato), probe (tipo, raggi, ns/raggio), check (conteggi, fallimenti, accordo V-buffer %), percorso scelto (strategia alpha, famiglia effettiva). Riga `RT` nel log come `SCENE`.

## Ordine dei pacchetti e deleghe

0. **Contratto (io)**: gpu_types, header `rt_scene.h`/`rt_proxy.h`/`rt_reference.h` con API e stub, `MemoryCategory::RayTracing`, opzioni CLI e `RtReport` (schema 9), stub `tools/apple-sdk-stubs` se servono nuovi header; ctest + `metal_syntax_check` verdi. Commit `F9.0`/ID pertinenti.
1. **Agenti Sonnet in parallelo** (worktree, `git merge --ff-only phase/f9`, `FETCHCONTENT_SOURCE_DIR_*` su `build/release/_deps`, solo i propri file, controlli negativi, nessuna misura GPU, wrapper GPU con lock):
   - **A1** `rt_scene.cpp` + `rt_reference.cpp` + test doctest (macchina a stati, retire, politica TLAS, riferimento contro casi noti).
   - **A2** `rt_proxy.cpp` + test + `tools/rt_proxy_cook` (politica S4 produttivizzata, GPU via harness `bench/soc`) → `assets/manifests/sponza.rtproxy.json`.
   - **A3** `shaders/rt_common.h` + `shaders/rt_scene.metal` contro il contratto, provati con un benchmark `F9-K1` in `bench/f9_spike` (kernel del motore compilati da sorgente come `F6-S3`).
   - **A4** CLI/report: parsing e test di `launch_options`, sezione `rt` di `bench_report` + test schema, `tools/perf_table.py`/testdata se servono.
2. **Percorso critico (io)**, nell'ordine F9.3 → F9.1 → F9.2 → F9.4 → F9.5: GpuMemory AS, `AccelerationStructures`, pass e barriere nel grafo, `PipelineDesc` linkato + IFT, integrazione nell'Engine, `--debug-view rt`, `--debug-rt`, probe, accordo V-buffer; un commit per task (`F9.3: …`, `F9.1: …`).
3. Verifica, misure, documenti, PR, CI, merge; rimozione worktree/branch degli agenti.

## Verifica (uscita)

- **F9.2 / uscita**: bench 8 `--instances 100000 --rt on` (istanze dinamiche, moto GPU), `--no-vsync --gpu-timing-serial`, 3 run x CV: unità `RT instances` + `RT TLAS` ≤ 0,5 ms sul M5 (T2 Max) — certificazione degli altri T2 `EXTERNAL_VALIDATION_PENDING`; degrado del traversal dopo K refit misurato per scegliere il default di `--rt-tlas-rebuild-every`.
- **Costo per raggio** per tipo (primary/shadow/ao/diffuse) con `--rt-probe` su Sponza 1080p, 3 run; Apple9 (`--force-family apple9`) per lo stesso preset.
- **Correttezza**: `--debug-rt 1` su bench a mesh piccole (1, 5, 7, 8 a 10K) e Sponza: 0 fallimenti; ogni `--debug-rt-corrupt` FAIL; accordo col V-buffer (% dichiarata, discordanze spiegate su spigoli/pareggi); alpha RT coerente: accordo V-buffer sui pixel con materiali MASK; `--force-family apple9` esercitato con gli stessi controlli.
- **Proxy**: errore dichiarato nel manifest e nel report, `--debug-rt` contro proxy vs piena. Diagnostica `--debug-rt-proxy-transition mask|emissive|reassign|full-upload`: partire da una mesh realmente ridotta, osservare modifica e tipo di upload, promozione Full e snapshot GPU di istanza/materiale coerente; il solo PASS del checker non basta. Il runner verifica il marker dedicato e fallisce se il controllo non è terminato. Nessun esito runtime è dichiarato dalla sola implementazione del controller.
- **Motore invariato**: ctest Debug/Release, `metal_syntax_check` (Linux), `visual_check` con `build/reference` senza flag e con ogni flag di debug (+ `--rt on` e `--debug-view rt` con riferimenti nuovi motivati), `f6_check`, `f7_f8_check` funzionale/qualità/contesto, `archive_check` (+ `harvest_pipelines` se cambiano i descrittori), `variant_check` (bench 4 modi 1/2 = divergenza nota OPT-2.0, non deve peggiorare), `hitch_check`, `hot_reload_check`, `metalfx_lifetime_check`, `leaks --atExit` 0 (anche con `--rt on` e switch), 0 allocazioni GPU nei frame misurati, somma unità = span, A/B/A contro OPT-4.16 (`f7_f8_bench`, `perf_record`), immagini guardate.
- **Documenti**: perf-log, opt-log (scoperte d'integrazione), ROADMAP (spunte solo verificate; F9.6 non spuntato), HARDWARE_VALIDATION (riga F9), CLAUDE.md (regole RT), README/plan F9; CI verde; PR con handoff dettagliato.

## Rischi e ripieghi

- Refit TLAS degradato su moti grandi → rebuild periodico (misurato) o su soglia.
- Pass RT vuoti che rompono i timestamp → solo pass con lavoro garantito nel grafo.
- Crash della shader validation al rilascio degli heap → rilascio sempre tramite `evict`; verifica con bench switch sotto validation.
- Linking statico nell'archivio AOT (`metal-tt`) non supportato → la pipeline RT resta fuori dall'archivio (fallback compilazione) e lo si dichiara.
