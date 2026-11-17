# Track Studio ↔ Phosphor: consumo, deriva, sotto-uso e lezioni sui materiali per F15

Audit in sola lettura del 2026-11-13 (letture fino alle 09:56 CET). In REA_RTB
non ho modificato, compilato né eseguito nulla, e non ho avviato processi GPU.
Non cito `Race Track Builder/` né `evidence/decompiled/`.

**Legenda.** **[V]** = verificato nel codice o in git al commit indicato. **[R]** = riportato
da documenti o ricevute di Track Studio, non rimisurato da me. **[I]** = mia inferenza.

**Versioni citate.**
- **TS** è il repository Git di Track Studio, che vive in `REA_RTB/work/parallel-base-20261024`.
  La radice `REA_RTB/` è un mirror (`REA_RTB/README.md`, «La root è il mirror»). Il suo
  `main` è `c91ff46` (12 nov, 21:47). Le citazioni `TS:percorso:riga` si riferiscono a
  `c91ff46`, salvo indicazione diversa.
- **geo** è il branch `geology/phase-1` a `993e1e3` (13 nov, 09:41), nel worktree
  `work/terrain3d-21-product`.
- **vendor** è `REA_RTB/vendor/phosphor`: un clone `--shared` di Phosphor a `a83fcf7`
  con modifiche locali non committate.
- **Phosphor** è `c8ca350` (main).

## 0. Sintesi

- Oggi TS consuma Phosphor per tre vie.
  1. **Bridge del prodotto** `libtrack_renderer`. Compila 16 unità Metal più il core portabile di
     vendor@`a83fcf7`, ma disegna tutto con un proprio shader `rtb_legacy.metal` in un grafo
     di quattro o cinque pass.
  2. **App Phosphor non modificata** (`be886ca`). Rende offline i glTF del nucleo geologico,
     fuori dall'editor.
  3. **Worktree Phosphor `rea/host-embedding`** su `c8ca350`. Aggiunge un'API C per ospitare
     l'`Engine` in un'app AppKit. Il lavoro è iniziato stamattina, è in corso e non è committato.
- **Deriva.** Il bridge è di fatto un secondo renderer con PBR, terreno a strati, acqua, ombre,
  riflesso planare, ambiente e tonemap propri. Contraddice la regola stessa di TS
  («nessun secondo renderer», `REA_RTB/CLAUDE.md`). La copia vendor ha cinque patch locali ed è
  96 commit dietro main.
- **Sotto-uso.** Al prodotto non è collegata alcuna capacità F5–F14. TS lo ha già riconosciuto:
  il piano geologico raccomanda l'opzione A, «terreno dentro Phosphor».
- **Materiali.** Ci sono buone idee trasferibili: pesi locali sparsi senza tetto di strati,
  height blend softmax con regole di attivazione, maschera di cavità in metri, partizione
  dell'unità con precedenza manuale, provenienza con hash e scala fisica, validazioni d'import.
  Altre soluzioni sono workaround che diventano requisiti F15: trasmissione e volume per l'acqua,
  subsurface, coat distinto da coverage, unità fisiche, convenzioni glTF.

## 1. Come Track Studio consuma Phosphor

### 1.1 Copie e versioni

| Canale | Phosphor | Stato | Evidenza |
|---|---|---|---|
| `vendor/phosphor`, il bridge del prodotto Terrain V1 | `a83fcf7` (8 ott, merge PR #21) con 16 file modificati e 1 nuovo, tutti datati 23 ott | In uso. Clone `--shared` con origin `/Users/danielsan/Documents/phosphor` | [V] `git log -1`, `git status`, `.git/config` |
| App Phosphor intatta, usata per le rese geologiche | `be886ca`, binario del worktree Phosphor `f13-f14-closure` | Uso offline via glTF | [V] `work/terrain3d-21-product/.checks/geology-phase-1/canyon-frame-01.json` (`renderer.commit`, `renderer.arguments`), `canyon/view-01/frame-01.log:1`; geo `photoreal-geology-plan.md:99` |
| `work/phosphor-host` | `c8ca350`, branch `rea/host-embedding`. È un worktree del repository Phosphor (gitdir `phosphor/.git/worktrees/phosphor-host`) | 11 file modificati e `src/host/` non committati, toccati fra le 09:44 e le 09:55 del 13 nov | [V] `git status`, `git diff --stat`: 346 righe aggiunte e 130 tolte |
| Radice `REA_RTB/native` | Snapshot del 22–26 ott, senza `terrain-v1`. `rtb_legacy.metal` vi pesa 8,6 KB contro 41,9 KB in `main` | Mirror vecchio, non è la sorgente corrente | [V] `ls` |

**Distanza da main.**
- `a83fcf7..c8ca350` conta 96 commit, 166 file, 28.331 righe aggiunte e 5.380 tolte [V].
- Lo stato dei tick in `docs/ROADMAP.md`:
  - a `a83fcf7`, da F10 a F14 nessun tick;
  - a `c8ca350`, tick su F13.1–F13.6 e F14.1–F14.4;
  - in entrambi, F10–F12 senza tick: anche upstream sono integrati come sorgente ma non accettati [V].
- La copia vendor ha quindi i sorgenti F9–F14, ma non la chiusura F13/F14 (PR #30 e #31). Esempi:
  - qualifica e pensionamento dei timing record MetalFX (`ed2ca22`, `112332d`);
  - ridefinizione di F13.3 (`f85c2c9`);
  - parcheggio degli slot TLAS morti (`4505ee7`);
  - storia delle nuvole nel volo veloce (`938edd9`);
  - gate della probe cotta (`ff322e7`);
  - le altre correzioni registrate in `docs/implementation/F13-F14-*`.

### 1.2 Build e link (bridge)

- `TS:native/CMakeLists.txt:7-8` fissa `PHOSPHOR_ROOT=../vendor/phosphor`. `PHOSPHOR_DEPS` è un
  percorso assoluto nel build del checkout Phosphor originale, `/Users/danielsan/Documents/phosphor/build/_deps` [V].
- Alle righe `:31-36` la libreria `phosphor_editor_core` è il glob di `src/core`, `diagnostics`,
  `pipeline`, `renderer`, `rendergraph` e `scene` [V].
- Alle righe `:37-41` entrano in `libtrack_renderer.dylib` 16 unità Metal: `metal_impl`,
  `metal_context`, `gpu_memory`, `residency_manager`, `upload_ring`, `transient_heap`,
  `memory_pressure`, `pipeline_cache`, `metalfx_lifetime`, `temporal_worker`,
  `metal_graph_executor`, `gpu_timestamps`, `scene_renderer`, `gpu_scene_buffers`,
  `metal_texture_manager` e `frame_capture` [V]. Non sono compilati `engine`, `mesh_renderer`,
  il percorso di visibilità, la luce diretta e indiretta, le ombre, l'RT, il post e l'atmosfera.
- Alle righe `:55-57` la metallib contiene tutti gli shader Phosphor più `rtb_legacy.metal` e
  `rtb_paint.metal` [V]. Il bridge non usa gli shader Phosphor tranne gli helper inclusi
  (§1.3) [I, da grep].
- Alle righe `:82-122` TS riusa i test CPU di Phosphor: SceneStore, mesh processing, gpu scene,
  ECS, glTF, pipeline key, worker pool e testbench [V].

### 1.3 Sottosistemi: usati, aggirati, re-implementati

| Sottosistema Phosphor | Uso nel bridge a `c91ff46` | Evidenza |
|---|---|---|
| `GpuMemory`, `frameUploads`, `stagingAllocate`, `enqueueUpload` | **Usato**: 5 `memory().newTexture`, 4 `memory().newBuffer`, 7 `frameUploads`, 10 `enqueueUpload`. Nessun `device->newBuffer/newTexture` | [V] `TS:native/renderer/track_renderer.cpp` intorno a 379–563 (grep) |
| `PipelineCache` | **Usato**: 17 `pipelines->request`, tutte su funzioni `rtb_*` | [V] `track_renderer.cpp:357-374, 846, 898` |
| Render graph ed executor | **Usato**, con un grafo proprio: `Terrain shadow depth`, `Terrain water reflection`, `RTB legacy`, `RTB multisample resolve` e `Frame capture` | [V] `track_renderer.cpp:797-840` |
| Barriere manuali | Una sola: `barrierAfterEncoderStages` blit→blit dentro un upload al caricamento. È accettabile | [V] `track_renderer.cpp:892` |
| Layout GPU condivisi (`gpu_types.h`) | **Aggirato**: `Params`, `Vertex` e `Instance` sono definiti due volte, in C++ (`track_renderer.cpp:53-69`, solo `static_assert` di dimensione) e in MSL (`rtb_legacy.metal:7-15`) | [V] |
| `SceneStore`, `GpuScene`, `SceneRenderer` (F5) | **Non usati**: zero riferimenti nel codice del bridge, anche se compilati e linkati. `PHOSPHOR_INTEGRATION.md:9-11` (22 ott) li dichiara in uso, quindi il documento non è aggiornato. Sono assenti fin dal primo snapshot Git del 24 ott (`498d36f`) | [V] `git grep`, `git log -S` |
| Mesh shader, Hi-Z, visibility buffer, risoluzione dei materiali (F6–F7) | **Non usati**: forward con una draw CPU per mesh (`drawIndexedPrimitives`, `track_renderer.cpp:665-794`) | [V] |
| PBR Phosphor | **Riuso parziale degli helper del vecchio forward**: `evaluateLight`, `envBRDFApprox`, `hemisphere` e `tonemapACES` da `material_shading.h` (`rtb_legacy.metal:3`), ricomposti con costanti proprie (sole `2.3*x`, ambiente `0.34`) | [V] `rtb_legacy.metal:413, 424, 426` |
| HDR, esposizione, post | **Re-implementati in LDR**: ACES e codifica sRGB per frammento, target `BGRA8Unorm` | [V] `rtb_legacy.metal:426-428`, `track_renderer.cpp:797` |
| F10 ombre | **Re-implementate**: una shadow map ortogonale 1024² adattata ai bounds del terreno | [V] `track_renderer.cpp:71, 798`; `TS:native/terrain-v1/shadow_math.hpp:11-25` |
| F13 riflessi | **Re-implementati**: riflesso planare dell'acqua a ¼ di risoluzione, con parallasse approssimata | [V] `track_renderer.cpp:73, 799-816`; `rtb_legacy.metal:144` |
| F13.5 probe | **Riuso offline**: un cubo da studio 32² con 6 mip, pre-filtrato col kernel Phosphor `reflection_probe_prefilter` e caricato da file | [V] `track_renderer.cpp:563`, `TS:native/terrain-v1/main.swift:1070-1077`; [R] `TS:docs/wiki/terrain-v1.md:52` |
| F9 RT, F11, F12, F13.1–13.4, F14, MetalFX | **Non collegati** (MetalFX è soltanto linkato) | [V] grep |
| Report e diagnostica | **Parziale**: `tr_gpu_bytes` legge la contabilità `GpuMemory`. Niente report JSON, tempi per pass, `GPU allocations` né `image_diff` di Phosphor. TS tiene ricevute proprie | [V] `TS:native/renderer/track_renderer.h:118`; [I] |
| Texture | `MetalTextureManager` usato, con le API aggiunte nella patch vendor (`uploadRgba8`, `loadCompressedTexture`) | [V] |

Il ramo **geo** è un consumatore diverso. Il nucleo `crates/geology-kernel` esporta glTF 2.0
standard, metallic-roughness con texture ORM condivisa per occlusione e metallic-roughness
(`geo:crates/geology-kernel/src/main.rs:185-189`), e lo rende con l'app Phosphor completa:
`--rt on --shadows rt --lighting clustered --gi ddgi --ao gtao --atmosphere on` [V].

## 2. Deriva

### 2.1 Patch alla copia vendor (rispetto a `a83fcf7`; nessuna presente in `c8ca350`)

| Patch | File | Natura | Stato upstream |
|---|---|---|---|
| `destinationBytesPerImage=0` nelle copie texture→buffer 2D (NATIVE-V016, nata dall'indagine sul panic del 23 ott) | `frame_capture.cpp`, `hiz_builder.cpp`, `mesh_renderer.cpp`, `post_processor.cpp`, `temporal_worker.cpp`, `visibility_check.cpp`, `direct_lighting_passes.cpp`, `memory_stress.cpp` | Possibile correzione generale. TS afferma che «Apple richiede 0 quando depth=1» (`TS:docs/reverse/STATUS.md:373`) | [V] `c8ca350` passa ancora `rowBytes * height_` (`frame_capture.cpp:36`) e `u64(w)*h*4` (`hiz_builder.cpp:134`). [I] La regola Apple non è verificata da me, e il nesso col panic non è dimostrato |
| `uploadRgba8` e `forEachRgba8Upload`: upload RGBA8 a righe intere entro la capacità dello staging ring | `metal_context.{h,cpp}`, `metal_texture_manager.cpp`, nuovo `core/memory/texture_upload.h` | Correzione di robustezza per texture più grandi dello staging (256 MiB) | [V] `c8ca350` fa ancora un solo `stagingAllocate(rowBytes*height)` (`metal_texture_manager.cpp:50`) |
| `loadCompressedTexture`: catena di mip BC1–7 autorata e validata (dimensioni ≤ 16384, numero di mip, payload esatto) | `metal_texture_manager.{h,cpp}` | Capacità mancante upstream (adattatore DDS) | [V] assente in `c8ca350` |
| `Blend::LegacyAlpha` e `SpriteAlpha`, `alphaToCoverage`, `sampleCount` (MSAA) nella `PipelineDesc` e nella chiave | `pipeline_desc.h`, `pipeline_key.cpp`, `pipeline_cache.cpp` | Compatibilità con gli stati RTB | [V] assente upstream. [I] Non serve alla linea Phosphor (visibilità + MetalFX); diventa superflua se TS abbandona il percorso legacy |
| `writePng(preserveAlpha)` | `frame_capture.{h,cpp}` | Diagnostica | [V] |

Tutte le modifiche risalgono al 23 ott e la copia non si è più mossa [V]. L'handoff TS dice
«Phosphor in sola lettura» (`REA_RTB/Terraformazione-Terrain3D-handoff.md:10`) e ammette che
«il checkout Phosphor presente sul Mac ha modifiche locali» (`:60`): la copia è patchata, quindi
non è in sola lettura.

### 2.2 Divergenze architetturali

1. **Un secondo renderer dentro il bridge** [V].
   - `rtb_legacy.metal` è nato come percorso di compatibilità RTB (`TS:docs/ARCHITECTURE.md:82`,
     `TS:docs/RTB_RENDERER_BRIDGE.md:3`). Dal 6 novembre ospita anche il PBR del terreno V1, le
     rocce, la natura, l'acqua, le ombre e il riflesso.
   - Le dimensioni sono passate da 8,6 a 41,9 KB per lo shader e da 83 a 143 KB per
     `track_renderer.cpp`.
   - TS stesso lo definisce «un adattatore attorno a Metal» (`geo:.scratch/rtb-solution/research/photoreal-geology-plan.md:15`).
2. **Uber-shader a flag runtime** [V].
   - `RtbParams.pad` porta 27 bit di flag distinti (`p.pad&1` … `p.pad&268435456`).
   - A questi si aggiungono `mode`/`ac`/`overlay` e una lerp per pixel fra shading «legacy» e
     «modern» (`rtb_legacy.metal:8-14, 426`).
   - È l'opposto di F15.2: permutazioni da function constant, niente uber-shader.
3. **Layout GPU duplicati** fra C++ e MSL, senza header condiviso, controllati solo da asserzioni
   di dimensione sul lato C++. Si veda §1.3 [V].
4. **Unità non fisiche** [V].
   - `Lighting { sun, ambient, exposure }` usa moltiplicatori 0–4 (`TS:crates/editor-core/src/terrain_v1/lighting.rs:8-26`).
   - Lo shader usa radianza del sole `2.3*x` e ambiente emisferico.
   - Il tonemap ACES è applicato per frammento prima della miscela e prima della codifica sRGB.
   - È incompatibile con l'esposizione e l'atmosfera fisica di F14.
5. **Documenti non aggiornati.** `PHOSPHOR_INTEGRATION.md:7-11` dichiara SceneRenderer e
   SceneStore in uso, nessun shader copiato e nessun renderer alternativo. `PHOSPHOR_REUSE.md`
   descrive il riuso al commit dell'8 ottobre [V].
6. **Portabilità.** `PHOSPHOR_DEPS` assoluto, clone `--shared` legato all'object store locale e
   SDL3 da Homebrew [V] (`TS:docs/PHOSPHOR_INTEGRATION.md:91-93`).
7. **Lavoro sul motore avviato dal lato TS** [V].
   - Il branch `rea/host-embedding` modifica `src/app/engine.{h,cpp}` (328 righe),
     `cmake/App.cmake`, `display_output`, `atmosphere` e `camera`.
   - Aggiunge `src/host/phosphor_host.{h,cpp}`: create, frame, resize, camera, ora del giorno,
     `load_scene` di un glTF intero e cattura.
   - [I] L'API è generica e coerente con `PRODUCT_PLATFORM` (editor), ma tocca la radice di
     composizione su cui lavorerà F15. Va integrata col flusso di Phosphor (PR di fase, test,
     `visual_check`), non tenuta come fork sotto REA_RTB.
   - La decisione 4 del piano geo («A o B, e quindi se Phosphor si può modificare») risulta
     ancora aperta a `993e1e3` (`photoreal-geology-plan.md:192-197`) [V]. Non ho trovato una
     registrazione dell'approvazione.

### 2.3 Limiti del motore che TS ha scoperto (input per Phosphor, non per F15)

- **Carico RT al primo frame** [R/V].
  - Il caso: 10,4 milioni di triangoli, 4.904 BLAS e 1,5 GB di BLAS.
  - La build termina in 46 ms, poi «Frame 0 GPU feedback failure: MTL4CommandQueueErrorDomain
    error 1» ed `EXIT 1`.
  - 3,9 milioni di triangoli con 993 BLAS funzionano.
  - Ricevuta: `canyon/view-01/FAILED-RENDER.json`; log `frame-01.log:14877-14906`.
  - La causa non è isolata.
- **Non esiste rimozione o sostituzione di una singola mesh** con le relative strutture RT. Solo
  aggiunta o svuotamento globale [R] (`photoreal-geology-plan.md:172, 179`).
- **Il costo di DDGI è circa 45 dei 55 ms** a 1080p con 3,9 milioni di triangoli; senza GI si
  scende a 10,5 ms con ombre e GTAO [R] (`photoreal-geology-plan.md:159-167`).
- **«Bande concentriche nel cielo a piena risoluzione»** a `be886ca` [R] (`photoreal-geology-plan.md:100`).

## 3. Sotto-uso

| Capacità Phosphor (`c8ca350`) | Stato in TS | Peso per un editor di terreno e piste [I] |
|---|---|---|
| HDR, esposizione e tonemap in post; F14.1 atmosfera di Hillaire, sole fisico; F14.4 ciclo del giorno | Assenti: LDR, ACES per frammento, ambiente emisferico | **Massimo.** È il salto di coerenza visiva più grande a costo basso. Dà prospettiva aerea per le viste panoramiche e luce di sole e cielo coerente su roccia e acqua. L'host aggiunge già `set_time_of_day` |
| F10 ombre (CSM, RT del sole, contact) e F13.2 GTAO/RTAO | Una shadow map ortogonale 1024²; nessuna AO a schermo | **Alto.** I «contatti credibili» di pietre e detrito e la lettura di giunti e cenge, richiesti da `Terraformazione-…-handoff.md`, «Qualità attesa». A 257 m di terreno 1024² vale circa 0,25 m per texel; a 2049² non regge |
| F13.1 riflessi RT e SSR, F13.5 probe | Riflesso planare a ¼ di risoluzione e cubo da studio fisso | **Alto per l'acqua** (riva, vista radente). TS ha già provato senza successo dissolvenze e mezza risoluzione (handoff, punto 4) |
| F7 visibility buffer + risoluzione dei materiali con binning; F5 GPU scene e SceneStore; F6 meshlet e Hi-Z | Forward con una draw CPU per mesh, ri-upload su commit | **Alto** per scene geologiche da milioni di triangoli e per le istanze di pietre e vegetazione (`tr_update_stone_instances` si presta ai delta SceneStore). Manca però la sostituzione per mesh (§2.3) |
| MetalFX temporale | Non usato (MSAA proprio) | Medio. Margine per i 60 fps a 1420×790 o Retina |
| F12 DDGI | Non usato | Medio-basso durante il gesto (45 ms); utile per l'immagine a mano ferma con un preset «fermo». TS ha già misurato due livelli |
| F11 molte luci | Non usato | Basso (esterni diurni) |
| Report JSON, tempi per pass, `GPU allocations`, `image_diff`, `visual_check` | Ricevute proprie, ottime ma separate | Medio. Le qualifiche T27 (p95, memoria) guadagnerebbero tempi per pass confrontabili con quelli di Phosphor |

Giudizio [I]: per Terrain V1 contano nell'ordine
1. la catena HDR con l'atmosfera,
2. ombre e AO,
3. riflessi per l'acqua,
4. il percorso di visibilità e GPU-driven per la geometria densa.

La GI è un preset «fermo», non un requisito del gesto. Le lacune vere del motore sono di Phosphor
e non sono sotto-uso: terreno a strati (F15.4/F19), clipmap e virtual texture (F19), acqua (F20),
sostituzione di singole mesh, ospitabilità.

## 4. Lezioni sui materiali per F15

### 4.1 Buone, trasferibili (in termini neutri rispetto a Phosphor)

1. **Pesi locali sparsi, senza tetto di strati** [V].
   - **Cosa fa TS.**
     - Per ogni vertice tiene una lista ordinata (id, peso) con stride variabile, che cresce
       solo dove serve (`TS:native/renderer/terrain_local_weights.h:8-50`).
     - Nel frammento unisce le liste dei tre vertici con i baricentri e senza limite sul numero
       di materiali (`rtb_legacy.metal:303-310`).
     - La motivazione misurata: la scena densa aveva circa il 95% dei pesi a zero, per 28,6–31,0 MiB
       sul socket [R] (handoff, punto 3).
   - **Per F15.4.** Il tetto di 4 strati per pixel, con top-K, è compatibile. La
     **rappresentazione di ingresso**, però, dovrebbe accettare più di 4 materiali per
     superficie: id e peso per texel o per vertice, con pruning top-K e rinormalizzazione. Una
     sola maschera RGBA non basta: TS ha 44 materiali per progetto (`TS:docs/wiki/terrain-v1.md:25`).
2. **Height blend softmax esatto agli estremi, già in uso** [V].
   - **La formula di TS.**
     - Il peso diventa `w'_i = w_i·exp(clamp((h_i−h_max)/c, −20, 0)) / Σ`, con contrasto
       `c = 0,02 m` (`rtb_legacy.metal:357, 372`).
     - La variante «contatto naturale» ancora l'esponente alla media pesata, con `c = 0,003 m` e
       clamp ±4.
   - **Le regole di attivazione.**
     - Il blend parte solo se contribuiscono almeno 2 strati (`:325`).
     - Se un contributore non ha altezza dichiarata (versione < 2, `:323`), si ricade sui pesi
       semplici.
     - Gli strati a peso 0 sono saltati, quindi restano 0; uno strato a peso 1 resta 1.
   - **Per F15.4.** È la formula che `layered-materials.md` §6 propone come «[D], non
     pubblicata»: TS ne è un precedente funzionante. Ne vanno ereditate tre cose: la
     sottrazione del massimo per la stabilità numerica, il clamp dell'esponente, e il contrasto
     **in metri**. Questo richiede altezze per strato in unità dichiarate.
3. **Altezza, normali e AO con unità e convenzioni esplicite** [V].
   - L'altezza è `f32` in metri (`library.rs:73-74`, con controllo `|h| ≤ 1000`).
   - La wiki fissa la regola «height con unità e origine dichiarate, albedo con spazio colore
     dichiarato, normal con convenzione assi esplicita, AO distinta dall'altezza»
     (`TS:docs/wiki/terrain-v1.md:47`).
   - L'import preferisce `normalgl` e inverte il verde per `normaldx` (`TS:native/terrain-v1/MaterialImport.swift:42-43, 63`).
4. **Maschera di cavità deterministica in unità fisiche** [V].
   - La formula: `mask = amount·clamp((rim − h)/relief_m, 0, 1)`, dove `rim` è la media di
     quattro campioni d'altezza a raggio `feature_m` (`TS:crates/editor-core/src/terrain_v1/library.rs:1549, 1627`).
   - Il deposito usa la stessa concavità (`:1393-1402`).
   - È esattamente il generatore che serve alla fixture F15.4 per la «finish nelle cavità»
     (decisione 12), con parametri in metri e seme dichiarato.
5. **Partizione dell'unità con precedenza manuale** [V].
   - I pesi automatici sono scalati per `1 − Σ manuali` (`TS:crates/editor-core/src/terrain_v1/surface.rs:83-87, 116-118`).
   - Se la somma è nulla, il codice ripiega su un ruolo di riserva normalizzato invece di emettere
     pesi zero, così non compaiono pixel neri (`:386-404`).
   - I test asseriscono `Σw = 1 ± 1e-12` (`:656, 799, 1040`).
   - È la regola d'energia F15.4 «pesi normalizzati e niente pixel a peso nullo», con un
     test pronto da replicare.
6. **Mip corretti per i canali** [V].
   - Il colore è mediato in lineare, con decodifica e ricodifica sRGB.
   - Le normali sono rinormalizzate a ogni livello (`TS:native/terrain-v1/material_mips.hpp:9-19`).
   - L'altezza ha mip `f32` con controllo dei valori finiti (`:21-28`).
   - Phosphor oggi usa `generateMipmaps` per tutte le texture, normali comprese
     (`c8ca350:src/platform/metal/metal_texture_manager.cpp:96`) [V]. Le normali filtrate
     non sono rinormalizzate e la rugosità non si allarga (manca Toksvig). Si veda §5.
7. **Validazione dell'import e limiti di risorsa** [V].
   - Ogni riferimento a una mappa MaterialX deve restare nella directory dopo la risoluzione dei
     symlink: niente percorsi assoluti, `..` o `:` (`MaterialImport.swift:73-79`).
   - Limiti per file e immagine: ≤ 64 MiB, lato ≤ 8192, ≤ 32 MP (`:18-21`).
   - Per un gruppo d'import si sceglie un solo livello di risoluzione, 256 o 1024, entro un
     budget di 48 MiB (`:81-93`).
   - I range dei parametri sono validati (`library.rs:51-63`).
   - I campi sconosciuti sono rifiutati con `deny_unknown_fields`.
   - Il loader glTF di Phosphor concatena `baseDir_ + image.uri` senza alcun contenimento
     (`c8ca350:src/scene/gltf_loader.cpp:454-456`) [V].
8. **Supporto dichiarato per materiale** [V].
   - Gli input MaterialX fuori dal sottoinsieme finiscono in `source_unsupported`
     (`library.rs:167`). Il pannello segnala la «resa incompleta» e un caso non
     rappresentabile (Gold) viene respinto (`TS:docs/wiki/terrain-v1.md:65-67, 78`).
   - Conferma, lato consumatore, il «support report» della decisione 07: va emesso **per
     materiale**, non solo per file.
9. **Provenienza della dotazione** [V].
   - `TS:crates/editor-core/assets/nature/provenance.json` descrive 44 materiali: 4 AMD MIT,
     2 ambientCG CC0, 32 Poly Haven CC0 e 6 acque da USD OpenPBR NVIDIA MIT-0.
   - Per ogni voce registra URL, licenza, SHA-256 per file e archivio, scala fisica con la sua
     **fonte** («Poly Haven dimensions», «assumed 1m UV domain…», «Physical scale unknown»)
     e la **riduzione applicata** (es. `Roughness = low + red·(high−low)`, «graph nodes
     beyond base, normal, roughness not evaluated»).
   - È il formato giusto per il corpus F15.3/F15.4, in particolare la dichiarazione della scala
     e del suo grado di certezza.
10. **Identità dei bake** [V].
    - Le mappe importate sono confrontate per digest SHA-256 (`library.rs:83-87, 116-118`).
    - Lo snapshot di un bake è rifiutato se fingerprint o `render_version` non coincidono
      (`:1680-1691`).
    - Per F15.2 ne segue che il manifest delle permutazioni sia indicizzato per hash della
      descrizione più la versione dello schema.
11. **Prova di non regressione «a funzione spenta»** [R].
    - Ogni aggiunta, come finiture o studio, è accompagnata dal campione storico identico pixel
      per pixel (`TS:docs/wiki/terrain-v1.md:52, 67`).
    - È la stessa regola delle varianti Phosphor: conviene citarla come requisito di ogni ticket F15.

### 4.2 Workaround causati da lacune di Phosphor (requisiti per F15 o per fasi successive)

| Workaround in TS | Lacuna | Requisito |
|---|---|---|
| Acqua resa come piano alpha-blended: tinta pari a `transmission_color`, opacità di default 0,62, Fresnel di Schlick fisso a 0,02. Assorbimento con Beer-Lambert approssimato (`tint^(path·density)`) e lunghezza del cammino ricavata dalla profondità del fondale lungo la vista (`TS:crates/editor-core/src/terrain_v1/water.rs:33, 382-395`; `rtb_legacy.metal:127-138`). Il codice lo marca come «shortcut … until scene-colour transmission is available» (`:132, 137`) [V] | Trasmissione e volume | **F15.1**: `KHR_materials_transmission` + `KHR_materials_volume` (`attenuationColor` e `attenuationDistance` corrispondono esattamente alla «trasmissione a 1 m» di TS) + `KHR_materials_ior` 1,33. [I] Per i volumi d'acqua lo spessore va preso dalla geometria (RT o differenza di profondità), non solo da `thicknessFactor`: inserire un caso «fondale visibile in acqua bassa» nel testbench. La superficie d'acqua (onde, riva, sott'acqua) resta F20 |
| Sei profili d'acqua OpenPBR con profondità di trasmissione, IOR e scattering conservati nello ZIP ma non resi («not represented in runtime map reduction», `provenance.json` alla voce `water_clean`, riga 3202) [V] | Scattering di volume | F15.1/F16.2 per trasmissione e volume; lo scattering di volume dell'acqua va dichiarato fuori F15 o approssimato, e registrato nella tabella delle approssimazioni (decisione 06) |
| Open Chess Set: trasmissione (pedoni) e subsurface (re e regine) conservati e modificabili ma non resi; il pannello segnala la resa incompleta (`TS:docs/wiki/terrain-v1.md:78`; test `library.rs:1802`) [V] | Lobi avanzati | Già in F15.1 (decisioni 03 e 10) |
| «Finitura» e «Finitura nelle cavità»: lerp per texel di metallicità, rugosità, intensità della normale, emissione e unlit, con colore e altezza della base invariati (`library.rs:1546-1565, 1622-1637`; wiki `:63`) [V] | Manca un coat con parametri ottici propri | **F15.4 / clearcoat.** [I] Una finitura metallica («Chrome nelle cavità», provata in TS) eredita come F0 il colore del suolo: la finitura confonde una *coverage* orizzontale (deposito metallico, che è uno strato con colore proprio) con un *coat* verticale (vernice, film d'acqua). F15.4 deve distinguere i due casi e testarli entrambi |
| Bagnato: `albedo·(1−0,35w)` e `rough·(1−0,45w)`, escludendo metallo e unlit, con umidità a byte per vertice (`rtb_legacy.metal:399-400`; `TS:native/terrain-v1/terrain_wetness.hpp`) [V] | Superfici bagnate | [I] Buon caso per il coat di F15.4: un film d'acqua come coat dielettrico con IOR circa 1,33 e rugosità bassa, pesato dall'umidità, sopra la base scurita per porosità. Il clearcoat glTF fissa l'IOR del coat a 1,5, il coat OpenPBR ha `coat_ior`: va registrato come scelta. Il meteo resta F14.5 |
| Colore per banco «per faccia e non per pixel»: le pareti lontane perdono le bande (`photoreal-geology-plan.md:137`). Il campo geologico deve essere «uguale in Rust e su GPU», con contatti valutati per pixel (`:50`) [R] | Maschere procedurali sulla GPU | [I] Per F15.4 basta accettare maschere da texture **e** da attributi di vertice. Una maschera da funzione procedurale valutata nello shader è F19: va dichiarata fuori F15 |
| Scansioni di ghiaia e calcare con rugosità media circa 0,5 «brillavano di bianco» in controsole. È stato alzato il valore di rugosità negli asset (`photoreal-geology-plan.md:136, 148`) [R] | Da diagnosticare | [I] Caso per F15.3: un dielettrico scansionato, ruvido, in controsole radente, confrontato con un riferimento indipendente. Se il riferimento non brilla, il difetto è del renderer (occlusione speculare o orizzonte dalle normali, Fresnel o multi-scatter); se brilla anche lui, la colpa è della mappa. In glTF `roughnessFactor ≤ 1`, quindi una correzione dell'asset va ricotta nella texture: il fattore 0–2 di TS non è glTF (`library.rs:55`) |
| Mappe normalizzate a 256², con un livello a 1024 (`MaterialImport.swift:5-15`; wiki `:54, :65`). Patch vendor per catene BC autorate [V] | Texture compresse e memoria | [I] Per il corpus a 2K: valutare l'ingestione di mip compressi autorati (BC/ASTC/KTX2) come candidato separato, non in F15 se non serve alla misura |
| Ombre, riflesso, ambiente e tonemap fatti in proprio (§1.3) [V] | Nessuna, perché esistono in Phosphor | È sotto-uso, non un requisito |

### 4.3 Da non importare in Phosphor

- La lerp per pixel fra modelli di shading («modern» o «legacy», `unlit` per texel) e la
  texture «response» RGBA (metallicità, emissione/8, unlit, flag) [V] (`rtb_legacy.metal:396-426`).
  Unlit e i lobi sono proprietà di classe di feature, cioè permutazioni, non canali per texel.
- Le convenzioni che non sono glTF [V].
  - **Intensità della normale.** Nel bake è applicata nello spazio delle pendenze
    (`x = nx/nz·s`, `library.rs:1561, 1637`). Lo shader invece la applica su `xy`
    (`rtb_legacy.metal:406`). glTF definisce `normalize((n·2−1)·(s,s,1))`: TS è incoerente anche
    con se stesso.
  - **Packing.** AO va nell'alpha del base color e la rugosità nell'alpha della normale
    (`MaterialImport.swift:63`), invece dell'ORM glTF.
  - **Emissione e rugosità.** L'emissione è adimensionale 0–8 invece di `emissive_strength`;
    il fattore di rugosità arriva fino a 2.
- La miscela lineare delle normali mappate in world space e la triplanare con pesi `|n|^4`
  (`rtb_legacy.metal:314, 409-412`). F15.4 ha già scelto i surface gradient. [I] La triplanare
  serve davvero sulle pareti di un heightfield: il framework a surface gradient la supporta, ma
  va decisa come caso F15.4 oppure rinviata a F19 dichiarandolo.
- L'anti-tiling stocastico (tre patch su griglia triangolare, `pow(w,1.35)`) e il crossfade di
  scala fra 24 e 80 m con rumore di macro-colore a 27 e 72 m (`rtb_legacy.metal:245-258, 318, 329`)
  [V]. Le idee sono buone ma appartengono al terreno di F19, non al modello di materiale
  standard. Nota utile: TS campiona base, normale e risposta **con gli stessi offset**, così i
  canali restano coerenti.
- L'import via `CGContext` in `DeviceRGB` con alpha premoltiplicata anche per le mappe dati
  (`MaterialImport.swift:22-26`) [V]. [I] Rischia gestione colore e premoltiplicazione su
  normali, rugosità e altezza, e il ridimensionamento non rinormalizza le normali. In Phosphor i
  canali dati vanno letti grezzi e lineari.

## 5. Raccomandazioni

### Per Phosphor F15

1. **F15.4: rappresentazione.**
   - Tenere K ≤ 4 strati per pixel, ma accettare in ingresso più di 4 materiali, come liste
     (id, peso) per texel o per vertice con top-K e rinormalizzazione. Prova misurata da TS:
     circa il 95% di zeri nella forma densa.
   - Aggiungere i pesi da attributo di vertice accanto alla maschera RGBA.
   - Dichiarare fuori F15 le maschere procedurali calcolate nello shader (F19).
2. **F15.4: height blend.**
   - Adottare la formula softmax con la stabilizzazione e i gate di TS: sottrazione del massimo,
     clamp dell'esponente, blend solo con almeno 2 contributori, ricaduta se uno strato non ha
     altezza.
   - Esprimere il contrasto in metri e dare all'altezza di ogni strato unità e origine dichiarate.
   - Test: estremi 0 e 1 esatti, monotonia, assenza di NaN con tutti i pesi a zero.
3. **F15.4: coverage contro coat.**
   - Nella fixture della decisione 12 generare la maschera di cavità con la formula
     `rim − h` in metri.
   - Testare due casi distinti: un deposito metallico come **coverage**, con colore proprio, e
     un film o vernice come **coat** (clearcoat glTF o coat OpenPBR).
   - Negativo atteso: una «finitura» che cambia solo la metallicità mantenendo l'albedo della
     base deve risultare un errore di modellazione, non un risultato valido.
4. **F15.4: energia.** Pesi con somma 1 asserita a 1e-12 sulla CPU e niente pixel a peso
   nullo: usare la regola di ripiego di TS come caso di test.
5. **F15.1: trasmissione e volume.**
   - Aggiungere al testbench un caso «acqua bassa sopra un fondale».
   - Validare `attenuationColor` e `attenuationDistance` contro Beer-Lambert analitico, con
     spessore preso dalla geometria (RT) e non dal solo `thicknessFactor`.
   - Registrare nella tabella delle approssimazioni (decisione 06) lo scattering di volume
     rimandato.
6. **F15.3: casi del testbench suggeriti da TS.**
   - Dielettrico ruvido scansionato in controsole radente, confrontato con un riferimento
     indipendente: decide se il «bianco» visto da TS è un difetto del renderer o della mappa.
   - Normal map con intensità diversa da 1, secondo la formula glTF.
   - Pipeline dei mip: rinormalizzazione delle normali e Toksvig/varianza per la rugosità.
     Oggi Phosphor usa `generateMipmaps` grezzo.
   - Prova «a funzione spenta», identica pixel per pixel, in ogni ticket.
7. **Robustezza dell'import (loader glTF, `extensionsRequired`).**
   - Contenere gli URI nella directory dell'asset (niente assoluti, `..` o fughe via symlink).
   - Porre limiti di byte e lato per le immagini.
   - Leggere le mappe dati come lineari e non premoltiplicate.
   - Emettere il support report **per materiale**, con la lista degli input ignorati.
8. **Corpus.** Usare per F15.3/F15.4 lo schema di provenienza della dotazione TS: URL, licenza,
   SHA-256 per file e archivio, scala fisica con fonte e grado di certezza, riduzione applicata.
9. **Fuori F15, da tracciare.**
   - Verificare la regola `destinationBytesPerImage = 0` per le copie 2D (patch TS, 9 file
     upstream ancora col valore pieno).
   - Upload RGBA8 a righe entro lo staging ring.
   - Ingestione di mip compressi autorati.
   - Fallimento al primo frame con circa 4.900 BLAS e 1,5 GB (§2.3): budget e frazionamento
     della build AS.
   - API per rimuovere o sostituire una mesh insieme al suo BLAS.
   - Bande del cielo.
   - L'host embedding (`rea/host-embedding`) va portato nel flusso di Phosphor, con PR, test e
     `visual_check`, prima che diverga da `engine.cpp` durante F15.

### Per Track Studio, da fare quando porterà gli sviluppi

1. **Una sola catena di resa.**
   - Completare l'opzione A, cioè l'`Engine` ospitato tramite l'API host, e abbandonare
     `rtb_legacy.metal` per il terreno V1.
   - Il percorso RTB legacy resta solo per la compatibilità RTB/AC, se ancora richiesta.
   - Rimuovere le ombre, il riflesso planare, l'ambiente da studio e il tonemap propri.
2. **Riallineare la copia di Phosphor.**
   - Eliminare `vendor/phosphor@a83fcf7` patchato e consumare un commit fissato di Phosphor main.
   - Proporre upstream, come PR verso Phosphor, le patch generali: copie 2D, upload a righe,
     BC. Lasciare cadere quelle RTB-specifiche (blend legacy, MSAA, A2C) insieme al percorso
     legacy.
3. **Materiali nel documento in termini standard.**
   - Packing ORM glTF e intensità della normale secondo glTF.
   - `roughnessFactor ≤ 1`, e correzioni della rugosità ricotte nelle texture.
   - `emissive_strength` per l'emissione.
   - Unlit come proprietà del materiale.
   - `KHR_materials_transmission`, `volume` e `ior` per i profili d'acqua.
   - Clearcoat, o coat OpenPBR, per finiture e bagnato.
   - Le finiture metalliche diventano strati di coverage con colore proprio.
   - Il compositore a blocchi resta uno strumento d'autore che cuoce in mappe standard, come fa
     già il nucleo geologico.
4. **Luce in unità fisiche.** Sole e cielo da F14, ora del giorno e preset «gesto» e «fermo»
   (senza o con GI), al posto dei moltiplicatori 0–4.
5. **Import.** Leggere le mappe dati senza gestione del colore né premoltiplicazione;
   rinormalizzare le normali ridotte; abbandonare il tetto 256/1024 a favore delle risoluzioni
   native con mip compressi, quando Phosphor li supporterà.
6. **Strati del terreno.** Mappare i pesi locali sparsi e la precedenza manuale sul
   formato di ingresso di F15.4 (top-K), conservando gli id e la partizione dell'unità.
   Spostare anti-tiling e macro-variazione su F19 quando arriverà.
7. **Documenti.** Aggiornare `PHOSPHOR_INTEGRATION.md` e `PHOSPHOR_REUSE.md`, oggi fermi all'8 e
   al 22 ottobre e in contrasto col codice. Registrare nel piano la decisione 4 (A o B, modifica
   di Phosphor), prima di proseguire `rea/host-embedding`.

## Metodo e limiti

- **Comandi usati**, tutti in sola lettura:
  - `git --no-optional-locks log`, `status`, `diff`, `show`, `grep` sul repository TS e sui
    worktree, sulla copia vendor e su Phosphor;
  - `ls`, `stat`, `jq` e `rg` sui file elencati.
- Le righe citate di `track_renderer.cpp` e `rtb_legacy.metal` sono molto lunghe. Dove do un
  intervallo, il costrutto citato sta su quelle righe.
- **Non verificati.**
  - La regola Apple su `destinationBytesPerImage`.
  - La causa del fallimento con 4.904 BLAS.
  - Le misure TS: p95, RSS, costo della GI, quota di pesi a zero.
  - L'approvazione dell'opzione A.
  - Lo stato finale del worktree `phosphor-host`, che era in modifica durante la lettura.
