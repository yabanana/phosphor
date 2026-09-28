# Phosphor — Roadmap implementativa 2026–2027 (e oltre)

Piano operativo per portare Phosphor dalle fondamenta Metal 4 (F0, completata)
a un **engine AAA nativo per Apple Silicon, bleeding edge e ottimizzato fino
all'ultimo byte di banda**, pensato per giochi che girano solo su piattaforme Apple.

Riferimenti:
- motivazioni tecniche → [`reports/Engine AAA nativo per Apple Silicon.md`](../reports/Engine%20AAA%20nativo%20per%20Apple%20Silicon.md) e `research_notes/`
- regole per le sessioni AI → [`CLAUDE.md`](../CLAUDE.md)

---

## Filosofia

1. **Nativo, non tradotto.** Ogni sistema nasce attorno a TBDR, memoria
   unificata, Metal 4 e acceleratori neurali. Nessuna astrazione "da PC".
2. **Misurare prima di ottimizzare, ottimizzare sempre.** Il profiler arriva
   in F4, prima di qualsiasi tecnica pesante; ogni fase chiude con numeri.
3. **Meno raggi, più ricostruzione.** Budget di ray tracing bassi, compensati
   da cache di radianza, riuso temporale e ricostruzione ML: è la direzione
   dell'hardware Apple.
4. **Tutto guidato dalla GPU.** La CPU descrive il mondo; la GPU decide cosa
   disegnare, a che LOD e con che qualità.
5. **Il pavimento conta quanto il vertice.** Apple9 (M3) è il minimo; ogni
   funzione ha un percorso T0 misurato su hardware reale.
6. **Spike di ricerca espliciti.** Le idee non provate diventano un prototipo
   a tempo con criterio "adotta / rimanda / scarta", mai un ramo infinito.

---

## Come usare questo documento

- Fasi in **ordine di dipendenza**; le fasi marcate *parallela* possono
  procedere accanto alla linea principale perché toccano sottosistemi separati.
- Etichette di priorità:
  - **[CORE]** indispensabile per avere un engine funzionante
  - **[AAA]** necessaria per qualità da gioco AAA
  - **[EDGE]** bleeding edge / ricerca: alto rischio, alto ritorno, sempre con fallback
- Ogni task ha un ID (`F7.3`) da citare in commit e PR; la casella si spunta
  solo quando il task è **verificato sul dispositivo**, non solo scritto.
- **Definizione di "fase chiusa"**: criteri di uscita misurati e annotati in
  `docs/perf-log.md`, `ctest` e CI verdi, zero errori di validazione Metal sui
  testbench, test visivi aggiornati, README aggiornato.
- I budget in millisecondi sono **ipotesi di progetto**, da ricalibrare con le
  misure reali alla fine di F8.

### Tier hardware

| Tier | Chip | Ruolo |
|---|---|---|
| **T0 Base** | M3/M4/M5 base, A18 Pro, iPad M3+ | Pavimento: tutto deve girare qui. Serve un **dispositivo T0 fisico** |
| **T1 Pro** | M3/M4/M5 Pro | Qualità alta |
| **T2 Max** | M3/M4/M5 Max | Qualità ultra; macchina di sviluppo (M5 Max 128 GB) |
| **T3 Neurale** | M5 Pro/Max/Ultra (Apple10) | Funzioni neurali, path tracing |

### Budget frame di riferimento (ipotesi, 60 fps)

| Pass | T0 1080p (upscaled) | T2 1440p→4K |
|---|---|---|
| GPU scene + culling + visibility buffer | 2,0 ms | 1,5 ms |
| Material resolve | 2,0 ms | 1,5 ms |
| Ombre | 1,5 ms | 1,5 ms |
| Luci dirette (ReSTIR DI) | 1,0 ms | 2,0 ms |
| GI + riflessioni | 2,0 ms | 3,5 ms |
| Atmosfera, volumetrici, acqua | 1,0 ms | 1,5 ms |
| Trasparenze, particelle | 1,0 ms | 1,0 ms |
| Post + MetalFX | 2,0 ms | 2,0 ms |
| Margine | 0,5–3,5 ms | 1,5 ms |
| **Totale** | **≤ 16,6 ms** | **≤ 16,6 ms** |

---

## Binario trasversale: ultra-ottimizzazione (regole O1–O12)

Non è una fase: sono regole che **ogni** fase deve rispettare e verificare.

| Regola | Contenuto | Come si verifica |
|---|---|---|
| **O1 Byte per pixel** | Ogni pass dichiara quanta banda DRAM legge/scrive; il totale per frame sta nel budget del tier (≈ 10 GB/frame a 60 fps su M5 Max, ¼ su T0) | Contatori di banda in Xcode, colonna "bytes" nel profiler |
| **O2 Tile memory first** | Intermedi `memoryless`, store `.dontCare`, pass fusi; fragment barrier vietate | Nessun load/store inutile nel report del render graph |
| **O3 Half ovunque possibile** | `half` per colore, normali, BRDF; `float` solo dove la precisione lo richiede (posizioni, profondità) | Revisione shader + test visivi invariati |
| **O4 Registri sotto controllo** | Niente uber-shader, niente array sullo stack; registri vivi per riga controllati in Xcode | Occupancy ≥ soglia per ogni shader "caldo" |
| **O5 SIMD-group** | Riduzioni, prefix sum e compattazioni con `simd_*`/`quad_*`, non con atomici in threadgroup memory | Microbenchmark in `bench/` |
| **O6 Compressione GPU** | Texture popolate via blit (mai `replaceRegion`), formati compressibili, private storage | Contatori di compressione (Xcode 26.4+) |
| **O7 Zero allocazioni per frame** | CPU: arena per frame; GPU: anelli e heap; nessun `new` nel ciclo caldo | Instruments Allocations piatto durante il gameplay |
| **O8 Zero draw call dalla CPU** | Dopo F5 la CPU non codifica draw per oggetto: ICB e dispatch indiretti costruiti dalla GPU | Conteggio comandi CPU per frame costante al variare della scena |
| **O9 Stutter zero** | Nessuna compilazione, allocazione di heap o I/O bloccante nel frame | Frame pacing a 2 bucket nel Performance HUD |
| **O10 Energia** | Modalità "efficienza" per batteria: fps cap, tier ridotto, E-core per i job non critici | Watt misurati con `powermetrics` |
| **O11 Specializzazione** | Function constant per eliminare rami statici; varianti generate, non `if` a runtime | Numero di varianti e tempo di compilazione tracciati |
| **O12 Misura su T0** | Ogni ottimizzazione verificata sul pavimento, non solo sull'M5 Max | Colonna T0 compilata in `docs/perf-log.md` |

---

## Panoramica

| # | Fase | Tag | Era |
|---|---|---|---|
| F0 | Fondamenta Metal 4 ✅ | CORE | I · Fondamenta |
| F1 | Memoria, heap e residency | CORE | I |
| F2 | Render graph e sincronizzazione automatica | CORE | I |
| F3 | Pipeline: compilazione asincrona e archivi AOT | CORE | I |
| F4 | Osservabilità: profiler, contatori, cattura, perf log | CORE | I |
| F5 | GPU scene persistente e submission guidata dalla GPU | CORE | II · Geometria |
| F6 | Mesh shader e culling a due fasi | CORE | II |
| F7 | Visibility buffer e shading ibrido TBDR | CORE | II |
| F8 | HDR, EDR, esposizione e MetalFX temporal | CORE | II |
| F9 | Infrastruttura ray tracing | CORE | III · Luce |
| F10 | Ombre ibride (CSM, RT, contact) | AAA | III |
| F11 | Migliaia di luci: ReSTIR DI, luci ad area ed emissive | AAA | III |
| F12 | Illuminazione globale: DDGI → radiance cache | AAA | III |
| F13 | Riflessioni, AO e denoiser MetalFX | AAA | III |
| F14 | Cielo, atmosfera, nuvole volumetriche, meteo | AAA | III |
| F15 | Materiali avanzati e generazione delle varianti shader | AAA | IV · Mondo |
| F16 | Trasparenze on-tile, particelle GPU, VFX | AAA | IV |
| F17 | Post-processing cinematografico | AAA | IV |
| F18 | Geometria virtualizzata (cluster LOD + raster software) | AAA | IV |
| F19 | Terreno, vegetazione e mondo procedurale GPU | AAA | IV |
| F20 | Acqua, oceano e fluidi | EDGE | IV |
| F21 | Asset pipeline e cooker | CORE | IV |
| F22 | Streaming: MTLIO, sparse, virtual texturing | CORE | IV |
| F23 | CPU ultra-ottimizzata: job system, QoS, NEON, SME | CORE | V · Simulazione (parallela) |
| F24 | Fisica, cloth e distruzione (Jolt + GPU) | AAA | V (parallela) |
| F25 | Animazione, skinning GPU, capelli a filamenti | AAA | V (parallela) |
| F26 | Audio spaziale e acustica in ray tracing | EDGE | V (parallela) |
| F27 | Runtime di gioco: ECS, input, scripting, rete | CORE | V (parallela) |
| — | *Checkpoint WWDC 2027* | — | — |
| F28 | Scalabilità: preset, DRS, rate map, frame interpolation, termica | CORE | VI · Frontiera |
| F29 | Autotuning per dispositivo | EDGE | VI |
| F30 | Rendering neurale I: tensori, NTC, radiance cache neurale | EDGE | VI |
| F31 | Rendering neurale II: reti addestrate in casa con MLX | EDGE | VI |
| F32 | Path tracing in tempo reale e riferimento | EDGE | VI |
| F33 | Gaussian splatting ibrido | EDGE | VI |
| F34 | Editor e strumenti di contenuto | CORE | VII · Prodotto |
| F35 | QA automatizzata: test visivi, perf bot, replay | CORE | VII |
| F36 | Piattaforme: iPad, iPhone Pro, visionOS | AAA | VII |
| F37 | Vertical slice | CORE | VII |
| F38 | Distribuzione e live ops | CORE | VII |

### Calendario indicativo e percorso critico

| Periodo | Linea principale (rendering) | Linea parallela (runtime/strumenti) |
|---|---|---|
| ott–nov 2026 | F1, F2, F3, F4 | — |
| dic 2026 – feb 2027 | F5, F6, F7, F8 | F21 (cooker minimo) |
| mar – mag 2027 | F9, F10, F11, F12, F13 | F23, F27 |
| giu 2027 | **Checkpoint WWDC 2027** | F24 |
| giu – set 2027 | F14, F15, F16, F17, F18, F22 | F25, F34 |
| ott – dic 2027 | F28, F30, F32 (T3), F37 | F35 |
| 2028 | F19, F20, F26, F29, F31, F33, F36, F38 | — |

**Percorso critico per "SOTA entro fine 2027"**: F1→F8 (fondamenta e
geometria GPU-driven), F9→F13 (luce in ray tracing con radiance cache e
MetalFX), F18 + F22 (geometria virtualizzata in streaming), F28 (scalabilità),
F30 + F32 (neurale e path tracing sui T3), F37 (vertical slice).
Le fasi [EDGE] fuori dal percorso critico sono il vantaggio competitivo del
2028: si iniziano come spike appena i prerequisiti esistono.

Onestamente: l'intero piano è molto ambizioso per un solo sviluppatore
affiancato da Claude Code. Le date vanno rivalutate alla fine di F8 con le
prime misure di velocità reali.

---

# Era I — Fondamenta

## F0 — Fondamenta Metal 4 ✅ [CORE]

- [x] F0.1 Codice Vulkan archiviato in `legacy/vulkan/`
- [x] F0.2 `phosphor_core` portabile + test doctest
- [x] F0.3 `MetalContext`: coda MTL4, 3 frame in volo, `MTLSharedEvent`, residency set, rilascio differito
- [x] F0.4 Texture bindless (tabella di `MTLResourceID`), mipmap via blit
- [x] F0.5 Pass forward PBR, depth memoryless, ImGui (coda Metal 3)
- [x] F0.6 CI Linux (test + `metal_syntax_check`) e macOS (app + shader)
- [ ] F0.7 Zero errori con `MTL_DEBUG_LAYER=1` sui testbench 1–7
- [ ] F0.8 Baseline FPS/GPU ms di ogni testbench su M5 Max (e T0) in `docs/perf-log.md`

## F1 — Memoria, heap e residency [CORE]

**Obiettivo**: controllo totale della memoria GPU, zero allocazioni nel frame (O7).

- [ ] F1.1 Heap `MTLHeapTypePlacement` per risorse transitorie; sub-allocatore TLSF per buffer persistenti
- [ ] F1.2 Anelli di upload per frame con suballocazione lineare (sostituiscono `UploadBuffer` di F0)
- [ ] F1.3 Residency: set statico + set per-streaming, aggiornamento incrementale e batch di `commit()`
- [ ] F1.4 Budget di memoria per tier, statistiche live (per heap, per categoria) in ImGui
- [ ] F1.5 Gestione della pressione di memoria di sistema (notifiche macOS → eviction)
- [ ] F1.6 Test di stress: 10.000 creazioni/distruzioni di risorse senza crescita di memoria

**Uscita**: Instruments Allocations piatto in gameplay; nessuna allocazione Metal nel frame.

## F2 — Render graph e sincronizzazione automatica [CORE]

**Obiettivo**: aggiungere decine di pass senza bug di sincronizzazione (Metal 4 non traccia gli hazard).

- [ ] F2.1 Render graph portabile (in `phosphor_core`, testato su Linux): pass, risorse virtuali, read/write per stage, culling dei pass inutili, ordinamento topologico
- [ ] F2.2 Aliasing delle risorse transitorie sugli heap di F1 (analisi degli intervalli di vita)
- [ ] F2.3 Barrier builder Metal 4: coppie di stage con `barrierAfterEncoderStages` / `barrierAfterQueueStages`; divieto di fragment/tile nel lato "after"
- [ ] F2.4 Fusione automatica dei pass TBDR compatibili; load/store e `memoryless` dedotti dal grafo (O2)
- [ ] F2.5 Encoding parallelo: più thread codificano pass diversi (allocatori per thread), render pass sospesi/ripresi tra command buffer
- [ ] F2.6 Async compute: pass compute indipendenti su una seconda coda MTL4 con eventi
- [ ] F2.7 Migrazione del forward di F0 sul grafo; backface culling verificato e riattivato
- [ ] F2.8 Dump del grafo (Graphviz) con banda stimata per risorsa (O1)

**Uscita**: stessa immagine di F0; unit test su ordinamento, aliasing e barriere attese.

## F3 — Pipeline: compilazione asincrona e archivi AOT [CORE]

**Obiettivo**: stutter di compilazione zero (O9) fin dall'inizio, non alla fine.

- [ ] F3.1 `MTL4Compiler` su thread dedicati con QoS inferiore al render thread, `maximumConcurrentCompilationTaskCount` worker
- [ ] F3.2 Pipeline "flessibili" subito, specializzate in background; chiave di cache hashata
- [ ] F3.3 Function constant per le varianti (O11); tabella delle varianti generata
- [ ] F3.4 Raccolta automatica dei descrittori in `.mtl4-json` durante i test, `metal-tt` in CI → `MTL4Archive`
- [ ] F3.5 Fallback obbligatorio su miss dell'archivio (OS o GPU diversi), con tasso di miss registrato
- [ ] F3.6 Hot reload degli shader in Debug (ricompilazione del `.metallib` e sostituzione delle pipeline)

**Uscita**: nessun hitch al cambio testbench; avvio a freddo con archivio senza compilazioni.

## F4 — Osservabilità [CORE]

**Obiettivo**: ogni decisione di ottimizzazione si basa su numeri.

- [ ] F4.1 `MTL4CounterHeap` con timestamp per pass, media/max su 60 frame, pannello Pass Timings
- [ ] F4.2 Tracy: zone CPU per sistema, zone GPU per pass, allocazioni
- [ ] F4.3 Cattura GPU da tasto (`MTLCaptureManager`), cattura automatica sui frame sopra soglia
- [ ] F4.4 Contatori hardware (occupancy, banda, compressione, stalli) esportati per pass quando disponibili
- [ ] F4.5 `docs/perf-log.md` + script che registra le misure dei testbench per commit
- [ ] F4.6 Modalità benchmark da riga di comando (`--bench N --frames 600 --report out.json`)
- [ ] F4.7 Sovrapposizioni di debug generiche (heatmap di costo per tile, overdraw, contatori)

**Uscita**: ogni pass ha tempo GPU e banda visibili; benchmark ripetibili da CLI.

---

# Era II — Geometria GPU-driven

## F5 — GPU scene persistente e submission guidata dalla GPU [CORE]

**Obiettivo**: la CPU non codifica più draw per oggetto (O8).

- [ ] F5.1 GPU scene persistente: istanze, materiali, mesh in buffer GPU con **aggiornamenti delta** (solo gli oggetti cambiati), memoria unificata con scrittura diretta
- [ ] F5.2 Gerarchia di transform aggiornata in compute
- [ ] F5.3 **Indirect command buffer costruiti dalla GPU**: un dispatch di culling scrive i comandi di draw
- [ ] F5.4 Catena di dispatch indiretti come sostituto dei work graph (Metal non li ha): code GPU a produttore/consumatore tra pass
- [ ] F5.5 Instance culling in compute (frustum + distanza + dimensione su schermo)
- [ ] F5.6 Testbench "1M istanze" dinamiche

**Uscita**: Stress Test con conteggio di comandi CPU costante; 1M istanze in movimento a 60 fps su T2.

## F6 — Mesh shader e culling a due fasi [CORE]

- [ ] F6.1 Meshlet ritarati per Apple: confronto 64/96/128 triangoli, `meshopt_buildMeshletsSpatial`, misure su M3 e M5
- [ ] F6.2 Object shader: culling per meshlet (frustum, cono di normali, area su schermo, occlusione Hi-Z)
- [ ] F6.3 Mesh shader con output minimi dichiarati e primitive scartate omesse (guida Apple)
- [ ] F6.4 Piramide Hi-Z in compute con riduzione SIMD-group (O5); percorso sampler min/max su Apple10
- [ ] F6.5 Culling a due fasi: frame precedente → nuovo test dei rifiutati sulla profondità corrente
- [ ] F6.6 Compattazione dei meshlet visibili con prefix sum SIMD-group, draw mesh indirette (Apple9)
- [ ] F6.7 Debug view: meshlet colorati, rifiutati per fase, Hi-Z

**Uscita**: Culling Viz a 60 fps su T0; conteggio dei cluster coerente con la visibilità.

## F7 — Visibility buffer e shading ibrido TBDR [CORE]

**Obiettivo**: il cuore del renderer, progettato per la tile memory.

- [ ] F7.1 Visibility buffer R32Uint con codifica **25 bit cluster + 7 bit triangolo**
- [ ] F7.2 Material resolve in compute: baricentriche analitiche, derivate per il mip, fetch attributi dalla GPU scene
- [ ] F7.3 **Binning per materiale** in compute (tile classification): uno shader specializzato per classe di materiale invece di un uber-shader (O4, O11)
- [ ] F7.4 **Spike [EDGE]**: deferred "on-tile" con tile shader e imageblock contro V-buffer + resolve in compute; misurare banda e tempo su T0 e T2; adottare un ibrido per tier se conviene
- [ ] F7.5 **Spike [EDGE]**: shading a frequenza variabile software nel resolve (2×2 dove il contrasto è basso), guidato dalla luminanza del frame precedente
- [ ] F7.6 Canali per il denoiser MetalFX emessi da subito: normali con segno, albedo diffusa, albedo speculare con Fresnel, roughness, motion vector, depth
- [ ] F7.7 Alpha test in fase raster separata; ordine opachi → alpha-test → traslucidi

**Uscita**: V-buffer + resolve più veloce del forward di F0 su Sponza; confronto on-tile documentato.

## F8 — HDR, EDR, esposizione e MetalFX temporal [CORE]

- [ ] F8.1 Target RGBA16F nella tile, istogramma di luminanza in compute con SIMD-group, esposizione automatica
- [ ] F8.2 Tonemapping configurabile (AgX, ACES, curva custom) e uscita **EDR** su display XDR con calibrazione della luminanza massima
- [ ] F8.3 Jitter Halton sub-pixel, motion vector per oggetti e camera
- [ ] F8.4 MetalFX temporal upscaler con risoluzione dinamica e reactive mask (scala max 2x)
- [ ] F8.5 Sharpening adattivo e mip bias corretto per la risoluzione di render

**Uscita**: stabilità temporale senza ghosting visibile sui testbench in movimento; misure base per ricalibrare i budget.

---

# Era III — Luce

## F9 — Infrastruttura ray tracing [CORE]

- [ ] F9.1 BLAS per mesh con build, compaction e refit nel `MTL4ComputeCommandEncoder`
- [ ] F9.2 TLAS per frame con istanze filtrate dalla GPU scene; build guidate da indirizzo (Apple9)
- [ ] F9.3 **Geometria proxy per l'RT** (LOD semplificati) separata dal raster: niente RT contro geometria a piena densità
- [ ] F9.4 Libreria di traversal con `intersector` (non `intersection_query`, che disabilita il reorder hardware)
- [ ] F9.5 Intersection function buffer per alpha test nell'RT; su M5 indicizzazione hardware
- [ ] F9.6 **Spike [EDGE]**: ordinamento software dei raggi per coerenza (binning per direzione/materiale) come sostituto dello shader execution reordering assente in Metal

**Uscita**: TLAS da 100K istanze aggiornato entro 0,5 ms su T2; costo per raggio misurato per tier.

## F10 — Ombre ibride [AAA]

- [ ] F10.1 CSM a 4 cascate renderizzate con mesh shader, stabilizzate, PCSS (fallback T0)
- [ ] F10.2 Ombre RT del sole con penombra fisica (1 raggio + denoise)
- [ ] F10.3 Ombre RT per luci locali integrate con ReSTIR
- [ ] F10.4 Contact shadow in screen space per i dettagli fini
- [ ] F10.5 Cache delle ombre statiche (aggiornamento solo delle regioni cambiate)

## F11 — Migliaia di luci [AAA]

- [ ] F11.1 ReSTIR DI (port da `legacy/vulkan/shaders/lighting/`): candidati, riuso temporale e spaziale, shading
- [ ] F11.2 Luci ad area (rettangoli, dischi, tubi) e mesh emissive campionate
- [ ] F11.3 Light BVH / alias table per il campionamento dei candidati
- [ ] F11.4 Rumore blu spazio-temporale (STBN) per tutte le decisioni stocastiche
- [ ] F11.5 Percorso T0: cluster di luci 3D + ReSTIR ridotto

**Uscita**: Many Lights (1.024 luci) stabile dopo il denoiser su T1; T0 ≥ 30 fps.

## F12 — Illuminazione globale [AAA]

- [ ] F12.1 DDGI in compute (port da `legacy/vulkan/shaders/gi/` con `intersector`): trace, atlanti, classificazione e riallocazione delle sonde
- [ ] F12.2 **Radiance cache a hash spaziale** (stile SHaRC) come cache unica per GI e riflessioni ruvide
- [ ] F12.3 ReSTIR GI su T2+ sopra la radiance cache
- [ ] F12.4 Superfici emissive che illuminano la scena tramite la cache
- [ ] F12.5 Confronto con riferimento (F32.1 quando disponibile, prima renderer offline esterno)

**Uscita**: Cornell Box fisicamente plausibile; Sponza con sole dinamico senza leak; budget GI rispettato.

## F13 — Riflessioni, AO e denoiser [AAA]

- [ ] F13.1 Riflessioni RT per roughness bassa, fallback su radiance cache e SSR (T0)
- [ ] F13.2 RTAO (T1+) / GTAO (T0)
- [ ] F13.3 **MetalFX denoised upscaler** (Apple9) collegato ai canali di F7.6
- [ ] F13.4 Denoiser custom leggero per T0 (temporal + atrous) quando il denoiser MetalFX non è usato
- [ ] F13.5 Probe di riflessione statiche filtrate per T0

## F14 — Cielo, atmosfera, nuvole, meteo [AAA]

- [ ] F14.1 Atmosfera fisica (Hillaire): LUT di trasmittanza, multi-scattering, sky-view
- [ ] F14.2 Nebbia volumetrica froxel con scattering di luci e ombre, integrata con la GI
- [ ] F14.3 **Nuvole volumetriche** raymarched (approccio stile Nubis) con ricostruzione temporale a bassa risoluzione
- [ ] F14.4 Ciclo giorno/notte, luna e stelle
- [ ] F14.5 **[EDGE]** Meteo dinamico: pioggia (particelle + superfici bagnate), neve con accumulo, fulmini come luci ReSTIR

---

# Era IV — Mondo

## F15 — Materiali avanzati e varianti shader [AAA]

- [ ] F15.1 Modello esteso: clearcoat, sheen, anisotropia, trasmissione, iridescenza, subsurface (estensioni glTF `KHR_materials_*`)
- [ ] F15.2 Descrizione dati dei materiali → generatore di permutazioni MSL offline (O11), niente uber-shader
- [ ] F15.3 Testbench materiali confrontato con i modelli di riferimento Khronos
- [ ] F15.4 Layered material (blend di strati) per terreni e superfici complesse

## F16 — Trasparenze on-tile, particelle, VFX [AAA]

- [ ] F16.1 **OIT on-tile** con imageblock + raster order group (strumenti TBDR che il desktop non ha)
- [ ] F16.2 Vetro e liquidi con rifrazione approssimata e assorbimento
- [ ] F16.3 Particelle GPU: simulazione in compute, rendering con mesh shader, collisione con la depth, luce dalla radiance cache
- [ ] F16.4 Decal deferred sul visibility buffer
- [ ] F16.5 **[EDGE]** Fumo e fuoco volumetrici simulati (griglia sparse in compute)

## F17 — Post-processing cinematografico [AAA]

- [ ] F17.1 Bloom fisico basato sull'energia, lens dirt
- [ ] F17.2 Depth of field con bokeh fisico
- [ ] F17.3 Motion blur per oggetto e camera
- [ ] F17.4 Color grading con LUT 3D, film grain, aberrazione cromatica, vignetta
- [ ] F17.5 Pass fusi in un unico compute dove possibile (O1)

**Uscita**: post completo entro 2 ms su T0 incluso MetalFX.

## F18 — Geometria virtualizzata [AAA]

- [ ] F18.1 DAG di cluster LOD offline con `clusterlod.h` (meshoptimizer ≥ 1.3)
- [ ] F18.2 Selezione LOD per livelli con dispatch indiretti (**no persistent thread**: Metal non garantisce il forward progress)
- [ ] F18.3 **Raster software** dei micro-triangoli in compute con atomici a 64 bit (Apple9), fuso con il raster hardware nel V-buffer; misurare l'impatto sull'HSR del TBDR
- [ ] F18.4 Compressione dei cluster (codec meshlet di meshoptimizer)
- [ ] F18.5 Ombre compatibili (CSM e RT contro proxy)
- [ ] F18.6 Testbench "Mega Geometry" (≥ 100M triangoli)

**Uscita**: Mega Geometry 60 fps su T2, ≥ 30 su T0; niente popping (errore < 1 pixel).

## F19 — Terreno, vegetazione, mondo procedurale GPU [AAA]

- [ ] F19.1 Terreno a clipmap con heightfield virtuale in streaming
- [ ] F19.2 Texturing del terreno con virtual texture e materiali a strati
- [ ] F19.3 Piazzamento procedurale di vegetazione e rocce in compute (regole + densità)
- [ ] F19.4 Vegetazione animata dal vento, impostor ottaedrici per la distanza
- [ ] F19.5 **[EDGE]** Generazione procedurale di interi biomi sulla GPU a runtime, deterministica per seed

## F20 — Acqua, oceano e fluidi [EDGE]

- [ ] F20.1 Oceano FFT in compute (cascate multiple), schiuma, interazione con gli oggetti
- [ ] F20.2 Caustiche in ray tracing
- [ ] F20.3 Rendering subacqueo con volumetrici
- [ ] F20.4 Fluidi locali (FLIP/SPH in compute) per fiumi, cascate, schizzi

## F21 — Asset pipeline e cooker [CORE]

- [ ] F21.1 `tools/cooker`: glTF e **OpenUSD** → formato Phosphor, build incrementale per hash
- [ ] F21.2 Texture ASTC (astcenc) e BC7, mip precalcolati, supercompressione per la distribuzione
- [ ] F21.3 Mesh: cluster, DAG LOD, proxy RT e meshlet compressi precalcolati
- [ ] F21.4 Formato pacchetto mappabile in memoria, allineato ai codec MTLIO
- [ ] F21.5 Cooker parallelo sui core del Mac (job system di F23)

## F22 — Streaming [CORE]

- [ ] F22.1 **Prototipo** MTLIO + risorse placement sparse per chiarire la sincronizzazione con le code MTL4 (lacuna aperta nel report)
- [ ] F22.2 Streaming a priorità dalla camera, budget per tier, eviction LRU
- [ ] F22.3 Virtual texturing con texture sparse (pagine 16/64 KB) e feedback buffer
- [ ] F22.4 Streaming delle pagine di cluster di F18
- [ ] F22.5 Streaming predittivo basato su velocità e direzione della camera

**Uscita**: volo in una scena più grande della RAM su T0 (16 GB) senza hitch > 33 ms.

---

# Era V — Simulazione e runtime (linea parallela)

## F23 — CPU ultra-ottimizzata [CORE]

- [ ] F23.1 Job system (enkiTS) mappato sulle classi QoS: P-core per render/simulazione, E-core per streaming e compilazione
- [ ] F23.2 Arena per frame, allocatori lineari, niente `new` nel ciclo caldo (O7)
- [ ] F23.3 Layout data-oriented (SoA) per ECS, transform, culling CPU residuo
- [ ] F23.4 NEON esplicito nei cicli caldi; **SME/AMX tramite Accelerate** (M4+) per skinning CPU, matrici e simulazione
- [ ] F23.5 Pipelining CPU/GPU: simulazione del frame N+1 mentre la GPU disegna il frame N
- [ ] F23.6 Latenza input→fotoni misurata e minimizzata (`CAMetalDisplayLink`, present timing)

## F24 — Fisica, cloth e distruzione [AAA]

- [ ] F24.1 Jolt Physics: corpi rigidi, character controller, raycast, trigger, debug draw
- [ ] F24.2 Backend GPU Metal di Jolt (dalla 5.6) per simulazioni massive
- [ ] F24.3 Cloth XPBD in compute con collisioni sulla depth e sulle capsule
- [ ] F24.4 Distruzione con frammenti precalcolati e simulazione GPU dei detriti

## F25 — Animazione e personaggi [AAA]

- [ ] F25.1 ozz-animation + ACL (compressione), blend tree, IK
- [ ] F25.2 Skinning in compute con output diretto ai buffer della GPU scene; BLAS refit per i personaggi
- [ ] F25.3 **Capelli a filamenti** renderizzati con mesh shader e simulati in compute
- [ ] F25.4 **[EDGE]** Motion matching, poi versione appresa (learned motion matching con rete piccola in tensori Metal)
- [ ] F25.5 Rendering della pelle (subsurface), occhi, denti

## F26 — Audio spaziale e acustica in ray tracing [EDGE]

- [ ] F26.1 AVAudioEngine + PHASE, mixer, streaming della musica
- [ ] F26.2 **Acustica in ray tracing** sulla stessa BVH del renderer: occlusione, riverbero e propagazione calcolati sulla GPU
- [ ] F26.3 Audio spaziale personalizzato (AirPods, head tracking)

## F27 — Runtime di gioco [CORE]

- [ ] F27.1 ECS scalabile (EnTT o flecs) con gerarchie, 100K+ entità dinamiche
- [ ] F27.2 Input: GameController (controller, haptics), mappatura azioni rimappabile, tastiera/mouse/trackpad
- [ ] F27.3 Scripting Luau con binding all'ECS e hot reload
- [ ] F27.4 Serializzazione di scene e salvataggi, iCloud opzionale
- [ ] F27.5 Rete (GameNetworkingSockets) per multiplayer, opzionale
- [ ] F27.6 Determinismo della simulazione (prerequisito per replay e test in F35)

---

## Checkpoint WWDC 2027 (giugno 2027)

- [ ] Sessioni Metal, MetalFX, Game Porting Toolkit, release note di OS 27
- [ ] Aggiornare metal-cpp; nuove famiglie GPU (M6?), deprecazioni
- [ ] Valutare nuove funzioni (estensioni RT, eventuale equivalente dei work graph, nuovi tipi di tensori) e riscrivere le fasi F28–F38 di conseguenza
- [ ] Aggiornare report e roadmap

---

# Era VI — Frontiera

## F28 — Scalabilità e presentazione [CORE]

- [ ] F28.1 Rilevamento famiglia GPU, core, memoria → tier; preset per modello di Mac ("For This Mac"), override utente
- [ ] F28.2 Risoluzione dinamica guidata dai tempi GPU
- [ ] F28.3 **Rasterization rate map** (shading a frequenza variabile nativo di Metal) per bordi, motion blur e foveazione
- [ ] F28.4 MetalFX frame interpolation con present thread dedicato; ingresso ≥ 30 fps
- [ ] F28.5 ProMotion 120 Hz, VRR, frame pacing a 2 bucket
- [ ] F28.6 Termica ed energia: riduzione progressiva della qualità, modalità batteria (O10), Game Mode
- [ ] F28.7 Matrice di compatibilità verificata su T0–T3 reali

## F29 — Autotuning per dispositivo [EDGE]

**Idea**: un engine che si ottimizza da solo su ogni Mac.

- [ ] F29.1 Parametri esposti: dimensioni dei threadgroup, tile, numero di sonde, dimensione dei meshlet, varianti di shader
- [ ] F29.2 Benchmark automatico al primo avvio (o in background) che cerca la configurazione migliore per il dispositivo
- [ ] F29.3 Risultati salvati per modello di chip e versione di OS, condivisibili tramite telemetria opzionale
- [ ] F29.4 Uso di Claude Code in locale come "perf engineer" automatico: cattura → analisi dei contatori → proposta di patch → misura

## F30 — Rendering neurale I [EDGE]

- [ ] F30.1 Infrastruttura: `MTLTensor`, TensorOps / Metal Performance Primitives negli shader, encoder ML per reti Core ML; microbenchmark reali (GEMM ≥ 32×32 per il picco)
- [ ] F30.2 **Radiance cache neurale** addestrata online negli shader (evoluzione di F12.2)
- [ ] F30.3 **Compressione neurale delle texture**: "on load" (transcodifica ad ASTC/BC, tutti i tier) poi "on sample" su T3
- [ ] F30.4 Upscaler MetalFX neurale (WWDC26) su M5 Pro/Max
- [ ] F30.5 Ogni funzione con fallback, interruttore nei preset e guadagno misurato

## F31 — Rendering neurale II: reti addestrate in casa [EDGE]

**Idea**: il Mac di sviluppo da 128 GB diventa la workstation di training con **MLX**.

- [ ] F31.1 Pipeline di dati: l'engine esporta coppie (input rumoroso, riferimento path traced da F32.1)
- [ ] F31.2 Denoiser e/o upscaler proprietari addestrati con MLX, esportati in tensori Metal
- [ ] F31.3 **Materiali neurali** per materiali stratificati complessi
- [ ] F31.4 **LOD e impostor neurali** per oggetti lontani
- [ ] F31.5 Esposizione e tonemapping appresi dal giudizio estetico (dataset curato)

## F32 — Path tracing in tempo reale [EDGE]

- [ ] F32.1 Path tracer di riferimento (accumulo progressivo) per validare GI, riflessioni e materiali
- [ ] F32.2 ReSTIR PT in tempo reale con terminazione nella radiance cache
- [ ] F32.3 Denoise MetalFX a 1 spp o denoiser di F31.2
- [ ] F32.4 Modalità foto con accumulo ad alta qualità

**Uscita**: modalità PT a 30 fps su M5 Max con upscaling + frame interpolation.

## F33 — Gaussian splatting ibrido [EDGE]

- [ ] F33.1 Rendering di 3D Gaussian splatting ordinato in compute con TBDR
- [ ] F33.2 Integrazione nel visibility buffer e nella depth: splat e mesh nella stessa scena
- [ ] F33.3 Illuminazione degli splat dalla radiance cache (rilluminazione approssimata)
- [ ] F33.4 Pipeline di import da cattura fotogrammetrica (anche da iPhone)

---

# Era VII — Prodotto

## F34 — Editor e strumenti [CORE]

- [ ] F34.1 Editor con ImGui docking (eventuale shell SwiftUI): scena, outliner, inspector, gizmo, undo/redo
- [ ] F34.2 Editor di materiali e grafo VFX
- [ ] F34.3 Browser degli asset collegato al cooker, reimport automatico, hot reload di tutto
- [ ] F34.4 Strumenti di illuminazione: posizionamento sonde, anteprima dei tier

## F35 — QA automatizzata [CORE]

- [ ] F35.1 Rendering deterministico dei testbench, **test visivi** (FLIP/PSNR) su runner macOS self-hosted
- [ ] F35.2 **Perf bot**: tempi per pass per commit su T0 e T2, allarme sopra soglia
- [ ] F35.3 Replay deterministico del gameplay (da F27.6) per bug e prestazioni
- [ ] F35.4 Fuzzing del cooker e dei loader, sanitizer in CI
- [ ] F35.5 Validazione shader (`MTL_SHADER_VALIDATION`) in una suite notturna

## F36 — Piattaforme Apple [AAA]

- [ ] F36.1 iPad M3+: input touch, preset T0, termica
- [ ] F36.2 iPhone Pro (A17 Pro+, Apple9): budget di memoria e termici mobili
- [ ] F36.3 **[EDGE] visionOS**: rendering stereo con Compositor Services, rendering foveato con rasterization rate map, reprojection

## F37 — Vertical slice [CORE]

- [ ] F37.1 Livello di 10–15 minuti AAA-like (area aperta + interni + meteo + personaggi)
- [ ] F37.2 Tutti i budget rispettati su T0–T3 con i preset automatici
- [ ] F37.3 Nessun crash in 1 ora di gioco, frame pacing a 2 bucket

## F38 — Distribuzione e live ops [CORE]

- [ ] F38.1 App bundle firmata, hardened runtime, notarizzazione
- [ ] F38.2 Build Steam e Mac App Store (sandbox)
- [ ] F38.3 Telemetria opzionale (prestazioni per modello, crash), aggiornamenti incrementali dei contenuti
- [ ] F38.4 Documentazione dell'engine per chi crea contenuti

---

## Cosa significa "SOTA 2026–2027" alla fine del percorso critico

| Area | Stato atteso |
|---|---|
| Geometria | GPU scene persistente, submission GPU-driven, cluster LOD virtualizzati, V-buffer, raster ibrido HW/SW |
| Luce | ReSTIR DI con migliaia di luci, ombre RT, GI a radiance cache (neurale su T3), ReSTIR GI, riflessioni RT |
| Ricostruzione | MetalFX temporal/denoised/neurale, frame interpolation, rasterization rate map |
| Path tracing | ReSTIR PT in tempo reale su M5 Max/Ultra |
| Neurale | Radiance cache e compressione delle texture neurali; reti proprietarie addestrate con MLX |
| Streaming | MTLIO + risorse sparse, virtual texturing, archivi di pipeline AOT |
| Efficienza | Banda per pixel budgettata, zero allocazioni e zero stutter per costruzione, autotuning per dispositivo |
| Piattaforma | Preset per ogni Mac Apple9+, EDR, ProMotion, Game Mode, iPad |

**Limiti di Metal rispetto ai motori PC** (dal report): niente shader execution
reordering generale, opacity micromap, cluster acceleration structure, TLAS
partizionati, work graph. La roadmap li aggira (proxy per l'RT, ordinamento
software dei raggi, radiance cache, catene di dispatch indiretti, ricostruzione
ML) invece di emularli.
