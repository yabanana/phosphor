# Apple SoC Playbook — spremere M3/M4/M5 fino all'ultimo ciclo

Manuale di riferimento per le fasi **OPT** di [`ROADMAP.md`](ROADMAP.md): tutto
quello che sappiamo (e quello che dobbiamo misurare) su ogni componente del SoC
Apple Silicon, con le tecniche per sfruttarlo e gli errori da evitare.

Ogni voce ha un ID (`S-TBDR-3`) citato dalle fasi OPT e ha quattro parti:

- **Fatto**: cosa è documentato, con fonte (Apple quando possibile)
- **Misura**: il microbenchmark (`B-xx`, sezione 0) che lo quantifica sul nostro hardware
- **Sfruttare**: come Phosphor lo usa a proprio vantaggio
- **Evitare**: gli anti-pattern che lo sprecano

Le affermazioni senza fonte sono marcate **(ipotesi)** e vanno verificate con
la misura indicata prima di basarci decisioni. Apple non pubblica clock, TFLOPS,
dimensioni delle cache né della tile memory: **quello che non è documentato si
misura**, non si assume.

Fonti principali (dettagli in `research_notes/…/apple_gpu_architecture.md` e
`metal4_api.md`): Apple Tech Talk 111375 (GPU M3/A17 Pro), Tech Talk 111431
(GPU M5/A19), Tech Talk 111432 (ML su M5), WWDC20-10632 (ottimizzazione per
Apple silicon), Metal Feature Set Tables (maggio 2026), benchmark di Philip
Turner, tzakharko e Michael's Tinkerings.

---

## 0. Caratterizzazione: la suite `bench/`

Prima di spremere un chip bisogna conoscerlo. La suite `bench/` (costruita in
OPT-0) esegue microbenchmark e salva i risultati in
`bench/results/<chip>-<os>.json`. Va eseguita su **ogni** Mac disponibile
(almeno un T0 e l'M5 Max) e ripetuta dopo ogni major release di macOS.

| ID | Misura | Serve per |
|---|---|---|
| B-01 | Throughput FP32, FP16, INT32 (FMA, add, mul) con catene dipendenti e indipendenti | S-ALU-1..3 |
| B-02 | Dual-issue FP16+FP32+INT al variare del numero di SIMD-group | S-ALU-2 |
| B-03 | Costo di trascendentali (rcp, rsqrt, exp, log, sin) FP32 vs FP16 vs `fast::` | S-ALU-4 |
| B-04 | Curva occupancy ↔ registri vivi, punto di thrashing del dynamic caching | S-OCC-1..3 |
| B-05 | Threadgroup memory: banda, latenza, conflitti di banco per pattern di accesso | S-SIMD-3 |
| B-06 | SIMD-group: shuffle, ballot, prefix sum, `quad_*`, riduzioni vs threadgroup memory | S-SIMD-1..2 |
| B-07 | Atomici 32/64 bit, device vs threadgroup, con e senza contesa | S-SIMD-4 |
| B-08 | Banda e latenza per livello: L1, SLC, DRAM (pointer chasing + stream); stima della dimensione di SLC | S-MEM-1..3 |
| B-09 | Contesa di banda CPU↔GPU (GPU in stream mentre la CPU fa memcpy) | S-MEM-4 |
| B-10 | Campionamento texture per formato (RGBA8, BC7, ASTC, RGBA16F, R11G11B10), compresse vs non, accesso coerente vs sparso | S-TEX-1..4 |
| B-11 | Compressione lossless: rapporto e costo per render target e texture scritte da shader (M5) | S-TEX-2 |
| B-12 | Soglia del parameter buffer: triangoli/vertici per pass prima del partial render | S-TBDR-3 |
| B-13 | Efficienza HSR in funzione dell'overdraw e dell'ordine di draw | S-TBDR-2 |
| B-14 | Costo di load/store per formato e risoluzione; costo di un render pass vuoto | S-TBDR-4 |
| B-15 | Dimensione massima dell'imageblock / tile memory per pixel alle varie tile size | S-TBDR-1 |
| B-16 | Mesh shader vs vertex shader: triangoli/s, costo del payload dell'object shader, dimensione ottimale dei meshlet | S-GEO-1..3 |
| B-17 | Costo di encoding di un ICB dalla GPU; overhead di dispatch vuoti e indiretti | S-GEO-4, S-SYNC-3 |
| B-18 | Costo delle barriere Metal 4: intra-encoder, di coda, per coppia di stage | S-SYNC-1 |
| B-19 | Sovrapposizione tra pass (vertex del pass N+1 durante fragment del pass N), async compute | S-SYNC-2 |
| B-20 | Raggi/s coerenti vs incoerenti, `intersector` vs `intersection_query`, con e senza intersection function | S-RT-1..3 |
| B-21 | Build, refit e compaction di BLAS/TLAS per numero di triangoli/istanze | S-RT-4 |
| B-22 | Neural Accelerator: GEMM per dimensione di tile, tipo (FP16, BF16, INT8, INT4, FP8) e scope; MLP piccoli in shader | S-NA-1..3 |
| B-23 | Neural Engine: latenza e throughput di reti Core ML tipiche, concorrenza con la GPU | S-ANE-1 |
| B-24 | CPU: NEON vs scalare, SME/AMX via Accelerate, P-core vs E-core, latenza dei thread per QoS | S-CPU-1..4 |
| B-25 | I/O: MTLIO per codec (LZ4, LZFSE, zlib, LZMA, LZBITMAP), dimensione di richiesta, code parallele | S-IO-1..2 |
| B-26 | Latenza input → fotoni, jitter di presentazione, `CAMetalDisplayLink` | S-DISP-1 |
| B-27 | Potenza (W) e frequenze sotto carico GPU, CPU, misto; tempo a regime termico (MacBook Air vs Pro vs Studio) | S-PWR-1..3 |
| B-28 | Latenza di commit di un command buffer MTL4 e di un segnale di evento visto dalla CPU | S-SYNC-4 |

---

## 1. Core GPU e ALU (S-ALU)

**S-ALU-1 Larghezza e throughput**
- **Fatto**: 128 ALU per core, 4 scheduler per core, SIMD-group da 32 thread (Turner). Base M5 misurato a ~3,85 TFLOPS FP32 a 1.578 MHz (Michael's Tinkerings). M5 raddoppia il throughput FP16 e delle operazioni "complesse" rispetto a M4 (Apple 111431).
- **Misura**: B-01.
- **Sfruttare**: codice scalare ricco di parallelismo a livello di istruzione (ILP); più catene indipendenti per thread nei kernel caldi.
- **Evitare**: lunghe catene di dipendenze; assumere che `float4` sia "vettoriale" (compila in 4 FMA scalari).

**S-ALU-2 Dual-issue FP16/FP32/INT**
- **Fatto**: da Apple9 FP16, FP32 e interi girano in parallelo "fino a 2x", ma **solo tra SIMD-group diversi**: serve occupancy (Apple 111375).
- **Misura**: B-02.
- **Sfruttare**: mescolare aritmetica `half` e intera (indirizzi, bit packing) nello stesso kernel; mantenere abbastanza SIMD-group attivi.
- **Evitare**: kernel a bassa occupancy che non possono sfruttare l'issue parallelo.

**S-ALU-3 FP16 ovunque possibile**
- **Fatto**: `half` gira al throughput massimo, usa meno registri e le conversioni sono gratuite (111375); su M5 il vantaggio raddoppia (111431). Usare il suffisso `h` sui letterali per evitare promozioni (WWDC20-10632).
- **Misura**: B-01, B-04.
- **Sfruttare**: colore, BRDF, normali, pesi di filtri, reservoir ReSTIR (pesi), probe GI in `half`; `float` solo per posizioni, profondità, accumulatori lunghi.
- **Evitare**: letterali senza `h` che promuovono l'intera espressione a FP32.

**S-ALU-4 Trascendentali e interi**
- **Fatto**: reciproci e radici FP32 sono multi-ciclo; la moltiplicazione intera è ~4x più lenta della somma (Turner, da verificare nelle tabelle del repo).
- **Misura**: B-03.
- **Sfruttare**: approssimazioni polinomiali in `half`, `fast::` dove accettabile, strength reduction degli indici (shift/mask invece di mul/div), tabelle in constant memory.
- **Evitare**: divisioni intere e modulo nei cicli caldi.

---

## 2. Registri, dynamic caching, occupancy (S-OCC)

**S-OCC-1 Il register file è una cache**
- **Fatto**: da M3 registri, threadgroup, tile, stack e buffer condividono le stesse cache on-chip; i registri si allocano dinamicamente in base all'uso effettivo; l'hardware abbassa l'occupancy per evitare il thrashing (111375).
- **Misura**: B-04.
- **Sfruttare**: tenere bassi i registri **vivi nel tempo**, non solo il picco: rilasciarli presto, spezzare le fasi del kernel, ricalcolare invece di conservare valori economici.
- **Evitare**: uber-shader con molti valori vivi attraverso tutto il kernel; array sullo stack; unrolling aggressivo.

**S-OCC-2 Occupancy Management Unit di M5**
- **Fatto**: su M5 l'occupancy viene ridotta per quattro cause: pressione di registri, pressione di memoria privata (threadgroup/stack), stalli di richieste di memoria (LLC/MMU/DRAM), stalli di decompressione delle texture. Xcode 26.4 mostra i contatori Occupancy Target / Target Influence e i registri vivi per riga (111431).
- **Misura**: B-04 + contatori Xcode per ogni shader caldo.
- **Sfruttare**: una tabella in `docs/perf-log.md` con occupancy target e causa dominante di ogni shader caldo; intervenire sulla causa, non a tentativi.
- **Evitare**: accessi casuali a buffer grandi (stalli di memoria); texture compresse con accesso sparso (stalli di decompressione, vedi S-TEX-2).

**S-OCC-3 Threadgroup memory e occupancy**
- **Fatto**: la threadgroup memory condivide lo spazio on-chip con i registri (111375); limite API da interrogare a runtime (`maxThreadgroupMemoryLength`).
- **Misura**: B-04, B-05.
- **Sfruttare**: usare la threadgroup memory solo dove il riuso lo giustifica; preferire SIMD-group intrinsics (S-SIMD-1) per scambi dentro 32 thread.
- **Evitare**: allocazioni di threadgroup memory "per sicurezza" di dimensione massima.

---

## 3. SIMD-group, threadgroup memory, atomici (S-SIMD)

**S-SIMD-1 SIMD-group da 32**
- **Fatto**: 32 thread per SIMD-group (Turner); MSL offre `simd_shuffle`, `simd_ballot`, `simd_prefix_*`, `simd_sum/min/max`, `quad_*`.
- **Misura**: B-06.
- **Sfruttare**: compattazione (liste di meshlet visibili, pixel da ombreggiare), riduzioni (Hi-Z, istogramma di luminanza), prefix sum per l'allocazione, votazioni per uniformità (salto di rami).
- **Evitare**: atomici su un contatore globale per ogni thread (usare prima un prefix per SIMD-group, poi un atomico per gruppo).

**S-SIMD-2 Uniformità e divergenza**
- **Fatto**: divergenza dentro il SIMD-group serializza i rami (comportamento SIMT standard) **(ipotesi sui costi esatti)**.
- **Misura**: B-06 con percentuali di divergenza variabili.
- **Sfruttare**: binning per materiale/classe prima dello shading (V-buffer resolve per tile classificate), `simd_all/any` per saltare lavoro.
- **Evitare**: uber-shader con rami per materiale dentro lo stesso SIMD-group.

**S-SIMD-3 Threadgroup memory**
- **Misura**: B-05 (banda, conflitti di banco per stride).
- **Sfruttare**: tile di dati riusati (filtri separabili, blocchi di GEMM, liste di luci per tile).
- **Evitare**: stride che causano conflitti di banco (misurare quali), barriere di threadgroup superflue.

**S-SIMD-4 Atomici**
- **Fatto**: atomici a 64 bit completi da Apple9; su M2 solo min/max a 64 bit, su M1 assenti (Feature Set Tables).
- **Misura**: B-07 (throughput con e senza contesa).
- **Sfruttare**: raster software nel V-buffer con `atomic_max` 64 bit (profondità|ID) su Apple9; atomici gerarchici (SIMD → threadgroup → device).
- **Evitare**: contesa alta su pochi indirizzi; atomici device dove basta threadgroup.

---

## 4. Tile memory, TBDR, HSR, parameter buffer (S-TBDR)

**S-TBDR-1 Tile memory e imageblock**
- **Fatto**: banda "molte volte" superiore alla DRAM, latenza molto inferiore, energia "significativamente" minore; gli imageblock sono strutture per pixel che persistono per tutta la vita della tile tra draw e dispatch; i tile shader girano dentro il render pass condividendo la tile memory (Apple, doc TBDR). Dimensione non pubblicata: da interrogare (`imageblockSampleLength`) e misurare.
- **Misura**: B-15.
- **Sfruttare**: G-buffer, accumulo di luce, OIT, liste di luci per tile **dentro la tile**; tile size scelte per il carico (es. 32×32 per il light culling).
- **Evitare**: scrivere in DRAM dati che il pass successivo rilegge subito.

**S-TBDR-2 Hidden Surface Removal**
- **Fatto**: con HSR massimizzato il depth prepass è ridondante; ordine opachi → feedback (alpha test, discard, depth write) → traslucidi; scrivere tutti i canali degli attachment per evitare il write-masking che disattiva l'HSR; `[[early_fragment_tests]]` negli shader che scrivono in memoria (WWDC20-10632).
- **Misura**: B-13.
- **Sfruttare**: nessun depth prepass; draw ordinate per classe; alpha test in un pass separato dopo gli opachi.
- **Evitare**: discard negli shader opachi; blending o depth write dallo shader mescolati agli opachi.

**S-TBDR-3 Parameter buffer e partial render**
- **Fatto**: il tiler conserva i vertici trasformati nel parameter buffer; quando si riempie, la GPU fa un *partial render* (flush delle tile a metà pass) con perdita di prestazioni (Rosenzweig). Il V-buffer riduce l'uso del parameter buffer (Apple 111431). Soglia non documentata.
- **Misura**: B-12 (e contatori in Xcode).
- **Sfruttare**: culling aggressivo prima del raster (mesh shader, occlusione), meshlet con output minimi, attributi minimi nel pass di visibilità (solo ID).
- **Evitare**: attributi pesanti interpolati nel pass di raster; geometria densa non culled.

**S-TBDR-4 Load/store e memoryless**
- **Fatto**: load/store "consumano la maggior parte della banda di sistema"; clear nella load action; `.dontCare` per ciò che non serve dopo; `memoryless` per attachment usati solo nel pass (solo texture); unire pass adiacenti con gli stessi attachment (WWDC20-10632).
- **Misura**: B-14.
- **Sfruttare**: il render graph (F2) deduce load/store e memoryless; depth, MSAA, G-buffer on-tile sempre memoryless.
- **Evitare**: pass multipli che si passano la stessa texture via DRAM; `.store` per default.

**S-TBDR-5 Barriere nello stadio fragment**
- **Fatto**: "operazione molto costosa" perché svuota la tile memory in memoria di sistema (WWDC20-10632); in Metal 4 le barriere con fragment/tile nel lato "after" non sono supportate da Apple3 ad Apple10 (Feature Set Tables).
- **Misurato (F2.3, M5 Max, 2026-09-29, `bench/barrier_spike`)**: una barriera di coda con Fragment sul lato consumatore di un render encoder è legale ed efficace, e costa ~16% meno che attendere in Vertex; Tile è accettato ma non sincronizza; dentro un render encoder il lato produttore di `barrierAfterEncoderStages` può essere solo Vertex/Object/Mesh (fragment/tile = abort della validazione). Il divieto riguarda quindi le barriere *dentro* il pass, non l'attesa di un pass successivo.
- **Sfruttare**: organizzare le dipendenze in modo che il consumo avvenga in compute o nel pass successivo.
- **Evitare**: qualunque sincronizzazione dentro il fragment.

**S-TBDR-6 Sovrapposizione tra pass**
- **Fatto**: il vertex stage di un pass successivo può partire mentre il fragment del pass precedente finisce in tile memory (doc TBDR).
- **Misura**: B-19.
- **Sfruttare**: ordinare i pass perché geometria pesante (shadow, V-buffer) si sovrapponga a fragment pesanti; lavoro compute indipendente su una seconda coda.
- **Evitare**: dipendenze artificiali che serializzano pass indipendenti (falsi hazard sulla stessa risorsa).

**S-TBDR-7 Programmable blending e raster order groups**
- **Fatto**: leggere gli attachment correnti nel fragment shader permette deferred in un solo pass; i ROG ordinano gli accessi per pixel e, con più gruppi, serializzano solo l'accumulo (doc TBDR, WWDC20-10632).
- **Sfruttare**: OIT multi-layer (MLAB) nella tile, decal, accumulo di luci on-tile, voxelizzazione.
- **Evitare**: usare atomici in device memory per ciò che un ROG fa nella tile.

**S-TBDR-8 MSAA on-chip**
- **Fatto**: l'hardware fa il blending per campione solo sui pixel di bordo; resolve nella tile con tile shader; su M5 MSAA 8x risolto on-chip con texture memoryless e compressione (111431).
- **Sfruttare**: MSAA 4x memoryless come opzione di qualità per pass specifici (UI, vegetazione in alpha-to-coverage) a costo di banda quasi nullo.

---

## 5. Texture e compressione (S-TEX)

**S-TEX-1 Formati**
- **Fatto**: BC e ASTC su tutti i Mac Apple silicon (Feature Set Tables).
- **Misura**: B-10.
- **Sfruttare**: ASTC per contenuti (qualità/bit regolabile), BC7 dove serve compatibilità degli strumenti; R11G11B10F / RGB9E5 per HDR intermedi dove la precisione basta.
- **Evitare**: RGBA16F ovunque "per sicurezza".

**S-TEX-2 Compressione lossless della GPU**
- **Fatto**: texture private popolate via blit sono compresse dalla GPU; `replaceRegion()` dalla CPU salta la compressione; da M5 anche le texture scritte dagli shader sono compresse ("universal compression"); per accessi sparsi conviene disattivarla (`allowGPUOptimizedContents = false`) per evitare overfetch di blocco; contatori Compression Ratio e Compressed Texture Write Inefficiency (111431).
- **Misura**: B-11 + contatori.
- **Sfruttare**: scritture a blocchi interi (tile allineate) negli output compute su M5; blit per ogni caricamento.
- **Evitare**: scritture parziali di blocco (read-modify-write); compressione su texture ad accesso casuale (tabelle, atlanti di probe consultati sparsamente → da misurare).

**S-TEX-3 Mip e cache delle texture**
- **Sfruttare**: mip bias corretto per la risoluzione di render (MetalFX), mip più bassi per effetti a bassa frequenza (GI, riflessioni ruvide), campionamento coerente (ordinare i lavori per regione di texture).
- **Evitare**: campionare il mip 0 in effetti che non lo richiedono (stalli di decompressione su M5).

**S-TEX-4 Texture sparse**
- **Fatto**: risorse placement sparse con pagine da 16 o 64 KB su heap placement, mapping sulla coda (Metal 4); texture fino a 32K su M5.
- **Misura**: B-10 (accesso a pagine mappate/non mappate), costo delle operazioni di mapping.
- **Sfruttare**: virtual texturing, shadow map virtuali se servissero, residency fine di cluster e texture.

---

## 6. Geometria: mesh shader, ICB, raster (S-GEO)

**S-GEO-1 Mesh shader**
- **Fatto**: da Apple9 i threadgroup object/mesh sono schedulati per tenere i meshlet on-chip; griglia mesh > 1M threadgroup; dichiarare vertici/primitive massimi solo quanto serve; omettere le primitive scartate invece di affidarsi al culling hardware successivo (111375). Payload fino a 16 KB. RT e function pointer nelle render pipeline sono incompatibili con il mesh shading (Feature Set Tables).
- **Misura**: B-16.
- **Sfruttare**: dimensione dei meshlet scelta dalle misure (64/96/128); culling nell'object shader; output minimi.
- **Evitare**: massimi sovradimensionati (più traffico, meno occupancy); RT nel mesh shader (impossibile).

**S-GEO-2 Throughput geometrico**
- **Fatto**: M5 raddoppia il throughput geometrico rispetto a M4 (111431).
- **Sfruttare**: su M5 soglie di LOD più fini e meno raster software; la soglia HW/SW del raster va calibrata **per generazione**.

**S-GEO-3 Dati di vertice**
- **Sfruttare**: posizioni quantizzate relative al cluster, normali ottaedriche, UV in `half`; attributi letti solo nel resolve (V-buffer) → il pass di raster tocca solo le posizioni.

**S-GEO-4 Indirect command buffer**
- **Fatto**: ICB codificabili dalla GPU; draw mesh negli ICB da Apple9; su Apple10 stato raster, depth/stencil, cull e winding impostabili per draw dalla GPU (111431).
- **Misura**: B-17.
- **Sfruttare**: submission interamente GPU-driven (O8); su M5 ICB unici per pass d'ombra con materiali misti.

**S-GEO-5 Funzioni specifiche Apple10**
- **Fatto**: valori per-vertex non interpolati accessibili nel fragment (utile al V-buffer), depth bounds test, riduzione min/max nel sampler (Hi-Z in un fetch), LOD bias (111431, Feature Set Tables).
- **Sfruttare**: percorsi dedicati Apple10 con fallback Apple9, attivati dal rilevamento della famiglia.

---

## 7. Ray tracing (S-RT)

**S-RT-1 Unità RT e reorder**
- **Fatto**: traversal in hardware da M3 con uno stadio di reorder che raggruppa le chiamate alle intersection function; usare l'API `intersector` perché `intersection_query` disattiva il reorder; evitare intersection function "uber" e payload grandi (111375). M4 dichiara RT 2x più veloce (Apple Newsroom).
- **Misura**: B-20.
- **Sfruttare**: `intersector` ovunque; payload minimi; intersection function specializzate e brevi.
- **Evitare**: `intersection_query` nei kernel caldi; alpha test complesso nelle intersection function.

**S-RT-2 Novità M5 (terza generazione)**
- **Fatto**: trasformazioni d'istanza in hardware; indicizzazione delle intersection function buffer interamente hardware (fino a −70% di tempo rispetto all'emulazione); allineamento delle acceleration structure da 16 KB a 1 KB; chiedere limiti estesi solo se servono; scegliere bene istanze statiche vs motion (111431).
- **Sfruttare**: su M5 molte istanze piccole costano poco (vegetazione, folle); su M3/M4 unire le istanze piccole in BLAS più grandi (re-braiding, OPT-5).
- **Evitare**: su M3/M4 migliaia di BLAS minuscole (spreco di 16 KB di padding ciascuna).

**S-RT-3 Coerenza dei raggi**
- **Fatto**: Metal non ha shader execution reordering generale (report).
- **Misura**: B-20 (coerenti vs incoerenti).
- **Sfruttare**: ordinamento software dei raggi per direzione/origine prima del trace, raggi "corti" dove possibile, geometria proxy semplice.

**S-RT-4 Build e aggiornamento**
- **Misura**: B-21.
- **Sfruttare**: refit per animazioni piccole, rebuild ammortizzato su più frame, compaction per le BLAS statiche, build guidate da indirizzo (Apple9).

---

## 8. Neural Accelerator (S-NA)

**S-NA-1 Throughput e forma del lavoro**
- **Fatto**: un Neural Accelerator per core GPU da M5; ~1.024 FLOP FP16 e ~2.048 op INT8 per core per ciclo; tile migliori ≥ 32×32; il picco supera ciò che la banda può alimentare; carichi piccoli o frammentati crollano sotto il picco (tzakharko). Una misura indipendente su M5 Max indica ~19,9 TFLOPS reali su GEMM FP16 grandi (Creative Strategies, da verificare).
- **Misura**: B-22.
- **Sfruttare**: reti progettate come GEMM a tile ≥ 32×32; batch di pixel per SIMD-group; pesi riusati in threadgroup memory.
- **Evitare**: MLP per pixel con dimensioni minuscole e dispatch separati.

**S-NA-2 Tipi di dato per versione di OS**
- **Fatto**: FP16 al lancio; BF16 da OS 26.1; cooperative tensor come input di matmul da 26.3; INT8 e INT4 da 26.4 (111432); FP8/FP4 e formati block-scaled da OS 27 (note Metal 4).
- **Sfruttare**: quantizzazione INT8/FP8 per inferenza nel frame, con fallback FP16 per OS più vecchi.

**S-NA-3 Cooperative tensor e fusione**
- **Fatto**: i cooperative tensor tengono l'output nei registri distribuiti tra i thread, per applicare attivazioni prima di un solo store; barriere di threadgroup ogni poche iterazioni K; traversal Morton/Hilbert dei threadgroup (111432).
- **Sfruttare**: MLP "fully fused" in un solo kernel (strati consecutivi senza passare dalla DRAM).

**S-NA-4 Competizione con la grafica**
- **Fatto**: i Neural Accelerator stanno nei core shader: condividono core, cache e banda con la grafica (111432, report).
- **Sfruttare**: budget ML come parte del tempo GPU del frame; spostare su ANE (S-ANE) ciò che non deve stare nel frame.

---

## 9. Gerarchia di memoria (S-MEM)

**S-MEM-1 Banda per chip**
- **Fatto**: M5 153,6 GB/s; M5 Pro 307; M5 Max 460/614; M5 Ultra 1,2 TB/s; M4 Max 546 (Apple Newsroom). STREAM misurato su base M5: 122 GB/s (~80% del picco).
- **Misura**: B-08.
- **Sfruttare**: budget di byte per frame per tier (O1): circa 10 GB/frame a 60 fps su M5 Max, **condivisi con la CPU**; la metà su Pro, un quarto su base.

**S-MEM-2 System Level Cache (SLC)**
- **Fatto**: 32 MB di SLC condivisa CPU/GPU su base M5 (Michael's Tinkerings); dimensioni su Pro/Max non pubblicate.
- **Misura**: B-08 (curva di banda vs working set per trovare la dimensione).
- **Sfruttare**: dimensionare i working set dei pass compute (tile di lavoro, liste, reservoir) per restare in SLC; ordinare i lavori per località (Morton) per massimizzare i riusi.

**S-MEM-3 Storage mode**
- **Fatto**: `shared` zero-copy, `private` solo GPU con layout ottimizzati e compressione, `memoryless` solo tile (Apple, storage modes).
- **Sfruttare**: `private` per tutto ciò che la GPU legge spesso; `shared` + write-combined per gli anelli di upload; `memoryless` per gli intermedi di pass.

**S-MEM-4 Memoria unificata**
- **Misura**: B-09 (contesa CPU/GPU).
- **Sfruttare**: la CPU scrive direttamente nei buffer GPU (animazione, decompressione, dati di gioco) senza copie; lavori leggeri spostabili tra CPU e GPU a seconda del carico; dataset enormi residenti (128 GB).
- **Evitare**: CPU che satura la banda durante i pass GPU più pesanti (pianificare i job CPU pesanti fuori dalle finestre critiche).

**S-MEM-5 Hazard tracking e residency**
- **Fatto**: Metal 4 non traccia gli hazard e i command buffer non rendono residenti né trattengono le risorse; massimo 32 residency set per coda (note Metal 4).
- **Sfruttare**: pochi residency set grandi; commit in batch; aggiornamenti incrementali per lo streaming.

**S-MEM-6 Fusion Architecture (M5 Pro/Max)**
- **Fatto**: M5 Pro/Max sono due die uniti (Apple Newsroom).
- **Misura**: B-08 da core diversi **(ipotesi: eventuali asimmetrie di latenza tra die, da verificare)**.
- **Sfruttare**: se esistono asimmetrie, località dei dati per metà del chip (da decidere solo dopo la misura).

---

## 10. Sincronizzazione e scheduling GPU (S-SYNC)

**S-SYNC-1 Costo delle barriere**
- **Fatto**: barriere Metal 4 per coppie di stage, intra-encoder o di coda (note Metal 4).
- **Misura**: B-18.
- **Sfruttare**: il render graph usa una tabella di costi misurata per scegliere tipo e posizione delle barriere, e le raggruppa.

**S-SYNC-2 Parallelismo tra pass e code**
- **Misura**: B-19.
- **Sfruttare**: seconda coda MTL4 per compute asincrono (GI, streaming, build di BVH) sincronizzata con eventi.

**S-SYNC-3 Overhead di dispatch**
- **Misura**: B-17.
- **Sfruttare**: fondere dispatch piccoli; catene di dispatch indiretti per il lavoro variabile (sostituto dei work graph).

**S-SYNC-4 Latenza CPU↔GPU**
- **Misura**: B-28.
- **Sfruttare**: numero di frame in volo scelto sulla base della latenza misurata e del target di input lag.

---

## 11. CPU (S-CPU)

**S-CPU-1 Core eterogenei**
- **Fatto**: M5 Pro/Max: CPU a 18 core, 6 "super" + 12 performance (Apple Newsroom); le classi QoS determinano il core; i thread in QoS background sono confinati sugli E-core; nessun pinning esplicito ai core (note middleware, Apple).
- **Misura**: B-24.
- **Sfruttare**: render thread e simulazione in QoS user-interactive; streaming e compilazione shader in QoS utility; lavori di fondo in background; work stealing.
- **Evitare**: presupporre un numero fisso di core; spin-wait (spreca energia e blocca core).

**S-CPU-2 NEON**
- **Sfruttare**: culling CPU residuo, trasformazioni, decompressione, fisica in SoA con intrinsics NEON.

**S-CPU-3 SME / AMX via Accelerate**
- **Fatto**: le unità matriciali della CPU sono raggiunte tramite Accelerate (BLAS, vDSP, BNNS) **(ipotesi: presenza di SME da M4, da verificare con B-24)**.
- **Misura**: B-24.
- **Sfruttare**: skinning CPU in batch, operazioni su matrici per folle e fisica, piccole reti su CPU.

**S-CPU-4 Latenza di scheduling**
- **Misura**: B-24 (wake-up per QoS).
- **Sfruttare**: code di lavoro persistenti invece di creare thread; job granulari ma non microscopici.

---

## 12. Neural Engine (S-ANE)

**S-ANE-1 ANE come coprocessore**
- **Fatto**: MetalFX neurale (WWDC26) usa Neural Engine **e** Neural Accelerator su M5 Pro/Max (note Metal 4); l'ANE è separato dai core GPU.
- **Misura**: B-23 (latenza, concorrenza con la GPU).
- **Sfruttare**: reti che non devono stare nel frame GPU (animazione appresa, audio, IA dei personaggi, previsione dello streaming) su ANE tramite Core ML, lasciando la GPU libera.
- **Evitare**: reti sul percorso critico del frame se la latenza di ANE non è compatibile (misurare).

---

## 13. Display e media (S-DISP)

**S-DISP-1 Presentazione**
- **Fatto**: frame interpolation MetalFX richiede un present thread dedicato e `CAMetalDisplayLink`; target di frame pacing a 2 bucket nell'istogramma del Performance HUD (WWDC25-211, WWDC26-357).
- **Misura**: B-26.
- **Sfruttare**: ProMotion 120 Hz e VRR; frame pacing prima del frame rate.

**S-DISP-2 EDR / XDR**
- **Sfruttare**: uscita EDR con headroom interrogato a runtime; tonemapping consapevole della luminanza del display; UI in SDR corretta.

**S-DISP-3 Media engine** **(ipotesi, da verificare)**
- **Sfruttare**: video in gioco (cinematiche, schermi nel mondo) decodificati dal media engine via VideoToolbox, importati senza copie come texture Metal (IOSurface), senza toccare CPU o shader.

---

## 14. Storage e I/O (S-IO)

**S-IO-1 SSD**
- **Fatto**: base M5 ~6,3 GB/s in lettura; ~13–15 GB/s riportati su un M5 Max (fonte singola); velocità dipendente dalla capacità (note GPU).
- **Misura**: B-25.
- **Sfruttare**: budget di streaming per tier e per capacità del disco rilevata.

**S-IO-2 MTLIO**
- **Fatto**: caricamento diretto in buffer/texture con decompressione (zlib, LZFSE, LZ4, LZMA, LZBITMAP o codec custom); non chiarito se la decompressione sia su CPU o hardware dedicato (note Metal 4).
- **Misura**: B-25 (throughput e uso CPU per codec).
- **Sfruttare**: scegliere il codec per tipo di dato in base alle misure; richieste grandi e allineate; code parallele.

---

## 15. Energia e termica (S-PWR)

**S-PWR-1 Consumi di riferimento**
- **Fatto**: M5 Max GPU ~75 W TDP; ~109 W di sistema medi in Cyberpunk (Notebookcheck).
- **Misura**: B-27 con `powermetrics`.

**S-PWR-2 Regime termico**
- **Misura**: B-27 (prestazioni dopo 10/30 minuti; MacBook Air senza ventola vs Pro vs Studio).
- **Sfruttare**: preset che tengono conto del regime termico, non dei primi secondi; riduzione progressiva della qualità prima del throttling.

**S-PWR-3 Efficienza**
- **Sfruttare**: frame cap intelligente, modalità batteria, meno banda = meno energia (la tile memory costa molto meno della DRAM, S-TBDR-1), E-core per i job non critici.
- **Evitare**: spin-wait della CPU; lavoro GPU speculativo non usato.

---

## 16. Cosa cambia per generazione

| Aspetto | M3 (Apple9) | M4 (Apple9) | M5 (Apple10) |
|---|---|---|---|
| Registri | Dynamic caching gen 1 | come M3 | Gen 2 + Occupancy Management Unit |
| FP16 | dual-issue | come M3 | throughput 2x |
| Geometria | mesh shader HW | come M3 | throughput 2x |
| Ray tracing | HW + reorder, allineamento AS 16 KB | "RT 2x" | istanze HW, IFB HW, allineamento 1 KB |
| Compressione | texture non scritte da shader | come M3 | universale (anche scritte da shader) |
| ML | TensorOps su ALU | come M3 | Neural Accelerator per core |
| Extra | — | — | depth bounds, sampler min/max, ICB estesi, MSAA 8x, texture 32K |

Regola: i percorsi di codice si scelgono per **famiglia** (Apple9 / Apple10),
i parametri numerici (dimensioni di meshlet, soglie di LOD, raggi per pixel)
per **chip misurato** tramite `bench/` e autotuning (F29).

---

## 17. Checklist rapida anti-spreco

- [ ] Nessun depth prepass; opachi → alpha test → traslucidi
- [ ] Tutti gli intermedi di pass `memoryless`, store `.dontCare` dove possibile
- [ ] Nessuna barriera nello stadio fragment
- [ ] Letterali `half` con suffisso `h`; FP32 solo dove serve
- [ ] Registri vivi controllati per ogni shader caldo (Xcode 26.4+)
- [ ] Occupancy target e causa di throttling annotati per ogni shader caldo
- [ ] Texture popolate via blit; compressione disattivata solo dove l'accesso è sparso
- [ ] `intersector` invece di `intersection_query`
- [ ] Meshlet con massimi dichiarati minimi, primitive scartate omesse
- [ ] Compattazioni e riduzioni con SIMD-group, non con atomici globali
- [ ] Nessuna allocazione, compilazione o I/O bloccante nel frame
- [ ] Budget di byte DRAM per frame rispettato per tier
- [ ] Misurato su T0, non solo sull'M5 Max
