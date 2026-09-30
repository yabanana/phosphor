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

Prima di spremere un chip bisogna conoscerlo. La suite [`bench/soc`](../bench/soc/README.md)
(`soc_bench`, costruita in OPT-0) esegue i microbenchmark e salva i risultati
in `bench/results/<chip>-<os>.json`; `tools/soc_bench_all.sh` fa i 3 run, la
validazione, i percorsi Apple9 e `leaks`; il modello di costo e il confronto
con le fonti esterne sono in [`soc-model.md`](soc-model.md). Va eseguita su
**ogni** Mac disponibile (almeno un T0 e l'M5 Max) e ripetuta dopo ogni major
release di macOS. **Misurato finora: solo M5 Max** (macOS 27.2, 2026-09-30);
le righe "Misura" qui sotto riportano quei valori.

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
- **Misura** (B-01, M5 Max, [`m5max-macos27.2.json`](../bench/results/m5max-macos27.2.json)): FP32 FMA 15,15 TFLOPS con 8 catene indipendenti = 116,9 FMA/core/clk a 1620 MHz su 40 core (il ciclo del kernel costa ~1 slot ogni 8 FMA: coerente con 128 ALU/core); catena dipendente 2,95 TFLOPS (5,1× più lenta: la latenza si nasconde solo con ILP/occupancy). Per core e per clock uguale al M5 base di terzi entro il 5% ([`soc-model.md`](soc-model.md)).
- **Sfruttare**: codice scalare ricco di parallelismo a livello di istruzione (ILP); più catene indipendenti per thread nei kernel caldi.
- **Evitare**: lunghe catene di dipendenze; assumere che `float4` sia "vettoriale" (compila in 4 FMA scalari).

**S-ALU-2 Dual-issue FP16/FP32/INT**
- **Fatto**: da Apple9 FP16, FP32 e interi girano in parallelo "fino a 2x", ma **solo tra SIMD-group diversi**: serve occupancy (Apple 111375).
- **Misura** (B-02): l'issue concorrente esiste: tempo(mix)/somma(tempi) = 0,81 (FP16+FP32), 0,76 (FP32+INT32), 0,71 (FP16+FP32+INT32), indipendente da 32…512 thread per threadgroup. FP16 raggiunge **1,85× FP32 solo con 32 catene indipendenti** (0,98× con 4, 1,15× con 8, 1,71× con 16); `half2` impacchettato 1,5–1,7× già con 8–16 catene.
- **Sfruttare**: mescolare aritmetica `half` e intera (indirizzi, bit packing) nello stesso kernel; mantenere abbastanza SIMD-group attivi.
- **Evitare**: kernel a bassa occupancy che non possono sfruttare l'issue parallelo.

**S-ALU-3 FP16 ovunque possibile**
- **Fatto**: `half` gira al throughput massimo, usa meno registri e le conversioni sono gratuite (111375); su M5 il vantaggio raddoppia (111431). Usare il suffisso `h` sui letterali per evitare promozioni (WWDC20-10632).
- **Misura** (B-01, B-02, B-04): FP16 add 12,8 contro FP32 6,8 Top/s (1,9×); FP16 FMA 17,4 contro 15,1 TFLOPS con 8 catene (il 2× richiede più ILP, vedi S-ALU-2); nessuna differenza di soglia di thrashing misurata tra FP16 e FP32 (B-04 usa FP32).
- **Sfruttare**: colore, BRDF, normali, pesi di filtri, reservoir ReSTIR (pesi), probe GI in `half`; `float` solo per posizioni, profondità, accumulatori lunghi.
- **Evitare**: letterali senza `h` che promuovono l'intera espressione a FP32.

**S-ALU-4 Trascendentali e interi**
- **Fatto**: reciproci e radici FP32 sono multi-ciclo; la moltiplicazione intera è ~4x più lenta della somma (Turner, da verificare nelle tabelle del repo).
- **Misura** (B-03, costo in FMA FP32 equivalenti): `fast::` rcp/rsqrt/exp2/log2 ≈ 0,9 FMA, sqrt 1,3, divisione 1,16, pow 2,8, sin/cos 6,5; `precise::` rsqrt 7, sqrt 8, divisione 12, sin/cos 20, pow 81; `half` ~1 FMA (sin/cos 5). Interi: shift 0,46, mask 0,04, divisione per costante 7 → 3,8 FMA (per potenze di 2 → 0,46). **Correzione**: la moltiplicazione intera costa 1,6× l'addizione (INT32 mul 4,06 contro add+xor 6,57 Top/s, B-01), non ~4×.
- **Sfruttare**: approssimazioni polinomiali in `half`, `fast::` dove accettabile, strength reduction degli indici (shift/mask invece di mul/div), tabelle in constant memory.
- **Evitare**: divisioni intere e modulo nei cicli caldi.

---

## 2. Registri, dynamic caching, occupancy (S-OCC)

**S-OCC-1 Il register file è una cache**
- **Fatto**: da M3 registri, threadgroup, tile, stack e buffer condividono le stesse cache on-chip; i registri si allocano dinamicamente in base all'uso effettivo; l'hardware abbassa l'occupancy per evitare il thrashing (111375).
- **Misura** (B-04): throughput stabile (6,9–7,2 Top/s) fino a **120 valori FP32 vivi** per thread, crollo a **128** (1,88 Top/s, ×0,27) e 256 (0,46): soglia di thrashing = 128 registri vivi (identica senza carichi di memoria: spill del compilatore). Un array indicizzato dinamicamente (stack) costa 62× la versione a registri.
- **Sfruttare**: tenere bassi i registri **vivi nel tempo**, non solo il picco: rilasciarli presto, spezzare le fasi del kernel, ricalcolare invece di conservare valori economici.
- **Evitare**: uber-shader con molti valori vivi attraverso tutto il kernel; array sullo stack; unrolling aggressivo.

**S-OCC-2 Occupancy Management Unit di M5**
- **Fatto**: su M5 l'occupancy viene ridotta per quattro cause: pressione di registri, pressione di memoria privata (threadgroup/stack), stalli di richieste di memoria (LLC/MMU/DRAM), stalli di decompressione delle texture. Xcode 26.4 mostra i contatori Occupancy Target / Target Influence e i registri vivi per riga (111431).
- **Misura**: B-04 (soglia sopra); i contatori Occupancy Target / Target Influence **non sono leggibili headless** (API e `xctrace` da CLI non espongono contatori hardware, F4.4): la tabella per shader caldo richiede Xcode con GUI.
- **Sfruttare**: una tabella in `docs/perf-log.md` con occupancy target e causa dominante di ogni shader caldo; intervenire sulla causa, non a tentativi.
- **Evitare**: accessi casuali a buffer grandi (stalli di memoria); texture compresse con accesso sparso (stalli di decompressione, vedi S-TEX-2).

**S-OCC-3 Threadgroup memory e occupancy**
- **Fatto**: la threadgroup memory condivide lo spazio on-chip con i registri (111375); limite API da interrogare a runtime (`maxThreadgroupMemoryLength`).
- **Misura** (B-04, B-05): `maxThreadgroupMemoryLength` = 32 KiB su M5 Max; banda threadgroup 6,2 TB/s a stride 1.
- **Sfruttare**: usare la threadgroup memory solo dove il riuso lo giustifica; preferire SIMD-group intrinsics (S-SIMD-1) per scambi dentro 32 thread.
- **Evitare**: allocazioni di threadgroup memory "per sicurezza" di dimensione massima.

---

## 3. SIMD-group, threadgroup memory, atomici (S-SIMD)

**S-SIMD-1 SIMD-group da 32**
- **Fatto**: 32 thread per SIMD-group (Turner); MSL offre `simd_shuffle`, `simd_ballot`, `simd_prefix_*`, `simd_sum/min/max`, `quad_*`.
- **Misura** (B-06, intrinseco contro emulazione in threadgroup memory): `simd_sum` 5,7×, `simd_prefix_exclusive_sum` 6,3×, `simd_ballot` 10,3×, `quad_shuffle` 1,9× più veloci; **`simd_shuffle` con lane dinamica 0,37×** (più lento dell'emulazione: anomalia da riverificare prima di usarlo nei kernel caldi).
- **Sfruttare**: compattazione (liste di meshlet visibili, pixel da ombreggiare), riduzioni (Hi-Z, istogramma di luminanza), prefix sum per l'allocazione, votazioni per uniformità (salto di rami).
- **Evitare**: atomici su un contatore globale per ogni thread (usare prima un prefix per SIMD-group, poi un atomico per gruppo).

**S-SIMD-2 Uniformità e divergenza**
- **Fatto**: divergenza dentro il SIMD-group serializza i rami (comportamento SIMT standard). **Misurato** (B-06): if/else con 25% o 50% delle lane nel ramo = 1,85× il tempo del caso uniforme (entrambi i rami eseguiti); if senza else con 0% delle lane = 0,18× (il ramo saltato non costa).
- **Misura**: B-06 (sopra).
- **Sfruttare**: binning per materiale/classe prima dello shading (V-buffer resolve per tile classificate), `simd_all/any` per saltare lavoro.
- **Evitare**: uber-shader con rami per materiale dentro lo stesso SIMD-group.

**S-SIMD-3 Threadgroup memory**
- **Misura** (B-05): 6,2 TB/s a stride 1, 1,2 TB/s a stride 32 (conflitti di banco: 5,2× più lento), 3,5 TB/s a stride 33 (32 banchi da 4 B coerenti); latenza 33 ns.
- **Sfruttare**: tile di dati riusati (filtri separabili, blocchi di GEMM, liste di luci per tile).
- **Evitare**: stride che causano conflitti di banco (misurare quali), barriere di threadgroup superflue.

**S-SIMD-4 Atomici**
- **Fatto**: atomici a 64 bit completi da Apple9; su M2 solo min/max a 64 bit, su M1 assenti (Feature Set Tables).
- **Misura** (B-07): device su 1 indirizzo 1,6 Gop/s, 32 indirizzi 6,3 (12,9 non spaziati su 128 B), per thread 204 Gop/s (137× la contesa totale); threadgroup su 1 indirizzo 104 Gop/s, 32+ indirizzi 828. 64 bit in MSL 4: solo `atomic_max`/`atomic_min` (senza valore restituito) su device; add/exchange/threadgroup a 64 bit non compilano.
- **Sfruttare**: raster software nel V-buffer con `atomic_max` 64 bit (profondità|ID) su Apple9; atomici gerarchici (SIMD → threadgroup → device).
- **Evitare**: contesa alta su pochi indirizzi; atomici device dove basta threadgroup.

---

## 4. Tile memory, TBDR, HSR, parameter buffer (S-TBDR)

**S-TBDR-1 Tile memory e imageblock**
- **Fatto**: banda "molte volte" superiore alla DRAM, latenza molto inferiore, energia "significativamente" minore; gli imageblock sono strutture per pixel che persistono per tutta la vita della tile tra draw e dispatch; i tile shader girano dentro il render pass condividendo la tile memory (Apple, doc TBDR). Dimensione non pubblicata: da interrogare (`imageblockSampleLength`) e misurare.
- **Misura** (B-15, OPT-0.6): imageblock esplicito massimo **24 B/pixel con tile 32×32** (tile memory 32 KiB = `maxThreadgroupMemoryLength`), **56 B/pixel con 32×16 e 16×16** (limite di 64 B per campione, riportati = espliciti + 8); tile 16×8 e 8×8 rifiutate dal descrittore del pass. **Oltre il limite la pipeline si crea e il pass termina senza errori ma il tile kernel non gira**: il limite va verificato a priori (lo segnala solo il layer di validazione).
- **Sfruttare**: G-buffer, accumulo di luce, OIT, liste di luci per tile **dentro la tile**; tile size scelte per il carico (es. 32×32 per il light culling).
- **Evitare**: scrivere in DRAM dati che il pass successivo rilegge subito.

**S-TBDR-2 Hidden Surface Removal**
- **Fatto**: con HSR massimizzato il depth prepass è ridondante; ordine opachi → feedback (alpha test, discard, depth write) → traslucidi; scrivere tutti i canali degli attachment per evitare il write-masking che disattiva l'HSR; `[[early_fragment_tests]]` negli shader che scrivono in memoria (WWDC20-10632).
- **Misura** (B-13, 1080p, fragment costoso): opachi 1→16 strati 0,12→0,15 ms in **entrambi gli ordini** (HSR rimuove anche back-to-front); blending back-to-front 0,14→1,26 ms (lineare); `discard` 4–8,5× (disattiva l'HSR, peggio in back-to-front).
- **Sfruttare**: nessun depth prepass; draw ordinate per classe; alpha test in un pass separato dopo gli opachi.
- **Evitare**: discard negli shader opachi; blending o depth write dallo shader mescolati agli opachi.

**S-TBDR-3 Parameter buffer e partial render**
- **Fatto**: il tiler conserva i vertici trasformati nel parameter buffer; quando si riempie, la GPU fa un *partial render* (flush delle tile a metà pass) con perdita di prestazioni (Rosenzweig). Il V-buffer riduce l'uso del parameter buffer (Apple 111431). Soglia non documentata.
- **Misura** (B-12, OPT-0.6): **nessun partial render rilevabile fino a 16,7 M triangoli** per pass (6 px ciascuno): 2,0–2,2 ns/triangolo senza varyings, 4,1–4,6 con 16 float4, curve piatte. La soglia è oltre questa scala (o non produce un salto nei tempi); contatori non disponibili headless.
- **Sfruttare**: culling aggressivo prima del raster (mesh shader, occlusione), meshlet con output minimi, attributi minimi nel pass di visibilità (solo ID).
- **Evitare**: attributi pesanti interpolati nel pass di raster; geometria densa non culled.

**S-TBDR-4 Load/store e memoryless**
- **Fatto**: load/store "consumano la maggior parte della banda di sistema"; clear nella load action; `.dontCare` per ciò che non serve dopo; `memoryless` per attachment usati solo nel pass (solo texture); unire pass adiacenti con gli stessi attachment (WWDC20-10632).
- **Misura** (B-14): un render pass piccolo costa **~6 µs oppure ~60 µs (bimodale)** tra due encoder compute, ~11 µs in catena; store RGBA32F ~0,8 TB/s efficaci (fit su 4 risoluzioni), store RGBA8 nascosto dallo shading; dontCare ≤ store sempre; memoryless ≈ private dontCare. Il load serve solo con un draw che non copre tutto (con copertura piena il driver lo elude).
- **Sfruttare**: il render graph (F2) deduce load/store e memoryless; depth, MSAA, G-buffer on-tile sempre memoryless.
- **Evitare**: pass multipli che si passano la stessa texture via DRAM; `.store` per default.

**S-TBDR-5 Barriere nello stadio fragment**
- **Fatto**: "operazione molto costosa" perché svuota la tile memory in memoria di sistema (WWDC20-10632); in Metal 4 le barriere con fragment/tile nel lato "after" non sono supportate da Apple3 ad Apple10 (Feature Set Tables).
- **Misurato (F2.3, M5 Max, 2026-09-29, `bench/barrier_spike`)**: una barriera di coda con Fragment sul lato consumatore di un render encoder è legale ed efficace, e costa ~16% meno che attendere in Vertex; Tile è accettato ma non sincronizza; dentro un render encoder il lato produttore di `barrierAfterEncoderStages` può essere solo Vertex/Object/Mesh (fragment/tile = abort della validazione). Il divieto riguarda quindi le barriere *dentro* il pass, non l'attesa di un pass successivo.
- **Sfruttare**: organizzare le dipendenze in modo che il consumo avvenga in compute o nel pass successivo.
- **Evitare**: qualunque sincronizzazione dentro il fragment.

**S-TBDR-6 Sovrapposizione tra pass**
- **Fatto**: il vertex stage di un pass successivo può partire mentre il fragment del pass precedente finisce in tile memory (doc TBDR).
- **Misura** (B-19): due pass render indipendenti (fragment-heavy + vertex-heavy) = 0,84 della somma; con barriera = 1,00; compute su seconda coda insieme a un render pass: 0,98 (ALU) e 0,78 (banda) della somma; latenza di un evento tra code ~0,07 ms.
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
- **Misura** (B-10, 8192², point): coerente BC7 318, ASTC 4×4 322, RGBA8 155, RGBA16F 83 Gtexel/s; sparso (coordinate hash) 13–14 (compresse) contro 2,3–2,7 (non compresse): le compresse leggono meno byte per texel.
- **Sfruttare**: ASTC per contenuti (qualità/bit regolabile), BC7 dove serve compatibilità degli strumenti; R11G11B10F / RGB9E5 per HDR intermedi dove la precisione basta.
- **Evitare**: RGBA16F ovunque "per sicurezza".

**S-TEX-2 Compressione lossless della GPU**
- **Fatto**: texture private popolate via blit sono compresse dalla GPU; `replaceRegion()` dalla CPU salta la compressione; da M5 anche le texture scritte dagli shader sono compresse ("universal compression"); per accessi sparsi conviene disattivarla (`allowGPUOptimizedContents = false`) per evitare overfetch di blocco; contatori Compression Ratio e Compressed Texture Write Inefficiency (111431).
- **Misura** (B-11): il rapporto di compressione **non è osservabile** (nessun contatore headless, F4.4); effetto indiretto su M5 Max: scrittura compute con `allowGPUOptimizedContents` 1,43× (contenuto costante) / 1,24× (liscio) / 0,94× (casuale) rispetto al piano, lettura 1,54× / 1,53× / 0,90×; scritture parziali (50% di ogni blocco 4×4) 1,35–1,69×.
- **Sfruttare**: scritture a blocchi interi (tile allineate) negli output compute su M5; blit per ogni caricamento.
- **Evitare**: scritture parziali di blocco (read-modify-write); compressione su texture ad accesso casuale (tabelle, atlanti di probe consultati sparsamente → da misurare).

**S-TEX-3 Mip e cache delle texture**
- **Sfruttare**: mip bias corretto per la risoluzione di render (MetalFX), mip più bassi per effetti a bassa frequenza (GI, riflessioni ruvide), campionamento coerente (ordinare i lavori per regione di texture).
- **Evitare**: campionare il mip 0 in effetti che non lo richiedono (stalli di decompressione su M5).

**S-TEX-4 Texture sparse**
- **Fatto**: risorse placement sparse con pagine da 16 o 64 KB su heap placement, mapping sulla coda (Metal 4); texture fino a 32K su M5.
- **Misura**: non coperta in OPT-0 (B-10 misura l'accesso sparso a una texture normale, non le texture sparse); da misurare con F22.
- **Sfruttare**: virtual texturing, shadow map virtuali se servissero, residency fine di cluster e texture.

---

## 6. Geometria: mesh shader, ICB, raster (S-GEO)

**S-GEO-1 Mesh shader**
- **Fatto**: da Apple9 i threadgroup object/mesh sono schedulati per tenere i meshlet on-chip; griglia mesh > 1M threadgroup; dichiarare vertici/primitive massimi solo quanto serve; omettere le primitive scartate invece di affidarsi al culling hardware successivo (111375). Payload fino a 16 KB. RT e function pointer nelle render pipeline sono incompatibili con il mesh shading (Feature Set Tables).
- **Misura** (B-16, 5,6 M triangoli da 0,7 px): vertex 8,5 Gtri/s, mesh 8,2–9,5 con meshlet 32–256 vertici (limite raster, non geometria); senza raster (back-face) vertex 22, mesh 15 (meshlet da 32) → 21 (256); payload dell'object shader 1 KiB e 16 KiB senza costo misurabile; culling della metà dei meshlet nell'object shader 1,8–1,9×.
- **Sfruttare**: dimensione dei meshlet scelta dalle misure (64/96/128); culling nell'object shader; output minimi.
- **Evitare**: massimi sovradimensionati (più traffico, meno occupancy); RT nel mesh shader (impossibile).

**S-GEO-2 Throughput geometrico**
- **Fatto**: M5 raddoppia il throughput geometrico rispetto a M4 (111431).
- **Sfruttare**: su M5 soglie di LOD più fini e meno raster software; la soglia HW/SW del raster va calibrata **per generazione**.

**S-GEO-3 Dati di vertice**
- **Sfruttare**: posizioni quantizzate relative al cluster, normali ottaedriche, UV in `half`; attributi letti solo nel resolve (V-buffer) → il pass di raster tocca solo le posizioni.

**S-GEO-4 Indirect command buffer**
- **Fatto**: ICB codificabili dalla GPU; draw mesh negli ICB da Apple9; su Apple10 stato raster, depth/stencil, cull e winding impostabili per draw dalla GPU (111431).
- **Misura** (B-17): dispatch vuoto 0,13 µs (0,82 con barriera), indiretto 0,13, catena indiretta con barriera 1,4 µs; ICB codificato dalla GPU 0,14 ns per comando, eseguito 0,043 µs per draw (draw diretto 0,027); codifica GPU + esecuzione nello stesso command buffer verificata.
- **Sfruttare**: submission interamente GPU-driven (O8); su M5 ICB unici per pass d'ombra con materiali misti.

**S-GEO-5 Funzioni specifiche Apple10**
- **Fatto**: valori per-vertex non interpolati accessibili nel fragment (utile al V-buffer), depth bounds test, riduzione min/max nel sampler (Hi-Z in un fetch), LOD bias (111431, Feature Set Tables).
- **Sfruttare**: percorsi dedicati Apple10 con fallback Apple9, attivati dal rilevamento della famiglia.

---

## 7. Ray tracing (S-RT)

**S-RT-1 Unità RT e reorder**
- **Fatto**: traversal in hardware da M3 con uno stadio di reorder che raggruppa le chiamate alle intersection function; usare l'API `intersector` perché `intersection_query` disattiva il reorder; evitare intersection function "uber" e payload grandi (111375). M4 dichiara RT 2x più veloce (Apple Newsroom).
- **Misura** (B-20, 1,2 M triangoli, 4,2 M raggi): `intersector` 8,8 Grays/s coerenti, 5,4 incoerenti; `intersection_query` 4,6 / 2,9 (~1,9× più lento); intersection function (alpha test 25%) −5…−15%.
- **Sfruttare**: `intersector` ovunque; payload minimi; intersection function specializzate e brevi.
- **Evitare**: `intersection_query` nei kernel caldi; alpha test complesso nelle intersection function.

**S-RT-2 Novità M5 (terza generazione)**
- **Fatto**: trasformazioni d'istanza in hardware; indicizzazione delle intersection function buffer interamente hardware (fino a −70% di tempo rispetto all'emulazione); allineamento delle acceleration structure da 16 KB a 1 KB; chiedere limiti estesi solo se servono; scegliere bene istanze statiche vs motion (111431).
- **Sfruttare**: su M5 molte istanze piccole costano poco (vegetazione, folle); su M3/M4 unire le istanze piccole in BLAS più grandi (re-braiding, OPT-5).
- **Evitare**: su M3/M4 migliaia di BLAS minuscole (spreco di 16 KB di padding ciascuna).

**S-RT-3 Coerenza dei raggi**
- **Fatto**: Metal non ha shader execution reordering generale (report).
- **Misura** (B-20): coerenti 1,64× più veloci degli incoerenti.
- **Sfruttare**: ordinamento software dei raggi per direzione/origine prima del trace, raggi "corti" dove possibile, geometria proxy semplice.

**S-RT-4 Build e aggiornamento**
- **Misura** (B-21): BLAS 10k/100k/1M triangoli 0,5/0,75/4,7 ms (costo fisso ~0,5 ms), refit 4,7–7,6× più veloce del rebuild, compaction a 0,48 della dimensione; TLAS 1k/10k/100k istanze 0,34/0,47/1,53 ms.
- **Sfruttare**: refit per animazioni piccole, rebuild ammortizzato su più frame, compaction per le BLAS statiche, build guidate da indirizzo (Apple9).

---

## 8. Neural Accelerator (S-NA)

**S-NA-1 Throughput e forma del lavoro**
- **Fatto**: un Neural Accelerator per core GPU da M5; ~1.024 FLOP FP16 e ~2.048 op INT8 per core per ciclo; tile migliori ≥ 32×32; il picco supera ciò che la banda può alimentare; carichi piccoli o frammentati crollano sotto il picco (tzakharko). Una misura indipendente su M5 Max indica ~19,9 TFLOPS reali su GEMM FP16 grandi (Creative Strategies, da verificare).
- **Misura** (B-22): matmul2d **FP16/BF16 58 TFLOPS, INT8 114 TOPS, INT4 92, FP8 62** (MSL 4.1) con il tile migliore (128×64 o 64×64); 64×32 con 4 SIMD-group solo 20–31 TFLOPS; `simdgroup_matrix` FP16 15 TFLOPS; MLP 64→64→64→3 fuso 1,53× la versione simdgroup. **Correzione**: i 19,9 TFLOPS di terzi dipendono dalla configurazione.
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
- **Misura** (B-08): lettura GPU sostenuta **572 GB/s** (93% dei 614 dichiarati), scrittura 486, copia 535 (lettura + scrittura).
- **Sfruttare**: budget di byte per frame per tier (O1): circa 10 GB/frame a 60 fps su M5 Max, **condivisi con la CPU**; la metà su Pro, un quarto su base.

**S-MEM-2 System Level Cache (SLC)**
- **Fatto**: 32 MB di SLC condivisa CPU/GPU su base M5 (Michael's Tinkerings); dimensioni su Pro/Max non pubblicate.
- **Misura** (B-08): banda on-chip ~8,6 TB/s fino a ~32 MiB, discesa tra 48 e 128 MiB, sopra la DRAM fino a ~384 MiB (cache resistente al thrashing); **stima dal fit h = C/WS: ~71 MiB** (modello, residuo 12%, non una misura diretta). Latenza: 30 ns (≤128 KiB), ~330–350 ns (4–64 MiB), 480–920 ns in DRAM secondo lo stato del fabric GPU (AFR, IOReport).
- **Sfruttare**: dimensionare i working set dei pass compute (tile di lavoro, liste, reservoir) per restare in SLC; ordinare i lavori per località (Morton) per massimizzare i riusi.

**S-MEM-3 Storage mode**
- **Fatto**: `shared` zero-copy, `private` solo GPU con layout ottimizzati e compressione, `memoryless` solo tile (Apple, storage modes).
- **Sfruttare**: `private` per tutto ciò che la GPU legge spesso; `shared` + write-combined per gli anelli di upload; `memoryless` per gli intermedi di pass.

**S-MEM-4 Memoria unificata**
- **Misura** (B-09): CPU e GPU condividono un solo budget di ~550–570 GB/s: 2/6/12 thread memcpy portano la lettura GPU da 573 a 440/335/330 GB/s, totale sempre ~551–553 GB/s.
- **Sfruttare**: la CPU scrive direttamente nei buffer GPU (animazione, decompressione, dati di gioco) senza copie; lavori leggeri spostabili tra CPU e GPU a seconda del carico; dataset enormi residenti (128 GB).
- **Evitare**: CPU che satura la banda durante i pass GPU più pesanti (pianificare i job CPU pesanti fuori dalle finestre critiche).

**S-MEM-5 Hazard tracking e residency**
- **Fatto**: Metal 4 non traccia gli hazard e i command buffer non rendono residenti né trattengono le risorse; massimo 32 residency set per coda (note Metal 4).
- **Sfruttare**: pochi residency set grandi; commit in batch; aggiornamenti incrementali per lo streaming.

**S-MEM-6 Fusion Architecture (M5 Pro/Max)**
- **Fatto**: M5 Pro/Max sono due die uniti (Apple Newsroom).
- **Misura**: **non misurabile qui**: l'API non permette di scegliere il core GPU (né il die) su cui gira un threadgroup; il chase a thread singolo di B-08 non ha mostrato bimodalità attribuibile a un die (la sua variabilità segue lo stato AFR). Resta un'ipotesi.
- **Sfruttare**: se esistono asimmetrie, località dei dati per metà del chip (da decidere solo dopo la misura).

---

## 10. Sincronizzazione e scheduling GPU (S-SYNC)

**S-SYNC-1 Costo delle barriere**
- **Fatto**: barriere Metal 4 per coppie di stage, intra-encoder o di coda (note Metal 4).
- **Misura** (B-18): barriera d'encoder 0,70 µs; barriere di coda ≤ 1 µs tra dispatch/blit/fragment → dispatch/vertex/fragment (dispatch→dispatch 0,97), **~11 µs tra render pass** (fragment/vertex → vertex/fragment: 10,8–11,8); senza barriera le corse si osservano su tutte le 10 coppie.
- **Sfruttare**: il render graph usa una tabella di costi misurata per scegliere tipo e posizione delle barriere, e le raggruppa.

**S-SYNC-2 Parallelismo tra pass e code**
- **Misura**: B-19 (vedi S-TBDR-6).
- **Sfruttare**: seconda coda MTL4 per compute asincrono (GI, streaming, build di BVH) sincronizzata con eventi.

**S-SYNC-3 Overhead di dispatch**
- **Misura** (B-17): dispatch vuoto 0,13 µs, con barriera 0,82 µs.
- **Sfruttare**: fondere dispatch piccoli; catene di dispatch indiretti per il lavoro variabile (sostituto dei work graph).

**S-SYNC-4 Latenza CPU↔GPU**
- **Misura** (B-28): commit quasi vuoto 3,4 µs di GPU; dalla `commit` all'evento visto dalla CPU 168 µs p50 (spin 141, listener 200); 1 ms di lavoro aggiunge ~1 ms.
- **Sfruttare**: numero di frame in volo scelto sulla base della latenza misurata e del target di input lag.

---

## 11. CPU (S-CPU)

**S-CPU-1 Core eterogenei**
- **Fatto**: M5 Pro/Max: CPU a 18 core, 6 "super" + 12 performance (Apple Newsroom); le classi QoS determinano il core; i thread in QoS background sono confinati sugli E-core; nessun pinning esplicito ai core (note middleware, Apple).
- **Misura** (B-24): 6 core "Super" + 12 "Performance" (sysctl perflevel), NEON ~86–88 GFLOPS per core con tutti i core attivi (112 con un thread); tutte le classi QoS girano alla stessa velocità a sistema scarico; risveglio di un thread 1,1 µs (user-interactive/initiated/utility) e 9,3 µs (background).
- **Sfruttare**: render thread e simulazione in QoS user-interactive; streaming e compilazione shader in QoS utility; lavori di fondo in background; work stealing.
- **Evitare**: presupporre un numero fisso di core; spin-wait (spreca energia e blocca core).

**S-CPU-2 NEON**
- **Sfruttare**: culling CPU residuo, trasformazioni, decompressione, fisica in SoA con intrinsics NEON.

**S-CPU-3 SME / AMX via Accelerate**
- **Fatto**: le unità matriciali della CPU sono raggiunte tramite Accelerate (BLAS, vDSP, BNNS). **Misurato su M5 Max** (B-24): `sysctl` riporta FEAT_SME/SME2/SME2p1 (vettore streaming 512 bit) e il compilatore accetta SME2 (`-mcpu=native+sme2`); presenza da M4: non verificabile su questo Mac.
- **Misura** (B-24): SGEMM Accelerate 2,6 TFLOPS FP32 (4096²); kernel FMOPA SME2 scritto a mano 2,2 TFLOPS su un thread; NEON 106 GFLOPS per core, scalare 33.
- **Sfruttare**: skinning CPU in batch, operazioni su matrici per folle e fisica, piccole reti su CPU.

**S-CPU-4 Latenza di scheduling**
- **Misura** (B-24): risveglio 1,1 µs (p99 3 µs) per user-interactive/user-initiated/utility, 9,3 µs per background.
- **Sfruttare**: code di lavoro persistenti invece di creare thread; job granulari ma non microscopici.

---

## 12. Neural Engine (S-ANE)

**S-ANE-1 ANE come coprocessore**
- **Fatto**: MetalFX neurale (WWDC26) usa Neural Engine **e** Neural Accelerator su M5 Pro/Max (note Metal 4); l'ANE è separato dai core GPU.
- **Misura** (B-23, modello scritto in codice, 4 conv 3×3 256→256 su 64×64): ANE 15,7 TFLOPS efficaci, 1,24 ms; GPU via Core ML 13,9, CPU 1,5 TFLOPS; `MLComputePlan` conferma l'ANE per layer. **Con il GPU carico la latenza ANE raddoppia (×2,08)**, il GPU non rallenta (0,995).
- **Sfruttare**: reti che non devono stare nel frame GPU (animazione appresa, audio, IA dei personaggi, previsione dello streaming) su ANE tramite Core ML, lasciando la GPU libera.
- **Evitare**: reti sul percorso critico del frame se la latenza di ANE non è compatibile (misurare).

---

## 13. Display e media (S-DISP)

**S-DISP-1 Presentazione**
- **Fatto**: frame interpolation MetalFX richiede un present thread dedicato e `CAMetalDisplayLink`; target di frame pacing a 2 bucket nell'istogramma del Performance HUD (WWDC25-211, WWDC26-357).
- **Misura** (B-26, `CAMetalDisplayLink` a 120 Hz): jitter di presentazione ~0,04 µs, presentato = target; anticipo callback → presentazione **41,6 ms** in finestra e a schermo intero, **16,3 ms** in finestra borderless grande quanto lo schermo; un carico GPU oltre il budget (9 ms) → 106–111 Hz. Latenza input→fotoni: serve un sensore, non misurata.
- **Sfruttare**: ProMotion 120 Hz e VRR; frame pacing prima del frame rate.

**S-DISP-2 EDR / XDR**
- **Sfruttare**: uscita EDR con headroom interrogato a runtime; tonemapping consapevole della luminanza del display; UI in SDR corretta.

**S-DISP-3 Media engine** **(ipotesi, da verificare: nessun B-xx la copre in OPT-0)**
- **Sfruttare**: video in gioco (cinematiche, schermi nel mondo) decodificati dal media engine via VideoToolbox, importati senza copie come texture Metal (IOSurface), senza toccare CPU o shader.

---

## 14. Storage e I/O (S-IO)

**S-IO-1 SSD**
- **Fatto**: base M5 ~6,3 GB/s in lettura; ~13–15 GB/s riportati su un M5 Max (fonte singola); velocità dipendente dalla capacità (note GPU).
- **Misura** (B-25, MTLIO): SSD a freddo 3,4 GB/s (non compresso, richieste 1–16 MiB, 1 coda; 4 code non aiutano), dalla page cache 62 GB/s. I 13–15 GB/s di terzi non sono riprodotti con MTLIO su un file da 128 MiB.
- **Sfruttare**: budget di streaming per tier e per capacità del disco rilevata.

**S-IO-2 MTLIO**
- **Fatto**: caricamento diretto in buffer/texture con decompressione (zlib, LZFSE, LZ4, LZMA, LZBITMAP o codec custom); non chiarito se la decompressione sia su CPU o hardware dedicato (note Metal 4).
- **Misura** (B-25): decompressione lz4 2,5, lzbitmap 2,9, lzfse 1,4, zlib 0,53, lzma 0,10 GB/s (a caldo); tempo CPU per GB decompresso 0,5 s (lz4), 0,8 (lzfse), 2,0 (zlib) contro 0,06 non compresso: **la decompressione MTLIO gira sulla CPU**.
- **Sfruttare**: scegliere il codec per tipo di dato in base alle misure; richieste grandi e allineate; code parallele.

---

## 15. Energia e termica (S-PWR)

**S-PWR-1 Consumi di riferimento**
- **Fatto**: M5 Max GPU ~75 W TDP; ~109 W di sistema medi in Cyberpunk (Notebookcheck).
- **Misura** (B-27, IOReport "GPU Energy", senza sudo): GPU ~63 W su un carico FMA a pieno clock (1620 MHz), 8 pJ per FMA; idle 0,03 W; il carico CPU su tutti i core non rallenta il GPU. Watt CPU/DRAM per fase non disponibili senza `powermetrics` (sudo): gli Energy Model di IOReport si aggiornano ogni ~5 min.

**S-PWR-2 Regime termico**
- **Misura**: B-27 (prestazioni dopo 10/30 minuti; MacBook Air senza ventola vs Pro vs Studio).
- **Sfruttare**: preset che tengono conto del regime termico, non dei primi secondi; riduzione progressiva della qualità prima del throttling.

**S-PWR-3 Efficienza**
- **Sfruttare**: frame cap intelligente, modalità batteria, meno banda = meno energia (la tile memory costa molto meno della DRAM, S-TBDR-1), E-core per i job non critici.
- **Evitare**: spin-wait della CPU; lavoro GPU speculativo non usato.

---

## 16. Cosa cambia per generazione

| Aspetto | M3 (Apple9) | M4 (Apple9) | M5 (Apple10) | Misurato su M5 Max (OPT-0) |
|---|---|---|---|---|
| Registri | Dynamic caching gen 1 | come M3 | Gen 2 + Occupancy Management Unit | throughput stabile fino a 120 valori FP32 vivi, crollo a 128 (B-04) |
| FP16 | dual-issue | come M3 | throughput 2x | 1,85× FP32 solo con 32 catene indipendenti (1,15× con 8); issue concorrente FP16+FP32+INT 0,71 della somma (B-01, B-02) |
| Geometria | mesh shader HW | come M3 | throughput 2x | 8,5–9,5 Gtri/s vertex e mesh (limite raster), front-end ~22 Gtri/s (B-16); il 2x rispetto a M4 non è verificabile senza un M4 |
| Ray tracing | HW + reorder, allineamento AS 16 KB | "RT 2x" | istanze HW, IFB HW, allineamento 1 KB | 8,8 / 5,4 Grays/s coerenti / incoerenti; `intersection_query` ~1,9× più lento (B-20) |
| Compressione | texture non scritte da shader | come M3 | universale (anche scritte da shader) | scritture compute su texture ottimizzate 1,24–1,43× più veloci con contenuto comprimibile, 0,94× casuale; rapporto non osservabile (B-11) |
| ML | TensorOps su ALU | come M3 | Neural Accelerator per core | matmul2d FP16 58 TFLOPS, INT8 114 TOPS (B-22); ANE 15,7 TFLOPS efficaci (B-23) |
| Extra | — | — | depth bounds, sampler min/max, ICB estesi, MSAA 8x, texture 32K | imageblock max 24 B/pixel (tile 32×32) / 56 B (32×16, 16×16) (B-15); nessun partial render fino a 16,7 M triangoli (B-12) |

Le colonne M3/M4 sono fonti esterne (Apple): **nessun Mac Apple9 (T0) è
stato misurato** in OPT-0; la suite gira anche con i soli percorsi Apple9
(`--force-family apple9`, [`m5max-macos27.2-apple9paths.json`](../bench/results/m5max-macos27.2-apple9paths.json))
ma su hardware M5 Max. Il modello per chip è in [`soc-model.md`](soc-model.md).

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
