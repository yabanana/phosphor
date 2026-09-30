# Modello di costo del SoC — Apple Synthetic

> **DATI SINTETICI DI ESEMPIO (chip «Apple Synthetic»): non sono misure. Il coordinatore rigenera questo file dai risultati reali di `bench/results/`.**

> Generato da `tools/soc_model.py` a partire dai risultati di `soc_bench` (OPT-0). Non modificare a mano: rigenerare. I valori nella colonna «misurato» sono misure di questo progetto; ogni numero di fonti esterne è nella sezione dedicata ed è marcato **esterno**.

## Intestazione

| | |
|---|---|
| Chip | Apple Synthetic (40 core GPU, famiglia Apple10) |
| Stati P GPU (MHz) | 338, 1620 |
| Sistema | macOS 0.0 (build X, SDK 0.0) |
| Data | 2026-01-01T00:00:00Z |
| Commit | `deadbee` |
| Alimentazione | AC |
| Termica (inizio / fine) | nominal / nominal |
| Run | 3 |

## Modello

Valore = mediana dei run; CV = coefficiente di variazione tra i run (**grassetto** se > 2%: misura da non citare senza riserva). «—» = non misurato.

| Parametro | Valore misurato | Unità | CV tra run | Benchmark/metrica |
|---|---:|---|---:|---|
| FP32 FMA (catene indipendenti) | 20 | TFLOPS | 0.4% | B-01/f32.fma.indep |
| FP16 FMA (catene indipendenti) | 40 | TFLOPS | **3.1%** | B-01/f16.fma.indep |
| FP32 add | — | Top/s | — | B-01/f32.add.indep |
| INT32 add | — | Top/s | — | B-01/i32.add.indep |
| INT32 mul | — | Top/s | — | B-01/i32.mul.indep |
| FP32 trascendenti (fast) | — | Top/s | — | B-03/f32.transcendental.fast |
| FP32 divisione (fast) | — | Top/s | — | B-03/f32.div.fast |
| Banda DRAM (streaming) | 500 | GB/s | 1.0% | B-08/dram_bw |
| Banda on-chip | — | GB/s | — | B-08/onchip_bw |
| Latenza L1 | — | ns | — | B-08/latency_l1 |
| Latenza DRAM | — | ns | — | B-08/latency_dram |
| Dimensione SLC (stima) | — | MiB | — | B-08/slc_size_estimate |
| Banda di scrittura DRAM | — | GB/s | — | B-08/write_bw.dram |
| Banda di copia DRAM | — | GB/s | — | B-08/copy_bw.dram |
| Render pass vuoto | — | us | — | B-14/empty_pass.us |
| Dispatch vuoto | — | us | — | B-17/dispatch.empty.us |
| Dispatch indiretto | — | us | — | B-17/dispatch.indirect.us |
| Barriera nell'encoder | — | us | — | B-18/barrier.encoder.us |
| Barriera di coda | — | us | — | B-18/barrier.queue.us |
| Commit (lato GPU) | — | us | — | B-28/commit.gpu_us |
| Commit -> CPU | — | us | — | B-28/commit_to_cpu.us |
| Raggi coerenti | — | Grays/s | — | B-20/rays.coherent |
| Raggi incoerenti | — | Grays/s | — | B-20/rays.incoherent |
| GEMM FP16 | — | Top/s | — | B-22/gemm.f16.tops |
| GEMM BF16 | — | Top/s | — | B-22/gemm.bf16.tops |
| GEMM INT8 | — | Top/s | — | B-22/gemm.i8.tops |
| GEMM simdgroup FP16 | — | Top/s | — | B-22/gemm.simd_f16.tops |

### TBDR: banda store/load delle attachment (B-14)

| Operazione | Formato | Risoluzione | GB/s | CV tra run |
|---|---|---|---:|---:|
| store | rgba8 | 1080p | 300 | 1.0% |

### Punti di ridge (roofline, [R4])

- FP32 / DRAM: 40 FLOP/byte
- FP32 / on-chip: — FLOP/byte

## Riepilogo per benchmark

| ID | Nome | Stato | Controllo negativo | Note | GPU stato massimo |
|---|---|---|---|---|---:|
| B-01 | alu.throughput | ok | pass: 2x work = 2x time | — | 98% |
| B-08 | mem.hierarchy | partial | pass | SLC non risolto | 50% |
| B-14 | tbdr | ok | n/a | — | — |

## Confronto con fonti esterne

Le colonne «esterno» **non sono misure di questo progetto**: sono valori dichiarati o misurati da terzi, riportati con la fonte. «Scarto spiegato» è da compilare dopo aver analizzato la differenza (diverso stato P, teorico vs sostenuto, altra configurazione...).

| Grandezza | Misurato (questo progetto) | Esterno (fonte) | Rapporto misurato/esterno | Scarto spiegato |
|---|---:|---|---:|---|
| Banda memoria M5 Max, 40 core GPU | 500 GB/s | **esterno**: 614 GB/s — Apple newsroom (specifica dichiarata, teorica di picco) | 0.81 | — |
| Banda memoria M5 Max, 32 core GPU | 500 GB/s | **esterno**: 460 GB/s — Apple newsroom (specifica dichiarata, teorica di picco) | 1.09 | — |
| FP32 M5 Max | 20 TFLOPS | **esterno**: 19.9 TFLOPS — Creative Strategies (stima di terzi) | 1.01 | — |
| FP32 = core x 128 FMA/clk x 2 x MHz (formula esterna) | 20 TFLOPS | **esterno**: 16.6 TFLOPS (derivato: 40 core x 128 x 2 x 1620 MHz) — [R7] Philip Turner, metal-benchmarks (128 ALU FP32 per core, misurato su M1/M2; non verificato su M5) | 1.21 | — |
| Latenza DRAM | —  | **esterno**: valore da inserire — [R7] Philip Turner, metal-benchmarks (valore da inserire) | — | — |
| Dimensione SLC | —  | **esterno**: valore da inserire — Michael's Tinkerings (valore da inserire) | — | — |
| FP32 M5 Max (recensione) | 20 TFLOPS | **esterno**: valore da inserire — Notebookcheck (valore da inserire) | — | — |

## Metodo

- Suite `bench/soc` (`soc_bench`), benchmark B-xx del [playbook](APPLE_SOC_PLAYBOOK.md) §0; protocollo e spike di misura in [`docs/opt-log.md`](opt-log.md), sezione «OPT-0».
- Ogni misura: mediana di ripetizioni singole da ~0.1–2 ms su timestamp GPU, GPU riscaldata, stato P registrato con IOReport; ogni benchmark ha un controllo negativo.
- Il modello (`src/diagnostics/soc_model.h`) è un **limite inferiore** roofline ([R4]): max(DRAM, on-chip, ALU), senza overhead fissi. Le FLOP contano FMA = 2 come in B-01.
- Predizioni contro l'engine: `soc_model predict` + `tools/roofline.py`.

