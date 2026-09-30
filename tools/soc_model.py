#!/usr/bin/env python3
"""soc_model.py -- generate docs/soc-model.md (Italian) from soc_bench results (OPT-0.3).

  tools/soc_model.py --results bench/results/m5max-macos27.2.json [--model model.json]
                     [--out docs/soc-model.md]
  tools/soc_model.py --self-test

Input: the JSON written by soc_bench (schema phosphor-soc-results, merged
runs) and, optionally, the cost-model JSON of `soc_model model` (used for the
ridge points).  Output sections: header (chip / OS / date / commit / power /
runs), the model table (value, unit, CV between runs, source benchmark), the
per-benchmark summary (status, negative control, notes, GPU top-state share),
the comparison with EXTERNAL sources (never presented as measured), and the
method.  Metrics with a between-run CV above 2% are marked **like this**.
Python 3 standard library only.
"""
import argparse
import datetime
import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CV_LIMIT = 0.02

# (key, benchmark, metric, label, unit) -- mirrors src/diagnostics/soc_model.cpp (kFields).
MODEL_FIELDS = [
    ("f32_fma_tflops", "B-01", "f32.fma.indep", "FP32 FMA (catene indipendenti)", "TFLOPS"),
    ("f16_fma_tflops", "B-01", "f16.fma.indep", "FP16 FMA (catene indipendenti)", "TFLOPS"),
    ("f32_add_tops", "B-01", "f32.add.indep", "FP32 add", "Top/s"),
    ("i32_add_tops", "B-01", "i32.add.indep", "INT32 add", "Top/s"),
    ("i32_mul_tops", "B-01", "i32.mul.indep", "INT32 mul", "Top/s"),
    ("f32_transcendental_tops", "B-03", "f32.transcendental.fast", "FP32 trascendenti (fast)", "Top/s"),
    ("f32_div_tops", "B-03", "f32.div.fast", "FP32 divisione (fast)", "Top/s"),
    ("dram_bw_gbps", "B-08", "dram_bw", "Banda DRAM (streaming)", "GB/s"),
    ("onchip_bw_gbps", "B-08", "onchip_bw", "Banda on-chip", "GB/s"),
    ("latency_l1_ns", "B-08", "latency_l1", "Latenza L1", "ns"),
    ("latency_dram_ns", "B-08", "latency_dram", "Latenza DRAM", "ns"),
    ("slc_size_mib", "B-08", "slc_size_estimate", "Dimensione SLC (stima)", "MiB"),
    ("write_bw_dram_gbps", "B-08", "write_bw.dram", "Banda di scrittura DRAM", "GB/s"),
    ("copy_bw_dram_gbps", "B-08", "copy_bw.dram", "Banda di copia DRAM", "GB/s"),
    ("empty_pass_us", "B-14", "empty_pass.us", "Render pass vuoto", "us"),
    ("dispatch_empty_us", "B-17", "dispatch.empty.us", "Dispatch vuoto", "us"),
    ("dispatch_indirect_us", "B-17", "dispatch.indirect.us", "Dispatch indiretto", "us"),
    ("barrier_encoder_us", "B-18", "barrier.encoder.us", "Barriera nell'encoder", "us"),
    ("barrier_queue_us", "B-18", "barrier.queue.us", "Barriera di coda", "us"),
    ("commit_gpu_us", "B-28", "commit.gpu_us", "Commit (lato GPU)", "us"),
    ("commit_to_cpu_us", "B-28", "commit_to_cpu.us", "Commit -> CPU", "us"),
    ("rays_coherent_grays", "B-20", "rays.coherent", "Raggi coerenti", "Grays/s"),
    ("rays_incoherent_grays", "B-20", "rays.incoherent", "Raggi incoerenti", "Grays/s"),
    ("gemm_f16_tops", "B-22", "gemm.f16.tops", "GEMM FP16", "Top/s"),
    ("gemm_bf16_tops", "B-22", "gemm.bf16.tops", "GEMM BF16", "Top/s"),
    ("gemm_i8_tops", "B-22", "gemm.i8.tops", "GEMM INT8", "Top/s"),
    ("gemm_simd_f16_tops", "B-22", "gemm.simd_f16.tops", "GEMM simdgroup FP16", "Top/s"),
]

# EXTERNAL values: NOT measured by this project.  Every row names its source.
# `value` None = not filled in (the coordinator fills it after reading the source);
# `key` links the row to a model field (measured / external ratio) or None.
# `per_core_clk`: an external per-core figure, combined with the MEASURED core count
# and top P-state to give a derived (still external-based) expectation.
# Cause of the metrics with CV between runs > 2%, per benchmark (measured or
# argued in docs/opt-log.md, "OPT-0 — suite").
CV_CAUSES = {
    "B-01": "catene dipendenti vicine alla risoluzione dei tempi",
    "B-02": "rapporti tra tempi di mix con occupancy ridotta (pochi SIMD-group): varianza dello scheduler",
    "B-03": "operazioni a basso costo (differenza tra due kernel quasi uguali)",
    "B-04": "punti oltre il thrashing (spill): tempi dipendenti dal traffico di memoria",
    "B-05": "latenza threadgroup (pochi cicli, un thread)",
    "B-07": "contesa sugli atomici: ordine di arrivo non deterministico",
    "B-08": "latenza e banda a working set nelle zone di transizione tra livelli (mapping fisico diverso a ogni "
            "run) e latenza DRAM legata allo stato del fabric GPU (AFR a P4-5 su 13 durante il chase a thread "
            "singolo, misurato con IOReport)",
    "B-09": "memcpy CPU non vincolate ai core (niente affinità su macOS) e condivise con gli altri processi",
    "B-10": "campionamento sparso di texture piccole (cache) e caratteristiche dipendenti dal layout",
    "B-11": "contenuto casuale e scritture parziali: costo della compressione dipendente dai dati",
    "B-12": "punti con pochi triangoli (tempo vicino al costo fisso del pass)",
    "B-13": "varianti con discard e pochi strati (differenze sotto 0,1 ms)",
    "B-14": "costo fisso bimodale dei render pass (~15 o ~60 us) e store nascosti dallo shading",
    "B-15": "tempi di pass tile piccoli (lavoro per pass che cresce con i byte)",
    "B-16": "throughput vincolato dal raster: frequenza del front-end e interferenza del compositor",
    "B-17": "dispatch vuoti bimodali (0,12 o 2,6 us) e catene indirette con barriera",
    "B-18": "costi di barriera di coda vicini a zero (differenze di pochi us tra varianti in parallelo)",
    "B-19": "sovrapposizione tra pass dipendente dallo scheduling",
    "B-20": "raggi incoerenti: dipendenza dalla distribuzione casuale",
    "B-22": "tile e dimensioni piccole (span brevi); FP8/INT4 con pochi campioni",
    "B-23": "tempi di compilazione/caricamento Core ML (cache del sistema) e latenze ANE di pochi ms",
    "B-24": "thread CPU senza affinità, altri processi, SME2 a un solo thread",
    "B-25": "letture fredde dall'SSD e page cache condivisa con il sistema",
    "B-26": "jitter di presentazione: composizione del WindowServer",
    "B-27": "fase idle con gli altri client del GPU (WindowServer, app) e carico misto CPU+GPU",
    "B-28": "latenze CPU di risveglio (scheduler) in microsecondi",
}

EXTERNAL = [
    # Only figures already cited by docs/APPLE_SOC_PLAYBOOK.md, with their
    # source.  `explained` = why the measurement differs (coordinator's analysis).
    {"key": "dram_bw_gbps", "label": "Banda memoria M5 Max (40 core GPU)", "value": 614.0, "unit": "GB/s",
     "source": "Apple newsroom (specifica dichiarata, picco teorico)", "ref": "Apple newsroom",
     "explained": "picco teorico contro lettura GPU sostenuta (1 GiB, float4 coalescenti): 93%; la base M5 misurata "
                  "da terzi con STREAM era all'80% (playbook S-MEM-1)"},
    {"key": "f32_fma_tflops", "label": "FP32 = core x 128 FMA/clk x 2 x MHz", "value": None,
     "unit": "TFLOPS", "per_core_clk": 128.0,
     "source": "[R7] Philip Turner, metal-benchmarks (128 ALU FP32 per core; misurato su M1/M2, non su M5)",
     "ref": "[R7]",
     "explained": "117 FMA/core/clk misurate con 8 catene: il ciclo del kernel aggiunge incremento, confronto e salto "
                  "ogni 8 FMA (8/9 = 0,89 del picco); nessuna prova di ALU FP32 diverse da 128 per core"},
    {"key": "f32_fma_tflops", "label": "FP32 per core e per clock, M5 base riportato a M5 Max", "value": None,
     "unit": "TFLOPS", "ext_tflops": 3.85, "ext_cores": 10, "ext_mhz": 1578.0,
     "source": "Michael's Tinkerings: M5 base 3,85 TFLOPS a 1.578 MHz, 10 core (convertito con i nostri 40 core e "
               "1.620 MHz)", "ref": "Michael's Tinkerings",
     "explained": "stesso throughput per core e per clock entro il 5%: architettura ALU coerente tra M5 e M5 Max"},
    {"key": "gemm_f16_tops", "label": "GEMM FP16 grande (Neural Accelerator)", "value": 19.9, "unit": "TFLOPS",
     "source": "Creative Strategies (misura di terzi su M5 Max, configurazione non pubblicata)",
     "ref": "Creative Strategies",
     "explained": "la configurazione conta: nel nostro sweep matmul2d 64x32/4 SIMD-group fa 20-31 TFLOPS, il tile "
                  "migliore (128x64) 58; lo spike iniziale con 64x32 misurava 32,2"},
    {"key": "slc_size_mib", "label": "SLC", "value": 32.0, "unit": "MiB",
     "source": "Michael's Tinkerings (M5 base, non M5 Max)", "ref": "Michael's Tinkerings",
     "explained": "chip diverso (base vs Max); il nostro valore è la stima di un modello (fit h = C/WS sulla curva di "
                  "banda), non una misura diretta: M5 Max non è documentato"},
]


def load(path):
    with open(path) as f:
        j = json.load(f)
    if j.get("schema") != "phosphor-soc-results":
        raise SystemExit("%s: not a phosphor-soc-results file" % path)
    return j


def find_metric(res, bench, name):
    for b in res.get("benchmarks", []):
        if b["id"] == bench:
            for m in b.get("metrics", []):
                if m["name"] == name:
                    return m
    return None


def run_cv(m):
    if len(m.get("runs", [])) >= 2:
        return m.get("run_cv")
    w = m.get("within", {})
    return w.get("cv") if w.get("n", 0) >= 2 else None


def fmt(v, digits=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if v == 0:
        return "0"
    return ("%." + str(digits) + "g") % v


def fmt_cv(cv):
    if cv is None:
        return "—"
    s = "%.1f%%" % (cv * 100)
    return "**%s**" % s if cv > CV_LIMIT else s


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def model_rows(res):
    rows = []
    for key, bench, name, label, unit in MODEL_FIELDS:
        m = find_metric(res, bench, name)
        if m is None:
            rows.append((key, label, None, unit, None, "%s/%s" % (bench, name)))
        else:
            rows.append((key, label, m["value"], m.get("unit", unit), run_cv(m), "%s/%s" % (bench, name)))
    return rows


def render(res, model=None):
    mach, run = res.get("machine", {}), res.get("run", {})
    out = []
    w = out.append
    w("# Modello di costo del SoC — %s" % mach.get("chip", "?"))
    w("")
    w("> Generato da `tools/soc_model.py` a partire dai risultati di `soc_bench` (OPT-0). "
      "Non modificare a mano: rigenerare. I valori nella colonna «misurato» sono misure di questo "
      "progetto; ogni numero di fonti esterne è nella sezione dedicata ed è marcato **esterno**.")
    w("")
    w("## Intestazione")
    w("")
    w("| | |")
    w("|---|---|")
    w("| Chip | %s (%s core GPU, famiglia %s) |" % (esc(mach.get("chip", "?")), mach.get("gpu_cores", "?"),
                                                    esc(mach.get("gpu_family", "?"))))
    pst = mach.get("gpu_pstate_mhz", [])
    if pst:
        w("| Stati P GPU (MHz) | %s |" % ", ".join(fmt(x, 4) for x in pst))
    w("| Sistema | %s (build %s, SDK %s) |" % (esc(mach.get("os", "?")), esc(mach.get("os_build", "?")),
                                               esc(mach.get("sdk", "?"))))
    w("| Data | %s |" % esc(run.get("date", "?")))
    w("| Commit | `%s` |" % esc(run.get("commit", "?")))
    w("| Alimentazione | %s |" % esc(mach.get("power_source", "?")))
    w("| Termica (inizio / fine) | %s / %s |" % (esc(mach.get("thermal_start", "?")), esc(mach.get("thermal_end", "?"))))
    w("| Run | %s%s |" % (run.get("runs", "?"), " (modalità quick)" if run.get("quick") else ""))
    if run.get("forced_apple9"):
        w("| Nota | eseguito con `--force-family apple9` |")
    w("")

    rows = model_rows(res)
    w("## Modello")
    w("")
    w("Valore = mediana dei run; CV = coefficiente di variazione tra i run (**grassetto** se > %d%%: "
      "misura da non citare senza riserva). «—» = non misurato." % int(CV_LIMIT * 100))
    w("")
    w("| Parametro | Valore misurato | Unità | CV tra run | Benchmark/metrica |")
    w("|---|---:|---|---:|---|")
    for key, label, value, unit, cv, src in rows:
        w("| %s | %s | %s | %s | %s |" % (esc(label), fmt(value), esc(unit), fmt_cv(cv), src))
    w("")
    tbdr = []
    for b in res.get("benchmarks", []):
        if b["id"] == "B-14":
            for m in b.get("metrics", []):
                parts = m["name"].split(".", 2)
                if len(parts) == 3 and parts[0] in ("store", "load"):
                    tbdr.append((parts[0], parts[1], parts[2], m["value"], run_cv(m)))
    if tbdr:
        w("### TBDR: banda store/load delle attachment (B-14)")
        w("")
        w("| Operazione | Formato | Risoluzione | GB/s | CV tra run |")
        w("|---|---|---|---:|---:|")
        for kind, f, r, v, cv in tbdr:
            w("| %s | %s | %s | %s | %s |" % (kind, esc(f), esc(r), fmt(v), fmt_cv(cv)))
        w("")

    vals = {k: v for k, _, v, _, _, _ in rows if v is not None}
    w("### Punti di ridge (roofline, [R4])")
    w("")
    ridge = ridge_on = None
    if model:
        d = model.get("derived", {})
        ridge, ridge_on = d.get("ridge_point_flop_per_byte"), d.get("ridge_point_onchip_flop_per_byte")
    if ridge is None and "f32_fma_tflops" in vals and "dram_bw_gbps" in vals:
        ridge = vals["f32_fma_tflops"] * 1e3 / vals["dram_bw_gbps"]
    if ridge_on is None and "f32_fma_tflops" in vals and "onchip_bw_gbps" in vals:
        ridge_on = vals["f32_fma_tflops"] * 1e3 / vals["onchip_bw_gbps"]
    w("- FP32 / DRAM: %s FLOP/byte" % fmt(ridge))
    w("- FP32 / on-chip: %s FLOP/byte" % fmt(ridge_on))
    w("")

    w("## Riepilogo per benchmark")
    w("")
    w("| ID | Nome | Stato | Controllo negativo | Note | GPU stato massimo |")
    w("|---|---|---|---|---|---:|")
    for b in res.get("benchmarks", []):
        nc = b.get("negative_control", {})
        ts = b.get("gpu", {}).get("top_state_share", -1)
        share = "%.0f%%" % (ts * 100) if ts is not None and ts >= 0 else "—"
        detail = nc.get("detail", "")
        w("| %s | %s | %s | %s%s | %s | %s |" % (b["id"], esc(b.get("name", "")), b.get("status", "?"),
                                                   nc.get("status", "?"), (": " + esc(detail)) if detail else "",
                                                   esc(b.get("notes", "")) or "—", share))
    w("")

    w("## Confronto con fonti esterne")
    w("")
    w("Le colonne «esterno» **non sono misure di questo progetto**: sono valori dichiarati o misurati da terzi, "
      "riportati con la fonte e con la spiegazione dello scarto.")
    w("")
    w("| Grandezza | Misurato (questo progetto) | Esterno (fonte) | Rapporto misurato/esterno | Scarto spiegato |")
    w("|---|---:|---|---:|---|")
    cores, top = mach.get("gpu_cores"), (max(pst) if pst else None)
    for e in EXTERNAL:
        measured = vals.get(e["key"])
        ext, note = e["value"], ""
        if ext is None and e.get("per_core_clk") and cores and top:
            ext = cores * e["per_core_clk"] * 2 * top * 1e-6
            note = " (derivato: %s core x %g x 2 x %s MHz)" % (cores, e["per_core_clk"], fmt(top, 4))
        if ext is None and e.get("ext_tflops") and cores and top:
            ext = e["ext_tflops"] / (e["ext_cores"] * e["ext_mhz"]) * cores * top
            note = " (derivato: %g TFLOPS / (%d core x %g MHz) x %s core x %s MHz)" % (
                e["ext_tflops"], e["ext_cores"], e["ext_mhz"], cores, fmt(top, 4))
        ratio = "—"
        if measured is not None and ext:
            ratio = "%.2f" % (measured / ext)
        ext_txt = "**esterno**: %s %s%s — %s" % (fmt(ext), e["unit"], note, esc(e["source"])) if ext is not None \
            else "**esterno**: valore da inserire — %s" % esc(e["source"])
        w("| %s | %s %s | %s | %s | %s |" % (esc(e["label"]), fmt(measured), e["unit"] if measured is not None else "",
                                           ext_txt, ratio, esc(e.get("explained", "—"))))
    w("")

    # Metrics with a CV between runs above 2%, grouped by benchmark, with
    # the measured cause (docs/opt-log.md, OPT-0).
    w("## Metriche con CV tra i run > 2%")
    w("")
    w("Obiettivo del protocollo: CV < 2%. Eccezioni per benchmark e causa: misurata per B-08 (stato AFR del fabric, "
      "IOReport), B-14 e B-19 (costo bimodale dei render pass), B-17 (dispatch bimodali), B-27 (altri client GPU); "
      "per gli altri benchmark è un'ipotesi da verificare quando la metrica servirà a una decisione (OPT-1/OPT-2).")
    w("")
    w("| Benchmark | Metriche con CV > 2% / totale | Causa |")
    w("|---|---:|---|")
    for b in res.get("benchmarks", []):
        ms = b.get("metrics", [])
        hi = [m for m in ms if (m.get("run_cv") or 0) > CV_LIMIT]
        if hi:
            w("| %s | %d / %d | %s |" % (b["id"], len(hi), len(ms), esc(CV_CAUSES.get(b["id"], "da analizzare"))))
    w("")

    w("## Metodo")
    w("")
    w("- Suite `bench/soc` (`soc_bench`), benchmark B-xx del [playbook](APPLE_SOC_PLAYBOOK.md) §0; "
      "protocollo e spike di misura in [`docs/opt-log.md`](opt-log.md), sezione «OPT-0».")
    w("- Ogni misura: mediana di ripetizioni singole da ~0.1–2 ms su timestamp GPU, GPU riscaldata, "
      "stato P registrato con IOReport; ogni benchmark ha un controllo negativo.")
    w("- Il modello (`src/diagnostics/soc_model.h`) è un **limite inferiore** roofline ([R4]): "
      "max(DRAM, on-chip, ALU), senza overhead fissi. Le FLOP contano FMA = 2 come in B-01.")
    w("- Predizioni contro l'engine: `soc_model predict` + `tools/roofline.py`.")
    w("")
    return "\n".join(out) + "\n"


def synthetic_results():
    def metric(name, unit, v, cv):
        return {"name": name, "unit": unit, "value": v, "higher_is_better": True,
                "within": {"median": v, "cv": 0.005, "n": 20}, "runs": [v, v, v], "run_cv": cv, "params": {}}

    return {
        "schema": "phosphor-soc-results", "schema_version": 1,
        "machine": {"chip": "Apple Synthetic", "slug": "synthetic", "gpu_family": "Apple10", "gpu_cores": 40,
                    "os": "macOS 0.0", "os_build": "X", "sdk": "0.0", "power_source": "AC",
                    "thermal_start": "nominal", "thermal_end": "nominal", "gpu_pstate_mhz": [338, 1000, 1620]},
        "run": {"commit": "deadbee", "date": "2026-01-01T00:00:00Z", "runs": 3, "quick": False},
        "benchmarks": [
            {"id": "B-01", "name": "alu.throughput", "status": "ok",
             "negative_control": {"status": "pass", "detail": "2x work = 2x time"}, "notes": "",
             "gpu": {"top_state_share": 0.98},
             "metrics": [metric("f32.fma.indep", "TFLOPS", 20.0, 0.004), metric("f16.fma.indep", "TFLOPS", 40.0, 0.031)]},
            {"id": "B-08", "name": "mem.hierarchy", "status": "partial",
             "negative_control": {"status": "pass", "detail": ""}, "notes": "SLC non risolto",
             "gpu": {"top_state_share": 0.5},
             "metrics": [metric("dram_bw", "GB/s", 500.0, 0.01)]},
            {"id": "B-14", "name": "tbdr", "status": "ok", "negative_control": {"status": "n/a", "detail": ""},
             "notes": "", "gpu": {"top_state_share": -1},
             "metrics": [metric("store.rgba8.1080p", "GB/s", 300.0, 0.01)]},
        ],
    }


def self_test():
    def check(cond, msg):
        if not cond:
            raise SystemExit("self-test FAILED: " + msg)

    md = render(synthetic_results())
    for needle in ("# Modello di costo del SoC", "## Intestazione", "## Modello", "## Riepilogo per benchmark",
                   "## Confronto con fonti esterne", "## Metodo", "opt-log.md", "TBDR: banda store/load",
                   "Apple Synthetic", "deadbee"):
        check(needle in md, "missing %r" % needle)
    check("**3.1%**" in md, "CV 3.1% must be highlighted")
    check("**0.4%**" not in md and "0.4%" in md, "CV 0.4% must not be highlighted")
    check("| Banda memoria M5 Max (40 core GPU) |" in md and "**esterno**: 614 GB/s" in md, "external comparison row")
    check("picco teorico contro lettura GPU sostenuta" in md, "external row carries its explanation")
    check("## Metriche con CV tra i run > 2%" in md, "CV causes section")
    check("1.01" in md, "ratio 20/19.9")
    check("| — |\n" in md, "scarto spiegato left empty")
    # external numbers only ever in the external column
    for line in md.splitlines():
        if line.startswith("| Banda memoria M5 Max"):
            cells = [c.strip() for c in line.split("|")]
            check("esterno" in cells[3] and "esterno" not in cells[2], "external marking: %s" % line)
    # derived per-core expectation uses measured cores x top P-state: 40*128*2*1620e6 = 16.5888 TFLOPS
    check("16.6 TFLOPS" in md, "derived external expectation")
    md2 = render(synthetic_results(), {"derived": {"ridge_point_flop_per_byte": 40.0}})
    check("FP32 / DRAM: 40 FLOP/byte" in md2, "ridge from model json")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "r.json")
        with open(p, "w") as f:
            json.dump(synthetic_results(), f)
        o = os.path.join(td, "soc-model.md")
        main(["--results", p, "--out", o])
        check(os.path.getsize(o) > 500, "output written")
    print("soc_model.py self-test OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--results")
    ap.add_argument("--model", help="model JSON from `soc_model model` (ridge points)")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "soc-model.md"))
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.results:
        ap.error("--results is required")
    res = load(a.results)
    model = None
    if a.model:
        with open(a.model) as f:
            model = json.load(f)
    md = render(res, model)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(md)
    print("soc_model.py: wrote %s" % a.out)


if __name__ == "__main__":
    sys.exit(main() or 0)
