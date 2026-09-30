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
EXTERNAL = [
    {"key": "dram_bw_gbps", "label": "Banda memoria M5 Max, 40 core GPU", "value": 614.0, "unit": "GB/s",
     "source": "Apple newsroom (specifica dichiarata, teorica di picco)", "ref": "Apple newsroom"},
    {"key": "dram_bw_gbps", "label": "Banda memoria M5 Max, 32 core GPU", "value": 460.0, "unit": "GB/s",
     "source": "Apple newsroom (specifica dichiarata, teorica di picco)", "ref": "Apple newsroom"},
    {"key": "f32_fma_tflops", "label": "FP32 M5 Max", "value": 19.9, "unit": "TFLOPS",
     "source": "Creative Strategies (stima di terzi)", "ref": "Creative Strategies"},
    {"key": "f32_fma_tflops", "label": "FP32 = core x 128 FMA/clk x 2 x MHz (formula esterna)", "value": None,
     "unit": "TFLOPS", "per_core_clk": 128.0,
     "source": "[R7] Philip Turner, metal-benchmarks (128 ALU FP32 per core, misurato su M1/M2; non verificato su M5)",
     "ref": "[R7]"},
    {"key": "latency_dram_ns", "label": "Latenza DRAM", "value": None, "unit": "ns",
     "source": "[R7] Philip Turner, metal-benchmarks (valore da inserire)", "ref": "[R7]"},
    {"key": "slc_size_mib", "label": "Dimensione SLC", "value": None, "unit": "MiB",
     "source": "Michael's Tinkerings (valore da inserire)", "ref": "Michael's Tinkerings"},
    {"key": "f32_fma_tflops", "label": "FP32 M5 Max (recensione)", "value": None, "unit": "TFLOPS",
     "source": "Notebookcheck (valore da inserire)", "ref": "Notebookcheck"},
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
      "riportati con la fonte. «Scarto spiegato» è da compilare dopo aver analizzato la differenza "
      "(diverso stato P, teorico vs sostenuto, altra configurazione...).")
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
        ratio = "—"
        if measured is not None and ext:
            ratio = "%.2f" % (measured / ext)
        ext_txt = "**esterno**: %s %s%s — %s" % (fmt(ext), e["unit"], note, esc(e["source"])) if ext is not None \
            else "**esterno**: valore da inserire — %s" % esc(e["source"])
        w("| %s | %s %s | %s | %s | — |" % (esc(e["label"]), fmt(measured), e["unit"] if measured is not None else "",
                                           ext_txt, ratio))
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
    check("FP32 M5 Max | 20 TFLOPS | **esterno**: 19.9" in md, "external comparison row")
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
