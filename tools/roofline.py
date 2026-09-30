#!/usr/bin/env python3
"""roofline.py -- roofline SVG of the engine's timed units (OPT-0.3, [R4]).

  tools/roofline.py --pred pred.json [--pred more.json ...] [--out-dir docs/img]
  tools/roofline.py --self-test

`pred.json` is the output of `soc_model predict`.  One SVG per chip
(docs/img/roofline-<slug>.svg): log-log axes, arithmetic intensity (FLOP/byte
of DRAM traffic) against TFLOPS; roofs FP32 (solid), FP16 (dashed), DRAM and
on-chip bandwidth, with the ridge points labelled; one point per timed unit
(achieved FLOPS = flops / measured p50, AI = flops / DRAM bytes).  A markdown
table with predicted vs measured time (docs/img/roofline-<slug>.md) is written
next to it.  The points use the STATIC AIR op counts of soc_model predict (see
its notes): they are estimates, not counter readings.
Python 3 standard library only.
"""
import argparse
import collections
import json
import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

W, H = 900, 620
ML, MR, MT, MB = 80, 30, 50, 70
PALETTE = ["#2a6fdb", "#d9541e", "#1a9850", "#8e44ad", "#c99a00", "#00838f", "#b03060", "#555555"]


def _num(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def build_svg(pred):
    roofs = pred.get("roofs", {})
    f32, f16 = roofs.get("f32_tflops"), roofs.get("f16_tflops")
    dram, onchip = roofs.get("dram_gbps"), roofs.get("onchip_gbps")
    peak = max([v for v in (f32, f16) if _num(v)] or [1.0])
    pts = []
    for u in pred.get("units", []):
        ai, ach = u.get("arithmetic_intensity"), u.get("achieved_tflops")
        if _num(ai) and _num(ach) and ai > 0 and ach > 0:
            pts.append((u["name"], ai, ach))
    xmin, xmax = 0.01, 1000.0
    ymin = 0.001
    for _, ai, ach in pts:
        xmin = min(xmin, ai / 2)
        xmax = max(xmax, ai * 2)
        ymin = min(ymin, ach / 2)
    ymax = peak * 3
    lx0, lx1 = math.log10(xmin), math.log10(xmax)
    ly0, ly1 = math.log10(ymin), math.log10(ymax)
    pw, ph = W - ML - MR, H - MT - MB

    def X(x):
        return ML + (math.log10(x) - lx0) / (lx1 - lx0) * pw

    def Y(y):
        y = min(max(y, ymin), ymax)
        return MT + (1 - (math.log10(y) - ly0) / (ly1 - ly0)) * ph

    o = []
    a = o.append
    a('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" '
      'font-family="Helvetica, Arial, sans-serif" font-size="12">' % (W, H, W, H))
    a('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H))
    a('<text x="%d" y="26" font-size="16" font-weight="bold">Roofline — %s</text>'
      % (ML, escape(pred.get("chip", "?"))))
    benches = str(pred.get("bench", ""))
    if benches.count(",") >= 2:
        benches = "%d testbench" % (benches.count(",") + 1)
    a('<text x="%d" y="42" fill="#555">%s, %sx%s — punti: unità cronometrate del motore; FLOP = cammino minimo AIR (limite inferiore)</text>'
      % (ML, escape(benches), pred.get("width", "?"), pred.get("height", "?")))
    # grid and ticks
    for e in range(int(math.floor(lx0)), int(math.ceil(lx1)) + 1):
        x = 10.0 ** e
        if xmin <= x <= xmax:
            a('<line class="grid" x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#e3e3e3"/>' % (X(x), MT, X(x), MT + ph))
            a('<text x="%.1f" y="%d" text-anchor="middle">%g</text>' % (X(x), MT + ph + 16, x))
    for e in range(int(math.floor(ly0)), int(math.ceil(ly1)) + 1):
        y = 10.0 ** e
        if ymin <= y <= ymax:
            a('<line class="grid" x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e3e3e3"/>' % (ML, Y(y), ML + pw, Y(y)))
            a('<text x="%d" y="%.1f" text-anchor="end">%g</text>' % (ML - 6, Y(y) + 4, y))
    a('<rect x="%d" y="%d" width="%d" height="%d" fill="none" stroke="#888"/>' % (ML, MT, pw, ph))
    a('<text x="%.1f" y="%d" text-anchor="middle">Intensità aritmetica (FLOP / byte DRAM)</text>' % (ML + pw / 2, H - 28))
    a('<text transform="translate(20 %.1f) rotate(-90)" text-anchor="middle">TFLOPS</text>' % (MT + ph / 2))

    def bw_line(gbps, colour, cls, label):
        # TFLOPS = GB/s * AI / 1000, clipped to the plot
        x_a = max(xmin, ymin * 1000.0 / gbps)
        x_b = min(xmax, ymax * 1000.0 / gbps)
        if x_a >= x_b:
            return
        a('<line class="%s" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2"/>'
          % (cls, X(x_a), Y(gbps * x_a / 1000.0), X(x_b), Y(gbps * x_b / 1000.0), colour))
        xl = min(x_b, max(x_a, xmin * 3))
        a('<text x="%.1f" y="%.1f" fill="%s" transform="rotate(-27 %.1f %.1f)">%s</text>'
          % (X(xl) + 4, Y(gbps * xl / 1000.0) - 6, colour, X(xl) + 4, Y(gbps * xl / 1000.0) - 6, escape(label)))

    def flat(tf, ridge_x, colour, cls, label, dash, below=True):
        x_a = max(xmin, ridge_x) if _num(ridge_x) else xmin
        a('<line class="%s" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2"%s/>'
          % (cls, X(x_a), Y(tf), X(xmax), Y(tf), colour, ' stroke-dasharray="7 4"' if dash else ""))
        a('<text x="%.1f" y="%.1f" fill="%s" text-anchor="end">%s</text>'
          % (X(xmax) - 6, Y(tf) + (16 if below else -6), colour, escape(label)))
        if _num(ridge_x) and xmin <= ridge_x <= xmax:
            a('<circle class="ridge" cx="%.1f" cy="%.1f" r="4" fill="%s"/>' % (X(ridge_x), Y(tf), colour))
            # FP32 label below its roof, FP16 above: the two ridges are close.
            a('<text x="%.1f" y="%.1f" fill="%s">ridge %.3g F/B</text>'
              % (X(ridge_x) + 6, Y(tf) + (16 if below else -8), colour, ridge_x))

    if _num(dram) and dram > 0:
        bw_line(dram, "#b2182b", "roof-dram", "DRAM %.0f GB/s" % dram)
    if _num(onchip) and onchip > 0:
        bw_line(onchip, "#ef8a62", "roof-onchip", "on-chip %.0f GB/s" % onchip)
    if _num(f32) and f32 > 0:
        flat(f32, f32 * 1000.0 / dram if _num(dram) and dram > 0 else None, "#2166ac", "roof-fp32",
             "FP32 %.1f TFLOPS" % f32, False)
    if _num(f16) and f16 > 0:
        flat(f16, f16 * 1000.0 / dram if _num(dram) and dram > 0 else None, "#67a9cf", "roof-fp16",
             "FP16 %.1f TFLOPS" % f16, True, below=False)
    for i, (name, ai, ach) in enumerate(pts):
        c = PALETTE[i % len(PALETTE)]
        a('<circle class="unit" cx="%.1f" cy="%.1f" r="5" fill="%s" stroke="#fff"/>' % (X(ai), Y(ach), c))
        left = X(ai) > ML + pw * 0.75  # keep the label inside the plot
        a('<text class="unit-label" x="%.1f" y="%.1f" fill="%s"%s>%s</text>'
          % (X(ai) + (-8 if left else 8), Y(ach) + (16 if i % 2 == 0 else -9), c, ' text-anchor="end"' if left else "",
             escape(name)))
    if not pts:
        a('<text x="%d" y="%d" fill="#888">nessuna unità con FLOP e traffico DRAM noti</text>' % (ML + 12, MT + 24))
    a("</svg>")
    return "\n".join(o) + "\n"


def build_table(pred):
    lines = ["# Predetto vs misurato — %s (%s)" % (pred.get("chip", "?"), pred.get("bench", "")), "",
             "Il predetto è un **limite inferiore** (roofline, nessun overhead): rapporto = misurato p50 / predetto.", "",
             "| Unità | Misurato ms | Predetto ms | Limite | AI (F/B) | Rapporto | Stima ms (non un limite) |",
             "|---|---:|---:|---|---:|---:|---:|"]

    def f(v, d=3):
        return "—" if not _num(v) else ("%." + str(d) + "g") % v

    for u in pred.get("units", []):
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            u["name"].replace("|", "\\|"), f(u.get("measured_ms")), f(u.get("predicted_ms")), u.get("bound", "—"),
            f(u.get("arithmetic_intensity")), f(u.get("ratio")), f(u.get("estimate_ms"))))
    lines.append("")
    return "\n".join(lines)


def merge_preds(preds):
    """One chip, several benches: units renamed '<bench>: <unit>' (every
    bench has a 'Forward' unit), roofs from the first."""
    if len(preds) == 1:
        return preds[0]
    merged = dict(preds[0])
    merged["bench"] = ", ".join(p.get("bench", "?") for p in preds)
    merged["units"] = []
    for p in preds:
        for u in p.get("units", []):
            v = dict(u)
            v["name"] = "%s: %s" % (p.get("bench", "?"), u.get("name", "?"))
            merged["units"].append(v)
    return merged


def write_outputs(pred, out_dir):
    slug = pred.get("slug") or "chip"
    os.makedirs(out_dir, exist_ok=True)
    svg = os.path.join(out_dir, "roofline-%s.svg" % slug)
    with open(svg, "w") as f:
        f.write(build_svg(pred))
    md = os.path.join(out_dir, "roofline-%s.md" % slug)
    with open(md, "w") as f:
        f.write(build_table(pred))
    return svg, md


def synthetic_pred():
    return {"chip": "Apple Synthetic", "slug": "synthetic", "bench": "many_lights", "width": 1920, "height": 1080,
            "roofs": {"f32_tflops": 20.0, "f16_tflops": 40.0, "dram_gbps": 500.0, "onchip_gbps": 2000.0,
                      "ridge_flop_per_byte": 40.0, "ridge_onchip_flop_per_byte": 10.0},
            "units": [
                {"name": "Forward", "measured_ms": 1.2, "predicted_ms": 1.09, "bound": "alu",
                 "arithmetic_intensity": 726.0, "achieved_tflops": 18.1, "ratio": 1.10},
                {"name": "Shadow <map>", "measured_ms": 0.4, "predicted_ms": 0.1, "bound": "dram",
                 "arithmetic_intensity": 2.5, "achieved_tflops": 0.3, "ratio": 4.0},
                {"name": "No work", "measured_ms": 0.1, "predicted_ms": 0.0, "bound": "none",
                 "arithmetic_intensity": None, "achieved_tflops": None, "ratio": None}]}


def self_test():
    def check(cond, msg):
        if not cond:
            raise SystemExit("self-test FAILED: " + msg)

    svg = build_svg(synthetic_pred())
    root = ET.fromstring(svg)  # well-formed (also proves "<map>" was escaped)
    check(root.tag.endswith("svg"), "root is svg")
    ns = "{http://www.w3.org/2000/svg}"
    classes = [e.get("class") for e in root.iter() if e.get("class")]
    for c in ("roof-fp32", "roof-fp16", "roof-dram", "roof-onchip", "ridge", "unit", "unit-label", "grid"):
        check(c in classes, "missing element class " + c)
    check(classes.count("unit") == 2, "two plotted units (the one without AI is skipped): %d" % classes.count("unit"))
    fp16 = [e for e in root.iter(ns + "line") if e.get("class") == "roof-fp16"][0]
    check(fp16.get("stroke-dasharray"), "FP16 roof is dashed")
    check(len([e for e in root.iter(ns + "circle") if e.get("class") == "ridge"]) == 2, "ridge point per flat roof")
    texts = " ".join("".join(e.itertext()) for e in root.iter(ns + "text"))
    for t in ("FP32 20.0 TFLOPS", "DRAM 500 GB/s", "ridge 40 F/B", "Forward", "Shadow <map>"):
        check(t in texts, "missing text %r" % t)
    # the FP32 roof must sit above the FP16-less unit points and the DRAM roof rises with AI
    dram = [e for e in root.iter(ns + "line") if e.get("class") == "roof-dram"][0]
    check(float(dram.get("y2")) < float(dram.get("y1")), "DRAM roof slopes upward")
    tab = build_table(synthetic_pred())
    check("| Forward | 1.2 | 1.09 | alu |" in tab and "| No work |" in tab, "table rows")
    with tempfile.TemporaryDirectory() as td:
        s, m = write_outputs(synthetic_pred(), td)
        ET.parse(s)
        check(os.path.basename(s) == "roofline-synthetic.svg", "file name")
        check(os.path.getsize(m) > 100, "table written")
    empty = ET.fromstring(build_svg({"chip": "x", "units": []}))
    check(empty is not None, "renders without roofs or units")
    print("roofline.py self-test OK")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pred", action="append", help="soc_model predict JSON (repeatable)")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(here), "docs", "img"))
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.pred:
        ap.error("--pred is required")
    by_chip = collections.OrderedDict()
    for p in a.pred:
        with open(p) as f:
            pred = json.load(f)
        if pred.get("schema") != "phosphor-soc-prediction":
            raise SystemExit("%s: not a soc_model predict output" % p)
        by_chip.setdefault(pred.get("slug") or "chip", []).append(pred)
    for preds in by_chip.values():
        svg, md = write_outputs(merge_preds(preds), a.out_dir)
        print("roofline.py: wrote %s and %s (%d report(s))" % (svg, md, len(preds)))


if __name__ == "__main__":
    sys.exit(main() or 0)
