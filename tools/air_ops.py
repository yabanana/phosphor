#!/usr/bin/env python3
"""air_ops.py -- static operation counts of every shader entry point (OPT-0.3).

  tools/air_ops.py [--shader-dir shaders] [--include src --include build/generated]
                   [--out bench/results/shader_ops.json] [--only forward.metal ...]
  tools/air_ops.py --ll FILE.ll                 count an already emitted AIR .ll
  tools/air_ops.py --self-test                  run on an embedded .ll (any OS)

Each shader is compiled with `xcrun metal -std=metal4.0 -O2 -S -emit-llvm`
(macOS only) and every entry point (vertex / fragment / kernel) is counted per
lane on the optimised AIR:

  flops            add/sub/mul/neg = 1, fma = 2, dot(n) = 2n, min/max/clamp/
                   saturate = 1, mix = 3, per vector lane (B-01 convention)
  transcendentals  rsqrt sqrt exp exp2 log log2 pow sin cos tan
  divides          fdiv
  int_ops          integer add/sub/mul/shift/logic
  samples          texture samples
  loads / stores   memory instructions (any address space)

Counts are STATIC: every basic block once, both sides of every branch (function
constants included).  Loops are found by back edges; a loop's `per_iteration`
is the cost of its blocks (head..tail in emission order).  Only OUTERMOST loops
are listed (back edges sharing a tail are one loop); loops nested in them are
counted in `nested_loops` and their cost stays inside the outer body, their trip
count is not modelled.  The JSON is consumed by tools/soc_model (predict).
Python 3 standard library only.
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FIELDS = ("flops", "transcendentals", "divides", "int_ops", "samples", "loads", "stores")

_TRANSC = r"(rsqrt|sqrt|exp2|log2|pow|cos|sin|exp|log|tan)(?![a-z0-9]*[a-z])"


def lanes(line):
    m = re.search(r"<(\d+) x (float|half|i32|i16)>", line)
    return int(m.group(1)) if m else 1


def cost(lines):
    """Per-lane operation counts of a list of LLVM IR lines."""
    c = collections.Counter()
    for l in lines:
        if re.search(r"= (fadd|fsub|fmul|fneg)\b", l):
            c["flops"] += lanes(l)
        elif re.search(r"= fdiv\b", l):
            c["divides"] += lanes(l)
        elif re.search(r"@(air|llvm)\.(fast_|precise_)?fma(?![a-z])", l):
            c["flops"] += 2 * lanes(l)
        elif re.search(r"@air\.(fast_|precise_)?dot\.v(\d)", l):
            c["flops"] += 2 * int(re.search(r"dot\.v(\d)", l).group(1))
        elif re.search(r"@(air|llvm)\.(fast_|precise_)?" + _TRANSC, l):
            c["transcendentals"] += lanes(l)
        elif re.search(r"@air\.(fast_|precise_)?(fmax|fmin|clamp|saturate|mix)", l):
            c["flops"] += lanes(l) * (3 if "mix" in l else 1)
        elif re.search(r"@air\.sample_", l):
            c["samples"] += 1
        elif re.search(r"= load\b", l):
            c["loads"] += 1
        elif re.search(r"^\s*store ", l):
            c["stores"] += 1
        elif re.search(r"= (add|sub|mul|shl|lshr|ashr|and|or|xor)\b", l):
            c["int_ops"] += lanes(l)
    return c


def entry_kinds(ll):
    """{function name: 'vertex'|'fragment'|'kernel'} from the air.* metadata."""
    nodes = {}
    for m in re.finditer(r"^!(\d+) = !\{.*?@(\w+),\s*!\d+", ll, re.M):
        nodes[m.group(1)] = m.group(2)
    kinds = {}
    for kind in ("vertex", "fragment", "kernel"):
        m = re.search(r"^!air\.%s = !\{(.*?)\}" % kind, ll, re.M)
        if not m:
            continue
        for ref in re.findall(r"!(\d+)", m.group(1)):
            if ref in nodes:
                kinds[nodes[ref]] = kind
    return kinds


def analyse(ll, file=""):
    """{function: entry} for the entry points of an AIR .ll text."""
    kinds = entry_kinds(ll)
    out = collections.OrderedDict()
    for fn in re.finditer(r"^define ([^\n]*?)@(\w+)\(.*?^}", ll, re.S | re.M):
        name, body = fn.group(2), fn.group(0)
        if kinds and name not in kinds:
            continue  # helper / static init
        if not kinds and "internal" in fn.group(1):
            continue
        blocks = collections.OrderedDict([("entry", [])])
        cur = "entry"
        for line in body.split("\n")[1:]:
            m = re.match(r"^(\d+):", line)
            if m:
                cur = m.group(1)
                blocks[cur] = []
                continue
            blocks[cur].append(line)
        order = list(blocks)
        idx = {b: i for i, b in enumerate(order)}
        edges = set()
        for b in order:
            for l in blocks[b]:
                for t in re.findall(r"label %(\d+)", l):
                    if t in idx and idx[t] <= idx[b]:
                        edges.add((t, b))
        # one loop per tail: the widest span (smallest head index)
        by_tail = {}
        for head, tail in edges:
            if tail not in by_tail or idx[head] < idx[by_tail[tail]]:
                by_tail[tail] = head
        spans = sorted(((idx[h], idx[t], h, t) for t, h in by_tail.items()), key=lambda s: (s[0], -s[1]))
        outer = []
        for s in spans:
            if any(o[0] <= s[0] and s[1] <= o[1] for o in outer):
                continue
            outer.append(s)
        nested = len(spans) - len(outer)
        total = collections.Counter()
        for b in order:
            total += cost(blocks[b])
        loops = []
        for hi, ti, h, t in outer:
            body_c = collections.Counter()
            for b in order[hi:ti + 1]:
                body_c += cost(blocks[b])
            loops.append({"head": h, "tail": t, "blocks": ti - hi + 1,
                          "per_iteration": {k: body_c.get(k, 0) for k in FIELDS}})
        out[name] = {"file": file, "kind": kinds.get(name, ""), "blocks": len(order),
                     "static": {k: total.get(k, 0) for k in FIELDS},
                     "loops": loops, "nested_loops": nested}
    return out


def compile_shader(path, includes):
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out.ll")
        cmd = ["xcrun", "metal", "-std=metal4.0", "-O2", "-S", "-emit-llvm"]
        for i in includes:
            cmd += ["-I", i]
        cmd += [path, "-o", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or "metal failed")
        with open(out) as f:
            return f.read()


def compiler_version():
    try:
        r = subprocess.run(["xcrun", "metal", "--version"], capture_output=True, text=True)
        return r.stdout.splitlines()[0] if r.stdout else ""
    except OSError:
        return ""


SELF_TEST_LL = r"""
define internal void @_GLOBAL__sub_I_x.metal() #5 section "air.static_init" {
  ret void
}
define <4 x half> @toy_fs(<4 x float> %0, i32 %1) local_unnamed_addr #1 {
  %3 = fadd <4 x float> %0, %0
  %4 = tail call float @air.fast_fmax.f32(float 1.0, float 2.0)
  br label %5

5:                                                ; preds = %2, %9
  %6 = tail call float @air.fma.f32(float 1.0, float 2.0, float 3.0)
  %7 = tail call float @air.fast_exp2.f32(float %6)
  %8 = fdiv float %6, %7
  br label %9

9:                                                ; preds = %5, %11
  %10 = tail call <4 x float> @air.sample_texture_2d.v4f32(i8 0)
  %11a = load i32, i32 addrspace(1)* null, align 4
  br label %11

11:                                               ; preds = %9
  %12 = add i32 %1, 1
  %13 = icmp eq i32 %12, 8
  br i1 %13, label %5, label %14

14:                                               ; preds = %11
  br i1 %13, label %9, label %15

15:                                               ; preds = %14
  store i32 0, i32 addrspace(1)* null
  ret <4 x half> zeroinitializer
}
define void @toy_kernel() {
  %1 = tail call float @air.dot.v3f32(float 1.0)
  ret void
}
!air.fragment = !{!27}
!air.kernel = !{!30}
!27 = !{<4 x half> (<4 x float>, i32)* @toy_fs, !28}
!30 = !{void ()* @toy_kernel, !28}
"""


def self_test():
    def check(cond, msg):
        if not cond:
            raise SystemExit("self-test FAILED: " + msg)

    r = analyse(SELF_TEST_LL, "toy.metal")
    check(list(r) == ["toy_fs", "toy_kernel"], "entry points only (static init skipped): %s" % list(r))
    fs = r["toy_fs"]
    check(fs["kind"] == "fragment" and r["toy_kernel"]["kind"] == "kernel", "kinds from air.* metadata")
    s = fs["static"]
    # fadd <4 x float> = 4, fmax = 1 (NOT counted as an fma), fma = 2
    check(s["flops"] == 4 + 1 + 2, "flops %s" % s["flops"])
    check(s["transcendentals"] == 1 and s["divides"] == 1, "transc/div %s" % s)
    check(s["samples"] == 1 and s["loads"] == 1 and s["stores"] == 1, "samples/loads/stores %s" % s)
    check(s["int_ops"] == 1, "int ops %s" % s)
    # back edges 11->5 and 14->9: two different tails; 9..14 is inside 5..11? 5(idx1)..11(idx3), 9..14 (idx2..4)
    # overlapping, not contained -> both outer; a strictly nested loop is exercised below.
    check(len(fs["loops"]) >= 1, "loop detected")
    body = fs["loops"][0]["per_iteration"]
    check(body["flops"] == 2 and body["transcendentals"] == 1 and body["divides"] == 1,
          "loop body cost %s" % body)
    check(r["toy_kernel"]["static"]["flops"] == 6, "dot.v3 = 2*3 flops")
    nested = SELF_TEST_LL.replace("br i1 %13, label %9, label %15", "br i1 %13, label %5, label %15")
    r2 = analyse(nested, "toy.metal")["toy_fs"]
    check(len(r2["loops"]) == 1 and r2["nested_loops"] == 1, "nested loop folded into the outer: %s" % r2)
    check(r2["loops"][0]["blocks"] == 4, "outer loop spans 5..14: %s" % r2["loops"])
    json.dumps(r)  # serialisable
    print("air_ops.py self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--shader-dir", default=os.path.join(ROOT, "shaders"))
    ap.add_argument("--include", action="append", default=None,
                    help="-I directory (repeatable; default src and build/generated)")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "results", "shader_ops.json"))
    ap.add_argument("--only", nargs="*", help="only these .metal file names")
    ap.add_argument("--ll", help="analyse an existing AIR .ll instead of compiling")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    functions = collections.OrderedDict()
    errors = {}
    if a.ll:
        with open(a.ll) as f:
            functions.update(analyse(f.read(), os.path.basename(a.ll)))
        version = ""
    else:
        includes = a.include or [os.path.join(ROOT, "src"), os.path.join(ROOT, "build", "generated")]
        version = compiler_version()
        for fn in sorted(os.listdir(a.shader_dir)):
            if not fn.endswith(".metal") or (a.only and fn not in a.only):
                continue
            try:
                ll = compile_shader(os.path.join(a.shader_dir, fn), includes)
            except (RuntimeError, OSError) as e:
                errors[fn] = str(e)[:400]
                print("air_ops: %s: FAILED: %s" % (fn, str(e)[:200]), file=sys.stderr)
                continue
            functions.update(analyse(ll, fn))
    doc = {"schema": "phosphor-shader-ops", "schema_version": 1, "compiler": version,
           "flags": "-std=metal4.0 -O2", "functions": functions, "errors": errors}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print("air_ops: %d entry points from %s -> %s" % (len(functions), a.shader_dir, a.out))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
