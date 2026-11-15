#!/usr/bin/env python3
"""Batch driver for build/<dir>/agx_link_stress (F3.2 / F15 ticket 24).

Runs N child processes per configuration (a crash kills only the child) and
prints one tally line per configuration: exit 0 = all compiled, 1 = a compile
failed, negative = killed by that signal.  Holds the shared GPU lock around
every child when --lock is given (one Metal process at a time on this machine).

usage: agx_link_stress.py BUILD_DIR --runs N --config "T=18 SV=1 salted=32" [--config ...]
config keys: T (threads), SV (shader validation + debug layer, 0/1), salted (K
synthetic kernels, cold every iteration), iters, serialize (0/1), link (0/1),
cpt (compiler per thread, 0/1), plain (K unlinked salted kernels), bind (0/1, the consumer also binds on a compute encoder), consume (0/1, a thread uses finished pipelines: function handle + IFT), fresh (fresh library per iteration, 0/1).
Without salted the app metallib's RT kernels are used (warm after the first run
because the driver caches them by content).
"""
import argparse, collections, fcntl, os, subprocess, sys, time


def run_child(exe, metallib, cfg, lock):
    env = dict(os.environ)
    sv = cfg.get("SV", "1") != "0"
    env.pop("MTL_SHADER_VALIDATION", None)
    env.pop("MTL_DEBUG_LAYER", None)
    if sv:
        env["MTL_SHADER_VALIDATION"] = "1"
        env["MTL_DEBUG_LAYER"] = "1"
    cmd = [exe, "--metallib", metallib, "--threads", cfg.get("T", "8"), "--iters", cfg.get("iters", "3"),
           "--serialize", cfg.get("serialize", "0"), "--link", cfg.get("link", "1"),
           "--compiler-per-thread", cfg.get("cpt", "0"), "--fresh-library", cfg.get("fresh", "1")]
    if cfg.get("salted"):
        cmd += ["--salted", cfg["salted"]]
    if cfg.get("plain"):
        cmd += ["--plain", cfg["plain"]]
    if cfg.get("bind"):
        cmd += ["--bind", cfg["bind"]]
    if cfg.get("consume"):
        cmd += ["--consume", cfg["consume"]]
    handle = None
    if lock:
        handle = open("/tmp/phosphor-gpu-verification.lock", "a+")
        fcntl.flock(handle, fcntl.LOCK_EX)
        while subprocess.run(["pgrep", "-x", "phosphor"], capture_output=True).returncode == 0:
            time.sleep(5)
    try:
        start = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
        return r.returncode, time.time() - start, r.stdout + r.stderr
    finally:
        if handle:
            handle.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("build_dir")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--lock", action="store_true", default=True)
    ap.add_argument("--keep-crash-logs", default="")
    a = ap.parse_args()
    exe = os.path.join(a.build_dir, "agx_link_stress")
    metallib = os.path.join(a.build_dir, "shaders", "phosphor.metallib")
    status = 0
    for text in a.config:
        cfg = dict(kv.split("=", 1) for kv in text.split())
        tally = collections.Counter()
        total = 0.0
        for i in range(a.runs):
            code, dt, out = run_child(exe, metallib, cfg, a.lock)
            tally[code] += 1
            total += dt
            if code not in (0, 1) and a.keep_crash_logs:
                with open("%s/crash_%s_%d.log" % (a.keep_crash_logs, text.replace(" ", "_"), i), "w") as f:
                    f.write(out)
        crashes = sum(n for c, n in tally.items() if c not in (0, 1))
        print("%-40s runs %d  ok %d  failed %d  crashed %d  %s  mean %.1fs" % (
            text, a.runs, tally[0], tally[1], crashes,
            {k: v for k, v in sorted(tally.items()) if k not in (0, 1)}, total / a.runs), flush=True)
        status |= 1 if crashes or tally[1] else 0
    return status


if __name__ == "__main__":
    sys.exit(main())
