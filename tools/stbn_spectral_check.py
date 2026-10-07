#!/usr/bin/env python3
"""Independent CPU FFT validation of original STBN ranks; never runs a renderer.

Protocol v1 is preregistered in tools/testdata/stbn/protocol-v1.json. A failing
result is saved with its original thresholds; this tool does not tune them.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np

SOURCE_FILES = ("src/renderer/stochastic_sampling.cpp", "src/renderer/stochastic_sampling.h")
PREREGISTRATION = "5ee62cd"
PROTOCOL_PATH = "tools/testdata/stbn/protocol-v1.json"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def spectrum(values, temporal=False):
    """Pool spectral power, never pool raw signals before transforming them."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("expected dimensions,time,height,width")
    if temporal:
        n = values.shape[1]
        data = np.moveaxis(values, 1, -1).reshape(-1, n)
        centered = data - data.mean(axis=-1, keepdims=True)
        power = np.abs(np.fft.fft(centered, axis=-1)) ** 2
        frequency = np.rint(np.fft.fftfreq(n) * n).astype(int)
        non_dc, low = frequency != 0, np.abs(frequency) == 1
        pooled = power.sum(axis=0)
    else:
        height, width = values.shape[-2:]
        data = values.reshape(-1, height, width)
        centered = data - data.mean(axis=(-2,-1), keepdims=True)
        power = np.abs(np.fft.fft2(centered, axes=(-2,-1))) ** 2
        fx = np.rint(np.fft.fftfreq(width) * width).astype(int)
        fy = np.rint(np.fft.fftfreq(height) * height).astype(int)
        squared = fy[:,None]**2 + fx[None,:]**2
        non_dc, low = squared > 0, (squared > 0) & (squared <= 2)
        pooled = power.sum(axis=0)
    denominator = float(pooled[non_dc].sum())
    numerator = float(pooled[low].sum())
    result = {"low_fraction": numerator / denominator if denominator > 0 else None,
              "non_dc_power": denominator, "low_power": numerator,
              "constant_sequences": int(np.count_nonzero(np.sum(centered**2,axis=-1 if temporal else (-2,-1)) == 0)),
              "sequence_count": int(data.shape[0]),
              "normalized_spectrum": (pooled / denominator).tolist() if denominator > 0 else None}
    if not temporal:
        # Diagnostic only: horizontal-vs-vertical power in the same low band.
        px, py = float(pooled[0,np.abs(fx)==1].sum()), float(pooled[np.abs(fy)==1,0].sum())
        result["xy_axis_relative_imbalance"] = abs(px-py)/(px+py) if px+py > 0 else None
    return result


def correlations(values):
    flat = np.asarray(values,dtype=np.float64).reshape(values.shape[0],-1)
    centered = flat-flat.mean(axis=1,keepdims=True)
    norms = np.sqrt(np.sum(centered**2,axis=1))
    if np.any(norms == 0):
        return {"valid":False,"p95_abs":None,"max_abs":None}
    matrix = (centered @ centered.T) / np.outer(norms,norms)
    samples = np.abs(matrix[np.triu_indices(len(matrix),k=1)])
    return {"valid":True,"p95_abs":float(np.quantile(samples,0.95)),"max_abs":float(samples.max()),
            "pair_count":int(len(samples))}


def distribution(ranks):
    flat = ranks.reshape(ranks.shape[0],-1)
    n = flat.shape[1]
    in_range = bool(np.all(flat < n))
    permutation = in_range and bool(np.all(np.sort(flat,axis=1) == np.arange(n)[None,:]))
    if not in_range:
        return {"rank_permutation":False,"exact_32_bin_histogram":False,"bin_count_min":None,"bin_count_max":None}
    bins = np.stack([np.bincount((row*32//n).astype(int),minlength=32) for row in flat])
    uniform = bins.shape[1] == 32 and bool(np.all(bins == n//32))
    return {"rank_permutation":permutation,"exact_32_bin_histogram":uniform,
            "bin_count_min":int(bins.min()),"bin_count_max":int(bins.max())}


def controls(seed, shape):
    dimensions, frames, height, width = shape
    n = frames*height*width
    permutation, white = np.empty(shape), np.empty(shape)
    for dimension in range(dimensions):
        a = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,1398030926,dimension,0])))
        b = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed,1398030926,dimension,1])))
        permutation[dimension] = ((a.permutation(n)+0.5)/n).reshape(frames,height,width)
        white[dimension] = b.random((frames,height,width))
    return {"uniform_permutation":permutation,"iid_uniform":white}


def summarize(runs, protocol):
    gates = protocol["gates"]
    results = []
    for threshold in protocol["thresholds"]:
        for axis in ("spatial","temporal"):
            for control in protocol["controls"]:
                ratios = []
                for run in runs:
                    group = run["thresholds"][str(threshold)][axis]
                    a, b = group["stbn"]["low_fraction"],group[control]["low_fraction"]
                    ratios.append(a/b if a is not None and b is not None and b > 0 else None)
                valid = all(v is not None and np.isfinite(v) for v in ratios)
                mean = float(np.mean(ratios)) if valid else None
                wins = sum(v is not None and v < 1 for v in ratios)
                passed = valid and mean <= gates["max_mean_low_band_ratio_to_each_control"] and wins >= gates["min_seeds_better_than_each_control"]
                results.append({"threshold":threshold,"axis":axis,"control":control,"per_seed_ratios":ratios,
                                "mean_ratio":mean,"better_seeds":wins,"passed":bool(passed)})
    uniform = all(r["distribution"]["rank_permutation"] and r["distribution"]["exact_32_bin_histogram"] for r in runs)
    converged = all(r["relaxation_converged"] for r in runs)
    correlated = all(r["correlations"]["stbn"]["valid"] and
                     r["correlations"]["stbn"]["p95_abs"] <= gates["per_seed_dimension_abs_pearson_p95_max"] and
                     r["correlations"]["stbn"]["max_abs"] <= gates["per_seed_dimension_abs_pearson_max"] for r in runs)
    passed = uniform and converged and correlated and all(r["passed"] for r in results)
    return {"passed":bool(passed),"distribution_passed":uniform,"relaxation_passed":converged,
            "dimension_correlation_passed":correlated,"spectral_gates":results,
            "generation_ms":{name:float(f([r["generation_ms"] for r in runs])) for name,f in
                             (("min",np.min),("mean",np.mean),("max",np.max))}}


def self_test():
    shape = (4,16,8,8)
    x, t = np.arange(8), np.arange(16)
    low_x = np.broadcast_to(np.sin(2*np.pi*x/8),shape)
    high_x = np.broadcast_to((-1.0)**x,shape)
    low_t = np.broadcast_to(np.sin(2*np.pi*t/16)[None,:,None,None],shape)
    high_t = np.broadcast_to(np.sin(8*np.pi*t/16)[None,:,None,None],shape)
    assert abs(spectrum(low_x)["low_fraction"]-1)<1e-12
    assert spectrum(high_x)["low_fraction"]<1e-12
    assert abs(spectrum(low_t,True)["low_fraction"]-1)<1e-12
    assert spectrum(high_t,True)["low_fraction"]<1e-12
    assert spectrum(np.ones(shape))["low_fraction"] is None
    permutation = np.tile(np.arange(1024),(4,1)).reshape(shape)
    assert distribution(permutation)["rank_permutation"]
    assert distribution(permutation)["exact_32_bin_histogram"]
    assert correlations(permutation)["max_abs"]>0.999999999
    bad = permutation.copy();bad[0,0,0,0]=1
    assert not distribution(bad)["rank_permutation"]
    bad[0,0,0,0]=0xffffffff
    assert not distribution(bad)["exact_32_bin_histogram"]
    white = controls(0,(64,16,8,8))["iid_uniform"]
    assert abs(spectrum(white)["low_fraction"]-8/63)<0.02
    assert abs(spectrum(white,True)["low_fraction"]-2/15)<0.02
    print("STBN analyzer: 12 independent CPU sanity/negative checks PASS; no generator invocation")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--self-test",action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test();return 0
    if not args.data or not args.output:ap.error("--data and --output are required")
    if args.output.exists():ap.error("output must be fresh; failed evidence is not overwritten")
    repo = Path(__file__).resolve().parents[1]
    protocol_bytes = (repo/PROTOCOL_PATH).read_bytes()
    frozen = subprocess.check_output(["git","show",f"{PREREGISTRATION}:{PROTOCOL_PATH}"],cwd=repo)
    if protocol_bytes != frozen:raise RuntimeError("protocol changed after preregistration")
    protocol = json.loads(protocol_bytes)
    source_hashes = {}
    for source in SOURCE_FILES:
        frozen = subprocess.check_output(["git","show",protocol["source_commit"]+":"+source],cwd=repo)
        actual = (repo/source).read_bytes()
        if frozen != actual:raise RuntimeError("generator source differs from the frozen snapshot: "+source)
        source_hashes[source] = sha256(actual)
    generation_bytes = (args.data/"generation.json").read_bytes()
    generation = json.loads(generation_bytes)
    if generation["generator_version"] != protocol["generator_version"] or generation["config"] != protocol["config"]:
        raise RuntimeError("generation configuration differs from preregistration")
    if [r["seed"] for r in generation["runs"]] != protocol["seeds"]:raise RuntimeError("seed corpus differs")
    config = protocol["config"]
    shape = (config["dimensions"],config["frames"],config["height"],config["width"])
    n = int(np.prod(shape[1:]))
    runs = []
    for entry in generation["runs"]:
        path = args.data/entry["file"]
        raw = path.read_bytes()
        ranks = np.frombuffer(raw,dtype="<u4").reshape(shape)
        values = {"stbn":(ranks.astype(np.float64)+0.5)/n,**controls(entry["seed"],shape)}
        run = {**entry,"ranks_sha256":sha256(raw),"distribution":distribution(ranks),
               "correlations":{name:correlations(value) for name,value in values.items()},
               "raw_rank_spectra":{name:{"spatial":spectrum(value),"temporal":spectrum(value,True)} for name,value in values.items()},
               "thresholds":{}}
        for threshold in protocol["thresholds"]:
            binary = {name:value<threshold for name,value in values.items()}
            run["thresholds"][str(threshold)]={axis:{name:spectrum(value,axis=="temporal") for name,value in binary.items()}
                                                   for axis in ("spatial","temporal")}
        runs.append(run)
    result = {"scope":"original STBN table CPU spectral verification only","preregistration_commit":PREREGISTRATION,
              "protocol_sha256":sha256(protocol_bytes),"protocol":protocol,"source_sha256":source_hashes,
              "generation_metadata_sha256":sha256(generation_bytes),"compiler":generation["compiler"],
              "python":platform.python_version(),"numpy":np.__version__,"platform":platform.platform(),
              "analyzer_sha256":sha256(Path(__file__).read_bytes()),
              "driver_sha256":sha256((repo/"tools/stbn_spectral_generate.cpp").read_bytes()),
              "summary":summarize(runs,protocol),"runs":runs}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result["summary"],indent=2))
    return 0 if result["summary"]["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
