#!/usr/bin/env python3
"""F15.2 (ticket 21): targeted archive update for the exact-maths resolve kernels (shaders/material_passes_exact.metal).

A scene with material features requests `phx_exact::visibility_resolve` / `phx_exact::visibility_lit_resolve` (pipeline/forward_variants.h,
exactModuleKernel): the class bins, the union-specialised pipeline, the all-lobes reference and their texture-transform / subsurface
variants, and with --material-binning off (the default) the legacy-class resolves.  tools/material_class_archive.py adds the planned class bins; this tool adds every other pipeline of those kernels that the
archive lacks, label independent, additive (existing entries are never touched or duplicated):

  exact_module_archive.py update --archive shaders/pipelines.mtl4-json --harvest RAW [RAW ...] --output OUT
      RAW = `phosphor --harvest-pipelines RAW ...` outputs (tools/f15_permutations.py harvest, tools/f15_layers_archive.py harvest).
  exact_module_archive.py check  --archive shaders/pipelines.mtl4-json --harvest RAW [RAW ...]
      the archive holds every exact-kernel pipeline of the harvests (exit 1 otherwise).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import filter_pipeline_harvest as fph  # noqa: E402
import merge_pipeline_harvests as mph  # noqa: E402
import material_class_archive as mca  # noqa: E402

EXACT_KERNELS = ("phx_exact::visibility_resolve", "phx_exact::visibility_lit_resolve")


def keep_exact(corpus, present=None):
    """The pipelines of `corpus` whose compute function is one of EXACT_KERNELS (specialised or not), minus those `present` holds."""
    functions = corpus.get("function_descriptors", {})
    specialized = {d["label"]: d for d in functions.get("specialized_function_descriptors", [])}
    library = {d["label"]: d for d in functions.get("library_function_descriptors", [])}
    wanted = {label for label, d in library.items() if d.get("name") in EXACT_KERNELS}
    wanted |= {label for label, d in specialized.items() if d["function_descriptor"][len(fph.PREFIX):] in wanted}
    # The legacy-class resolves (constant 21 with MATERIAL_FEATURES, no RESOLVE_BIN) are kept: --material-binning off is the DEFAULT, and a
    # scene with features requests them there (F15 closure round 2: the budget, which runs binning on, never saw them).
    skip = fph.semantic_keys(present) if present else set()
    single = {"function_descriptors": functions}
    kept_pipelines, used = {}, set()
    for kind, descriptors in corpus.get("pipeline_descriptors", {}).items():
        kept = []
        for d in descriptors:
            if not (wanted & set(fph.references(d))):
                continue
            if skip and next(iter(fph.semantic_keys({**single, "pipeline_descriptors": {kind: [d]}}))) in skip:
                continue
            kept.append(d)
            used.update(fph.references(d))
        kept_pipelines[kind] = kept
    for label in list(used):
        if label in specialized:
            used.update(fph.references(specialized[label]))
    out = {k: v for k, v in corpus.items() if k not in ("function_descriptors", "pipeline_descriptors")}
    out["function_descriptors"] = {
        "library_function_descriptors": [d for label, d in library.items() if label in used],
        "specialized_function_descriptors": [d for label, d in specialized.items() if label in used]}
    out["pipeline_descriptors"] = kept_pipelines
    return out


STALE_BASE_KERNELS = ("visibility_resolve", "visibility_lit_resolve")


def prune_stale(archive):
    """Drop the base-module resolve pipelines specialised with MATERIAL_FEATURES (22) or RESOLVE_BIN (24): a scene with features now requests the
    exact copy, so these are never looked up.  Their specialised function descriptors go with them (library descriptors stay).  Returns
    (archive, number of pipelines dropped)."""
    functions = archive.get("function_descriptors", {})
    specialized = {d["label"]: d for d in functions.get("specialized_function_descriptors", [])}
    library = {d["label"]: d for d in functions.get("library_function_descriptors", [])}
    stale = {label for label, d in specialized.items()
             if library.get(d["function_descriptor"][len(fph.PREFIX):], {}).get("name") in STALE_BASE_KERNELS
             and any(c.get("id", {}).get("data") in (22, 24) for c in d.get("constant_values", []))}
    kept_pipelines, dropped = {}, 0
    for kind, descriptors in archive.get("pipeline_descriptors", {}).items():
        kept = [d for d in descriptors if not (stale & set(fph.references(d)))]
        dropped += len(descriptors) - len(kept)
        kept_pipelines[kind] = kept
    still = {r for descriptors in kept_pipelines.values() for d in descriptors for r in fph.references(d)}
    out = dict(archive)
    out["pipeline_descriptors"] = kept_pipelines
    out["function_descriptors"] = {
        "library_function_descriptors": list(library.values()),
        "specialized_function_descriptors": [d for label, d in specialized.items() if label not in stale or label in still]}
    return out, dropped


def combine(paths):
    combined, label = {}, None
    for path in paths:
        raw = mca.normalise(json.loads(Path(path).read_text()))
        label = label or raw["libraries"][0]["label"]
        combined = mph.merge(combined, mph.normalize_library(raw, label))
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["update", "check", "prune"])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--harvest", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    archive = json.loads(args.archive.read_text())
    if args.mode == "prune":
        pruned, dropped = prune_stale(archive)
        args.output.write_text(json.dumps(pruned, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"dropped_stale_pipelines": dropped}))
        return 0
    kept = keep_exact(combine(args.harvest), present=archive)
    counts = {kind: len(v) for kind, v in kept["pipeline_descriptors"].items()}
    if args.mode == "check":
        missing = sum(counts.values())
        print(json.dumps({"missing_exact_pipelines": counts}))
        return 1 if missing else 0
    if not args.output:
        parser.error("update needs --output")
    merged = mph.merge(archive, mph.normalize_library(kept, archive["libraries"][0]["label"]))
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"added_exact_pipelines": counts}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
