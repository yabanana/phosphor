#!/usr/bin/env python3
"""Keep only the pipelines of a harvest that specialise one function constant (F15 targeted archive updates).

A full re-harvest of shaders/pipelines.mtl4-json duplicates entries (the serializer records every pipeline the
run creates), so F15 tickets add their specialised pipelines by a targeted update instead:

  1. run the scenario with `--harvest-pipelines RAW` and normalise it like tools/harvest_pipelines.sh
     (`jq -S '.libraries |= map(.path = "@PHOSPHOR_METALLIB@")'`);
  2. `tools/filter_pipeline_harvest.py RAW --constant 22 --output KEPT` keeps the pipeline descriptors whose
     functions are specialised with that function-constant index (22 = MATERIAL_FEATURES), plus exactly the
     function descriptors they reference;
  3. `tools/merge_pipeline_harvests.py --output MERGED shaders/pipelines.mtl4-json KEPT ...` merges (identity =
     complete state, so repeated pipelines are kept once), then `jq -S . MERGED > shaders/pipelines.mtl4-json`.
"""
import argparse
import json
import sys
from pathlib import Path

PREFIX = "fnd:"


def references(value):
    """Every "fnd:<label>" string inside a descriptor."""
    if isinstance(value, str):
        return [value[len(PREFIX):]] if value.startswith(PREFIX) else []
    if isinstance(value, dict):
        return [r for v in value.values() for r in references(v)]
    if isinstance(value, list):
        return [r for v in value for r in references(v)]
    return []


def specialises(descriptor, constant):
    return any(c.get("id_type") == "FunctionConstantIndex" and c.get("id", {}).get("data") == constant
               for c in descriptor.get("constant_values", []))


def filter_harvest(corpus, constant):
    functions = corpus.get("function_descriptors", {})
    specialized = {d["label"]: d for d in functions.get("specialized_function_descriptors", [])}
    library = {d["label"]: d for d in functions.get("library_function_descriptors", [])}
    wanted = {label for label, d in specialized.items() if specialises(d, constant)}
    kept_pipelines, used = {}, set()
    for kind, descriptors in corpus.get("pipeline_descriptors", {}).items():
        kept = [d for d in descriptors if wanted & set(references(d))]
        kept_pipelines[kind] = kept
        for d in kept:
            used.update(references(d))
    # Specialised functions reference their library function.
    for label in list(used):
        if label in specialized:
            used.update(references(specialized[label]))
    missing = [label for label in used if label not in specialized and label not in library]
    if missing:
        raise ValueError(f"descriptor references unknown functions: {sorted(missing)[:3]}")
    out = {key: value for key, value in corpus.items() if key not in ("function_descriptors", "pipeline_descriptors")}
    out["function_descriptors"] = {
        "library_function_descriptors": [d for label, d in library.items() if label in used],
        "specialized_function_descriptors": [d for label, d in specialized.items() if label in used]}
    out["pipeline_descriptors"] = kept_pipelines
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("harvest", type=Path)
    parser.add_argument("--constant", type=int, required=True, help="function-constant index (22 = MATERIAL_FEATURES)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    kept = filter_harvest(json.loads(args.harvest.read_text()), args.constant)
    args.output.write_text(json.dumps(kept, indent=2, sort_keys=True) + "\n")
    counts = {kind: len(v) for kind, v in kept["pipeline_descriptors"].items()}
    print(json.dumps({"constant": args.constant, "pipelines": counts}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
