#!/usr/bin/env python3
"""Keep only the pipelines of a harvest that specialise one function constant (F15 targeted archive updates).

A full re-harvest of shaders/pipelines.mtl4-json duplicates entries (the serializer records every pipeline the
run creates), so F15 tickets add their specialised pipelines by a targeted update instead:

  1. run the scenario with `--harvest-pipelines RAW` and normalise it like tools/harvest_pipelines.sh
     (`jq -S '.libraries |= map(.path = "@PHOSPHOR_METALLIB@")'`);
  2. `tools/filter_pipeline_harvest.py RAW --constant 22 --skip-present shaders/pipelines.mtl4-json --output KEPT`
     keeps the pipeline descriptors whose functions are specialised with that function-constant index
     (22 = MATERIAL_FEATURES) and that the archive does not hold yet, plus exactly the function descriptors they
     reference.  Labels are hashes of the metallib and change with every shader edit, so presence is compared
     by function names, constants and state (without --skip-present, a later harvest duplicates entries);
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


def semantic_keys(corpus):
    """Label-independent identity of every pipeline descriptor: function names and constant values in place of
    the function labels (hashes of the metallib, which change with every shader edit) plus the remaining
    pipeline state."""
    functions = corpus.get("function_descriptors", {})
    library = {d["label"]: d["name"] for d in functions.get("library_function_descriptors", [])}
    specialized = {}
    for d in functions.get("specialized_function_descriptors", []):
        base = d["function_descriptor"][len(PREFIX):]
        constants = sorted((c["id"]["data"], json.dumps(c["value"], sort_keys=True)) for c in d.get("constant_values", []))
        specialized[d["label"]] = [library.get(base, base), constants]

    def resolve(value):
        if isinstance(value, str) and value.startswith(PREFIX):
            label = value[len(PREFIX):]
            return specialized.get(label) or library.get(label) or label
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    return {(kind, json.dumps(resolve(d), sort_keys=True))
            for kind, descriptors in corpus.get("pipeline_descriptors", {}).items() for d in descriptors}


def filter_harvest(corpus, constant, present=None, functions_by_name=()):
    """`present`: a corpus whose pipelines are dropped from the result (semantic identity, see semantic_keys).
    `functions_by_name`: library functions (new kernels without a specialisation constant, e.g. a new pass) whose pipelines are kept too."""
    functions = corpus.get("function_descriptors", {})
    specialized = {d["label"]: d for d in functions.get("specialized_function_descriptors", [])}
    library = {d["label"]: d for d in functions.get("library_function_descriptors", [])}
    wanted = {label for label, d in specialized.items() if specialises(d, constant)}
    wanted |= {label for label, d in library.items() if d.get("name") in set(functions_by_name)}
    skip = semantic_keys(present) if present else set()
    own = {}
    if skip:
        single = {"function_descriptors": functions}
        for kind, descriptors in corpus.get("pipeline_descriptors", {}).items():
            for index, d in enumerate(descriptors):
                own[(kind, index)] = next(iter(semantic_keys({**single, "pipeline_descriptors": {kind: [d]}})))
    kept_pipelines, used = {}, set()
    for kind, descriptors in corpus.get("pipeline_descriptors", {}).items():
        kept = [d for index, d in enumerate(descriptors)
                if wanted & set(references(d)) and own.get((kind, index)) not in skip]
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
    parser.add_argument("--function", action="append", default=[],
                        help="also keep the pipelines of this library function (a new kernel without the constant); repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-present", type=Path,
                        help="drop pipelines already in this corpus (e.g. shaders/pipelines.mtl4-json), compared by "
                             "function names, constants and state rather than by the metallib-dependent labels")
    args = parser.parse_args(argv)
    present = json.loads(args.skip_present.read_text()) if args.skip_present else None
    kept = filter_harvest(json.loads(args.harvest.read_text()), args.constant, present, args.function)
    args.output.write_text(json.dumps(kept, indent=2, sort_keys=True) + "\n")
    counts = {kind: len(v) for kind, v in kept["pipeline_descriptors"].items()}
    print(json.dumps({"constant": args.constant, "pipelines": counts}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
