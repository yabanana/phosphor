#!/usr/bin/env python3
"""Targeted archive update for the material class pipelines (F15.2, ticket 21; ADR-0003).

The permutation generator (`variant_gen classes`, tools/variant_gen/class_gen.cpp) writes
`<build>/generated/pipeline/material_class_plan.json`: one entry per (resolve kernel, class) with its function
constants.  This tool keeps shaders/pipelines.mtl4-json equal to "the archive as it was, plus exactly the plan":

  check   --plan PLAN --archive ARCHIVE
          every plan pipeline exists exactly once, the archive holds at most `archived_permutations` distinct
          MATERIAL_FEATURES values, every MATERIAL_FEATURES value of the archive is a planned permutation, and the
          duplicate-identity count of the archive is reported (history duplicates are never added to).
  update  --plan PLAN --archive ARCHIVE --harvest RAW [RAW ...] --output OUT
          RAW = outputs of `phosphor --harvest-pipelines RAW ...` (tools/harvest_material_classes.sh).  Each is
          normalised like tools/harvest_pipelines.sh, filtered with tools/filter_pipeline_harvest.py semantics to the
          pipelines specialised with RESOLVE_BIN that the archive does not hold (label independent identity),
          checked to be inside the plan, merged (tools/merge_pipeline_harvests.py identity = complete state) and
          written as `jq -S` would.  No full re-harvest: the existing entries are untouched.

Exit codes: 0 ok, 1 check failed / harvest outside the plan or incomplete, 2 usage.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import filter_pipeline_harvest as fph  # noqa: E402
import merge_pipeline_harvests as mph  # noqa: E402

PLACEHOLDER = "@PHOSPHOR_METALLIB@"
MATERIAL_FEATURES_INDEX = 22


def function_names(corpus):
    """label -> (library function name, ((constant index, value), ...)) of every function descriptor."""
    functions = corpus.get("function_descriptors", {})
    library = {d["label"]: d["name"] for d in functions.get("library_function_descriptors", [])}
    out = {label: (name, ()) for label, name in library.items()}
    for d in functions.get("specialized_function_descriptors", []):
        base = d["function_descriptor"][len(fph.PREFIX):]
        constants = tuple(sorted((c["id"]["data"], c["value"]["data"]) for c in d.get("constant_values", [])))
        out[d["label"]] = (library.get(base, base), constants)
    return out


def compute_identities(corpus):
    """[(kernel, ((index, value), ...))] of every compute pipeline descriptor, in file order (duplicates kept)."""
    names = function_names(corpus)
    out = []
    for p in corpus.get("pipeline_descriptors", {}).get("compute_pipeline_descriptors", []):
        label = p["compute_function_descriptor"][len(fph.PREFIX):]
        out.append(names.get(label, (label, ())))
    return out


def plan_identities(plan):
    out = []
    for p in plan["pipelines"]:
        out.append((p["kernel"], tuple(sorted((c["index"], c["value"]) for c in p["constants"]))))
    return out


def archived_permutations(corpus):
    """Distinct nonzero MATERIAL_FEATURES values over every function of the archive."""
    return {v for _, constants in function_names(corpus).values() for i, v in constants if i == MATERIAL_FEATURES_INDEX and v}


def duplicate_identities(corpus):
    """Number of pipeline descriptors that repeat another one's label-independent identity (history, informational)."""
    total = 0
    functions = corpus.get("function_descriptors", {})
    single = {"function_descriptors": functions}
    for kind, descriptors in corpus.get("pipeline_descriptors", {}).items():
        keys = Counter(next(iter(fph.semantic_keys({**single, "pipeline_descriptors": {kind: [d]}}))) for d in descriptors)
        total += sum(n - 1 for n in keys.values())
    return total


def check(plan, archive):
    problems = []
    have = Counter(compute_identities(archive))
    for ident in plan_identities(plan):
        if have[ident] != 1:
            problems.append(f"plan pipeline {ident[0]} {dict(ident[1])} occurs {have[ident]} times in the archive (want 1)")
    permutations = archived_permutations(archive)
    budget = plan["budgets"]["archived_permutations"]
    if len(permutations) > budget:
        problems.append(f"{len(permutations)} archived material permutations > budget {budget}")
    stray = permutations - set(plan["permutations"])
    if stray:
        problems.append(f"MATERIAL_FEATURES values in the archive outside the plan: {sorted(stray)}")
    report = {"plan_pipelines": len(plan["pipelines"]), "archived_permutations": len(permutations), "budget": budget,
              "duplicate_identities_in_archive": duplicate_identities(archive)}
    return problems, report


def normalise(raw):
    """Same as `jq -S '.libraries |= map(.path = "@PHOSPHOR_METALLIB@")'` (the sorting happens at serialisation)."""
    for library in raw.get("libraries", []):
        library["path"] = PLACEHOLDER
    return raw


def update(plan, archive, harvests, bin_index):
    plan_set = set(plan_identities(plan))
    # The harvests of one build agree on every label: merge them (identity = complete state) and filter once.
    combined, label = {}, None
    for raw in harvests:
        raw = normalise(raw)
        label = label or raw["libraries"][0]["label"]
        combined = mph.merge(combined, mph.normalize_library(raw, label))
    kept = fph.filter_harvest(combined, bin_index, present=archive)
    outside = [i for i in compute_identities(kept) if i not in plan_set]
    if outside:
        raise ValueError(f"harvested pipelines outside the plan: {outside[:3]}")
    # RESOLVE_BIN belongs to compute kernels: nothing else may be specialised with it.
    for kind, descriptors in kept["pipeline_descriptors"].items():
        if kind != "compute_pipeline_descriptors" and descriptors:
            raise ValueError(f"unexpected {kind} specialised with RESOLVE_BIN")
    return mph.merge(archive, mph.normalize_library(kept, archive["libraries"][0]["label"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("--plan", type=Path, required=True)
    c.add_argument("--archive", type=Path, required=True)
    u = sub.add_parser("update")
    u.add_argument("--plan", type=Path, required=True)
    u.add_argument("--archive", type=Path, required=True)
    u.add_argument("--harvest", type=Path, nargs="+", action="extend", required=True)
    u.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    archive = json.loads(args.archive.read_text())
    bin_index = next(c["index"] for p in plan["pipelines"] for c in p["constants"] if c["value"] & 0x3000)
    if args.command == "check":
        problems, report = check(plan, archive)
        print(json.dumps(report, sort_keys=True))
        for p in problems:
            print("FAIL:", p, file=sys.stderr)
        return 1 if problems else 0
    try:
        merged = update(plan, archive, [json.loads(h.read_text()) for h in args.harvest], bin_index)
    except ValueError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    problems, report = check(plan, merged)
    for p in problems:
        print("FAIL:", p, file=sys.stderr)
    if problems:
        return 1
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
