#!/usr/bin/env python3
"""Merge actual Metal serializer corpora without dropping unlabeled pipelines."""
import argparse
import json
from pathlib import Path


def merge(left, right, path="root"):
    if type(left) is not type(right):
        raise ValueError("incompatible harvest types at " + path)
    if isinstance(left, dict):
        result = dict(left)
        for key, value in right.items():
            if key in result:
                result[key] = merge(result[key], value, path + "." + key)
            elif isinstance(value, (dict, list)):
                result[key] = merge(type(value)(), value, path + "." + key)
            else:
                result[key] = value
        return result
    if isinstance(left, list):
        # Pipeline records have no label. Their complete state is their identity.
        result, seen, labels = [], set(), {}
        for value in left + right:
            identity = json.dumps(value, sort_keys=True, separators=(",", ":"))
            if isinstance(value, dict) and "label" in value:
                label = value["label"]
                if label in labels and labels[label] != identity:
                    raise ValueError("conflicting labeled descriptor at " + path + ": " + label)
                labels[label] = identity
            if identity not in seen:
                result.append(value)
                seen.add(identity)
        return result
    if left != right:
        raise ValueError("incompatible harvest metadata at " + path)
    return left


def normalize_library(corpus, label):
    # The corpus sources must contain only the engine metallib. All scripts
    # reject framework captures before invoking this merge. Library labels
    # change between builds, while function labels can remain identical.
    libraries = corpus["libraries"]
    if not libraries or any(Path(lib["path"]).name not in ("phosphor.metallib", "@PHOSPHOR_METALLIB@") for lib in libraries):
        raise ValueError("only engine metallib corpora can be merged")
    previous = {lib["label"] for lib in libraries}
    for function in corpus["function_descriptors"]["library_function_descriptors"]:
        if function["library"] not in previous:
            raise ValueError("unknown function library reference")
        function["library"] = label
    for library in libraries:
        library["label"] = label
        library["path"] = "@PHOSPHOR_METALLIB@"
    return corpus


def compute_name(corpus, pipeline):
    descriptors=corpus["function_descriptors"]
    functions={f["label"]:f["name"] for f in descriptors["library_function_descriptors"]}
    for f in descriptors.get("specialized_function_descriptors",[]):
        functions[f["label"]]=functions[f["function_descriptor"].removeprefix("fnd:")]
    return functions[pipeline["compute_function_descriptor"].removeprefix("fnd:")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace-compute",action="append",default=[],help="replace named compute kernels from the first corpus with newly harvested records")
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()
    corpora=[json.loads(path.read_text()) for path in args.inputs]
    if args.replace_compute:
        names=set(args.replace_compute)
        replacements={compute_name(c,p) for c in corpora[1:] for p in c["pipeline_descriptors"]["compute_pipeline_descriptors"]}
        if not names<=replacements:raise ValueError("replacement kernels missing actual harvested descriptors: "+str(names-replacements))
        baseline=corpora[0]
        baseline["pipeline_descriptors"]["compute_pipeline_descriptors"]=[p for p in baseline["pipeline_descriptors"]["compute_pipeline_descriptors"] if compute_name(baseline,p) not in names]
    result, library_label = {}, None
    for corpus in corpora:
        if library_label is None:
            library_label = corpus["libraries"][0]["label"]
        result = merge(result, normalize_library(corpus, library_label))
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
