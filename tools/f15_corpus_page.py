#!/usr/bin/env python3
"""F15.3 (ticket 22): generate the side-by-side page and the text defect report of a Khronos corpus run (stdlib only).

  tools/f15_corpus_page.py --run build/f15-corpus --corpus build/khronos-corpus --page build/f15-corpus/page \\
                           [--report-md docs/research/2026-11-14-f15-khronos-corpus-report.md]

Inputs: <run>/summary.json (and summary-validation.json when present) from tools/f15_corpus.py, the per-model Phosphor captures, the
corpus folder (the Khronos screenshots and the per-model licence records) and tools/f15_corpus_notes.json (what renders correctly, the named
approximations and the defects, written from the captures).

The PAGE shows the Khronos screenshots next to the captures.  Those screenshots are third-party content (CC-BY-4.0 / CC0 per model, the
attribution is printed under each one): the page is generated under build/ and is never committed.  The REPORT is text only (no third-party
image) and is committed.  Neither claims image parity: Phosphor has no image-based lighting and a different tonemap (see DISCLAIMER).
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import f15_corpus  # noqa: E402

DISCLAIMER = (
    "No global parity is claimed and none is ticked. The Khronos images were made with image-based lighting (an HDR environment) and "
    "the tonemap of the tool that produced them (Sample Viewer: Khronos PBR Neutral; Render Fidelity goldens: ACES Hill; some screenshots come from "
    "path tracers such as dspbr-pt, Stellar or Substance Stager, per each model's README). Phosphor has no image-based lighting: "
    "the declared rig is one white sun (3 W/m2) from the upper left behind the camera over a constant isotropic environment (radiance 0.5), "
    "exposure 1.0, ACES tonemap, a 60 degree field of view framed from the model bounds, one settled frame. Background, exposure, "
    "reflections of the environment and the look of every smooth or anisotropic surface therefore differ by construction. The comparison is "
    "qualitative and per feature: the numeric gates of F15 are the independent per-lobe references (tools/f15_check.py), not these images.")

DEVIATIONS = [
    ("Clearcoat F0", "Phosphor follows the ratified text: coat F0 fixed at IOR 1.5 (0.04). The Sample Renderer derives the coat F0 from the "
                     "material IOR; where a model sets KHR_materials_ior the coat highlight strength differs (negative control `coat-f0-from-ior`, ticket 09)."),
    ("Base under coat / sheen", "Phosphor weights the base by (1 - weight * max(F)) as the extension texts specify; the Sample Renderer composes the layers with its own albedo scaling."),
    ("Iridescence", "Phosphor evaluates the thin film per light sample at V.H and the environment / albedo terms at N.V (ticket 12); the Sample Renderer uses N.V everywhere, "
                    "so the hue shift under punctual lights differs near grazing half-vectors."),
    ("Anisotropy in the environment", "The Sample Renderer fetches the image-based light along a bent normal; Phosphor's declared environment model is a split sum at an effective "
                                      "roughness, which is invisible in a constant environment (ticket 11). Anisotropic stretching of environment reflections, which the Khronos "
                                      "AnisotropyRotationTest README says needs image-based lighting, cannot appear here."),
    ("Sheen table", "Phosphor generates its own Charlie E(mu, roughness) table (tools/sheen_table.py); the Sample Renderer ships a pre-baked LUT image. Both clamp at the table edge."),
    ("Lighting and tonemap", "Image-based lighting, the HDR environment itself, its orientation, the clear colour / skybox and the tonemap are different in every Khronos image; "
                             "see DISCLAIMER."),
    ("Unsupported by design", "KHR_materials_unlit and KHR_lights_punctual are reported unsupported (decision 02): a model that requires them is refused, a model that merely uses "
                              "unlit materials renders them lit."),
]


def load_json(path):
    return json.loads(pathlib.Path(path).read_text())


def khronos_screenshot(corpus, name, notes):
    """The Khronos screenshot to show: the notes' choice, else the largest screenshot file of the model."""
    folder = corpus / name / "screenshot"
    choice = notes.get(name, {}).get("khronos_screenshot")
    if choice and (folder / choice).is_file():
        return folder / choice
    candidates = sorted((p for p in folder.glob("screenshot*") if p.is_file()), key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0] if candidates else None


def licence_line(manifest, name):
    """Attribution line of a model from its manifest licence records (artist, owner, year, SPDX)."""
    records = manifest["models"].get(name, {}).get("license")
    if not isinstance(records, list) or not records:
        return "licence: see the model's LICENSE.md"
    parts = []
    for record in records:
        who = ", ".join(str(record[k]) for k in ("artist", "owner", "year") if record.get(k) and record[k] != "Public")
        what = f" ({record['what']})" if record.get("what") and len(records) > 1 else ""
        parts.append(f"{record.get('spdx') or record.get('license', '?')}{' - ' + who if who else ''}{what}")
    return "; ".join(parts)


def support_text(record):
    materials = record.get("materials")
    if not materials:
        return ""
    parts = []
    for feature, entry in sorted(materials["features"].items()):
        text = f"{feature}: {entry['support']}"
        if entry.get("approximation"):
            text += f" ({entry['approximation']})"
        parts.append(f"{text} x{entry['materials']}")
    return "; ".join(parts)


def status_cell(record):
    if record["status"] == "rendered":
        return "rendered"
    if record["status"] == "refused":
        return "REFUSED"
    return "FAILED"


def build_page(summary, validation, notes, corpus, manifest, page):
    images = page / "img"
    images.mkdir(parents=True, exist_ok=True)
    esc = html.escape
    rows = []
    for record in summary["models"]:
        name = record["model"]
        note = notes.get(name, {})
        engine = None
        if record.get("capture") and pathlib.Path(record["capture"]).is_file():
            engine = f"img/{name}-phosphor.png"
            shutil.copyfile(record["capture"], page / engine)
        khronos = khronos_screenshot(corpus, name, notes)
        reference = None
        if khronos:
            reference = f"img/{name}-khronos{khronos.suffix}"
            shutil.copyfile(khronos, page / reference)
        val = next((m for m in (validation or {}).get("models", []) if m["model"] == name), None)
        pipelines = record.get("pipelines", {})
        facts = [f"status: {status_cell(record)}" + (f" - {esc(record['reason'])}" if record.get("reason") else ""),
                 f"archive misses {pipelines.get('archive_misses', '?')}, fallback draws {pipelines.get('fallback_draws', '?')}, "
                 f"GPU allocations in measured frames {record.get('gpu_allocations_measured', '?')}",
                 f"validation (API + shader): {esc(validation_status(val)) if val else 'not run'}"]
        lists = []
        for title, key in (("Renders correctly", "renders"), ("Named approximations", "approximations")):
            if note.get(key):
                lists.append(f"<h4>{title}</h4><ul>{''.join(f'<li>{esc(x)}</li>' for x in note[key])}</ul>")
        if note.get("defects"):
            lists.append("<h4>Defects</h4><ul>" + "".join(
                f"<li><b>{esc(d['id'])}</b> [{esc(d['status'])}] {esc(d['summary'])}<br><i>Root-cause hypothesis:</i> {esc(d['hypothesis'])}</li>"
                for d in note["defects"]) + "</ul>")
        support = support_text(record)
        rows.append(f"""
<section id="{esc(name)}">
<h3>{esc(name)} <small>tier {esc(record['tier'])}</small></h3>
<p class="shows">{esc(f15_corpus.MODELS[name]['shows'])}</p>
<div class="pair">
<figure>{f'<img src="{engine}" alt="Phosphor capture of {esc(name)}">' if engine else '<div class="none">no capture (refused or failed)</div>'}<figcaption>Phosphor, declared rig</figcaption></figure>
<figure>{f'<img src="{reference}" alt="Khronos screenshot of {esc(name)}">' if reference else '<div class="none">no Khronos screenshot</div>'}
<figcaption>Khronos screenshot (qualitative; made with image-based lighting). {esc(licence_line(manifest, name))}, glTF-Sample-Assets {esc(manifest['commit'][:7])}.</figcaption></figure>
</div>
<ul class="facts">{''.join(f'<li>{x}</li>' for x in facts)}</ul>
{f'<p class="support">Engine support report: {esc(support)}</p>' if support else ''}
{''.join(lists)}
</section>""")
    deviations = "".join(f"<li><b>{esc(t)}.</b> {esc(d)}</li>" for t, d in DEVIATIONS)
    toc = " ".join(f'<a href="#{esc(m["model"])}">{esc(m["model"])}</a>' for m in summary["models"])
    rig = summary["rig"]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phosphor vs Khronos corpus</title>
<style>
:root {{ --bg:#fff; --fg:#1b1f23; --muted:#57606a; --card:#f6f8fa; --warn:#fff8c5; --line:#d0d7de; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#8b949e; --card:#161b22; --warn:#3a2f00; --line:#30363d; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,sans-serif; margin:0 auto; max-width:1180px; padding:0 16px 48px; }}
.banner {{ background:var(--warn); border:1px solid var(--line); padding:12px 16px; border-radius:8px; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 16px 12px; margin:20px 0; }}
.pair {{ display:flex; gap:12px; flex-wrap:wrap; }} figure {{ margin:0; flex:1 1 440px; }} img {{ width:100%; height:auto; display:block; border:1px solid var(--line); }}
figcaption, .shows, .support, small {{ color:var(--muted); font-size:13px; }} .none {{ border:1px dashed var(--line); padding:60px 12px; text-align:center; color:var(--muted); }}
.toc a {{ margin-right:10px; white-space:nowrap; }} h4 {{ margin:10px 0 2px; }} ul.facts {{ padding-left:18px; }}
</style></head><body>
<h1>Phosphor against the Khronos corpus</h1>
<p>glTF-Sample-Assets {esc(manifest['commit'][:7])}, default glTF variant. Captures: {esc(rig['resolution'])}, one settled frame after {rig['warmup']} warm-up frames,
sun {rig['sun_irradiance']} W/m2, constant environment {rig['environment_radiance']}, exposure {rig['exposure']}, {esc(rig['tonemap'])} tonemap, field of view {rig['fov_y_degrees']} degrees.
Generated {esc(summary['generated_utc'])} from {esc(summary['binary_sha256'][:12])}.</p>
<div class="banner"><b>Read this first.</b> {esc(DISCLAIMER)}</div>
<h2>Sample Renderer deviations that explain differences</h2><ul>{deviations}</ul>
<p class="toc">{toc}</p>
{''.join(rows)}
<p><small>Third-party images are shown for qualitative comparison only, under each model's licence (attribution printed under the image). This page is generated under build/ and not part of the repository.</small></p>
</body></html>"""
    (page / "index.html").write_text(document)


def validation_status(record):
    if record is None:
        return "not run"
    if record["status"] == "refused":
        return "refused (no render)"
    problems = record.get("validation_failures") or []
    others = record.get("other_messages") or []
    if record["status"] == "rendered" and not problems and not others:
        return "clean"
    if record["status"] == "rendered":
        return f"rendered, {len(others)} unclassified log lines" if not problems else f"{len(problems)} validation failures"
    return f"failed: {record.get('reason')}"


def build_report(summary, validation, notes, manifest, audits):
    esc = lambda s: str(s).replace("|", "\\|")  # noqa: E731
    lines = ["# F15.3 Khronos corpus run and defect report (ticket 22)", "",
             f"Run `{summary['generated_utc']}` of `tools/f15_corpus.py` on binary `{summary['binary_sha256'][:12]}`; corpus glTF-Sample-Assets `{manifest['commit'][:7]}` "
             "(default glTF variant, owner-approved subset, never committed). Text only: the side-by-side page is generated under `build/` by "
             "`tools/f15_corpus_page.py` because the Khronos screenshots are third-party content.", "",
             "## Declared rig and what this report is not", "", DISCLAIMER, "",
             f"Rig constants: {summary['rig']['resolution']}, {summary['rig']['warmup']} warm-up + {summary['rig']['frames']} measured frames, one settled frame captured, "
             "`--rt on --shadows rt --lighting restir --reflections rt --lighting-denoise custom --post --tonemap aces`, `--corpus-rig YAW,PITCH` "
             "(`src/testbench/corpus_rig.h`).", "", "## Sample Renderer deviations cited", ""]
    lines += [f"- **{t}.** {d}" for t, d in DEVIATIONS]
    lines += ["", "## Load / refusal table", "",
              "| Model | Tier | Required extensions | Result | Archive misses | Fallback draws | O7 allocs | Validation |", "|---|---|---|---|---|---|---|---|"]
    for record in summary["models"]:
        name = record["model"]
        audit = (audits or {}).get(name, {})
        required = ", ".join(audit.get("extensions_required", [])) or "-"
        val = next((m for m in (validation or {}).get("models", []) if m["model"] == name), None)
        pipelines = record.get("pipelines", {})
        result = status_cell(record) + (f": {record['reason']}" if record.get("reason") else "")
        lines.append(f"| {name} | {record['tier']} | {esc(required)} | {esc(result)} | {pipelines.get('archive_misses', '-')} | {pipelines.get('fallback_draws', '-')} | "
                     f"{record.get('gpu_allocations_measured', '-')} | {esc(validation_status(val))} |")
    lines += ["", "## Per model", ""]
    defect_rows = []
    for record in summary["models"]:
        name = record["model"]
        note = notes.get(name, {})
        lines += [f"### {name}", "", f"Shows: {f15_corpus.MODELS[name]['shows']}.", ""]
        support = support_text(record)
        if support:
            lines += [f"Engine support report: {support}.", ""]
        if record.get("rig"):
            lines += [f"Framing: {record['rig']['instances']} instances, bounds {record['rig']['bounds_min']} to {record['rig']['bounds_max']}, camera distance {record['rig']['distance']}.", ""]
        for title, key in (("Renders correctly", "renders"), ("Named approximations", "approximations")):
            if note.get(key):
                lines += [f"{title}:", ""] + [f"- {x}" for x in note[key]] + [""]
        if note.get("defects"):
            lines += ["Defects:", ""]
            for d in note["defects"]:
                lines += [f"- **{d['id']}** [{d['status']}] {d['summary']} Root-cause hypothesis: {d['hypothesis']}"]
                defect_rows.append((d["id"], name, d["status"], d["summary"]))
            lines += [""]
        if not any(note.get(k) for k in ("renders", "approximations", "defects")):
            lines += ["(no notes: " + ("refused, see the table" if record["status"] == "refused" else "not reviewed") + ")", ""]
    lines += ["## Defect index", "", "| Id | Model | Status | Summary |", "|---|---|---|---|"]
    lines += [f"| {i} | {m} | {s} | {esc(t)} |" for i, m, s, t in defect_rows]
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--page", type=pathlib.Path)
    parser.add_argument("--report-md", type=pathlib.Path)
    parser.add_argument("--notes", type=pathlib.Path, default=REPO / "tools/f15_corpus_notes.json")
    args = parser.parse_args(argv)
    summary = load_json(args.run / "summary.json")
    validation = load_json(args.run / "summary-validation.json") if (args.run / "summary-validation.json").is_file() else None
    notes = load_json(args.notes) if args.notes.is_file() else {}
    manifest = load_json(REPO / "tools/khronos_corpus.json")
    audits = load_json(args.run / "static-audit.json") if (args.run / "static-audit.json").is_file() else {}
    if args.page:
        if REPO.resolve() in args.page.resolve().parents and args.page.resolve().relative_to(REPO.resolve()).parts[0] != "build":
            raise SystemExit("the page holds third-party images: generate it under build/, never in a tracked folder")
        build_page(summary, validation, notes, args.corpus, manifest, args.page)
        print(f"page: {args.page / 'index.html'}")
    if args.report_md:
        args.report_md.write_text(build_report(summary, validation, notes, manifest, audits) + "\n")
        print(f"report: {args.report_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
