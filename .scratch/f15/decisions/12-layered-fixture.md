# 12: F15.4 consumer fixture and real-world cost measurement

Type: grilling
Status: resolved
Blocked by: 05

## Question

Which scene proves F15.4 (layer equivalence, transitions, energy) and measures
"costo resolve nel mondo reale" when Phosphor has no terrain system yet?

## Answer

- Fixture: a procedural heightfield patch (existing procedural meshes) with four
  CC0 PBR materials (ambientCG, fetched by a pinned script with SHA-256 hashes
  like Sponza, never committed), one RGBA weight mask generated
  deterministically (noise + slope/height rules, seed declared) plus an optional
  height-contrast parameter, and a "finish" coat layer in cavities to exercise a
  feature layer. Named scenarios: weights 0/1 per layer, a linear transition
  ramp, a furnace patch, and the measured view.
- Real-world cost: GPU time of the material resolve at 1080p on the measured
  view, per active-layer count (1–4) and per feature class, A/B/A with 3
  replicas on a quiet machine, compared with the single-material resolve on the
  same view; reported in `docs/perf-log.md`, not gated beyond the binning
  decision rule and Sponza no-regression.
- The Track Studio audit (`research/track-studio-audit.md`) is read for
  transferable layering lessons before the F15.4 ticket starts.
