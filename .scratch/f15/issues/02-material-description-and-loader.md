# 02: Material description, glTF extension parsing and support report

Roadmap: F15.1, F15.3

**What to build:** A versioned, portable material description with one optional block per material feature and a bounded layer stack; the glTF loader parses every F15 extension (and texture transforms, emissive strength, required extensions) into it and prints a support report; a versioned material-description file can define materials or override a glTF's materials by name. Rendering does not change yet: every feature is reported `unsupported` until its ticket lands.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] Doctest seam 1: for each extension (clearcoat, sheen, anisotropy, transmission, volume, ior, specular, iridescence, emissive_strength, texture_transform, diffuse_transmission RC, draft KHR_materials_scatter at PR head a6cd5b4) parameters, defaults, ranges, texture channel and colour space match the specification text cited in research/khr-materials.md
- [ ] Uniform UV0 transform is baked into the primitive's UV0 (unit test with an independent worked example); a non-uniform transform is reported unsupported for that material
- [ ] An asset whose `extensionsRequired` lists an unsupported extension is refused with an error naming it; optional unknown extensions are listed and the asset loads
- [ ] `MATERIALS` log line and report-JSON section (schema bump) with implemented / approximated / unsupported per material and feature; draft/RC inputs labelled
- [ ] Material-description file load/save round-trip and name-keyed override over a glTF, with validation errors for out-of-range values
- [ ] No Track Studio type or name appears in the schema (ADR-0001); Sponza loads with an empty feature set and identical images (visual_check 0 px)
