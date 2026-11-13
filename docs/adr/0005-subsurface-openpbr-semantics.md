# Subsurface follows OpenPBR semantics; the draft KHR_materials_scatter is accepted

Owner decision, 2026-11-13. glTF has no ratified subsurface extension
(`KHR_materials_diffuse_transmission` is a release candidate;
`KHR_materials_scatter`, PR #2579, is a draft that replaces `volume_scatter` and
mirrors OpenPBR). Phosphor's subsurface feature uses the OpenPBR subsurface
parameters (weight, multiple-scattering colour, per-channel mean free path as
radius × scale in metres, Henyey–Greenstein anisotropy; thin or volumetric). It
is fed by Phosphor's versioned material-description file, by the draft
`KHR_materials_scatter` at a pinned revision (reported as `draft`) and by
`KHR_materials_diffuse_transmission` for the thin mode. A future ratified
extension changes only the loader mapping, not the feature.
