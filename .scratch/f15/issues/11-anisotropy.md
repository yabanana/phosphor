# 11: Anisotropy lobe

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_anisotropy` per decision 06: anisotropic GGX D and V with the rotated tangent frame, anisotropic VNDF sampling in the transport, bent-normal environment term (declared).

**Blocked by:** 07, 08

**Status:** ready-for-agent

- [ ] Anisotropy fixtures (strength, rotation, direction texture) pass decision-11 thresholds; furnace bound holds
- [ ] Rotation direction and tangent handedness correct on mirrored instances (glTF conventions)
- [ ] Negative control (rotation sign flipped) fails
- [ ] AnisotropyStrengthTest, AnisotropyRotationTest, AnisotropyDiscTest side by side
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
