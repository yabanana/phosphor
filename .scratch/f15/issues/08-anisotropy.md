# 08: Anisotropy lobe

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_anisotropy` (strength, rotation, direction texture) renders per decision 06: anisotropic GGX D and V with the rotated tangent frame, anisotropic VNDF sampling in the specular transport, bent-normal environment term (declared).

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] Anisotropy fixtures (strength sweep, rotation sweep, direction texture) pass decision-11 thresholds; furnace bound holds
- [ ] Rotation direction and tangent handedness match the specification on mirrored instances (glTF conventions in CLAUDE.md)
- [ ] Negative control (rotation sign flipped) fails the rotation fixture
- [ ] AnisotropyStrengthTest, AnisotropyRotationTest, AnisotropyDiscTest side by side under the declared rig
