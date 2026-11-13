# 22: Khronos corpus run and defect report

Roadmap: F15.3

**What to build:** A pinned fetch script brings Tier A/B models (glTF-Sample-Assets 8f49239, hashes and licence records, never committed); the bench renders each under the declared Phosphor rig; a side-by-side page against the Khronos screenshots/goldens documents per-feature defects and the explained Sample Renderer deviations.

**Blocked by:** 19

**Status:** ready-for-agent

- [ ] Fetch script reproducible, verifies hashes, stores licence/attribution, skips Draco/BasisU/WebP variants
- [ ] Every Tier A model loads or is refused for a stated reason (e.g. DiffuseTransmissionTest requires unlit/lights_punctual)
- [ ] Side-by-side page with per-feature defects; no global parity claim; Sample Renderer deviations cited
- [ ] Captures run serially with the runner's GPU guard
