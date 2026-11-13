# 19: Critical lobe combinations and cross-lobe furnace

Roadmap: F15.1, F15.3

**What to build:** The layering order holds when lobes combine: hand-authored clearcoat + sheen (no Khronos sample has it), iridescence + transmission + volume, anisotropy + clearcoat, specular + sheen, transmission overriding diffuse transmission; furnace across every implemented combination.

**Blocked by:** 09, 10, 11, 12, 13, 16

**Status:** ready-for-agent

- [ ] Each combination passes decision-11 thresholds against the independent reference
- [ ] Furnace bound holds for every combination; an additive (non-layered) negative control exceeds it
- [ ] AnisotropyBarnLamp, IridescenceSuzanne, ToyCar, SpecularSilkPouf side by side
- [ ] The F15.1 roadmap line re-read literally; every lobe's status (exact/approximated) listed
