# 12: Critical lobe combinations and cross-lobe furnace

Roadmap: F15.1, F15.3

**What to build:** The layering order of research 01 holds when lobes combine: hand-authored clearcoat + sheen fixture (no Khronos sample has it), iridescence + transmission + volume, anisotropy + clearcoat, specular + sheen, transmission overriding diffuse transmission; furnace across every implemented combination.

**Blocked by:** 05, 06, 07, 08, 09, 10, 11

**Status:** ready-for-agent

- [ ] Each combination fixture passes decision-11 thresholds against the independent reference
- [ ] Furnace bound holds for every combination; an additive (non-layered) combination negative control exceeds it
- [ ] AnisotropyBarnLamp, IridescenceSuzanne, ToyCar, SpecularSilkPouf side by side under the declared rig
- [ ] The F15.1 roadmap line is re-read literally and every lobe's status (exact/approximated) is listed
