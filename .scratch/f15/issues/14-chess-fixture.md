# 14: Chess fixture: the roadmap's Track Studio case

Roadmap: F15.1, F15.3

**What to build:** The six Open Chess Set materials of the roadmap case render from public assets: ABeautifulGame (pinned) unchanged for the pawn tops, and a material-description override file for the four kings/queens carrying the MaterialX values (weight map, colour = base colour, radius = base colour × scale 0.003 / Queen_W 0.001); back-light and side-light rigs and parameter sweeps prove the effects; the conversion of the MaterialX values is documented so the fixture is reproducible.

**Blocked by:** 10, 12

**Status:** ready-for-agent

- [ ] Back-light rig: the scene is visible through the pawn tops; transmission 0 → 1 increases the transmitted luminance on the glass mask monotonically; tint changes chroma as specified
- [ ] Side-light rig: subsurface weight 0 → 1 and scale × 0.5 / × 2 change the terminator softness monotonically; ground-truth comparison per ticket 12 bound on the piece masks
- [ ] Support report for the scene lists every feature as implemented or a named approximation
- [ ] Conversion note (MaterialX Standard Surface values → Phosphor description, transmission_depth = 0 rule, subsurface conversion) committed with the fixture
