# 13: Track Studio case: the six Open Chess Set materials

Roadmap: F15.1, F15.3

**What to build:** ABeautifulGame (pinned) loads unchanged for the pawn tops and with a material-description override file for the four kings/queens carrying the MaterialX values (weight map, colour = base colour, radius = base colour × scale 0.003 / Queen_W 0.001); back-light and side-light rigs and parameter sweeps prove the effects; the MaterialX → Phosphor mapping note is written.

**Blocked by:** 09, 11

**Status:** ready-for-agent

- [ ] Back-light rig: the scene is visible through the pawn tops; transmission 0 → 1 increases the transmitted luminance on the glass mask monotonically; tint changes chroma as specified
- [ ] Side-light rig: subsurface weight 0 → 1 and scale × 0.5 / × 2 change the terminator softness monotonically; ground-truth comparison per ticket 11 bound on the piece masks
- [ ] Support report for the scene lists every feature as implemented or a named approximation
- [ ] Mapping note MaterialX Standard Surface → Phosphor description committed (docs), with the transmission_depth = 0 rule and the subsurface conversion
- [ ] No file of Track Studio read at runtime or modified (ADR-0001)
