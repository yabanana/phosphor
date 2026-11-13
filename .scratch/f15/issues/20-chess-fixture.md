# 20: Chess fixture: the roadmap's Track Studio case

Roadmap: F15.1, F15.3

**What to build:** The six Open Chess Set materials render from public assets: ABeautifulGame (pinned) unchanged for the pawn tops, a material-description override file for the four kings/queens with the MaterialX values (weight map, colour = base colour, radius = base colour × scale 0.003 / Queen_W 0.001); back-light and side-light rigs and parameter sweeps; coloured shadows of the pawn tops; conversion note committed with the fixture.

**Blocked by:** 13, 14, 17

**Status:** ready-for-agent

- [ ] Back-light rig: the scene is visible through the pawn tops; transmission 0 → 1 increases transmitted luminance on the glass mask monotonically; tint changes chroma as specified
- [ ] Side-light rig: subsurface weight 0 → 1 and scale × 0.5 / × 2 change the terminator softness monotonically; ticket-17 bound on the piece masks
- [ ] Pawn-top shadows attenuated and tinted (qualitative: curved glass)
- [ ] Support report lists every feature as implemented or a named approximation; conversion note committed
