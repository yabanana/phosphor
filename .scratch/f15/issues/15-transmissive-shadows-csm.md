# 15: Coloured transmittance map for transmissive casters (CSM)

Roadmap: F15.1, F15.3

**What to build:** The CSM path gets a single-layer transmittance map per cascade (nearest transmissive caster depth + RGB transmittance with multiplicative accumulation) so receivers behind glass are attenuated and coloured without RT (declared single-layer approximation).

**Blocked by:** 13

**Status:** ready-for-agent

- [ ] Planar fixtures with `--shadows csm` match the RT result of ticket 14 within the decision-11 transmission bound (or the CSM filtering envelope, declared before measuring)
- [ ] Scenes without transmissive casters unchanged (visual_check 0 px); negative control (map ignored) fails
- [ ] Memory and cost of the map per cascade measured
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
