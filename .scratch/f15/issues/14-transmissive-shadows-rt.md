# 14: Attenuated coloured shadows through transmissive casters (ray tracing)

Roadmap: F15.1, F15.3

**What to build:** Owner decision 2026-11-13: shadow rays through transmissive surfaces accumulate an RGB transmittance instead of stopping — per-surface factor from transmission, tint and Fresnel, plus Beer–Lambert from the signed sum of entry/exit distances per medium (order-independent, so any-hit traversal needs no sorting); straight line, caustics excluded (declared). Used by the RT sun shadow mask (scalar → RGB), DI visibility, GI shadow rays and the reflection secondary lighting.

**Blocked by:** 13

**Status:** ready-for-agent

- [ ] Spike first (timeboxed, results in opt-log): Metal intersection-function invocation multiplicity for non-opaque triangles (duplicate invocations would double-count) and cost; mitigation chosen by measurement
- [ ] Planar-interface fixtures (glass pane, flat shallow water with sun within 30° of the normal) vs Mitsuba within the decision-11 transmission bound; curved-glass shadows (pawn tops) qualitative (caustics excluded)
- [ ] Opaque casters unchanged (visual_check 0 px on scenes without transmissive materials); negative control (transmittance ignored → opaque) fails the planar fixtures
- [ ] Sun shadow mask format change and DI/GI visibility changes keep F10/F11/F12 corpus cases green
- [ ] Cost on the glass fixture and Sponza D2 within band
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
