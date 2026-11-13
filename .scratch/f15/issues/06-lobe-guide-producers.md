# 06: Lobe guide producers: resolve, guide kernel and overflow fallback

Roadmap: F15.1

**What to build:** A per-pixel lobe guide (layout for all features frozen with static_asserts) is written by the visibility resolve, the guide kernel and the indexed overflow fallback for pixels of extended materials, allocated and written only when the scene feature union is non-empty. Consumers come in tickets 07 and 08.

**Blocked by:** 05

**Status:** ready-for-agent

- [ ] Guide content equals a CPU reconstruction of the resolved feature parameters on the ior/specular fixture (readback check with a corruption control that must FAIL)
- [ ] Forced-overflow capture (`--debug-meshlets 1 --debug-meshlets-corrupt count`, as `tools/f13_overflow_emission_check.sh`) equals the meshlet path on the ticket's fixture
- [ ] Guide allocated only for a non-empty union (memory per category in the report); Sponza unchanged (visual_check 0 px, D2 within band)
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
