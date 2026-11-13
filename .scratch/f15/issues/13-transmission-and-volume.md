# 13: Specular transmission and volume attenuation (ray-traced)

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_transmission` + `KHR_materials_volume` per ADR-0004: Fresnel-chosen reflect/refract sample, rough GGX BTDF, always traced with RT whatever the reflection band, interface traversal that accepts the inside back faces of the transmissive medium (single-sided closed meshes included), ≤ 4 crossings, Beer–Lambert along the real inside path (a medium entered at a front face ends at any hit), thin-walled without offset; probes/environment along the refracted ray without RT; refract lobe id/axis/IOR-scaled roughness to the denoiser; emitters hit by refracted rays owned by the transport, delta lights by the thin-walled DI BTDF.

**Blocked by:** 07, 08

**Status:** ready-for-agent

- [ ] Transmission fixtures (factor, roughness × ior, thin vs volume, attenuation) vs Mitsuba 3.9.1 rough dielectric (same roughness) + homogeneous absorbing medium on the glass mask, with nothing behind the glass lit through glass: decision-11 transmission bound
- [ ] Single-sided closed glass mesh renders the exit correctly; negative control keeping today's back-face rejection (`reflections.metal:21`) fails
- [ ] Glass behind glass is not black; crossings never exceed the bound; command buffers far below 1 s
- [ ] `--rt off`: image compared with the declared probes/environment approximation (image check + negative control), and the support report says so
- [ ] Emissive panel behind thin glass: no double count
- [ ] Ray budget per pixel unchanged on non-transmissive scenes (Sponza D2 within band); extra cost on the glass fixture measured
- [ ] Temporal clip (orbit, moving light) within F13.6; transmissive pixels reactive for MetalFX temporal
- [ ] Forced-overflow capture (`--debug-meshlets 1 --debug-meshlets-corrupt count`, as `tools/f13_overflow_emission_check.sh`) equals the meshlet path on the ticket's fixture
- [ ] TransmissionTest, TransmissionRoughnessTest, TransmissionThinwallTestGrid, AttenuationTest side by side
- [ ] Pipelines added or specialised by this ticket enter the archive by a targeted update; `tools/archive_check.sh` and `tools/hitch_check.sh` green; 0 render-thread compiles and 0 GPU allocations in measured frames
