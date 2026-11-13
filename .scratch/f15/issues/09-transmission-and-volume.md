# 09: Specular transmission and volume attenuation (ray-traced)

Roadmap: F15.1, F15.3

**What to build:** `KHR_materials_transmission` + `KHR_materials_volume` render per ADR-0004: the specular transport chooses reflection or refraction by Fresnel, refracts through the rough GGX BTDF, crosses up to four interfaces with Beer–Lambert attenuation along the real inside path (thin-walled without offset), falls back to probes/environment along the refracted ray without RT, keeps opaque shadows, and the denoiser demodulates by the combined reflect+transmit albedo. ABeautifulGame pawn tops show the scene through the glass when backlit.

**Blocked by:** 04

**Status:** ready-for-agent

- [ ] Transmission fixtures (factor, roughness × ior, thin vs volume, attenuation colour/distance) compared with Mitsuba 3.9.1 rough dielectric + homogeneous absorbing medium on the glass mask: mean ≤ 5 %, p95 ≤ 15 % (static accumulation)
- [ ] Glass behind glass is not black (interface-crossing fixture); crossings never exceed the bound; command buffers stay far below 1 s
- [ ] `--rt off` uses the declared probes/environment approximation and the support report says so
- [ ] Negative control (no refraction / attenuation off) fails the bounds
- [ ] Ray budget per pixel unchanged on non-transmissive scenes (Sponza D2 within noise); extra cost on the glass fixture measured
- [ ] TransmissionTest, TransmissionRoughnessTest, TransmissionThinwallTestGrid, AttenuationTest side by side; pawn-top backlit capture stored
