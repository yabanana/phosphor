# 19: Critical lobe combinations and cross-lobe furnace

Roadmap: F15.1, F15.3

**What to build:** The layering order holds when lobes combine: hand-authored clearcoat + sheen (no Khronos sample has it), iridescence + transmission + volume, anisotropy + clearcoat, specular + sheen, transmission overriding diffuse transmission; furnace across every implemented combination.

**Blocked by:** 09, 10, 11, 12, 13, 16

**Status:** ready-for-agent

- [ ] Each combination passes decision-11 thresholds against the independent reference
- [ ] Furnace bound holds for every combination; an additive (non-layered) negative control exceeds it
- [ ] AnisotropyBarnLamp, IridescenceSuzanne, ToyCar, SpecularSilkPouf side by side
- [ ] The F15.1 roadmap line re-read literally; every lobe's status (exact/approximated) listed

## Comments

2026-11-14 - test-first harness (reference, fixtures, runner keys, archive; no engine or shader change). Commits `62a5aae`, `12307ef`,
`85a2a79`, `e879f18`, `6172e06`. STATUS: the final device run (`--run --expect green` over all keys) and the transmission-combination
captures were NOT done when this note was written (the GPU lock was held by another agent's run); the red-state numbers below come
from 31 captures of the engine before the ticket-13 integration (binary of `85a2a79`), evaluated with `--evaluate`.

Reference (`tools/f15_reference.py`, from research/khr-materials.md section 12, never from the MSL). Order: clearcoat > sheen > base
(ior/specular, iridescence, anisotropy inside the base); thin transmitted lobe is part of the base under the coat.
- coat + sheen: f = (1 - c Fc)(f_sheen + f_mat * scaling) + c Fc D V. Key `combo-coat-sheen-{sun,env,env@resolve}`. Green. Controls
  coat-ignored, sheen-ignored, order-swapped (sheen over coat), additive-layering: all fail.
- anisotropy + coat / + sheen: anisotropic D V under the layers. `combo-aniso-coat-*`, `combo-aniso-sheen-*`. Green. env keys drop
  aniso-rotation-flipped (passes the 5 % sphere-mean gate there).
- ior/specular + sheen: `combo-specular-sheen-*`. Green.
- iridescence + coat / + sheen: film replaces the Schlick Fresnel, coat/sheen above (interface coat-film ignored, per the text).
  `combo-irid-coat-*`, `combo-irid-sheen-*`. Green.
- iridescence + anisotropy: film under the anisotropic lobe; the exact environment uses V.H histograms of the anisotropic quadrature
  (agrees with a wi-space Monte Carlo). `combo-irid-aniso-*` (sun and env@resolve green; the exact env key has no capture yet).
- iridescence + ior/specular: base_f0 under the film = weight * f0 (README), strength 1 in the fixtures (for fractional strength
  the two README forms of the diffuse weight differ). `combo-irid-specular-*`. sun and env@resolve green;
  `combo-irid-specular-env` PENDING "ticket 19" without a matching control (see open engine item).
- thin subsurface + coat: coat attenuates the transmitted lobe by 1 - c Fc(N.V). `combo-thin-coat-sun` PENDING "ticket 19",
  pending_matches coat-ignored (the sun path of a thin pixel drops the coat); `-env` and `-env@resolve` green.
- ior/specular + coat + sheen + anisotropy: `combo-all-lobes-*`. Green.
- furnace: `furnace-combinations` (+ `@resolve`), 20 spheres, every pair, thin + coat and all five lobes: green; additive-layering
  control fails (declared twin too).
- Refused by the reference: thin subsurface with sheen, anisotropy, iridescence, ior/specular.
- Transmission combinations (declared --rt off closed form only): `transmission-rtoff-combinations`, iridescence + transmission +
  volume, transmission replacing the diffuse transmission (M = (1-t)((1-w) base + w colour (1 - 0.16 N.V)) + t tint T). Key pending
  "ticket 19", pending_matches declared-transmission-ignored; NOT captured yet. No ray-traced Mitsuba ground truth of a thin-film
  dielectric exists (not built).

OPEN ENGINE ITEM (orchestrator decision 2026-11-14: the film ior 1.8 row stays in the fixture): `combo-irid-specular-env`, sphere means
against the exact quadrature (rows: film 1.3/400, 1.3/800, 1.8/400, metal; columns: spec A..D): row0 0.0275 0.0327 0.0433 0.0286; row1
0.0311 0.0325 0.0481 0.0283; row2 0.0212 0.0224 **0.0501** 0.0270; metal 0.0145-0.0148. One sphere (ior 2.4, film 1.8) exceeds 5 % by 0.01
points; `iridescence-at-nv` does not model it (9 % on metals). The declared environment gate (`@resolve`) passes with 3e-4. Hypothesis: a
composition difference of the transport weight (film evaluated at N.V with base_f0 = weight f0 for iridescence + specular) rather than
noise; to be re-measured on the final capture and investigated by the engine agent of ticket 19.

Dropped controls: aniso-rotation-flipped on aniso+coat / aniso+sheen env keys, transmission-uncoated on the thin-coat env key (do not
move the exact environment sphere mean by 5 %).

F15.1 line (re-read literally: clearcoat, sheen, anisotropy, transmission, iridescence, subsurface + the Track Studio case). Lobe status
from `material_support.cpp` on `cl/f15-plan-c38b51`: base implemented; ior/specular, clearcoat, sheen, anisotropy, iridescence,
subsurface (thin only), transmission and volume are all APPROXIMATED with named approximations (none is "exact"); subsurface volumetric
and layers are not yet supported there. The Track Studio case (pawns transmission, king/queen subsurface) is ticket 20's.
