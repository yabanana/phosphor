# 03: The six Track Studio materials (Open Chess Set) in MaterialX and glTF terms

Type: research
Status: resolved
Blocked by: none

## Question

Track Studio imports the official MaterialX Open Chess Set (15 materials);
Pawn_Top_B/W use transmission and King_B/W, Queen_B/W use subsurface
(Track Studio `library.rs` test `plan_chess_fixture`). What are their exact
Standard Surface inputs (transmission, transmission_color/depth/scatter,
subsurface, subsurface_color/radius/scale, maps), the license and pinned
source, the MaterialX-defined translation to glTF (standard_surface to
gltf_pbr nodegraph), and how the Khronos "ABeautifulGame" glTF encodes the
same six materials (which KHR extensions, which values)?
Findings: `.scratch/f15/research/open-chess-set.md`.

## Answer

Findings: [open-chess-set.md](../research/open-chess-set.md) (MaterialX
`9176de8`, glTF-Sample-Assets `8f49239`, glTF specs `f93f8af`).

- Source of the Track Studio case: MaterialX `standard_surface_chess_set.mtlx`.
  Pawn tops: `transmission` 1, `transmission_depth` 0 (tint on the closure,
  no absorption), white base colour, shared roughness and normal maps. Kings
  and queens: `subsurface` weight from a scattering map, `subsurface_color`
  and `subsurface_radius` bound to the base-colour texture, `subsurface_scale`
  0.003 (Queen_W 0.001); mean free path = radius × scale.
- Khronos `ABeautifulGame` (CC-BY-4.0) encodes the pawn tops exactly as the
  official MaterialX→glTF translation prescribes (transmissionFactor 1, tint in
  baseColorFactor, KHR_materials_volume thickness 0.22, infinite attenuation
  distance) and deliberately omits subsurface on kings/queens (draft
  extensions). The official translation drops every subsurface input.
- Decision: the pawn tops are verified by loading `ABeautifulGame` unchanged.
  The kings and queens need a subsurface feature in Phosphor's material
  description plus a glTF encoding chosen in the subsurface decision (ticket
  10); the F15 fixture carries the MaterialX values through that encoding with a
  documented conversion. Per ADR-0001 no Track Studio format enters Phosphor.
- Owner clarification (2026-11-13): Track Studio is a separate consumer; the
  developments are ported to it after F15. Verification happens in Phosphor.
