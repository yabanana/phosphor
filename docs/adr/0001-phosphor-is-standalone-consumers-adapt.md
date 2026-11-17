# Phosphor is a standalone product; consumers adapt to its material model

Owner direction, 2026-11-13 (F15 planning): Phosphor is a separate product that
Track Studio builds on and uses, not the other way around. Phosphor's material
model follows public standards (glTF 2.0 and ratified `KHR_materials_*`
extensions; where no ratified standard exists, a Phosphor-defined, documented
encoding based on published shading models). A consumer's needs (Track Studio's
six Open Chess Set materials) are requirements and test cases for Phosphor, not
a format Phosphor mirrors. Developments are brought back to Track Studio after
they land in Phosphor; Track Studio is never modified from Phosphor work.

## Consequences

- No Track Studio type, field name or file format appears in Phosphor's schema,
  loader or tests; the Track Studio case is reproduced in Phosphor with the same
  public assets and parameter values.
- The conversion of the chess fixture's MaterialX values into Phosphor's
  description is documented with the fixture so it is reproducible.

## Addendum (owner, 2026-11-13)

Phosphor stays a separate product; F15 considers only Phosphor. The owner
pauses Track Studio's development; F15 does no work for Track Studio and no
host-API work (the `rea/host-embedding` branch stays outside F15). The six Open
Chess Set materials remain a Phosphor fixture built from public assets.
