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
- A mapping note (MaterialX Standard Surface inputs to Phosphor's material
  description) is part of the F15 delivery, so the later port is mechanical.
