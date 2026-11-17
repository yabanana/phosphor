# Phosphor

Native Metal 4 renderer for Apple silicon; a standalone product that other
applications (such as Track Studio) build on. This glossary fixes the words
used for its concepts in plans, tickets, tests and code.

## Materials

**Material description**:
The versioned, renderer-independent record of one material: its base
parameters, the material features it enables and their parameters.
_Avoid_: GPU material, material component (those are representations of it)

**Material feature**:
An optional part of the material model that a material enables, each defined
by a public specification (a glTF `KHR_materials_*` extension) or, where none is
ratified, by a documented Phosphor encoding.
_Avoid_: extension (the glTF file encoding of a feature), effect, lobe

**Lobe**:
One scattering component of the material model with its own angular
distribution: base diffuse, base specular, clearcoat, sheen, specular
transmission, diffuse transmission, subsurface.
_Avoid_: layer, feature

**Feature class**:
The exact set of material features a material uses; materials with the same
feature class are shaded by the same generated shader permutation.
_Avoid_: variant (the forward-pass axes of F3), resolve class (the older
normal-map/emissive tile classes)

**Support report**:
The per-material statement of which features the renderer implements,
approximates or does not support, produced when a scene is loaded.
_Avoid_: warning list

**Layer stack**:
A bounded ordered list of materials blended over one surface by masks, used for
terrain and other complex surfaces.
_Avoid_: lobe, coat, multi-material

**Layer**:
One material of a layer stack together with its blend mask.
_Avoid_: lobe, feature

## Consumers

**Track Studio case**:
The six Open Chess Set materials (two pawn tops with specular transmission,
two kings and two queens with subsurface scattering) that Track Studio reported
as rendered incompletely; a requirement and test case for Phosphor.
_Avoid_: Track Studio materials, REA materials
