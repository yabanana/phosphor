# F11 live emissive UV correction — NON VERIFIED

The emitter transform producer now refreshes local positions AND UVs from the
current raster GPUVertex stream, then writes its per-slot metadata through an
explicit graph ShaderWrite version. Candidate/visibility/reference consumers
read that same updated version. The existing geometry content epoch rejects
history after updateVertices; proposals still retain full support.

No build/MSL/runtime test was performed. Tester must deform UVs across MASK and
emission texels, compare F9 alpha and DI sampling plus exported metadata, then
restore geometry and inspect history. The helper needs no new binary layout.
