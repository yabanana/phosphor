# F13 — exact binary16 exposure control

The first manual exposure run is retained as a failure. The R16Float texture
contained 22*2^-24 although the independent CPU round-to-nearest conversion
predicted 23*2^-24. Native Metal texture write rounding does not promise that
CPU rounding mode (Metal Shading Language specification, texture writes).

This next bounded experiment supplies the exact representable value23*2^-24.
Metadata still records the ideal requested E=0.5/368640 separately. Target RGB
(368640,128,64), preExposure1/64, output restoration and all gates stay unchanged.
No compiler rounding flag is changed. The exposure readback must now match
exactly before interpreting the SDK output. Production policy is not promoted.
