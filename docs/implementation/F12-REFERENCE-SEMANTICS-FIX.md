# F12 offline role/emission reviewer corrections — NON VERIFIED

The exporter preserves every valid instance that is camera/indirect visible OR
casts shadows, retaining its original flags. It no longer drops a caster-only
object and silently changes the exported scene's shadow domain.

Standard Mitsuba path rays cannot express the engine's independent primary/
indirect versus shadow masks with this adapter. A snapshot with visible-only,
caster-only or inactive exported roles is therefore rejected before any
renderer/dependency import. --allow-model-differences cannot override it.
Strict comparison requires the reference report's equivalent-role provenance
and converged/non-model-different state. This is an unsupported ray-domain
constraint, not an approval gate.

Sampled type6 triangles are deduplicated only if their valid material metadata
identifies an extracted full-scene mesh emitter. Standalone legal API triangles
with materialINVALID are exported as actual world-space PLY triangle area
emitters with constant Le, winding/side flags and zero reflective albedo.
Their geometry is part of the scene SHA256 and is not added to the snapshot
as a side effect of rendering.

The emissive texture independently applies MASK using filtered LOD0 alpha
converted to half, float32 base-alpha multiplication and the original >=cutoff
rule. BSDF opacity alone does not mask an area emitter's radiance. Emission
and BSDF mask use the same predicate. Current primary Mitsuba area.cpp supports
explicit twosided and uniform surface sampling (sample_texture=false), used
here for arbitrary full-scene UV layouts. An older renderer rejecting those
properties fails the reference instead of guessing behavior.

Primary source consulted:
https://raw.githubusercontent.com/mitsuba-renderer/mitsuba3/master/src/emitters/area.cpp

Written tests: test_gi.cpp caster-only snapshot/full flags and standalone
triangle PLY; test_f12_reference_contract.py independently checks masked/
surviving/equality/opaque emission, mandatory role rejection even with model-
difference override and provenance-based triangle dedup. None was executed.
No reference image or acceptance claim was generated.
