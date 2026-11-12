# Frozen surface interpolation regression

The fixture replays the exact ray, triangle, transform and hardware hit from
native Sponza diffuse frame37/ray162. Only the scene slot and primitive ordinal
are remapped to zero for the standalone one-triangle input. It invokes the
production MSL rtRefineSurfaceHit helper. No traversal, camera, tolerance or
test-ray adjustment is involved.

Using the configured Metal toolchain (metal/metallib), from the repository root:

    metal -std=metal4.0 -I src -I shaders -I build/closure-release/generated -fpreserve-invariance -c bench/f9_spike/shaders/surface_refinement_fixture.metal -o fixture.air
    metallib fixture.air -o fixture.metallib
    xcrun clang++ -std=c++20 -fobjc-arc -I src -framework Metal -framework Foundation bench/f9_spike/surface_refinement_fixture.mm -o fixture
    MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./fixture fixture.metallib

The original 3.5292790874e-5 m backward error must fail the unchanged
3.14801329545e-5 m bound. Refined output must pass it; shadow and miss controls
must be byte-identical to their input. On M5 Max the refined residual is
4.49953218682e-7 m. This fixture does not cover missed edge intersections.
The full native/forced-Apple9 matrix remains required.
