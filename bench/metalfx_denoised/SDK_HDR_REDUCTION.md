# Public MetalFX HDR radiometry reduction

This standalone program uses Foundation, Metal and MetalFX only. It has no
Phosphor headers, shaders, render graph, packing stage or private framework
workaround. It does not measure lifetime.

Build and run on a Metal4FX-capable device/SDK:

```sh
xcrun clang++ -std=c++20 -fobjc-arc -framework Foundation -framework Metal -framework MetalFX sdk_hdr.mm -o sdk_hdr
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./sdk_hdr --constant > constant.log 2>&1
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ./sdk_hdr --wide > wide.log 2>&1
```

The target and 1% error criterion are authored application requirements, not
an assertion that Apple promises this precision for arbitrary HDR inputs.
The reference is a constant color and uses no stochastic transport.

Both cases encode 96 SDK frames at 128x96, reset at 0/48, and measure all RGB
pixels in the 80 steady frames. Packed input is independently copied back
and checked before encoding. Normal, material, motion, reverse-Z depth and
projection describe a static analytic plane. Public manual exposure is used.
Only output division by the authored input scale restores physical units;
there is no output fitting, compensation or channel remapping.

- Constant: physical/input RGB (0.5, 0.25, 0.125), preExposure 1, exposure 1.
- Wide: physical RGB (368640, 128, 64), packed RGB (5760, 2, 1),
  preExposure 1/64, exact R16 exposure 0.00008678436279296875.
- Exit 0 means the program's numerical gate passed. Exit 1 is a completed
  numerical failure. Exit 2 is usage/setup/input/readback failure.

Observed on Apple M5 Max, macOS 27.3 build 26C5035r, Xcode 27.2:
constant passes with maximum steady relative error 0.00390625;
wide fails at 0.0517578125. The latter matches the engine fixture's error.
This demonstrates reproduction outside the engine. It does not establish
that Apple has acknowledged an SDK defect, nor does it qualify native HDR.
