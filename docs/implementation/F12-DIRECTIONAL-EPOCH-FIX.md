# F12 reviewer correction: directional/environment radiance — NON VERIFIED

Base is frozen4b80e36 plus isolated review corrections; no ref was rewritten.
No configure/build/test/GPU execution. Git diff formatting check only.

F11's local-light sampling list omits directionals intentionally. Its radiance
revision alone cannot invalidate DDGI/radiance cache/reservoir values when a
sun moves or switches off. The GI caller now supplies the actual SAME-FRAME
full GPULight list; a collision-free exact semantic tuple includes directional
orientation/color/intensity, all other analytic parameters, sky radiance,
solar angular radius and an external atmosphere/LUT revision. Any change bumps
one GI light epoch used by probe/cache/reservoir generation and view history.

The environment setter is ready for F14. Defaults remain the previous zero-sky
and0.00465rad sun. No fabricated atmosphere/light measurement is introduced.
Written tests cover sun ONLY, stable state, direction movement, intensity zero,
restoration, sky/disk/LUT changes; the local-only epoch is the negative control.
