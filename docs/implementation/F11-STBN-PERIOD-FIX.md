# F11 temporal-block scramble — NON VERIFIED

CPU and MSL keep the same within-block rank lookup, but the Cranley-Patterson
rotation now depends on floor(frame/stbnFrames), dimension and seed. A16-frame
mask no longer produces exactly the same sampling sequence at every new block.
Generator version2 names this contract change. No spectral/quality/adoption claim.

Written tests compare different blocks and use a complete16-rank fixture with
an independent uniform-stratification oracle: one sample per bin per block and
constant-integral mean bounded by one half-bin. No tests were executed.
The tester still needs GPU/CPU seed agreement, DI energetic reference and many
independent seeds; the consistency checker is not a convergence proof.
