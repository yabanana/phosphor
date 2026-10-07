# F11 reviewer correction — NON VERIFIED

Backportable onto frozen `4b80e36`, without changing that ref. No build/test/GPU
execution. `git diff --check` only.

Rigid endpoint mapping preserves world-area measure. Reuse with J=1 is unsafe
for an emitter whose scale/shear changes world area. Uniform scale2 maps area
by4; old W then underestimates a constant-integrand current integral by4.

The host now evaluates the same world poses as the scene transform producer,
using the frame's actual uploaded motion phases. An exact identity/incarnation
plus linear-metric check permits rigid movement and rejects non-rigid changes,
then increments the history content epoch. The tiny comparison tolerance2e-5
covers FP32 matrix arithmetic, not a measured quality/estimator budget. Shear is
conservatively rejected even if one triangle's area is accidentally unchanged.
Proposal support remains full and unchanged; a fresh candidate uses CURRENT
world area/PDF. No unimplemented Jacobian is substituted in `diMerge`.

Written tests include independent transformed-area expectations for scale2 and
shear, rigid translation/rotation retention, epoch rejection, and the old-W
negative yielding one-quarter of the correct area integral. Tests NOT EXECUTED.
