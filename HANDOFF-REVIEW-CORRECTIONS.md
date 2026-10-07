# F10-F12 reviewer corrections — NON VERIFIED

Frozen delivery ref `codex/f10-f12-delivery` at4b80e36 was NOT changed.
This branch starts there; cherry-pick the isolated corrections into the tester's
already compiled/integrated branch. No build, shader compile, tests, rendering
or measurement was executed by this writer. Git diff formatting only.

Ordered commits currently ready:

1. dfe8cdb5f054d7fa6135649cfc2e11b57a4d273b — F11 emitter scale/shear domain tracker + area/history negative.
2. 5eb70f03f19fbfefee2c884f72f6d0decaa2b276 — F12 WORLD-facing to material-facing on mirrored instances.
3. 3f9ffd984151c39545c330df9568018a1f46b189 — F12 full analytic directional/sky/solar GI epoch.
4. d147149dc54d66ea4d88cf389a853ec735d88605 — exact conservative affine-domain comparison; no small-scale allowance.
5. ab64225a5275acb68d62dc3310bfabfbc460f4e9 — preserve sampled-light radiance revision alongside directionals.
6. a4a1388c8c6040004db588735a2202a9fbb101af — F11 persistent pre-sanitization numeric errors and finite-overflow negative.
7. bf3535ef5a70aca5b59495c838ecca5d7ae312f5 — F11 live emissive positions/UV and graph metadata write version.

Individual source contracts/commands are in docs/implementation/F11-AREA-DOMAIN-FIX,
F12-FACING-FIX, F12-DIRECTIONAL-EPOCH-FIX, F11-NUMERICS-FIX and F11-LIVE-EMISSIVE-UV-FIX.md.

Do not drop the tester's2afbf20 syntax/addressspace/receiver-alpha fixes or F9
main integration. The writer is NOT duplicating those edits. The tester owns
shadow drawCascade/bucket dispatch optimization; this branch does not touch it.

Pending separate writer corrections: GI validzero proposal M normalization,
geometry-vs-radiometry DDGI relocation state, independent reference MASK/roles
and standalone triangle endpoints. These will be appended as concrete commits.

F13-F14 source writing continues on a separate branch/worktree; no F15/OPT.
