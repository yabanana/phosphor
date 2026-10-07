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

Additional ready source corrections:

8. eddfe4d — valid ZERO GI proposal histories count in M; independent Bernoulli .5/.625 negative.
9. 1f5d44b — shader/CPU geometric-vs-radiometric probe state; host patch artifact included.
10. 9a0fef6 — masked emitter radiance, exact exported ray roles and standalone triangle emitters.

The following host-epoch commit applies the reviewed patch to production
GiPasses; use it as well, rather than leaving a patch artifact unapplied.
Reference role mismatches are rejected BEFORE rendering, without a model-difference
bypass; full equivariant role filtering is not claimed. Fixture/contract tests are
written and NOT EXECUTED. The physical/reference fixes do not lower a quality gate.

F13-F14 source writing continues on a separate branch/worktree; no F15/OPT.

Additional DI sampling corrections (source-only):

- 55ce9c9a6e4398d08aacc5353ee48974b170b5ff: coherent CP scramble per temporal STBN block.
- 52431e0c852a1169797aa628251b8092343ea383: zero counted DI proposal streams survive helper/prefilter normalization.
- f996edd971ef851db62645c50bc72335969933f3: production host geometric/radiometric probe epoch split.

The subsequent numeric-zero guard commit rejects poisoned histories before M;
source tests include NaN/Inf. Neither energetic nor spectral evidence was run.
