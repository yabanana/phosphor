# F11/F12 history-expiry estimator review

Conclusion: a real estimator defect, not an insufficient-precision result. All 22 rejected RGB comparisons are FAIL outside their preregistered CI+FP32 budget; none is INCONCLUSIVE and every CI is narrower than its2% precision bound. The four cases are point8 temporal/full and area temporal/full. The area ensemble is about9.8 standard errors high on whole-image RGB.

The mechanism is in DI diStream copying sample.age only when the source endpoint wins, while diReusable rejects age>=16. A fresh endpoint can therefore rejuvenate a reservoir that still contains old proposal mass. GI has the same dependency with age32, including a zero-source path that advances age only when the destination has no selected endpoint.

An independent exact rational enumeration with two positive proposal weights {1,3}, equal probability, maxM2 and age1 gives expectation182666318/91265265=2.0014878387741493 at the fifth frame, against integral2. With history-chain age advancing independently of endpoint choice, expectation remains exactly2 at every enumerated frame. This counterexample has unchanged receiver, support and target: neither visibility rejection, zero-source loss nor an FP32 allowance explains it.

The IID CPU model at the DI production caps (8 fresh candidates, maxM64, age16) uses131072 independent paths, 32 warmup and256 measured frames. Endpoint age produces1.00530663195 against expected1.000001 (SE0.0001823633); history-chain age gives1.00002968221 (SE0.0002099970). This model is independent of shader/STBN code and does not claim to reproduce the exact GPU magnitude.

The separate DI and GI patches use age for the complete incorporated history chain. Counted histories advance it even when a fresh endpoint wins and for zero/blocked proposal mass. M caps, age caps, proposal PDFs, target, weights, finalization and compatibility rules are unchanged. DI spatial merges propagate the maximum chain age without a temporal increment. GI keeps its existing increment per reuse hop, bounded at32. History buffers still store temporal output only.

This can increase variance or produce periodic refreshes in a static scene; it removes a demonstrated conditioning error rather than hiding it with a looser test. GI remains the explicitly biased baseline for mismatched visibility/support, an independent limitation.

Artifacts: `f11-history-chain-age.patch`, `f12-history-chain-age.patch`, `corrective-protocol.json`, `exact_history_expiry.py`, `exact-proof.json`, `iid-toy.json`. Patches are against immutable f4629c8 and both pass git apply --check on the integration tree. Candidate/base file snapshots are retained locally. Python exact proof and IID analysis ran on CPU only. C++ regression tests are written but have NOT been compiled by this agent. No GPU, Metal compilation, integration build, source mutation, or evidence overwrite occurred.

The corrective protocol was frozen before any corrective GPU run. Root should rerun the original nine-case ensemble (family459), using a new output directory, with identical seeds/frame ranges/ROIs/thresholds. Failure or insufficient precision retains its original meaning.
