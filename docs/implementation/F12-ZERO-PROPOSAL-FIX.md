# F12 zero-proposal reviewer correction — NON VERIFIED

GI_SAMPLE_VALID denotes a selected positive endpoint. GI_PROPOSAL_VALID now
separately denotes an eligible receiver/proposal history with M>0, including an
all-zero sample stream. Epoch/view/surface checks still reject invalid histories;
a zero stream has no selected secondary slot to dereference.

Fresh candidate finalization preserves the proposal flag when W is zero.
CPU/MSL merges add the bounded source M before considering positive selection.
An eligible zero source adds zero weight and its M, without inventing an
endpoint. The existing biased basic reconnection estimator remains explicitly
experimental; this fixes an additional bias that existed even for identical,
independent proposal domains.

The written test independently enumerates four equiprobable outcomes of two
Bernoulli0.5 samples: the corrected estimator's mean is0.5; the previous rule
that skipped an all-zero source gives0.625. No random generator, shader or test
was executed. The raw GPU checker accepts proposal-only state while checking
its receiver epochs/normal and leaves secondary checks for selected endpoints.
