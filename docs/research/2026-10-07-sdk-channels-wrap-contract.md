# SDK channels fixture: invalidate the known source replacement

The preserved 96-frame native capture supplied a 7×7 patch moving one input
pixel per frame, then replaced it at frame 64: center x95→x32, motion 0, no
corresponding history reset. Independent analytic-image comparison exposed
an eight-frame recovery, exceeding the frozen four-frame cap. The complete
failed evidence and original protocol remain unchanged.

The channels fixture now invalidates history at each known modulo wrap,
in addition to its existing 48-frame reset/pre-exposure phases. The source
domain includes the wrap count, so it stays changed on the following frame;
this is not a transient marker that disappears after one reset. Continuous
one-pixel movement does not introduce additional invalidations. Constant,
impulse, lifecycle and wide-HDR scenarios retain their previous phase policy.

The portable `fxFixtureHistoryPolicy` is used by the actual fixture and has
written CPU regressions over 192 frames, wrap boundaries 63/64/65, existing
phase resets, other scenarios and invalid/overflow inputs. C++ compilation
and the corrective native run belong to the root tester.

Per-frame native JSON now records `requested_history_reset`,
`requested_camera_cut`, `input_signal_epoch`, `source_signal_epoch`,
`channels_wrap`, `sdk_reset_submitted`, `sdk_encode_delta` and
`history_hint_passed`. The actual reset observation comes from the adapter's
reset/encode counters surrounding that frame: those counters advance after
`setShouldResetHistory(reset)` and the real SDK encode call. The fixture
requires exactly one native encode and verifies that a requested reset/cut
reached that call. It also marks the first eight frames after a wrap as
unsettled metadata; the quality analysis continues to include every frame.

No image, ROI, radiance scale, threshold or reconstruction support changed.
The next run must preserve the original cap/ROI protocol in a fresh output
directory and verify reset/epoch metadata at frame 64 before assessing
recovery. The original unannounced replacement is retained as an adverse
signal-change control. This correction does not establish a general SDK bug,
motion-vector proof or full F13.6 acceptance.
