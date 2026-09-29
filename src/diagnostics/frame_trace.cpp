#include "diagnostics/frame_trace.h"

namespace phosphor {

// F3 contract stub: implemented by the F3.0 measurement work package.
void FrameTrace::reserve(u32, u32) {}
void FrameTrace::addFrame(const FrameRecord&) {}
void FrameTrace::addSwitch(const SwitchRecord&) {}
void FrameTrace::setGpuTimes(const std::vector<float>&) {}
HitchReport analyzeHitches(const FrameTrace&, const HitchConfig&) { return {}; }
std::string formatHitchReport(const HitchReport&) { return "SWITCH"; }
std::string traceToCsv(const FrameTrace&) { return {}; }

} // namespace phosphor
