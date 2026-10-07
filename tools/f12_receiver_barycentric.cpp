// F12 CPU probe bridge. Compile as a tiny shared library against the exact source
// under review; f12_receiver_probe.py compares its result with double ray/triangle
// intersections and checks captured self-hits using an independent quadrature.
#include "renderer/visibility_math.h"
extern "C" unsigned f12ReceiverWeights(const float* clip, float x, float y, float width, float height,
                                      float* weights) {
    const auto result = phosphor::visibilityBarycentrics(clip[0], clip[1], clip[3],
        clip[4], clip[5], clip[7], clip[8], clip[9], clip[11], x, y, width, height);
    for (unsigned i = 0; i < 3; ++i) weights[i] = result.value[i];
    return result.valid;
}
