#pragma once

#include <glm/glm.hpp>

namespace phosphor {

// F15.3 (ticket 22): the declared Phosphor light rig and camera framing of the Khronos corpus run (tools/f15_corpus.py).  Portable, so the
// framing maths is unit-tested on Linux.  The rig is NOT an attempt to reproduce the Khronos screenshots (they use image-based lighting
// and their own tonemap); it is one fixed, stated lighting for every model:
//   - one white sun disk, irradiance 3 W/m^2 on a surface facing it, coming from the upper left behind the camera;
//   - one constant isotropic environment of radiance 0.5 (no image-based lighting, no sky model);
//   - fixed exposure 1.0 (no auto exposure) and the ACES tonemap of the post chain;
//   - camera: vertical field of view 60 degrees (the engine's), looking at the centre of the model's world bounds along a fixed yaw/pitch
//     (yaw 0 = from +Z, the glTF front), at the distance where the bounds' projection fills the view with a 12 % margin.
struct CorpusRig {
    static constexpr float sunIrradiance = 3.0f;
    static constexpr float environmentRadiance = 0.5f;
    static constexpr float margin = 1.12f;
};

/// Request of the scene viewer (from --corpus-rig YAW,PITCH): `aspect` is the drawable's width / height.
struct CorpusRigRequest {
    bool  enabled = false;
    float yawDegrees = 0.0f, pitchDegrees = 0.0f;
    float aspect = 16.0f / 9.0f;
};

struct CorpusRigView {
    glm::vec3 position{0.0f}, target{0.0f};
    glm::vec3 towardSun{0.0f, 1.0f, 0.0f}; // unit vector from the scene towards the sun
    float     distance = 0.0f;             // camera to target
};

/// Frame the world-space bounds [minimum, maximum] for a camera of vertical field of view `fovYRadians`; empty or non-finite bounds
/// give a unit box around the origin (the caller reports it).
[[nodiscard]] CorpusRigView frameCorpusRig(glm::vec3 minimum, glm::vec3 maximum, float yawDegrees, float pitchDegrees,
                                           float fovYRadians, float aspect);

} // namespace phosphor
