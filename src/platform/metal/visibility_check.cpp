#include "platform/metal/visibility_renderer.h"
#include "platform/metal/scene_renderer.h"
#include "platform/metal/mesh_renderer.h"
#include "renderer/gpu_scene.h"
#include "renderer/visibility_math.h"
#include "rendergraph/pass_context.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace phosphor {
namespace {
float halfFloat(u16 bits) {
    const u32 exponent = (bits >> 10) & 31, mantissa = bits & 1023;
    float v = exponent == 0 ? std::ldexp(float(mantissa), -24)
              : exponent == 31
                  ? (mantissa ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity())
                  : std::ldexp(float(1024 + mantissa), int(exponent) - 25);
    return bits & 0x8000 ? -v : v;
}
} // namespace
void VisibilityRenderer::addChecks(rg::RenderGraph &g) {
    if (!checks_)
        return;
    using namespace rg;
    const std::array<TextureRef, 8> images = {visibility_, depth_,      outputs_[0], outputs_[1],
                                              outputs_[2], outputs_[3], outputs_[4], outputs_[5]};
    g.addPass(
        "Visibility guide readback", PassType::Blit,
        [&](PassBuilder &b) {
            for (auto t : images)
                b.read(t, Usage::CopySrc, StageBlit);
            b.read(poses_, Usage::CopySrc, StageBlit);
            b.read(scene_.dataRef(), Usage::CopySrc, StageBlit);
            b.setSideEffect();
        },
        [this, images](PassContext &ctx) {
            auto *enc = static_cast<MTL4::ComputeCommandEncoder *>(ctx.encoder());
            for (u32 i = 0; i < images.size(); ++i)
                enc->copyFromTexture(static_cast<MTL::Texture *>(ctx.texture(images[i])), 0, 0,
                                     MTL::Origin::Make(0, 0, 0), MTL::Size::Make(readWidth_, readHeight_, 1),
                                     readbacks_[i], 0, pitches_[i], pitches_[i] * readHeight_);
            enc->copyFromBuffer(scene_.buffers().instances(), 0, currentReadback_, 0,
                                std::min<u64>(poseCapacity_, scene_.buffers().instances()->length()));
            enc->copyFromBuffer(previousInstances_[view_], 0, previousReadback_, 0, poseCapacity_);
        });
}
bool VisibilityRenderer::check(const GpuScene &geometry) {
    if (!checks_)
        return true;
    ++checksCount_;
    u32 idErrors = 0, guideErrors = 0, motionErrors = 0, covered = 0, motionSamples = 0;
    const auto *current = static_cast<const GPUInstance *>(currentReadback_->contents());
    const auto *previous = static_cast<const GPUInstance *>(previousReadback_->contents());
    const auto *a = static_cast<const GPUMeshletCandidate *>(mesh_.frame(slot_).candidates->contents());
    const auto *b = static_cast<const GPUMeshletCandidate *>(mesh_.frame(slot_).bList->contents());
    const bool overflow = *static_cast<const u32 *>(mesh_.frame(slot_).gate->contents()) != 0;
    const glm::mat4 raster = glm::make_mat4(constants_.viewProjection);
    const glm::mat4 nowVP = glm::make_mat4(temporal_.currentViewProjection),
                    oldVP = glm::make_mat4(temporal_.previousViewProjection);
    u32 poseHistoryErrors = 0;
    const auto &expectedPrevious = checkedPreviousPoses_[view_];
    if (temporal_.historyValid && !expectedPrevious.empty()) {
        for (size_t i = 0; i < std::min<size_t>(expectedPrevious.size(), params_.pad); ++i)
            if (std::memcmp(&previous[i], &expectedPrevious[i], sizeof(GPUInstance)) != 0)
                ++poseHistoryErrors;
    }
    float worstMotion = 0;
    for (u32 y = 0; y < params_.height; ++y)
        for (u32 x = 0; x < params_.width; ++x) {
            const auto *row = static_cast<const u8 *>(readbacks_[0]->contents()) + pitches_[0] * y;
            u32 id;
            std::memcpy(&id, row + x * 4, 4);
            const auto halfAt = [&](u32 image, u32 components, u32 component) {
                const auto *data = static_cast<const u8 *>(readbacks_[image]->contents()) + pitches_[image] * y +
                                   (x * components + component) * 2;
                u16 v;
                std::memcpy(&v, data, 2);
                return halfFloat(v);
            };
            float depth;
            std::memcpy(&depth, static_cast<const u8 *>(readbacks_[1]->contents()) + pitches_[1] * y + x * 4, 4);
            const bool shaded = depth > 0;
            if (!shaded) {
                if (id != VISIBILITY_BACKGROUND)
                    ++guideErrors;
                continue;
            }
            ++covered;
            for (u32 image = 2; image <= 5; ++image)
                for (u32 c = 0; c < 4; ++c)
                    if (!std::isfinite(halfAt(image, 4, c)))
                        ++guideErrors;
            const glm::vec3 n(halfAt(3, 4, 0), halfAt(3, 4, 1), halfAt(3, 4, 2));
            if (std::abs(glm::length(n) - 1.0f) > 0.003f)
                ++guideErrors;
            const float roughness = halfAt(3, 4, 3);
            if (roughness < 0.039f || roughness > 1.001f)
                ++guideErrors;
            const glm::vec2 actual(halfAt(6, 2, 0), halfAt(6, 2, 1));
            if (!std::isfinite(actual.x) || !std::isfinite(actual.y))
                ++guideErrors;
            if (overflow)
                continue; // indexed fallback has guides but no visibility ID
            if (id == VISIBILITY_BACKGROUND || visibilityCluster(id) >= 2 * params_.candidateCapacity) {
                ++idErrors;
                continue;
            }
            const u32 cluster = visibilityCluster(id), triangle = visibilityTriangle(id);
            const auto candidate =
                cluster < params_.candidateCapacity ? a[cluster] : b[cluster - params_.candidateCapacity];
            if (candidate.slot >= params_.pad || candidate.meshlet >= geometry.meshlets().size()) {
                ++idErrors;
                continue;
            }
            const auto &instance = current[candidate.slot];
            const auto &m = geometry.meshlets()[candidate.meshlet];
            if (!(instance.flags & INSTANCE_FLAG_VALID) || triangle >= m.triangleCount) {
                ++idErrors;
                continue;
            }
            if ((x % 4) != 0 || (y % 4) != 0)
                continue;
            glm::vec4 local[3], world[3], clip[3];
            const glm::mat4 model = glm::make_mat4(instance.modelMatrix);
            for (u32 i = 0; i < 3; ++i) {
                const u32 vi =
                    geometry.meshletVertices()[m.vertexOffset +
                                               geometry.meshletTriangles()[m.triangleOffset + triangle * 3 + i]];
                const auto &v = geometry.vertices()[vi];
                local[i] = glm::vec4(v.px, v.py, v.pz, 1);
                world[i] = model * local[i];
                clip[i] = raster * world[i];
            }
            const auto weights = visibilityBarycentrics(clip[0].x, clip[0].y, clip[0].w, clip[1].x, clip[1].y,
                                                        clip[1].w, clip[2].x, clip[2].y, clip[2].w, float(x) + 0.5f,
                                                        float(y) + 0.5f, float(params_.width), float(params_.height));
            if (!weights.valid) {
                ++idErrors;
                continue;
            }
            glm::vec2 expected(0);
            const auto &old = previous[candidate.slot];
            if (temporal_.historyValid && old.generation == instance.generation && (old.flags & INSTANCE_FLAG_VALID)) {
                const auto point =
                    local[0] * weights.value[0] + local[1] * weights.value[1] + local[2] * weights.value[2];
                const auto c = nowVP * model * point, p = oldVP * glm::make_mat4(old.modelMatrix) * point;
                if (c.w > 1e-6f && p.w > 1e-6f)
                    expected = (glm::vec2(p) / p.w - glm::vec2(c) / c.w) *
                               glm::vec2(params_.width * 0.5f, -float(params_.height) * 0.5f);
                expected = glm::clamp(expected, glm::vec2(-16384), glm::vec2(16384));
            }
            const float error = glm::length(expected - actual);
            worstMotion = std::max(worstMotion, error);
            ++motionSamples;
            // RG16Float quantization plus independently reconstructed FP matrices.
            // Fixed before the GPU tests: 1/32 pixel or 0.1% of the displacement.
            if (error > std::max(0.03125f, glm::length(expected) * 0.001f))
                ++motionErrors;
        }
    checkedPreviousPoses_[view_].assign(current, current + params_.pad);
    const bool pass = idErrors == 0 && guideErrors == 0 && motionErrors == 0 && poseHistoryErrors == 0;
    if (!pass)
        ++checkFailures_;
    std::printf("VISIBILITY view %u | pixels %u | ids %u guides %u motion %u (%u samples, max %.6f px) | pose history "
                "%u | %s\n",
                view_, covered, idErrors, guideErrors, motionErrors, motionSamples, double(worstMotion),
                poseHistoryErrors, pass ? "PASS" : "FAIL");
    std::fflush(stdout);
    return pass;
}
} // namespace phosphor
