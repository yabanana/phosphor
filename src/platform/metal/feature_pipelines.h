#pragma once

#include "core/types.h"
#include "pipeline/forward_variants.h"
#include "pipeline/pipeline_desc.h"
#include "pipeline/pipeline_registry.h"
#include "platform/metal/pipeline_cache.h"

#include <unordered_map>

namespace phosphor {

// ---------------------------------------------------------------------------
// FeaturePipelines (F15.1, ticket 05; ADR-0003): chooses, per frame, the shading
// pipeline of a scene with material features.
//
// The shading pipelines (forward, visibility resolve, ...) are requested as
// before for a scene without features (feature union 0: the very same
// descriptors, hence the same keys and archive entries).  For a union U != 0
// serve() additionally requests
//   - the pipeline specialised with the reserved constant MATERIAL_FEATURES = U
//   - the all-lobes pipeline (MATERIAL_FEATURES = every bit), the reference
//     that serves while the specialisation compiles (never the base pipeline,
//     which has no extension lobes).
// Requests happen on first use per (slot, union) and compile on the cache's
// workers; the render thread never compiles (O9).  If neither is final yet the
// cache's fallback object is used when it has one (render pipelines); otherwise
// the base pipeline serves (compute pipelines, which have no fallback object).
// ---------------------------------------------------------------------------
class FeaturePipelines {
public:
    /// `slot` identifies one shading pipeline of the caller; `makeDesc` builds its
    /// union-0 descriptor (called only when a request is needed).  `base` is the
    /// caller's handle for union 0.
    template <typename MakeDesc>
    pipe::PipelineHandle serve(PipelineCache& cache, u32 slot, pipe::PipelineHandle base, u32 mask, MakeDesc&& makeDesc) {
        if (mask == 0) return base;
        // F15.1 (ticket 22): FEATURE_UNION_TEXTURE_TRANSFORM rides on the mask (SceneStore::pipelineFeatureUnion): the lobe bits keep
        // their meaning, the marker becomes the TEXTURE_TRANSFORM constant of the specialised and of the all-lobes pipeline.
        const u32 lobes = mask & MATERIAL_FEATURES_ALL;
        const u32 tt = mask & FEATURE_UNION_TEXTURE_TRANSFORM;
        const pipe::PipelineHandle spec = handleFor(cache, slot, mask, makeDesc);
        if (cache.isFinal(spec)) return spec;
        pipe::PipelineHandle all = pipe::INVALID_PIPELINE;
        if (lobes != 0 && lobes != MATERIAL_FEATURES_ALL) {
            all = handleFor(cache, slot, MATERIAL_FEATURES_ALL | tt, makeDesc);
            if (cache.isFinal(all)) return all;
        }
        // Render pipelines get a flexible fallback object while they compile;
        // compute pipelines get nothing (binding a null compute pipeline crashes
        // the driver). Until one of the two has an object, the scene's base
        // pipeline (no extension lobes) serves those frames, counted as a
        // fallback use (transient after a load or a feature toggle).
        if (cache.available(spec)) return spec;
        if (all != pipe::INVALID_PIPELINE && cache.available(all)) return all;
        cache.noteFallbackUse();
        return base;
    }

    /// Requests issued so far (tests / logs).
    [[nodiscard]] size_t requests() const { return handles_.size(); }

private:
    template <typename MakeDesc>
    pipe::PipelineHandle handleFor(PipelineCache& cache, u32 slot, u32 mask, MakeDesc& makeDesc) {
        const u64 key = (u64(slot) << 32) | mask;
        auto it = handles_.find(key);
        if (it == handles_.end()) {
            pipe::PipelineDesc desc = pipe::forward::withMaterialFeatures(makeDesc(), mask & MATERIAL_FEATURES_ALL);
            if (mask & FEATURE_UNION_TEXTURE_TRANSFORM) desc = pipe::forward::withTextureTransform(desc);
            it = handles_.emplace(key, cache.request(desc)).first;
        }
        return it->second;
    }

    std::unordered_map<u64, pipe::PipelineHandle> handles_;
};

// ---------------------------------------------------------------------------
// TransmissionPipelines (F15 ticket 24): the kernels that carry the transmittance of glass (the sun RT visibility with its 124-byte payload,
// the temporal moments with their RGB history) are requested in their cheap form (shadow_sun_rt_lite, shadow_temporal_scalar); a scene whose
// feature union has a transmissive material (MATERIAL_FEATURE_TRANSMISSION) is served by the ticket-14/15 kernel, requested on first use.
// While that pipeline is not available the cheap one serves (no coloured transmittance, counted as a fallback use): transient after
// loading such a scene, never in a scene without transmission.
// ---------------------------------------------------------------------------
class TransmissionPipelines {
public:
    template <typename MakeFull>
    pipe::PipelineHandle serve(PipelineCache& cache, u32 slot, pipe::PipelineHandle cheap, bool transmissive, MakeFull&& makeFull) {
        if (!transmissive) return cheap;
        auto it = full_.find(slot);
        if (it == full_.end()) it = full_.emplace(slot, cache.request(makeFull())).first;
        if (cache.isFinal(it->second) || cache.available(it->second)) return it->second;
        cache.noteFallbackUse();
        return cheap;
    }
    [[nodiscard]] size_t requests() const { return full_.size(); }

private:
    std::unordered_map<u32, pipe::PipelineHandle> full_;
};

} // namespace phosphor
