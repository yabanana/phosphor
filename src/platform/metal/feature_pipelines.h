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
// cache's own fallback object is used.
// ---------------------------------------------------------------------------
class FeaturePipelines {
public:
    /// `slot` identifies one shading pipeline of the caller; `makeDesc` builds its
    /// union-0 descriptor (called only when a request is needed).  `base` is the
    /// caller's handle for union 0.
    template <typename MakeDesc>
    pipe::PipelineHandle serve(PipelineCache& cache, u32 slot, pipe::PipelineHandle base, u32 mask, MakeDesc&& makeDesc) {
        if (mask == 0) return base;
        const pipe::PipelineHandle spec = handleFor(cache, slot, mask, makeDesc);
        if (cache.isFinal(spec)) return spec;
        if (mask != MATERIAL_FEATURES_ALL) {
            const pipe::PipelineHandle all = handleFor(cache, slot, MATERIAL_FEATURES_ALL, makeDesc);
            if (cache.isFinal(all)) return all;
        }
        return spec;
    }

    /// Requests issued so far (tests / logs).
    [[nodiscard]] size_t requests() const { return handles_.size(); }

private:
    template <typename MakeDesc>
    pipe::PipelineHandle handleFor(PipelineCache& cache, u32 slot, u32 mask, MakeDesc& makeDesc) {
        const u64 key = (u64(slot) << 32) | mask;
        auto it = handles_.find(key);
        if (it == handles_.end()) {
            it = handles_.emplace(key, cache.request(pipe::forward::withMaterialFeatures(makeDesc(), mask))).first;
        }
        return it->second;
    }

    std::unordered_map<u64, pipe::PipelineHandle> handles_;
};

} // namespace phosphor
