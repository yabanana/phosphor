#pragma once

#include "rhi/vk_common.h"
#include "core/types.h"
#include <glm/glm.hpp>

namespace phosphor {

class DebugUtils {
public:
    static void setObjectName(VkDevice device, u64 objectHandle, VkObjectType type, const char* name);
    static void beginLabel(VkCommandBuffer cmd, const char* name, glm::vec4 color = {1, 1, 1, 1});
    static void endLabel(VkCommandBuffer cmd);
    static void insertLabel(VkCommandBuffer cmd, const char* name, glm::vec4 color = {1, 1, 1, 1});

    // Queue labels
    static void beginQueueLabel(VkQueue queue, const char* name, glm::vec4 color = {1, 1, 1, 1});
    static void endQueueLabel(VkQueue queue);
};

// Scoped label helper (RAII)
class ScopedDebugLabel {
public:
    ScopedDebugLabel(VkCommandBuffer cmd, const char* name, glm::vec4 color = {1, 1, 1, 1})
        : cmd_(cmd) { DebugUtils::beginLabel(cmd, name, color); }
    ~ScopedDebugLabel() { DebugUtils::endLabel(cmd_); }
    ScopedDebugLabel(const ScopedDebugLabel&) = delete;
    ScopedDebugLabel& operator=(const ScopedDebugLabel&) = delete;
private:
    VkCommandBuffer cmd_;
};

#define PHOSPHOR_GPU_LABEL(cmd, name) ::phosphor::ScopedDebugLabel _gpu_label_##__LINE__(cmd, name)

} // namespace phosphor
