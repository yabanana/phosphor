#include "diagnostics/debug_utils.h"
#include <mutex>

namespace phosphor {

// Cached function pointers (loaded once, thread-safe)
static PFN_vkCmdBeginDebugUtilsLabelEXT    pfnCmdBeginLabel = nullptr;
static PFN_vkCmdEndDebugUtilsLabelEXT      pfnCmdEndLabel = nullptr;
static PFN_vkCmdInsertDebugUtilsLabelEXT   pfnCmdInsertLabel = nullptr;
static PFN_vkQueueBeginDebugUtilsLabelEXT  pfnQueueBeginLabel = nullptr;
static PFN_vkQueueEndDebugUtilsLabelEXT    pfnQueueEndLabel = nullptr;
static PFN_vkSetDebugUtilsObjectNameEXT    pfnSetObjectName = nullptr;
static std::once_flag loadOnce;

static void loadFunctions() {
    pfnCmdBeginLabel   = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(vkGetInstanceProcAddr(nullptr, "vkCmdBeginDebugUtilsLabelEXT"));
    pfnCmdEndLabel     = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(vkGetInstanceProcAddr(nullptr, "vkCmdEndDebugUtilsLabelEXT"));
    pfnCmdInsertLabel  = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(vkGetInstanceProcAddr(nullptr, "vkCmdInsertDebugUtilsLabelEXT"));
    pfnQueueBeginLabel = reinterpret_cast<PFN_vkQueueBeginDebugUtilsLabelEXT>(vkGetInstanceProcAddr(nullptr, "vkQueueBeginDebugUtilsLabelEXT"));
    pfnQueueEndLabel   = reinterpret_cast<PFN_vkQueueEndDebugUtilsLabelEXT>(vkGetInstanceProcAddr(nullptr, "vkQueueEndDebugUtilsLabelEXT"));
}

static VkDebugUtilsLabelEXT makeLabel(const char* name, glm::vec4 color) {
    VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    label.pLabelName = name;
    label.color[0] = color.r;
    label.color[1] = color.g;
    label.color[2] = color.b;
    label.color[3] = color.a;
    return label;
}

void DebugUtils::setObjectName(VkDevice device, u64 objectHandle, VkObjectType type, const char* name) {
    if (!pfnSetObjectName) {
        pfnSetObjectName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
            vkGetDeviceProcAddr(device, "vkSetDebugUtilsObjectNameEXT"));
    }
    if (!pfnSetObjectName) return;

    VkDebugUtilsObjectNameInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    info.objectType = type;
    info.objectHandle = objectHandle;
    info.pObjectName = name;
    pfnSetObjectName(device, &info);
}

void DebugUtils::beginLabel(VkCommandBuffer cmd, const char* name, glm::vec4 color) {
    std::call_once(loadOnce, loadFunctions);
    if (!pfnCmdBeginLabel) return;
    auto label = makeLabel(name, color);
    pfnCmdBeginLabel(cmd, &label);
}

void DebugUtils::endLabel(VkCommandBuffer cmd) {
    if (!pfnCmdEndLabel) return;
    pfnCmdEndLabel(cmd);
}

void DebugUtils::insertLabel(VkCommandBuffer cmd, const char* name, glm::vec4 color) {
    std::call_once(loadOnce, loadFunctions);
    if (!pfnCmdInsertLabel) return;
    auto label = makeLabel(name, color);
    pfnCmdInsertLabel(cmd, &label);
}

void DebugUtils::beginQueueLabel(VkQueue queue, const char* name, glm::vec4 color) {
    std::call_once(loadOnce, loadFunctions);
    if (!pfnQueueBeginLabel) return;
    auto label = makeLabel(name, color);
    pfnQueueBeginLabel(queue, &label);
}

void DebugUtils::endQueueLabel(VkQueue queue) {
    if (!pfnQueueEndLabel) return;
    pfnQueueEndLabel(queue);
}

} // namespace phosphor
