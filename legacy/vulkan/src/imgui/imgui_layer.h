#pragma once

#include "core/types.h"
#include "rhi/vk_common.h"
#include <SDL3/SDL_events.h>

namespace phosphor {

class VulkanDevice;
class Window;

// ---------------------------------------------------------------------------
// ImGuiLayer -- manages the Dear ImGui lifecycle with SDL3 + Vulkan backend.
// Uses dynamic rendering (no VkRenderPass objects).
// ---------------------------------------------------------------------------

class ImGuiLayer {
public:
    ImGuiLayer(VulkanDevice& device, Window& window, VkFormat swapchainFormat);
    ~ImGuiLayer();

    ImGuiLayer(const ImGuiLayer&) = delete;
    ImGuiLayer& operator=(const ImGuiLayer&) = delete;

    /// Start a new ImGui frame. Call once per frame before any ImGui::Begin/End.
    void newFrame();

    /// Feed an SDL event to ImGui. Returns true if ImGui consumed the event
    /// (i.e. mouse is over an ImGui window, keyboard captured, etc.).
    bool processEvent(const SDL_Event& event);

    /// Finalize the ImGui frame and record draw commands into the command
    /// buffer. Renders into @p targetView using dynamic rendering.
    void render(VkCommandBuffer cmd, VkImageView targetView, VkExtent2D extent);

    /// Explicit shutdown. Also called by the destructor.
    void shutdown();

private:
    void createDescriptorPool();
    void setupStyle();

    VulkanDevice&   device_;
    VkDescriptorPool imguiPool_  = VK_NULL_HANDLE;
    bool             initialized_ = false;
};

} // namespace phosphor
