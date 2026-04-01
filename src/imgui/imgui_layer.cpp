#include "imgui/imgui_layer.h"
#include "core/log.h"
#include "core/window.h"
#include "rhi/vk_device.h"

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>

namespace phosphor {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

ImGuiLayer::ImGuiLayer(VulkanDevice& device, Window& window, VkFormat swapchainFormat)
    : device_(device)
{
    createDescriptorPool();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    setupStyle();

    // --- SDL3 backend init ---
    ImGui_ImplSDL3_InitForVulkan(window.getSDLWindow());

    // --- Vulkan backend init (dynamic rendering, no VkRenderPass) ---
    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance        = device_.getInstance();
    initInfo.PhysicalDevice  = device_.getPhysicalDevice();
    initInfo.Device          = device_.getDevice();
    initInfo.QueueFamily     = device_.getQueueFamilyIndices().graphics.value();
    initInfo.Queue           = device_.getQueues().graphics;
    initInfo.DescriptorPool  = imguiPool_;
    initInfo.MinImageCount   = FRAMES_IN_FLIGHT;
    initInfo.ImageCount      = FRAMES_IN_FLIGHT;
    initInfo.MSAASamples     = VK_SAMPLE_COUNT_1_BIT;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    initInfo.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    initInfo.PipelineRenderingCreateInfo.pColorAttachmentFormats = &swapchainFormat;

    ImGui_ImplVulkan_Init(&initInfo);

    // Upload font atlas
    ImGui_ImplVulkan_CreateFontsTexture();

    initialized_ = true;
    LOG_INFO("ImGui layer initialized (SDL3 + Vulkan dynamic rendering)");
}

ImGuiLayer::~ImGuiLayer() {
    shutdown();
}

// ---------------------------------------------------------------------------
// Per-frame interface
// ---------------------------------------------------------------------------

void ImGuiLayer::newFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

bool ImGuiLayer::processEvent(const SDL_Event& event) {
    ImGui_ImplSDL3_ProcessEvent(&event);
    const ImGuiIO& io = ImGui::GetIO();
    // Return true if ImGui wants to capture this event type
    switch (event.type) {
        case SDL_EVENT_MOUSE_MOTION:
        case SDL_EVENT_MOUSE_BUTTON_DOWN:
        case SDL_EVENT_MOUSE_BUTTON_UP:
        case SDL_EVENT_MOUSE_WHEEL:
            return io.WantCaptureMouse;
        case SDL_EVENT_KEY_DOWN:
        case SDL_EVENT_KEY_UP:
        case SDL_EVENT_TEXT_INPUT:
            return io.WantCaptureKeyboard;
        default:
            return false;
    }
}

void ImGuiLayer::render(VkCommandBuffer cmd, VkImageView targetView, VkExtent2D extent) {
    ImGui::Render();
    ImDrawData* drawData = ImGui::GetDrawData();
    if (!drawData) return;

    // Begin dynamic rendering targeting the swapchain image
    VkRenderingAttachmentInfo colorAttach{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttach.imageView   = targetView;
    colorAttach.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttach.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD; // preserve existing content
    colorAttach.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderInfo.renderArea          = {{0, 0}, extent};
    renderInfo.layerCount          = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments   = &colorAttach;

    vkCmdBeginRendering(cmd, &renderInfo);
    ImGui_ImplVulkan_RenderDrawData(drawData, cmd);
    vkCmdEndRendering(cmd);
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

void ImGuiLayer::shutdown() {
    if (!initialized_) return;

    device_.waitIdle();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    if (imguiPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_.getDevice(), imguiPool_, nullptr);
        imguiPool_ = VK_NULL_HANDLE;
    }

    initialized_ = false;
    LOG_INFO("ImGui layer shut down");
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

void ImGuiLayer::createDescriptorPool() {
    // ImGui needs a descriptor pool for its font atlas and any user textures.
    // Over-allocate slightly to leave room for debug texture displays.
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
    };

    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets       = 64;
    poolInfo.poolSizeCount = static_cast<u32>(std::size(poolSizes));
    poolInfo.pPoolSizes    = poolSizes;

    VK_CHECK(vkCreateDescriptorPool(device_.getDevice(), &poolInfo, nullptr, &imguiPool_));
}

void ImGuiLayer::setupStyle() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding   = 4.0f;
    style.FrameRounding    = 2.0f;
    style.GrabRounding     = 2.0f;
    style.ScrollbarRounding = 3.0f;
    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize  = 0.0f;
    style.Alpha            = 0.95f;

    // Dark color scheme with blue accent
    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg]          = ImVec4(0.06f, 0.06f, 0.08f, 0.94f);
    colors[ImGuiCol_TitleBg]           = ImVec4(0.04f, 0.04f, 0.06f, 1.00f);
    colors[ImGuiCol_TitleBgActive]     = ImVec4(0.08f, 0.10f, 0.18f, 1.00f);
    colors[ImGuiCol_FrameBg]           = ImVec4(0.10f, 0.10f, 0.14f, 1.00f);
    colors[ImGuiCol_FrameBgHovered]    = ImVec4(0.15f, 0.15f, 0.22f, 1.00f);
    colors[ImGuiCol_FrameBgActive]     = ImVec4(0.20f, 0.20f, 0.30f, 1.00f);
    colors[ImGuiCol_Button]            = ImVec4(0.14f, 0.18f, 0.30f, 1.00f);
    colors[ImGuiCol_ButtonHovered]     = ImVec4(0.20f, 0.26f, 0.42f, 1.00f);
    colors[ImGuiCol_ButtonActive]      = ImVec4(0.24f, 0.32f, 0.52f, 1.00f);
    colors[ImGuiCol_Header]            = ImVec4(0.14f, 0.18f, 0.30f, 1.00f);
    colors[ImGuiCol_HeaderHovered]     = ImVec4(0.20f, 0.26f, 0.42f, 1.00f);
    colors[ImGuiCol_HeaderActive]      = ImVec4(0.24f, 0.32f, 0.52f, 1.00f);
    colors[ImGuiCol_SliderGrab]        = ImVec4(0.30f, 0.40f, 0.65f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]  = ImVec4(0.36f, 0.48f, 0.78f, 1.00f);
    colors[ImGuiCol_CheckMark]         = ImVec4(0.40f, 0.55f, 0.85f, 1.00f);
    colors[ImGuiCol_Tab]               = ImVec4(0.10f, 0.12f, 0.20f, 1.00f);
    colors[ImGuiCol_TabHovered]        = ImVec4(0.20f, 0.26f, 0.42f, 1.00f);
    colors[ImGuiCol_TabSelected]       = ImVec4(0.16f, 0.22f, 0.36f, 1.00f);
    colors[ImGuiCol_PlotLines]         = ImVec4(0.40f, 0.55f, 0.85f, 1.00f);
    colors[ImGuiCol_PlotHistogram]     = ImVec4(0.40f, 0.55f, 0.85f, 1.00f);
}

} // namespace phosphor
