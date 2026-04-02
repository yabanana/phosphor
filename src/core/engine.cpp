#include "core/engine.h"
#include "core/log.h"
#include "core/window.h"
#include "core/input.h"
#include "core/timer.h"
#include <csignal>
#include <atomic>

namespace { std::atomic<bool> g_signalQuit{false}; }
static void signalHandler(int) { g_signalQuit.store(true, std::memory_order_relaxed); }

#include "rhi/vk_device.h"
#include "rhi/vk_allocator.h"
#include "rhi/vk_swapchain.h"
#include "rhi/vk_commands.h"
#include "rhi/vk_sync.h"
#include "rhi/vk_descriptors.h"
#include "rhi/vk_pipeline.h"

#include "scene/ecs.h"
#include "scene/camera.h"
#include "scene/components.h"
#include "scene/texture_manager.h"
#include "renderer/gpu_scene.h"
#include "renderer/visibility_buffer.h"
#include "renderer/mesh_pass.h"
#include "renderer/material_resolve.h"
#include "renderer/tonemap_pass.h"
#include "renderer/composite_pass.h"
#include "renderer/push_constants.h"

#include "diagnostics/gpu_profiler.h"
#include "diagnostics/frame_stats.h"
#include "diagnostics/renderdoc_capture.h"
#include "diagnostics/resource_tracker.h"
#include "diagnostics/debug_utils.h"

#include "imgui/imgui_layer.h"
#include "imgui/ui_panels.h"
#include "testbench/testbench.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cstring>

namespace phosphor {

// ---------------------------------------------------------------------------
// Construction / Destruction
// ---------------------------------------------------------------------------

Engine::Engine(int argc, char* argv[]) {
    (void)argc;
    (void)argv;
    LOG_INFO("Phosphor Vulkan Renderer starting...");
    initSystems();
}

Engine::~Engine() {
    shutdownSystems();
    LOG_INFO("Phosphor shut down.");
}

// ---------------------------------------------------------------------------
// System lifecycle
// ---------------------------------------------------------------------------

void Engine::initSystems() {
    // ---- Core ----
    window_ = std::make_unique<Window>("Phosphor", 1920, 1080);
    input_  = std::make_unique<Input>();
    timer_  = std::make_unique<Timer>();

    // ---- RHI ----
#ifdef PHOSPHOR_ENABLE_VALIDATION
    constexpr bool enableValidation = true;
#else
    constexpr bool enableValidation = false;
#endif

    device_      = std::make_unique<VulkanDevice>(*window_, enableValidation);
    allocator_   = std::make_unique<GpuAllocator>(*device_);
    sync_        = std::make_unique<SyncManager>(*device_);
    commands_    = std::make_unique<CommandManager>(*device_);
    swapchain_   = std::make_unique<Swapchain>(*device_, window_->getExtent());
    descriptors_ = std::make_unique<BindlessDescriptorManager>(*device_);
    pipelines_   = std::make_unique<PipelineManager>(*device_, *descriptors_);

    // ---- Scene ----
    ecs_      = std::make_unique<ECS>();
    textures_ = std::make_unique<TextureManager>(*device_, *allocator_, *descriptors_, *commands_);
    gpuScene_ = std::make_unique<GpuScene>(*device_, *allocator_, *commands_);

    VkExtent2D extent = swapchain_->getExtent();
    float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
    camera_ = std::make_unique<Camera>(glm::radians(60.0f), aspect, 0.1f, 1000.0f);

    // ---- Renderer passes ----
    visBuffer_       = std::make_unique<VisibilityBuffer>(*device_, *allocator_, extent);
    meshPass_        = std::make_unique<MeshPass>(*device_, *pipelines_, *descriptors_);
    materialResolve_ = std::make_unique<MaterialResolve>(*device_, *allocator_, *pipelines_, *descriptors_, extent);
    tonemapPass_     = std::make_unique<TonemapPass>(*device_, *allocator_, *pipelines_, *descriptors_, extent);
    compositePass_   = std::make_unique<CompositePass>(*device_, *pipelines_);

    // ---- Diagnostics ----
    profiler_        = std::make_unique<GpuProfiler>(*device_);
    frameStats_      = std::make_unique<FrameStats>();
    renderdoc_       = std::make_unique<RenderDocCapture>();
    resourceTracker_ = std::make_unique<ResourceTracker>(*allocator_);

    // ---- ImGui ----
    imgui_ = std::make_unique<ImGuiLayer>(*device_, *window_, swapchain_->getFormat());

    // ---- Initial test bench ----
    switchTestBench(currentBenchType_);

    LOG_INFO("Initialization complete. Resolution: %ux%u", extent.width, extent.height);
}

void Engine::shutdownSystems() {
    if (device_) device_->waitIdle();

    // Tear down active bench first (it references ECS/GpuScene)
    if (activeBench_ && ecs_ && gpuScene_) {
        activeBench_->teardown(*ecs_, *gpuScene_);
    }
    activeBench_.reset();

    // Shut down ImGui before Vulkan resources
    if (imgui_) imgui_->shutdown();
    imgui_.reset();

    // Save pipeline cache
    if (pipelines_) pipelines_->saveCacheToDisk();

    // Destroy in reverse order
    resourceTracker_.reset();
    renderdoc_.reset();
    frameStats_.reset();
    profiler_.reset();

    compositePass_.reset();
    tonemapPass_.reset();
    materialResolve_.reset();
    meshPass_.reset();
    visBuffer_.reset();

    textures_.reset();
    gpuScene_.reset();
    camera_.reset();
    ecs_.reset();

    pipelines_.reset();
    descriptors_.reset();
    swapchain_.reset();
    commands_.reset();
    sync_.reset();
    allocator_.reset();
    device_.reset();

    window_.reset();
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

void Engine::run() {
    std::signal(SIGTERM, signalHandler);
    std::signal(SIGINT, signalHandler);
    LOG_INFO("Entering main loop.");

    while (running_ && !window_->shouldClose() && !g_signalQuit.load(std::memory_order_relaxed)) {
        window_->pollEvents();
        timer_->tick();

        float dt = timer_->getDeltaTime();

        // Feed SDL events to input and ImGui
        for (const auto& event : window_->getPendingEvents()) {
            bool consumed = imgui_->processEvent(event);
            if (!consumed) {
                input_->processEvent(event);
            }
        }

        handleInput();
        update(dt);
        render();

        input_->resetFrameState();
    }

    if (device_) device_->waitIdle();
}

// ---------------------------------------------------------------------------
// Input handling
// ---------------------------------------------------------------------------

void Engine::handleInput() {
    // Quit on Escape
    if (input_->isKeyPressed(SDL_SCANCODE_ESCAPE)) {
        running_ = false;
    }

    // Test bench switching with 1-7 keys
    for (int i = 0; i < static_cast<int>(TestBenchType::COUNT); ++i) {
        SDL_Scancode key = static_cast<SDL_Scancode>(SDL_SCANCODE_1 + i);
        if (input_->isKeyPressed(key)) {
            auto type = static_cast<TestBenchType>(i);
            if (type != currentBenchType_) {
                switchTestBench(type);
            }
        }
    }

    // RenderDoc capture on F12
    if (input_->isKeyPressed(SDL_SCANCODE_F12)) {
        renderdoc_->triggerCapture();
    }

    // Toggle debug overlay with F1-F8
    if (input_->isKeyPressed(SDL_SCANCODE_F1)) overlayMode_ = OverlayMode::None;
    if (input_->isKeyPressed(SDL_SCANCODE_F2)) overlayMode_ = OverlayMode::Meshlets;
    if (input_->isKeyPressed(SDL_SCANCODE_F3)) overlayMode_ = OverlayMode::Normals;
    if (input_->isKeyPressed(SDL_SCANCODE_F4)) overlayMode_ = OverlayMode::Depth;
    if (input_->isKeyPressed(SDL_SCANCODE_F5)) overlayMode_ = OverlayMode::Overdraw;
}

// ---------------------------------------------------------------------------
// Update
// ---------------------------------------------------------------------------

void Engine::update(float dt) {
    // Camera
    if (camera_) {
        camera_->updateFPS(*input_, dt);
        camera_->setAspect(static_cast<float>(swapchain_->getExtent().width) /
                           static_cast<float>(swapchain_->getExtent().height));
        camera_->updateMatrices();
    }

    // Active bench
    if (activeBench_) {
        activeBench_->update(dt, *ecs_);
    }

    // Frame stats
    frameStats_->update(*timer_, profiler_->getTotalGpuMs());
    frameStats_->incrementFrame();
}

// ---------------------------------------------------------------------------
// Upload ECS data to GPU buffers
// ---------------------------------------------------------------------------

void Engine::uploadSceneData() {
    auto& transforms = ecs_->getArray<TransformComponent>();
    auto& meshInsts  = ecs_->getArray<MeshInstanceComponent>();
    auto& materials  = ecs_->getArray<MaterialComponent>();
    auto& lights     = ecs_->getArray<LightComponent>();

    // Build GPU instance array
    std::vector<GPUInstance> gpuInstances;
    gpuInstances.reserve(meshInsts.size());

    for (u32 i = 0; i < meshInsts.size(); ++i) {
        EntityID entity = meshInsts.entities()[i];
        const auto& inst = meshInsts.data()[i];

        if (!inst.isVisible()) continue;
        if (!transforms.has(entity)) continue;

        const auto& xform = transforms.get(entity);

        GPUInstance gi{};
        std::memcpy(gi.modelMatrix, &xform.worldMatrix[0][0], sizeof(float) * 16);
        gi.meshIndex     = inst.meshHandle;
        gi.materialIndex = inst.materialIndex;
        gi.flags         = inst.flags;
        gpuInstances.push_back(gi);
    }
    gpuScene_->updateInstances(gpuInstances);

    // Build GPU material array
    std::vector<GPUMaterial> gpuMaterials;
    gpuMaterials.reserve(materials.size());

    for (u32 i = 0; i < materials.size(); ++i) {
        const auto& mat = materials.data()[i];
        GPUMaterial gm{};
        gm.baseColor[0]          = mat.baseColorFactor.r;
        gm.baseColor[1]          = mat.baseColorFactor.g;
        gm.baseColor[2]          = mat.baseColorFactor.b;
        gm.baseColor[3]          = mat.baseColorFactor.a;
        gm.metallic              = mat.metallicFactor;
        gm.roughness             = mat.roughnessFactor;
        gm.normalScale           = mat.normalScale;
        gm.occlusionStrength     = mat.occlusionStrength;
        gm.baseColorTex          = mat.baseColorTexIndex;
        gm.normalTex             = mat.normalTexIndex;
        gm.metallicRoughnessTex  = mat.metallicRoughnessTexIndex;
        gm.occlusionTex          = mat.occlusionTexIndex;
        gm.emissiveTex           = mat.emissiveTexIndex;
        gm.emissive[0]           = mat.emissiveFactor.x;
        gm.emissive[1]           = mat.emissiveFactor.y;
        gm.emissive[2]           = mat.emissiveFactor.z;
        gm.alphaCutoff           = mat.alphaCutoff;
        gpuMaterials.push_back(gm);
    }
    gpuScene_->updateMaterials(gpuMaterials);

    // Build GPU light array
    std::vector<GPULight> gpuLights;
    gpuLights.reserve(lights.size());

    for (u32 i = 0; i < lights.size(); ++i) {
        EntityID entity = lights.entities()[i];
        const auto& lc = lights.data()[i];

        glm::vec3 position{0.0f};
        glm::vec3 direction{0.0f, -1.0f, 0.0f};

        if (transforms.has(entity)) {
            const auto& xform = transforms.get(entity);
            position  = xform.position;
            // Direction from rotation (forward = -Z in local space)
            direction = glm::normalize(glm::mat3(xform.worldMatrix) * glm::vec3(0.0f, 0.0f, -1.0f));
        }

        GPULight gl{};
        gl.type          = static_cast<u32>(lc.type);
        gl.position[0]   = position.x;
        gl.position[1]   = position.y;
        gl.position[2]   = position.z;
        gl.direction[0]  = direction.x;
        gl.direction[1]  = direction.y;
        gl.direction[2]  = direction.z;
        gl.color[0]      = lc.color.r;
        gl.color[1]      = lc.color.g;
        gl.color[2]      = lc.color.b;
        gl.intensity     = lc.intensity;
        gl.range         = lc.range;
        gl.innerCone     = lc.innerConeAngle;
        gl.outerCone     = lc.outerConeAngle;
        gl.shadowMapIndex = lc.shadowMapIndex;
        gpuLights.push_back(gl);
    }
    gpuScene_->updateLights(gpuLights);
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

void Engine::render() {
    renderdoc_->beginFrame();

    auto& frameSync = sync_->getFrameSync(currentFrame_);

    // 1. Acquire swapchain image
    auto [imageIndex, needsRecreate] = swapchain_->acquireNextImage(frameSync.imageAvailable);
    if (needsRecreate) {
        device_->waitIdle();
        VkExtent2D newExtent = window_->getExtent();
        swapchain_->recreate(newExtent);
        visBuffer_->recreate(newExtent);
        materialResolve_->recreate(newExtent);
        tonemapPass_->recreate(newExtent);
        return;
    }

    // 2. Wait for this frame's GPU work + begin command buffer
    if (!sync_->waitForFrame(currentFrame_)) {
        running_ = false; // device lost — stop gracefully
        return;
    }
    commands_->resetPools(currentFrame_);

    VkCommandBuffer cmd = commands_->getCommandBuffer(currentFrame_);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    profiler_->beginFrame(currentFrame_);

    // 3. Upload ECS data to GPU scene buffers
    uploadSceneData();

    VkExtent2D extent = swapchain_->getExtent();

    // 4. Record mesh pass (task/mesh shaders -> visibility buffer)
    {
        profiler_->beginPass(cmd, "MeshPass");
        meshPass_->recordPass(cmd, *gpuScene_, *visBuffer_, *camera_,
                              currentFrame_, exposure_);
        profiler_->endPass(cmd, "MeshPass");
    }

    // 5-6. Skip material resolve and tonemap for now — blit clear color to swapchain
    // The HDR and LDR images stay black, which is fine for pipeline testing

    // 7. Transition swapchain: UNDEFINED -> COLOR_ATTACHMENT for ImGui direct render
    {
        VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        barrier.srcStageMask  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier.srcAccessMask = 0;
        barrier.dstStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.image         = swapchain_->getImage(imageIndex);
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    // 8. ImGui pass (swapchain is in COLOR_ATTACHMENT_OPTIMAL from composite)
    {
        profiler_->beginPass(cmd, "ImGui");
        imgui_->newFrame();

        // Draw all UI panels
        int benchInt = static_cast<int>(currentBenchType_);
        bool benchChanged = false;
        UIPanels::drawTestBenchSelector(benchInt, benchChanged);
        UIPanels::drawPerformancePanel(*frameStats_, *profiler_);
        UIPanels::drawResourcePanel(*resourceTracker_);
        UIPanels::drawDebugPanel(overlayMode_, aaMode_, exposure_, ddgiEnabled_, restirEnabled_);
        UIPanels::drawRenderDocPanel(*renderdoc_);

        if (benchChanged) {
            switchTestBench(static_cast<TestBenchType>(benchInt));
        }

        imgui_->render(cmd, swapchain_->getImageView(imageIndex), extent);
        profiler_->endPass(cmd, "ImGui");
    }

    // 11. Transition swapchain to PRESENT_SRC_KHR
    {
        VkImageMemoryBarrier2 toPresent{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        toPresent.srcStageMask  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toPresent.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toPresent.dstStageMask  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        toPresent.dstAccessMask = 0;
        toPresent.oldLayout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toPresent.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toPresent.image         = swapchain_->getImage(imageIndex);
        toPresent.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

        VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dep.imageMemoryBarrierCount = 1;
        dep.pImageMemoryBarriers    = &toPresent;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    VK_CHECK(vkEndCommandBuffer(cmd));

    // 12. Submit
    VkCommandBufferSubmitInfo cmdSubmit{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    cmdSubmit.commandBuffer = cmd;

    VkSemaphoreSubmitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    waitInfo.semaphore = frameSync.imageAvailable;
    waitInfo.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSemaphoreSubmitInfo signalInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    signalInfo.semaphore = frameSync.renderFinished;
    signalInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submitInfo.commandBufferInfoCount   = 1;
    submitInfo.pCommandBufferInfos      = &cmdSubmit;
    submitInfo.waitSemaphoreInfoCount   = 1;
    submitInfo.pWaitSemaphoreInfos      = &waitInfo;
    submitInfo.signalSemaphoreInfoCount = 1;
    submitInfo.pSignalSemaphoreInfos    = &signalInfo;

    VK_CHECK(vkQueueSubmit2(device_->getQueues().graphics, 1, &submitInfo, frameSync.inFlight));

    // 13. Present
    bool presentNeedsRecreate = swapchain_->present(
        device_->getQueues().present, frameSync.renderFinished, imageIndex);

    if (presentNeedsRecreate || window_->wasResized()) {
        device_->waitIdle();
        VkExtent2D newExtent = window_->getExtent();
        swapchain_->recreate(newExtent);
        visBuffer_->recreate(newExtent);
        materialResolve_->recreate(newExtent);
        tonemapPass_->recreate(newExtent);
    }

    profiler_->resolveQueries(currentFrame_);
    renderdoc_->endFrame();
    currentFrame_ = (currentFrame_ + 1) % FRAMES_IN_FLIGHT;
}

// ---------------------------------------------------------------------------
// Test bench switching
// ---------------------------------------------------------------------------

void Engine::switchTestBench(TestBenchType type) {
    if (device_) device_->waitIdle();

    // Tear down current bench
    if (activeBench_) {
        activeBench_->teardown(*ecs_, *gpuScene_);
        activeBench_.reset();
    }

    currentBenchType_ = type;
    activeBench_ = createTestBench(type);

    LOG_INFO("Switching to test bench: %s", activeBench_->getName());

    activeBench_->setup(*ecs_, *gpuScene_, *textures_);

    // Apply default camera
    CameraSetup camSetup = activeBench_->getDefaultCamera();
    camera_->setPosition(camSetup.position);

    if (camSetup.orbit) {
        camera_->setOrbitMode(camSetup.target, camSetup.distance);
    }

    // Look at target
    glm::vec3 dir = glm::normalize(camSetup.target - camSetup.position);
    float yaw = glm::degrees(std::atan2(dir.x, -dir.z));
    float pitch = glm::degrees(std::asin(dir.y));
    camera_->setYawPitch(yaw - 90.0f, pitch);
    camera_->updateMatrices();
}

} // namespace phosphor
