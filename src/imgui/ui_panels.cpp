#include "imgui/ui_panels.h"
#include "diagnostics/frame_stats.h"
#include "diagnostics/gpu_profiler.h"
#include "diagnostics/resource_tracker.h"
#include "diagnostics/renderdoc_capture.h"
#include "diagnostics/debug_overlay.h"

#include <imgui.h>
#include <cstdio>
#include <algorithm>

namespace phosphor {

// ---------------------------------------------------------------------------
// Forward-declare testBenchName from testbench.h to avoid a circular header.
// The linker resolves this from testbench.cpp.
// ---------------------------------------------------------------------------
extern const char* testBenchName(int index);
extern int testBenchCount();

// ---------------------------------------------------------------------------
// TestBench Selector
// ---------------------------------------------------------------------------

void UIPanels::drawTestBenchSelector(int& currentBench, bool& changed) {
    changed = false;
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(280, 0), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Test Bench", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        const int count = testBenchCount();
        const char* currentName = testBenchName(currentBench);

        if (ImGui::BeginCombo("Scene", currentName)) {
            for (int i = 0; i < count; ++i) {
                bool selected = (i == currentBench);
                if (ImGui::Selectable(testBenchName(i), selected)) {
                    if (i != currentBench) {
                        currentBench = i;
                        changed = true;
                    }
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        ImGui::TextDisabled("Press 1-7 to switch quickly");
    }
    ImGui::End();
}

// ---------------------------------------------------------------------------
// Performance Panel
// ---------------------------------------------------------------------------

void UIPanels::drawPerformancePanel(const FrameStats& stats, const GpuProfiler& profiler) {
    ImGui::SetNextWindowPos(ImVec2(10, 80), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Performance")) {
        // --- Headline numbers ---
        ImGui::Text("FPS: %.1f", stats.getFPS());
        ImGui::Text("CPU: %.2f ms", stats.getCpuMs());
        ImGui::Text("GPU: %.2f ms", stats.getGpuMs());
        ImGui::Separator();

        // --- FPS graph ---
        {
            const auto& hist = stats.getFpsHistory();
            u32 count = stats.getSampleCount();
            if (count > 0) {
                float maxFps = *std::max_element(hist.begin(),
                    hist.begin() + std::min(count, static_cast<u32>(hist.size())));
                char overlay[32];
                std::snprintf(overlay, sizeof(overlay), "%.0f fps", stats.getFPS());
                ImGui::PlotLines("##fps", hist.data(), static_cast<int>(count),
                                 0, overlay, 0.0f, maxFps * 1.2f, ImVec2(0, 50));
            }
        }

        // --- CPU / GPU time graphs ---
        {
            const auto& cpuHist = stats.getCpuHistory();
            const auto& gpuHist = stats.getGpuHistory();
            u32 count = stats.getSampleCount();
            if (count > 0) {
                char cpuOverlay[32];
                std::snprintf(cpuOverlay, sizeof(cpuOverlay), "CPU %.2f ms", stats.getCpuMs());
                ImGui::PlotLines("##cpu", cpuHist.data(), static_cast<int>(count),
                                 0, cpuOverlay, 0.0f, 33.3f, ImVec2(0, 35));

                char gpuOverlay[32];
                std::snprintf(gpuOverlay, sizeof(gpuOverlay), "GPU %.2f ms", stats.getGpuMs());
                ImGui::PlotLines("##gpu", gpuHist.data(), static_cast<int>(count),
                                 0, gpuOverlay, 0.0f, 33.3f, ImVec2(0, 35));
            }
        }

        ImGui::Separator();

        // --- Per-pass GPU timings ---
        if (ImGui::TreeNodeEx("Pass Timings", ImGuiTreeNodeFlags_DefaultOpen)) {
            const auto& timings = profiler.getPassTimings();
            for (const auto& [name, timing] : timings) {
                ImGui::Text("%-20s %6.2f ms (avg %6.2f, max %6.2f)",
                            name.c_str(), timing.gpuMs, timing.avgMs, timing.maxMs);
            }
            ImGui::Text("%-20s %6.2f ms", "TOTAL", profiler.getTotalGpuMs());
            ImGui::TreePop();
        }
    }
    ImGui::End();
}

// ---------------------------------------------------------------------------
// Resource Panel
// ---------------------------------------------------------------------------

void UIPanels::drawResourcePanel(const ResourceTracker& tracker) {
    ImGui::SetNextWindowPos(ImVec2(10, 400), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 0), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("GPU Resources")) {
        ImGui::Text("Total: %u resources, %.1f MB",
                     tracker.getResourceCount(), tracker.getTotalAllocatedMB());
        ImGui::Separator();

        for (u32 i = 0; i < static_cast<u32>(TrackedResourceType::COUNT); ++i) {
            auto type = static_cast<TrackedResourceType>(i);
            u32 count = tracker.getCountByType(type);
            if (count > 0) {
                ImGui::Text("  %-18s %u", ResourceTracker::typeName(type), count);
            }
        }
    }
    ImGui::End();
}

// ---------------------------------------------------------------------------
// Debug Panel
// ---------------------------------------------------------------------------

void UIPanels::drawDebugPanel(OverlayMode& overlayMode, AAMode& aaMode,
                               float& exposure, bool& ddgiEnabled,
                               bool& restirEnabled) {
    ImGui::SetNextWindowPos(ImVec2(10, 560), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300, 0), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Debug & Rendering")) {
        // --- Overlay mode ---
        if (ImGui::TreeNodeEx("Debug Overlays", ImGuiTreeNodeFlags_DefaultOpen)) {
            int current = static_cast<int>(overlayMode);
            for (int i = 0; i < static_cast<int>(OverlayMode::COUNT); ++i) {
                if (ImGui::RadioButton(overlayModeName(static_cast<OverlayMode>(i)), current == i)) {
                    current = i;
                }
                if (i % 3 != 2 && i < static_cast<int>(OverlayMode::COUNT) - 1) {
                    ImGui::SameLine();
                }
            }
            overlayMode = static_cast<OverlayMode>(current);
            ImGui::TreePop();
        }

        ImGui::Separator();

        // --- Anti-aliasing ---
        if (ImGui::TreeNodeEx("Anti-Aliasing", ImGuiTreeNodeFlags_DefaultOpen)) {
            int aaInt = static_cast<int>(aaMode);
            ImGui::RadioButton("None", &aaInt, 0); ImGui::SameLine();
            ImGui::RadioButton("FXAA", &aaInt, 1); ImGui::SameLine();
            ImGui::RadioButton("TAA",  &aaInt, 2);
            aaMode = static_cast<AAMode>(aaInt);
            ImGui::TreePop();
        }

        ImGui::Separator();

        // --- Tone mapping ---
        if (ImGui::TreeNodeEx("Tone Mapping", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Exposure", &exposure, 0.1f, 10.0f, "%.2f");
            if (ImGui::Button("Reset##exposure")) exposure = 1.0f;
            ImGui::TreePop();
        }

        ImGui::Separator();

        // --- Lighting features ---
        if (ImGui::TreeNodeEx("Lighting", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("ReSTIR Direct", &restirEnabled);
            ImGui::Checkbox("DDGI Global Illumination", &ddgiEnabled);
            ImGui::TreePop();
        }
    }
    ImGui::End();
}

// ---------------------------------------------------------------------------
// RenderDoc Panel
// ---------------------------------------------------------------------------

void UIPanels::drawRenderDocPanel(RenderDocCapture& renderdoc) {
    ImGui::SetNextWindowPos(ImVec2(10, 750), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(250, 0), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("RenderDoc")) {
        if (renderdoc.isAvailable()) {
            if (ImGui::Button("Capture Frame (F12)")) {
                renderdoc.triggerCapture();
            }
            ImGui::Text("Captures taken: %u", renderdoc.getCaptureCount());
            if (renderdoc.isCapturing()) {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.2f, 1.0f), "CAPTURING...");
            }
        } else {
            ImGui::TextDisabled("RenderDoc not attached.");
            ImGui::TextDisabled("Launch from RenderDoc to enable.");
        }
    }
    ImGui::End();
}

} // namespace phosphor
