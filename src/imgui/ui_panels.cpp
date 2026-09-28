#include "imgui/ui_panels.h"
#include "diagnostics/frame_stats.h"
#include "testbench/testbench.h"

#include <imgui.h>

#include <algorithm>
#include <cstdio>

namespace phosphor {

void UIPanels::drawTestBenchSelector(int& currentBench, bool& changed) {
    changed = false;
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Test Bench", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        if (ImGui::BeginCombo("Scene", testBenchName(currentBench))) {
            for (int i = 0; i < testBenchCount(); ++i) {
                const bool selected = (i == currentBench);
                if (ImGui::Selectable(testBenchName(i), selected) && !selected) {
                    currentBench = i;
                    changed = true;
                }
                if (selected) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::TextDisabled("Press 1-7 to switch quickly");
    }
    ImGui::End();
}

void UIPanels::drawPerformancePanel(const FrameStats& stats, const RendererInfo& info) {
    ImGui::SetNextWindowPos(ImVec2(10, 90), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330, 0), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Performance")) {
        ImGui::Text("%s%s", info.gpuName, info.apple9 ? "" : "  (below Apple9 baseline)");
        ImGui::Text("%u x %u", info.width, info.height);
        ImGui::Separator();

        ImGui::Text("FPS %.1f   CPU %.2f ms   GPU %.2f ms", stats.getFPS(), stats.getCpuMs(), info.gpuMs);

        const u32 count = std::min(stats.getSampleCount(), FrameStats::HISTORY_SIZE);
        if (count > 0) {
            char overlay[32];
            std::snprintf(overlay, sizeof(overlay), "CPU %.2f ms", stats.getCpuMs());
            ImGui::PlotLines("##cpu", stats.getCpuHistory().data(), static_cast<int>(count),
                             0, overlay, 0.0f, 33.3f, ImVec2(0, 40));
            std::snprintf(overlay, sizeof(overlay), "GPU %.2f ms", info.gpuMs);
            ImGui::PlotLines("##gpu", stats.getGpuHistory().data(), static_cast<int>(count),
                             0, overlay, 0.0f, 33.3f, ImVec2(0, 40));
        }

        ImGui::Separator();
        ImGui::Text("Instances  %u", info.instances);
        ImGui::Text("Draws      %u", info.drawBatches);
        ImGui::Text("Triangles  %u", info.triangles);
        ImGui::Text("Meshlets   %u", info.meshlets);
        ImGui::Text("Textures   %u", info.textures);
    }
    ImGui::End();
}

void UIPanels::drawRenderPanel(RenderSettings& settings) {
    ImGui::SetNextWindowPos(ImVec2(10, 380), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Rendering", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        static const char* kModes[] = {"Lit", "Normals", "Base color"};
        ImGui::Combo("View", &settings.debugMode, kModes, IM_ARRAYSIZE(kModes));
        ImGui::SliderFloat("Exposure", &settings.exposure, 0.1f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::Checkbox("VSync", &settings.vsync);
        ImGui::TextDisabled("F1-F3: view modes");
    }
    ImGui::End();
}

} // namespace phosphor
