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

        ImGui::Text("FPS %.1f   Frame %.2f ms   GPU %.2f ms", stats.getFPS(), stats.getCpuMs(), info.gpuMs);

        const u32 count = std::min(stats.getSampleCount(), FrameStats::HISTORY_SIZE);
        if (count > 0) {
            char overlay[32];
            std::snprintf(overlay, sizeof(overlay), "Frame %.2f ms", stats.getCpuMs());
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

namespace {

double mib(u64 bytes) { return static_cast<double>(bytes) / (1 << 20); }

ImVec4 levelColor(MemoryBudget::Level level) {
    switch (level) {
    case MemoryBudget::Level::Ok:      return ImVec4(0.55f, 0.85f, 0.55f, 1.0f);
    case MemoryBudget::Level::Warning: return ImVec4(0.95f, 0.75f, 0.30f, 1.0f);
    case MemoryBudget::Level::Over:    return ImVec4(0.95f, 0.35f, 0.30f, 1.0f);
    }
    return ImVec4(1, 1, 1, 1);
}

} // namespace

void UIPanels::drawMemoryPanel(const MemoryPanelInfo& info) {
    ImGui::SetNextWindowPos(ImVec2(350, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(480, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Memory")) {
        ImGui::Text("%s  working set %.1f GiB  engine budget %.1f GiB", info.tier, mib(info.workingSet) / 1024.0,
                    mib(info.engineLimit) / 1024.0);
        ImGui::Text("Device allocated %.1f MiB   GPU allocations %llu", mib(info.deviceAllocated),
                    static_cast<unsigned long long>(info.gpuAllocations));
        ImGui::Text("Memory pressure: %s (%u events)", info.pressure, info.pressureEvents);

        if (ImGui::BeginTable("categories", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Category");
            ImGui::TableSetupColumn("MiB");
            ImGui::TableSetupColumn("Count");
            ImGui::TableSetupColumn("Budget");
            ImGui::TableHeadersRow();
            for (u32 c = 0; c < MEMORY_CATEGORY_COUNT; ++c) {
                const auto& cat = info.categories[c];
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(memoryCategoryName(static_cast<MemoryCategory>(c)));
                ImGui::TableNextColumn();
                ImGui::TextColored(levelColor(cat.level), "%.2f", mib(cat.bytes));
                ImGui::TableNextColumn();
                ImGui::Text("%u", cat.count);
                ImGui::TableNextColumn();
                ImGui::ProgressBar(cat.limit ? static_cast<float>(static_cast<double>(cat.bytes) / cat.limit) : 0.0f,
                                   ImVec2(-1, 0), nullptr);
            }
            ImGui::EndTable();
        }

        ImGui::SeparatorText("Placement heaps");
        for (const auto& h : info.heaps) {
            ImGui::Text("%-9s %6.1f / %6.1f MiB  %4u res  frag %.2f", h.streaming ? "streaming" : "static",
                        mib(h.used), mib(h.size), h.allocations, h.fragmentation);
        }

        ImGui::SeparatorText("Upload rings");
        for (const auto& r : info.rings) {
            ImGui::Text("%-13s %6.1f MiB  in flight %6.2f  peak frame %6.2f  overflows %u", r.name,
                        mib(r.capacity), mib(r.inFlight), mib(r.peakFrame), r.overflows);
        }

        ImGui::SeparatorText("Residency sets");
        for (const auto& r : info.residency) {
            ImGui::Text("%-9s %4u allocations  %7.1f MiB  %u commits", r.name, r.allocations, mib(r.bytes),
                        r.commits);
        }
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

void UIPanels::drawPipelinePanel(const PipelinePanelInfo& info) {
    if (!info.stats) return;
    const pipe::PipelineStats& s = *info.stats;
    ImGui::SetNextWindowPos(ImVec2(10, 480), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Pipelines", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Archive: %s", info.archive);
        ImGui::Text("Entries %u, compile threads %u", info.entries, info.workers);
        ImGui::Text("Archive hits %u, misses %u, unavailable %u (miss rate %.1f%%)", s.archiveHits, s.archiveMisses,
                    s.archiveUnavailable, static_cast<double>(s.archiveMissRate()) * 100.0);
        ImGui::Text("Compiler calls %u: %.1f ms total, max %.1f ms", s.compilerCalls, s.compileMs,
                    static_cast<double>(s.compileMsMax));
        ImGui::Text("Render-thread compiles %u (%.1f ms)", s.renderThreadCompiles, s.renderThreadCompileMs);
        ImGui::Text("Fallbacks %u, fallback frames %llu%s", s.fallbacksServed,
                    static_cast<unsigned long long>(s.fallbackDraws), info.fallback ? "  [drawing fallback]" : "");
        ImGui::Text("Reloads %u (failed %u), failures %u", s.reloads, s.reloadFailures, s.failures);
    }
    ImGui::End();
}

} // namespace phosphor
