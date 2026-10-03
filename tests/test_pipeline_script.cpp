// The committed pipelines script (shaders/pipelines.mtl4-json, F3.4) must
// cover every pipeline the engine can request, or the archive built from it
// misses at runtime.  It goes stale when variants.def or a pipeline's
// functions/constants/output state change: this test catches it on every
// platform (the harvest itself needs the GPU: tools/harvest_pipelines.sh).
#include "pipeline/forward_variants.h"

#include <doctest/doctest.h>
#include <json.hpp> // nlohmann::json, bundled with tinygltf

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace phosphor;
using nlohmann::json;

namespace {

struct Function {
    std::string name;
    std::map<u32, u32> constants; // index -> value bits (bool: 0/1)
    bool specialised = false;
};

std::string stripPrefix(const std::string& ref) { return ref.rfind("fnd:", 0) == 0 ? ref.substr(4) : ref; }

struct Script {
    std::map<std::string, Function> functions; // by label
    std::set<std::string> render;              // "vs|fs|constants|format|blend"
    std::set<std::string> compute;             // "kernel"
    std::set<std::string> mesh;                // F6: "object|mesh|fs+constants|format|limits"
    std::set<std::string> tile;

    static std::string key(const Function& f) {
        std::string k = f.name;
        for (const auto& [index, value] : f.constants) k += "," + std::to_string(index) + "=" + std::to_string(value);
        return k;
    }
};

Script loadScript() {
    std::ifstream in(PHOSPHOR_SOURCE_DIR "/shaders/pipelines.mtl4-json");
    REQUIRE(in.good());
    const json doc = json::parse(in);
    Script script;
    const json& fds = doc.at("function_descriptors");
    for (const json& f : fds.at("library_function_descriptors")) {
        script.functions[f.at("label").get<std::string>()] = {f.at("name").get<std::string>(), {}, false};
    }
    if (fds.contains("specialized_function_descriptors")) {
        for (const json& f : fds.at("specialized_function_descriptors")) {
            Function spec = script.functions.at(stripPrefix(f.at("function_descriptor").get<std::string>()));
            spec.specialised = true;
            // A specialisation with no constant (a generic pipeline) has no
            // "constant_values" key at all.
            for (const json& c : f.value("constant_values", json::array())) {
                REQUIRE(c.at("id_type").get<std::string>() == "FunctionConstantIndex");
                const json& value = c.at("value").at("data");
                spec.constants[c.at("id").at("data").get<u32>()] =
                    value.is_boolean() ? (value.get<bool>() ? 1u : 0u) : value.get<u32>();
            }
            script.functions[f.at("label").get<std::string>()] = spec;
        }
    }
    const json& pds = doc.at("pipeline_descriptors");
    for (const json& p : pds.at("render_pipeline_descriptors")) {
        const Function& vs = script.functions.at(stripPrefix(p.at("vertex_function_descriptor").get<std::string>()));
        const Function& fs = script.functions.at(stripPrefix(p.at("fragment_function_descriptor").get<std::string>()));
        const json& color = p.at("color_attachments").at(0);
        const std::string blend = color.value("blending_state", std::string("Disabled"));
        // The serializer records a function without function constants as a
        // plain library function even when it was specialised, so only the
        // fragment function carries constants in the key.
        script.render.insert(vs.name + "|" + Script::key(fs) + "|" +
                             color.at("pixel_format").get<std::string>() + "|" + blend);
    }
    for (const json& p : pds.value("mesh_render_pipeline_descriptors", json::array())) {
        const auto name = [&](const char* field) {
            return p.contains(field) ? script.functions.at(stripPrefix(p.at(field).get<std::string>())).name : std::string();
        };
        const Function& fs = script.functions.at(stripPrefix(p.at("fragment_function_descriptor").get<std::string>()));
        script.mesh.insert(name("object_function_descriptor") + "|" + name("mesh_function_descriptor") + "|" +
                           Script::key(fs) + "|" + p.at("color_attachments").at(0).at("pixel_format").get<std::string>() +
                           "|" + std::to_string(p.value("max_total_threads_per_object_threadgroup", 0)) + "," +
                           std::to_string(p.value("max_total_threads_per_mesh_threadgroup", 0)) + "," +
                           std::to_string(p.value("payload_memory_length", 0)) + "," +
                           std::to_string(p.value("max_total_threadgroups_per_mesh_grid", 0)));
    }
    for (const json& p : pds.at("compute_pipeline_descriptors")) {
        script.compute.insert(Script::key(
            script.functions.at(stripPrefix(p.at("compute_function_descriptor").get<std::string>()))));
    }
    for (const json &p : pds.value("tile_render_pipeline_descriptors", json::array())) {
        script.tile.insert(
            Script::key(script.functions.at(stripPrefix(p.at("tile_function_descriptor").get<std::string>()))));
    }
    return script;
}

/// The script key of a render pipeline as the engine requests it (the
/// forward constants live in the fragment function).
std::string renderKey(const pipe::PipelineDesc& desc, const char* format, const char* blend) {
    Function fs{desc.functions[1], {}, true};
    for (u32 i = 0; i < desc.constantCount; ++i) fs.constants[desc.constants[i].index] = desc.constants[i].bits;
    return desc.functions[0] + "|" + Script::key(fs) + "|" + format + "|" + blend;
}

} // namespace

TEST_CASE("pipelines script: covers every forward variant and the generic pipeline") {
    const Script script = loadScript();
    u32 missing = 0;
    for (u32 i = 0; i < pipe::forward::variantCount(); ++i) {
        const pipe::PipelineDesc desc = pipe::forward::pipelineDesc(pipe::forward::variantAt(i), rg::Format::BGRA8Srgb);
        if (!script.render.count(renderKey(desc, "BGRA8Unorm_sRGB", "Disabled"))) {
            ++missing;
            MESSAGE("variant " << i << " missing from shaders/pipelines.mtl4-json (run tools/harvest_pipelines.sh)");
        }
    }
    CHECK(missing == 0);
    CHECK(script.render.count(renderKey(pipe::forward::genericDesc(rg::Format::BGRA8Srgb), "BGRA8Unorm_sRGB",
                                        "Disabled")) == 1);
}

TEST_CASE("pipelines script: covers the F6 mesh path (every variant, generic, debug, kernels)") {
    const Script script = loadScript();
    // Must match platform/metal/mesh_renderer.cpp meshDesc() and meshlet_layout.h.
    const auto meshKey = [](const pipe::PipelineDesc& forward, const char* mesh, const char* fsName) {
        Function fs{fsName, {}, true};
        if (std::string(fsName) == "forward_fs") {
            for (u32 i = 0; i < forward.constantCount; ++i) fs.constants[forward.constants[i].index] = forward.constants[i].bits;
        }
        return std::string("meshlet_object|") + mesh + "|" + Script::key(fs) + "|BGRA8Unorm_sRGB|32,128,384,32";
    };
    u32 missing = 0;
    for (u32 i = 0; i < pipe::forward::variantCount(); ++i) {
        const pipe::PipelineDesc desc = pipe::forward::pipelineDesc(pipe::forward::variantAt(i), rg::Format::BGRA8Srgb);
        if (!script.mesh.count(meshKey(desc, "meshlet_mesh", "forward_fs"))) {
            ++missing;
            MESSAGE("mesh variant " << i << " missing from shaders/pipelines.mtl4-json (run tools/harvest_pipelines.sh)");
        }
    }
    CHECK(missing == 0);
    const pipe::PipelineDesc generic = pipe::forward::genericDesc(rg::Format::BGRA8Srgb);
    CHECK(script.mesh.count(meshKey(generic, "meshlet_mesh", "forward_fs")) == 1);
    CHECK(script.mesh.count(meshKey(generic, "meshlet_mesh_debug", "meshlet_debug_fs")) == 1);
    pipe::PipelineDesc view;
    view.functions = {"hiz_view_vs", "hiz_view_fs"};
    CHECK(script.render.count(renderKey(view, "BGRA8Unorm_sRGB", "Disabled")) == 1);
    for (const char* kernel : {"meshlet_cand_count", "meshlet_cand_scan", "meshlet_cand_write", "meshlet_b_count",
                               "meshlet_b_scan", "meshlet_b_write", "hiz_level0", "hiz_reduce_simd",
                               "hiz_reduce_sampler"}) {
        CAPTURE(kernel);
        CHECK(script.compute.count(kernel) == 1);
    }
}

TEST_CASE("pipelines script: covers ImGui and the F2 self-check pipelines") {
    const Script script = loadScript();
    pipe::PipelineDesc imgui;
    imgui.functions = {"imgui_vs", "imgui_fs"};
    CHECK(script.render.count(renderKey(imgui, "BGRA8Unorm_sRGB", "Enabled")) == 1);
    pipe::PipelineDesc raster;
    raster.functions = {"debug_vs", "debug_fs"};
    CHECK(script.render.count(renderKey(raster, "R32Uint", "Disabled")) == 1);
    for (const char* kernel : {"debug_fill", "debug_reduce", "debug_expand", "debug_checksum", "async_seed",
                               "async_reduce", "async_consume", "known_cost",
                               "timestamp_anchor"}) {
        CAPTURE(kernel);
        CHECK(script.compute.count(kernel) == 1);
    }
}

TEST_CASE("pipelines script: covers the F4.7 overlay pipelines") {
    const Script script = loadScript();
    const auto render = [&](const char* vs, const char* fs, const char* format, const char* blend) {
        pipe::PipelineDesc desc;
        desc.functions = {vs, fs};
        return script.render.count(renderKey(desc, format, blend));
    };
    CHECK(render("overlay_vs", "overdraw_fs", "R16Float", "Disabled") == 1);
    CHECK(render("overlay_vs", "lightcount_fs", "R16Float", "Disabled") == 1);
    CHECK(render("overlay_composite_vs", "overlay_composite_fs", "BGRA8Unorm_sRGB", "Enabled") == 1);
    CHECK(script.compute.count("overlay_tilecost") == 1);
}

TEST_CASE("pipelines script: covers visibility, HDR and optional F7 experiments") {
    const Script script = loadScript();
    for (const char *name :
         {"visibility_clear", "visibility_classify", "visibility_resolve", "visibility_adaptive", "visibility_history",
          "exposure_clear", "exposure_histogram", "exposure_reduce", "post_native", "post_curve_probe"}) {
        CAPTURE(name);
        CHECK(script.compute.count(name) == 1);
    }
    for (u32 cls = 0; cls < 4; ++cls)
        CHECK(script.compute.count("visibility_resolve,21=" + std::to_string(cls)) == 1);
    CHECK(script.tile.count("visibility_tile") == 1);
    const auto render = [&](const char *vs, const char *fs, const char *format) {
        pipe::PipelineDesc d;
        d.functions = {vs, fs};
        return script.render.count(renderKey(d, format, "Disabled"));
    };
    CHECK(render("visibility_present_vs", "visibility_present_fs", "BGRA8Unorm_sRGB") == 1);
    CHECK(render("forward_surface_vs", "forward_surface_fs", "RGBA16Float") == 1);
    CHECK(render("post_vs", "post_fs", "BGRA8Unorm_sRGB") == 1);
    CHECK(render("post_vs", "post_fs", "RGBA16Float") == 1);
    CHECK(render("post_vs", "post_capture_fs", "BGRA8Unorm_sRGB") == 1);
    for (const auto &entry : script.functions) {
        CHECK_FALSE(entry.second.name.starts_with("BBRNet"));
        CHECK_FALSE(entry.second.name.starts_with("brnet"));
    }
}
