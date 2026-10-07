cmake_minimum_required(VERSION 3.25)

# Metal 4 requires macOS 26 on Apple silicon.  Must be set before project().
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0" CACHE STRING "Minimum macOS version")
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Target architecture")
endif()

project(phosphor LANGUAGES C CXX)

if(APPLE)
    # Objective-C++ (ImGui's Metal backend) must be enabled before any
    # subdirectory is added, or CMake cannot generate its rules.
    enable_language(OBJC OBJCXX)
endif()

# ---------------------------------------------------------------------------
# Phosphor -- native Metal 4 renderer for Apple Silicon.
#
# Two layers:
#   phosphor_core  API-agnostic C++20 (scene, ECS, meshlets, test benches).
#                  Builds everywhere; this is what runs in Linux CI.
#   phosphor       The macOS app: Metal 4 backend (metal-cpp) + SDL3 + ImGui.
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
    set(PHOSPHOR_APP_DEFAULT ON)
else()
    set(PHOSPHOR_APP_DEFAULT OFF)
endif()

option(PHOSPHOR_BUILD_APP     "Build the macOS Metal application" ${PHOSPHOR_APP_DEFAULT})
option(PHOSPHOR_BUILD_TESTS   "Build unit tests for phosphor_core" ON)
option(PHOSPHOR_METAL_VALIDATION "Enable Metal API/shader validation hooks in the app" ON)

include(cmake/Dependencies.cmake)
# F3.3: generated variant tables (C++ and MSL) in ${CMAKE_BINARY_DIR}/generated.
include(cmake/Variants.cmake)

set(PHOSPHOR_WARNINGS -Wall -Wextra -Wpedantic -Werror=format -Wno-unused-parameter)

# --- Portable core ---------------------------------------------------------
add_library(phosphor_core STATIC
    src/core/file_watcher.cpp
    src/core/input.cpp
    src/core/launch_options.cpp
    src/core/memory/linear_ring.cpp
    src/core/memory/memory_budget.cpp
    src/core/memory/tlsf_allocator.cpp
    src/core/process.cpp
    src/core/timer.cpp
    src/core/worker_pool.cpp
    src/diagnostics/bench_report.cpp
    src/diagnostics/frame_stats.cpp
    src/diagnostics/frame_trace.cpp
    src/diagnostics/overlay_math.cpp
    src/diagnostics/pass_timings.cpp
    src/diagnostics/soc_model.cpp
    src/diagnostics/soc_results.cpp
    src/diagnostics/tracy_gpu.cpp
    src/pipeline/compile_queue.cpp
    src/pipeline/forward_variants.cpp
    src/pipeline/pipeline_desc.cpp
    src/pipeline/pipeline_key.cpp
    src/pipeline/pipeline_registry.cpp
    src/renderer/cull_reference.cpp
    src/renderer/history_registry.cpp
    src/renderer/exposure.cpp
    src/renderer/gpu_scene.cpp
    src/renderer/meshlet_builder.cpp
    src/renderer/rt_proxy.cpp
    src/renderer/rt_scene.cpp
    src/renderer/rt_reference.cpp
    src/renderer/rt_check.cpp
    src/renderer/shadow_settings.cpp
    src/renderer/meshlet_cull_reference.cpp
    src/renderer/meshlet_check.cpp
    src/renderer/scene_check.cpp
    src/renderer/scene_extract.cpp
    src/renderer/scene_store.cpp
    src/renderer/transform_reference.cpp
    src/rendergraph/aliasing.cpp
    src/rendergraph/barrier_plan.cpp
    src/rendergraph/async_probe_reference.cpp
    src/rendergraph/graph_debug_reference.cpp
    src/rendergraph/graph_budget.cpp
    src/rendergraph/graph_dump.cpp
    src/rendergraph/graph_lint.cpp
    src/rendergraph/optimizer/cost_model.cpp
    src/rendergraph/optimizer/optimizer.cpp
    src/rendergraph/optimizer/plan.cpp
    src/rendergraph/render_graph.cpp
    src/rendergraph/scenario.cpp
    src/rendergraph/tbdr_passes.cpp
    src/rendergraph/timing_plan.cpp
    src/scene/camera.cpp
    src/scene/ecs.cpp
    src/scene/gltf_loader.cpp
    src/scene/mesh_processing.cpp
    src/scene/procedural.cpp
    src/scene/texture_manager.cpp
    src/testbench/cornell_box.cpp
    src/testbench/culling_viz.cpp
    src/testbench/many_lights.cpp
    src/testbench/million_instances.cpp
    src/testbench/pbr_grid.cpp
    src/testbench/scene_viewer.cpp
    src/testbench/stress_test.cpp
    src/testbench/testbench.cpp
    src/testbench/torus_demo.cpp
)
add_dependencies(phosphor_core phosphor_variants)
target_include_directories(phosphor_core PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR}/generated)
find_package(Threads REQUIRED)
target_link_libraries(phosphor_core PUBLIC Threads::Threads glm::glm SDL3::SDL3 meshoptimizer tinygltf mikktspace)
target_compile_definitions(phosphor_core PUBLIC
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_RADIANS
    GLM_ENABLE_EXPERIMENTAL
)
target_compile_options(phosphor_core PRIVATE ${PHOSPHOR_WARNINGS})
if(PHOSPHOR_TRACY)
    # PUBLIC: PHOSPHOR_TRACY selects the real macros of core/profile.h in the
    # app and the tests as well.
    target_link_libraries(phosphor_core PUBLIC TracyClient)
    target_compile_definitions(phosphor_core PUBLIC PHOSPHOR_TRACY)
endif()
# Third-party single-header implementations compiled in gltf_loader.cpp.
set_source_files_properties(src/scene/gltf_loader.cpp PROPERTIES COMPILE_OPTIONS "-w")

# --- Tools -------------------------------------------------------------------
# Pixel comparison for visual regression checks of --capture images.
add_executable(image_diff tools/image_diff/image_diff.cpp)
target_link_libraries(image_diff PRIVATE tinygltf) # stb_image / stb_image_write
set_source_files_properties(tools/image_diff/image_diff.cpp PROPERTIES COMPILE_OPTIONS "-w")
# OPT-0.3: SoC cost model CLI (results -> model JSON, engine report -> predictions). Portable.
add_executable(soc_model tools/soc_model/soc_model.cpp)
target_link_libraries(soc_model PRIVATE phosphor_core tinygltf) # tinygltf: nlohmann json.hpp
target_compile_options(soc_model PRIVATE ${PHOSPHOR_WARNINGS})
# OPT-1.1: offline graph optimiser CLI (scenarios -> plans + report). Portable.
add_executable(graph_opt tools/graph_opt/graph_opt.cpp)
target_link_libraries(graph_opt PRIVATE phosphor_core)
target_include_directories(graph_opt PRIVATE ${CMAKE_SOURCE_DIR}/tools)
target_compile_options(graph_opt PRIVATE ${PHOSPHOR_WARNINGS})
# F6 spike S1 (CPU part): meshlet cook comparison over a procedural corpus. Portable.
add_executable(meshlet_cook tools/meshlet_cook/meshlet_cook.cpp)
target_link_libraries(meshlet_cook PRIVATE phosphor_core tinygltf) # tinygltf: nlohmann json.hpp
target_compile_definitions(meshlet_cook PRIVATE PHOSPHOR_BUILD_TYPE="$<CONFIG>")
target_compile_options(meshlet_cook PRIVATE ${PHOSPHOR_WARNINGS})

# --- macOS app ---------------------------------------------------------------
if(PHOSPHOR_BUILD_APP)
    if(NOT APPLE)
        message(FATAL_ERROR "PHOSPHOR_BUILD_APP requires macOS (Metal 4).")
    endif()
    include(cmake/App.cmake)
endif()

# --- Metal syntax check (Linux) ---------------------------------------------
# Type-checks the Metal backend against the real metal-cpp headers using the
# minimal SDK stubs in tools/apple-sdk-stubs.  Nothing is linked or run; this
# exists so changes made on a Linux machine are at least compile-clean.
if(PHOSPHOR_METAL_SYNTAX_CHECK AND NOT APPLE)
    include(cmake/MetalSyntaxCheck.cmake)
endif()

# --- Tests -------------------------------------------------------------------
if(PHOSPHOR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
