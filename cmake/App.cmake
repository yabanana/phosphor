# App.cmake -- the macOS Metal 4 application (included only on Apple hosts).

# --- metal-cpp (header-only) ---
add_library(metal_cpp INTERFACE)
target_include_directories(metal_cpp SYSTEM INTERFACE ${metal_cpp_SOURCE_DIR})
target_link_libraries(metal_cpp INTERFACE
    "-framework Foundation"
    "-framework Metal"
    "-framework QuartzCore"
)

# --- Dear ImGui with the SDL3 platform backend ---
# The renderer backend is Phosphor's own (src/imgui/imgui_renderer.cpp, Metal 4).
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
)
target_include_directories(imgui SYSTEM PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC SDL3::SDL3)

# --- Metal toolchain check ---
# Since Xcode 26 the Metal compiler ships as a separate component; fail at
# configure time with the fix instead of halfway through the build.
execute_process(
    COMMAND xcrun -sdk macosx metal --version
    RESULT_VARIABLE _metal_rc
    OUTPUT_QUIET ERROR_QUIET
)
if(NOT _metal_rc EQUAL 0)
    message(FATAL_ERROR
        "The Metal Toolchain is not installed, so shaders cannot be compiled.\n"
        "Install it with:\n"
        "    xcodebuild -downloadComponent MetalToolchain\n"
        "then re-run CMake.")
endif()

# --- Shaders: .metal -> .air -> phosphor.metallib ---
file(GLOB PHOSPHOR_METAL_SHADERS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/shaders/*.metal)
# Headers the shaders include (C++/MSL shared): an edit recompiles every shader.
set(PHOSPHOR_SHADER_HEADERS
    ${CMAKE_SOURCE_DIR}/src/renderer/gpu_types.h
    ${CMAKE_SOURCE_DIR}/src/renderer/gpu_scene_layout.h
    ${CMAKE_SOURCE_DIR}/src/renderer/gpu_queue.h
    ${CMAKE_SOURCE_DIR}/src/renderer/cull_math.h
    ${CMAKE_SOURCE_DIR}/src/renderer/transform_math.h
    ${CMAKE_SOURCE_DIR}/src/renderer/meshlet_layout.h
    ${CMAKE_SOURCE_DIR}/src/renderer/meshlet_cull_math.h
    ${CMAKE_SOURCE_DIR}/src/diagnostics/overlay_math.h)
set(PHOSPHOR_SHADER_OUT ${CMAKE_BINARY_DIR}/shaders)
set(PHOSPHOR_METALLIB ${PHOSPHOR_SHADER_OUT}/phosphor.metallib)
set(PHOSPHOR_METAL_FLAGS -std=metal4.0 -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}
    -I ${CMAKE_SOURCE_DIR}/src -I ${CMAKE_BINARY_DIR}/generated -Wall)
# Source-level shader debugging and profiling in Xcode (Debug/RelWithDebInfo).
# metal-tt cannot translate specialised functions (F3.3 function constants)
# from a metallib with debug info ("cannot find private metadata", measured
# with toolchain 27.1), so the pipeline archive is built only without it.
option(PHOSPHOR_SHADER_DEBUG_INFO "Compile shaders with debug info in Debug/RelWithDebInfo" ON)
set(PHOSPHOR_SHADER_HAS_DEBUG_INFO OFF)
if(PHOSPHOR_SHADER_DEBUG_INFO AND CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo")
    list(APPEND PHOSPHOR_METAL_FLAGS -gline-tables-only -frecord-sources)
    set(PHOSPHOR_SHADER_HAS_DEBUG_INFO ON)
endif()

set(PHOSPHOR_AIR_FILES)
foreach(shader IN LISTS PHOSPHOR_METAL_SHADERS)
    get_filename_component(name ${shader} NAME_WE)
    set(air ${PHOSPHOR_SHADER_OUT}/${name}.air)
    add_custom_command(
        OUTPUT ${air}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PHOSPHOR_SHADER_OUT}
        COMMAND xcrun -sdk macosx metal ${PHOSPHOR_METAL_FLAGS} -c ${shader} -o ${air}
        DEPENDS ${shader} ${PHOSPHOR_SHADER_HEADERS}
                ${PHOSPHOR_VARIANTS_MSL_HEADER} phosphor_variants
        COMMENT "Compiling Metal shader ${name}.metal"
        VERBATIM
    )
    list(APPEND PHOSPHOR_AIR_FILES ${air})
endforeach()

add_custom_command(
    OUTPUT ${PHOSPHOR_METALLIB}
    COMMAND xcrun -sdk macosx metallib ${PHOSPHOR_AIR_FILES} -o ${PHOSPHOR_METALLIB}
    DEPENDS ${PHOSPHOR_AIR_FILES}
    COMMENT "Linking phosphor.metallib"
    VERBATIM
)
# OPT-1.1: offline graph plans (tools/graph_opt) next to the shader library.
set(PHOSPHOR_GRAPH_PLANS ${PHOSPHOR_SHADER_OUT}/graph-plans.json)
add_custom_command(
    OUTPUT ${PHOSPHOR_GRAPH_PLANS}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/shaders/graph-plans.json ${PHOSPHOR_GRAPH_PLANS}
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/graph-plans.json
    COMMENT "Copying graph-plans.json"
    VERBATIM
)
add_custom_target(phosphor_shaders DEPENDS ${PHOSPHOR_METALLIB} ${PHOSPHOR_GRAPH_PLANS})

# F3.4: pipeline archive (metal-tt) built from shaders/pipelines.mtl4-json.
include(cmake/PipelineArchive.cmake)

# F3.6 self-test (--debug-hot-reload): the same shaders with
# PHOSPHOR_HOT_RELOAD_PROBE, whose forward pass outputs a constant colour.
set(PHOSPHOR_PROBE_OUT ${PHOSPHOR_SHADER_OUT}/probe)
set(PHOSPHOR_PROBE_METALLIB ${PHOSPHOR_SHADER_OUT}/hot-reload-probe.metallib)
set(PHOSPHOR_PROBE_AIR_FILES)
foreach(shader IN LISTS PHOSPHOR_METAL_SHADERS)
    get_filename_component(name ${shader} NAME_WE)
    set(air ${PHOSPHOR_PROBE_OUT}/${name}.air)
    add_custom_command(
        OUTPUT ${air}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PHOSPHOR_PROBE_OUT}
        COMMAND xcrun -sdk macosx metal ${PHOSPHOR_METAL_FLAGS} -DPHOSPHOR_HOT_RELOAD_PROBE=1 -c ${shader} -o ${air}
        DEPENDS ${shader} ${PHOSPHOR_SHADER_OUT}/${name}.air
        COMMENT "Compiling hot-reload probe shader ${name}.metal"
        VERBATIM
    )
    list(APPEND PHOSPHOR_PROBE_AIR_FILES ${air})
endforeach()
add_custom_command(
    OUTPUT ${PHOSPHOR_PROBE_METALLIB}
    COMMAND xcrun -sdk macosx metallib ${PHOSPHOR_PROBE_AIR_FILES} -o ${PHOSPHOR_PROBE_METALLIB}
    DEPENDS ${PHOSPHOR_PROBE_AIR_FILES}
    COMMENT "Linking hot-reload-probe.metallib"
    VERBATIM
)
add_custom_target(phosphor_probe_shaders DEPENDS ${PHOSPHOR_PROBE_METALLIB})

# --- Executable ---
add_executable(phosphor
    src/main.cpp
    src/app/engine.cpp
    src/imgui/imgui_renderer.cpp
    src/imgui/ui_panels.cpp
    src/platform/metal/async_compute_probe.cpp
    src/platform/metal/debug_overlays.cpp
    src/platform/metal/frame_capture.cpp
    src/platform/metal/gpu_capture.cpp
    src/platform/metal/gpu_memory.cpp
    src/platform/metal/gpu_scene_buffers.cpp
    src/platform/metal/gpu_scene_check.cpp
    src/platform/metal/gpu_timestamps.cpp
    src/platform/metal/graph_debug_passes.cpp
    src/platform/metal/known_cost_pass.cpp
    src/platform/metal/scenario_passes.cpp
    src/platform/metal/metal_context.cpp
    src/platform/metal/metal_graph_executor.cpp
    src/platform/metal/metal_impl.cpp
    src/platform/metal/memory_pressure.cpp
    src/platform/metal/memory_stress.cpp
    src/platform/metal/metal_texture_manager.cpp
    src/platform/metal/pipeline_cache.cpp
    src/platform/metal/residency_manager.cpp
    src/platform/metal/scene_renderer.cpp
    src/platform/metal/shader_reloader.cpp
    src/platform/metal/transient_heap.cpp
    src/platform/metal/upload_ring.cpp
)
target_link_libraries(phosphor PRIVATE phosphor_core imgui metal_cpp)
if(PHOSPHOR_TRACY)
    # Global operator new/delete replacement: part of the executable (not of
    # phosphor_core) so it is always linked and the unit tests, which count
    # allocations with their own operators, are unaffected.
    target_sources(phosphor PRIVATE src/core/tracy_memory.cpp)
endif()
target_compile_options(phosphor PRIVATE ${PHOSPHOR_WARNINGS})
add_dependencies(phosphor phosphor_shaders phosphor_probe_shaders)
# F3.6 hot reload rebuilds the metallib with exactly these flags ('|'-joined:
# a ';' would split the definition).
string(REPLACE ";" "|" _phosphor_shader_flags "${PHOSPHOR_METAL_FLAGS}")
target_compile_definitions(phosphor PRIVATE
    "PHOSPHOR_SHADER_FLAGS=\"${_phosphor_shader_flags}\""
    "PHOSPHOR_SHADER_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}/shaders\""
    "PHOSPHOR_RENDERER_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}/src/renderer\""
)
if(PHOSPHOR_METAL_VALIDATION)
    # main() enables the Metal API validation layer in Debug builds.
    target_compile_definitions(phosphor PRIVATE $<$<CONFIG:Debug>:PHOSPHOR_METAL_VALIDATION=1>)
endif()

# Development builds are signed ad hoc with get-task-allow so profiling tools
# (Instruments, leaks, heap, malloc_history) can attach without root.  Turn
# off for anything distributed.
option(PHOSPHOR_DEBUGGABLE "Sign the app with the get-task-allow entitlement" ON)
if(PHOSPHOR_DEBUGGABLE)
    add_custom_command(TARGET phosphor POST_BUILD
        COMMAND codesign --force --sign - --entitlements ${CMAKE_SOURCE_DIR}/cmake/debuggable.entitlements
                $<TARGET_FILE:phosphor>
        COMMENT "Signing phosphor with get-task-allow (profiling)"
        VERBATIM
    )
endif()

# --- F2.3 barrier legality spike (measurement tool, not engine code) ---
# Self-contained offscreen Metal 4 tool; see bench/barrier_spike/README.md.
add_executable(barrier_spike
    bench/barrier_spike/barrier_spike.cpp
)
target_link_libraries(barrier_spike PRIVATE metal_cpp)
target_compile_features(barrier_spike PRIVATE cxx_std_20)

# --- F4.1 timestamp spike (measurement tool, not engine code) ---
add_executable(timestamp_spike
    bench/timestamp_spike/timestamp_spike.cpp
)
target_link_libraries(timestamp_spike PRIVATE metal_cpp)
target_compile_features(timestamp_spike PRIVATE cxx_std_20)

# --- OPT-0.1 SoC characterisation suite (measurement tool, not engine code) ---
# bench/soc: one file per benchmark group, registered with SOC_BENCH; MSL in
# bench/soc/shaders compiled at run time.  See bench/soc/README.md.
file(GLOB SOC_BENCH_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/bench/soc/*.cpp
    ${CMAKE_SOURCE_DIR}/bench/soc/*.mm)
execute_process(COMMAND xcrun --show-sdk-version OUTPUT_VARIABLE SOC_SDK_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
add_executable(soc_bench ${SOC_BENCH_SOURCES})
target_link_libraries(soc_bench PRIVATE phosphor_core metal_cpp IOReport
    "-framework IOKit" "-framework CoreFoundation" "-framework CoreML" "-framework Accelerate"
    "-framework AppKit" "-framework QuartzCore")
target_compile_features(soc_bench PRIVATE cxx_std_20)
target_compile_options(soc_bench PRIVATE ${PHOSPHOR_WARNINGS})
target_compile_definitions(soc_bench PRIVATE
    "SOC_SHADER_DIR=\"${CMAKE_SOURCE_DIR}/bench/soc/shaders\""
    "SOC_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\""
    "SOC_SDK_VERSION=\"${SOC_SDK_VERSION}\"")
foreach(src IN LISTS SOC_BENCH_SOURCES)
    if(src MATCHES "\\.mm$")
        set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    endif()
endforeach()
if(PHOSPHOR_DEBUGGABLE)
    add_custom_command(TARGET soc_bench POST_BUILD
        COMMAND codesign --force --sign - --entitlements ${CMAKE_SOURCE_DIR}/cmake/debuggable.entitlements
                $<TARGET_FILE:soc_bench>
        COMMENT "Signing soc_bench with get-task-allow (leaks)"
        VERBATIM
    )
endif()

# --- F5 spikes S2-S7 (measurement tool, not engine code) ---
# The bench/soc harness and runner with the F5 spike benchmarks (ids F5-Sn);
# their MSL lives in bench/f5_spike/shaders.  See bench/f5_spike/README.md.
file(GLOB F5_SPIKE_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/bench/f5_spike/*.cpp)
add_executable(f5_spike ${CMAKE_SOURCE_DIR}/bench/soc/harness.cpp ${CMAKE_SOURCE_DIR}/bench/soc/soc_bench.cpp
    ${F5_SPIKE_SOURCES})
target_include_directories(f5_spike PRIVATE ${CMAKE_SOURCE_DIR}/bench/soc)
target_link_libraries(f5_spike PRIVATE phosphor_core metal_cpp IOReport
    "-framework IOKit" "-framework CoreFoundation" "-framework AppKit" "-framework QuartzCore")
target_compile_features(f5_spike PRIVATE cxx_std_20)
target_compile_options(f5_spike PRIVATE ${PHOSPHOR_WARNINGS})
target_compile_definitions(f5_spike PRIVATE
    "SOC_SHADER_DIR=\"${CMAKE_SOURCE_DIR}/bench/soc/shaders\""
    "F5_SHADER_DIR=\"${CMAKE_SOURCE_DIR}/bench/f5_spike/shaders\""
    "SOC_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\""
    "SOC_RESULTS_DIR=\"${CMAKE_SOURCE_DIR}/bench/results/f5-spike\""
    "SOC_SDK_VERSION=\"${SOC_SDK_VERSION}\"")
if(PHOSPHOR_DEBUGGABLE)
    add_custom_command(TARGET f5_spike POST_BUILD
        COMMAND codesign --force --sign - --entitlements ${CMAKE_SOURCE_DIR}/cmake/debuggable.entitlements
                $<TARGET_FILE:f5_spike>
        COMMENT "Signing f5_spike with get-task-allow (leaks)"
        VERBATIM
    )
endif()

# --- F6 spikes S3-S4 (measurement tool, not engine code) ---
# The bench/soc harness and runner with the F6 spike benchmarks (ids F6-Sn);
# their MSL lives in bench/f6_spike/shaders, engine shaders are compiled from
# shaders/ with their includes.  See bench/f6_spike/README.md.
file(GLOB F6_SPIKE_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/bench/f6_spike/*.cpp)
add_executable(f6_spike ${CMAKE_SOURCE_DIR}/bench/soc/harness.cpp ${CMAKE_SOURCE_DIR}/bench/soc/soc_bench.cpp
    ${F6_SPIKE_SOURCES})
target_include_directories(f6_spike PRIVATE ${CMAKE_SOURCE_DIR}/bench/soc)
target_link_libraries(f6_spike PRIVATE phosphor_core metal_cpp IOReport
    "-framework IOKit" "-framework CoreFoundation" "-framework AppKit" "-framework QuartzCore")
target_compile_features(f6_spike PRIVATE cxx_std_20)
target_compile_options(f6_spike PRIVATE ${PHOSPHOR_WARNINGS})
add_dependencies(f6_spike phosphor_shaders) # generated variant headers for the engine shaders
target_compile_definitions(f6_spike PRIVATE
    "SOC_SHADER_DIR=\"${CMAKE_SOURCE_DIR}/bench/soc/shaders\""
    "SOC_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\""
    "F6_GENERATED_DIR=\"${CMAKE_BINARY_DIR}/generated\""
    "SOC_RESULTS_DIR=\"${CMAKE_SOURCE_DIR}/bench/results/f6-spike\""
    "SOC_SDK_VERSION=\"${SOC_SDK_VERSION}\"")
if(PHOSPHOR_DEBUGGABLE)
    add_custom_command(TARGET f6_spike POST_BUILD
        COMMAND codesign --force --sign - --entitlements ${CMAKE_SOURCE_DIR}/cmake/debuggable.entitlements
                $<TARGET_FILE:f6_spike>
        COMMENT "Signing f6_spike with get-task-allow (leaks)")
endif()

# Test assets are looked up relative to the working directory.
if(NOT EXISTS ${CMAKE_BINARY_DIR}/assets)
    file(CREATE_LINK ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets SYMBOLIC)
endif()
