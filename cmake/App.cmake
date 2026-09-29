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
        DEPENDS ${shader} ${CMAKE_SOURCE_DIR}/src/renderer/gpu_types.h
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
add_custom_target(phosphor_shaders DEPENDS ${PHOSPHOR_METALLIB})

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
    src/platform/metal/gpu_timestamps.cpp
    src/platform/metal/graph_debug_passes.cpp
    src/platform/metal/known_cost_pass.cpp
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

# Test assets are looked up relative to the working directory.
if(NOT EXISTS ${CMAKE_BINARY_DIR}/assets)
    file(CREATE_LINK ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets SYMBOLIC)
endif()
