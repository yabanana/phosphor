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
    -I ${CMAKE_SOURCE_DIR}/src -Wall)
if(CMAKE_BUILD_TYPE MATCHES "Debug|RelWithDebInfo")
    # Source-level shader debugging and profiling in Xcode.
    list(APPEND PHOSPHOR_METAL_FLAGS -gline-tables-only -frecord-sources)
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

# --- Executable ---
add_executable(phosphor
    src/main.cpp
    src/app/engine.cpp
    src/imgui/imgui_renderer.cpp
    src/imgui/ui_panels.cpp
    src/platform/metal/frame_capture.cpp
    src/platform/metal/gpu_memory.cpp
    src/platform/metal/metal_context.cpp
    src/platform/metal/metal_impl.cpp
    src/platform/metal/memory_pressure.cpp
    src/platform/metal/metal_texture_manager.cpp
    src/platform/metal/residency_manager.cpp
    src/platform/metal/scene_renderer.cpp
    src/platform/metal/upload_ring.cpp
)
target_link_libraries(phosphor PRIVATE phosphor_core imgui metal_cpp)
target_compile_options(phosphor PRIVATE ${PHOSPHOR_WARNINGS})
add_dependencies(phosphor phosphor_shaders)
if(PHOSPHOR_METAL_VALIDATION)
    # main() enables the Metal API validation layer in Debug builds.
    target_compile_definitions(phosphor PRIVATE $<$<CONFIG:Debug>:PHOSPHOR_METAL_VALIDATION=1>)
endif()

# Test assets are looked up relative to the working directory.
if(NOT EXISTS ${CMAKE_BINARY_DIR}/assets)
    file(CREATE_LINK ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets SYMBOLIC)
endif()
