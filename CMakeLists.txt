cmake_minimum_required(VERSION 3.25)
project(phosphor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- Options ---
option(PHOSPHOR_ENABLE_TRACY      "Enable Tracy profiler integration" OFF)
option(PHOSPHOR_ENABLE_AFTERMATH  "Enable NVIDIA Aftermath crash diagnostics" OFF)
option(PHOSPHOR_ENABLE_VALIDATION "Enable Vulkan validation layers" ON)

# --- Find system packages ---
find_package(Vulkan REQUIRED)
find_package(SDL3 REQUIRED)
find_package(glm REQUIRED)

# --- Fetch dependencies ---
include(cmake/FetchDependencies.cmake)

# --- Shader compilation ---
include(cmake/CompileShaders.cmake)

# --- Collect sources ---
file(GLOB_RECURSE PHOSPHOR_SOURCES CONFIGURE_DEPENDS
    src/*.cpp
)

# --- Executable ---
add_executable(phosphor ${PHOSPHOR_SOURCES})

target_include_directories(phosphor PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
)

target_link_libraries(phosphor PRIVATE
    Vulkan::Vulkan
    SDL3::SDL3
    glm::glm
    VulkanMemoryAllocator
    meshoptimizer
    tinygltf
    imgui
    ${CMAKE_DL_LIBS}  # dlopen for RenderDoc
    pthread
)

# --- Compile definitions ---
target_compile_definitions(phosphor PRIVATE
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_RADIANS
    GLM_ENABLE_EXPERIMENTAL
)

if(PHOSPHOR_ENABLE_VALIDATION)
    target_compile_definitions(phosphor PRIVATE PHOSPHOR_ENABLE_VALIDATION=1)
endif()

if(PHOSPHOR_ENABLE_TRACY)
    target_compile_definitions(phosphor PRIVATE TRACY_ENABLE=1)
    target_link_libraries(phosphor PRIVATE TracyClient)
endif()

if(PHOSPHOR_ENABLE_AFTERMATH)
    include(cmake/FindAftermathSDK.cmake)
    if(AFTERMATH_FOUND)
        target_compile_definitions(phosphor PRIVATE PHOSPHOR_ENABLE_AFTERMATH=1)
        target_include_directories(phosphor PRIVATE ${AFTERMATH_INCLUDE_DIR})
        target_link_libraries(phosphor PRIVATE ${AFTERMATH_LIBRARY})
    else()
        message(WARNING "Aftermath SDK not found in third_party/aftermath/")
    endif()
endif()

# --- Warnings ---
target_compile_options(phosphor PRIVATE
    -Wall -Wextra -Wpedantic -Wno-unused-parameter
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG>
)

# --- Shader compilation ---
phosphor_compile_shaders(
    TARGET phosphor
    SHADERS_DIR ${CMAKE_SOURCE_DIR}/shaders
    OUTPUT_DIR ${CMAKE_BINARY_DIR}/shaders
)

# --- Copy assets reference ---
file(CREATE_LINK ${CMAKE_SOURCE_DIR}/assets ${CMAKE_BINARY_DIR}/assets SYMBOLIC)
