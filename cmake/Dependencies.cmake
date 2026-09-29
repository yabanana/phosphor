# Dependencies.cmake -- third-party libraries, fetched and pinned.

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# --- glm ---
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE
)
set(GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLM_BUILD_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glm)

# --- SDL3 (window, input, timing) ---
find_package(SDL3 CONFIG QUIET)
if(NOT SDL3_FOUND)
    FetchContent_Declare(SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG        release-3.4.16
        GIT_SHALLOW    TRUE
    )
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    if(NOT APPLE)
        # Linux builds (CI, cloud sessions) only need SDL for timing and the
        # input types; skip every subsystem that pulls in system packages.
        set(SDL_UNIX_CONSOLE_BUILD ON CACHE BOOL "" FORCE)
        foreach(sub VIDEO AUDIO RENDER GPU CAMERA JOYSTICK HAPTIC SENSOR HIDAPI DIALOG TRAY)
            set(SDL_${sub} OFF CACHE BOOL "" FORCE)
        endforeach()
    endif()
    FetchContent_MakeAvailable(SDL3)
endif()

# --- meshoptimizer ---
FetchContent_Declare(meshoptimizer
    GIT_REPOSITORY https://github.com/zeux/meshoptimizer.git
    GIT_TAG        v1.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(meshoptimizer)

# --- tinygltf (header-only; also provides stb_image) ---
FetchContent_Declare(tinygltf
    GIT_REPOSITORY https://github.com/syoyo/tinygltf.git
    GIT_TAG        v2.9.7
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  do-not-build   # only the headers are used
)
FetchContent_MakeAvailable(tinygltf)
add_library(tinygltf INTERFACE)
target_include_directories(tinygltf SYSTEM INTERFACE ${tinygltf_SOURCE_DIR})

# --- MikkTSpace (reference tangent generator required by glTF, zlib) ---
# Pinned by commit: the repository has no releases.
FetchContent_Declare(mikktspace
    GIT_REPOSITORY https://github.com/mmikk/MikkTSpace.git
    GIT_TAG        3e895b49d05ea07e4c2133156cfa94369e19e409
    SOURCE_SUBDIR  do-not-build   # built below as a plain C library
)
FetchContent_MakeAvailable(mikktspace)
add_library(mikktspace STATIC ${mikktspace_SOURCE_DIR}/mikktspace.c)
target_include_directories(mikktspace SYSTEM PUBLIC ${mikktspace_SOURCE_DIR})
set_target_properties(mikktspace PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(mikktspace PRIVATE -w)

# --- Apple-side sources: metal-cpp and Dear ImGui ---
# Needed by the macOS app; also fetched on Linux so the Metal host code can be
# type-checked there (see the metal_syntax_check target and
# tools/apple-sdk-stubs).
option(PHOSPHOR_METAL_SYNTAX_CHECK "Add the metal_syntax_check target (non-Apple hosts)" ON)
if(PHOSPHOR_BUILD_APP OR (PHOSPHOR_METAL_SYNTAX_CHECK AND NOT APPLE))
    FetchContent_Declare(metal_cpp
        GIT_REPOSITORY https://github.com/apple/metal-cpp.git
        GIT_TAG        release/metal-cpp_macOS26.4_iOS26.4
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(metal_cpp)

    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.91.8
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(imgui)
endif()

# --- Tracy profiler client (F4.2, optional) ---
# Off by default: no fetch, no define, every PH_* macro (core/profile.h)
# expands to nothing.  ONLY_LOCALHOST keeps the listener off the network;
# broadcast discovery and frame-image capture are not used.  The capture GUI /
# tools are built from this same source by tools/tracy_check.sh.
option(PHOSPHOR_TRACY "Link the Tracy profiler client (CPU/GPU zones, allocations)" OFF)
if(PHOSPHOR_TRACY)
    FetchContent_Declare(tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG        v0.14.1
        GIT_SHALLOW    TRUE
    )
    set(TRACY_ENABLE ON CACHE BOOL "" FORCE)
    set(TRACY_ONLY_LOCALHOST ON CACHE BOOL "" FORCE)
    set(TRACY_NO_BROADCAST ON CACHE BOOL "" FORCE)
    set(TRACY_NO_FRAME_IMAGE ON CACHE BOOL "" FORCE)
    set(TRACY_NO_SYSTEM_TRACING ON CACHE BOOL "" FORCE)
    set(TRACY_STATIC ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(tracy)
endif()
