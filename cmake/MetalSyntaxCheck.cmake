# MetalSyntaxCheck.cmake -- `cmake --build <dir> --target metal_syntax_check`

find_program(PHOSPHOR_CLANGXX NAMES clang++ clang++-18 clang++-17)
if(NOT PHOSPHOR_CLANGXX)
    message(STATUS "metal_syntax_check: clang++ not found, target disabled")
    return()
endif()

file(GLOB PHOSPHOR_METAL_HOST_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/platform/metal/*.cpp
    ${CMAKE_SOURCE_DIR}/src/app/*.cpp
    ${CMAKE_SOURCE_DIR}/src/imgui/*.cpp
    ${CMAKE_SOURCE_DIR}/src/main.cpp
)

set(_flags
    -std=c++20 -fsyntax-only -fblocks -Wall -Wextra -Wno-unused-parameter
    -DGLM_FORCE_DEPTH_ZERO_TO_ONE -DGLM_FORCE_RADIANS -DGLM_ENABLE_EXPERIMENTAL
    -DPHOSPHOR_METAL_SYNTAX_CHECK
    -isystem ${CMAKE_SOURCE_DIR}/tools/apple-sdk-stubs
    -isystem ${metal_cpp_SOURCE_DIR}
    -isystem ${glm_SOURCE_DIR}
    -isystem ${imgui_SOURCE_DIR}
    -isystem ${imgui_SOURCE_DIR}/backends
    -isystem ${tinygltf_SOURCE_DIR}
    -I ${CMAKE_SOURCE_DIR}/src
)
if(DEFINED sdl3_SOURCE_DIR)
    list(APPEND _flags -isystem ${sdl3_SOURCE_DIR}/include)
endif()

set(_commands)
foreach(src IN LISTS PHOSPHOR_METAL_HOST_SOURCES)
    list(APPEND _commands COMMAND ${PHOSPHOR_CLANGXX} ${_flags} ${src})
endforeach()

add_custom_target(metal_syntax_check
    ${_commands}
    COMMENT "Type-checking Metal host code against metal-cpp (syntax only)"
    VERBATIM
)
