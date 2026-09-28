# CompileShaders.cmake — compile GLSL sources to SPIR-V via glslc
#
# Usage: phosphor_compile_shaders(TARGET <target> SHADERS_DIR <dir> OUTPUT_DIR <dir>)
# Files in common/ are includes, not compiled directly.
# Shader stage is inferred from filename: *.vert.glsl, *.frag.glsl, *.comp.glsl,
# *.task.glsl, *.mesh.glsl, *.rgen.glsl, *.rmiss.glsl, *.rchit.glsl

function(phosphor_compile_shaders)
    cmake_parse_arguments(SHADER "" "TARGET;SHADERS_DIR;OUTPUT_DIR" "" ${ARGN})

    file(GLOB_RECURSE SHADER_SOURCES CONFIGURE_DEPENDS
        "${SHADER_SHADERS_DIR}/*.glsl"
    )

    # Filter out common/ includes
    list(FILTER SHADER_SOURCES EXCLUDE REGEX ".*/common/.*")

    set(SPV_FILES "")

    foreach(SHADER_SRC ${SHADER_SOURCES})
        # Get relative path for output structure
        file(RELATIVE_PATH REL_PATH "${SHADER_SHADERS_DIR}" "${SHADER_SRC}")
        string(REGEX REPLACE "\\.glsl$" ".spv" SPV_REL "${REL_PATH}")
        set(SPV_OUTPUT "${SHADER_OUTPUT_DIR}/${SPV_REL}")

        # Get parent dir for output
        get_filename_component(SPV_DIR "${SPV_OUTPUT}" DIRECTORY)

        # Infer shader stage from filename
        set(STAGE "")
        if(SHADER_SRC MATCHES "\\.vert\\.glsl$")
            set(STAGE "vertex")
        elseif(SHADER_SRC MATCHES "\\.frag\\.glsl$")
            set(STAGE "fragment")
        elseif(SHADER_SRC MATCHES "\\.comp\\.glsl$")
            set(STAGE "compute")
        elseif(SHADER_SRC MATCHES "\\.task\\.glsl$")
            set(STAGE "task")
        elseif(SHADER_SRC MATCHES "\\.mesh\\.glsl$")
            set(STAGE "mesh")
        elseif(SHADER_SRC MATCHES "\\.rgen\\.glsl$")
            set(STAGE "rgen")
        elseif(SHADER_SRC MATCHES "\\.rmiss\\.glsl$")
            set(STAGE "rmiss")
        elseif(SHADER_SRC MATCHES "\\.rchit\\.glsl$")
            set(STAGE "rchit")
        else()
            message(WARNING "Cannot infer shader stage for: ${SHADER_SRC}, skipping")
            continue()
        endif()

        # Collect common/ headers as dependencies
        file(GLOB COMMON_HEADERS CONFIGURE_DEPENDS
            "${SHADER_SHADERS_DIR}/common/*.glsl"
        )

        add_custom_command(
            OUTPUT "${SPV_OUTPUT}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${SPV_DIR}"
            COMMAND Vulkan::glslc
                --target-env=vulkan1.3
                -I "${SHADER_SHADERS_DIR}/common"
                -fshader-stage=${STAGE}
                -O
                -Werror
                -o "${SPV_OUTPUT}"
                "${SHADER_SRC}"
            DEPENDS "${SHADER_SRC}" ${COMMON_HEADERS}
            COMMENT "Compiling ${REL_PATH} -> ${SPV_REL}"
            VERBATIM
        )

        list(APPEND SPV_FILES "${SPV_OUTPUT}")
    endforeach()

    add_custom_target(${SHADER_TARGET}_shaders DEPENDS ${SPV_FILES})
    add_dependencies(${SHADER_TARGET} ${SHADER_TARGET}_shaders)
endfunction()
