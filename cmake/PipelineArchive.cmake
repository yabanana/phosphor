# PipelineArchive.cmake -- F3.4: build the MTL4Archive with metal-tt.
#
# Included from App.cmake after PHOSPHOR_METALLIB / PHOSPHOR_SHADER_OUT are set.
# shaders/pipelines.mtl4-json (a "pipelines script" harvested by
# tools/harvest_pipelines.sh, library path = @PHOSPHOR_METALLIB@) is
# specialised with the real metallib path and translated into
# ${PHOSPHOR_SHADER_OUT}/phosphor-archive.metallib, which the app looks for
# next to phosphor.metallib.  metal-tt recomputes the keys from the real
# metallib, so the JSON stays valid when shader bodies change.
#
#   PHOSPHOR_ARCHIVE_ARCHS = native (default: the local GPU, `xcrun metal-arch`)
#                            | all   (every GPU arch; ~16 s instead of ~0.1 s.
#                                     An archive without the local slice is
#                                     rejected at open time.)

set(PHOSPHOR_ARCHIVE_ARCHS "native" CACHE STRING "GPU architectures of the pipeline archive: native | all")
set_property(CACHE PHOSPHOR_ARCHIVE_ARCHS PROPERTY STRINGS native all)

execute_process(
    COMMAND xcrun -sdk macosx -f metal-tt
    OUTPUT_VARIABLE PHOSPHOR_METAL_TT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _tt_rc
    ERROR_QUIET
)
set(PHOSPHOR_PIPELINES_JSON ${CMAKE_SOURCE_DIR}/shaders/pipelines.mtl4-json)
set(PHOSPHOR_ARCHIVE ${PHOSPHOR_SHADER_OUT}/phosphor-archive.metallib)
set(PHOSPHOR_ARCHIVE_JSON ${PHOSPHOR_SHADER_OUT}/pipelines.mtl4-json)

if(NOT _tt_rc EQUAL 0 OR NOT PHOSPHOR_METAL_TT)
    message(WARNING "metal-tt not found (xcrun -sdk macosx -f metal-tt): the pipeline "
                    "archive is not built; the app will compile every pipeline.")
elseif(NOT EXISTS ${PHOSPHOR_PIPELINES_JSON})
    message(WARNING "${PHOSPHOR_PIPELINES_JSON} missing: pipeline archive not built "
                    "(run tools/harvest_pipelines.sh).")
else()
    set(PHOSPHOR_ARCHIVE_ARCH_ARGS)
    if(PHOSPHOR_ARCHIVE_ARCHS STREQUAL "native")
        execute_process(
            COMMAND xcrun metal-arch
            OUTPUT_VARIABLE _native_arch
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _arch_rc
            ERROR_QUIET
        )
        if(_arch_rc EQUAL 0 AND _native_arch MATCHES "^applegpu_")
            set(PHOSPHOR_ARCHIVE_ARCH_ARGS -arch ${_native_arch})
            message(STATUS "Pipeline archive: native arch ${_native_arch}")
        else()
            message(STATUS "Pipeline archive: metal-arch failed, building for all architectures")
        endif()
    elseif(NOT PHOSPHOR_ARCHIVE_ARCHS STREQUAL "all")
        message(FATAL_ERROR "PHOSPHOR_ARCHIVE_ARCHS must be 'native' or 'all'")
    else()
        message(STATUS "Pipeline archive: all architectures")
    endif()

    # Placeholder substitution as a -P script (portable, no sed flavours).
    set(_subst ${CMAKE_BINARY_DIR}/substitute_metallib.cmake)
    file(WRITE ${_subst} [=[
file(READ "${IN}" _content)
string(REPLACE "@PHOSPHOR_METALLIB@" "${METALLIB}" _content "${_content}")
file(WRITE "${OUT}" "${_content}")
]=])

    add_custom_command(
        OUTPUT ${PHOSPHOR_ARCHIVE}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${PHOSPHOR_SHADER_OUT}
        COMMAND ${CMAKE_COMMAND} -DIN=${PHOSPHOR_PIPELINES_JSON} -DOUT=${PHOSPHOR_ARCHIVE_JSON}
                -DMETALLIB=${PHOSPHOR_METALLIB} -P ${_subst}
        COMMAND xcrun -sdk macosx metal-tt ${PHOSPHOR_ARCHIVE_JSON}
                ${PHOSPHOR_ARCHIVE_ARCH_ARGS} -o ${PHOSPHOR_ARCHIVE}
        DEPENDS ${PHOSPHOR_METALLIB} ${PHOSPHOR_PIPELINES_JSON} ${_subst}
        COMMENT "Translating pipelines.mtl4-json into phosphor-archive.metallib (metal-tt)"
        VERBATIM
    )
    add_custom_target(phosphor_archive ALL DEPENDS ${PHOSPHOR_ARCHIVE})
    # `phosphor` is defined after this include: attach the dependency at the
    # end of the directory scope.
    cmake_language(DEFER DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        CALL add_dependencies phosphor phosphor_archive)
endif()
