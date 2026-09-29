# Variants.cmake -- F3.3 variant table generator.
#
# tools/variant_gen turns shaders/variants.def into two headers in
# ${CMAKE_BINARY_DIR}/generated/pipeline/: the C++ axis table used by
# src/pipeline/forward_variants.cpp and the MSL function constants included by
# shaders/forward.metal.  Consumers depend on the `phosphor_variants` target
# (phosphor_core, the shader compilation and the Linux syntax check).

add_executable(variant_gen tools/variant_gen/variant_gen.cpp)
target_compile_features(variant_gen PRIVATE cxx_std_20)

set(PHOSPHOR_GENERATED_DIR ${CMAKE_BINARY_DIR}/generated)
set(PHOSPHOR_VARIANTS_DEF ${CMAKE_SOURCE_DIR}/shaders/variants.def)
set(PHOSPHOR_VARIANTS_CPP_HEADER ${PHOSPHOR_GENERATED_DIR}/pipeline/forward_variants.generated.h)
set(PHOSPHOR_VARIANTS_MSL_HEADER ${PHOSPHOR_GENERATED_DIR}/pipeline/forward_variants.generated.metal.h)

add_custom_command(
    OUTPUT ${PHOSPHOR_VARIANTS_CPP_HEADER} ${PHOSPHOR_VARIANTS_MSL_HEADER}
    COMMAND variant_gen ${PHOSPHOR_VARIANTS_DEF} ${PHOSPHOR_GENERATED_DIR}
    DEPENDS variant_gen ${PHOSPHOR_VARIANTS_DEF}
    COMMENT "Generating forward variant tables from variants.def"
    VERBATIM
)
add_custom_target(phosphor_variants DEPENDS ${PHOSPHOR_VARIANTS_CPP_HEADER} ${PHOSPHOR_VARIANTS_MSL_HEADER})
