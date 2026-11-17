# Variants.cmake -- F3.3 variant table generator.
#
# tools/variant_gen turns shaders/variants.def into two headers in
# ${CMAKE_BINARY_DIR}/generated/pipeline/: the C++ axis table used by
# src/pipeline/forward_variants.cpp and the MSL function constants included by
# shaders/forward.metal.  Consumers depend on the `phosphor_variants` target
# (phosphor_core, the shader compilation and the Linux syntax check).

add_executable(variant_gen tools/variant_gen/variant_gen.cpp tools/variant_gen/class_gen.cpp)
target_compile_features(variant_gen PRIVATE cxx_std_20)
# F15.2: the class generator shares the manifest parser (header only, renderer/material_class_manifest.h) with the exporter.
target_include_directories(variant_gen PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(variant_gen PRIVATE tinygltf) # nlohmann json.hpp

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

# F15.2 (ticket 21): the material permutation generator -- class table, MSL constants and the archive plan from
# shaders/material_classes.def and the feature-class manifests it names (shaders/material_classes/*.json).
set(PHOSPHOR_CLASSES_DEF ${CMAKE_SOURCE_DIR}/shaders/material_classes.def)
file(GLOB PHOSPHOR_CLASS_MANIFESTS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/shaders/material_classes/*.json)
set(PHOSPHOR_CLASSES_CPP_HEADER ${PHOSPHOR_GENERATED_DIR}/pipeline/material_classes.generated.h)
set(PHOSPHOR_CLASSES_MSL_HEADER ${PHOSPHOR_GENERATED_DIR}/pipeline/material_classes.generated.metal.h)
set(PHOSPHOR_CLASSES_PLAN ${PHOSPHOR_GENERATED_DIR}/pipeline/material_class_plan.json)
add_custom_command(
    OUTPUT ${PHOSPHOR_CLASSES_CPP_HEADER} ${PHOSPHOR_CLASSES_MSL_HEADER} ${PHOSPHOR_CLASSES_PLAN}
    COMMAND variant_gen classes ${PHOSPHOR_VARIANTS_DEF} ${PHOSPHOR_CLASSES_DEF} ${PHOSPHOR_GENERATED_DIR}
    DEPENDS variant_gen ${PHOSPHOR_VARIANTS_DEF} ${PHOSPHOR_CLASSES_DEF} ${PHOSPHOR_CLASS_MANIFESTS}
    COMMENT "Generating the material class table and archive plan from material_classes.def"
    VERBATIM
)
add_custom_target(phosphor_variants DEPENDS ${PHOSPHOR_VARIANTS_CPP_HEADER} ${PHOSPHOR_VARIANTS_MSL_HEADER}
    ${PHOSPHOR_CLASSES_CPP_HEADER} ${PHOSPHOR_CLASSES_MSL_HEADER} ${PHOSPHOR_CLASSES_PLAN})
