FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG        v2.5.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(doctest)

add_executable(phosphor_tests
    test_main.cpp
    test_aliasing.cpp
    test_barrier_plan.cpp
    test_gltf_loader.cpp
    test_async_probe_reference.cpp
    test_graph_debug_reference.cpp
    test_graph_dump.cpp
    test_gpu_scene.cpp
    test_launch_options.cpp
    test_linear_ring.cpp
    test_memory_budget.cpp
    test_mesh_processing.cpp
    test_procedural.cpp
    test_render_graph.cpp
    test_scene_extract.cpp
    test_tbdr_passes.cpp
    test_texture_manager.cpp
    test_testbenches.cpp
    test_tlsf.cpp
    test_worker_pool.cpp
)
target_link_libraries(phosphor_tests PRIVATE phosphor_core doctest::doctest)
target_compile_options(phosphor_tests PRIVATE ${PHOSPHOR_WARNINGS})
add_test(NAME phosphor_tests COMMAND phosphor_tests)
