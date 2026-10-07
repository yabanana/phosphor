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
    test_bench_report.cpp
    test_compile_queue.cpp
    test_cull_math.cpp
    test_ecs_changes.cpp
    test_file_watcher.cpp
    test_forward_variants.cpp
    test_frame_trace.cpp
    test_gltf_loader.cpp
    test_async_probe_reference.cpp
    test_graph_debug_reference.cpp
    test_graph_dump.cpp
    test_graph_lint.cpp
    test_graph_opt_tool.cpp
    test_graph_plan.cpp
    test_gpu_queue.cpp
    test_gpu_scene.cpp
    test_launch_options.cpp
    test_linear_ring.cpp
    test_memory_budget.cpp
    test_meshlet_builder.cpp
    test_meshlet_cull.cpp
    test_meshlet_check.cpp
    test_mesh_processing.cpp
    test_overlay_math.cpp
    test_pass_timings.cpp
    test_pipeline_key.cpp
    test_pipeline_registry.cpp
    test_pipeline_script.cpp
    test_procedural.cpp
    test_render_graph.cpp
    test_rt_proxy.cpp
    test_rt_scene.cpp
    test_rt_reference.cpp
    test_rt_check.cpp
    test_shadows.cpp
    test_scene_check.cpp
    test_scene_extract.cpp
    test_scene_store.cpp
    test_soc_model.cpp
    test_soc_results.cpp
    test_tbdr_passes.cpp
    test_timing_plan.cpp
    test_texture_manager.cpp
    test_testbenches.cpp
    test_tlsf.cpp
    test_transform_math.cpp
    test_temporal.cpp
    test_visibility_math.cpp
    test_worker_pool.cpp
)
target_link_libraries(phosphor_tests PRIVATE phosphor_core doctest::doctest)
# test_pipeline_script.cpp reads the committed shaders/pipelines.mtl4-json.
target_compile_definitions(phosphor_tests PRIVATE PHOSPHOR_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(phosphor_tests PRIVATE ${CMAKE_SOURCE_DIR}/tools)
target_compile_options(phosphor_tests PRIVATE ${PHOSPHOR_WARNINGS})
add_test(NAME phosphor_tests COMMAND phosphor_tests)
