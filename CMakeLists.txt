cmake_minimum_required(VERSION 3.20)
project(browseg CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TSC_SOURCES
    src/tsc/threadpool.cpp
    src/tsc/resample.cpp
)
file(GLOB TSC_EXTRA_SOURCES CONFIGURE_DEPENDS
    src/tsc/dicom.cpp src/tsc/unet.cpp src/tsc/nnunet.cpp src/tsc/pipeline.cpp src/tsc/mesh.cpp src/tsc/weights.cpp)
list(APPEND TSC_SOURCES ${TSC_EXTRA_SOURCES})

add_library(tsc_core STATIC ${TSC_SOURCES})
target_include_directories(tsc_core PUBLIC src)

option(TSC_SINGLE_THREAD "WebAssembly build without threads (web/dist-st; runs on pages that are not cross-origin isolated)" OFF)

if(EMSCRIPTEN)
    # wasm: SIMD128 + pthreads, no FMA contraction (keeps scipy-exact resampling)
    # -fexceptions: C++ exceptions are caught (default emscripten builds abort on the first throw)
    if(TSC_SINGLE_THREAD)
        target_compile_options(tsc_core PUBLIC -O3 -msimd128 -ffp-contract=off -fexceptions -DTSC_SINGLE_THREAD)
    else()
        target_compile_options(tsc_core PUBLIC -O3 -msimd128 -pthread -ffp-contract=off -fexceptions)
    endif()
elseif(MSVC)
    target_compile_options(tsc_core PUBLIC /O2 /arch:AVX2 /fp:precise /EHsc /utf-8 /W3 /wd4244 /wd4267)
    target_compile_definitions(tsc_core PUBLIC _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(tsc_core PUBLIC -O3 -mavx2 -mfma -ffp-contract=off)
    find_package(Threads REQUIRED)
    target_link_libraries(tsc_core PUBLIC Threads::Threads)
endif()

if(NOT EMSCRIPTEN)
    add_executable(test_resample tests/test_resample.cpp)
    target_link_libraries(test_resample tsc_core)
    foreach(t test_unet test_task mesh_from_npy dump_dicom_volume)
        if(EXISTS ${CMAKE_SOURCE_DIR}/tests/${t}.cpp)
            add_executable(${t} tests/${t}.cpp)
            target_link_libraries(${t} tsc_core)
        endif()
    endforeach()
    if(EXISTS ${CMAKE_SOURCE_DIR}/cli/main.cpp)
        add_executable(tsc_liver cli/main.cpp)
        target_link_libraries(tsc_liver tsc_core)
    endif()
endif()

if(EMSCRIPTEN AND EXISTS ${CMAKE_SOURCE_DIR}/web/wasm_api.cpp)
    add_executable(tsc_wasm web/wasm_api.cpp web/liver_bundle.cpp)
    target_include_directories(tsc_wasm PRIVATE web)
    target_link_libraries(tsc_wasm tsc_core)
    if(TSC_SINGLE_THREAD)
        set(TSC_DIST dist-st)
        set(TSC_THREAD_FLAGS "")
    else()
        set(TSC_DIST dist)
        set(TSC_THREAD_FLAGS -pthread -sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency)
    endif()
    set_target_properties(tsc_wasm PROPERTIES OUTPUT_NAME "tsc" SUFFIX ".js"
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/web/${TSC_DIST})
    target_link_options(tsc_wasm PRIVATE
        -O3 -msimd128 ${TSC_THREAD_FLAGS} -fexceptions
        -sMODULARIZE=1 -sEXPORT_NAME=createTSC -sENVIRONMENT=web,worker,node
        -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=256MB -sMAXIMUM_MEMORY=4GB
        -sSTACK_SIZE=4MB
        -sEXPORTED_FUNCTIONS=_malloc,_free,_tsc_version,_tsc_set_threads,_tsc_set_resampling_order,_tsc_load_weights,_tsc_has_model,_tsc_load_classmaps,_tsc_task_models,_tsc_clear_dicom,_tsc_add_dicom,_tsc_load_dicom_volume,_tsc_list_series,_tsc_load_dicom_series,_tsc_set_volume,_tsc_set_volume_raw,_tsc_ct_ptr,_tsc_ct_affine,_tsc_ct_info,_tsc_run,_tsc_begin,_tsc_stage,_tsc_stage_name,_tsc_num_tiles,_tsc_tile_input,_tsc_tile_output,_tsc_tile_accumulate,_tsc_tile_count,_tsc_logits_ptr,_tsc_gaussian_ptr,_tsc_tile_geom,_tsc_advance,_tsc_finish,_tsc_labels_ptr,_tsc_num_label_names,_tsc_label_name,_tsc_set_labels,_tsc_build_meshes,_tsc_result_ptr,_tsc_result_int,_tsc_label_info,_tsc_result_double,_tsc_free_results,_tsc_bundle_clear,_tsc_bundle_keep,_tsc_bundle_build,_tsc_bundle_blob_ptr,_tsc_bundle_blob_len
        -sEXPORTED_RUNTIME_METHODS=wasmMemory,HEAPU8,HEAPF32,HEAP32,HEAPU32,HEAPF64,UTF8ToString,stringToUTF8,lengthBytesUTF8
    )
endif()
