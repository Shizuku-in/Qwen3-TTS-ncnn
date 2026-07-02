cmake_minimum_required(VERSION 3.16)
project(qwen3_tts_ncnn CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- locate ncnn ----------------------------------------------------------
# Point -DncnnDIR=/path/to/ncnn/build/install/lib/cmake/ncnn at an installed
# ncnn, or -Dncnn_DIR / -DNCNN_BUILD_DIR at an in-tree build dir. We first try
# find_package; if that fails we fall back to a manual static-lib + headers.
set(NCNN_BUILD_DIR "" CACHE PATH "Path to an in-tree ncnn build directory")

find_package(ncnn QUIET)

if(NOT ncnn_FOUND AND NCNN_BUILD_DIR)
    message(STATUS "ncnn package not found; using NCNN_BUILD_DIR=${NCNN_BUILD_DIR}")
    add_library(ncnn STATIC IMPORTED)
    set_target_properties(ncnn PROPERTIES
        IMPORTED_LOCATION "${NCNN_BUILD_DIR}/src/libncnn.a")
    target_include_directories(ncnn INTERFACE
        "${NCNN_BUILD_DIR}/../src"   # source headers (net.h, mat.h, layer.h)
        "${NCNN_BUILD_DIR}/src")     # generated headers (platform.h, ...)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(ncnn INTERFACE OpenMP::OpenMP_CXX)
    endif()
endif()

if(NOT TARGET ncnn)
    message(FATAL_ERROR
        "ncnn not found. Either install ncnn and set -Dncnn_DIR, or pass "
        "-DNCNN_BUILD_DIR=/path/to/ncnn/build")
endif()

# ---- library: shared inference sources ------------------------------------
add_library(qwen3_tts_core INTERFACE)
target_include_directories(qwen3_tts_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(qwen3_tts_core INTERFACE ncnn)

# ---- main synthesizer (built once src/qwen3_tts.cpp exists) ----------------
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/qwen3_tts.cpp)
    add_executable(qwen3_tts src/qwen3_tts.cpp)
    target_link_libraries(qwen3_tts PRIVATE qwen3_tts_core)
endif()

# ---- tests / validation harnesses -----------------------------------------
enable_testing()

# conv-tail parity: exercises the ncnn link + the custom SnakeBeta layer
add_executable(test_convtail tests/test_convtail.cpp)
target_link_libraries(test_convtail PRIVATE qwen3_tts_core)
add_test(NAME convtail COMMAND test_convtail)

# full vocoder chain: codes -> wav, validated vs golden
add_executable(test_vocoder tests/test_vocoder.cpp)
target_link_libraries(test_vocoder PRIVATE qwen3_tts_core)
add_test(NAME vocoder COMMAND test_vocoder)

# AR decode loop: Talker outer + Code Predictor nested, validated vs golden codes
add_executable(test_ar_loop tests/test_ar_loop.cpp)
target_link_libraries(test_ar_loop PRIVATE qwen3_tts_core)
add_test(NAME ar_loop COMMAND test_ar_loop)
