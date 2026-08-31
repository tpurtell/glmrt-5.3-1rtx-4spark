cmake_minimum_required(VERSION 3.24)
project(glmrt_native LANGUAGES C CXX)

option(GLMRT_ENABLE_CUDA "Build CUDA kernels and CUDA-backed ABI functions" OFF)
option(GLMRT_ENABLE_RDMA "Build RDMA helper components when libibverbs is available" OFF)
option(GLMRT_ENABLE_SPARKINFER_AOT "Build Spark-side SparkInfer AOT kernels" OFF)
option(GLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT
  "Build coordinator SparkInfer AOT kernels" OFF)
option(GLMRT_ENABLE_B12X_AOT
  "Deprecated alias for GLMRT_ENABLE_SPARKINFER_AOT" OFF)
option(GLMRT_ENABLE_B12X_COORDINATOR_AOT
  "Deprecated alias for GLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT" OFF)
option(GLMRT_ENABLE_W8A16_AOT "Build coordinator row-major W8A16 Triton AOT kernels" OFF)
option(GLMRT_ENABLE_FLASHINFER_PACKED_MLA
  "Build the exact grouped-split SM120 packed-FP8 MLA kernel" OFF)
option(GLMRT_ENABLE_NCCL "Build multi-node NCCL helper components" OFF)
option(GLMRT_ENABLE_XGRAMMAR "Build constrained-decoding grammar support" ON)
set(GLMRT_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA architectures for native kernels")
set(GLMRT_SPARKINFER_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sparkinfer"
  CACHE PATH "Pinned SparkInfer source tree used by every SparkInfer-backed AOT export")
set(GLMRT_SPARKINFER_LOCK_FILE
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/sparkinfer.lock.json"
  CACHE FILEPATH "Provenance lock for GLMRT_SPARKINFER_SOURCE_DIR")
set(GLMRT_XGRAMMAR_SOURCE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xgrammar"
  CACHE PATH "Pinned XGrammar source tree")
set(GLMRT_XGRAMMAR_LOCK_FILE
  "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xgrammar.lock.json"
  CACHE FILEPATH "Provenance lock for GLMRT_XGRAMMAR_SOURCE_DIR")

if(GLMRT_ENABLE_B12X_AOT)
  message(DEPRECATION
    "GLMRT_ENABLE_B12X_AOT is deprecated; use GLMRT_ENABLE_SPARKINFER_AOT")
  set(GLMRT_ENABLE_SPARKINFER_AOT ON)
endif()
if(GLMRT_ENABLE_B12X_COORDINATOR_AOT)
  message(DEPRECATION
    "GLMRT_ENABLE_B12X_COORDINATOR_AOT is deprecated; "
    "use GLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT")
  set(GLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(GLMRT_NATIVE_SOURCES src/glmrt_native.cc)

if(GLMRT_ENABLE_XGRAMMAR)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  if(NOT EXISTS "${GLMRT_XGRAMMAR_SOURCE_DIR}/include/xgrammar/compiler.h" OR
     NOT EXISTS "${GLMRT_XGRAMMAR_SOURCE_DIR}/3rdparty/dlpack/include/dlpack/dlpack.h" OR
     NOT EXISTS "${GLMRT_XGRAMMAR_SOURCE_DIR}/3rdparty/picojson/picojson.h")
    message(FATAL_ERROR
      "Pinned XGrammar source is incomplete under ${GLMRT_XGRAMMAR_SOURCE_DIR}; "
      "initialize third_party/xgrammar and its nested submodules")
  endif()
  set(GLMRT_XGRAMMAR_VERIFY_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/verify-xgrammar-source.py")
  if(NOT EXISTS "${GLMRT_XGRAMMAR_VERIFY_SCRIPT}")
    message(FATAL_ERROR
      "XGrammar source verifier was not found: ${GLMRT_XGRAMMAR_VERIFY_SCRIPT}")
  endif()
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${GLMRT_XGRAMMAR_VERIFY_SCRIPT}"
      --source "${GLMRT_XGRAMMAR_SOURCE_DIR}"
      --lock "${GLMRT_XGRAMMAR_LOCK_FILE}"
    RESULT_VARIABLE GLMRT_XGRAMMAR_VERIFY_RESULT
    OUTPUT_VARIABLE GLMRT_XGRAMMAR_VERIFY_OUTPUT
    ERROR_VARIABLE GLMRT_XGRAMMAR_VERIFY_ERROR
  )
  if(NOT GLMRT_XGRAMMAR_VERIFY_RESULT EQUAL 0)
    message(FATAL_ERROR
      "XGrammar source verification failed:\n${GLMRT_XGRAMMAR_VERIFY_ERROR}")
  endif()
  string(STRIP "${GLMRT_XGRAMMAR_VERIFY_OUTPUT}" GLMRT_XGRAMMAR_VERIFY_OUTPUT)
  message(STATUS "${GLMRT_XGRAMMAR_VERIFY_OUTPUT}")
  file(GLOB_RECURSE GLMRT_XGRAMMAR_SOURCES
    CONFIGURE_DEPENDS
    "${GLMRT_XGRAMMAR_SOURCE_DIR}/cpp/*.cc"
  )
  list(FILTER GLMRT_XGRAMMAR_SOURCES EXCLUDE REGEX "/cpp/tvm_ffi/.*\\.cc$")
  add_library(glmrt_xgrammar STATIC ${GLMRT_XGRAMMAR_SOURCES})
  target_include_directories(glmrt_xgrammar
    PUBLIC "${GLMRT_XGRAMMAR_SOURCE_DIR}/include"
  )
  target_include_directories(glmrt_xgrammar SYSTEM PUBLIC
    "${GLMRT_XGRAMMAR_SOURCE_DIR}/3rdparty/dlpack/include"
    "${GLMRT_XGRAMMAR_SOURCE_DIR}/3rdparty/picojson"
  )
  target_compile_definitions(glmrt_xgrammar PUBLIC
    XGRAMMAR_ENABLE_CPPTRACE=0
    XGRAMMAR_ENABLE_INTERNAL_CHECK=0
  )
  find_package(Threads REQUIRED)
  target_link_libraries(glmrt_xgrammar PUBLIC Threads::Threads)
  list(APPEND GLMRT_NATIVE_SOURCES src/xgrammar_adapter.cc)
endif()

set(GLMRT_BUILD_SPARKINFER_AOT OFF)
if(GLMRT_ENABLE_SPARKINFER_AOT OR GLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT OR GLMRT_ENABLE_W8A16_AOT)
  set(GLMRT_BUILD_SPARKINFER_AOT ON)
endif()
if(GLMRT_BUILD_SPARKINFER_AOT)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  if(NOT EXISTS "${GLMRT_SPARKINFER_SOURCE_DIR}/b12x/__init__.py")
    message(FATAL_ERROR
      "SparkInfer was not found under GLMRT_SPARKINFER_SOURCE_DIR="
      "${GLMRT_SPARKINFER_SOURCE_DIR}. Initialize third_party/sparkinfer or "
      "select an explicit pinned source tree.")
  endif()
  set(GLMRT_SPARKINFER_VERIFY_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/verify-sparkinfer-source.py")
  set(GLMRT_SPARKINFER_PYTHON_BOOTSTRAP
    "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/_pinned_sparkinfer.py")
  if(NOT EXISTS "${GLMRT_SPARKINFER_VERIFY_SCRIPT}")
    message(FATAL_ERROR
      "SparkInfer source verifier was not found: ${GLMRT_SPARKINFER_VERIFY_SCRIPT}")
  endif()
  if(NOT EXISTS "${GLMRT_SPARKINFER_PYTHON_BOOTSTRAP}")
    message(FATAL_ERROR
      "SparkInfer Python bootstrap was not found: "
      "${GLMRT_SPARKINFER_PYTHON_BOOTSTRAP}")
  endif()
  set(GLMRT_SPARKINFER_VERIFY_COMMAND
    "${Python3_EXECUTABLE}" "${GLMRT_SPARKINFER_VERIFY_SCRIPT}"
    --source "${GLMRT_SPARKINFER_SOURCE_DIR}"
    --lock "${GLMRT_SPARKINFER_LOCK_FILE}"
  )
  set(GLMRT_SPARKINFER_PYTHON_ENV
    "GLMRT_SPARKINFER_SOURCE_DIR=${GLMRT_SPARKINFER_SOURCE_DIR}"
    "GLMRT_SPARKINFER_LOCK_FILE=${GLMRT_SPARKINFER_LOCK_FILE}"
    "PYTHONDONTWRITEBYTECODE=1"
    "PYTHONPATH=${GLMRT_SPARKINFER_SOURCE_DIR}"
  )
  execute_process(
    COMMAND ${GLMRT_SPARKINFER_VERIFY_COMMAND}
    RESULT_VARIABLE GLMRT_SPARKINFER_VERIFY_RESULT
    OUTPUT_VARIABLE GLMRT_SPARKINFER_VERIFY_OUTPUT
    ERROR_VARIABLE GLMRT_SPARKINFER_VERIFY_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
  )
  if(NOT GLMRT_SPARKINFER_VERIFY_RESULT EQUAL 0)
    message(FATAL_ERROR
      "Pinned SparkInfer verification failed:\n${GLMRT_SPARKINFER_VERIFY_ERROR}")
  endif()
  message(STATUS "${GLMRT_SPARKINFER_VERIFY_OUTPUT}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${GLMRT_SPARKINFER_LOCK_FILE}"
    "${GLMRT_SPARKINFER_VERIFY_SCRIPT}"
  )
  set(GLMRT_SPARKINFER_PROVENANCE_INPUTS
    "${GLMRT_SPARKINFER_LOCK_FILE}"
    "${GLMRT_SPARKINFER_VERIFY_SCRIPT}"
    "${GLMRT_SPARKINFER_PYTHON_BOOTSTRAP}"
  )
  file(GLOB_RECURSE GLMRT_SPARKINFER_EXPORT_INPUTS
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "${GLMRT_SPARKINFER_SOURCE_DIR}/b12x/*.py"
    "${GLMRT_SPARKINFER_SOURCE_DIR}/b12x/*.cu"
    "${GLMRT_SPARKINFER_SOURCE_DIR}/b12x/*.cuh"
    "${GLMRT_SPARKINFER_SOURCE_DIR}/b12x/*.h"
    "${GLMRT_SPARKINFER_SOURCE_DIR}/b12x/*.hpp"
  )
  list(APPEND GLMRT_SPARKINFER_EXPORT_INPUTS
    "${GLMRT_SPARKINFER_SOURCE_DIR}/pyproject.toml"
  )
  add_custom_target(glmrt_verify_sparkinfer_source
    COMMAND ${GLMRT_SPARKINFER_VERIFY_COMMAND}
    COMMENT "Verifying pinned SparkInfer source"
    VERBATIM
  )
endif()

if(GLMRT_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  list(APPEND GLMRT_NATIVE_SOURCES
    cuda/kernels/norm.cu
    cuda/kernels/mlp.cu
    cuda/kernels/residual.cu
    cuda/kernels/kv_cache.cu
    cuda/kernels/dsa_indexer.cu
    cuda/kernels/router.cu
    cuda/kernels/linear.cu
    cuda/kernels/attention.cu
    cuda/kernels/embedding.cu
    cuda/kernels/sampling.cu
    cuda/kernels/nibbles.cu
    cuda/kernels/nvfp4_payload.cu
  )
endif()

set(GLMRT_BUILD_FLASHINFER_PACKED_MLA OFF)
if(GLMRT_ENABLE_FLASHINFER_PACKED_MLA OR GLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT OR GLMRT_ENABLE_W8A16_AOT)
  set(GLMRT_BUILD_FLASHINFER_PACKED_MLA ON)
endif()

if(GLMRT_BUILD_FLASHINFER_PACKED_MLA)
  if(NOT GLMRT_ENABLE_CUDA)
    message(FATAL_ERROR "GLMRT_ENABLE_FLASHINFER_PACKED_MLA=ON requires GLMRT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "import pathlib, flashinfer; print(pathlib.Path(flashinfer.__file__).resolve().parent / 'data' / 'include')"
    OUTPUT_VARIABLE GLMRT_FLASHINFER_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  if(NOT EXISTS "${GLMRT_FLASHINFER_INCLUDE_DIR}/flashinfer/attention/sparse_mla_sm120/decode_dsv3_2_kernel.cuh")
    message(FATAL_ERROR
      "FlashInfer SM120 sparse-MLA headers were not found under ${GLMRT_FLASHINFER_INCLUDE_DIR}")
  endif()
  list(APPEND GLMRT_NATIVE_SOURCES
    cuda/kernels/packed_fp8_mla_exact.cu
  )
  set_source_files_properties(cuda/kernels/packed_fp8_mla_exact.cu PROPERTIES
    COMPILE_OPTIONS "-use_fast_math;-O3"
  )
  if(GLMRT_CUDA_ARCHITECTURES STREQUAL "120")
    set(GLMRT_CUDA_ARCHITECTURES "120f")
  endif()
endif()

if(GLMRT_ENABLE_SPARKINFER_AOT)
  if(NOT GLMRT_ENABLE_CUDA)
    message(FATAL_ERROR "GLMRT_ENABLE_SPARKINFER_AOT=ON requires GLMRT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(GLMRT_B12X_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/b12x_aot")
  set(GLMRT_B12X_AOT_REGIMES 1 2 4 8 16 32 64 128 256)
  set(GLMRT_B12X_AOT_TOPK8_PREFILL_REGIMES 2 4 8 16 32 64 128 256 512 1024 2048 2064)
  set(GLMRT_B12X_EXL3_K3_REGIMES 1 2 4 8 9 16 32 64 128 256 257 512 1024 2048 2064)
  set(GLMRT_B12X_EXL3_K4_REGIMES
    1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
    17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32
    64 128 256 257 512 1024 2048 2064
  )
  set(GLMRT_B12X_AOT_OBJECTS)
  set(GLMRT_B12X_AOT_HEADERS)
  list(APPEND GLMRT_B12X_AOT_OBJECTS
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1.o"
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1_fused_sum.o"
  )
  foreach(regime IN LISTS GLMRT_B12X_AOT_TOPK8_PREFILL_REGIMES)
    list(APPEND GLMRT_B12X_AOT_OBJECTS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_prefill_m${regime}_topk8.o"
    )
  endforeach()
  foreach(regime IN LISTS GLMRT_B12X_AOT_REGIMES)
    list(APPEND GLMRT_B12X_AOT_OBJECTS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_top1_m${regime}.o"
    )
  endforeach()
  foreach(regime IN LISTS GLMRT_B12X_EXL3_K3_REGIMES)
    list(APPEND GLMRT_B12X_AOT_OBJECTS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_exl3_k3_m${regime}_topk8.o"
    )
  endforeach()
  foreach(regime IN LISTS GLMRT_B12X_EXL3_K4_REGIMES)
    list(APPEND GLMRT_B12X_AOT_OBJECTS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_exl3_k4_m${regime}_topk8.o"
    )
  endforeach()
  list(APPEND GLMRT_B12X_AOT_OBJECTS
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_exl3_k3_topk_sum.o"
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_exl3_k3_topk_sum_bf16.o"
  )
  list(APPEND GLMRT_B12X_AOT_HEADERS
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1.h"
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_decode_m1_fused_sum.h"
    "${GLMRT_B12X_AOT_DIR}/b12x_spark_moe_aot_config.h"
  )
  foreach(regime IN LISTS GLMRT_B12X_AOT_TOPK8_PREFILL_REGIMES)
    list(APPEND GLMRT_B12X_AOT_HEADERS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_prefill_m${regime}_topk8.h"
    )
  endforeach()
  foreach(regime IN LISTS GLMRT_B12X_AOT_REGIMES)
    list(APPEND GLMRT_B12X_AOT_HEADERS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_top1_m${regime}.h"
    )
  endforeach()
  foreach(regime IN LISTS GLMRT_B12X_EXL3_K3_REGIMES)
    list(APPEND GLMRT_B12X_AOT_HEADERS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_exl3_k3_m${regime}_topk8.h"
    )
  endforeach()
  foreach(regime IN LISTS GLMRT_B12X_EXL3_K4_REGIMES)
    list(APPEND GLMRT_B12X_AOT_HEADERS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_exl3_k4_m${regime}_topk8.h"
    )
  endforeach()
  list(APPEND GLMRT_B12X_AOT_HEADERS
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_exl3_k3_topk_sum.h"
    "${GLMRT_B12X_AOT_DIR}/moe_tp4_exl3_k3_topk_sum_bf16.h"
  )
  add_custom_command(
    OUTPUT
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_aot.stamp"
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_moe_aot.meta"
      ${GLMRT_B12X_AOT_OBJECTS}
      ${GLMRT_B12X_AOT_HEADERS}
    COMMAND ${GLMRT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${GLMRT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_moe_aot.py"
      --output-dir "${GLMRT_B12X_AOT_DIR}"
      --target-sms 48
    COMMAND "${CMAKE_COMMAND}" -E touch
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_aot.stamp"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_moe_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/_b12x_exl3_k3_profile.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/_b12x_exl3_k4_profile.py"
      ${GLMRT_SPARKINFER_PROVENANCE_INPUTS}
      ${GLMRT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting Spark-side SparkInfer AOT kernels"
    VERBATIM
  )
  add_custom_target(glmrt_b12x_aot_export
    DEPENDS
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_aot.stamp"
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_moe_aot.meta"
      ${GLMRT_B12X_AOT_OBJECTS}
      ${GLMRT_B12X_AOT_HEADERS}
  )
  add_dependencies(glmrt_b12x_aot_export glmrt_verify_sparkinfer_source)
  set_source_files_properties(${GLMRT_B12X_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND GLMRT_NATIVE_SOURCES
    cuda/kernels/b12x_direct.cu
    ${GLMRT_B12X_AOT_OBJECTS}
  )

  set(GLMRT_B12X_M1_PARITY_AOT_REGIMES 2 3 4 5 6 7 8)
  set(GLMRT_B12X_M1_PARITY_AOT_OBJECTS)
  set(GLMRT_B12X_M1_PARITY_AOT_HEADERS
    "${GLMRT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot_config.h"
  )
  foreach(regime IN LISTS GLMRT_B12X_M1_PARITY_AOT_REGIMES)
    list(APPEND GLMRT_B12X_M1_PARITY_AOT_OBJECTS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_m${regime}_topk8.o"
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_m${regime}_topk8.o"
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_wide_m${regime}_topk8.o"
    )
    list(APPEND GLMRT_B12X_M1_PARITY_AOT_HEADERS
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_m${regime}_topk8.h"
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_m${regime}_topk8.h"
      "${GLMRT_B12X_AOT_DIR}/moe_tp4_w4a16_m1_parity_grouped_wide_m${regime}_topk8.h"
    )
  endforeach()
  add_custom_command(
    OUTPUT
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.stamp"
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.meta"
      ${GLMRT_B12X_M1_PARITY_AOT_OBJECTS}
      ${GLMRT_B12X_M1_PARITY_AOT_HEADERS}
    COMMAND ${GLMRT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${GLMRT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_w4a16_m1_parity_aot.py"
      --output-dir "${GLMRT_B12X_AOT_DIR}"
      --target-sms 48
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_spark_w4a16_m1_parity_aot.py"
      ${GLMRT_SPARKINFER_PROVENANCE_INPUTS}
      ${GLMRT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting ordered Spark W4A16 M=2..8 parity candidates"
    VERBATIM
  )
  add_custom_target(glmrt_b12x_m1_parity_aot_export
    DEPENDS
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.stamp"
      "${GLMRT_B12X_AOT_DIR}/b12x_spark_w4a16_m1_parity_aot.meta"
      ${GLMRT_B12X_M1_PARITY_AOT_OBJECTS}
      ${GLMRT_B12X_M1_PARITY_AOT_HEADERS}
  )
  add_dependencies(
    glmrt_b12x_m1_parity_aot_export
    glmrt_verify_sparkinfer_source
  )
  set_source_files_properties(${GLMRT_B12X_M1_PARITY_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND GLMRT_NATIVE_SOURCES ${GLMRT_B12X_M1_PARITY_AOT_OBJECTS})

endif()

if(GLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT)
  if(NOT GLMRT_ENABLE_CUDA)
    message(FATAL_ERROR
      "GLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT=ON requires GLMRT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(GLMRT_B12X_COORDINATOR_AOT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/b12x_coordinator_aot"
  )
  set(GLMRT_B12X_COORDINATOR_AOT_OBJECTS
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m8.o"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m16_candidate.o"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1.o"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m16_candidate.o"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1_tn64_candidate.o"
  )
  set(GLMRT_B12X_COORDINATOR_AOT_HEADERS
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m8.h"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_q_b_m16_candidate.h"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1.h"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m16_candidate.h"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/coordinator_w4a16_o_proj_m1_tn64_candidate.h"
    "${GLMRT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot_config.h"
  )
  add_custom_command(
    OUTPUT
      "${GLMRT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.stamp"
      "${GLMRT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.meta"
      ${GLMRT_B12X_COORDINATOR_AOT_OBJECTS}
      ${GLMRT_B12X_COORDINATOR_AOT_HEADERS}
    COMMAND ${GLMRT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${GLMRT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_coordinator_aot.py"
      --output-dir "${GLMRT_B12X_COORDINATOR_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_b12x_coordinator_aot.py"
      ${GLMRT_SPARKINFER_PROVENANCE_INPUTS}
      ${GLMRT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting coordinator SparkInfer AOT kernels"
    VERBATIM
  )
  add_custom_target(glmrt_b12x_coordinator_aot_export
    DEPENDS
      "${GLMRT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.stamp"
      "${GLMRT_B12X_COORDINATOR_AOT_DIR}/b12x_coordinator_aot.meta"
      ${GLMRT_B12X_COORDINATOR_AOT_OBJECTS}
      ${GLMRT_B12X_COORDINATOR_AOT_HEADERS}
  )
  add_dependencies(
    glmrt_b12x_coordinator_aot_export
    glmrt_verify_sparkinfer_source
  )
  set_source_files_properties(${GLMRT_B12X_COORDINATOR_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND GLMRT_NATIVE_SOURCES
    cuda/kernels/b12x_coordinator.cu
    ${GLMRT_B12X_COORDINATOR_AOT_OBJECTS}
  )
endif()

if(GLMRT_ENABLE_W8A16_AOT)
  if(NOT GLMRT_ENABLE_CUDA)
    message(FATAL_ERROR "GLMRT_ENABLE_W8A16_AOT=ON requires GLMRT_ENABLE_CUDA=ON")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(GLMRT_W8A16_AOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/w8a16_aot")
  set(GLMRT_W8A16_AOT_HEADER
    "${GLMRT_W8A16_AOT_DIR}/w8a16_row_major_aot.h"
  )
  set(GLMRT_W8A16_AOT_SOURCE
    "${GLMRT_W8A16_AOT_DIR}/w8a16_row_major_aot.cc"
  )
  set(GLMRT_W8A16_PACKED_O_AOT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/w8a16_packed_o_aot"
  )
  set(GLMRT_W8A16_PACKED_O_AOT_OBJECTS
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m16.o"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m32.o"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m64.o"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m128.o"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m256.o"
  )
  set(GLMRT_W8A16_PACKED_O_AOT_HEADERS
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m16.h"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m32.h"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m64.h"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m128.h"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_m256.h"
    "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot_config.h"
  )
  add_custom_command(
    OUTPUT
      "${GLMRT_W8A16_AOT_DIR}/w8a16_row_major_aot.stamp"
      "${GLMRT_W8A16_AOT_HEADER}"
      "${GLMRT_W8A16_AOT_SOURCE}"
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_row_major_aot.py"
      --output-dir "${GLMRT_W8A16_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_row_major_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/tune_w8a16_triton_prefill.py"
    COMMENT "Exporting coordinator row-major W8A16 AOT kernels"
    VERBATIM
  )
  add_custom_target(glmrt_w8a16_aot_export
    DEPENDS
      "${GLMRT_W8A16_AOT_DIR}/w8a16_row_major_aot.stamp"
      "${GLMRT_W8A16_AOT_HEADER}"
      "${GLMRT_W8A16_AOT_SOURCE}"
  )
  add_custom_command(
    OUTPUT
      "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.stamp"
      "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.meta"
      ${GLMRT_W8A16_PACKED_O_AOT_OBJECTS}
      ${GLMRT_W8A16_PACKED_O_AOT_HEADERS}
    COMMAND ${GLMRT_SPARKINFER_VERIFY_COMMAND}
    COMMAND "${CMAKE_COMMAND}" -E env
      ${GLMRT_SPARKINFER_PYTHON_ENV}
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_packed_o_aot.py"
      --output-dir "${GLMRT_W8A16_PACKED_O_AOT_DIR}"
    DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/export_w8a16_packed_o_aot.py"
      "${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/tune_w8a16_cute_packed_prefill.py"
      ${GLMRT_SPARKINFER_PROVENANCE_INPUTS}
      ${GLMRT_SPARKINFER_EXPORT_INPUTS}
    COMMENT "Exporting coordinator packed W8A16 O-projection AOT kernels"
    VERBATIM
  )
  add_custom_target(glmrt_w8a16_packed_o_aot_export
    DEPENDS
      "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.stamp"
      "${GLMRT_W8A16_PACKED_O_AOT_DIR}/w8a16_packed_o_aot.meta"
      ${GLMRT_W8A16_PACKED_O_AOT_OBJECTS}
      ${GLMRT_W8A16_PACKED_O_AOT_HEADERS}
  )
  add_dependencies(
    glmrt_w8a16_packed_o_aot_export
    glmrt_verify_sparkinfer_source
  )
  set_source_files_properties("${GLMRT_W8A16_AOT_SOURCE}" PROPERTIES
    GENERATED TRUE
  )
  set_source_files_properties(${GLMRT_W8A16_PACKED_O_AOT_OBJECTS} PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE
  )
  list(APPEND GLMRT_NATIVE_SOURCES
    "${GLMRT_W8A16_AOT_SOURCE}"
    cuda/kernels/w8a16_packed.cu
    ${GLMRT_W8A16_PACKED_O_AOT_OBJECTS}
  )
endif()

if(GLMRT_ENABLE_SPARKINFER_AOT OR GLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT OR GLMRT_ENABLE_W8A16_AOT)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m cutlass.cute.export.aot_config --libdir
    OUTPUT_VARIABLE GLMRT_B12X_AOT_RUNTIME_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  find_library(GLMRT_B12X_AOT_RUNTIME_LIBRARY
    NAMES cute_dsl_runtime
    PATHS "${GLMRT_B12X_AOT_RUNTIME_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
  )
endif()

add_library(glmrt_native SHARED ${GLMRT_NATIVE_SOURCES})
target_include_directories(glmrt_native PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(GLMRT_ENABLE_XGRAMMAR)
  target_link_libraries(glmrt_native PRIVATE glmrt_xgrammar)
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_XGRAMMAR=1)
else()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_XGRAMMAR=0)
endif()
if(GLMRT_ENABLE_CUDA)
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_CUDA=1)
  target_link_libraries(glmrt_native PUBLIC CUDA::cudart CUDA::cublas CUDA::cuda_driver)
  set_target_properties(glmrt_native PROPERTIES
    CUDA_ARCHITECTURES "${GLMRT_CUDA_ARCHITECTURES}"
  )
else()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_CUDA=0)
endif()
if(GLMRT_BUILD_FLASHINFER_PACKED_MLA)
  target_include_directories(glmrt_native PRIVATE "${GLMRT_FLASHINFER_INCLUDE_DIR}")
endif()
if(GLMRT_ENABLE_SPARKINFER_AOT)
  add_dependencies(
    glmrt_native
    glmrt_b12x_aot_export
    glmrt_b12x_m1_parity_aot_export
  )
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_B12X_AOT=1)
  target_include_directories(glmrt_native PRIVATE "${GLMRT_B12X_AOT_DIR}")
else()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_B12X_AOT=0)
endif()
if(GLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT)
  add_dependencies(glmrt_native glmrt_b12x_coordinator_aot_export)
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_B12X_COORDINATOR_AOT=1)
  target_include_directories(glmrt_native PRIVATE "${GLMRT_B12X_COORDINATOR_AOT_DIR}")
else()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_B12X_COORDINATOR_AOT=0)
endif()
if(GLMRT_ENABLE_W8A16_AOT)
  add_dependencies(glmrt_native glmrt_w8a16_aot_export)
  add_dependencies(glmrt_native glmrt_w8a16_packed_o_aot_export)
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_W8A16_AOT=1)
  target_include_directories(glmrt_native PRIVATE "${GLMRT_W8A16_AOT_DIR}")
  target_include_directories(glmrt_native PRIVATE "${GLMRT_W8A16_PACKED_O_AOT_DIR}")
else()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_W8A16_AOT=0)
endif()
if(GLMRT_ENABLE_SPARKINFER_AOT OR GLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT OR GLMRT_ENABLE_W8A16_AOT)
  target_link_libraries(glmrt_native PUBLIC "${GLMRT_B12X_AOT_RUNTIME_LIBRARY}")
  set_property(TARGET glmrt_native APPEND PROPERTY BUILD_RPATH "${GLMRT_B12X_AOT_RUNTIME_DIR}")
endif()
if(GLMRT_ENABLE_RDMA)
  find_path(GLMRT_IBVERBS_INCLUDE_DIR infiniband/verbs.h)
  find_library(GLMRT_IBVERBS_LIBRARY NAMES ibverbs)
  if(NOT GLMRT_IBVERBS_INCLUDE_DIR OR NOT GLMRT_IBVERBS_LIBRARY)
    message(FATAL_ERROR "GLMRT_ENABLE_RDMA=ON requires libibverbs headers and library")
  endif()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_RDMA=1)
  target_include_directories(glmrt_native PRIVATE ${GLMRT_IBVERBS_INCLUDE_DIR})
  target_link_libraries(glmrt_native PUBLIC ${GLMRT_IBVERBS_LIBRARY})
else()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_RDMA=0)
endif()
if(GLMRT_ENABLE_NCCL)
  if(NOT GLMRT_ENABLE_CUDA)
    message(FATAL_ERROR "GLMRT_ENABLE_NCCL=ON requires GLMRT_ENABLE_CUDA=ON")
  endif()
  find_path(GLMRT_NCCL_INCLUDE_DIR nccl.h REQUIRED)
  find_library(GLMRT_NCCL_LIBRARY NAMES nccl REQUIRED)
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_NCCL=1)
  target_include_directories(glmrt_native PRIVATE ${GLMRT_NCCL_INCLUDE_DIR})
  target_link_libraries(glmrt_native PUBLIC ${GLMRT_NCCL_LIBRARY})
else()
  target_compile_definitions(glmrt_native PRIVATE GLMRT_NATIVE_ENABLE_NCCL=0)
endif()

add_executable(glmrt_native_selftest tests/native_selftest.cc)
target_link_libraries(glmrt_native_selftest PRIVATE glmrt_native)
# The self-test intentionally uses assertions for both the ABI call and its
# result check. Keep those expressions live in Release builds; defining
# NDEBUG would otherwise skip allocations and make the following copy probe
# dereference an uninitialized buffer.
if(MSVC)
  target_compile_options(glmrt_native_selftest PRIVATE /UNDEBUG)
else()
  target_compile_options(glmrt_native_selftest PRIVATE -UNDEBUG)
endif()

enable_testing()
add_test(NAME glmrt_native_selftest COMMAND glmrt_native_selftest)

if(GLMRT_ENABLE_XGRAMMAR)
  add_executable(glmrt_xgrammar_selftest tests/xgrammar_selftest.cc)
  target_link_libraries(glmrt_xgrammar_selftest PRIVATE glmrt_native)
  target_compile_definitions(glmrt_xgrammar_selftest PRIVATE
    "GLMRT_XGRAMMAR_TEST_TOKENIZER=\"${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/xgrammar_tiny_tokenizer.json\""
  )
  if(MSVC)
    target_compile_options(glmrt_xgrammar_selftest PRIVATE /UNDEBUG)
  else()
    target_compile_options(glmrt_xgrammar_selftest PRIVATE -UNDEBUG)
  endif()
  add_test(NAME glmrt_xgrammar_selftest COMMAND glmrt_xgrammar_selftest)
endif()

if(GLMRT_ENABLE_CUDA)
  add_executable(glmrt_cuda_selftest tests/cuda_selftest.cc)
  target_link_libraries(glmrt_cuda_selftest PRIVATE glmrt_native CUDA::cudart CUDA::cublas CUDA::cuda_driver)
  add_test(NAME glmrt_cuda_selftest COMMAND glmrt_cuda_selftest)
endif()
