#[[
Copyright (c) 2026 LightSeek Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
]]

cmake_minimum_required(VERSION 3.16)
project(tokenspeed_kernel_npu_ascend_c LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(
    VLLM_ASCEND_CSRC
    "${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/vllm-ascend/csrc"
)
if(NOT EXISTS "${VLLM_ASCEND_CSRC}/aclnn_torch_adapter/op_api_common.h")
    message(FATAL_ERROR "vllm-ascend source submodule is required")
endif()

if(NOT DEFINED PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE python3)
endif()

if(NOT DEFINED ASCEND_HOME_PATH)
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
if(ASCEND_HOME_PATH STREQUAL "")
    set(ASCEND_HOME_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_NPU_IMPORT_RESULT
)
if(NOT TORCH_NPU_IMPORT_RESULT EQUAL 0)
    message(FATAL_ERROR "torch_npu is required to build tokenspeed-kernel-npu")
endif()

find_package(Torch REQUIRED)

add_library(tokenspeed_ascend_c SHARED
    torch_binding.cpp
    "${VLLM_ASCEND_CSRC}/aclnn_torch_adapter/NPUBridge.cpp"
    "${VLLM_ASCEND_CSRC}/aclnn_torch_adapter/NPUStorageImpl.cpp"
)

target_include_directories(tokenspeed_ascend_c PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${VLLM_ASCEND_CSRC}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_PATH}/include
    ${ASCEND_HOME_PATH}/include
)

target_link_directories(tokenspeed_ascend_c PRIVATE
    ${TORCH_LIBRARY_DIRS}
    ${TORCH_NPU_PATH}/lib
    ${ASCEND_HOME_PATH}/lib64
)

target_link_libraries(tokenspeed_ascend_c PRIVATE
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    opapi
    dl
)

set_target_properties(tokenspeed_ascend_c PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    INSTALL_RPATH "$ORIGIN"
)
