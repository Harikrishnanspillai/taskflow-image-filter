cmake_minimum_required(VERSION 3.24)
project(TaskFlowImageFilter LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architecture (RTX 3050 = 86)")

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

set(TASKFLOW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/taskflow" CACHE PATH
    "Path to Taskflow checkout")

if(NOT EXISTS "${TASKFLOW_DIR}/taskflow/taskflow.hpp")
  message(FATAL_ERROR
    "Taskflow headers not found at ${TASKFLOW_DIR}. Clone Taskflow into third_party/taskflow.")
endif()

add_executable(image_filter
  src/main.cpp
  src/preprocessing.cpp
  src/cpu_filters.cpp
  src/gpu_filters.cu
  src/taskflow_pipeline.cpp
  src/benchmark.cpp
)

target_include_directories(image_filter PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${TASKFLOW_DIR}"
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(image_filter PRIVATE ${OpenCV_LIBS})
target_link_options(image_filter PRIVATE -static-libstdc++)
target_compile_options(image_filter PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O2>
  $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)
set_target_properties(image_filter PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
