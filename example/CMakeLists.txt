cmake_minimum_required(VERSION 3.8)
project(VectorCopy)

# FindCUDA
include(CheckLanguage)
check_language(CUDA QUIET)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
  add_library(vc.cuda STATIC ${CMAKE_CURRENT_SOURCE_DIR}/vc.cu)
endif()
