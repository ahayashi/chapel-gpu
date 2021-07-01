cmake_minimum_required(VERSION 3.8)
project(VectorCopy)

# FindCUDA
include(CheckLanguage)
check_language(CUDA QUIET)

# FindHIP
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
find_package(HIP QUIET)

# FindOpenCL
find_package(OpenCL QUIET)

# FindDPC++
find_program(DPCPP_FOUND dpcpp QUIET)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
  add_library(vc.cuda STATIC ${CMAKE_CURRENT_SOURCE_DIR}/vc.cu)
endif()

if(HIP_FOUND)
  if(EXISTS "${HIP_ROOT_DIR}/hip/bin/hipify-perl")
    message(STATUS "Found HIP: " ${HIP_VERSION})
    message(STATUS "Found HIPIFY: " ${HIP_ROOT_DIR}/hip/bin/hipify-perl)
    add_custom_command(
      OUTPUT vc.hip.cc
      COMMAND ${HIP_ROOT_DIR}/hip/bin/hipify-perl ${CMAKE_CURRENT_SOURCE_DIR}/vc.cu > vc.hip.cc
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vc.cu
      COMMENT "Convering vc.cu to vc.hip.cc"
      )
    set(CMAKE_CXX_COMPILER "${HIP_ROOT_DIR}/hip/bin/hipcc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-gpu-rdc -fPIC")
    hip_add_library(vc.hip STATIC vc.hip.cc)
  else ()
    message(STATUS "Found HIP, but HIPIFY NOTFOUND")
    set(HIP_FOUND OFF)
  endif()
else()
    message(STATUS "HIP NOTFOUND")
endif()

if(OpenCL_FOUND)
  message(STATUS "Found OpenCL: " ${OpenCL_VERSION_STRING})
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
  add_library(vc.opencl STATIC ${CMAKE_CURRENT_SOURCE_DIR}/vc.opencl.c)
  target_link_libraries(vc.opencl OpenCL::OpenCL)
else()
  message(STATUS "OpenCL Not Found")
endif()

if(DPCPP_FOUND)
  find_program(DPCT_FOUND dpct QUIET)
  if(DPCT_FOUND)
    message(STATUS "Found DPC++: " ${DPCPP_FOUND})
    message(STATUS "Found DPCT: " ${DPCT_FOUND})
    if(CMAKE_CUDA_COMPILER)
      set(CUDA_INCLUDE_PATH ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    elseif(DEFINED ENV{CUDA_HOME})
      set(CUDA_INCLUDE_PATH $ENV{CUDA_HOME}/include)
    else()
      message(FATAL_ERROR "Unable to find CUDA cmake or CUDA_HOME environment variable")
    endif()
    add_custom_command(
      OUTPUT vc.dp.cpp
      COMMAND dpct --in-root=${CMAKE_CURRENT_SOURCE_DIR}/ --out-root=. --cuda-include-path=${CUDA_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/vc.cu
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vc.cu
      COMMENT "Convering vc.cu to vc.dp.cpp"
    )
  else()
	message(STATUS "Found DPC++, but DPCT NOTFOUND")
    set(DPCPP_FOUND OFF)
  endif()
  set(CMAKE_CXX_COMPILER "dpcpp")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
  add_library(vc.dpcpp STATIC vc.dp.cpp)
else()
	message(STATUS "DPC++ Not Found")
endif()
