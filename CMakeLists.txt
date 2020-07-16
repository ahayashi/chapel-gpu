cmake_minimum_required(VERSION 3.8)
project(ChapelGPUAPI)

# FindCUDA
include(CheckLanguage)
check_language(CUDA)

# FindHIP
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
find_package(HIP)

# FindOpenCL
find_package(OpenCL)

set(GPU_COMPILER_FOUND OFF)

if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
  add_library(GPUAPICUDA SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.cu)
  add_library(GPUAPICUDA_static STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.cu)
  set_target_properties(GPUAPI PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.h)
  set(GPU_COMPILER_FOUND ON)
endif()

if(HIP_FOUND)
  if(EXISTS "${HIP_ROOT_DIR}/hip/bin/hipify-perl")
    message(STATUS "Found HIPIFY: " ${HIP_ROOT_DIR}/hip/bin/hipify-perl)
    add_custom_command(
      OUTPUT GPUAPI.hip.cc
      COMMAND ${HIP_ROOT_DIR}/hip/bin/hipify-perl ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.cu > GPUAPI.hip.cc
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.cu
      COMMENT "Convering GPUAPI.cu to GPUAPI.hip.cc"
      )
    set(CMAKE_CXX_COMPILER "${HIP_ROOT_DIR}/hip/bin/hipcc")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-gpu-rdc")
    hip_add_library(GPUAPIHIP SHARED GPUAPI.hip.cc)
    hip_add_library(GPUAPIHIP_static STATIC GPUAPI.hip.cc)
    set_target_properties(GPUAPIHIP PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.h)
    set(GPU_COMPILER_FOUND ON)
  else ()
    set(HIP_FOUND OFF)
  endif()
endif()

if(OpenCL_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
  add_library(GPUAPIOPENCL SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.opencl.c)
  add_library(GPUAPIOPENCL_static STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.opencl.c)
  set(GPU_COMPILER_FOUND ON)
endif()

if(NOT GPU_COMPILER_FOUND)
  message(FATAL_ERROR "No GPU compiler found")
endif()

# Installation

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set (CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "default install path" FORCE)
endif()

include(GNUInstallDirs)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR} CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKEDIR CMake)
else()
  set(DEF_INSTALL_CMAKEDIR share/cmake/${PROJECT_NAME})
endif()
set(INSTALL_CMAKEDIR ${DEF_INSTALL_CMAKEDIR} CACHE PATH "Installation directory for CMake files")

# Report to user
foreach(p LIB INCLUDE)
  file(TO_NATIVE_PATH ${CMAKE_INSTALL_PREFIX}/${INSTALL_${p}DIR} _path )
  message(STATUS "Installing ${p} components to ${_path}")
  unset(_path)
endforeach()

if(CMAKE_CUDA_COMPILER)
  install(
    TARGETS GPUAPICUDA GPUAPICUDA_static
    ARCHIVE DESTINATION ${INSTALL_LIBDIR}
    LIBRARY DESTINATION ${INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${INSTALL_INCLUDEDIR}
    )
endif()

if (HIP_FOUND)
  install(
    TARGETS GPUAPIHIP GPUAPIHIP_static
    ARCHIVE DESTINATION ${INSTALL_LIBDIR}
    LIBRARY DESTINATION ${INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${INSTALL_INCLUDEDIR}
    )
endif()

if(OpenCL_FOUND)
  install(
    TARGETS GPUAPIOPENCL GPUAPIOPENCL_static
    ARCHIVE DESTINATION ${INSTALL_LIBDIR}
    LIBRARY DESTINATION ${INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${INSTALL_INCLUDEDIR}
    )
endif()
