cmake_minimum_required(VERSION 3.8)
project(ChapelGPUAPI)

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
  add_library(GPUAPICUDA SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.cu)
  add_library(GPUAPICUDA_static STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.cu)
  set_target_properties(GPUAPICUDA PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.h)
endif()

if(HIP_FOUND)
  find_program(
    CHPL_GPU_HIPIFY_EXECUTABLE
    NAMES hipify-perl
    PATHS
    ${HIP_ROOT_DIR}
    PATH_SUFFIXES bin
    NO_DEFAULT_PATH)
  if(CHPL_GPU_HIPIFY_EXECUTABLE)
    message(STATUS "Found HIP: " ${HIP_VERSION})
    message(STATUS "Found HIPIFY: " ${CHPL_GPU_HIPIFY_EXECUTABLE})
    add_custom_command(
      OUTPUT GPUAPI.hip.cc
      COMMAND ${CHPL_GPU_HIPIFY_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.cu > GPUAPI.hip.cc
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.cu
      COMMENT "Convering GPUAPI.cu to GPUAPI.hip.cc"
      )
    set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-gpu-rdc -fPIC")
    hip_add_library(GPUAPIHIP SHARED GPUAPI.hip.cc)
    hip_add_library(GPUAPIHIP_static STATIC GPUAPI.hip.cc)
    set_target_properties(GPUAPIHIP PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.h)
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
  add_library(GPUAPIOPENCL SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.opencl.c)
  add_library(GPUAPIOPENCL_static STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.opencl.c)
  target_link_libraries(GPUAPIOPENCL OpenCL::OpenCL)
  target_link_libraries(GPUAPIOPENCL_static OpenCL::OpenCL)
  set_target_properties(GPUAPIOPENCL PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.h)
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
      OUTPUT GPUAPI.dp.cpp
      COMMAND dpct --in-root=${CMAKE_CURRENT_SOURCE_DIR}/src/ --out-root=. --cuda-include-path=${CUDA_INCLUDE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.cu
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.cu
      COMMENT "Convering GPUAPI.cu to GPUAPI.dp.cpp"
    )
  else()
	message(STATUS "Found DPC++, but DPCT NOTFOUND")
    set(DPCPP_FOUND OFF)
  endif()
  set(CMAKE_CXX_COMPILER "dpcpp")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
  add_library(GPUAPIDPCPP SHARED GPUAPI.dp.cpp)
  add_library(GPUAPIDPCPP_static STATIC GPUAPI.dp.cpp)
  set_target_properties(GPUAPIDPCPP PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.h)
else()
	message(STATUS "DPC++ Not Found")
endif()

if(NOT (CMAKE_CUDA_COMPILER OR HIP_FOUND OR OpenCL_FOUND OR DPCPP_FOUND))
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
set(INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR} CACHE PATH "Installation directory for header files")
set(INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR} CACHE PATH "Installation directory for executables")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKEDIR CMake)
else()
  set(DEF_INSTALL_CMAKEDIR share/cmake/${PROJECT_NAME})
endif()
set(INSTALL_CMAKEDIR ${DEF_INSTALL_CMAKEDIR} CACHE PATH "Installation directory for CMake files")
set(INSTALL_MODULEDIR modules CACHE PATH "Installation directory for Chapel module files")

# Report to user
foreach(p LIB INCLUDE MODULE CMAKE)
  file(TO_NATIVE_PATH ${CMAKE_INSTALL_PREFIX}/${INSTALL_${p}DIR} _path )
  message(STATUS "${p} components will be installed to ${_path}")
  unset(_path)
endforeach()

if(CMAKE_CUDA_COMPILER)
  install(
    TARGETS GPUAPICUDA GPUAPICUDA_static
    EXPORT gpuapicuda-export
    ARCHIVE DESTINATION ${INSTALL_LIBDIR}
    LIBRARY DESTINATION ${INSTALL_LIBDIR}
    INCLUDES DESTINATION ${INSTALL_INCLUDEDIR}
    PUBLIC_HEADER DESTINATION ${INSTALL_INCLUDEDIR}
    )
  install(
    EXPORT gpuapicuda-export
    FILE gpuapicuda-config.cmake
    DESTINATION ${INSTALL_CMAKEDIR}
    EXPORT_LINK_INTERFACE_LIBRARIES
    )
endif()

if (HIP_FOUND)
  install(
    TARGETS GPUAPIHIP GPUAPIHIP_static
    EXPORT gpuapihip-export
    ARCHIVE DESTINATION ${INSTALL_LIBDIR}
    LIBRARY DESTINATION ${INSTALL_LIBDIR}
    INCLUDES DESTINATION ${INSTALL_INCLUDEDIR}
    PUBLIC_HEADER DESTINATION ${INSTALL_INCLUDEDIR}
    )
  install(
    EXPORT gpuapihip-export
    FILE gpuapihip-config.cmake
    DESTINATION ${INSTALL_CMAKEDIR}
    EXPORT_LINK_INTERFACE_LIBRARIES
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

if(DPCPP_FOUND)
  install(
    TARGETS GPUAPIDPCPP GPUAPIDPCPP_static
    EXPORT gpuapidpcpp-export
    ARCHIVE DESTINATION ${INSTALL_LIBDIR}
    LIBRARY DESTINATION ${INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${INSTALL_INCLUDEDIR}
    )
  install(
    EXPORT gpuapidpcpp-export
    FILE gpuapidpcpp-config.cmake
    DESTINATION ${INSTALL_CMAKEDIR}
    EXPORT_LINK_INTERFACE_LIBRARIES
    )
endif()

if(CMAKE_CUDA_COMPILER OR HIP_FOUND OR DPCPP_FOUND)
  install(
    FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lambda.h
    DESTINATION ${INSTALL_INCLUDEDIR}
    )
endif()

install(
  FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildGPUCode.cmake
  DESTINATION ${INSTALL_CMAKEDIR}
  )

install(
  FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUIterator.chpl
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GPUAPI.chpl
  DESTINATION ${INSTALL_MODULEDIR}
  )

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/env.sh.in env.sh @ONLY)
install(
  PROGRAMS
  ${CMAKE_CURRENT_BINARY_DIR}/env.sh
  DESTINATION ${INSTALL_BINDIR}
  )
