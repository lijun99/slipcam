# -*- cmake -*-
# slipcam, with framecor (C++, CUDA and CPU backends, a pybind11 module)
#
# Install with pip, in a (conda) environment with numpy, pybind11, scikit-build-core and FFTW:
#   pip install . --no-deps --no-build-isolation -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=native
#
# Options:
#   FRAMECOR_CUDA   AUTO (default, build the CUDA backend if a CUDA compiler is found), ON, OFF

cmake_minimum_required(VERSION 3.18)

project(slipcam VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FRAMECOR_CUDA AUTO CACHE STRING "Build the CUDA backend (AUTO, ON, OFF)")
set_property(CACHE FRAMECOR_CUDA PROPERTY STRINGS AUTO ON OFF)

# CUDA backend
if(FRAMECOR_CUDA STREQUAL "AUTO")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(FRAMECOR_WITH_CUDA ON)
    else()
        set(FRAMECOR_WITH_CUDA OFF)
    endif()
elseif(FRAMECOR_CUDA)
    set(FRAMECOR_WITH_CUDA ON)
else()
    set(FRAMECOR_WITH_CUDA OFF)
endif()

if(FRAMECOR_WITH_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
            set(CMAKE_CUDA_ARCHITECTURES all-major)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
        endif()
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    find_package(CUDAToolkit 11 COMPONENTS cudart cufft REQUIRED)
    message(STATUS "Building the CUDA backend with CUDA ${CUDAToolkit_VERSION}")
else()
    message(STATUS "Not building the CUDA backend")
endif()

# CPU backend
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(FFTW REQUIRED COMPONENTS Float)
find_package(OpenMP COMPONENTS CXX)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP is not found; the CPU backend will use a single thread")
endif()

# python and pybind11; the python of the active (e.g., conda) environment first
set(Python_FIND_UNVERSIONED_NAMES FIRST)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

set(FRAMECOR_SOURCES cxx/common.cpp cxx/cpu.cpp cxx/bindings.cpp)
if(FRAMECOR_WITH_CUDA)
    list(APPEND FRAMECOR_SOURCES cxx/cuda.cu)
endif()

pybind11_add_module(_framecor ${FRAMECOR_SOURCES})
target_include_directories(_framecor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cxx)
target_compile_definitions(_framecor PRIVATE FRAMECOR_VERSION="${PROJECT_VERSION}")
target_link_libraries(_framecor PRIVATE FFTW::Float)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_framecor PRIVATE OpenMP::OpenMP_CXX)
endif()
if(FRAMECOR_WITH_CUDA)
    target_compile_definitions(_framecor PRIVATE FRAMECOR_WITH_CUDA)
    target_link_libraries(_framecor PRIVATE CUDA::cudart CUDA::cufft)
    set_target_properties(_framecor PROPERTIES CUDA_VISIBILITY_PRESET hidden)
endif()

install(TARGETS _framecor LIBRARY DESTINATION slipcam/framecor)

# version file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/_version.py "__version__ = \"${PROJECT_VERSION}\"\n")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_version.py DESTINATION slipcam)
