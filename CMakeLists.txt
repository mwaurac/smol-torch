cmake_minimum_required(VERSION 3.15)
project(smol_torch C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  add_compile_options(-g -O0 -fno-omit-frame-pointer)
  add_link_options(-fno-omit-frame-pointer)

  include(CheckCCompilerFlag)
  check_c_compiler_flag(-fsanitize=address HAS_ASAN)
  check_c_compiler_flag(-fsanitize=undefined HAS_UBSAN)
  if(HAS_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
  else()
    message(WARNING "AddressSanitizer not supported by compiler")
  endif()
  if(HAS_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
  else()
    message(WARNING "UndefinedBehaviorSanitizer not supported by compiler")
  endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-O3 -march=native -flto)
  add_link_options(-flto)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(Python3 COMPONENTS Development.Module REQUIRED)

add_library(smol_torch_core
  smol-torch/src/tensor.c
        smol-torch/src/dtype.c
        smol-torch/src/ops.c
)

add_library(smol_torch MODULE
  smol-torch/cpython/python_tensor.c
)
target_link_libraries(smol_torch PRIVATE smol_torch_core Python3::Module)
target_include_directories(smol_torch PRIVATE ${Python3_INCLUDE_DIRS})

set_target_properties(smol_torch PROPERTIES
  PREFIX ""
  OUTPUT_NAME "smol_torch"
  SUFFIX ".${Python3_SOABI}${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

install(TARGETS smol_torch_core
  ARCHIVE DESTINATION lib
)

install(TARGETS smol_torch
  LIBRARY DESTINATION lib/python${Python3_VERSION_MAJOR}.$
  # {Python3_VERSION_MINOR}/site-packages
)

install(FILES include/tensor.h
  DESTINATION include/smol_torch
)

