cmake_minimum_required(VERSION 2.6)

project(deepstream-odtk)
enable_language(CXX)
#include(FindCUDA)
#project(face_detection)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(DeepStream_DIR /opt/nvidia/deepstream/deepstream-6.0) 
find_package(CUDA REQUIRED)


#set(CMAKE_CXX_STANDARD 14)
#find_package(CUDA REQUIRED)
#find_package(OpenCV REQUIRED)

if(DEFINED TensorRT_DIR)
  include_directories("${TensorRT_DIR}/include")
  link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
if(DEFINED DeepStream_DIR)
  include_directories("${DeepStream_DIR}/sources/includes")
endif(DEFINED DeepStream_DIR)
include_directories(${CUDA_INCLUDE_DIRS})


if(NOT DEFINED ARCH)
  set(ARCH "sm_70")
endif(NOT DEFINED ARCH)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

#include_directories(/usr/local/cuda/include)
#link_directories(/usr/local/cuda/lib64)

cuda_add_library(decodeplugin SHARED ${PROJECT_SOURCE_DIR}/decode.cu)
target_link_libraries(decodeplugin nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

#target_link_libraries(deepstream-odtk nvinfer)
#target_link_libraries(deepstream-odtk cudart)
#target_link_libraries(deepstream-odtk decodeplugin)
#target_link_libraries(deepstream-odtk ${OpenCV_LIBRARIES})




cuda_add_library(deepstream-odtk SHARED
  decode.h
#  decode.cu 
  nvdsparsebbox_retinanet.cpp
  OPTIONS -arch ${ARCH} -std=c++14 --expt-extended-lambda
)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(deepstream-odtk ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvonnxparser ${OpenCV_LIBS})
