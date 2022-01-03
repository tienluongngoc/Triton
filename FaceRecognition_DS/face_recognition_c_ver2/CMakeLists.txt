cmake_minimum_required(VERSION 2.6)

project(facerecognition)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)



include_directories(${PROJECT_SOURCE_DIR}/include/extraction)
link_directories(${PROJECT_SOURCE_DIR}/include/extraction)
include_directories(${PROJECT_SOURCE_DIR}/include/detection_with_mask)
link_directories(${PROJECT_SOURCE_DIR}/include/detection_with_mask)
# include_directories(${PROJECT_SOURCE_DIR}/include/detection)
# link_directories(${PROJECT_SOURCE_DIR}/include/detection)
include_directories(${PROJECT_SOURCE_DIR}/include/manager)
link_directories(${PROJECT_SOURCE_DIR}/include/manager)
include_directories(${PROJECT_SOURCE_DIR}/include/antispoofing)
link_directories(${PROJECT_SOURCE_DIR}/include/antispoofing)
include_directories(${PROJECT_SOURCE_DIR}/include/headpose_estimation)
link_directories(${PROJECT_SOURCE_DIR}/include/headpose_estimation)


if(WIN32)
enable_language(CUDA) 
endif(WIN32)

# cuda
#include_directories(/usr/local/cuda-11.2/include)
#link_directories(/usr/local/cuda-11.2/lib64)


# cuda_add_library(decodeplugin SHARED ${PROJECT_SOURCE_DIR}/include/detection/decode.cu)
cuda_add_library(decodeplugin SHARED ${PROJECT_SOURCE_DIR}/include/detection_with_mask/decode.cu)
cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/include/antispoofing/yololayer.cu)
# cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/include/extraction/prelu.cu)

target_link_libraries(decodeplugin nvinfer cudart myplugins)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(facerecognition ${PROJECT_SOURCE_DIR}/facerecognition.cpp)
add_executable(facedetection ${PROJECT_SOURCE_DIR}/face_detection.cpp )
# add_executable(facerecognition ${PROJECT_SOURCE_DIR}/test_extract.cpp)

add_executable(addface ${PROJECT_SOURCE_DIR}/add_face.cpp)
# add_executable(retina_mnet ${PROJECT_SOURCE_DIR}/include/calibrator.cpp ${PROJECT_SOURCE_DIR}/test.cpp)
target_link_libraries(facerecognition nvinfer)
target_link_libraries(facerecognition cudart)
target_link_libraries(facerecognition myplugins)
target_link_libraries(facerecognition decodeplugin)
target_link_libraries(facerecognition ${OpenCV_LIBRARIES})

target_link_libraries(facedetection nvinfer)
target_link_libraries(facedetection cudart)
target_link_libraries(facedetection myplugins)
target_link_libraries(facedetection decodeplugin)
target_link_libraries(facedetection ${OpenCV_LIBRARIES})



target_link_libraries(addface nvinfer)
target_link_libraries(addface cudart)
target_link_libraries(addface decodeplugin)
target_link_libraries(addface ${OpenCV_LIBRARIES})


#extraction 



# target_link_libraries(arcface-r100 nvinfer)
# target_link_libraries(arcface-r100 cudart)
# target_link_libraries(arcface-r100 myplugins)
# target_link_libraries(arcface-r100 ${OpenCV_LIBS})

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)

