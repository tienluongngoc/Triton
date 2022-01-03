cmake_minimum_required(VERSION 2.6)

project(face_detection)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

# include_directories(${PROJECT_SOURCE_DIR}/include)
# link_directories(${PROJECT_SOURCE_DIR}/include)
if(WIN32)
enable_language(CUDA) 
endif(WIN32)

# cuda
include_directories(/usr/local/cuda-11.2/include)
link_directories(/usr/local/cuda-11.2/lib64)

# tensorrt
# include_directories(/home/nms173341/TensorRT-8.0.0.3/include)
# link_directories(/home/nms173341/TensorRT-8.0.0.3/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(decodeplugin SHARED ${PROJECT_SOURCE_DIR}/decode.cu)
target_link_libraries(decodeplugin nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})


# include_directories("/home/haobk/Mydata/Engine/server/res2tensort_server/face_detection/align")
# link_directories("/home/haobk/Mydata/Engine/server/res2tensort_server/face_detection/align")
# include_directories("/home/haobk/Mydata/Engine/server/res2tensort_server/face_detection/include")
# link_directories("/home/haobk/Mydata/Engine/server/res2tensort_server/face_detection/include")
#add_executable(retina_r50 ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/retina_r50.cpp)
#target_link_libraries(retina_r50 nvinfer)
#target_link_libraries(retina_r50 cudart)
#target_link_libraries(retina_r50 decodeplugin)
#target_link_libraries(retina_r50 ${OpenCV_LIBRARIES})

add_executable(convert_mnet ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/convert_mnet.cpp)
# add_executable(retina_mnet ${PROJECT_SOURCE_DIR}/include/calibrator.cpp ${PROJECT_SOURCE_DIR}/test.cpp)
target_link_libraries(convert_mnet nvinfer)
target_link_libraries(convert_mnet cudart)
target_link_libraries(convert_mnet decodeplugin)
target_link_libraries(convert_mnet ${OpenCV_LIBRARIES})

if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)

