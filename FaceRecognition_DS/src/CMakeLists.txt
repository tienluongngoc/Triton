set(LIBRARY_NAME deep_stream)
set(DEEP_STREAM_LIBS /opt/nvidia/deepstream/deepstream-6.0/lib)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lstdc++ -lsupc++ -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
find_package (Threads)
find_package(PkgConfig) 
pkg_search_module(GLIB REQUIRED glib-2.0)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
list(APPEND INCLUDE_DIRS ${GLIB_INCLUDE_DIRS}${GSTREAMER_INCLUDE_DIRS})
list(APPEND DEPS ${GLIB_LIBRARY_DIRS}${GSTREAMER_LIBRARY_DIRS})


find_package(yaml-cpp REQUIRED )


list(APPEND DEPS "${DEEP_STREAM_LIBS}/libnvdsgst_meta.so")
list(APPEND DEPS "${DEEP_STREAM_LIBS}/libnvds_meta.so")
list(APPEND DEPS "${DEEP_STREAM_LIBS}/libnvbufsurface.so")
list(APPEND DEPS "${DEEP_STREAM_LIBS}/libnvds_msgconv.so")
list(APPEND DEPS "${DEEP_STREAM_LIBS}/libnvds_msgbroker.so")
list(APPEND DEPS "${DEEP_STREAM_LIBS}/libnvds_msgconv_audio.so")



find_package(yaml-cpp REQUIRED )

list(APPEND INCLUDE_DIRS "/opt/nvidia/deepstream/deepstream/sources/includes")


file(GLOB_RECURSE HEADER_LIST "${PROJECT_SOURCE_DIR}/include/*[.hpp, .h]")
file(GLOB_RECURSE SOURCE_LIST "${PROJECT_SOURCE_DIR}/src/*.cpp")
list(APPEND INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include/")



find_package(CUDA REQUIRED)
message(STATUS "Find CUDA include at ${CUDA_INCLUDE_DIRS}")
message(STATUS "Find CUDA libraries: ${CUDA_LIBRARIES}")

set(YAML_INCLUDE "${PROJECT_SOURCE_DIR}/includes/yaml-cpp/include")
set(YAML_LIB_DIR "${PROJECT_SOURCE_DIR}/includes/yaml-cpp/libs")






# OPENCV
find_package(OpenCV REQUIRED core imgcodecs imgproc video videoio dnn calib3d highgui cudaimgproc)
list(APPEND INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS})
list(APPEND DEPS opencv_core opencv_highgui opencv_imgcodecs opencv_cudaimgproc opencv_imgproc opencv_video opencv_videoio  opencv_calib3d)




add_library(${LIBRARY_NAME} ${SOURCE_LIST} ${HEADER_LIST} ${yaml_LIST})
target_include_directories(${LIBRARY_NAME}  PRIVATE ${GLIB_INCLUDE_DIRS}  )
target_link_libraries(${LIBRARY_NAME}  INTERFACE ${GLIB_LDFLAGS})
target_compile_features(${LIBRARY_NAME} PUBLIC cxx_std_17)
target_include_directories(${LIBRARY_NAME} PUBLIC ${INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(${LIBRARY_NAME} PUBLIC ${GSTREAMER_LIBRARIES} ${DEPS} ${CUDA_LIBRARIES})

target_include_directories(${LIBRARY_NAME} PRIVATE ${GST_INCLUDE_DIRS})
target_link_libraries(${LIBRARY_NAME} PUBLIC ${GST_LIBRARIES} ${TENSORRT_LIBRARY} yaml-cpp)


