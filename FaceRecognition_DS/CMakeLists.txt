cmake_minimum_required(VERSION 3.10.2)
project(DEEPSTREAM VERSION 0.1.0)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_subdirectory(src)
add_subdirectory(apps)

set(YAML_INCLUDE include/yaml-cpp/include)
set(YAML_LIB_DIR include/yaml-cpp/libs)

include_directories(${YAML_INCLUDE})
link_directories(${YAML_LIB_DIR})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)