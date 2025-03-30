cmake_minimum_required(VERSION 3.10) # 3.1 << C_STANDARD 11

project(HelloOpenCL LANGUAGES CXX)

find_package(OpenCL CONFIG REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::HeadersCpp OpenCL::Headers OpenCL::OpenCL)

