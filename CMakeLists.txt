cmake_minimum_required(VERSION 3.20)

project(Stencil LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

find_package(OpenCL REQUIRED)

add_executable(main main.cc)
target_link_libraries(main PRIVATE ${OpenCL_LIBRARY})
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR}/include)

