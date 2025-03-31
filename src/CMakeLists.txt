find_package(OpenCL CONFIG REQUIRED)
# Copy all kernels to the build directory
file(GLOB_RECURSE KERNELS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cl)
foreach(KERNEL IN LISTS KERNELS)
    file(COPY ${KERNEL} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})
endforeach()


foreach(EXAMPLE IN ITEMS Stencil1d VectorAdd)
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} PRIVATE OpenCL::HeadersCpp OpenCL::Headers OpenCL::OpenCL)
endforeach()


