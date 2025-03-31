#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#define N 16
#define BLOCK_SIZE 8
#define RADIUS 3

using std::chrono::microseconds;

int main()
{
  try
  {
    const int size = N * sizeof(int);
    std::vector<int> in(N), out(N);

    std::cout << "INFO:" << std::endl;
    // Query for platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::cout << "Platform: " << platforms.front().getInfo<CL_PLATFORM_NAME>()
              << std::endl;

    // Get a list of devices on this platform
    std::vector<cl::Device> devices;
    // Select the platform.
    platforms.front().getDevices(CL_DEVICE_TYPE_GPU, &devices);
    std::cout << "Device: " << devices.front().getInfo<CL_DEVICE_NAME>()
              << std::endl
              << std::endl;

    // Create a context
    cl::Context context(devices);

    // Create a command queue
    // Select the device.
    cl::CommandQueue queue(context, devices.front());

    // Create the memory buffers
    cl::Buffer inBuff(context, CL_MEM_READ_WRITE, size);
    cl::Buffer outBuff(context, CL_MEM_READ_WRITE, size);

    // Assign values to host variables
    auto t_start = std::chrono::high_resolution_clock::now();
    for (auto &id : in)
      id = 1;
    auto t_end = std::chrono::high_resolution_clock::now();
    auto t_create_data =
        std::chrono::duration_cast<microseconds>(t_end - t_start).count();

    // Copy values from host variables to device
    t_start = std::chrono::high_resolution_clock::now();
    // usar CL_FALSE para hacerlo asíncrono
    queue.enqueueWriteBuffer(inBuff, CL_TRUE, 0, size, in.data());
    t_end = std::chrono::high_resolution_clock::now();
    auto t_copy_to_device =
        std::chrono::duration_cast<microseconds>(t_end - t_start).count();

    // Read the program source
    std::ifstream sourceFile("stencil.cl");
    std::stringstream sourceCode;
    sourceCode << sourceFile.rdbuf();

    // Make and build program from the source code
    cl::Program program(context, sourceCode.str(), true);

    // Make kernel
    cl::Kernel stencil(program, "stencil_1d");

    // Set the kernel arguments
    stencil.setArg(0, inBuff);
    stencil.setArg(1, outBuff);

    // Execute the function on the device (using 32 threads here)
    cl::NDRange globalSize(N);
    cl::NDRange localSize(BLOCK_SIZE);

    t_start = std::chrono::high_resolution_clock::now();
    cl::Event event;
    queue.enqueueNDRangeKernel(stencil, cl::NullRange, globalSize, localSize,
                               nullptr, &event);
    event.wait();
    t_end = std::chrono::high_resolution_clock::now();
    auto t_kernel =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
            .count();

    // Copy the output variable from device to host
    t_start = std::chrono::high_resolution_clock::now();
    queue.enqueueReadBuffer(outBuff, CL_TRUE, 0, size, out.data());
    t_end = std::chrono::high_resolution_clock::now();
    auto t_copy_to_host =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
            .count();

    // Print the result
    std::cout << "RESULTS: " << std::endl;
    for (int i = 0; i < N; i++)
      std::cout << "  out[" << i << "]: " << out[i] << "\n";

    std::cout << "Time to create data: " << t_create_data << " microseconds\n";
    std::cout << "Time to copy data to device: " << t_copy_to_device
              << " microseconds\n";
    std::cout << "Time to execute kernel: " << t_kernel << " microseconds\n";
    std::cout << "Time to copy data to host: " << t_copy_to_host
              << " microseconds\n";
    std::cout << "Time to execute the whole program: "
              << t_create_data + t_copy_to_device + t_kernel + t_copy_to_host
              << " microseconds\n";
  }
  catch (cl::Error err)
  {
    std::cerr << "Error (" << err.err() << "): " << err.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
