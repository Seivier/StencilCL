#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
// C++ standard includes
#include <iostream>
#include <vector>

// OpenCL C++ includes
#include <CL/opencl.hpp>

int main()
{
    try
    {
        // Get all platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        std::cout << platforms.size() << " platform(s) found:\n";

        for (const auto &platform : platforms)
        {
            std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
            std::cout << "Platform: " << platformName << "\n";

            // Get all devices for the current platform
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

            for (const auto &device : devices)
            {
                std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
                std::cout << "  Device: " << deviceName << "\n";
            }
        }
    }
    catch (const cl::Error &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
    }

    return 0;
}
