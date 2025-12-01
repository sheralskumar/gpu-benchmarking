
#include <CL/cl.h>  // library for OpenCL
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <cmath>
#include "openMP_functions.h"
#include "openCL_functions.h"

// function to load the contents of a file into a string
// used to read OpenCL kernel source code from a file
std::string load_file(const std::filesystem::path& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
        exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int openCL_run(int device_index, int N, int mode, std::string parent_path) {
    // 1. Select openCL platform using the clGetPlatformIDs function
    // note platform vs device: platform is the vendor (e.g., NVIDIA, AMD, Intel), device is the actual hardware (e.g., GPU, CPU)
    cl_uint num_platforms;  // cl_uint is OpenCL-defined (I think unsigned int32) to make sure code is platform independent

    // when num_entries is 0 and platforms is NULL, clGetPlatformIDs returns the number of platforms available
    CHECK(clGetPlatformIDs(0, NULL, &num_platforms));
    // when num_entries is non-zero and platforms is not NULL, clGetPlatformIDs fills the platforms array with platform IDs
    std::vector<cl_platform_id> platforms(num_platforms);
    CHECK(clGetPlatformIDs(num_platforms, platforms.data(), NULL));

    // 2. Now select a device on the platform using clGetDeviceIDs
    std::vector<cl_device_id> devices;
    for (auto p : platforms) {
        cl_uint dev_count; // store number of devices on the platform
        cl_uint err;

        // get number of devices on a platform p, similar to the way we got number of platforms
        // CL_DEVICE_TYPE_ALL specifies we want all types of devices (CPU, GPU, etc.)
        // CL_DEVICE_TYPE_GPU could be used to select only GPU devices
        err = clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, NULL, &dev_count);
        if (err == CL_DEVICE_NOT_FOUND) { continue; } // skip platform with no devices
        CHECK(err);

        // we can now get the device IDs for platform p, similar to how we got platform IDs
        std::vector<cl_device_id> devs(dev_count);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, dev_count, devs.data(), NULL);
        devices.insert(devices.end(), devs.begin(), devs.end());    // append this platform's devices to the overall devices list
    }

    // error check to see if device index requested by the user is valid
    if (device_index >= devices.size()) {
    std::cerr << "Device index out of range.\n";
    return 1;
    }

    // grab the device ID for the requested device
    cl_device_id dev = devices[device_index];

    // get the name of the device and print it out
    char name[128];
    clGetDeviceInfo(dev, CL_DEVICE_NAME, 128, name, NULL);
    std::cout << "Using device: " << name << std::endl;

    // 3. Create an OpenCL context and command queue on that device
    // the context is like a container for all OpenCL objects (buffers, programs, kernels, etc.) associated with the device
    // the queue is used to submit work (kernel executions, memory transfers, etc.) to the device
    // operations must be done through the command queue
    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err); // properties, num_devices, array of device ids (dev), call back for errors, user data for call back, error code
    CHECK(err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(ctx, dev, 0, &err); // context, device, properties, error code
    CHECK(err);

    // 4. Create and build the OpenCL program from source
    // need to know about program objects and kernel objects
    // program object is container for OpenCL code (kernels) that will be executed on the device (can contain multiple kernels)
    // kernel objects created/managed by program object represent individual functions (kernels) in the OpenCL code that can be executed on the device
    // for ex. an algebraic program might have separate kernels for vector addition, matrix multiplication, etc. in a single program object
    // to create a kernel, we need to prepare source code in OpenCL C (similar to C99) and load it into a string

    // std::string src = load_file("../src/kernel.cl"); // load file wth the C source code as a string
    std::filesystem::path exe_path = std::filesystem::absolute(parent_path).parent_path();
    std::filesystem::path kernel_path = exe_path.parent_path() / "src" / "kernel.cl";
    std::string src = load_file(kernel_path);

    const char* csrc = src.c_str(); // convert it to C-style character array
    size_t len = src.size();
    // create the program object from the source code string
    cl_program program = clCreateProgramWithSource(ctx, 1, &csrc, &len, &err); // context, number of strings, array of strings, array of string lengths, error code
    CHECK(err);

    // build the program object
    err = clBuildProgram(program, 1, &dev, NULL, NULL, NULL); // program, num_devices, array of device ids, build options, call back, user data
    if (err != CL_SUCCESS) { // if that fails, get and print the build log
        size_t log_size;
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size, &log[0], NULL);
        std::cerr << "Build error:\n" << log << std::endl;
        return 1;
    }

    // run the vector addition benchmark
    if (mode == 0) { run_vector_add(program, ctx, queue, N); }
    else if (mode == 1) { run_vector_mul(program, ctx, queue, N); }
    else if (mode == 2) { run_relu(program, ctx, queue, N); }
    else if (mode == 3) { run_sigmoid(program, ctx, queue, N); }

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    return 0;
}

// main function 
int main(int argc, char** argv) {
    // read in command line arguments for device index and vector size
    int device_index = (argc > 1) ? atoi(argv[1]) : 0;
    int N = (argc > 2) ? atoi(argv[2]) : (1 << 26);
    int framework_mode = (argc > 3) ? atoi(argv[3]) : 0;
    int mode = (argc > 4) ? atoi(argv[4]) : 0;

    if (framework_mode == 0) {
        // run OpenCL benchmark
        return openCL_run(device_index, N, mode, argv[0]);
    }
    else if (framework_mode == 1)
    {
        return cpu_benchmark(N, mode);
    }
    
    else {
        std::cerr << "Unsupported framework mode.\n";
        return 1;
    }

}