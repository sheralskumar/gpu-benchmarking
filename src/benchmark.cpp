
#include <CL/cl.h>  // library for OpenCL
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <cmath>


// error checking macro for OpenCL calls to determine whether they were successful
// CL_SUCCESS is defined in cl.h as 0
#define CHECK(x) if((x) != CL_SUCCESS){std::cerr << "OpenCL error " << x << std::endl; exit(1);}

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

void run_vector_add(cl_program program, cl_context ctx, cl_command_queue queue, int N) {
    cl_int err;
    // 5. Create kernel object from the built program, extract it by its name
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    CHECK(err);

    // 6. Create device buffers and transfer input data to the device
    // buffers are memory allocations on the device that can be read from and written to by kernels
    // openCL defines three memory types: buffer, image, and pipe
    // buffer stores contiguous linear data (like arrays/vectors/matrices) and can be accessed on the device using pointers, created using clCreateBuffer
    // image is optimized for 2D/3D data and provides built-in functions for sampling and filtering (used in graphics/vision applications)
    // pipe is used for streaming data between kernels
    // here we use buffer memory type to create buffers for our input/output vectors
    
    // make three buffers: two for input vectors and one for output vector, fill the input buffers with data (host)
    size_t bytes = N*sizeof(float);
    std::vector<float> A(N), B(N), C(N);
    for(int i=0;i<N;i++){
        A[i] = i;
        B[i] = 2*i;
    }

    // create device buffers and copy input data from host to device
    // note bitwise or operator (|) is used to combine multiple memory flags
    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, A.data(), &err); // context, flags, size in bytes, host pointer (data to copy), error code
    CHECK(err);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, B.data(), &err);
    CHECK(err);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    CHECK(err);

    // 7. Set kernel arguments as the buffers created above
    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA)); // kernel, argument index, size of argument, pointer to argument value
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC));
    CHECK(clSetKernelArg(kernel, 3, sizeof(int), &N));

    // 8. Kernel execution
    // specify the global and local work sizes. Global work size is total number of work items (threads) to execute the kernel
    // local work size is number of work items in a work group (like CUDA blocks), just a way to organize work items for better performance
    // 256 is a common choice for local work size
    size_t global = N;
    size_t local = 256;

    auto start = std::chrono::high_resolution_clock::now(); // start timer for benchmarking
    CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL)); // command queue, kernel, work dimension, global work offset, global work size, local work size, num events in wait list, event wait list, event
    clFinish(queue); // wait for all commands in the queue to finish
    auto end = std::chrono::high_resolution_clock::now(); // end timer

    double ms = std::chrono::duration<double, std::milli>(end-start).count();
    std::cout << "Time: " << ms << " ms\n"; // print elapsed time

    // read back the result from device to host, queue, buffer, blocking read, offset, size, host pointer, num events in wait list, event wait list, event
    CHECK(clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, bytes, C.data(), 0, NULL, NULL));

    // 9. Verify the result
    bool ok = true;
    for (int i=0;i<10;i++) {
        if (C[i] != A[i] + B[i]) { ok = false;    }
    }
    std::cout << "Correct: " << (ok ? "yes" : "no") << "\n";

    // 10. Cleanup OpenCL resources
    // release device buffers
    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);

    // release kernel, program, queue, and context
    clReleaseKernel(kernel);
}


void run_vector_mul(cl_program program, cl_context ctx, cl_command_queue queue, int N) {
    cl_int err;
    // 5. Create kernel object from the built program, extract it by its name
    cl_kernel kernel = clCreateKernel(program, "vector_mul", &err);
    CHECK(err);

    // 6. Create device buffers and transfer input data to the device
    size_t bytes = N*sizeof(float);
    std::vector<float> A(N), B(N), C(N);
    for(int i=0;i<N;i++){
        A[i] = i;
        B[i] = 2*i;
    }

    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, A.data(), &err); // context, flags, size in bytes, host pointer (data to copy), error code
    CHECK(err);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, B.data(), &err);
    CHECK(err);
    cl_mem dC = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
    CHECK(err);

    // 7. Set kernel arguments as the buffers created above
    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA)); // kernel, argument index, size of argument, pointer to argument value
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB));
    CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC));
    CHECK(clSetKernelArg(kernel, 3, sizeof(int), &N));

    // 8. Kernel execution
    size_t global = N;
    size_t local = 256;

    auto start = std::chrono::high_resolution_clock::now(); // start timer for benchmarking
    CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL)); // command queue, kernel, work dimension, global work offset, global work size, local work size, num events in wait list, event wait list, event
    clFinish(queue); // wait for all commands in the queue to finish
    auto end = std::chrono::high_resolution_clock::now(); // end timer

    double ms = std::chrono::duration<double, std::milli>(end-start).count();
    std::cout << "Time: " << ms << " ms\n"; // print elapsed time

    CHECK(clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, bytes, C.data(), 0, NULL, NULL));

    // 9. Verify the result
    bool ok = true;
    float eps = 1e-5f;
    for (int i=0;i<10;i++) {
        if (fabsf(C[i] - A[i]*B[i]) > eps) { ok = false;    }
    }
    std::cout << "Correct: " << (ok ? "yes" : "no") << "\n";

    // 10. Cleanup OpenCL resources
    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
    clReleaseKernel(kernel);
}

void run_relu(cl_program program, cl_context ctx, cl_command_queue queue, int N) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "relu", &err); 
    CHECK(err);

    size_t bytes = N*sizeof(float);
    std::vector<float> A(N), B(N);
    for(size_t i=0;i<N;i++){ A[i]=i-500; } // some negative numbers

    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, A.data(), &err); 
    CHECK(err);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err); 
    CHECK(err);
    
    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA)); // kernel, argument index, size of argument, pointer to argument value
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB));
    CHECK(clSetKernelArg(kernel, 2, sizeof(int), &N));

    // 8. Kernel execution
    size_t global = N;
    size_t local = 256;

    auto start = std::chrono::high_resolution_clock::now(); // start timer for benchmarking
    CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL)); // command queue, kernel, work dimension, global work offset, global work size, local work size, num events in wait list, event wait list, event
    clFinish(queue); // wait for all commands in the queue to finish
    auto end = std::chrono::high_resolution_clock::now(); // end timer

    double ms = std::chrono::duration<double, std::milli>(end-start).count();
    std::cout << "Time: " << ms << " ms\n"; // print elapsed time

    CHECK(clEnqueueReadBuffer(queue, dB, CL_TRUE, 0, bytes, B.data(), 0, NULL, NULL));

    // 9. Verify the result
    bool ok = true;
    float eps = 1e-5f;
    for (int i = 0; i < 10; i++) {  // just check first few for quick sanity
        if (fabsf(B[i] - std::fmaxf(0.0f, A[i])) > eps) {
            ok = false;
            break;
        }
    }
    std::cout << "Correct: " << (ok ? "yes" : "no") << "\n";

    // 10. Cleanup OpenCL resources
    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseKernel(kernel);
}

void run_sigmoid(cl_program program, cl_context ctx, cl_command_queue queue, int N) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, "sigmoid", &err); 
    CHECK(err);

    size_t bytes = N*sizeof(float);
    std::vector<float> A(N), B(N);
    for(size_t i=0;i<N;i++){ A[i]=i-500; } // some negative numbers

    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, A.data(), &err); 
    CHECK(err);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, &err); 
    CHECK(err);
    
    CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA)); // kernel, argument index, size of argument, pointer to argument value
    CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB));
    CHECK(clSetKernelArg(kernel, 2, sizeof(int), &N));

    // 8. Kernel execution
    size_t global = N;
    size_t local = 256;

    auto start = std::chrono::high_resolution_clock::now(); // start timer for benchmarking
    CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL)); // command queue, kernel, work dimension, global work offset, global work size, local work size, num events in wait list, event wait list, event
    clFinish(queue); // wait for all commands in the queue to finish
    auto end = std::chrono::high_resolution_clock::now(); // end timer

    double ms = std::chrono::duration<double, std::milli>(end-start).count();
    std::cout << "Time: " << ms << " ms\n"; // print elapsed time

    CHECK(clEnqueueReadBuffer(queue, dB, CL_TRUE, 0, bytes, B.data(), 0, NULL, NULL));

    // 9. Verify the result
    bool ok = true;
    float eps = 1e-5f;
    for (int i = 0; i < 10; i++) {  // just check first few for quick sanity
        if (fabsf(B[i] - 1.0f / (1.0f + expf(-A[i]))) > eps) {
            ok = false;
            break;
        }
    }
    std::cout << "Correct: " << (ok ? "yes" : "no") << "\n";

    // 10. Cleanup OpenCL resources
    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseKernel(kernel);
}


// main function 
int main(int argc, char** argv) {
    // read in command line arguments for device index and vector size
    int device_index = (argc > 1) ? atoi(argv[1]) : 0;
    int N = (argc > 2) ? atoi(argv[2]) : (1 << 26);
    int mode = (argc > 3) ? atoi(argv[3]) : 0;

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
        err = clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, NULL, &dev_count);
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
    std::filesystem::path exe_path = std::filesystem::absolute(argv[0]).parent_path();
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