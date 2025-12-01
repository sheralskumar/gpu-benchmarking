#include <CL/cl.h>  // library for OpenCL
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <cmath>
#include "openMP_functions.h"


// error checking macro for OpenCL calls to determine whether they were successful
// CL_SUCCESS is defined in cl.h as 0
#define CHECK(x) if((x) != CL_SUCCESS){std::cerr << "OpenCL error " << x << std::endl; exit(1);}


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