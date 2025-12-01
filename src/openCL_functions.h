#ifndef OPEN_CL_FUNCTIONS_H
#define OPEN_CL_FUNCTIONS_H

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

void run_vector_add(cl_program program, cl_context ctx, cl_command_queue queue, int N); 

void run_vector_mul(cl_program program, cl_context ctx, cl_command_queue queue, int N);

void run_relu(cl_program program, cl_context ctx, cl_command_queue queue, int N);

void run_sigmoid(cl_program program, cl_context ctx, cl_command_queue queue, int N);

#endif