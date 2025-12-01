#include <thread>  
#include <omp.h>    // library for openMP
#include <iostream>
#include <vector>
#include <cmath>


void add_vectors(int N) {
    std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N);
    #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

void multiply_vectors(int N) {
    std::vector<float> a(N, 1.0f), b(N, 2.0f), c(N);
    #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        c[i] = a[i] * b[i];
    }
}

void relu_activation(int N) {
    std::vector<float> a(N), b(N);
    for(int i = 0; i < N; i++) {
        a[i] = i - N/2; // some negative and positive values
    }
    #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        b[i] = std::fmax(0.0f, a[i]);
    }
}

void sigmoid_activation(int N) {
    std::vector<float> a(N), b(N);
    for(int i = 0; i < N; i++) {
        a[i] = i - N/2; // some negative and positive values
    }
    #pragma omp parallel for
    for(int i = 0; i < N; i++) {
        b[i] = 1.0f / (1.0f + std::exp(-a[i]));
    }
}

int cpu_benchmark(int N, int mode) {
    // Number of OpenMP threads
    int num_threads = omp_get_max_threads();

    // Optional: get CPU info (number of logical cores)
    unsigned int hw_threads = std::thread::hardware_concurrency();

    std::cout << "Running on CPU with " << num_threads 
              << " OpenMP threads (logical cores available: " 
              << hw_threads << ")\n";

    double start = omp_get_wtime();

    if (mode == 0) { add_vectors(N); }
    else if (mode == 1) { multiply_vectors(N); }
    else if (mode == 2) { relu_activation(N); }
    else if (mode == 3) { sigmoid_activation(N); }
    else {
        std::cerr << "Unsupported mode for OpenMP benchmark.\n";
        return -1;
    }

    double end = omp_get_wtime();
    std::cout << "OpenMP CPU time: " << (end - start) * 1000 << " ms\n";
    return 0;
}