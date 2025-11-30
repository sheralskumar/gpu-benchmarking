# Simple GPU Benchmarking Project

A GPU benchmarking project using **OpenCL** and **C++** with **Python** automation.  
Demonstrates low-level GPU programming, kernel execution, and automation for performance testing.

## Requirements

- **C++ compiler** supporting C++17 (MSVC, GCC, Clang)
- **CMake** ≥ 3.10
- **OpenCL SDK** (AMD, NVIDIA, Intel, or generic)
- **Python** 3.x (for automation scripts)
- **Windows** 11

## Building and Running

### 1. Build the C++ Benchmark

Open a terminal in the project root. Create and navigate to the build directory:

```bash
mkdir build
cd build
```
### 2. Configure the project with CMake:

```bash 
cmake ..
```
### 3. Build the executable:
```bash
cmake --build .
```

The executable will be created in `build/` as `benchmark.exe`.

### 4. Running
#### From Terminal:
```bash
./build/benchmark [device_index] [vector_size]
```


- `device_index` (optional, default 0): GPU device to use

- `vector_size` (optional, default 2^26): Number of elements in the vectors

Example:

```bash
./build/benchmark 0 8192
```

#### From Python Automation Script:

Navigate to the automation/ folder. Run the script:

```bash
python run_benchmarks.py
```

Results saved to `automation\benchmark_results.csv`
