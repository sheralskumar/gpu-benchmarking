import subprocess
import csv
import os
import sys
from pathlib import Path
import pyopencl as cl
import matplotlib.pyplot as plt
import pandas as pd
import platform

# Define path to the benchmark executable
SCRIPT_DIR = Path(__file__).parent.parent
SAVE_DIR = SCRIPT_DIR / "automation" / "benchmark_results.csv"

if platform.system() == "Windows":
    BENCHMARK_EXECUTABLE = SCRIPT_DIR / "build" / "benchmark.exe"
    if not BENCHMARK_EXECUTABLE.exists():
        BENCHMARK_EXECUTABLE = SCRIPT_DIR / "build" / "Release" / "benchmark.exe"
else:
    BENCHMARK_EXECUTABLE = SCRIPT_DIR / "build" / "benchmark"


# Automatically detect all GPU devices
def get_gpu_devices():
    platforms = cl.get_platforms()
    devices = [dev for p in platforms for dev in p.get_devices(device_type=cl.device_type.ALL)]

    for i, dev in enumerate(devices):
        print(f"{i}: {dev.name}")
        # print(f"  Vendor: {dev.vendor}")
        # print(f"  Max Compute Units: {dev.max_compute_units}")
        # print(f"  Max Clock Frequency: {dev.max_clock_frequency} MHz")
        # print(f"  Global Memory Size: {dev.global_mem_size // (1024**2)} MB")
        # print(f"  Local Memory Size: {dev.local_mem_size // 1024} KB")

    return devices


def run_benchmark(device, N, mode=0):
    if not BENCHMARK_EXECUTABLE.exists():
        print(f"Benchmark executable not found at {BENCHMARK_EXECUTABLE}")
        return None
    try: 
        result = subprocess.run([str(BENCHMARK_EXECUTABLE), str(device), str(N), str(mode)], text=True, capture_output=True)

        if result.returncode != 0:
            print(f"Error running benchmark for device {device} with N={N}: {result.stderr}")
            return None
        if "Correct:" in result.stdout and "no" in result.stdout:
            print("Validation failed")
            return None

        for line in result.stdout.splitlines():
            if line.startswith("Time:"):
                return float(line.split()[1])
            
    except Exception as e:
        print(f"Exception occurred while running benchmark for device {device} with N={N}: {e}")
        return None
    
def plot_results():
    if not SAVE_DIR.exists():
        print("No results file. Run benchmarks first.")
        return

    df = pd.read_csv(SAVE_DIR)

    for device_index in df["Device"].unique():
        subset = df[df["Device"] == device_index]
        plt.plot(subset["N"], subset["Time (ms)"], label=f"Device {device_index}")

    plt.xlabel("N (size)")
    plt.ylabel("Time (ms)")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.title("GPU Benchmark Results")
    plt.legend()
    plt.grid(True)
    plt.savefig(SCRIPT_DIR / "automation" / "benchmark_results.png")
    plt.show()

def main(mode):
    devices = get_gpu_devices() # List of device IDs to benchmark
    N_values = [1 << 10, 1<< 15, 1 << 22, 1 << 24, 1 << 28]  # Different sizes for the benchmark

    results = []
    for N in N_values:
        for i, dev in enumerate(devices):
            print(f"Running benchmark for device {i} with N={N}...")
            output = run_benchmark(i, N, mode)
            if output is not None:
                results.append((i, N, output))
                print(f"Result: {output}")

    # Save results to CSV
    with open(SAVE_DIR, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Device", "N", "Time (ms)"])
        csvwriter.writerows(results)

    print("Benchmarking completed. Results saved to ", SAVE_DIR)

    plot_results()

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else 0
    main(mode)