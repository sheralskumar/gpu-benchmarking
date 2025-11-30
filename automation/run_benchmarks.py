import subprocess
import csv
import os
import sys
from pathlib import Path
import pyopencl as cl
import matplotlib.pyplot as plt
import pandas as pd

# Define path to the benchmark executable
SCRIPT_DIR = Path(__file__).parent.parent
BENCHMARK_EXECUTABLE = SCRIPT_DIR / "build" / "benchmark.exe"
SCRIPT_DIR = Path(__file__).parent.parent
SAVE_DIR = SCRIPT_DIR / "automation" / "benchmark_results.csv"


# Automatically detect all GPU devices
def get_gpu_devices():
    platforms = cl.get_platforms()
    devices = [dev for p in platforms for dev in p.get_devices(device_type=cl.device_type.GPU)]

    unique_devices = []
    seen_names = set()

    for dev in devices:
        if dev.name not in seen_names:
            unique_devices.append(dev)
            seen_names.add(dev.name)

    for i, dev in enumerate(unique_devices):
        print(f"{i}: {dev.name}")
        print(f"  Vendor: {dev.vendor}")
        print(f"  Max Compute Units: {dev.max_compute_units}")
        print(f"  Max Clock Frequency: {dev.max_clock_frequency} MHz")
        print(f"  Global Memory Size: {dev.global_mem_size // (1024**2)} MB")
        print(f"  Local Memory Size: {dev.local_mem_size // 1024} KB")

    return unique_devices


def run_benchmark(device, N):
    if not BENCHMARK_EXECUTABLE.exists():
        print(f"Benchmark executable not found at {BENCHMARK_EXECUTABLE}")
        return None
    try: 
        result = subprocess.run([str(BENCHMARK_EXECUTABLE), str(device), str(N)], text=True, capture_output=True)

        if result.returncode != 0:
            print(f"Error running benchmark for device {device} with N={N}: {result.stderr}")
            return None
        if "Correct: yes" not in result.stdout:
            print("Validation failed")
            return None
        for line in result.stdout.splitlines():
            if line.startswith("Time:"):
                return float(line.split()[1])
            
    except Exception as e:
        print(f"Exception occurred while running benchmark for device {device} with N={N}: {e}")
        return None
    
def plot_results():
    df = pd.read_csv(SAVE_DIR)

    for device_index in df["Device"].unique():
        subset = df[df["Device"] == device_index]
        plt.plot(subset["N"], subset["Time (ms)"], label=f"Device {device_index}")

    plt.xlabel("N (size)")
    plt.ylabel("Time (s)")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.title("GPU Benchmark Results")
    plt.legend()
    plt.grid(True)
    plt.savefig(SCRIPT_DIR / "automation" / "benchmark_results.png")
    plt.show()

def main():
    devices = get_gpu_devices() # List of device IDs to benchmark
    N_values = [1 << 22, 1 << 24, 1 << 26, 1 << 28]  # Different sizes for the benchmark

    results = []
    for N in N_values:
        for i, dev in enumerate(devices):
            print(f"Running benchmark for device {i} with N={N}...")
            output = run_benchmark(i, N)
            if output:
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
    main()