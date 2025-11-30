import subprocess
import csv
import os
import sys
from pathlib import Path

# Define path to the benchmark executable
SCRIPT_DIR = Path(__file__).parent.parent
BENCHMARK_EXECUTABLE = SCRIPT_DIR / "build" / "benchmark.exe"
SCRIPT_DIR = Path(__file__).parent.parent
SAVE_DIR = SCRIPT_DIR / "automation" / "benchmark_results.csv"



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
    
def main():
    devices = [0, 1]  # List of device IDs to benchmark
    N_values = [1 << 22, 1 << 24, 1 << 26]  # Different sizes for the benchmark

    results = []
    for N in N_values:
        for device in devices:
            print(f"Running benchmark for device {device} with N={N}...")
            output = run_benchmark(device, N)
            if output:
                results.append((device, N, output))
                print(f"Result: {output}")

    # Save results to CSV
    with open(SAVE_DIR, "w", newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Device", "N", "Output"])
        csvwriter.writerows(results)

    print("Benchmarking completed. Results saved to ", SAVE_DIR)

if __name__ == "__main__":
    main()