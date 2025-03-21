"""
Improved benchmark for BetterFFTW vs NumPy FFT performance.

This script better demonstrates the strengths of FFTW by:
1. Using more repetitions to amortize planning costs
2. Separating first-time planning costs from execution
3. Testing both power-of-2 sizes and non-power-of-2 sizes
"""

# Add the src directory to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import time
import matplotlib.pyplot as plt

# Test both power-of-2 sizes and non-power-of-2 sizes
POW2_SIZES = [256, 512, 1024, 2048, 4096, 8192]
NONPOW2_SIZES = [384, 768, 1536, 3072, 6144]  # Non-power-of-2 sizes where FFTW should excel

# Add larger sizes for long-running tests
LARGE_POW2_SIZE = 4096  # 4K x 4K array 
LARGE_NONPOW2_SIZE = 6144  # ~6K x 6K array

# Add 3D sizes that are much more computationally intensive
POW2_3D_SIZE = 256  # 256^3 is very computationally intensive
NONPOW2_3D_SIZE = 320  # 320^3 is even more intensive with non-power-of-2

def benchmark_numpy(sizes, repeats=20):
    """Benchmark NumPy's native FFT implementation."""
    results = {}
    
    for size in sizes:
        # Create random data
        data = np.random.random((size, size)).astype(np.float64)
        
        # Warm-up run (not timed)
        _ = np.fft.fft2(data)
        
        # Measure time for FFT
        start = time.time()
        for _ in range(repeats):
            spectrum = np.fft.fft2(data)
            # Force computation by accessing a value
            _ = spectrum[0, 0]
        end = time.time()
        
        avg_time = (end - start) / repeats
        results[size] = avg_time
        print(f"NumPy FFT: {size}x{size} array: {avg_time:.6f} seconds")
    
    return results

def benchmark_betterfftw(sizes, repeats=20):
    """Benchmark BetterFFTW implementation."""
    import betterfftw
    
    # Configure BetterFFTW for best performance
    betterfftw.set_planner_effort('FFTW_ESTIMATE')  # Start with fast planning
    
    # Make BetterFFTW the default FFT implementation
    betterfftw.use_as_default(register_scipy=False)
    
    results_first_call = {}
    results_subsequent = {}
    
    for size in sizes:
        # Create random data
        data = np.random.random((size, size)).astype(np.float64)
        
        # Clear cache to ensure fair comparison for first run
        betterfftw.clear_cache()
        
        # Measure first call (includes planning)
        start = time.time()
        spectrum = np.fft.fft2(data)
        first_call_time = time.time() - start
        results_first_call[size] = first_call_time
        
        # Now measure subsequent calls
        start = time.time()
        for _ in range(repeats):
            spectrum = np.fft.fft2(data)
            # Force computation by accessing a value
            _ = spectrum[0, 0]
        end = time.time()
        
        avg_time = (end - start) / repeats
        results_subsequent[size] = avg_time
        print(f"BetterFFTW: {size}x{size} array: first call {results_first_call[size]:.6f}s, subsequent {avg_time:.6f}s")
    
    # Restore the original NumPy implementation
    betterfftw.restore_default(unregister_scipy=False)
    
    return results_first_call, results_subsequent

def benchmark_3d_numpy(size, repeats=3):
    """Benchmark NumPy's native FFT implementation on 3D arrays."""
    print(f"\nBenchmarking NumPy FFT on 3D array of size {size}x{size}x{size}...")
    
    # Create random data
    data = np.random.random((size, size, size)).astype(np.float64)
    
    # Warm-up run (not timed)
    _ = np.fft.fftn(data)
    
    # Measure time for FFT
    start = time.time()
    for _ in range(repeats):
        spectrum = np.fft.fftn(data)
        # Force computation by accessing a value
        _ = spectrum[0, 0, 0]
    end = time.time()
    
    avg_time = (end - start) / repeats
    print(f"NumPy 3D FFT: {size}x{size}x{size} array: {avg_time:.6f} seconds")
    
    return avg_time

def benchmark_3d_betterfftw(size, repeats=3):
    """Benchmark BetterFFTW implementation on 3D arrays."""
    import betterfftw
    
    print(f"\nBenchmarking BetterFFTW on 3D array of size {size}x{size}x{size}...")
    
    # Configure BetterFFTW for best performance
    betterfftw.set_planner_effort('FFTW_ESTIMATE')  # Start with fast planning
    
    # Make BetterFFTW the default FFT implementation
    betterfftw.use_as_default(register_scipy=False)
    
    # Create random data
    data = np.random.random((size, size, size)).astype(np.float64)
    
    # Clear cache to ensure fair comparison for first run
    betterfftw.clear_cache()
    
    # Measure first call (includes planning)
    start = time.time()
    spectrum = np.fft.fftn(data)
    first_call_time = time.time() - start
    
    # Now measure subsequent calls
    start = time.time()
    for _ in range(repeats):
        spectrum = np.fft.fftn(data)
        # Force computation by accessing a value
        _ = spectrum[0, 0, 0]
    end = time.time()
    
    avg_time = (end - start) / repeats
    print(f"BetterFFTW 3D FFT: {size}x{size}x{size} array: first call {first_call_time:.6f}s, subsequent {avg_time:.6f}s")
    
    # Restore the original NumPy implementation
    betterfftw.restore_default(unregister_scipy=False)
    
    return first_call_time, avg_time

def benchmark_large_2d(repeats=3):
    """Benchmark large 2D arrays that take several seconds."""
    print("\n--- Large 2D Arrays (Several Seconds Each) ---")
    
    results = {}
    
    # Power of 2 size
    print(f"\nTesting large power-of-2 array: {LARGE_POW2_SIZE}x{LARGE_POW2_SIZE}")
    print("Running NumPy native FFT...")
    data = np.random.random((LARGE_POW2_SIZE, LARGE_POW2_SIZE)).astype(np.float64)
    
    # NumPy benchmark
    start = time.time()
    for _ in range(repeats):
        spectrum = np.fft.fft2(data)
        _ = spectrum[0, 0]
    numpy_time = (time.time() - start) / repeats
    print(f"NumPy FFT: {LARGE_POW2_SIZE}x{LARGE_POW2_SIZE} array: {numpy_time:.6f} seconds")
    
    # BetterFFTW benchmark
    import betterfftw
    print("\nRunning BetterFFTW...")
    betterfftw.set_planner_effort('FFTW_ESTIMATE')
    betterfftw.use_as_default(register_scipy=False)
    
    betterfftw.clear_cache()
    start = time.time()
    spectrum = np.fft.fft2(data)
    first_call_time = time.time() - start
    
    start = time.time()
    for _ in range(repeats):
        spectrum = np.fft.fft2(data)
        _ = spectrum[0, 0]
    bfft_time = (time.time() - start) / repeats
    
    print(f"BetterFFTW: {LARGE_POW2_SIZE}x{LARGE_POW2_SIZE} array: first call {first_call_time:.6f}s, subsequent {bfft_time:.6f}s")
    speedup = numpy_time / bfft_time
    print(f"Speedup: {speedup:.2f}x")
    
    results['pow2'] = {'numpy': numpy_time, 'betterfftw': bfft_time, 'speedup': speedup}
    
    # Non-power of 2 size
    print(f"\nTesting large non-power-of-2 array: {LARGE_NONPOW2_SIZE}x{LARGE_NONPOW2_SIZE}")
    print("Running NumPy native FFT...")
    data = np.random.random((LARGE_NONPOW2_SIZE, LARGE_NONPOW2_SIZE)).astype(np.float64)
    
    # NumPy benchmark
    start = time.time()
    for _ in range(repeats):
        spectrum = np.fft.fft2(data)
        _ = spectrum[0, 0]
    numpy_time = (time.time() - start) / repeats
    print(f"NumPy FFT: {LARGE_NONPOW2_SIZE}x{LARGE_NONPOW2_SIZE} array: {numpy_time:.6f} seconds")
    
    # BetterFFTW benchmark
    print("\nRunning BetterFFTW...")
    
    betterfftw.clear_cache()
    start = time.time()
    spectrum = np.fft.fft2(data)
    first_call_time = time.time() - start
    
    start = time.time()
    for _ in range(repeats):
        spectrum = np.fft.fft2(data)
        _ = spectrum[0, 0]
    bfft_time = (time.time() - start) / repeats
    
    print(f"BetterFFTW: {LARGE_NONPOW2_SIZE}x{LARGE_NONPOW2_SIZE} array: first call {first_call_time:.6f}s, subsequent {bfft_time:.6f}s")
    speedup = numpy_time / bfft_time
    print(f"Speedup: {speedup:.2f}x")
    
    betterfftw.restore_default(unregister_scipy=False)
    
    results['nonpow2'] = {'numpy': numpy_time, 'betterfftw': bfft_time, 'speedup': speedup}
    
    return results

def plot_comparison(sizes, numpy_times, betterfftw_times, title_suffix=""):
    """Plot performance comparison between NumPy and BetterFFTW."""
    plt.figure(figsize=(10, 6))
    
    # Convert to ms for better readability
    numpy_ms = [t * 1000 for t in numpy_times.values()]
    betterfftw_ms = [t * 1000 for t in betterfftw_times.values()]
    
    plt.bar(np.arange(len(sizes)) - 0.2, numpy_ms, width=0.4, label='NumPy FFT', color='skyblue')
    plt.bar(np.arange(len(sizes)) + 0.2, betterfftw_ms, width=0.4, label='BetterFFTW', color='coral')
    
    plt.xlabel('Array Size (NxN)')
    plt.ylabel('Time (milliseconds, lower is better)')
    plt.title(f'FFT Performance Comparison {title_suffix}')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    
    # Add size labels and speedup annotations
    plt.xticks(np.arange(len(sizes)), [f"{s}x{s}" for s in sizes])
    
    for i, (size, np_time, bfft_time) in enumerate(zip(sizes, numpy_ms, betterfftw_ms)):
        speedup = np_time / bfft_time
        plt.annotate(f'{speedup:.1f}x', 
                   xy=(i, min(np_time, bfft_time) - 5),
                   ha='center', va='bottom',
                   fontweight='bold', color='green' if speedup > 1 else 'red')
    
    plt.tight_layout()
    plt.savefig(f'fft_comparison{title_suffix.replace(" ", "_")}.png')

def main():
    """Run the benchmarks and plot the results."""
    print("Benchmarking FFT performance...\n")
    total_start_time = time.time()
    
    # Standard 2D benchmarks (original code)
    repeats = 20  # Increased repetitions to better amortize planning costs
    
    # Benchmark power-of-2 sizes
    print("\n--- Power of 2 Sizes ---")
    print("Running NumPy native FFT...")
    numpy_pow2 = benchmark_numpy(POW2_SIZES, repeats)
    
    print("\nRunning BetterFFTW...")
    bfft_first_pow2, bfft_subsequent_pow2 = benchmark_betterfftw(POW2_SIZES, repeats)
    
    # Benchmark non-power-of-2 sizes (where FFTW typically excels)
    print("\n--- Non-Power of 2 Sizes ---")
    print("Running NumPy native FFT...")
    numpy_nonpow2 = benchmark_numpy(NONPOW2_SIZES, repeats)
    
    print("\nRunning BetterFFTW...")
    bfft_first_nonpow2, bfft_subsequent_nonpow2 = benchmark_betterfftw(NONPOW2_SIZES, repeats)
    
    # Plot comparisons for standard tests
    plot_comparison(POW2_SIZES, numpy_pow2, bfft_subsequent_pow2, "- Power of 2 Sizes")
    plot_comparison(NONPOW2_SIZES, numpy_nonpow2, bfft_subsequent_nonpow2, "- Non-Power of 2 Sizes")
    
    # Add 3D FFT benchmarks (these take much longer)
    print("\n--- 3D FFT Benchmarks (Long Running) ---")
    numpy_3d_pow2 = benchmark_3d_numpy(POW2_3D_SIZE)
    bfft_first_3d_pow2, bfft_subsequent_3d_pow2 = benchmark_3d_betterfftw(POW2_3D_SIZE)
    
    numpy_3d_nonpow2 = benchmark_3d_numpy(NONPOW2_3D_SIZE)
    bfft_first_3d_nonpow2, bfft_subsequent_3d_nonpow2 = benchmark_3d_betterfftw(NONPOW2_3D_SIZE)
    
    # Benchmark very large 2D arrays
    large_2d_results = benchmark_large_2d()
    
    # Show overall stats
    print("\nPerformance Summary:")
    
    # Power of 2 sizes
    pow2_speedups = [numpy_pow2[s]/bfft_subsequent_pow2[s] for s in POW2_SIZES]
    avg_pow2_speedup = sum(pow2_speedups) / len(pow2_speedups)
    print(f"Power of 2 sizes - Average speedup: {avg_pow2_speedup:.2f}x")
    
    # Non-power of 2 sizes
    nonpow2_speedups = [numpy_nonpow2[s]/bfft_subsequent_nonpow2[s] for s in NONPOW2_SIZES]
    avg_nonpow2_speedup = sum(nonpow2_speedups) / len(nonpow2_speedups)
    print(f"Non-power of 2 sizes - Average speedup: {avg_nonpow2_speedup:.2f}x")
    
    # 3D FFT speedups
    pow2_3d_speedup = numpy_3d_pow2 / bfft_subsequent_3d_pow2
    nonpow2_3d_speedup = numpy_3d_nonpow2 / bfft_subsequent_3d_nonpow2
    print(f"3D Power of 2 ({POW2_3D_SIZE}^3) - Speedup: {pow2_3d_speedup:.2f}x")
    print(f"3D Non-power of 2 ({NONPOW2_3D_SIZE}^3) - Speedup: {nonpow2_3d_speedup:.2f}x")
    
    # Large 2D speedups
    print(f"Large 2D Power of 2 ({LARGE_POW2_SIZE}x{LARGE_POW2_SIZE}) - Speedup: {large_2d_results['pow2']['speedup']:.2f}x")
    print(f"Large 2D Non-power of 2 ({LARGE_NONPOW2_SIZE}x{LARGE_NONPOW2_SIZE}) - Speedup: {large_2d_results['nonpow2']['speedup']:.2f}x")
    
    # Planning overhead analysis (original code)
    print("\nPlanning Overhead Analysis:")
    for size_list, first_calls, label in [
        (POW2_SIZES, bfft_first_pow2, "Power of 2"),
        (NONPOW2_SIZES, bfft_first_nonpow2, "Non-power of 2")
    ]:
        for size in size_list:
            overhead_ratio = first_calls[size] / bfft_subsequent_pow2[size] if size in POW2_SIZES else first_calls[size] / bfft_subsequent_nonpow2[size]
            print(f"{label} {size}x{size}: First call is {overhead_ratio:.1f}x slower than subsequent calls")
    
    # Total benchmark time
    total_time = time.time() - total_start_time
    print(f"\nTotal benchmark time: {total_time:.2f} seconds")
    
    print("\nDone!")

if __name__ == "__main__":
    main()