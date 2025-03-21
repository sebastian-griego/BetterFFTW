#!/usr/bin/env python
"""
Simple FFT Benchmarking Tool

A straightforward script to compare BetterFFTW with NumPy's FFT implementations.
This avoids the complexity of the comprehensive benchmark while still providing useful data.
"""

import numpy as np
import time
import os
import sys
import gc
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Try to import SciPy for comparison
try:
    import scipy.fft as scipy_fft
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# Set up benchmark parameters
REPEATS = 10  # Number of times to repeat each measurement for consistency
WARMUP_RUNS = 3  # Number of warmup runs before timing

# Test array shapes and types
POW2_SIZES = [256, 512, 1024, 2048, 4096]  # Power of 2 sizes
NONPOW2_SIZES = [384, 768, 1500, 3000]  # Non-power of 2 sizes
RECT_SIZES = [(512, 256), (1024, 512), (2048, 1024)]  # Rectangular sizes
D3_SIZES = [(64, 64, 64), (128, 128, 128)]  # 3D sizes

# Avoid too large sizes on smaller systems
if os.environ.get('SIMPLE_BENCHMARK', '').lower() == 'small':
    POW2_SIZES = [256, 512, 1024]
    NONPOW2_SIZES = [384, 768]
    RECT_SIZES = [(512, 256)]
    D3_SIZES = [(64, 64, 64)]


def benchmark_function(func, array, repeats=REPEATS):
    """Benchmark a function with given input array."""
    # Warmup runs
    for _ in range(WARMUP_RUNS):
        result = func(array.copy())
        # Force computation
        _ = result[0]
        
    # Garbage collect to reduce interference
    gc.collect()
    
    # Timed runs
    times = []
    for _ in range(repeats):
        start = time.time()
        result = func(array.copy())
        # Force computation
        _ = result[0]
        times.append(time.time() - start)
    
    # Return statistics
    return {
        'mean': sum(times) / len(times),
        'min': min(times),
        'max': max(times),
        'times': times
    }


def format_time(seconds):
    """Format time in a human-readable way."""
    if seconds < 0.001:
        return f"{seconds * 1e6:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.4f} s"


def run_1d_benchmarks():
    """Run benchmarks for 1D FFT."""
    print("\n--- 1D FFT Benchmarks ---")
    results = {'pow2': {}, 'nonpow2': {}}
    
    # Set up implementations
    implementations = {
        'NumPy': np.fft.fft,
    }
    
    if HAVE_SCIPY:
        implementations['SciPy'] = scipy_fft.fft
    
    # Import BetterFFTW only if available
    try:
        import betterfftw
        implementations['BetterFFTW'] = betterfftw.fft
        
        # Clear cache for fair comparison
        betterfftw.clear_cache()
    except ImportError:
        print("BetterFFTW not found, skipping its benchmarks")
    
    # Benchmark power of 2 sizes
    print("\nPower of 2 sizes:")
    for size in POW2_SIZES:
        print(f"\nBenchmarking 1D FFT with size {size}")
        results['pow2'][size] = {}
        
        # Create test array
        array = np.random.random(size).astype(np.complex128)
        
        for name, func in implementations.items():
            print(f"  Testing {name}...", end="", flush=True)
            stats = benchmark_function(func, array)
            results['pow2'][size][name] = stats
            print(f" {format_time(stats['mean'])}")
    
    # Benchmark non-power of 2 sizes
    print("\nNon-power of 2 sizes:")
    for size in NONPOW2_SIZES:
        print(f"\nBenchmarking 1D FFT with size {size}")
        results['nonpow2'][size] = {}
        
        # Create test array
        array = np.random.random(size).astype(np.complex128)
        
        for name, func in implementations.items():
            print(f"  Testing {name}...", end="", flush=True)
            stats = benchmark_function(func, array)
            results['nonpow2'][size][name] = stats
            print(f" {format_time(stats['mean'])}")
    
    return results


def run_2d_benchmarks():
    """Run benchmarks for 2D FFT."""
    print("\n--- 2D FFT Benchmarks ---")
    results = {'pow2': {}, 'nonpow2': {}, 'rect': {}}
    
    # Set up implementations
    implementations = {
        'NumPy': np.fft.fft2,
    }
    
    if HAVE_SCIPY:
        implementations['SciPy'] = scipy_fft.fft2
    
    # Import BetterFFTW only if available
    try:
        import betterfftw
        implementations['BetterFFTW'] = betterfftw.fft2
        
        # Clear cache for fair comparison
        betterfftw.clear_cache()
    except ImportError:
        print("BetterFFTW not found, skipping its benchmarks")
    
    # Benchmark power of 2 square sizes
    print("\nPower of 2 square sizes:")
    for size in POW2_SIZES[:3]:  # Limit to first 3 sizes to avoid too large arrays
        shape = (size, size)
        print(f"\nBenchmarking 2D FFT with shape {shape}")
        results['pow2'][shape] = {}
        
        # Create test array
        array = np.random.random(shape).astype(np.complex128)
        
        for name, func in implementations.items():
            print(f"  Testing {name}...", end="", flush=True)
            stats = benchmark_function(func, array)
            results['pow2'][shape][name] = stats
            print(f" {format_time(stats['mean'])}")
    
    # Benchmark non-power of 2 square sizes
    print("\nNon-power of 2 square sizes:")
    for size in NONPOW2_SIZES[:3]:  # Limit to first 3 sizes
        shape = (size, size)
        print(f"\nBenchmarking 2D FFT with shape {shape}")
        results['nonpow2'][shape] = {}
        
        # Create test array
        array = np.random.random(shape).astype(np.complex128)
        
        for name, func in implementations.items():
            print(f"  Testing {name}...", end="", flush=True)
            stats = benchmark_function(func, array)
            results['nonpow2'][shape][name] = stats
            print(f" {format_time(stats['mean'])}")
    
    # Benchmark rectangular sizes
    print("\nRectangular sizes:")
    for shape in RECT_SIZES:
        print(f"\nBenchmarking 2D FFT with shape {shape}")
        results['rect'][shape] = {}
        
        # Create test array
        array = np.random.random(shape).astype(np.complex128)
        
        for name, func in implementations.items():
            print(f"  Testing {name}...", end="", flush=True)
            stats = benchmark_function(func, array)
            results['rect'][shape][name] = stats
            print(f" {format_time(stats['mean'])}")
    
    return results


def run_3d_benchmarks():
    """Run benchmarks for 3D FFT."""
    print("\n--- 3D FFT Benchmarks ---")
    results = {}
    
    # Set up implementations
    implementations = {
        'NumPy': np.fft.fftn,
    }
    
    if HAVE_SCIPY:
        implementations['SciPy'] = scipy_fft.fftn
    
    # Import BetterFFTW only if available
    try:
        import betterfftw
        implementations['BetterFFTW'] = betterfftw.fftn
        
        # Clear cache for fair comparison
        betterfftw.clear_cache()
    except ImportError:
        print("BetterFFTW not found, skipping its benchmarks")
    
    # Benchmark 3D sizes
    for shape in D3_SIZES:
        print(f"\nBenchmarking 3D FFT with shape {shape}")
        results[shape] = {}
        
        # Create test array
        array = np.random.random(shape).astype(np.complex128)
        
        for name, func in implementations.items():
            print(f"  Testing {name}...", end="", flush=True)
            stats = benchmark_function(func, array)
            results[shape][name] = stats
            print(f" {format_time(stats['mean'])}")
    
    return results


def run_real_benchmarks():
    """Run benchmarks for real FFT."""
    print("\n--- Real FFT Benchmarks ---")
    results = {'pow2': {}, 'nonpow2': {}}
    
    # Set up implementations
    implementations = {
        'NumPy': {'1d': np.fft.rfft, '2d': np.fft.rfft2},
    }
    
    if HAVE_SCIPY:
        implementations['SciPy'] = {'1d': scipy_fft.rfft, '2d': scipy_fft.rfft2}
    
    # Import BetterFFTW only if available
    try:
        import betterfftw
        implementations['BetterFFTW'] = {'1d': betterfftw.rfft, '2d': betterfftw.rfft2}
        
        # Clear cache for fair comparison
        betterfftw.clear_cache()
    except ImportError:
        print("BetterFFTW not found, skipping its benchmarks")
    
    # Benchmark 1D rfft with power of 2 sizes
    print("\n1D RFFT - Power of 2 sizes:")
    for size in POW2_SIZES:
        print(f"\nBenchmarking 1D RFFT with size {size}")
        key = ('1d', size)
        results['pow2'][key] = {}
        
        # Create test array (real input)
        array = np.random.random(size).astype(np.float64)
        
        for name, funcs in implementations.items():
            func = funcs['1d']
            print(f"  Testing {name}...", end="", flush=True)
            stats = benchmark_function(func, array)
            results['pow2'][key][name] = stats
            print(f" {format_time(stats['mean'])}")
    
    # Benchmark 2D rfft with power of 2 shapes
    print("\n2D RFFT - Power of 2 shapes:")
    for size in POW2_SIZES[:3]:  # Limit to first 3 sizes
        shape = (size, size)
        print(f"\nBenchmarking 2D RFFT with shape {shape}")
        key = ('2d', shape)
        results['pow2'][key] = {}
        
        # Create test array (real input)
        array = np.random.random(shape).astype(np.float64)
        
        for name, funcs in implementations.items():
            func = funcs['2d']
            print(f"  Testing {name}...", end="", flush=True)
            stats = benchmark_function(func, array)
            results['pow2'][key][name] = stats
            print(f" {format_time(stats['mean'])}")
    
    # Benchmark 1D rfft with non-power of 2 sizes
    print("\n1D RFFT - Non-power of 2 sizes:")
    for size in NONPOW2_SIZES:
        print(f"\nBenchmarking 1D RFFT with size {size}")
        key = ('1d', size)
        results['nonpow2'][key] = {}
        
        # Create test array (real input)
        array = np.random.random(size).astype(np.float64)
        
        for name, funcs in implementations.items():
            func = funcs['1d']
            print(f"  Testing {name}...", end="", flush=True)
            stats = benchmark_function(func, array)
            results['nonpow2'][key][name] = stats
            print(f" {format_time(stats['mean'])}")
    
    # Benchmark 2D rfft with non-power of 2 shapes
    print("\n2D RFFT - Non-power of 2 shapes:")
    for size in NONPOW2_SIZES[:3]:  # Limit to first 3 sizes
        shape = (size, size)
        print(f"\nBenchmarking 2D RFFT with shape {shape}")
        key = ('2d', shape)
        results['nonpow2'][key] = {}
        
        # Create test array (real input)
        array = np.random.random(shape).astype(np.float64)
        
        for name, funcs in implementations.items():
            func = funcs['2d']
            print(f"  Testing {name}...", end="", flush=True)
            stats = benchmark_function(func, array)
            results['nonpow2'][key][name] = stats
            print(f" {format_time(stats['mean'])}")
    
    return results


def run_first_call_benchmark():
    """Benchmark first call overhead (planning cost)."""
    print("\n--- First Call Overhead Benchmarks ---")
    results = {}
    
    # Set up implementations
    implementations = {
        'NumPy': np.fft.fft2,
    }
    
    if HAVE_SCIPY:
        implementations['SciPy'] = scipy_fft.fft2
    
    # Import BetterFFTW with different planning strategies
    try:
        import betterfftw
        
        # Test different planning strategies
        planning_strategies = {
            'ESTIMATE': 'FFTW_ESTIMATE',
            'MEASURE': 'FFTW_MEASURE',
            'PATIENT': 'FFTW_PATIENT'
        }
        
        for name, strategy in planning_strategies.items():
            # Set up BetterFFTW with this strategy
            betterfftw.set_planner_effort(strategy)
            implementations[f'BetterFFTW-{name}'] = betterfftw.fft2
    except ImportError:
        print("BetterFFTW not found, skipping its benchmarks")
    
    # Test sizes
    test_sizes = [(512, 512), (1024, 1024)]
    
    for shape in test_sizes:
        print(f"\nBenchmarking first call overhead with shape {shape}")
        results[shape] = {}
        
        for name, func in implementations.items():
            print(f"  Testing {name}...")
            
            first_times = []
            subsequent_times = []
            
            # Repeat several times to get consistent measurements
            for _ in range(5):
                # Clear the cache if it's BetterFFTW
                if 'BetterFFTW' in name:
                    betterfftw.clear_cache()
                
                # Create a fresh array each time
                array = np.random.random(shape).astype(np.complex128)
                
                # Measure first call
                start = time.time()
                result = func(array.copy())
                _ = result[0]  # Force computation
                first_call_time = time.time() - start
                
                # Measure subsequent call
                start = time.time()
                result = func(array.copy())
                _ = result[0]  # Force computation
                subsequent_call_time = time.time() - start
                
                first_times.append(first_call_time)
                subsequent_times.append(subsequent_call_time)
            
            # Calculate average times
            avg_first = sum(first_times) / len(first_times)
            avg_subsequent = sum(subsequent_times) / len(subsequent_times)
            
            results[shape][name] = {
                'first_call': avg_first,
                'subsequent_call': avg_subsequent,
                'overhead_ratio': avg_first / avg_subsequent if avg_subsequent > 0 else float('inf')
            }
            
            print(f"    First call: {format_time(avg_first)}")
            print(f"    Subsequent call: {format_time(avg_subsequent)}")
            print(f"    Overhead ratio: {results[shape][name]['overhead_ratio']:.2f}x")
    
    return results


def run_thread_scaling_benchmark():
    """Benchmark how performance scales with thread count."""
    print("\n--- Thread Scaling Benchmarks ---")
    results = {}
    
    # Import BetterFFTW only if available
    try:
        import betterfftw
        
        # Get CPU count for scaling
        max_threads = os.cpu_count() or 4
        thread_counts = [1, 2, 4, max_threads]
        # Remove duplicates
        thread_counts = sorted(list(set(thread_counts)))
        
        # Only test a subset of sizes to keep benchmark time reasonable
        test_sizes = [(1024, 1024), (2048, 2048)]
        
        for shape in test_sizes:
            print(f"\nBenchmarking thread scaling with shape {shape}")
            results[shape] = {}
            
            # Create test array
            array = np.random.random(shape).astype(np.complex128)
            
            # Benchmark NumPy for reference
            print(f"  Testing NumPy (baseline)...", end="", flush=True)
            numpy_stats = benchmark_function(np.fft.fft2, array)
            results[shape]['NumPy'] = numpy_stats
            print(f" {format_time(numpy_stats['mean'])}")
            
            # Benchmark BetterFFTW with different thread counts
            for threads in thread_counts:
                print(f"  Testing BetterFFTW with {threads} threads...", end="", flush=True)
                
                # Set thread count
                betterfftw.set_num_threads(threads)
                
                # Clear cache for fair comparison
                betterfftw.clear_cache()
                
                # Warmup call to ensure planning is done
                _ = betterfftw.fft2(array.copy())
                
                # Benchmark
                stats = benchmark_function(betterfftw.fft2, array)
                results[shape][f'BetterFFTW-{threads}threads'] = stats
                
                # Calculate speedup vs NumPy
                speedup = numpy_stats['mean'] / stats['mean']
                
                print(f" {format_time(stats['mean'])} ({speedup:.2f}x speedup)")
    
    except ImportError:
        print("BetterFFTW not found, skipping its benchmarks")
    
    return results


def plot_results(results):
    """Generate plots from benchmark results."""
    print("\nGenerating plots...")
    
    # Create output directory
    output_dir = Path('benchmark_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1D FFT results
    if '1d_results' in results:
        plot_1d_results(results['1d_results'], output_dir)
    
    # Plot 2D FFT results
    if '2d_results' in results:
        plot_2d_results(results['2d_results'], output_dir)
    
    # Plot 3D FFT results
    if '3d_results' in results:
        plot_3d_results(results['3d_results'], output_dir)
    
    # Plot Real FFT results
    if 'real_results' in results:
        plot_real_results(results['real_results'], output_dir)
    
    # Plot first call overhead
    if 'first_call_results' in results:
        plot_first_call_results(results['first_call_results'], output_dir)
    
    # Plot thread scaling
    if 'thread_results' in results:
        plot_thread_results(results['thread_results'], output_dir)
    
    print(f"Plots saved to {output_dir.absolute()}")


def plot_1d_results(results, output_dir):
    """Plot 1D FFT benchmark results."""
    # Plot power of 2 sizes
    plt.figure(figsize=(10, 6))
    
    for impl_name in next(iter(results['pow2'].values())).keys():
        sizes = []
        times = []
        
        for size, impl_results in results['pow2'].items():
            sizes.append(size)
            times.append(impl_results[impl_name]['mean'])
        
        plt.plot(sizes, times, 'o-', label=impl_name)
    
    plt.xlabel('Array Size')
    plt.ylabel('Time (seconds)')
    plt.title('1D FFT Performance - Power of 2 Sizes')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / '1d_fft_pow2.png', dpi=150)
    plt.close()
    
    # Plot non-power of 2 sizes
    plt.figure(figsize=(10, 6))
    
    for impl_name in next(iter(results['nonpow2'].values())).keys():
        sizes = []
        times = []
        
        for size, impl_results in results['nonpow2'].items():
            sizes.append(size)
            times.append(impl_results[impl_name]['mean'])
        
        plt.plot(sizes, times, 'o-', label=impl_name)
    
    plt.xlabel('Array Size')
    plt.ylabel('Time (seconds)')
    plt.title('1D FFT Performance - Non-Power of 2 Sizes')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / '1d_fft_nonpow2.png', dpi=150)
    plt.close()
    
    # Plot speedups vs NumPy
    plt.figure(figsize=(10, 6))
    
    impl_names = [name for name in next(iter(results['pow2'].values())).keys() if name != 'NumPy']
    
    for impl_name in impl_names:
        pow2_sizes = []
        pow2_speedups = []
        nonpow2_sizes = []
        nonpow2_speedups = []
        
        for size, impl_results in results['pow2'].items():
            numpy_time = impl_results['NumPy']['mean']
            impl_time = impl_results[impl_name]['mean']
            speedup = numpy_time / impl_time
            pow2_sizes.append(size)
            pow2_speedups.append(speedup)
        
        for size, impl_results in results['nonpow2'].items():
            numpy_time = impl_results['NumPy']['mean']
            impl_time = impl_results[impl_name]['mean']
            speedup = numpy_time / impl_time
            nonpow2_sizes.append(size)
            nonpow2_speedups.append(speedup)
        
        plt.plot(pow2_sizes, pow2_speedups, 'o-', label=f"{impl_name} (Power of 2)")
        plt.plot(nonpow2_sizes, nonpow2_speedups, 's--', label=f"{impl_name} (Non-Power of 2)")
    
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='NumPy baseline')
    
    plt.xlabel('Array Size')
    plt.ylabel('Speedup vs NumPy')
    plt.title('1D FFT Performance Speedup Relative to NumPy')
    plt.xscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / '1d_fft_speedup.png', dpi=150)
    plt.close()


def plot_2d_results(results, output_dir):
    """Plot 2D FFT benchmark results."""
    # Plot power of 2 sizes
    plt.figure(figsize=(10, 6))
    
    for impl_name in next(iter(results['pow2'].values())).keys():
        sizes = []
        times = []
        
        for shape, impl_results in results['pow2'].items():
            # Use first dimension as size (assuming square arrays)
            size = shape[0]
            sizes.append(size)
            times.append(impl_results[impl_name]['mean'])
        
        plt.plot(sizes, times, 'o-', label=impl_name)
    
    plt.xlabel('Array Size (NxN)')
    plt.ylabel('Time (seconds)')
    plt.title('2D FFT Performance - Power of 2 Square Arrays')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / '2d_fft_pow2.png', dpi=150)
    plt.close()
    
    # Plot non-power of 2 sizes
    plt.figure(figsize=(10, 6))
    
    for impl_name in next(iter(results['nonpow2'].values())).keys():
        sizes = []
        times = []
        
        for shape, impl_results in results['nonpow2'].items():
            # Use first dimension as size (assuming square arrays)
            size = shape[0]
            sizes.append(size)
            times.append(impl_results[impl_name]['mean'])
        
        plt.plot(sizes, times, 'o-', label=impl_name)
    
    plt.xlabel('Array Size (NxN)')
    plt.ylabel('Time (seconds)')
    plt.title('2D FFT Performance - Non-Power of 2 Square Arrays')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / '2d_fft_nonpow2.png', dpi=150)
    plt.close()
    
    # Plot speedups vs NumPy
    plt.figure(figsize=(10, 6))
    
    impl_names = [name for name in next(iter(results['pow2'].values())).keys() if name != 'NumPy']
    
    for impl_name in impl_names:
        pow2_sizes = []
        pow2_speedups = []
        nonpow2_sizes = []
        nonpow2_speedups = []
        
        for shape, impl_results in results['pow2'].items():
            # Use first dimension as size (assuming square arrays)
            size = shape[0]
            numpy_time = impl_results['NumPy']['mean']
            impl_time = impl_results[impl_name]['mean']
            speedup = numpy_time / impl_time
            pow2_sizes.append(size)
            pow2_speedups.append(speedup)
        
        for shape, impl_results in results['nonpow2'].items():
            # Use first dimension as size (assuming square arrays)
            size = shape[0]
            numpy_time = impl_results['NumPy']['mean']
            impl_time = impl_results[impl_name]['mean']
            speedup = numpy_time / impl_time
            nonpow2_sizes.append(size)
            nonpow2_speedups.append(speedup)
        
        plt.plot(pow2_sizes, pow2_speedups, 'o-', label=f"{impl_name} (Power of 2)")
        plt.plot(nonpow2_sizes, nonpow2_speedups, 's--', label=f"{impl_name} (Non-Power of 2)")
    
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='NumPy baseline')
    
    plt.xlabel('Array Size (NxN)')
    plt.ylabel('Speedup vs NumPy')
    plt.title('2D FFT Performance Speedup Relative to NumPy')
    plt.xscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / '2d_fft_speedup.png', dpi=150)
    plt.close()


def plot_3d_results(results, output_dir):
    """Plot 3D FFT benchmark results."""
    plt.figure(figsize=(10, 6))
    
    impl_names = list(next(iter(results.values())).keys())
    shapes = list(results.keys())
    
    # Use first dimension as label (assuming cubic arrays)
    x_labels = [shape[0] for shape in shapes]
    
    # For each implementation, collect times for all shapes
    for impl_name in impl_names:
        times = [results[shape][impl_name]['mean'] for shape in shapes]
        plt.bar(x_labels, times, label=impl_name, alpha=0.7)
    
    plt.xlabel('Array Size (NxNxN)')
    plt.ylabel('Time (seconds)')
    plt.title('3D FFT Performance')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / '3d_fft_performance.png', dpi=150)
    plt.close()
    
    # Plot speedups vs NumPy
    plt.figure(figsize=(10, 6))
    
    impl_names = [name for name in impl_names if name != 'NumPy']
    
    # For each implementation, calculate speedup vs NumPy
    for impl_name in impl_names:
        speedups = []
        
        for shape in shapes:
            numpy_time = results[shape]['NumPy']['mean']
            impl_time = results[shape][impl_name]['mean']
            speedup = numpy_time / impl_time
            speedups.append(speedup)
        
        plt.bar(x_labels, speedups, label=impl_name, alpha=0.7)
    
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='NumPy baseline')
    
    plt.xlabel('Array Size (NxNxN)')
    plt.ylabel('Speedup vs NumPy')
    plt.title('3D FFT Performance Speedup Relative to NumPy')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / '3d_fft_speedup.png', dpi=150)
    plt.close()


def plot_real_results(results, output_dir):
    """Plot real FFT benchmark results."""
    # Get implementations
    impl_names = list(next(iter(results['pow2'].values())).keys())
    
    # Split 1D and 2D results
    dims = ['1d', '2d']
    
    for dim in dims:
        # Collect data
        pow2_sizes = []
        nonpow2_sizes = []
        
        for key in results['pow2'].keys():
            if key[0] == dim:
                if dim == '1d':
                    pow2_sizes.append(key[1])
                else:
                    pow2_sizes.append(key[1][0])  # Use first dimension for 2D
        
        for key in results['nonpow2'].keys():
            if key[0] == dim:
                if dim == '1d':
                    nonpow2_sizes.append(key[1])
                else:
                    nonpow2_sizes.append(key[1][0])  # Use first dimension for 2D
        
        pow2_sizes.sort()
        nonpow2_sizes.sort()
        
        # Skip if we don't have any sizes
        if not pow2_sizes and not nonpow2_sizes:
            continue
        
        # Plot power of 2 sizes
        if pow2_sizes:
            plt.figure(figsize=(10, 6))
            
            for impl_name in impl_names:
                times = []
                
                for size in pow2_sizes:
                    if dim == '1d':
                        key = (dim, size)
                    else:
                        key = (dim, (size, size))
                    
                    times.append(results['pow2'][key][impl_name]['mean'])
                
                plt.plot(pow2_sizes, times, 'o-', label=impl_name)
            
            plt.xlabel(f'Array Size ({dim.upper()})')
            plt.ylabel('Time (seconds)')
            plt.title(f'{dim.upper()} RFFT Performance - Power of 2 Sizes')
            plt.xscale('log', base=2)
            plt.yscale('log')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(output_dir / f'{dim}_rfft_pow2.png', dpi=150)
            plt.close()
        
        # Plot non-power of 2 sizes
        if nonpow2_sizes:
            plt.figure(figsize=(10, 6))
            
            for impl_name in impl_names:
                times = []
                
                for size in nonpow2_sizes:
                    if dim == '1d':
                        key = (dim, size)
                    else:
                        key = (dim, (size, size))
                    
                    times.append(results['nonpow2'][key][impl_name]['mean'])
                
                plt.plot(nonpow2_sizes, times, 'o-', label=impl_name)
            
            plt.xlabel(f'Array Size ({dim.upper()})')
            plt.ylabel('Time (seconds)')
            plt.title(f'{dim.upper()} RFFT Performance - Non-Power of 2 Sizes')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(output_dir / f'{dim}_rfft_nonpow2.png', dpi=150)
            plt.close()
        
        # Plot speedups vs NumPy
        plt.figure(figsize=(10, 6))
        
        non_numpy_impls = [name for name in impl_names if name != 'NumPy']
        
        for impl_name in non_numpy_impls:
            pow2_speedups = []
            nonpow2_speedups = []
            
            for size in pow2_sizes:
                if dim == '1d':
                    key = (dim, size)
                else:
                    key = (dim, (size, size))
                
                numpy_time = results['pow2'][key]['NumPy']['mean']
                impl_time = results['pow2'][key][impl_name]['mean']
                speedup = numpy_time / impl_time
                pow2_speedups.append(speedup)
            
            for size in nonpow2_sizes:
                if dim == '1d':
                    key = (dim, size)
                else:
                    key = (dim, (size, size))
                
                numpy_time = results['nonpow2'][key]['NumPy']['mean']
                impl_time = results['nonpow2'][key][impl_name]['mean']
                speedup = numpy_time / impl_time
                nonpow2_speedups.append(speedup)
            
            if pow2_sizes:
                plt.plot(pow2_sizes, pow2_speedups, 'o-', label=f"{impl_name} (Power of 2)")
            if nonpow2_sizes:
                plt.plot(nonpow2_sizes, nonpow2_speedups, 's--', label=f"{impl_name} (Non-Power of 2)")
        
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='NumPy baseline')
        
        plt.xlabel(f'Array Size ({dim.upper()})')
        plt.ylabel('Speedup vs NumPy')
        plt.title(f'{dim.upper()} RFFT Performance Speedup Relative to NumPy')
        plt.xscale('log')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(output_dir / f'{dim}_rfft_speedup.png', dpi=150)
        plt.close()


def plot_first_call_results(results, output_dir):
    """Plot first call overhead benchmark results."""
    plt.figure(figsize=(12, 6))
    
    shapes = list(results.keys())
    impl_names = list(next(iter(results.values())).keys())
    
    # Create x-axis labels
    x_labels = [f"{shape[0]}x{shape[1]}" for shape in shapes]
    x = np.arange(len(x_labels))
    width = 0.15  # Width of bars
    
    # Plot overhead ratios
    for i, impl_name in enumerate(impl_names):
        ratios = [results[shape][impl_name]['overhead_ratio'] for shape in shapes]
        plt.bar(x + i*width - (len(impl_names)-1)*width/2, ratios, width, label=impl_name)
    
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='No overhead')
    
    plt.xlabel('Array Shape')
    plt.ylabel('First Call / Subsequent Call Ratio')
    plt.title('FFT First Call Overhead')
    plt.xticks(x, x_labels)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / 'first_call_overhead.png', dpi=150)
    plt.close()
    
    # Plot actual times
    plt.figure(figsize=(12, 10))
    
    for i, shape in enumerate(shapes):
        plt.subplot(len(shapes), 1, i+1)
        
        first_times = []
        subsequent_times = []
        
        for impl_name in impl_names:
            first_times.append(results[shape][impl_name]['first_call'])
            subsequent_times.append(results[shape][impl_name]['subsequent_call'])
        
        x = np.arange(len(impl_names))
        width = 0.35
        
        plt.bar(x - width/2, first_times, width, label='First Call')
        plt.bar(x + width/2, subsequent_times, width, label='Subsequent Calls')
        
        plt.xlabel('Implementation')
        plt.ylabel('Time (seconds)')
        plt.title(f'FFT Timing for {shape[0]}x{shape[1]} Array')
        plt.xticks(x, impl_names, rotation=45, ha='right')
        plt.yscale('log')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
        plt.tight_layout()
    
    plt.savefig(output_dir / 'first_call_times.png', dpi=150)
    plt.close()


def plot_thread_results(results, output_dir):
    """Plot thread scaling benchmark results."""
    plt.figure(figsize=(10, 6))
    
    shapes = list(results.keys())
    
    for shape in shapes:
        # Get thread counts from implementation names
        thread_counts = []
        speedups = []
        
        for impl_name, stats in results[shape].items():
            if impl_name == 'NumPy':
                numpy_time = stats['mean']
            elif 'BetterFFTW' in impl_name:
                threads = int(impl_name.split('-')[1].replace('threads', ''))
                thread_counts.append(threads)
                impl_time = stats['mean']
                speedup = numpy_time / impl_time
                speedups.append(speedup)
        
        # Sort by thread count
        sorted_data = sorted(zip(thread_counts, speedups))
        thread_counts = [x[0] for x in sorted_data]
        speedups = [x[1] for x in sorted_data]
        
        plt.plot(thread_counts, speedups, 'o-', label=f"{shape[0]}x{shape[1]}")
    
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='NumPy baseline')
    
    plt.xlabel('Thread Count')
    plt.ylabel('Speedup vs NumPy')
    plt.title('BetterFFTW Performance vs Thread Count')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / 'thread_scaling.png', dpi=150)
    plt.close()
    
    # Plot ideal scaling
    plt.figure(figsize=(10, 6))
    
    for shape in shapes:
        # Get thread counts from implementation names
        thread_counts = []
        speedups = []
        
        # Find single thread performance
        single_thread_time = None
        for impl_name, stats in results[shape].items():
            if 'BetterFFTW-1threads' in impl_name:
                single_thread_time = stats['mean']
        
        if single_thread_time is None:
            continue
        
        for impl_name, stats in results[shape].items():
            if 'BetterFFTW' in impl_name:
                threads = int(impl_name.split('-')[1].replace('threads', ''))
                thread_counts.append(threads)
                impl_time = stats['mean']
                # Speedup relative to single thread
                speedup = single_thread_time / impl_time
                speedups.append(speedup)
        
        # Sort by thread count
        sorted_data = sorted(zip(thread_counts, speedups))
        thread_counts = [x[0] for x in sorted_data]
        speedups = [x[1] for x in sorted_data]
        
        plt.plot(thread_counts, speedups, 'o-', label=f"{shape[0]}x{shape[1]} (actual)")
        
        # Add ideal scaling line
        max_threads = max(thread_counts)
        plt.plot([1, max_threads], [1, max_threads], '--', color='gray', alpha=0.7)
    
    plt.plot([1, max_threads], [1, max_threads], '--', color='gray', alpha=0.7, label='Ideal scaling')
    
    plt.xlabel('Thread Count')
    plt.ylabel('Speedup vs Single Thread')
    plt.title('BetterFFTW Thread Scaling Efficiency')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(output_dir / 'thread_efficiency.png', dpi=150)
    plt.close()


def main():
    """Run the benchmark and plot results."""
    print("Starting simple FFT benchmark...")
    
    # Run benchmarks
    results = {}
    
    print("\nRunning 1D FFT benchmarks...")
    results['1d_results'] = run_1d_benchmarks()
    
    print("\nRunning 2D FFT benchmarks...")
    results['2d_results'] = run_2d_benchmarks()
    
    print("\nRunning 3D FFT benchmarks...")
    results['3d_results'] = run_3d_benchmarks()
    
    print("\nRunning Real FFT benchmarks...")
    results['real_results'] = run_real_benchmarks()
    
    print("\nRunning first call overhead benchmarks...")
    results['first_call_results'] = run_first_call_benchmark()
    
    print("\nRunning thread scaling benchmarks...")
    results['thread_results'] = run_thread_scaling_benchmark()
    
    # Generate plots
    plot_results(results)
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()