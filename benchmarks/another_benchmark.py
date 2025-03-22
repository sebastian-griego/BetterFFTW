"""
Comprehensive benchmark for BetterFFTW

This script performs extensive benchmarking of FFT implementations across:
- Different transform types (complex FFT, real FFT, etc.)
- Different array dimensions (1D, 2D, 3D)
- Different array sizes (small to very large)
- Different data types (float32/64, complex64/128)
- Different implementations (NumPy, SciPy, PyFFTW, BetterFFTW)
- Different thread counts
- Different planning strategies
- Different usage patterns (single vs repeated transforms)

Results are saved to CSV files and plotted for easy analysis.
"""

import os
import time
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import psutil
import platform
import scipy.fft
import pyfftw
import betterfftw
from datetime import datetime
import multiprocessing

# Create results directory
results_dir = "benchmark_results"
os.makedirs(results_dir, exist_ok=True)

# Get system information
def get_system_info():
    """Get detailed system information for benchmark context."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pyfftw_version": pyfftw.__version__,
        "betterfftw_version": betterfftw.__version__,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return info

# Try to detect if NumPy is using MKL
def is_numpy_using_mkl():
    """Try to determine if NumPy is using Intel MKL."""
    try:
        # Check configuration
        np_config = np.show_config()
        return "mkl" in str(np_config).lower()
    except:
        # Alternative method: check for specific MKL functions
        try:
            from ctypes import cdll, c_char_p
            mkl = cdll.LoadLibrary("libmkl_rt.so")
            return True
        except:
            pass
    return False

# Record system info
system_info = get_system_info()
system_info["numpy_mkl"] = is_numpy_using_mkl()

# Print system information
print("=" * 50)
print("SYSTEM INFORMATION")
print("=" * 50)
for key, value in system_info.items():
    print(f"{key}: {value}")
print("=" * 50)

# Save system info to file
with open(os.path.join(results_dir, "system_info.txt"), "w") as f:
    for key, value in system_info.items():
        f.write(f"{key}: {value}\n")

# ===============================================================================
# Benchmark configuration
# ===============================================================================

# FFT implementations to test
implementations = {
    "numpy": {
        "name": "NumPy FFT",
        "module": np.fft,
        "setup": lambda: None,
        "cleanup": lambda: None,
        "kwargs": {}
    },
    "scipy": {
        "name": "SciPy FFT",
        "module": scipy.fft,
        "setup": lambda: None,
        "cleanup": lambda: None,
        "kwargs": {}
    },
    "pyfftw_estimate": {
        "name": "PyFFTW (ESTIMATE)",
        "module": pyfftw.interfaces.numpy_fft,
        "setup": lambda: pyfftw.interfaces.cache.enable(),
        "cleanup": lambda: pyfftw.interfaces.cache.disable(),
        "kwargs": {"planner_effort": "FFTW_ESTIMATE"}
    },
    "pyfftw_measure": {
        "name": "PyFFTW (MEASURE)",
        "module": pyfftw.interfaces.numpy_fft,
        "setup": lambda: pyfftw.interfaces.cache.enable(),
        "cleanup": lambda: pyfftw.interfaces.cache.disable(),
        "kwargs": {"planner_effort": "FFTW_MEASURE"}
    },
    "betterfftw_default": {
        "name": "BetterFFTW (Default)",
        "module": betterfftw,
        "setup": lambda: betterfftw.clear_cache(),
        "cleanup": lambda: None,
        "kwargs": {}
    },
    "betterfftw_estimate": {
        "name": "BetterFFTW (ESTIMATE)",
        "module": betterfftw,
        "setup": lambda: betterfftw.clear_cache(),
        "cleanup": lambda: None,
        "kwargs": {"planner": "FFTW_ESTIMATE"}
    },
    "betterfftw_measure": {
        "name": "BetterFFTW (MEASURE)",
        "module": betterfftw,
        "setup": lambda: betterfftw.clear_cache(),
        "cleanup": lambda: None,
        "kwargs": {"planner": "FFTW_MEASURE"}
    },
    "betterfftw_patient": {
        "name": "BetterFFTW (PATIENT)",
        "module": betterfftw,
        "setup": lambda: betterfftw.clear_cache(),
        "cleanup": lambda: None,
        "kwargs": {"planner": "FFTW_PATIENT"}
    }
}

# Transform types to test
transform_types = {
    "fft": {"name": "Complex FFT (1D)", "func_name": "fft"},
    "ifft": {"name": "Inverse Complex FFT (1D)", "func_name": "ifft"},
    "rfft": {"name": "Real FFT (1D)", "func_name": "rfft"},
    "irfft": {"name": "Inverse Real FFT (1D)", "func_name": "irfft"},
    "fft2": {"name": "Complex FFT (2D)", "func_name": "fft2"},
    "ifft2": {"name": "Inverse Complex FFT (2D)", "func_name": "ifft2"},
    "rfft2": {"name": "Real FFT (2D)", "func_name": "rfft2"},
    "irfft2": {"name": "Inverse Real FFT (2D)", "func_name": "irfft2"},
    "fftn": {"name": "Complex FFT (N-D)", "func_name": "fftn"},
    "ifftn": {"name": "Inverse Complex FFT (N-D)", "func_name": "ifftn"},
    "rfftn": {"name": "Real FFT (N-D)", "func_name": "rfftn"},
    "irfftn": {"name": "Inverse Real FFT (N-D)", "func_name": "irfftn"}
}

# Array sizes to test (powers of 2 and non-powers of 2)
small_sizes = [16, 32, 64, 100]               # Small arrays
medium_sizes = [128, 256, 512, 1000]          # Medium arrays
large_sizes = [1024, 2048, 4096, 10000]       # Large arrays
very_large_sizes = [8192, 16384, 32768]       # Very large arrays (may require significant memory)

# Array dimensions
dimensions = {
    "1d": {"name": "1D", "shape_func": lambda n: (n,)},
    "2d_square": {"name": "2D Square", "shape_func": lambda n: (n, n)},
    "2d_rect": {"name": "2D Rectangle", "shape_func": lambda n: (n, n//2)},
    "3d": {"name": "3D", "shape_func": lambda n: (n, n, n)},
    "4d": {"name": "4D", "shape_func": lambda n: (n, n, n, n)}
}

# Data types
dtypes = {
    "float32": {"name": "float32", "complex_dtype": np.complex64, "real_dtype": np.float32},
    "float64": {"name": "float64", "complex_dtype": np.complex128, "real_dtype": np.float64}
}

# Thread counts to test
max_threads = os.cpu_count()
thread_counts = [1, 2, 4, min(8, max_threads), max_threads]
thread_counts = sorted(list(set(thread_counts)))  # Remove duplicates

# Function to create test array
def create_test_array(shape, dtype, is_complex=False):
    """Create a test array with specified shape and dtype."""
    if is_complex:
        real_part = np.random.random(shape).astype(dtype)
        imag_part = np.random.random(shape).astype(dtype)
        return real_part + 1j * imag_part
    else:
        return np.random.random(shape).astype(dtype)

# ===============================================================================
# Benchmark functions
# ===============================================================================

def run_single_benchmark(implementation, transform_type, array, n_runs=5, threads=None, **kwargs):
    """
    Run a single benchmark test and measure performance.
    
    Args:
        implementation: Implementation dictionary
        transform_type: Transform type dictionary
        array: Input array
        n_runs: Number of times to run the transform
        threads: Number of threads to use (for threaded implementations)
        **kwargs: Additional keyword arguments
    
    Returns:
        Dictionary with benchmark results
    """
    module = implementation["module"]
    func_name = transform_type["func_name"]
    impl_kwargs = implementation["kwargs"].copy()
    
    # Add threads parameter for FFTW implementations if specified
    if threads is not None and implementation["name"] != "NumPy FFT" and implementation["name"] != "SciPy FFT":
        impl_kwargs["threads"] = threads
    
    # Add any additional kwargs
    impl_kwargs.update(kwargs)
    
    # Get function
    func = getattr(module, func_name)
    
    # Setup implementation (e.g., enable caching)
    implementation["setup"]()
    
    # Warmup run (not timed) to handle any first-time overhead
    result = func(array.copy(), **impl_kwargs)
    
    # Run the benchmark
    times = []
    for _ in range(n_runs):
        # Create a copy to avoid in-place modifications affecting subsequent runs
        array_copy = array.copy()
        
        # Force garbage collection before timing
        gc.collect()
        
        start_time = time.time()
        result = func(array_copy, **impl_kwargs)
        end_time = time.time()
        
        times.append(end_time - start_time)
    
    # Cleanup implementation (e.g., disable caching)
    implementation["cleanup"]()
    
    # Calculate statistics
    min_time = min(times)
    max_time = max(times)
    mean_time = sum(times) / len(times)
    median_time = sorted(times)[len(times) // 2]
    
    # Get memory usage of result
    result_memory = result.nbytes / (1024 * 1024)  # in MB
    
    # Get input array shape and size
    input_shape = array.shape
    input_size = array.size
    input_memory = array.nbytes / (1024 * 1024)  # in MB
    
    # Return benchmark results
    return {
        "implementation": implementation["name"],
        "transform_type": transform_type["name"],
        "func_name": func_name,
        "input_shape": input_shape,
        "input_size": input_size,
        "input_memory_mb": input_memory,
        "result_memory_mb": result_memory,
        "threads": threads,
        "n_runs": n_runs,
        "min_time": min_time,
        "max_time": max_time,
        "mean_time": mean_time,
        "median_time": median_time,
        "throughput_gflops": calculate_fft_gflops(input_shape, func_name) / median_time,
        **impl_kwargs
    }

def calculate_fft_gflops(shape, func_name):
    """
    Calculate the approximate GFLOPS for an FFT operation.
    
    This is a rough approximation based on the FFT complexity.
    The actual FLOP count can vary by implementation.
    
    Args:
        shape: Input array shape
        func_name: Name of the FFT function
    
    Returns:
        Approximate GFLOPS for the operation
    """
    # Determine dimensionality
    if func_name in ["fft", "ifft", "rfft", "irfft"]:
        # 1D transform
        n = shape[0]
        # FFT is approximately 5*n*log2(n) floating point operations
        flops = 5 * n * np.log2(n)
    elif func_name in ["fft2", "ifft2", "rfft2", "irfft2"]:
        # 2D transform
        n1, n2 = shape[0], shape[1]
        # 2D FFT is roughly 5*n1*n2*log2(n1*n2)
        flops = 5 * n1 * n2 * np.log2(n1 * n2)
    else:
        # N-D transform
        n_total = np.prod(shape)
        # N-D FFT is roughly 5*n_total*log2(n_total)
        flops = 5 * n_total * np.log2(n_total)
    
    # Convert to GFLOPS (billion floating point operations)
    return flops / 1e9
def run_benchmark_suite(config):
    """
    Run a full benchmark suite based on configuration.
    
    Args:
        config: Dictionary with benchmark configuration
    
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    # Print benchmark configuration
    print("\n" + "=" * 50)
    print(f"RUNNING BENCHMARK: {config['name']}")
    print("=" * 50)
    print(f"Implementations: {', '.join([impl['name'] for impl in config['implementations']])}")
    print(f"Transform types: {', '.join([t['name'] for t in config['transform_types']])}")
    print(f"Sizes: {config['sizes']}")
    print(f"Dimensions: {', '.join([dim['name'] for dim in config['dimensions']])}")
    print(f"Data types: {', '.join([dt['name'] for dt in config['dtypes']])}")
    print(f"Thread counts: {config['thread_counts']}")
    print(f"Runs per test: {config['n_runs']}")
    print("=" * 50)
    
    # Count total tests, accounting for skipped combinations due to dimensionality
    valid_tests = 0
    for transform_type in config['transform_types']:
        for dimension in config['dimensions']:
            transform_name = transform_type['func_name']
            dim_name = dimension['name']
            
            # Check compatibility
            if transform_name in ['fft2', 'ifft2', 'rfft2', 'irfft2'] and dim_name == '1D':
                continue
            if transform_name in ['fftn', 'ifftn', 'rfftn', 'irfftn'] and dim_name == '1D' and min(config['sizes']) < 1000:
                continue
                
            # Count valid tests for this transform-dimension pair
            for implementation in config['implementations']:
                for size in config['sizes']:
                    for dtype_info in config['dtypes']:
                        for thread_count in config['thread_counts']:
                            # Skip thread benchmarks for NumPy and SciPy if thread_count > 1
                            if thread_count > 1 and implementation['name'] in ['NumPy FFT', 'SciPy FFT']:
                                continue
                                
                            # Skip higher dimensions for very large sizes to avoid OOM
                            if size in very_large_sizes and dimension['name'] in ['3D', '4D']:
                                continue
                                
                            valid_tests += 1
                            
    print(f"Total valid tests to run: {valid_tests}")
    
    # Counters for progress tracking
    completed_tests = 0
    
    # Start timer for the entire suite
    suite_start_time = time.time()
    
    # Iterate over all combinations
    for implementation in config['implementations']:
        for transform_type in config['transform_types']:
            for size in config['sizes']:
                for dimension in config['dimensions']:
                    # Check dimension compatibility with transform
                    transform_name = transform_type['func_name']
                    dim_name = dimension['name']
                    
                    # Skip incompatible dimension-transform combinations
                    if transform_name in ['fft2', 'ifft2', 'rfft2', 'irfft2'] and dim_name == '1D':
                        print(f"Skipping {transform_type['name']} on {dim_name} array (incompatible dimensions)")
                        continue
                        
                    if transform_name in ['fftn', 'ifftn', 'rfftn', 'irfftn'] and dim_name == '1D' and size < 1000:
                        # fftn works on 1D but is inefficient - only test for large arrays where the difference matters
                        print(f"Skipping {transform_type['name']} on {dim_name} array (inefficient combination)")
                        continue
                    
                    for dtype_info in config['dtypes']:
                        for thread_count in config['thread_counts']:
                            # Skip higher dimensions for very large sizes to avoid OOM
                            if size in very_large_sizes and dimension['name'] in ['3D', '4D']:
                                print(f"Skipping {dimension['name']} with size {size} (too large)")
                                completed_tests += 1
                                continue
                            
                            # Skip thread benchmarks for NumPy and SciPy if thread_count > 1
                            if thread_count > 1 and implementation['name'] in ['NumPy FFT', 'SciPy FFT']:
                                # NumPy/SciPy don't support threading directly
                                completed_tests += 1
                                continue
                            
                            # Calculate shape based on dimension
                            shape = dimension['shape_func'](size)
                            
                            # Check if the shape is too large
                            array_size_bytes = np.prod(shape) * (8 if dtype_info['name'] == 'float64' else 4)
                            if array_size_bytes > 8 * 1024**3:  # 8 GB limit
                                print(f"Skipping {shape} with {dtype_info['name']} (requires >{array_size_bytes/1024**3:.1f} GB)")
                                completed_tests += 1
                                continue
                            
                            # Determine if complex input is required
                            is_complex_input = transform_type['func_name'] in [
                                'ifft', 'ifft2', 'ifftn', 'irfft', 'irfft2', 'irfftn'
                            ]
                            
                            # Determine dtype
                            dtype = dtype_info['complex_dtype'] if is_complex_input else dtype_info['real_dtype']
                            
                            # Create test array
                            try:
                                array = create_test_array(shape, dtype, is_complex=is_complex_input)
                            except MemoryError:
                                print(f"Memory error creating array of shape {shape} and dtype {dtype}")
                                completed_tests += 1
                                continue
                            
                            # Print progress
                            completed_tests += 1
                            progress = completed_tests / valid_tests * 100
                            elapsed_time = time.time() - suite_start_time
                            eta = elapsed_time / completed_tests * (valid_tests - completed_tests)
                            
                            print(f"[{progress:.1f}%] Testing {implementation['name']} {transform_type['name']} "
                                  f"on {dimension['name']} array of size {size} ({dtype_info['name']}) "
                                  f"with {thread_count} threads | ETA: {eta:.1f}s")
                            
                            # Run benchmark
                            try:
                                result = run_single_benchmark(
                                    implementation=implementation,
                                    transform_type=transform_type,
                                    array=array,
                                    n_runs=config['n_runs'],
                                    threads=thread_count
                                )
                                results.append(result)
                                
                                # Print result summary
                                print(f"  Result: median time = {result['median_time']:.6f}s, "
                                      f"throughput = {result['throughput_gflops']:.2f} GFLOPS")
                            except Exception as e:
                                print(f"  Error: {str(e)}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    results_file = os.path.join(results_dir, f"{config['name'].replace(' ', '_').lower()}.csv")
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    # Print summary statistics
    print("\nSummary statistics by implementation:")
    summary = df.groupby('implementation')['median_time'].agg(['mean', 'min', 'max'])
    print(tabulate(summary, headers='keys', tablefmt='grid'))
    
    return df

# Helper function to check transform-dimension compatibility
def is_compatible(transform_name, dimension_name, size=None):
    """
    Check if a transform type is compatible with the given dimension.
    
    Args:
        transform_name: Name of the transform function
        dimension_name: Name of the dimension
        size: Size of the array (some combinations only make sense for large arrays)
        
    Returns:
        True if compatible, False otherwise
    """
    # 2D transforms need at least 2D arrays
    if transform_name in ['fft2', 'ifft2', 'rfft2', 'irfft2'] and dimension_name == '1D':
        return False
        
    # N-D transforms on 1D arrays are inefficient for small sizes
    if transform_name in ['fftn', 'ifftn', 'rfftn', 'irfftn'] and dimension_name == '1D' and size is not None and size < 1000:
        return False
        
    return True

# Update repeated transform benchmark
def run_repeated_transform_benchmark(config):
    """
    Benchmark the performance impact of repeating the same transform.
    
    Args:
        config: Dictionary with benchmark configuration
    
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    print("\n" + "=" * 50)
    print(f"RUNNING REPEATED TRANSFORM BENCHMARK: {config['name']}")
    print("=" * 50)
    
    # Iterate over all combinations
    for implementation in config['implementations']:
        for transform_type in config['transform_types']:
            for size in config['sizes']:
                for dimension in config['dimensions']:
                    # Check compatibility
                    transform_name = transform_type['func_name']
                    dim_name = dimension['name']
                    
                    if not is_compatible(transform_name, dim_name, size):
                        print(f"Skipping {transform_type['name']} on {dim_name} array (incompatible dimensions)")
                        continue
                
                    # Calculate shape based on dimension
                    shape = dimension['shape_func'](size)
                    
                    # Create test array (always use float64/complex128 for consistency)
                    is_complex_input = transform_type['func_name'] in [
                        'ifft', 'ifft2', 'ifftn', 'irfft', 'irfft2', 'irfftn'
                    ]
                    dtype = np.complex128 if is_complex_input else np.float64
                    array = create_test_array(shape, dtype, is_complex=is_complex_input)
                    
                    # Get function
                    module = implementation["module"]
                    func_name = transform_type["func_name"]
                    func = getattr(module, func_name)
                    impl_kwargs = implementation["kwargs"].copy()
                    
                    if implementation["name"] != "NumPy FFT" and implementation["name"] != "SciPy FFT":
                        impl_kwargs["threads"] = config['thread_count']
                    
                    # Setup implementation
                    implementation["setup"]()
                    
                    # Print info
                    print(f"Testing {implementation['name']} {transform_type['name']} "
                          f"on {dimension['name']} array of size {size}")
                    
                    try:
                        # Track time for each repetition
                        times = []
                        
                        # Run with increasing repetition counts
                        for repetitions in config['repetition_counts']:
                            # Force garbage collection
                            gc.collect()
                            
                            start_time = time.time()
                            for _ in range(repetitions):
                                array_copy = array.copy()
                                result = func(array_copy, **impl_kwargs)
                            end_time = time.time()
                            
                            total_time = end_time - start_time
                            time_per_transform = total_time / repetitions
                            
                            times.append(time_per_transform)
                            
                            # Record result
                            results.append({
                                "implementation": implementation['name'],
                                "transform_type": transform_type['name'],
                                "dimension": dimension['name'],
                                "size": size,
                                "repetitions": repetitions,
                                "total_time": total_time,
                                "time_per_transform": time_per_transform
                            })
                            
                            print(f"  {repetitions} repetitions: {total_time:.6f}s total, "
                                  f"{time_per_transform:.6f}s per transform")
                        
                        # Calculate improvement from first to last
                        if times[0] > 0 and times[-1] > 0:
                            improvement = times[0] / times[-1]
                            print(f"  Improvement: {improvement:.2f}x")
                    except Exception as e:
                        print(f"  Error: {str(e)}")
                    finally:
                        # Cleanup implementation
                        implementation["cleanup"]()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    results_file = os.path.join(results_dir, f"{config['name'].replace(' ', '_').lower()}.csv")
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    return df

def run_non_power_of_two_benchmark(config):
    """
    Benchmark performance on power-of-two vs non-power-of-two sizes.
    
    Args:
        config: Dictionary with benchmark configuration
    
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    print("\n" + "=" * 50)
    print(f"RUNNING NON-POWER-OF-TWO BENCHMARK: {config['name']}")
    print("=" * 50)
    
    # Function to factorize a number
    def factorize(n):
        factors = []
        d = 2
        while n > 1:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
            if d*d > n:
                if n > 1: factors.append(n)
                break
        return factors
    
    # Process each size to determine its factors
    size_info = []
    for size in config['sizes']:
        factors = factorize(size)
        if len(factors) == 1 and factors[0] == 2:
            # Pure power of 2
            size_type = "power_of_2"
            factor_str = f"2^{int(np.log2(size))}"
        elif len(factors) == 1:
            # Prime number
            size_type = "prime"
            factor_str = str(factors[0])
        else:
            # Mixed factors
            factor_counts = {}
            for f in factors:
                factor_counts[f] = factor_counts.get(f, 0) + 1
            factor_str = " × ".join([f"{f}^{c}" if c > 1 else str(f) for f, c in factor_counts.items()])
            
            # Determine if it's mostly power of 2 or mostly other
            if 2 in factor_counts and factor_counts[2] > len(factors) / 2:
                size_type = "mostly_power_of_2"
            else:
                size_type = "mixed_factors"
        
        size_info.append({
            "size": size,
            "factors": factors,
            "factor_str": factor_str,
            "size_type": size_type
        })
    
    # Iterate over implementations and sizes
    for implementation in config['implementations']:
        for transform_type in config['transform_types']:
            for size_data in size_info:
                size = size_data['size']
                
                # Print info
                print(f"Testing {implementation['name']} {transform_type['name']} on size {size} ({size_data['factor_str']})")
                
                # Create 1D array (for simplicity)
                is_complex_input = transform_type['func_name'] in [
                    'ifft', 'ifft2', 'ifftn', 'irfft', 'irfft2', 'irfftn'
                ]
                dtype = np.complex128 if is_complex_input else np.float64
                array = create_test_array((size,), dtype, is_complex=is_complex_input)
                
                # Get function
                module = implementation["module"]
                func_name = transform_type["func_name"]
                func = getattr(module, func_name)
                impl_kwargs = implementation["kwargs"].copy()
                
                # Setup implementation
                implementation["setup"]()
                
                # Warm-up run
                _ = func(array.copy(), **impl_kwargs)
                
                # Run the benchmark
                times = []
                for _ in range(config['n_runs']):
                    array_copy = array.copy()
                    gc.collect()
                    
                    start_time = time.time()
                    result = func(array_copy, **impl_kwargs)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                
                # Calculate statistics
                min_time = min(times)
                max_time = max(times)
                mean_time = sum(times) / len(times)
                median_time = sorted(times)[len(times) // 2]
                
                # Record result
                results.append({
                    "implementation": implementation['name'],
                    "transform_type": transform_type['name'],
                    "size": size,
                    "factor_str": size_data['factor_str'],
                    "size_type": size_data['size_type'],
                    "min_time": min_time,
                    "max_time": max_time,
                    "mean_time": mean_time,
                    "median_time": median_time
                })
                
                print(f"  Median time: {median_time:.6f}s")
                
                # Cleanup implementation
                implementation["cleanup"]()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    results_file = os.path.join(results_dir, f"{config['name'].replace(' ', '_').lower()}.csv")
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    return df

# Update threading scaling benchmark
def run_threading_scaling_benchmark(config):
    """
    Benchmark how performance scales with thread count.
    
    Args:
        config: Dictionary with benchmark configuration
    
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    print("\n" + "=" * 50)
    print(f"RUNNING THREADING SCALING BENCHMARK: {config['name']}")
    print("=" * 50)
    
    # Iterate over implementations, sizes, dimensions
    for implementation in config['implementations']:
        # Skip NumPy and SciPy as they don't support direct threading
        if implementation['name'] in ['NumPy FFT', 'SciPy FFT']:
            continue
            
        for transform_type in config['transform_types']:
            for size in config['sizes']:
                for dimension in config['dimensions']:
                    # Check compatibility
                    transform_name = transform_type['func_name']
                    dim_name = dimension['name']
                    
                    if not is_compatible(transform_name, dim_name, size):
                        print(f"Skipping {transform_type['name']} on {dim_name} array (incompatible dimensions)")
                        continue
                    
                    # Calculate shape based on dimension
                    shape = dimension['shape_func'](size)
                    
                    # Create test array
                    is_complex_input = transform_type['func_name'] in [
                        'ifft', 'ifft2', 'ifftn', 'irfft', 'irfft2', 'irfftn'
                    ]
                    dtype = np.complex128 if is_complex_input else np.float64
                    
                    try:
                        array = create_test_array(shape, dtype, is_complex=is_complex_input)
                    except MemoryError:
                        print(f"Memory error creating array of shape {shape}")
                        continue
                    
                    # Get function
                    module = implementation["module"]
                    func_name = transform_type["func_name"]
                    func = getattr(module, func_name)
                    
                    # Print info
                    print(f"Testing {implementation['name']} {transform_type['name']} "
                          f"on {dimension['name']} array of size {size}")
                    
                    try:
                        # Reference time with single thread
                        impl_kwargs = implementation["kwargs"].copy()
                        impl_kwargs["threads"] = 1
                        
                        # Setup implementation
                        implementation["setup"]()
                        
                        # Run single-thread benchmark
                        _ = func(array.copy(), **impl_kwargs)  # Warm-up
                        
                        start_time = time.time()
                        for _ in range(config['n_runs']):
                            array_copy = array.copy()
                            _ = func(array_copy, **impl_kwargs)
                        single_thread_time = (time.time() - start_time) / config['n_runs']
                        
                        print(f"  1 thread: {single_thread_time:.6f}s")
                        
                        # Test each thread count
                        for thread_count in config['thread_counts']:
                            if thread_count == 1:
                                # Already measured
                                current_time = single_thread_time
                                speedup = 1.0
                                efficiency = 1.0
                            else:
                                # Update thread count
                                impl_kwargs["threads"] = thread_count
                                
                                # Run multi-thread benchmark
                                _ = func(array.copy(), **impl_kwargs)  # Warm-up
                                
                                start_time = time.time()
                                for _ in range(config['n_runs']):
                                    array_copy = array.copy()
                                    _ = func(array_copy, **impl_kwargs)
                                current_time = (time.time() - start_time) / config['n_runs']
                                
                                # Calculate speedup and parallel efficiency
                                speedup = single_thread_time / current_time
                                efficiency = speedup / thread_count
                                
                                print(f"  {thread_count} threads: {current_time:.6f}s, "
                                      f"speedup: {speedup:.2f}x, efficiency: {efficiency:.2f}")
                            
                            # Record result
                            results.append({
                                "implementation": implementation['name'],
                                "transform_type": transform_type['name'],
                                "dimension": dimension['name'],
                                "size": size,
                                "thread_count": thread_count,
                                "time": current_time,
                                "speedup": speedup,
                                "efficiency": efficiency
                            })
                    except Exception as e:
                        print(f"  Error: {str(e)}")
                    finally:
                        # Cleanup implementation
                        implementation["cleanup"]()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    results_file = os.path.join(results_dir, f"{config['name'].replace(' ', '_').lower()}.csv")
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    return df

# Update planning strategy benchmark
def run_planning_strategy_benchmark(config):
    """
    Benchmark the impact of different planning strategies.
    
    Args:
        config: Dictionary with benchmark configuration
    
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    print("\n" + "=" * 50)
    print(f"RUNNING PLANNING STRATEGY BENCHMARK: {config['name']}")
    print("=" * 50)
    
    # Only test BetterFFTW and PyFFTW
    implementations = [impl for impl in config['implementations'] 
                      if impl['name'] not in ['NumPy FFT', 'SciPy FFT']]
    
    # Planning strategies to test
    planning_strategies = [
        "FFTW_ESTIMATE",
        "FFTW_MEASURE",
        "FFTW_PATIENT",
        "FFTW_EXHAUSTIVE"
    ]
    
    # Iterate over implementations, transform types, etc.
    for implementation in implementations:
        for transform_type in config['transform_types']:
            for size in config['sizes']:
                for dimension in config['dimensions']:
                    # Check compatibility
                    transform_name = transform_type['func_name']
                    dim_name = dimension['name']
                    
                    if not is_compatible(transform_name, dim_name, size):
                        print(f"Skipping {transform_type['name']} on {dim_name} array (incompatible dimensions)")
                        continue
                    
                    # Calculate shape based on dimension
                    shape = dimension['shape_func'](size)
                    
                    # Create test array
                    is_complex_input = transform_type['func_name'] in [
                        'ifft', 'ifft2', 'ifftn', 'irfft', 'irfft2', 'irfftn'
                    ]
                    dtype = np.complex128 if is_complex_input else np.float64
                    
                    try:
                        array = create_test_array(shape, dtype, is_complex=is_complex_input)
                    except MemoryError:
                        print(f"Memory error creating array of shape {shape}")
                        continue
                    
                    # Get function
                    module = implementation["module"]
                    func_name = transform_type["func_name"]
                    func = getattr(module, func_name)
                    
                    # Print info
                    print(f"Testing {implementation['name']} {transform_type['name']} "
                          f"on {dimension['name']} array of size {size}")
                    
                    # Test each planning strategy
                    for strategy in planning_strategies:
                        # Skip EXHAUSTIVE for large arrays (takes too long)
                        if strategy == "FFTW_EXHAUSTIVE" and size > 1024:
                            print(f"  Skipping {strategy} for size {size} (would take too long)")
                            continue
                        
                        try:
                            # Setup implementation
                            implementation["setup"]()
                            
                            # Set planning strategy
                            impl_kwargs = implementation["kwargs"].copy()
                            if "PyFFTW" in implementation['name']:
                                impl_kwargs["planner_effort"] = strategy
                            else:
                                impl_kwargs["planner"] = strategy
                            
                            # Measure planning time
                            gc.collect()
                            plan_start_time = time.time()
                            _ = func(array.copy(), **impl_kwargs)  # First call includes planning
                            planning_time = time.time() - plan_start_time
                            
                            # Measure execution time (after planning)
                            exec_times = []
                            for _ in range(config['n_runs']):
                                array_copy = array.copy()
                                gc.collect()
                                
                                start_time = time.time()
                                _ = func(array_copy, **impl_kwargs)
                                end_time = time.time()
                                
                                exec_times.append(end_time - start_time)
                            
                            # Calculate statistics
                            exec_median = sorted(exec_times)[len(exec_times) // 2]
                            
                            # Record result
                            results.append({
                                "implementation": implementation['name'],
                                "transform_type": transform_type['name'],
                                "dimension": dimension['name'],
                                "size": size,
                                "planning_strategy": strategy,
                                "planning_time": planning_time,
                                "execution_time": exec_median,
                                "total_time": planning_time + (exec_median * config['n_runs'])
                            })
                            
                            print(f"  {strategy}: planning={planning_time:.6f}s, "
                                  f"execution={exec_median:.6f}s")
                        except Exception as e:
                            print(f"  Error: {str(e)}")
                        finally:
                            # Cleanup implementation
                            implementation["cleanup"]()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    results_file = os.path.join(results_dir, f"{config['name'].replace(' ', '_').lower()}.csv")
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    return df

def run_dtype_benchmark(config):
    """
    Benchmark performance across different data types.
    
    Args:
        config: Dictionary with benchmark configuration
    
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    print("\n" + "=" * 50)
    print(f"RUNNING DATA TYPE BENCHMARK: {config['name']}")
    print("=" * 50)
    
    # Data types to test
    dtypes = [
        {"name": "float32", "complex_dtype": np.complex64, "real_dtype": np.float32},
        {"name": "float64", "complex_dtype": np.complex128, "real_dtype": np.float64}
    ]
    
    # Iterate over all combinations
    for implementation in config['implementations']:
        for transform_type in config['transform_types']:
            for size in config['sizes']:
                for dimension in config['dimensions']:
                    # Check dimension compatibility with transform
                    transform_name = transform_type['func_name']
                    dim_name = dimension['name']
                    
                    # Skip incompatible dimension-transform combinations
                    if transform_name in ['fft2', 'ifft2', 'rfft2', 'irfft2'] and dim_name == '1D':
                        print(f"Skipping {transform_type['name']} on {dim_name} array (incompatible dimensions)")
                        continue
                        
                    if transform_name in ['fftn', 'ifftn', 'rfftn', 'irfftn'] and dim_name == '1D' and size < 1000:
                        # fftn works on 1D but is inefficient - only test for large arrays where the difference matters
                        print(f"Skipping {transform_type['name']} on {dim_name} array (inefficient combination)")
                        continue
                    
                    for dtype_info in dtypes:
                        # Calculate shape based on dimension
                        shape = dimension['shape_func'](size)
                        
                        # Determine if complex input is required
                        is_complex_input = transform_type['func_name'] in [
                            'ifft', 'ifft2', 'ifftn', 'irfft', 'irfft2', 'irfftn'
                        ]
                        
                        # Determine dtype
                        dtype = dtype_info['complex_dtype'] if is_complex_input else dtype_info['real_dtype']
                        
                        # Create test array
                        array = create_test_array(shape, dtype, is_complex=is_complex_input)
                        
                        # Get function
                        module = implementation["module"]
                        func_name = transform_type["func_name"]
                        func = getattr(module, func_name)
                        impl_kwargs = implementation["kwargs"].copy()
                        
                        # Print info
                        print(f"Testing {implementation['name']} {transform_type['name']} "
                              f"on {dimension['name']} array of size {size} ({dtype_info['name']})")
                        
                        try:
                            # Setup implementation
                            implementation["setup"]()
                            
                            # Run benchmark
                            _ = func(array.copy(), **impl_kwargs)  # Warm-up
                            
                            times = []
                            for _ in range(config['n_runs']):
                                array_copy = array.copy()
                                gc.collect()
                                
                                start_time = time.time()
                                result = func(array_copy, **impl_kwargs)
                                end_time = time.time()
                                
                                times.append(end_time - start_time)
                            
                            # Calculate statistics
                            median_time = sorted(times)[len(times) // 2]
                            
                            # Record result
                            results.append({
                                "implementation": implementation['name'],
                                "transform_type": transform_type['name'],
                                "dimension": dimension['name'],
                                "size": size,
                                "dtype": dtype_info['name'],
                                "time": median_time
                            })
                            
                            print(f"  Median time: {median_time:.6f}s")
                            
                        except Exception as e:
                            print(f"  Error: {str(e)}")
                        finally:
                            # Cleanup implementation
                            implementation["cleanup"]()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    results_file = os.path.join(results_dir, f"{config['name'].replace(' ', '_').lower()}.csv")
    df.to_csv(results_file, index=False)
    print(f"\nResults saved to {results_file}")
    
    return df

def plot_benchmark_results(results, plot_name, x_axis='size', y_axis='median_time', 
                          group_by='implementation', log_scale=True,
                          x_label=None, y_label=None, title=None):
    """
    Plot benchmark results.
    
    Args:
        results: DataFrame with results
        plot_name: Name for the output file
        x_axis: Column to use for x-axis
        y_axis: Column to use for y-axis
        group_by: Column to group by (for different lines)
        log_scale: Whether to use logarithmic scale
        x_label: Label for x-axis
        y_label: Label for y-axis
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Use Seaborn for better aesthetics
    sns.set_style("whitegrid")
    
    # Group data for plotting
    groups = results.groupby(group_by)
    
    # Create line plot
    for name, group in groups:
        plt.plot(group[x_axis], group[y_axis], marker='o', linestyle='-', label=name)
    
    # Set log scale if requested
    if log_scale:
        plt.xscale('log', base=2)
        plt.yscale('log')
    
    # Set labels
    plt.xlabel(x_label or x_axis)
    plt.ylabel(y_label or y_axis)
    plt.title(title or f"Benchmark Results: {y_axis} vs {x_axis}")
    
    # Add legend
    plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_file = os.path.join(results_dir, f"{plot_name}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")
    
    plt.close()

# ===============================================================================
# Run the benchmarks
# ===============================================================================

if __name__ == "__main__":
    # Check if NumPy is linked against MKL
    if is_numpy_using_mkl():
        print("NumPy appears to be using Intel MKL")
    else:
        print("NumPy does not appear to be using Intel MKL")
    
    # Subset of implementations for faster benchmarks
    fast_implementations = [
        implementations["numpy"],
        implementations["scipy"],
        implementations["betterfftw_default"]
    ]
    
    # Subset of transform types for faster benchmarks
    fast_transforms = [
        transform_types["fft"],
        transform_types["fft2"],
        transform_types["fftn"]
    ]
    
    # Benchmarks to run (comment out benchmarks you don't want to run)
    benchmarks = [
        # 1. Basic performance comparison - small to medium sizes
        {
            "name": "Basic Performance (Small)", 
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["fft"]],  # 1D transform
            "sizes": small_sizes,
            "dimensions": [dimensions["1d"]],  # 1D arrays
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 5
        },
        
        # 2. 2D transform performance - small to medium sizes
        {
            "name": "2D Transform Performance (Small)",
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["fft2"]],  # 2D transform
            "sizes": small_sizes,
            "dimensions": [dimensions["2d_square"], dimensions["2d_rect"]],  # 2D arrays
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 5
        },
        
        # 3. Basic performance comparison - medium to large sizes
        {
            "name": "Basic Performance (Medium)",
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["fft"]],  # 1D transform
            "sizes": medium_sizes,
            "dimensions": [dimensions["1d"]],  # 1D arrays
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 5
        },
        
        # 4. 2D transform performance - medium to large sizes
        {
            "name": "2D Transform Performance (Medium)",
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["fft2"]],  # 2D transform
            "sizes": medium_sizes,
            "dimensions": [dimensions["2d_square"], dimensions["2d_rect"]],  # 2D arrays
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 5
        },
        
        # 5. Basic performance comparison - large sizes
        {
            "name": "Basic Performance (Large)",
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["fft"]],  # 1D transform
            "sizes": large_sizes,
            "dimensions": [dimensions["1d"]],  # 1D arrays
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 3
        },
        
        # 6. 2D transform performance - large sizes
        {
            "name": "2D Transform Performance (Large)",
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["fft2"]],  # 2D transform
            "sizes": large_sizes,
            "dimensions": [dimensions["2d_square"]],  # 2D arrays
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 3
        },
        
        # 7. Real FFT performance - 1D
        {
            "name": "Real FFT Performance (1D)",
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["rfft"]],  # Real 1D transform
            "sizes": medium_sizes,
            "dimensions": [dimensions["1d"]],  # 1D arrays
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 5
        },
        
        # 8. Real FFT performance - 2D
        {
            "name": "Real FFT Performance (2D)",
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["rfft2"]],  # Real 2D transform
            "sizes": medium_sizes,
            "dimensions": [dimensions["2d_square"]],  # 2D arrays
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 5
        },
        
        # 9. 3D FFT performance
        {
            "name": "3D FFT Performance",
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["fftn"]],  # N-D transform
            "sizes": small_sizes + [128],  # Only small sizes for 3D
            "dimensions": [dimensions["3d"]],  # 3D arrays
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 3
        },
        
        # 10. Threading performance
        {
            "name": "Threading Scaling",
            "implementations": [implementations["betterfftw_default"], implementations["pyfftw_measure"]],
            "transform_types": [transform_types["fft2"]],  # 2D transform
            "sizes": [512, 1024, 2048],
            "dimensions": [dimensions["2d_square"]],  # 2D arrays
            "thread_counts": thread_counts,
            "n_runs": 3
        },
        
        # 11. Non-power-of-two performance
        {
            "name": "Non-Power-of-Two Performance",
            "implementations": [implementations["numpy"], implementations["betterfftw_default"]],
            "transform_types": [transform_types["fft"]],  # 1D transform only
            "sizes": [128, 256, 512, 1024] + [100, 101, 127, 253, 509, 1001],  # Powers of 2 + non-powers
            "n_runs": 5
        },
        
        # 12. Repeated transform performance - 1D
        {
            "name": "Repeated Transform Performance (1D)",
            "implementations": [implementations["numpy"], implementations["betterfftw_default"]],
            "transform_types": [transform_types["fft"]],  # 1D transform
            "sizes": [1024, 2048],
            "dimensions": [dimensions["1d"]],  # 1D arrays
            "repetition_counts": [1, 2, 5, 10, 20, 50, 100],
            "thread_count": 4
        },
        
        # 13. Repeated transform performance - 2D
        {
            "name": "Repeated Transform Performance (2D)",
            "implementations": [implementations["numpy"], implementations["betterfftw_default"]],
            "transform_types": [transform_types["fft2"]],  # 2D transform
            "sizes": [512, 1024],
            "dimensions": [dimensions["2d_square"]],  # 2D arrays
            "repetition_counts": [1, 2, 5, 10, 20, 50, 100],
            "thread_count": 4
        },
        
        # 14. Planning strategy comparison - 1D
        {
            "name": "Planning Strategy Comparison (1D)",
            "implementations": [implementations["betterfftw_default"], implementations["pyfftw_estimate"]],
            "transform_types": [transform_types["fft"]],  # 1D transform
            "sizes": [128, 256, 512, 1024],
            "dimensions": [dimensions["1d"]],  # 1D arrays
            "n_runs": 5
        },
        
        # 15. Planning strategy comparison - 2D
        {
            "name": "Planning Strategy Comparison (2D)",
            "implementations": [implementations["betterfftw_default"], implementations["pyfftw_estimate"]],
            "transform_types": [transform_types["fft2"]],  # 2D transform
            "sizes": [128, 256, 512, 1024],
            "dimensions": [dimensions["2d_square"]],  # 2D arrays
            "n_runs": 5
        },
        
        # 16. Data type comparison - 1D
        {
            "name": "Data Type Comparison (1D)",
            "implementations": [implementations["numpy"], implementations["betterfftw_default"]],
            "transform_types": [transform_types["fft"]],  # 1D transform
            "sizes": [512, 1024, 2048],
            "dimensions": [dimensions["1d"]],  # 1D arrays
            "n_runs": 5
        },
        
        # 17. Data type comparison - 2D
        {
            "name": "Data Type Comparison (2D)",
            "implementations": [implementations["numpy"], implementations["betterfftw_default"]],
            "transform_types": [transform_types["fft2"]],  # 2D transform
            "sizes": [128, 256, 512],
            "dimensions": [dimensions["2d_square"]],  # 2D arrays
            "n_runs": 5
        }
    ]
    # Run each benchmark
    results = {}
    for benchmark in benchmarks:
        name = benchmark["name"]
        
        if "Basic Performance" in name or "Real FFT Performance" in name or "3D FFT Performance" in name:
            results[name] = run_benchmark_suite(benchmark)
        elif name == "Threading Scaling":
            results[name] = run_threading_scaling_benchmark(benchmark)
        elif name == "Non-Power-of-Two Performance":
            results[name] = run_non_power_of_two_benchmark(benchmark)
        elif name == "Repeated Transform Performance":
            results[name] = run_repeated_transform_benchmark(benchmark)
        elif name == "Planning Strategy Comparison":
            results[name] = run_planning_strategy_benchmark(benchmark)
        elif name == "Data Type Comparison":
            results[name] = run_dtype_benchmark(benchmark)
    
    # Create plots for each benchmark
    for name, df in results.items():
        if "Basic Performance" in name:
            # Plot time vs size for each implementation
            plot_benchmark_results(
                df[df['transform_type'].str.contains('1D')],
                f"{name.replace(' ', '_').lower()}_1d",
                x_axis='input_size',
                y_axis='median_time',
                group_by='implementation',
                title=f"{name} - 1D FFT",
                x_label="Array Size",
                y_label="Time (seconds)"
            )
            
            # Plot time vs size for 2D transforms
            plot_benchmark_results(
                df[df['transform_type'].str.contains('2D')],
                f"{name.replace(' ', '_').lower()}_2d",
                x_axis='input_size',
                y_axis='median_time',
                group_by='implementation',
                title=f"{name} - 2D FFT",
                x_label="Array Size (each dimension)",
                y_label="Time (seconds)"
            )
            
            # Plot throughput
            plot_benchmark_results(
                df,
                f"{name.replace(' ', '_').lower()}_throughput",
                x_axis='input_size',
                y_axis='throughput_gflops',
                group_by='implementation',
                title=f"{name} - Throughput",
                x_label="Array Size",
                y_label="Throughput (GFLOPS)"
            )
        
        elif name == "Real FFT Performance":
            # Plot time vs size for real FFTs
            plot_benchmark_results(
                df,
                f"{name.replace(' ', '_').lower()}",
                x_axis='input_size',
                y_axis='median_time',
                group_by='implementation',
                title=name,
                x_label="Array Size",
                y_label="Time (seconds)"
            )
            
            # Compare real vs complex FFT (if data available)
            if "Basic Performance" in results:
                # Combine data
                real_data = df[df['transform_type'].str.contains('Real')]
                complex_data = results["Basic Performance (Medium)"][
                    results["Basic Performance (Medium)"]["transform_type"].str.contains('Complex')
                ]
                
                # Add a type column
                real_data['fft_type'] = 'Real FFT'
                complex_data['fft_type'] = 'Complex FFT'
                
                # Combine
                combined = pd.concat([real_data, complex_data])
                
                # Plot comparison
                plot_benchmark_results(
                    combined,
                    "real_vs_complex_comparison",
                    x_axis='input_size',
                    y_axis='median_time',
                    group_by='fft_type',
                    title="Real vs Complex FFT Performance",
                    x_label="Array Size",
                    y_label="Time (seconds)"
                )
        
        elif name == "3D FFT Performance":
            # Plot time vs size for 3D FFTs
            plot_benchmark_results(
                df,
                f"{name.replace(' ', '_').lower()}",
                x_axis='input_size',
                y_axis='median_time',
                group_by='implementation',
                title=name,
                x_label="Array Size (per dimension)",
                y_label="Time (seconds)"
            )
        
        elif name == "Threading Scaling":
            # Group by size and implementation
            for size in df['size'].unique():
                # Filter for this size
                size_data = df[df['size'] == size]
                
                # Plot speedup vs thread count
                plt.figure(figsize=(10, 6))
                
                for impl in size_data['implementation'].unique():
                    impl_data = size_data[size_data['implementation'] == impl]
                    plt.plot(impl_data['thread_count'], impl_data['speedup'], 
                            marker='o', label=impl)
                
                # Add ideal scaling line
                max_threads = size_data['thread_count'].max()
                plt.plot([1, max_threads], [1, max_threads], 'k--', label='Ideal Scaling')
                
                plt.xlabel('Thread Count')
                plt.ylabel('Speedup')
                plt.title(f'Threading Speedup for Size {size}')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Save plot
                plt.savefig(os.path.join(results_dir, f"threading_speedup_size_{size}.png"), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Plot efficiency vs thread count
                plt.figure(figsize=(10, 6))
                
                for impl in size_data['implementation'].unique():
                    impl_data = size_data[size_data['implementation'] == impl]
                    plt.plot(impl_data['thread_count'], impl_data['efficiency'], 
                            marker='o', label=impl)
                
                # Add ideal efficiency line
                plt.axhline(y=1.0, color='k', linestyle='--', label='Ideal Efficiency')
                
                plt.xlabel('Thread Count')
                plt.ylabel('Parallel Efficiency')
                plt.title(f'Threading Efficiency for Size {size}')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Save plot
                plt.savefig(os.path.join(results_dir, f"threading_efficiency_size_{size}.png"), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        elif name == "Non-Power-of-Two Performance":
            # Group by size type
            plt.figure(figsize=(12, 8))
            
            # Calculate the ratio of BetterFFTW to NumPy performance
            pivot = df.pivot_table(
                index=['size', 'size_type', 'factor_str'],
                columns='implementation',
                values='median_time'
            ).reset_index()
            
            # Add a ratio column
            pivot['speedup'] = pivot['NumPy FFT'] / pivot['BetterFFTW (Default)']
            
            # Group by size type
            for size_type in pivot['size_type'].unique():
                type_data = pivot[pivot['size_type'] == size_type]
                plt.scatter(type_data['size'], type_data['speedup'], 
                           label=size_type.replace('_', ' ').title())
            
            plt.axhline(y=1.0, color='k', linestyle='--', label='Equal Performance')
            
            plt.xscale('log')
            plt.xlabel('Array Size')
            plt.ylabel('Speedup (NumPy Time / BetterFFTW Time)')
            plt.title('BetterFFTW Speedup for Different Array Sizes')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            plt.savefig(os.path.join(results_dir, "non_power_of_two_speedup.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also create a direct comparison plot
            plt.figure(figsize=(12, 8))
            
            for impl in df['implementation'].unique():
                for size_type in df['size_type'].unique():
                    data = df[(df['implementation'] == impl) & (df['size_type'] == size_type)]
                    plt.scatter(data['size'], data['median_time'], 
                               label=f"{impl} - {size_type.replace('_', ' ').title()}")
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Array Size')
            plt.ylabel('Time (seconds)')
            plt.title('Performance for Different Array Size Types')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            plt.savefig(os.path.join(results_dir, "non_power_of_two_comparison.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        elif name == "Repeated Transform Performance":
            # Plot time per transform vs repetition count
            plt.figure(figsize=(12, 8))
            
            for impl in df['implementation'].unique():
                for size in df['size'].unique():
                    data = df[(df['implementation'] == impl) & (df['size'] == size)]
                    plt.plot(data['repetitions'], data['time_per_transform'], 
                            marker='o', label=f"{impl} - Size {size}")
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Number of Repetitions')
            plt.ylabel('Time per Transform (seconds)')
            plt.title('Impact of Repeated Transforms')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            plt.savefig(os.path.join(results_dir, "repeated_transform_performance.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate and plot speedup vs first call
            plt.figure(figsize=(12, 8))
            
            for impl in df['implementation'].unique():
                for size in df['size'].unique():
                    data = df[(df['implementation'] == impl) & (df['size'] == size)]
                    first_time = data.iloc[0]['time_per_transform']
                    speedup = [first_time / t for t in data['time_per_transform']]
                    plt.plot(data['repetitions'], speedup, 
                            marker='o', label=f"{impl} - Size {size}")
            
            plt.xscale('log')
            plt.axhline(y=1.0, color='k', linestyle='--', label='First Call')
            plt.xlabel('Number of Repetitions')
            plt.ylabel('Speedup vs First Call')
            plt.title('Performance Improvement with Repeated Transforms')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            plt.savefig(os.path.join(results_dir, "repeated_transform_speedup.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        elif name == "Planning Strategy Comparison":
            # Plot planning time vs strategy
            plt.figure(figsize=(14, 8))
            
            for impl in df['implementation'].unique():
                for size in df['size'].unique():
                    data = df[(df['implementation'] == impl) & (df['size'] == size)]
                    plt.plot(data['planning_strategy'], data['planning_time'], 
                            marker='o', label=f"{impl} - Size {size}")
            
            plt.yscale('log')
            plt.xlabel('Planning Strategy')
            plt.ylabel('Planning Time (seconds)')
            plt.title('Planning Time for Different Strategies')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(results_dir, "planning_strategy_planning_time.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot execution time vs strategy
            plt.figure(figsize=(14, 8))
            
            for impl in df['implementation'].unique():
                for size in df['size'].unique():
                    data = df[(df['implementation'] == impl) & (df['size'] == size)]
                    plt.plot(data['planning_strategy'], data['execution_time'], 
                            marker='o', label=f"{impl} - Size {size}")
            
            plt.yscale('log')
            plt.xlabel('Planning Strategy')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Execution Time for Different Planning Strategies')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(results_dir, "planning_strategy_execution_time.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot break-even point: at what repetition count does more planning pay off?
            plt.figure(figsize=(14, 8))
            
            for impl in df['implementation'].unique():
                for size in df['size'].unique():
                    data = df[(df['implementation'] == impl) & (df['size'] == size)]
                    
                    # Get baseline (ESTIMATE)
                    baseline = data[data['planning_strategy'] == 'FFTW_ESTIMATE']
                    if len(baseline) == 0:
                        continue
                        
                    baseline_planning = baseline.iloc[0]['planning_time']
                    baseline_execution = baseline.iloc[0]['execution_time']
                    
                    strategies = []
                    breakeven = []
                    
                    for _, row in data.iterrows():
                        if row['planning_strategy'] == 'FFTW_ESTIMATE':
                            continue
                            
                        planning_diff = row['planning_time'] - baseline_planning
                        execution_diff = baseline_execution - row['execution_time']
                        
                        if execution_diff <= 0:
                            # More expensive planning but no execution benefit
                            breakeven_point = float('inf')
                        else:
                            # Calculate how many repeats for planning to pay off
                            breakeven_point = planning_diff / execution_diff
                        
                        strategies.append(row['planning_strategy'])
                        breakeven.append(breakeven_point)
                    
                    plt.plot(strategies, breakeven, 
                            marker='o', label=f"{impl} - Size {size}")
            
            plt.yscale('log')
            plt.xlabel('Planning Strategy')
            plt.ylabel('Break-even Point (number of executions)')
            plt.title('Number of Executions for Planning Strategy to Pay Off')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            # Save plot
            plt.savefig(os.path.join(results_dir, "planning_strategy_breakeven.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        elif name == "Data Type Comparison":
            # Plot performance by data type
            plt.figure(figsize=(12, 8))
            
            for impl in df['implementation'].unique():
                for transform in df['transform_type'].unique():
                    for dim in df['dimension'].unique():
                        data = df[(df['implementation'] == impl) & 
                                  (df['transform_type'] == transform) & 
                                  (df['dimension'] == dim)]
                        
                        if len(data) == 0:
                            continue
                            
                        # Group by size and dtype
                        pivot = data.pivot_table(
                            index='size',
                            columns='dtype',
                            values='time'
                        ).reset_index()
                        
                        # Calculate ratio float32/float64
                        if 'float32' in pivot.columns and 'float64' in pivot.columns:
                            pivot['ratio'] = pivot['float64'] / pivot['float32']
                            
                            plt.plot(pivot['size'], pivot['ratio'], 
                                    marker='o', label=f"{impl} - {transform} - {dim}")
            
            plt.axhline(y=1.0, color='k', linestyle='--', label='Equal Performance')
            plt.axhline(y=2.0, color='k', linestyle=':', label='float64 Twice as Slow')
            
            plt.xscale('log', base=2)
            plt.xlabel('Array Size')
            plt.ylabel('Performance Ratio (float64 time / float32 time)')
            plt.title('Performance Impact of Data Type')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            plt.savefig(os.path.join(results_dir, "data_type_comparison.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    print("\nAll benchmarks complete!")
    print(f"Results saved to {results_dir}/")