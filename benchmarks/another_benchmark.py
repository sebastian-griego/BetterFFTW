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
- Realistic application scenarios

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
from functools import wraps
import tempfile
import shutil
import subprocess
import sys
import warnings

# Create results directory
results_dir = "benchmark_results"
os.makedirs(results_dir, exist_ok=True)

# Try to import cpuinfo for better CPU detection
try:
    import cpuinfo
    has_cpuinfo = True
except ImportError:
    has_cpuinfo = False
    print("Warning: py-cpuinfo not installed. CPU details will be limited.")
    print("To install: pip install py-cpuinfo")

# Get system information
def get_system_info():
    """Get detailed system information for benchmark context."""
    
    # Get CPU info
    if has_cpuinfo:
        cpu_info = cpuinfo.get_cpu_info()
        cpu_model = cpu_info.get("brand_raw", platform.processor())
        cpu_arch = cpu_info.get("arch", platform.machine())
        
        # Detect SIMD capabilities
        simd_capabilities = []
        cpu_flags = cpu_info.get("flags", [])
        if isinstance(cpu_flags, str):
            cpu_flags = cpu_flags.split()
            
        for feature in ["avx", "avx2", "avx512f", "sse4_1", "sse4_2"]:
            if feature in cpu_flags:
                simd_capabilities.append(feature)
    else:
        cpu_model = platform.processor()
        cpu_arch = platform.machine()
        simd_capabilities = ["unknown"]
    
    # Get memory info
    mem = psutil.virtual_memory()
    mem_total = mem.total / (1024**3)  # GB
    
    # Get cache information (Linux only)
    cache_info = {}
    if sys.platform.startswith('linux'):
        try:
            # Try to get cache info from lscpu
            output = subprocess.check_output(['lscpu']).decode('utf-8')
            for line in output.split('\n'):
                if 'cache' in line.lower():
                    parts = line.split(':')
                    if len(parts) == 2:
                        cache_info[parts[0].strip()] = parts[1].strip()
        except:
            pass
    
    # Framework versions
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "processor": cpu_model,
        "processor_architecture": cpu_arch,
        "simd_capabilities": simd_capabilities,
        "cpu_count": os.cpu_count(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_gb": round(mem_total, 2),
        "memory_available_gb": round(mem.available / (1024**3), 2),
        "cache_info": cache_info,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pyfftw_version": pyfftw.__version__,
        "betterfftw_version": betterfftw.__version__,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Estimate memory bandwidth (simple test)
    try:
        size = 1024 * 1024 * 100  # 100 MB array
        a = np.ones(size, dtype=np.float64)
        b = np.ones(size, dtype=np.float64)
        
        # Time a simple operation that should be memory-bound
        start = time.time()
        for _ in range(5):
            c = a + b
            c[0] = 0  # Ensure it's not optimized away
        end = time.time()
        
        # Calculate bandwidth (bytes/sec)
        bytes_processed = 5 * 3 * 8 * size  # read a, read b, write c, 8 bytes per element, 5 iterations
        bw = bytes_processed / (end - start) / (1024**3)  # GB/s
        info["estimated_memory_bandwidth_gb_s"] = round(bw, 2)
    except:
        info["estimated_memory_bandwidth_gb_s"] = "unknown"
    
    return info

# Try to detect if NumPy is using MKL
def is_numpy_using_mkl():
    """Try to determine if NumPy is using Intel MKL."""
    using_mkl = False
    
    # Check configuration 
    np_config = str(np.__config__).lower()
    if 'mkl' in np_config or 'intel' in np_config:
        using_mkl = True
    
    # Check loaded libraries
    if not using_mkl and sys.platform == 'linux':
        try:
            # List loaded libraries for the Python process
            maps_file = f"/proc/{os.getpid()}/maps"
            if os.path.exists(maps_file):
                with open(maps_file, 'r') as f:
                    for line in f:
                        if 'mkl' in line.lower() or 'intel' in line.lower():
                            using_mkl = True
                            break
        except:
            pass
    
    # Try to import mkl module
    if not using_mkl:
        try:
            import mkl
            using_mkl = True
        except ImportError:
            pass
    
    # Finally, try the dot product test (MKL is much faster for large dot products)
    if not using_mkl:
        try:
            # Generate large arrays for dot product
            n = 10000
            a = np.random.random((n, n)).astype(np.float64)
            b = np.random.random((n, n)).astype(np.float64)
            
            # Time the dot product
            start = time.time()
            np.dot(a, b)
            end = time.time()
            
            # If the dot product is very fast for this size, it's likely using MKL
            if end - start < 1.0:  # Threshold depends on hardware
                using_mkl = True
        except:
            pass
            
    return using_mkl

# Record system info
system_info = get_system_info()
system_info["numpy_mkl"] = is_numpy_using_mkl()

# Print system information
print("=" * 50)
print("SYSTEM INFORMATION")
print("=" * 50)
for key, value in system_info.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    else:
        print(f"{key}: {value}")
print("=" * 50)

# Save system info to file
with open(os.path.join(results_dir, "system_info.txt"), "w") as f:
    for key, value in system_info.items():
        if isinstance(value, dict):
            f.write(f"{key}:\n")
            for k, v in value.items():
                f.write(f"  {k}: {v}\n")
        else:
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
    "pyfftw_patient": {
        "name": "PyFFTW (PATIENT)",
        "module": pyfftw.interfaces.numpy_fft,
        "setup": lambda: pyfftw.interfaces.cache.enable(),
        "cleanup": lambda: pyfftw.interfaces.cache.disable(),
        "kwargs": {"planner_effort": "FFTW_PATIENT"}
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
    },
    "betterfftw_exhaustive": {
        "name": "BetterFFTW (EXHAUSTIVE)",
        "module": betterfftw,
        "setup": lambda: betterfftw.clear_cache(),
        "cleanup": lambda: None,
        "kwargs": {"planner": "FFTW_EXHAUSTIVE"}
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

# Array sizes to test (improved with more realistic sizes)
small_sizes = [64, 256, 512, 1024]                     # Small arrays
medium_sizes = [2048, 4096, 8192]                      # Medium arrays
large_sizes = [16384, 32768, 65536]                    # Large arrays
very_large_sizes = [131072, 262144, 524288]            # Very large arrays
# Non-power-of-2 sizes for testing prime factors and mixed sizes
non_power_of_2_sizes = [100, 160, 486, 729, 1000, 1331, 2187, 10000]

# Array dimensions
dimensions = {
    "1d": {"name": "1D", "shape_func": lambda n: (n,)},
    "2d_square": {"name": "2D Square", "shape_func": lambda n: (n, n)},
    "2d_rect": {"name": "2D Rectangle", "shape_func": lambda n: (n, n//2)},
    "3d": {"name": "3D", "shape_func": lambda n: (n, n, n)},
    # Realistic image and volume dimensions for practical benchmarks
    "2d_image_hd": {"name": "2D HD Image", "shape_func": lambda n: (1080, 1920)},
    "2d_image_4k": {"name": "2D 4K Image", "shape_func": lambda n: (2160, 3840)}, 
    "3d_volume_small": {"name": "3D Small Volume", "shape_func": lambda n: (64, 64, 64)},
    "3d_volume_medical": {"name": "3D Medical", "shape_func": lambda n: (256, 256, 128)}
}

# Data types
dtypes = {
    "float32": {"name": "float32", "complex_dtype": np.complex64, "real_dtype": np.float32},
    "float64": {"name": "float64", "complex_dtype": np.complex128, "real_dtype": np.float64}
}

# Thread counts to test
max_threads = os.cpu_count()
thread_counts = [1]
if max_threads >= 2:
    thread_counts.append(2)
if max_threads >= 4:
    thread_counts.append(4)
if max_threads >= 8:
    thread_counts.append(8)
if max_threads > 8:
    thread_counts.append(max_threads // 2)
    thread_counts.append(max_threads)
thread_counts = sorted(list(set(thread_counts)))  # Remove duplicates

# Function to create test array
def create_test_array(shape, dtype, is_complex=False, out_of_cache=False):
    """
    Create a test array with specified shape and dtype.
    
    Args:
        shape: Tuple of dimensions
        dtype: NumPy dtype
        is_complex: Whether to create a complex array
        out_of_cache: Force array to be out of cache
        
    Returns:
        NumPy array
    """
    if out_of_cache and hasattr(create_test_array, 'cache_buster'):
        # Touch the cache buster array to flush cache
        create_test_array.cache_buster[:] = np.random.random(create_test_array.cache_buster.shape)
    
    if is_complex:
        real_part = np.random.random(shape).astype(dtype)
        imag_part = np.random.random(shape).astype(dtype)
        return real_part + 1j * imag_part
    else:
        return np.random.random(shape).astype(dtype)

# Initialize a large array to flush cache when needed
def init_cache_buster():
    """Create a large array to flush CPU cache when needed."""
    # Estimate cache size - aim for 2x the largest cache
    est_cache_size = 32 * 1024 * 1024  # Default: assume 32MB cache
    
    # Try to get actual cache size from system info
    if 'cache_info' in system_info:
        for cache_name, cache_size in system_info['cache_info'].items():
            if 'L3' in cache_name and 'size' in cache_name.lower():
                try:
                    size_str = cache_size.lower().replace('k', '').replace('m', '000').replace('g', '000000')
                    size_bytes = int(size_str) * 1024
                    est_cache_size = size_bytes * 2  # Use 2x the cache size
                except:
                    pass
    
    # Create the array
    array_size = est_cache_size // 8  # 8 bytes per float64
    create_test_array.cache_buster = np.zeros(array_size, dtype=np.float64)
    print(f"Initialized cache buster array: {array_size} elements ({est_cache_size / (1024*1024):.1f} MB)")

# Call the function to initialize the cache buster
init_cache_buster()

# Function to get current memory usage
def get_memory_usage():
    """Get current memory usage of the process in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)

# ===============================================================================
# Benchmark functions
# ===============================================================================

def run_single_benchmark(implementation, transform_type, array, n_runs=5, threads=None, 
                         include_planning=True, out_of_cache=False, profile=False, **kwargs):
    """
    Run a single benchmark test and measure performance.
    
    Args:
        implementation: Implementation dictionary
        transform_type: Transform type dictionary
        array: Input array
        n_runs: Number of times to run the transform
        threads: Number of threads to use (for threaded implementations)
        include_planning: Whether to measure planning time
        out_of_cache: Whether to flush cache between runs
        profile: Whether to profile the execution
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
    
    # Memory usage before benchmark
    memory_before = get_memory_usage()
    
    # Setup implementation (e.g., enable caching)
    implementation["setup"]()
    
    # Track if we've warmed up
    warmed_up = False
    
    # Measure planning time (first-call overhead)
    if include_planning:
        planning_times = []
        
        for _ in range(max(1, n_runs // 2)):  # Do fewer planning runs than execution runs
            # Force cleanup between runs to measure planning overhead
            implementation["cleanup"]()
            implementation["setup"]()
            
            # Create a fresh copy for each run
            array_copy = array.copy()
            
            # Flush cache if needed
            if out_of_cache:
                create_test_array.cache_buster[:] = np.random.random(create_test_array.cache_buster.shape)
            
            # Force garbage collection
            gc.collect()
            
            # Time the first run (includes planning)
            start_time = time.time()
            result = func(array_copy, **impl_kwargs)
            end_time = time.time()
            
            planning_times.append(end_time - start_time)
            warmed_up = True
    
    # If we haven't warmed up yet, do a single warm-up run
    if not warmed_up:
        # Warm-up run (not timed) to handle any first-time overhead
        result = func(array.copy(), **impl_kwargs)
    
    # Profile if requested
    if profile:
        try:
            import cProfile
            import pstats
            from io import StringIO
            
            # Create a profile
            profiler = cProfile.Profile()
            
            # Run with profiling
            array_copy = array.copy()
            gc.collect()
            
            profiler.enable()
            result = func(array_copy, **impl_kwargs)
            profiler.disable()
            
            # Get profile data
            s = StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(30)  # top 30 functions
            profile_data = s.getvalue()
        except Exception as e:
            profile_data = f"Profiling error: {str(e)}"
    else:
        profile_data = None
    
    # Run the execution benchmark
    execution_times = []
    for _ in range(n_runs):
        # Create a copy to avoid in-place modifications affecting subsequent runs
        array_copy = array.copy()
        
        # Flush cache if needed
        if out_of_cache:
            create_test_array.cache_buster[:] = np.random.random(create_test_array.cache_buster.shape)
        
        # Force garbage collection before timing
        gc.collect()
        
        start_time = time.time()
        result = func(array_copy, **impl_kwargs)
        end_time = time.time()
        
        execution_times.append(end_time - start_time)
    
    # Calculate memory usage
    memory_after = get_memory_usage()
    memory_increase = memory_after - memory_before
    
    # Cleanup implementation (e.g., disable caching)
    implementation["cleanup"]()
    
    # Calculate statistics
    min_exec = min(execution_times)
    max_exec = max(execution_times)
    mean_exec = sum(execution_times) / len(execution_times)
    median_exec = sorted(execution_times)[len(execution_times) // 2]
    
    # Planning time statistics if measured
    if include_planning:
        min_plan = min(planning_times)
        max_plan = max(planning_times)
        mean_plan = sum(planning_times) / len(planning_times)
        median_plan = sorted(planning_times)[len(planning_times) // 2]
    else:
        min_plan = max_plan = mean_plan = median_plan = None
    
    # Get memory usage of result
    result_memory = result.nbytes / (1024 * 1024)  # in MB
    
    # Get input array shape and size
    input_shape = array.shape
    input_size = array.size
    input_memory = array.nbytes / (1024 * 1024)  # in MB
    
    # Calculate throughput metrics (operations per second, data processed per second)
    # Assume we're using median execution time for throughput
    array_size_mb = input_memory
    throughput_mb_per_s = array_size_mb / median_exec
    
    # Calculate FLOPS estimate based on FFT complexity
    flops = calculate_fft_gflops(input_shape, func_name) * 1e9  # Convert GFLOPS to FLOPS
    flops_per_s = flops / median_exec
    
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
        "memory_increase_mb": memory_increase,
        # Execution time (without planning)
        "min_exec_time": min_exec,
        "max_exec_time": max_exec,
        "mean_exec_time": mean_exec,
        "median_exec_time": median_exec,
        # Planning time (first-call overhead)
        "min_plan_time": min_plan,
        "max_plan_time": max_plan,
        "mean_plan_time": mean_plan,
        "median_plan_time": median_plan,
        # Total time (planning + execution)
        "total_median_time": (median_plan or 0) + median_exec,
        # Performance metrics
        "throughput_gflops": flops_per_s / 1e9,
        "throughput_mb_per_s": throughput_mb_per_s,
        "operations_per_s": 1.0 / median_exec,
        # Profile data
        "profile_data": profile_data,
        # Implementation details
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
    # Helper function to calculate FFT complexity for a single dimension
    def fft_complexity(n):
        # Determine if n is a power of 2
        is_power_of_2 = (n & (n-1) == 0) and n > 0
        
        if is_power_of_2:
            # Fast path: Power of 2 is 5*n*log2(n)
            return 5 * n * np.log2(n)
        else:
            # Slower path: Non-power of 2
            # Estimate as 5*n*log2(n) + additional work
            return 5 * n * np.log2(n) * 1.5
    
    # Determine dimensionality and compute complexity
    if func_name in ["fft", "ifft", "rfft", "irfft"]:
        # 1D transform
        n = shape[0]
        flops = fft_complexity(n)
        
        # Adjust for real transforms (roughly half the work)
        if "rfft" in func_name:
            flops *= 0.5
            
    elif func_name in ["fft2", "ifft2", "rfft2", "irfft2"]:
        # 2D transform
        n1, n2 = shape[0], shape[1]
        
        # 2D FFT is computed as 1D FFTs along each dimension
        row_ffts = n2 * fft_complexity(n1)  # n2 FFTs of size n1
        col_ffts = n1 * fft_complexity(n2)  # n1 FFTs of size n2
        flops = row_ffts + col_ffts
        
        # Adjust for real transforms
        if "rfft" in func_name:
            flops *= 0.6  # Real 2D is more than half due to storage pattern
            
    else:
        # N-D transform
        flops = 0
        
        # Multi-dimensional FFT is computed recursively along each dimension
        if len(shape) == 3:
            n1, n2, n3 = shape
            
            # Compute as successive 1D FFTs along each dimension
            depth_ffts = n1 * n2 * fft_complexity(n3)  # n1*n2 FFTs of size n3
            row_ffts = n1 * n3 * fft_complexity(n2)    # n1*n3 FFTs of size n2
            col_ffts = n2 * n3 * fft_complexity(n1)    # n2*n3 FFTs of size n1
            flops = depth_ffts + row_ffts + col_ffts
            
            # Adjust for real transforms
            if "rfft" in func_name:
                flops *= 0.6  # Complex pattern for real 3D transforms
        else:
            # General case for arbitrary dimensions (could be optimized further)
            n_total = np.prod(shape)
            log_term = np.log2(n_total)
            flops = 5 * n_total * log_term
            
            # Adjust for real transforms
            if "rfft" in func_name:
                flops *= 0.5
    
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
    print(f"Thread counts: {config.get('thread_counts', [1])}")
    print(f"Runs per test: {config['n_runs']}")
    print(f"Include planning time: {config.get('include_planning', True)}")
    print(f"Out-of-cache testing: {config.get('out_of_cache', False)}")
    print(f"Profiling: {config.get('profile', False)}")
    print("=" * 50)
    
    # Count total tests, accounting for skipped combinations due to dimensionality
    valid_tests = 0
    for transform_type in config['transform_types']:
        for dimension in config['dimensions']:
            transform_name = transform_type['func_name']
            dim_name = dimension['name']
            
            # Check compatibility
            if not is_compatible(transform_name, dim_name):
                continue
                
            # Count valid tests for this transform-dimension pair
            for implementation in config['implementations']:
                for size in config['sizes']:
                    for dtype_info in config['dtypes']:
                        for thread_count in config.get('thread_counts', [1]):
                            # Skip thread benchmarks for NumPy and SciPy if thread_count > 1
                            if thread_count > 1 and implementation['name'] in ['NumPy FFT', 'SciPy FFT']:
                                continue
                                
                            # Skip higher dimensions for very large sizes to avoid OOM
                            if size in very_large_sizes and 'volume' in dimension['name'].lower():
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
                    if not is_compatible(transform_name, dim_name):
                        print(f"Skipping {transform_type['name']} on {dim_name} (incompatible)")
                        continue
                    
                    for dtype_info in config['dtypes']:
                        for thread_count in config.get('thread_counts', [1]):
                            # Skip higher dimensions for very large sizes to avoid OOM
                            if size in very_large_sizes and 'volume' in dimension['name'].lower():
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
                                out_of_cache = config.get('out_of_cache', False)
                                array = create_test_array(shape, dtype, is_complex=is_complex_input, out_of_cache=out_of_cache)
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
                                    threads=thread_count,
                                    include_planning=config.get('include_planning', True),
                                    out_of_cache=out_of_cache,
                                    profile=config.get('profile', False) and completed_tests % 10 == 0  # Profile every 10th test
                                )
                                
                                # Save profile data if generated
                                if result['profile_data']:
                                    profile_dir = os.path.join(results_dir, "profiles")
                                    os.makedirs(profile_dir, exist_ok=True)
                                    
                                    profile_file = os.path.join(
                                        profile_dir, 
                                        f"{implementation['name']}_{transform_type['name']}_{size}.txt".replace(" ", "_")
                                    )
                                    
                                    with open(profile_file, 'w') as f:
                                        f.write(result['profile_data'])
                                    
                                    # Remove from result to keep CSV clean
                                    result.pop('profile_data')
                                
                                results.append(result)
                                
                                # Print result summary
                                print(f"  Execution: {result['median_exec_time']:.6f}s, "
                                      f"Planning: {result.get('median_plan_time', 'N/A')}, "
                                      f"Throughput: {result['throughput_gflops']:.2f} GFLOPS")
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
    if 'median_exec_time' in df.columns:
        summary = df.groupby('implementation')['median_exec_time'].agg(['mean', 'min', 'max'])
    else:
        summary = df.groupby('implementation')['mean_time'].agg(['mean', 'min', 'max'])
    print(tabulate(summary, headers='keys', tablefmt='grid'))
    
    return df

# Helper function to check transform-dimension compatibility
def is_compatible(transform_name, dimension_name):
    """
    Check if a transform type is compatible with the given dimension.
    
    Args:
        transform_name: Name of the transform function
        dimension_name: Name of the dimension
        
    Returns:
        True if compatible, False otherwise
    """
    # 2D transforms need at least 2D arrays
    if transform_name in ['fft2', 'ifft2', 'rfft2', 'irfft2'] and dimension_name == '1D':
        return False
    
    # 2D image dimensions only work with 2D transforms
    if 'image' in dimension_name.lower() and not any(x in transform_name for x in ['fft2', 'fftn']):
        return False
        
    # Volume dimensions only work with 3D transforms
    if 'volume' in dimension_name.lower() and not any(x in transform_name for x in ['fftn']):
        return False
        
    return True

# Function to track the wall-clock time of a function (for benchmark analysis)
def time_this(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        
        # Add to a global tracker if exists
        if hasattr(time_this, 'timings'):
            time_this.timings[func.__name__] = end_time - start_time
        else:
            time_this.timings = {func.__name__: end_time - start_time}
            
        return result
    return wrapper

# Update repeated transform benchmark with realistic application scenario
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
                    
                    if not is_compatible(transform_name, dim_name):
                        print(f"Skipping {transform_type['name']} on {dim_name} (incompatible)")
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
                    
                    # Add threads if supported
                    if implementation["name"] != "NumPy FFT" and implementation["name"] != "SciPy FFT":
                        impl_kwargs["threads"] = config.get('thread_count', 1)
                    
                    # Setup implementation
                    implementation["setup"]()
                    
                    # Print info
                    print(f"Testing {implementation['name']} {transform_type['name']} "
                          f"on {dimension['name']} array of size {size}")
                    
                    try:
                        # Track time for each repetition
                        times = []
                        times_with_planning = []
                        times_per_transform = []
                        
                        # Run with increasing repetition counts
                        for repetitions in config['repetition_counts']:
                            # Test with cache warm (planning done)
                            # Force garbage collection
                            gc.collect()
                            
                            # Run a single warm-up to establish the plan
                            _ = func(array.copy(), **impl_kwargs)
                            
                            # Now time the execution phase
                            start_time = time.time()
                            for _ in range(repetitions):
                                array_copy = array.copy()
                                result = func(array_copy, **impl_kwargs)
                            end_time = time.time()
                            
                            exec_time = end_time - start_time
                            exec_time_per_transform = exec_time / repetitions
                            
                            # Now measure with planning for each transform (worst case)
                            implementation["cleanup"]()
                            implementation["setup"]()
                            
                            start_time = time.time()
                            for _ in range(repetitions):
                                array_copy = array.copy()
                                # Clear cache each time to force planning
                                implementation["cleanup"]()
                                implementation["setup"]()
                                result = func(array_copy, **impl_kwargs)
                            end_time = time.time()
                            
                            total_time_with_planning = end_time - start_time
                            
                            # Record the times
                            times.append(exec_time)
                            times_with_planning.append(total_time_with_planning)
                            times_per_transform.append(exec_time_per_transform)
                            
                            # Record result
                            results.append({
                                "implementation": implementation['name'],
                                "transform_type": transform_type['name'],
                                "dimension": dimension['name'],
                                "size": size,
                                "repetitions": repetitions,
                                "total_time": exec_time,
                                "total_time_with_planning": total_time_with_planning,
                                "time_per_transform": exec_time_per_transform,
                                "time_per_transform_with_planning": total_time_with_planning / repetitions,
                                "throughput_transforms_per_s": repetitions / exec_time,
                                "planning_overhead_ratio": total_time_with_planning / exec_time
                            })
                            
                            print(f"  {repetitions} repetitions: {exec_time:.6f}s ({exec_time_per_transform:.6f}s per transform), "
                                  f"with planning: {total_time_with_planning:.6f}s")
                        
                        # Calculate improvement from first to last
                        if times_per_transform[0] > 0 and times_per_transform[-1] > 0:
                            improvement = times_per_transform[0] / times_per_transform[-1]
                            planning_impact = times_with_planning[-1] / times[-1]
                            
                            print(f"  Improvement with repetition: {improvement:.2f}x")
                            print(f"  Planning overhead impact: {planning_impact:.2f}x")
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

# Improved non-power-of-two benchmark with detailed analysis
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
    
    # Function to categorize a number based on its prime factorization
    def categorize_size(n, factors):
        if n == 1:
            return "trivial", "1"
            
        # Check if it's a power of 2
        if len(factors) == 1 and factors[0] == 2:
            log2 = int(np.log2(n))
            return "power_of_2", f"2^{log2}"
        
        # Check if it's a power of 3
        if len(factors) == 1 and factors[0] == 3:
            log3 = int(np.log(n) / np.log(3))
            return "power_of_3", f"3^{log3}"
            
        # Check if it's a single large prime
        if len(factors) == 1:
            return "prime", str(factors[0])
            
        # Count frequencies of each factor
        factor_counts = {}
        for f in factors:
            factor_counts[f] = factor_counts.get(f, 0) + 1
            
        # Format the factorization
        factor_str = " × ".join([f"{f}^{c}" if c > 1 else str(f) for f, c in sorted(factor_counts.items())])
        
        # Check if it's dominated by powers of 2
        if 2 in factor_counts and factor_counts[2] > len(factors) / 2:
            return "mostly_power_of_2", factor_str
            
        # Check if it has small primes only
        small_primes = [2, 3, 5, 7, 11, 13]
        if all(f in small_primes for f in factors):
            return "small_primes", factor_str
        
        # Check if it has large primes
        has_large_primes = any(f > 100 for f in factors)
        if has_large_primes:
            return "large_prime_factors", factor_str
            
        # Default: mixed factors
        return "mixed_factors", factor_str
    
    # Process each size to determine its factors
    size_info = []
    for size in config['sizes']:
        factors = factorize(size)
        size_type, factor_str = categorize_size(size, factors)
        
        size_info.append({
            "size": size,
            "factors": factors,
            "factor_str": factor_str,
            "size_type": size_type
        })
        
        print(f"Size {size}: {size_type} ({factor_str})")
    
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
                
                # Test with and without planning
                try:
                    # First, test with planning time included (first call)
                    # Cleanup and setup to ensure fresh state
                    implementation["cleanup"]()
                    implementation["setup"]()
                    
                    gc.collect()
                    start_time = time.time()
                    result = func(array.copy(), **impl_kwargs)
                    first_call_time = time.time() - start_time
                    
                    # Now test execution time (after planning)
                    execution_times = []
                    for _ in range(config['n_runs']):
                        array_copy = array.copy()
                        gc.collect()
                        
                        start_time = time.time()
                        result = func(array_copy, **impl_kwargs)
                        end_time = time.time()
                        
                        execution_times.append(end_time - start_time)
                    
                    # Calculate execution statistics
                    min_time = min(execution_times)
                    max_time = max(execution_times)
                    mean_time = sum(execution_times) / len(execution_times)
                    median_time = sorted(execution_times)[len(execution_times) // 2]
                    
                    # Estimate planning time
                    planning_time = first_call_time - median_time if first_call_time > median_time else 0
                    
                    # Record result
                    results.append({
                        "implementation": implementation['name'],
                        "transform_type": transform_type['name'],
                        "size": size,
                        "factor_str": size_data['factor_str'],
                        "size_type": size_data['size_type'],
                        "first_call_time": first_call_time,
                        "planning_time": planning_time,
                        "min_time": min_time,
                        "max_time": max_time,
                        "mean_time": mean_time,
                        "median_time": median_time,
                        "planning_ratio": planning_time / median_time if median_time > 0 else 0
                    })
                    
                    print(f"  First call: {first_call_time:.6f}s, Execution: {median_time:.6f}s, "
                          f"Planning: {planning_time:.6f}s")
                    
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

# Improved threading scaling benchmark with efficiency metrics
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
                    
                    if not is_compatible(transform_name, dim_name):
                        print(f"Skipping {transform_type['name']} on {dim_name} (incompatible)")
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
                        
                        # Run single-thread benchmark (with warm-up)
                        _ = func(array.copy(), **impl_kwargs)  # Warm-up
                        
                        single_thread_times = []
                        for _ in range(config['n_runs']):
                            array_copy = array.copy()
                            gc.collect()
                            
                            start_time = time.time()
                            _ = func(array_copy, **impl_kwargs)
                            end_time = time.time()
                            
                            single_thread_times.append(end_time - start_time)
                            
                        single_thread_time = sum(single_thread_times) / len(single_thread_times)
                        print(f"  1 thread: {single_thread_time:.6f}s")
                        
                        # Calculate array size and operations
                        array_size_mb = array.nbytes / (1024 * 1024)
                        flops = calculate_fft_gflops(shape, func_name) * 1e9
                        
                        # Test each thread count
                        for thread_count in config['thread_counts']:
                            # Update thread count
                            impl_kwargs["threads"] = thread_count
                            
                            # Run multi-thread benchmark
                            _ = func(array.copy(), **impl_kwargs)  # Warm-up
                            
                            times = []
                            for _ in range(config['n_runs']):
                                array_copy = array.copy()
                                gc.collect()
                                
                                start_time = time.time()
                                _ = func(array_copy, **impl_kwargs)
                                end_time = time.time()
                                
                                times.append(end_time - start_time)
                                
                            current_time = sum(times) / len(times)
                            
                            # Calculate metrics
                            speedup = single_thread_time / current_time
                            efficiency = speedup / thread_count
                            throughput_mb_per_s = array_size_mb / current_time
                            flops_per_s = flops / current_time
                            
                            # Calculate real threads
                            real_threads = min(thread_count, os.cpu_count())
                            real_efficiency = speedup / real_threads
                            
                            print(f"  {thread_count} threads: {current_time:.6f}s, "
                                  f"speedup: {speedup:.2f}x, efficiency: {efficiency:.2f}, "
                                  f"throughput: {throughput_mb_per_s:.2f} MB/s")
                            
                            # Record result
                            results.append({
                                "implementation": implementation['name'],
                                "transform_type": transform_type['name'],
                                "dimension": dimension['name'],
                                "size": size,
                                "array_size_mb": array_size_mb,
                                "thread_count": thread_count,
                                "actual_threads": real_threads,
                                "time": current_time,
                                "speedup": speedup,
                                "efficiency": efficiency,
                                "real_efficiency": real_efficiency,
                                "throughput_mb_per_s": throughput_mb_per_s,
                                "throughput_gflops": flops_per_s / 1e9
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

# Improved planning strategy benchmark with detailed overhead analysis
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
                    
                    if not is_compatible(transform_name, dim_name):
                        print(f"Skipping {transform_type['name']} on {dim_name} (incompatible)")
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
                    
                    # Reference data for ESTIMATE strategy
                    estimate_planning_time = None
                    estimate_execution_time = None
                    
                    # Test each planning strategy
                    for strategy in planning_strategies:
                        # Skip EXHAUSTIVE for large arrays (takes too long)
                        if strategy == "FFTW_EXHAUSTIVE" and (size > 1024 or 'volume' in dimension['name'].lower()):
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
                            
                            # Measure planning time with repeated runs
                            planning_times = []
                            
                            for _ in range(max(1, config['n_runs'] // 2)):  # Fewer runs for planning
                                # Cleanup and setup for a fresh state
                                implementation["cleanup"]()
                                implementation["setup"]()
                                
                                gc.collect()
                                plan_start_time = time.time()
                                _ = func(array.copy(), **impl_kwargs)  # First call includes planning
                                planning_times.append(time.time() - plan_start_time)
                            
                            # Take median planning time
                            planning_time = sorted(planning_times)[len(planning_times) // 2]
                            
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
                            execution_time = sorted(exec_times)[len(exec_times) // 2]
                            
                            # Store reference data for ESTIMATE
                            if strategy == "FFTW_ESTIMATE":
                                estimate_planning_time = planning_time
                                estimate_execution_time = execution_time
                            
                            # Calculate relative metrics
                            planning_overhead = planning_time - (estimate_planning_time or 0)
                            execution_improvement = (estimate_execution_time or execution_time) - execution_time
                            
                            # Calculate break-even point: at what repetition count does planning pay off?
                            if execution_improvement <= 0:
                                # More expensive planning but no execution benefit
                                breakeven_point = float('inf')
                            else:
                                # Calculate how many repeats for planning to pay off
                                breakeven_point = planning_overhead / execution_improvement
                            
                            # Record result
                            results.append({
                                "implementation": implementation['name'],
                                "transform_type": transform_type['name'],
                                "dimension": dimension['name'],
                                "size": size,
                                "planning_strategy": strategy,
                                "planning_time": planning_time,
                                "execution_time": execution_time,
                                "total_time": planning_time + (execution_time * config['n_runs']),
                                "planning_overhead": planning_overhead,
                                "execution_improvement": execution_improvement,
                                "breakeven_repeats": breakeven_point
                            })
                            
                            print(f"  {strategy}: planning={planning_time:.6f}s, "
                                  f"execution={execution_time:.6f}s, "
                                  f"breakeven={breakeven_point:.1f} repeats")
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

# Improved data type benchmark 
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
                    if not is_compatible(transform_name, dim_name):
                        print(f"Skipping {transform_type['name']} on {dim_name} (incompatible)")
                        continue
                    
                    # Calculate shape based on dimension
                    shape = dimension['shape_func'](size)
                    
                    # Reference data for float64 (will be used to calculate ratios)
                    float64_time = None
                    
                    for dtype_info in dtypes:
                        # Determine if complex input is required
                        is_complex_input = transform_type['func_name'] in [
                            'ifft', 'ifft2', 'ifftn', 'irfft', 'irfft2', 'irfftn'
                        ]
                        
                        # Determine dtype
                        dtype = dtype_info['complex_dtype'] if is_complex_input else dtype_info['real_dtype']
                        
                        # Calculate array memory footprint
                        element_size = 8 if dtype_info['name'] == 'float64' else 4  # bytes per element
                        if is_complex_input:
                            element_size *= 2  # complex uses 2x memory
                            
                        memory_footprint = np.prod(shape) * element_size / (1024 * 1024)  # MB
                        
                        # Create test array
                        try:
                            array = create_test_array(shape, dtype, is_complex=is_complex_input)
                        except MemoryError:
                            print(f"Memory error creating array of shape {shape} with {dtype}")
                            continue
                        
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
                            
                            # First run to warm up and handle planning
                            _ = func(array.copy(), **impl_kwargs)
                            
                            # Run benchmark
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
                            
                            # Calculate throughput
                            throughput_mb_per_s = memory_footprint / median_time
                            
                            # Store reference time for float64
                            if dtype_info['name'] == 'float64':
                                float64_time = median_time
                            
                            # Calculate ratio to float64 if available
                            ratio_to_float64 = None
                            if float64_time is not None and dtype_info['name'] != 'float64':
                                ratio_to_float64 = float64_time / median_time
                            
                            # Record result
                            results.append({
                                "implementation": implementation['name'],
                                "transform_type": transform_type['name'],
                                "dimension": dimension['name'],
                                "size": size,
                                "shape": str(shape),
                                "dtype": dtype_info['name'],
                                "is_complex": is_complex_input,
                                "memory_footprint_mb": memory_footprint,
                                "min_time": min_time,
                                "max_time": max_time,
                                "mean_time": mean_time,
                                "median_time": median_time,
                                "throughput_mb_per_s": throughput_mb_per_s,
                                "ratio_to_float64": ratio_to_float64
                            })
                            
                            print(f"  Median time: {median_time:.6f}s, "
                                  f"Throughput: {throughput_mb_per_s:.2f} MB/s")
                            
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

# New: Real-world application scenario benchmark
def run_application_benchmark(config):
    """
    Benchmark a realistic application scenario with multiple FFT operations.
    
    Args:
        config: Dictionary with benchmark configuration
    
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    print("\n" + "=" * 50)
    print(f"RUNNING APPLICATION BENCHMARK: {config['name']}")
    print("=" * 50)
    
    # Define application scenarios
    scenarios = config.get('scenarios', [
        {
            "name": "Image filtering",
            "steps": [
                {"op": "fft2", "desc": "Forward FFT"},
                {"op": "custom", "desc": "Apply filter in frequency domain"},
                {"op": "ifft2", "desc": "Inverse FFT"}
            ]
        },
        {
            "name": "Signal processing",
            "steps": [
                {"op": "rfft", "desc": "Forward real FFT"},
                {"op": "custom", "desc": "Spectrum analysis"},
                {"op": "irfft", "desc": "Inverse real FFT"}
            ]
        },
        {
            "name": "Volume rendering",
            "steps": [
                {"op": "fftn", "desc": "3D FFT"},
                {"op": "custom", "desc": "3D frequency domain processing"},
                {"op": "ifftn", "desc": "Inverse 3D FFT"}
            ]
        }
    ])
    
    # Iterate over all combinations
    for implementation in config['implementations']:
        for scenario in scenarios:
            for dimension in config['dimensions']:
                # Check if this dimension is compatible with the scenario
                scenario_name = scenario['name'].lower()
                if 'image' in scenario_name and 'image' not in dimension['name'].lower():
                    continue
                if 'volume' in scenario_name and 'volume' not in dimension['name'].lower():
                    continue
                if 'signal' in scenario_name and not dimension['name'] == '1D':
                    continue
                
                # Get the shape
                shape = dimension['shape_func'](1)  # Size parameter is ignored for application benchmarks
                
                # Create test array (real input)
                try:
                    array = create_test_array(shape, np.float64, is_complex=False)
                except MemoryError:
                    print(f"Memory error creating array of shape {shape}")
                    continue
                
                # Setup implementation
                implementation["setup"]()
                
                # Print info
                print(f"Testing {implementation['name']} on {scenario['name']} scenario "
                      f"with {dimension['name']} data of shape {shape}")
                
                try:
                    # Execute the scenario
                    step_times = []
                    
                    # Create copies for each run
                    result_arrays = []
                    
                    # Run the full pipeline multiple times
                    for run in range(config['n_runs']):
                        data = array.copy()
                        current_step_times = []
                        
                        for step in scenario['steps']:
                            if step['op'] == 'custom':
                                # Simulate custom processing by doing a simple operation
                                start_time = time.time()
                                
                                # Simple frequency domain operation (e.g., filtering)
                                if len(data.shape) == 1:
                                    # 1D: Simple lowpass filter
                                    mid = len(data) // 2
                                    cutoff = mid // 2
                                    data[cutoff:mid+cutoff] *= 0.5
                                elif len(data.shape) == 2:
                                    # 2D: Simple circular filter
                                    h, w = data.shape
                                    center_y, center_x = h // 2, w // 2
                                    y, x = np.ogrid[:h, :w]
                                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                                    mask = dist <= min(h, w) // 4
                                    data[~mask] *= 0.1
                                else:
                                    # 3D: Simple operation
                                    data *= 0.9
                                    
                                end_time = time.time()
                            else:
                                # Get FFT function
                                func = getattr(implementation['module'], step['op'])
                                impl_kwargs = implementation['kwargs'].copy()
                                
                                # Execute transform
                                start_time = time.time()
                                data = func(data, **impl_kwargs)
                                end_time = time.time()
                            
                            current_step_times.append(end_time - start_time)
                            
                        # Store result and times
                        result_arrays.append(data)
                        step_times.append(current_step_times)
                    
                    # Calculate average time for each step
                    avg_step_times = []
                    for i in range(len(scenario['steps'])):
                        avg_time = sum(run_times[i] for run_times in step_times) / len(step_times)
                        avg_step_times.append(avg_time)
                    
                    # Calculate total time
                    total_time = sum(avg_step_times)
                    
                    # Record result
                    results.append({
                        "implementation": implementation['name'],
                        "scenario": scenario['name'],
                        "dimension": dimension['name'],
                        "shape": str(shape),
                        "array_size_mb": array.nbytes / (1024 * 1024),
                        "total_time": total_time,
                        "steps": len(scenario['steps']),
                        "step_times": str(avg_step_times)  # Store as string for CSV
                    })
                    
                    print(f"  Total time: {total_time:.6f}s")
                    for i, step in enumerate(scenario['steps']):
                        print(f"    {step['desc']}: {avg_step_times[i]:.6f}s")
                    
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

# New: Out-of-cache performance benchmark
def run_out_of_cache_benchmark(config):
    """
    Benchmark performance when data doesn't fit in CPU cache.
    
    Args:
        config: Dictionary with benchmark configuration
    
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    print("\n" + "=" * 50)
    print(f"RUNNING OUT-OF-CACHE BENCHMARK: {config['name']}")
    print("=" * 50)
    
    # Iterate over all combinations
    for implementation in config['implementations']:
        for transform_type in config['transform_types']:
            for size in config['sizes']:
                for dimension in config['dimensions']:
                    # Check dimension compatibility with transform
                    transform_name = transform_type['func_name']
                    dim_name = dimension['name']
                    
                    if not is_compatible(transform_name, dim_name):
                        print(f"Skipping {transform_type['name']} on {dim_name} (incompatible)")
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
                    impl_kwargs = implementation["kwargs"].copy()
                    
                    # Setup implementation
                    implementation["setup"]()
                    
                    # Print info
                    print(f"Testing {implementation['name']} {transform_type['name']} "
                          f"on {dimension['name']} array of size {size}")
                    
                    try:
                        # Test with cache warm (standard case)
                        # Warm-up run
                        _ = func(array.copy(), **impl_kwargs)
                        
                        # Now benchmark with warm cache
                        warm_cache_times = []
                        for _ in range(config['n_runs']):
                            array_copy = array.copy()
                            gc.collect()
                            
                            start_time = time.time()
                            result = func(array_copy, **impl_kwargs)
                            end_time = time.time()
                            
                            warm_cache_times.append(end_time - start_time)
                        
                        warm_cache_time = sorted(warm_cache_times)[len(warm_cache_times) // 2]
                        
                        # Test with cache flushed between runs
                        cold_cache_times = []
                        for _ in range(config['n_runs']):
                            array_copy = array.copy()
                            
                            # Flush cache by accessing a large array
                            create_test_array.cache_buster[:] = np.random.random(create_test_array.cache_buster.shape)
                            
                            gc.collect()
                            
                            start_time = time.time()
                            result = func(array_copy, **impl_kwargs)
                            end_time = time.time()
                            
                            cold_cache_times.append(end_time - start_time)
                            
                        cold_cache_time = sorted(cold_cache_times)[len(cold_cache_times) // 2]
                        
                        # Calculate the cache impact
                        cache_impact = cold_cache_time / warm_cache_time
                        
                        # Record result
                        results.append({
                            "implementation": implementation['name'],
                            "transform_type": transform_type['name'],
                            "dimension": dimension['name'],
                            "size": size,
                            "shape": str(shape),
                            "array_size_mb": array.nbytes / (1024 * 1024),
                            "warm_cache_time": warm_cache_time,
                            "cold_cache_time": cold_cache_time,
                            "cache_impact": cache_impact
                        })
                        
                        print(f"  Warm cache: {warm_cache_time:.6f}s, "
                              f"Cold cache: {cold_cache_time:.6f}s, "
                              f"Impact: {cache_impact:.2f}x")
                        
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
    if group_by in results.columns:
        groups = results.groupby(group_by)
        
        # Create line plot
        for name, group in groups:
            plt.plot(group[x_axis], group[y_axis], marker='o', linestyle='-', label=name)
    else:
        # No grouping, just plot the data
        plt.plot(results[x_axis], results[y_axis], marker='o', linestyle='-')
    
    # Set log scale if requested
    if log_scale:
        if x_axis == 'size' or 'size' in x_axis:
            plt.xscale('log', base=2)
        if y_axis == 'time' or 'time' in y_axis:
            plt.yscale('log')
    
    # Set labels
    plt.xlabel(x_label or x_axis)
    plt.ylabel(y_label or y_axis)
    plt.title(title or f"Benchmark Results: {y_axis} vs {x_axis}")
    
    # Add legend if grouping was used
    if group_by in results.columns:
        plt.legend()
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_file = os.path.join(results_dir, f"{plot_name}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")
    
    plt.close()

# Function to create combined plots and comparisons
def create_summary_plots(results):
    """
    Create summary plots combining data from multiple benchmarks.
    
    Args:
        results: Dictionary with benchmark results (name -> DataFrame)
    """
    print("\n" + "=" * 50)
    print("CREATING SUMMARY PLOTS")
    print("=" * 50)
    
    # Create directory for summary plots
    summary_dir = os.path.join(results_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # 1. Performance comparison across implementations
    try:
        # Combine data from basic benchmarks
        basic_dfs = []
        for name, df in results.items():
            if 'basic_performance' in name and 'median_exec_time' in df.columns:
                df = df.copy()
                df['benchmark'] = name
                basic_dfs.append(df)
        
        if basic_dfs:
            combined_df = pd.concat(basic_dfs)
            
            # Plot by implementation
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='implementation', y='median_exec_time', data=combined_df)
            plt.yscale('log')
            plt.xticks(rotation=45, ha='right')
            plt.title('Performance Comparison Across All Basic Benchmarks')
            plt.ylabel('Execution Time (seconds, log scale)')
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, 'implementation_comparison.png'), dpi=300)
            plt.close()
            
            # Plot throughput by implementation
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='implementation', y='throughput_gflops', data=combined_df)
            plt.yscale('log')
            plt.xticks(rotation=45, ha='right')
            plt.title('Throughput Comparison Across All Basic Benchmarks')
            plt.ylabel('Throughput (GFLOPS, log scale)')
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, 'throughput_comparison.png'), dpi=300)
            plt.close()
            
            # Plot planning overhead ratio
            if 'median_plan_time' in combined_df.columns:
                combined_df['planning_overhead_ratio'] = combined_df['median_plan_time'] / combined_df['median_exec_time']
                
                plt.figure(figsize=(12, 8))
                sns.boxplot(x='implementation', y='planning_overhead_ratio', data=combined_df)
                plt.yscale('log')
                plt.xticks(rotation=45, ha='right')
                plt.title('Planning Overhead Ratio (Planning Time / Execution Time)')
                plt.ylabel('Ratio (log scale)')
                plt.tight_layout()
                plt.savefig(os.path.join(summary_dir, 'planning_overhead_ratio.png'), dpi=300)
                plt.close()
            
            print("Created implementation comparison summary plots")
    except Exception as e:
        print(f"Error creating implementation comparison plots: {str(e)}")
    
    # 2. Scaling with array size
    try:
        # Filter benchmarks that have size information
        size_dfs = []
        for name, df in results.items():
            if 'basic_performance' in name and 'size' in df.columns and 'median_exec_time' in df.columns:
                df = df.copy()
                df['benchmark'] = name
                size_dfs.append(df)
        
        if size_dfs:
            size_df = pd.concat(size_dfs)
            
            # Plot execution time vs size for each implementation
            plt.figure(figsize=(12, 8))
            for impl, group in size_df.groupby('implementation'):
                # Get average time for each size
                size_avg = group.groupby('size')['median_exec_time'].mean()
                plt.plot(size_avg.index, size_avg.values, marker='o', label=impl)
            
            plt.xscale('log', base=2)
            plt.yscale('log')
            plt.title('Execution Time Scaling with Array Size')
            plt.xlabel('Array Size')
            plt.ylabel('Execution Time (seconds, log scale)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(summary_dir, 'size_scaling.png'), dpi=300)
            plt.close()
            
            print("Created size scaling summary plot")
    except Exception as e:
        print(f"Error creating size scaling plot: {str(e)}")
    
    # 3. Threading efficiency visualization
    try:
        thread_df = None
        for name, df in results.items():
            if 'threading' in name and 'thread_count' in df.columns and 'efficiency' in df.columns:
                thread_df = df
                break
        
        if thread_df is not None:
            # Plot efficiency vs thread count
            plt.figure(figsize=(12, 8))
            for (impl, size), group in thread_df.groupby(['implementation', 'size']):
                plt.plot(group['thread_count'], group['efficiency'], 
                         marker='o', label=f"{impl} - Size {size}")
            
            plt.axhline(y=1.0, color='k', linestyle='--', label='Ideal Efficiency')
            plt.axhline(y=0.8, color='k', linestyle=':', label='80% Efficiency')
            plt.title('Threading Efficiency')
            plt.xlabel('Thread Count')
            plt.ylabel('Efficiency (Speedup / Thread Count)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(summary_dir, 'threading_efficiency.png'), dpi=300)
            plt.close()
            
            print("Created threading efficiency summary plot")
    except Exception as e:
        print(f"Error creating threading efficiency plot: {str(e)}")
    
    # 4. Planning strategy comparison
    try:
        planning_df = None
        for name, df in results.items():
            if 'planning' in name and 'planning_strategy' in df.columns:
                planning_df = df
                break
        
        if planning_df is not None:
            # Plot planning time vs execution time
            plt.figure(figsize=(12, 8))
            
            # Use different colors for different implementations
            implementations = planning_df['implementation'].unique()
            colors = plt.cm.viridis(np.linspace(0, 1, len(implementations)))
            
            for i, impl in enumerate(implementations):
                impl_df = planning_df[planning_df['implementation'] == impl]
                
                for strategy in impl_df['planning_strategy'].unique():
                    strategy_df = impl_df[impl_df['planning_strategy'] == strategy]
                    
                    # Calculate average for each size
                    size_avg = strategy_df.groupby('size').agg({
                        'planning_time': 'mean',
                        'execution_time': 'mean'
                    })
                    
                    plt.scatter(size_avg['planning_time'], size_avg['execution_time'], 
                               s=100, color=colors[i], 
                               label=f"{impl} - {strategy}")
            
            plt.xscale('log')
            plt.yscale('log')
            plt.title('Planning Time vs Execution Time')
            plt.xlabel('Planning Time (seconds, log scale)')
            plt.ylabel('Execution Time (seconds, log scale)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(summary_dir, 'planning_vs_execution.png'), dpi=300)
            plt.close()
            
            # Plot break-even point by planning strategy
            if 'breakeven_repeats' in planning_df.columns:
                plt.figure(figsize=(12, 8))
                sns.boxplot(x='planning_strategy', y='breakeven_repeats', data=planning_df)
                plt.yscale('log')
                plt.title('Break-even Point by Planning Strategy')
                plt.xlabel('Planning Strategy')
                plt.ylabel('Number of Transforms to Break Even (log scale)')
                plt.savefig(os.path.join(summary_dir, 'breakeven_by_strategy.png'), dpi=300)
                plt.close()
            
            print("Created planning strategy summary plots")
    except Exception as e:
        print(f"Error creating planning strategy plots: {str(e)}")
    
    # 5. Real-world application performance
    try:
        app_df = None
        for name, df in results.items():
            if 'application' in name and 'scenario' in df.columns:
                app_df = df
                break
        
        if app_df is not None:
            # Plot total time by implementation and scenario
            plt.figure(figsize=(14, 8))
            sns.barplot(x='scenario', y='total_time', hue='implementation', data=app_df)
            plt.title('Application Scenario Performance')
            plt.ylabel('Total Time (seconds)')
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, 'application_performance.png'), dpi=300)
            plt.close()
            
            print("Created application performance summary plot")
    except Exception as e:
        print(f"Error creating application performance plot: {str(e)}")
    
    # 6. Out-of-cache impact
    try:
        cache_df = None
        for name, df in results.items():
            if 'cache' in name and 'warm_cache_time' in df.columns:
                cache_df = df
                break
        
        if cache_df is not None:
            # Plot cache impact by implementation
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='implementation', y='cache_impact', data=cache_df)
            plt.title('Cache Impact by Implementation')
            plt.ylabel('Cold Cache Time / Warm Cache Time')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, 'cache_impact.png'), dpi=300)
            plt.close()
            
            print("Created cache impact summary plot")
    except Exception as e:
        print(f"Error creating cache impact plot: {str(e)}")
    
    # 7. Create a performance summary table
    try:
        summary_rows = []
        
        # Select basic benchmark data if available
        basic_large = None
        for name, df in results.items():
            if name == 'basic_performance_(large)' and 'median_exec_time' in df.columns:
                basic_large = df
                break
        
        if basic_large is not None:
            # Compute speedup over NumPy
            pivot = basic_large.pivot_table(
                index=['transform_type', 'input_size'],
                columns='implementation',
                values='median_exec_time'
            )
            
            # Get average speedup over NumPy for each implementation
            if 'NumPy FFT' in pivot.columns:
                for col in pivot.columns:
                    if col != 'NumPy FFT':
                        speedup = pivot['NumPy FFT'] / pivot[col]
                        avg_speedup = speedup.mean()
                        summary_rows.append({
                            'Implementation': col,
                            'Metric': 'Avg Speedup vs NumPy',
                            'Value': f"{avg_speedup:.2f}x"
                        })
            
            # Get best and worst case speedups
            if 'BetterFFTW (Default)' in pivot.columns and 'NumPy FFT' in pivot.columns:
                speedup = pivot['NumPy FFT'] / pivot['BetterFFTW (Default)']
                summary_rows.append({
                    'Implementation': 'BetterFFTW (Default)',
                    'Metric': 'Best Speedup vs NumPy',
                    'Value': f"{speedup.max():.2f}x"
                })
                summary_rows.append({
                    'Implementation': 'BetterFFTW (Default)',
                    'Metric': 'Worst Speedup vs NumPy',
                    'Value': f"{speedup.min():.2f}x"
                })
        
        # Gather threading scaling data
        thread_df = None
        for name, df in results.items():
            if 'threading' in name and 'speedup' in df.columns:
                thread_df = df
                break
        
        if thread_df is not None:
            # Get max speedup for each implementation
            for impl in thread_df['implementation'].unique():
                impl_df = thread_df[thread_df['implementation'] == impl]
                max_speedup = impl_df['speedup'].max()
                max_threads = impl_df.loc[impl_df['speedup'].idxmax(), 'thread_count']
                
                summary_rows.append({
                    'Implementation': impl,
                    'Metric': f'Max Threading Speedup ({max_threads} threads)',
                    'Value': f"{max_speedup:.2f}x"
                })
        
        # Create summary table
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_file = os.path.join(summary_dir, 'performance_summary.csv')
            summary_df.to_csv(summary_file, index=False)
            print(f"Created performance summary table: {summary_file}")
    except Exception as e:
        print(f"Error creating performance summary: {str(e)}")
    
    print("\nSummary plots and data saved to:", summary_dir)

# ===============================================================================
# Run the benchmarks
# ===============================================================================

if __name__ == "__main__":
    # Check if NumPy is linked against MKL
    if is_numpy_using_mkl():
        print("NumPy appears to be using Intel MKL")
    else:
        print("NumPy does not appear to be using Intel MKL")
    
    # Subset of implementations for faster benchmarks or diagnostics
    fast_implementations = [
        implementations["numpy"],
        implementations["scipy"],
        implementations["betterfftw_default"]
    ]
    
    # All FFTW-based implementations
    fftw_implementations = [impl for name, impl in implementations.items() 
                           if "fftw" in name.lower()]
    
    # Subset of transform types for faster benchmarks
    fast_transforms = [
        transform_types["fft"],
        transform_types["fft2"],
        transform_types["fftn"]
    ]
    
    # Configure benchmarks
    benchmarks = [
        # 1. Basic performance comparison with realistic sizes - small
        {
            "name": "Basic Performance (Small)", 
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["fft"]],  # 1D transform
            "sizes": small_sizes,
            "dimensions": [dimensions["1d"]],  # 1D arrays
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 5,
            "include_planning": True,
            "out_of_cache": False,
            "profile": False
        },
        
        # 2. Basic performance comparison with realistic sizes - medium
        {
            "name": "Basic Performance (Medium)",
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["fft"]],  # 1D transform
            "sizes": medium_sizes,
            "dimensions": [dimensions["1d"]],  # 1D arrays
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 5,
            "include_planning": True,
            "out_of_cache": False,
            "profile": False
        },
        
        # 3. Basic performance comparison with realistic sizes - large
        {
            "name": "Basic Performance (Large)",
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["fft"]],  # 1D transform
            "sizes": large_sizes,
            "dimensions": [dimensions["1d"]],  # 1D arrays
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 3,
            "include_planning": True,
            "out_of_cache": False,
            "profile": True  # Enable profiling for large arrays
        },
        
        # 4. 2D transform performance - small/medium/large with realistic image sizes
        {
            "name": "2D Transform Performance",
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["fft2"]],  # 2D transform
            "sizes": [512, 1024, 2048],  # Size parameter for square arrays
            "dimensions": [
                dimensions["2d_square"], 
                dimensions["2d_rect"],
                dimensions["2d_image_hd"]
            ],
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 5,
            "include_planning": True,
            "out_of_cache": False,
            "profile": False
        },
        
        # 5. Real FFT performance
        {
            "name": "Real FFT Performance",
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["rfft"], transform_types["rfft2"]],
            "sizes": [1024, 2048, 4096],
            "dimensions": [dimensions["1d"], dimensions["2d_square"]],
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 5,
            "include_planning": True,
            "out_of_cache": False,
            "profile": False
        },
        
        # 6. 3D FFT performance with medical volumes
        {
            "name": "3D FFT Performance",
            "implementations": list(implementations.values()),
            "transform_types": [transform_types["fftn"]],  # N-D transform
            "sizes": [64, 128],  # Only small sizes for 3D
            "dimensions": [
                dimensions["3d"], 
                dimensions["3d_volume_small"],
                dimensions["3d_volume_medical"]
            ],
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1],
            "n_runs": 3,
            "include_planning": True,
            "out_of_cache": False,
            "profile": False
        },
        
        # 7. Threading performance (detailed)
        {
            "name": "Threading Scaling",
            "implementations": fftw_implementations,  # Only FFTW-based implementations
            "transform_types": [transform_types["fft2"], transform_types["fftn"]],
            "sizes": [512, 1024, 2048],
            "dimensions": [dimensions["2d_square"], dimensions["3d_volume_small"]],
            "thread_counts": thread_counts,
            "n_runs": 3
        },
        
        # 8. Non-power-of-two performance (detailed)
        {
            "name": "Non-Power-of-Two Performance",
            "implementations": [implementations["numpy"], implementations["betterfftw_default"]],
            "transform_types": [transform_types["fft"], transform_types["fft2"]],
            "sizes": small_sizes + medium_sizes + non_power_of_2_sizes,
            "n_runs": 5
        },
        
        # 9. Repeated transform performance (realistic workflow)
        {
            "name": "Repeated Transform Performance",
            "implementations": [implementations["numpy"], implementations["betterfftw_default"]],
            "transform_types": [transform_types["fft"], transform_types["fft2"]],
            "sizes": [1024, 2048],
            "dimensions": [dimensions["1d"], dimensions["2d_square"]],
            "repetition_counts": [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
            "thread_count": 4
        },
        
        # 10. Planning strategy comparison (detailed)
        {
            "name": "Planning Strategy Comparison",
            "implementations": [implementations["betterfftw_default"], implementations["pyfftw_estimate"]],
            "transform_types": [transform_types["fft"], transform_types["fft2"]],
            "sizes": [512, 1024, 2048, 4096],
            "dimensions": [dimensions["1d"], dimensions["2d_square"]],
            "n_runs": 5
        },
        
        # 11. Data type comparison (float32 vs float64)
        {
            "name": "Data Type Comparison",
            "implementations": [implementations["numpy"], implementations["betterfftw_default"]],
            "transform_types": [transform_types["fft"], transform_types["fft2"]],
            "sizes": [512, 1024, 2048, 4096],
            "dimensions": [dimensions["1d"], dimensions["2d_square"]],
            "n_runs": 5
        },
        
        # 12. Application scenario benchmark
        {
            "name": "Application Benchmark",
            "implementations": list(implementations.values()),
            "dimensions": [
                dimensions["1d"], 
                dimensions["2d_image_hd"],
                dimensions["3d_volume_small"]
            ],
            "n_runs": 5
        },
        
        # 13. Out-of-cache performance
        {
            "name": "Out of Cache Performance",
            "implementations": [implementations["numpy"], implementations["betterfftw_default"]],
            "transform_types": [transform_types["fft"], transform_types["fft2"]],
            "sizes": [4096, 8192, 16384],
            "dimensions": [dimensions["1d"], dimensions["2d_square"]],
            "n_runs": 5
        },
        
        # 14. Very large array performance (memory bound)
        {
            "name": "Very Large Array Performance",
            "implementations": [implementations["numpy"], implementations["betterfftw_default"]],
            "transform_types": [transform_types["fft"]],
            "sizes": very_large_sizes,
            "dimensions": [dimensions["1d"]],
            "dtypes": [dtypes["float64"]],
            "thread_counts": [1, max_threads],
            "n_runs": 3,
            "include_planning": True,
            "out_of_cache": True,
            "profile": True
        }
    ]
    
    # Run selected benchmarks (comment out ones you don't want to run)
    active_benchmarks = benchmarks.copy()
    
    # Dictionary to store results
    results = {}
    
    # Run each benchmark
    for benchmark in active_benchmarks:
        name = benchmark["name"]
        
        if "Basic Performance" in name or "2D Transform" in name or "Real FFT" in name or "3D FFT" in name or "Very Large" in name:
            results[name] = run_benchmark_suite(benchmark)
        elif "Threading Scaling" in name:
            results[name] = run_threading_scaling_benchmark(benchmark)
        elif "Non-Power-of-Two" in name:
            results[name] = run_non_power_of_two_benchmark(benchmark)
        elif "Repeated Transform" in name:
            results[name] = run_repeated_transform_benchmark(benchmark)
        elif "Planning Strategy" in name:
            results[name] = run_planning_strategy_benchmark(benchmark)
        elif "Data Type" in name:
            results[name] = run_dtype_benchmark(benchmark)
        elif "Application" in name:
            results[name] = run_application_benchmark(benchmark)
        elif "Out of Cache" in name:
            results[name] = run_out_of_cache_benchmark(benchmark)
        else:
            print(f"WARNING: No matching benchmark function for '{name}'")
    
    # Create plots for each benchmark
    for name, df in results.items():
        if "Basic Performance" in name:
            if 'median_exec_time' in df.columns:
                time_col = 'median_exec_time'
            else:
                time_col = 'median_time'
            
            # Plot time vs size for each implementation
            plot_benchmark_results(
                df,
                f"{name.replace(' ', '_').lower()}_time",
                x_axis='input_size',
                y_axis=time_col,
                group_by='implementation',
                title=f"{name} - Execution Time",
                x_label="Array Size",
                y_label="Time (seconds)"
            )
            
            # Plot throughput
            if 'throughput_gflops' in df.columns:
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
            
            # Plot planning overhead if available
            if 'median_plan_time' in df.columns:
                plot_benchmark_results(
                    df,
                    f"{name.replace(' ', '_').lower()}_planning",
                    x_axis='input_size',
                    y_axis='median_plan_time',
                    group_by='implementation',
                    title=f"{name} - Planning Time",
                    x_label="Array Size",
                    y_label="Planning Time (seconds)"
                )
        
        elif "Transform Performance" in name:
            # Group by dimension type
            for dim in df['input_shape'].unique():
                # Filter for this dimension
                dim_data = df[df['input_shape'] == dim]
                
                # Plot time vs implementation
                if 'median_exec_time' in dim_data.columns:
                    time_col = 'median_exec_time'
                else:
                    time_col = 'median_time'
                
                plot_benchmark_results(
                    dim_data,
                    f"{name.replace(' ', '_').lower()}_{dim}".replace("(", "").replace(")", "").replace(",", "_").replace(" ", "_"),
                    x_axis='implementation',
                    y_axis=time_col,
                    title=f"{name} - {dim}",
                    x_label="Implementation",
                    y_label="Time (seconds)",
                    log_scale=True
                )
        
        elif "FFT Performance" in name:
            # Plot by transform type and dimension
            for transform in df['transform_type'].unique():
                transform_data = df[df['transform_type'] == transform]
                
                if 'median_exec_time' in transform_data.columns:
                    time_col = 'median_exec_time'
                else:
                    time_col = 'median_time'
                
                plot_benchmark_results(
                    transform_data,
                    f"{name.replace(' ', '_').lower()}_{transform.replace(' ', '_').lower()}",
                    x_axis='implementation',
                    y_axis=time_col,
                    title=f"{name} - {transform}",
                    x_label="Implementation",
                    y_label="Time (seconds)",
                    log_scale=True
                )
        
        # Specialized plots for other benchmark types are handled in their respective functions
    
    # Create summary plots combining data from all benchmarks
    create_summary_plots(results)
    
    print("\nAll benchmarks complete!")
    print(f"Results saved to {results_dir}/")