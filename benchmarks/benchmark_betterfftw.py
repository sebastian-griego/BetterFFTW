"""
Enhanced BetterFFTW Benchmark Suite

This script extends the original benchmark to test larger arrays, 
run more iterations, and provide more statistically significant results.
"""

import os
import sys
import time
import gc
import numpy as np
import scipy.fft
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import traceback
import threading
from datetime import datetime
from tabulate import tabulate
from functools import partial
import argparse

# Try to import pyfftw and betterfftw
try:
    import pyfftw
    HAVE_PYFFTW = True
    # Enable the PyFFTW cache
    pyfftw.interfaces.cache.enable()
except ImportError:
    HAVE_PYFFTW = False
    print("PyFFTW not found. Skipping PyFFTW benchmarks.")

try:
    import betterfftw
    HAVE_BETTERFFTW = True
except ImportError:
    HAVE_BETTERFFTW = False
    print("BetterFFTW not found. Please install it to run these benchmarks.")
    sys.exit(1)

# Create results directory
RESULTS_DIR = "benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Current timestamp for unique filenames
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Setup logging to file
import logging
LOG_FILE = os.path.join(RESULTS_DIR, f"benchmark_{TIMESTAMP}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("benchmark")
# Define the FFT implementations to benchmark
FFT_IMPLEMENTATIONS = {
    "numpy": {
        "name": "NumPy FFT",
        "module": np.fft,
        "setup": lambda: None,
        "cleanup": lambda: None,
        "available": True
    },
    "scipy": {
        "name": "SciPy FFT",
        "module": scipy.fft,
        "setup": lambda: None,
        "cleanup": lambda: None,
        "available": True
    },
    "pyfftw": {
        "name": "PyFFTW",
        "module": pyfftw.interfaces.numpy_fft if HAVE_PYFFTW else None,
        "setup": lambda: pyfftw.interfaces.cache.enable() if HAVE_PYFFTW else None,
        "cleanup": lambda: None,
        "available": HAVE_PYFFTW
    },
    "betterfftw": {
        "name": "BetterFFTW",
        "module": betterfftw,
        "setup": lambda: None,
        "cleanup": lambda: None,  # Changed from betterfftw.clear_cache() to None
        "available": True
    }
}

# Force garbage collection between benchmarks to reduce interference
def force_gc():
    """Force garbage collection to clean memory between tests."""
    gc.collect()

# Get current memory usage in MB
def get_memory_usage():
    """Return current memory usage of this process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

# Create array with specified shape and type
def create_test_array(shape, dtype=np.float64, is_complex=False):
    """Create a test array with the specified shape and type."""
    if is_complex:
        # For complex input (e.g., for ifft)
        real_part = np.random.random(shape).astype(dtype)
        imag_part = np.random.random(shape).astype(dtype)
        return real_part + 1j * imag_part
    else:
        # For real input
        return np.random.random(shape).astype(dtype)

# Utility to accurately measure function execution time
def time_function(func, *args, **kwargs):
    """Measure execution time of a function call."""
    # Force garbage collection before timing
    gc.collect()
    
    # Measure execution time
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    return result, end_time - start_time

def benchmark_direct_comparison(n_datasets=3, save_results=True):
    """
    Compare BetterFFTW defaults against other libraries across
    various array sizes and dimensions.
    
    Args:
        n_datasets: Number of different random datasets to test
        save_results: Whether to save results to disk
    """
    logger.info("Starting enhanced direct comparison benchmark...")
    
    results = []
    
    # Test configurations
    dimensions = [1, 2, 3]  # 1D, 2D, 3D
    
    # Power-of-2 sizes (extended to larger sizes)
    power2_sizes = [2**n for n in range(8, 19)]  # 256 to 262,144
    
    # Non-power-of-2 sizes (extended with more prime numbers and challenging sizes)
    nonpower2_sizes = [
        300,      # 2^2 * 3 * 5^2
        720,      # 2^4 * 3^2 * 5
        1000,     # 2^3 * 5^3
        1331,     # 11^3 (prime cubed)
        1920,     # 2^6 * 3 * 5 (HD width)
        2187,     # 3^7 (power of 3)
        10000,    # 2^4 * 5^4
        12289,    # Prime number (large prime)
        15625,    # 5^6 (power of 5)
        20000,    # 2^5 * 5^4
        32805,    # 3 * 5 * 7 * 313 (multiple prime factors)
        65537,    # Fermat prime (2^16 + 1)
        100000,   # 2^5 * 5^5
        131071,   # Mersenne number (2^17 - 1)
        262143    # Mersenne number (2^18 - 1)
    ]
    
    # Combined sizes list
    all_sizes = sorted(power2_sizes + nonpower2_sizes)
    
    # Data types to test (float32/complex64 and float64/complex128)
    dtypes = [
        {"name": "float32/complex64", "real_dtype": np.float32, "complex_dtype": np.complex64},
        {"name": "float64/complex128", "real_dtype": np.float64, "complex_dtype": np.complex128}
    ]
    
    # Transform types
    transforms = [
        {"name": "FFT", "func": "fft", "is_real": False},
        {"name": "Real FFT", "func": "rfft", "is_real": True}
    ]
    
    # Run benchmarks for each configuration
    for dim in dimensions:
        for transform in transforms:
            transform_name = transform["name"]
            func_name = transform["func"]
            is_real = transform["is_real"]
            
            # Get the right function name for each dimension
            if dim == 1:
                func_suffix = ""
            elif dim == 2:
                func_suffix = "2"
            else:  # dim == 3
                func_suffix = "n"
            
            full_func_name = func_name + func_suffix
            logger.info(f"Testing {dim}D {transform_name} ({full_func_name})...")
            
            # Different size limits based on dimension to avoid memory issues
            if dim == 1:
                sizes = all_sizes
            elif dim == 2:
                # Limit 2D arrays to avoid memory issues
                sizes = [s for s in all_sizes if s <= 4096]  # Expanded from 2048
            else:  # dim == 3
                # Limit 3D arrays to avoid memory issues
                sizes = [s for s in all_sizes if s <= 512]   # Expanded from 256
            
            for dtype_info in dtypes:
                dtype_name = dtype_info["name"]
                real_dtype = dtype_info["real_dtype"]
                complex_dtype = dtype_info["complex_dtype"]
                
                logger.info(f"  Testing dtype: {dtype_name}")
                
                for size in sizes:
                    # Determine if this is a power of 2
                    is_power_of_2 = (size & (size - 1) == 0)
                    
                    # Create shape based on dimension
                    if dim == 1:
                        shape = (size,)
                    elif dim == 2:
                        shape = (size, size)
                    else:  # dim == 3
                        shape = (size, size, size)
                    
                    logger.info(f"  Size {size} ({'power of 2' if is_power_of_2 else 'non-power of 2'})")
                    
                    # Dictionary to collect average times for each implementation
                    implementation_times = {impl_key: [] for impl_key, impl in FFT_IMPLEMENTATIONS.items() if impl["available"]}
                    
                    # Test multiple datasets for statistical significance
                    for dataset_idx in range(n_datasets):
                        logger.info(f"    Dataset {dataset_idx+1}/{n_datasets}")
                        
                        # Create test array with appropriate dtype
                        try:
                            if is_real:
                                # Real input for rfft
                                array = create_test_array(shape, dtype=real_dtype, is_complex=False)
                            else:
                                # Complex input for regular fft
                                array = create_test_array(shape, dtype=complex_dtype, is_complex=True)
                                
                            # Test each implementation
                            for impl_key, impl in FFT_IMPLEMENTATIONS.items():
                                if not impl["available"]:
                                    continue
                                
                                # Get the transform function
                                try:
                                    impl_module = impl["module"]
                                    impl_func = getattr(impl_module, full_func_name)
                                    
                                    # Setup implementation
                                    if impl["setup"]:
                                        impl["setup"]()
                                    
                                    # SIMPLIFIED APPROACH: Run warmup, then single measurement
                                    # Warmup run (not measured)
                                    _ = impl_func(array)
                                    
                                    # Single measured run
                                    _, execution_time = time_function(impl_func, array)
                                    
                                    # Store the time for averaging across datasets
                                    implementation_times[impl_key].append(execution_time)
                                    
                                    # Cleanup implementation
                                    if impl["cleanup"]:
                                        impl["cleanup"]()
                                    
                                    # Log result for this dataset
                                    logger.info(f"       {impl['name']}: {execution_time:.6f}s")
                                    
                                except Exception as e:
                                    logger.error(f"Error with {impl['name']} on {transform_name} size {size}: {str(e)}")
                                    logger.debug(traceback.format_exc())
                        except Exception as e:
                            logger.error(f"Error creating test array: {str(e)}")
                            logger.debug(traceback.format_exc())
                    
                    # After testing all datasets, compute and store average results
                    for impl_key, times in implementation_times.items():
                        if times:
                            avg_time = sum(times) / len(times)
                            impl_name = FFT_IMPLEMENTATIONS[impl_key]["name"]
                            
                            # Create result entry for plotting/saving
                            result = {
                                "dimension": dim,
                                "transform": transform_name,
                                "size": size,
                                "is_power_of_2": is_power_of_2,
                                "dtype": dtype_name,
                                "implementation": impl_name,
                                "time": avg_time
                            }
                            results.append(result)

def benchmark_repeated_use(n_runs=5, n_datasets=3, save_results=True):
    """
    Measure performance when the same transform is used repeatedly,
    showing the benefits of intelligent plan caching.
    
    Args:
        n_runs: Number of measurement runs
        n_datasets: Number of different random datasets to test
        save_results: Whether to save results to disk
    """
    logger.info("Starting enhanced repeated use benchmark...")
    
    results = []
    
    # Test configurations - extended with larger arrays
    transforms = [
        {"name": "1D FFT", "func": "fft", "shape": (2048,), "is_complex": True},
        {"name": "1D FFT (Large)", "func": "fft", "shape": (16384,), "is_complex": True},
        {"name": "1D FFT (Very Large)", "func": "fft", "shape": (131072,), "is_complex": True},
        {"name": "2D FFT", "func": "fft2", "shape": (512, 512), "is_complex": True},
        {"name": "2D FFT (Large)", "func": "fft2", "shape": (2048, 2048), "is_complex": True},
        {"name": "1D RFFT", "func": "rfft", "shape": (2048,), "is_complex": False},
        {"name": "1D RFFT (Large)", "func": "rfft", "shape": (32768,), "is_complex": False},
        {"name": "2D RFFT", "func": "rfft2", "shape": (512, 512), "is_complex": False},
        {"name": "3D FFT", "func": "fftn", "shape": (64, 64, 64), "is_complex": True},
        {"name": "3D FFT (Large)", "func": "fftn", "shape": (128, 128, 128), "is_complex": True},
    ]
    
    # More fine-grained repetition counts to see caching effects
    repetition_counts = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    
    # Add test for reusing same array vs creating new arrays
    reuse_strategies = [
        {"name": "New array each time", "reuse_array": False},
        {"name": "Reuse same array", "reuse_array": True}
    ]
    
    for transform in transforms:
        transform_name = transform["name"]
        func_name = transform["func"]
        shape = transform["shape"]
        is_complex = transform["is_complex"]
        
        logger.info(f"Testing {transform_name} with shape {shape}...")
        
        for dataset_idx in range(n_datasets):
            logger.info(f"  Dataset {dataset_idx+1}/{n_datasets}")
            
            # Create array once per dataset
            try:
                if is_complex:
                    array = create_test_array(shape, dtype=np.complex128, is_complex=True)
                else:
                    array = create_test_array(shape, dtype=np.float64, is_complex=False)
            
                for reuse_strategy in reuse_strategies:
                    strategy_name = reuse_strategy["name"]
                    reuse_array = reuse_strategy["reuse_array"]
                    
                    logger.info(f"    Strategy: {strategy_name}")
                    
                    for impl_key, impl in FFT_IMPLEMENTATIONS.items():
                        if not impl["available"]:
                            continue
                            
                        logger.info(f"      Implementation: {impl['name']}")
                        
                        # Get the transform function
                        try:
                            impl_module = impl["module"]
                            impl_func = getattr(impl_module, func_name)
                            
                            for repetitions in repetition_counts:
                                logger.info(f"        Repetitions: {repetitions}")
                                
                                # Run the test multiple times for statistical reliability
                                first_transform_times = []
                                total_times = []
                                
                                for run_idx in range(n_runs):
                                    # Setup implementation
                                    impl["setup"]()
                                    
                                    # Prepare arrays - either create copies or reuse the same array
                                    if reuse_array:
                                        # We'll reuse the same array for all repetitions
                                        array_copies = [array] * repetitions
                                    else:
                                        # Create separate copies for each repetition
                                        array_copies = [array.copy() for _ in range(repetitions)]
                                    
                                    # First transform (with planning)
                                    start_time = time.time()
                                    _ = impl_func(array_copies[0].copy() if reuse_array else array_copies[0])
                                    first_transform_time = time.time() - start_time
                                    first_transform_times.append(first_transform_time)
                                    
                                    # Measure total time for all repetitions
                                    start_time = time.time()
                                    for i in range(repetitions):
                                        _ = impl_func(array_copies[i].copy() if reuse_array else array_copies[i])
                                    total_time = time.time() - start_time
                                    total_times.append(total_time)
                                    
                                    # Cleanup
                                    impl["cleanup"]()
                                    force_gc()
                                
                                # Calculate statistics
                                avg_first_time = sum(first_transform_times) / len(first_transform_times)
                                avg_total_time = sum(total_times) / len(total_times)
                                std_total_time = np.std(total_times) if len(total_times) > 1 else 0
                                
                                # Amortized time per transform
                                amortized_time = avg_total_time / repetitions
                                
                                # Record result
                                result = {
                                    "implementation": impl["name"],
                                    "transform": transform_name,
                                    "shape": str(shape),
                                    "dataset_idx": dataset_idx,
                                    "strategy": strategy_name,
                                    "repetitions": repetitions,
                                    "first_transform_time": avg_first_time,
                                    "total_time": avg_total_time,
                                    "std_dev_total": std_total_time,
                                    "amortized_time": amortized_time,
                                    "amortization_factor": avg_first_time / amortized_time if amortized_time > 0 else 0,
                                }
                                
                                results.append(result)
                                logger.info(f"          First: {avg_first_time:.6f}s, Amortized: {amortized_time:.6f}s")
                                
                        except Exception as e:
                            logger.error(f"Error with {impl['name']} on {func_name}: {str(e)}")
                            traceback.print_exc()
            except Exception as e:
                logger.error(f"Error creating array with shape {shape}: {str(e)}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if save_results and not df.empty:
        csv_file = os.path.join(RESULTS_DIR, f"repeated_use_{TIMESTAMP}.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file}")
        
        # Create enhanced visualization
        # Group by transform type
        for transform_name in df['transform'].unique():
            transform_df = df[df['transform'] == transform_name]
            
            # Create plots for each reuse strategy
            for strategy in transform_df['strategy'].unique():
                strategy_df = transform_df[transform_df['strategy'] == strategy]
                
                plt.figure(figsize=(12, 9))
                
                # Plot 1: Amortized time vs repetitions
                plt.subplot(2, 1, 1)
                for impl_name in strategy_df['implementation'].unique():
                    impl_df = strategy_df[strategy_df['implementation'] == impl_name]
                    if not impl_df.empty:
                        # Group by repetitions and calculate mean and std
                        grouped = impl_df.groupby('repetitions')[['amortized_time']].agg(['mean', 'std']).reset_index()
                        # Flatten the multi-level columns
                        grouped.columns = ['repetitions', 'mean_time', 'std_time']
                        
                        # Plot with error bars
                        plt.errorbar(
                            grouped['repetitions'],
                            grouped['mean_time'],
                            yerr=grouped['std_time'],
                            marker='o',
                            label=impl_name
                        )
                
                plt.xscale('log', base=10)
                plt.yscale('log', base=10)
                plt.title(f'Amortized Time per Transform ({strategy})')
                plt.xlabel('Number of Repetitions')
                plt.ylabel('Amortized Time (seconds)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Plot 2: Amortization factor vs repetitions
                plt.subplot(2, 1, 2)
                for impl_name in strategy_df['implementation'].unique():
                    impl_df = strategy_df[strategy_df['implementation'] == impl_name]
                    if not impl_df.empty:
                        # Group by repetitions and calculate mean and std
                        grouped = impl_df.groupby('repetitions')[['amortization_factor']].agg(['mean', 'std']).reset_index()
                        # Flatten the multi-level columns
                        grouped.columns = ['repetitions', 'mean_factor', 'std_factor']
                        
                        # Plot with error bars
                        plt.errorbar(
                            grouped['repetitions'],
                            grouped['mean_factor'],
                            yerr=grouped['std_factor'],
                            marker='o',
                            label=impl_name
                        )
                
                plt.xscale('log', base=10)
                plt.title(f'Amortization Factor ({strategy})')
                plt.xlabel('Number of Repetitions')
                plt.ylabel('Amortization Factor (first_time / amortized_time)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.suptitle(f'{transform_name} - Plan Reuse Efficiency')
                plt.tight_layout()
                
                filename = f"repeated_use_{transform_name.replace(' ', '_')}_{strategy.replace(' ', '_')}_{TIMESTAMP}.png"
                plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
                plt.close()
        
        # Create a summary plot: Max amortization factor across implementations
        plt.figure(figsize=(14, 8))
        
        # Group by implementation and transform, find the max amortization at max repetitions
        max_reps = df['repetitions'].max()
        max_reps_df = df[df['repetitions'] == max_reps]
        
        pivot = max_reps_df.pivot_table(
            index='transform',
            columns=['implementation', 'strategy'],
            values='amortization_factor',
            aggfunc='mean'
        )
        
        if not pivot.empty:
            ax = pivot.plot(kind='bar', figsize=(14, 8))
            plt.title(f'Maximum Amortization Factor at {max_reps} Repetitions')
            plt.ylabel('Amortization Factor (first_time / amortized_time)')
            plt.xlabel('Transform Type')
            plt.grid(axis='y', alpha=0.3)
            plt.legend(title='Implementation / Strategy')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(os.path.join(RESULTS_DIR, f"max_amortization_factor_{TIMESTAMP}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    return df

def benchmark_real_world(n_runs=5, n_datasets=3, save_results=True):
    """
    Test with realistic application scenarios.
    
    Args:
        n_runs: Number of runs per scenario
        n_datasets: Number of different random datasets to test
        save_results: Whether to save results to disk
    """
    logger.info("Starting enhanced real-world scenario benchmark...")
    
    results = []
    
    # Define real-world scenarios - extended with larger arrays and more complex operations
    scenarios = [
        {
            "name": "Image Filtering",
            "description": "Apply a low-pass filter to an image in frequency domain",
            "shape": (1080, 1920),  # HD image
            "steps": [
                {"name": "Forward FFT", "func": "fft2", "is_complex": False},
                {"name": "Apply Filter", "is_filter": True},
                {"name": "Inverse FFT", "func": "ifft2", "is_complex": True}
            ]
        },
        {
            "name": "Image Filtering (4K)",
            "description": "Apply a low-pass filter to a 4K image",
            "shape": (2160, 3840),  # 4K image
            "steps": [
                {"name": "Forward FFT", "func": "fft2", "is_complex": False},
                {"name": "Apply Filter", "is_filter": True},
                {"name": "Inverse FFT", "func": "ifft2", "is_complex": True}
            ]
        },
        {
            "name": "Signal Analysis",
            "description": "Analyze a signal with FFT, apply window, calculate power spectrum",
            "shape": (16384,),  # Long 1D signal - doubled from original
            "steps": [
                {"name": "Forward FFT", "func": "rfft", "is_complex": False},
                {"name": "Calculate Power", "is_power": True},
                {"name": "Find Peaks", "is_peaks": True}
            ]
        },
        {
            "name": "Long Signal Analysis",
            "description": "Analyze a very long signal with FFT",
            "shape": (131072,),  # Very long 1D signal
            "steps": [
                {"name": "Forward FFT", "func": "rfft", "is_complex": False},
                {"name": "Calculate Power", "is_power": True},
                {"name": "Find Peaks", "is_peaks": True}
            ]
        },
        {
            "name": "Volume Processing",
            "description": "Process 3D medical imaging data with FFT",
            "shape": (64, 64, 64),  # Small 3D volume
            "steps": [
                {"name": "Forward FFT", "func": "fftn", "is_complex": False},
                {"name": "Apply Filter", "is_filter": True},
                {"name": "Inverse FFT", "func": "ifftn", "is_complex": True}
            ]
        },
        {
            "name": "Large Volume Processing",
            "description": "Process large 3D medical imaging data with FFT",
            "shape": (128, 128, 128),  # Larger 3D volume
            "steps": [
                {"name": "Forward FFT", "func": "fftn", "is_complex": False},
                {"name": "Apply Filter", "is_filter": True},
                {"name": "Inverse FFT", "func": "ifftn", "is_complex": True}
            ]
        },
        {
            "name": "Multiple FFT Pipeline",
            "description": "Pipeline with multiple FFTs",
            "shape": (1024, 1024),  # Medium 2D image
            "steps": [
                {"name": "First Forward FFT", "func": "fft2", "is_complex": False},
                {"name": "Apply Filter 1", "is_filter": True},
                {"name": "First Inverse FFT", "func": "ifft2", "is_complex": True},
                {"name": "Threshold", "is_threshold": True},
                {"name": "Second Forward FFT", "func": "fft2", "is_complex": False},
                {"name": "Apply Filter 2", "is_filter": True},
                {"name": "Second Inverse FFT", "func": "ifft2", "is_complex": True}
            ]
        }
    ]
    
    for scenario in scenarios:
        scenario_name = scenario["name"]
        shape = scenario["shape"]
        steps = scenario["steps"]
        
        logger.info(f"Testing scenario: {scenario_name} with shape {shape}")
        
        # Test with multiple datasets
        for dataset_idx in range(n_datasets):
            logger.info(f"  Dataset {dataset_idx+1}/{n_datasets}")
            
            # Create initial data once per dataset
            try:
                data_original = create_test_array(shape, dtype=np.float64, is_complex=False)
            
                for impl_key, impl in FFT_IMPLEMENTATIONS.items():
                    if not impl["available"]:
                        continue
                        
                    impl_module = impl["module"]
                    logger.info(f"    Implementation: {impl['name']}")
                    
                    # Run the scenario multiple times for statistical robustness
                    scenario_times = []
                    step_times_all = []
                    
                    for run in range(n_runs):
                        # Setup implementation
                        impl["setup"]()
                        
                        # Create a fresh copy of the data for this run
                        data = data_original.copy()
                        step_times = []
                        
                        # Process each step
                        for step in steps:
                            if "func" in step:
                                # FFT step
                                func_name = step["func"]
                                func = getattr(impl_module, func_name)
                                
                                start_time = time.time()
                                data = func(data)
                                step_time = time.time() - start_time
                                
                            elif step.get("is_filter", False):
                                # Apply filter in frequency domain
                                start_time = time.time()
                                
                                # Create a simple low-pass filter
                                if len(shape) == 1:
                                    # 1D filter: zero out high frequencies
                                    mid = len(data) // 2
                                    cutoff = mid // 4
                                    mask = np.ones_like(data)
                                    mask[cutoff:-cutoff] = 0.1
                                    data *= mask
                                    
                                elif len(shape) == 2:
                                    # 2D filter: radial low-pass
                                    rows, cols = data.shape
                                    crow, ccol = rows // 2, cols // 2
                                    
                                    # Create a mask with a circular center region = 1
                                    y, x = np.ogrid[:rows, :cols]
                                    mask = ((y - crow)**2 + (x - ccol)**2 <= (min(crow, ccol) // 3)**2)
                                    
                                    # Apply the mask
                                    data[~mask] *= 0.1
                                    
                                else:  # 3D
                                    # 3D filter: spherical low-pass
                                    d, h, w = data.shape
                                    cd, ch, cw = d // 2, h // 2, w // 2
                                    
                                    # Create coordinates
                                    z, y, x = np.ogrid[:d, :h, :w]
                                    
                                    # Create spherical mask
                                    mask = ((z - cd)**2 + (y - ch)**2 + (x - cw)**2 <= (min(cd, ch, cw) // 3)**2)
                                    
                                    # Apply the mask
                                    data[~mask] *= 0.1
                                
                                step_time = time.time() - start_time
                                
                            elif step.get("is_power", False):
                                # Calculate power spectrum
                                start_time = time.time()
                                data = np.abs(data)**2
                                step_time = time.time() - start_time
                                
                            elif step.get("is_peaks", False):
                                # Find peaks in spectrum
                                start_time = time.time()
                                # Simple peak finding - local maxima
                                peaks = (data[1:-1] > data[:-2]) & (data[1:-1] > data[2:])
                                peak_indices = np.where(peaks)[0] + 1
                                peak_values = data[peak_indices]
                                
                                # Keep only top 5 peaks
                                if len(peak_indices) > 5:
                                    top_indices = np.argsort(peak_values)[-5:]
                                    peak_indices = peak_indices[top_indices]
                                    peak_values = peak_values[top_indices]
                                
                                step_time = time.time() - start_time
                                
                            elif step.get("is_threshold", False):
                                # Apply thresholding
                                start_time = time.time()
                                threshold = np.median(np.abs(data))
                                data[np.abs(data) < threshold] = 0
                                step_time = time.time() - start_time
                                
                            else:
                                # Unknown step type
                                logger.warning(f"Unknown step type: {step}")
                                step_time = 0
                                
                            step_times.append(step_time)
                        
                        # Calculate total scenario time
                        total_time = sum(step_times)
                        scenario_times.append(total_time)
                        step_times_all.append(step_times)
                        
                        # Cleanup
                        impl["cleanup"]()
                        force_gc()
                    
                    # Calculate statistics
                    avg_scenario_time = sum(scenario_times) / len(scenario_times)
                    std_dev = np.std(scenario_times) if len(scenario_times) > 1 else 0
                    
                    # Calculate average time for each step across runs
                    avg_step_times = np.mean(step_times_all, axis=0) if step_times_all else []
                    
                    # Record result
                    result = {
                        "implementation": impl["name"],
                        "scenario": scenario_name,
                        "shape": str(shape),
                        "dataset_idx": dataset_idx,
                        "avg_time": avg_scenario_time,
                        "std_dev": std_dev,
                        "min_time": min(scenario_times),
                        "max_time": max(scenario_times),
                        "step_times": ','.join([f"{t:.6f}" for t in avg_step_times]) if avg_step_times else ""
                    }
                    
                    results.append(result)
                    logger.info(f"      Average time: {avg_scenario_time:.6f}s ±{std_dev:.6f}s")
            except Exception as e:
                logger.error(f"Error creating array with shape {shape}: {str(e)}")
                traceback.print_exc()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if save_results and not df.empty:
        csv_file = os.path.join(RESULTS_DIR, f"real_world_{TIMESTAMP}.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file}")
        
        # Create enhanced visualization
        plt.figure(figsize=(14, 10))
        
        # Group results by scenario and implementation, and compute statistics
        stats_df = df.groupby(['scenario', 'implementation']).agg({
            'avg_time': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        # Flatten multi-level columns
        stats_df.columns = ['scenario', 'implementation', 'mean_time', 'std_time', 'min_time', 'max_time']
        
        # Create a grouped bar chart with error bars
        plt.figure(figsize=(14, 10))
        for i, scenario in enumerate(stats_df['scenario'].unique()):
            scenario_df = stats_df[stats_df['scenario'] == scenario]
            x = np.arange(len(scenario_df))
            width = 0.8
            
            plt.subplot(len(stats_df['scenario'].unique()) // 2 + 1, 2, i + 1)
            bars = plt.bar(x, scenario_df['mean_time'], width, yerr=scenario_df['std_time'],
                          capsize=5, label=scenario_df['implementation'])
            
            plt.title(scenario)
            plt.xticks(x, scenario_df['implementation'], rotation=45, ha='right')
            plt.ylabel('Time (seconds)')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}',
                        ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"real_world_scenarios_{TIMESTAMP}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create speedup comparison plot
        # Pivot to calculate speedup vs NumPy
        pivot = df.pivot_table(index=['scenario', 'dataset_idx'], columns='implementation', values='avg_time')
        
        if 'NumPy FFT' in pivot.columns:
            for col in pivot.columns:
                if col != 'NumPy FFT':
                    pivot[f"{col} Speedup"] = pivot['NumPy FFT'] / pivot[col]
            
            # Calculate average speedup by scenario
            speedup_by_scenario = pivot.groupby('scenario').mean()
            
            # Plot speedups
            speedup_cols = [col for col in speedup_by_scenario.columns if 'Speedup' in col]
            if speedup_cols:
                plt.figure(figsize=(12, 8))
                ax = speedup_by_scenario[speedup_cols].plot(kind='bar', figsize=(12, 8))
                plt.title('Speedup vs NumPy in Real-World Scenarios')
                plt.ylabel('Speedup Factor')
                plt.grid(axis='y', alpha=0.3)
                plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Add text labels
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.2f', padding=3)
                
                plt.savefig(os.path.join(RESULTS_DIR, f"real_world_speedup_{TIMESTAMP}.png"), dpi=300, bbox_inches='tight')
                plt.close()
    
    return df

def benchmark_thread_scaling(n_runs=5, n_datasets=3, save_results=True):
    """
    Measure how performance scales with thread count for large arrays.
    
    Args:
        n_runs: Number of runs per test configuration
        n_datasets: Number of different random datasets to test
        save_results: Whether to save results to disk
    """
    logger.info("Starting enhanced thread scaling benchmark...")
    
    # Skip if only NumPy/SciPy available (no threading support)
    if not HAVE_PYFFTW and not HAVE_BETTERFFTW:
        logger.warning("Skipping thread scaling benchmark: No threading-capable libraries available")
        return pd.DataFrame()
    
    results = []
    
    # Test configurations - extended with larger arrays and varied data types
    transform_types = [
        {"name": "2D FFT", "func": "fft2", "shape": (2048, 2048), "is_complex": False},
        {"name": "2D FFT (Large)", "func": "fft2", "shape": (4096, 4096), "is_complex": False},
        {"name": "2D FFT (Very Large)", "func": "fft2", "shape": (8192, 8192), "is_complex": False},
        {"name": "3D FFT", "func": "fftn", "shape": (128, 128, 128), "is_complex": False},
        {"name": "3D FFT (Large)", "func": "fftn", "shape": (256, 256, 256), "is_complex": False}
    ]
    
    # Data types to test
    dtypes = [
        {"name": "float32", "real_dtype": np.float32, "complex_dtype": np.complex64},
        {"name": "float64", "real_dtype": np.float64, "complex_dtype": np.complex128}
    ]
    
    # Get available CPU count
    max_threads = os.cpu_count()
    
    # Create more comprehensive thread counts
    thread_counts = [1]  # Always test single-threaded
    
    # Add powers of 2 thread counts
    for i in range(1, int(np.log2(max_threads)) + 1):
        thread_count = 2**i
        if thread_count <= max_threads:
            thread_counts.append(thread_count)
    
    # Add some non-power-of-2 thread counts if applicable
    if max_threads >= 6 and 6 not in thread_counts:
        thread_counts.append(6)
    if max_threads >= 10 and 10 not in thread_counts:
        thread_counts.append(10)
    if max_threads >= 12 and 12 not in thread_counts:
        thread_counts.append(12)
    
    # Ensure max_threads is included
    if max_threads not in thread_counts:
        thread_counts.append(max_threads)
    
    # Sort thread counts
    thread_counts = sorted(thread_counts)
    
    # Filter for threading-capable implementations
    threading_impls = {
        k: v for k, v in FFT_IMPLEMENTATIONS.items()
        if k in ['pyfftw', 'betterfftw'] and v['available']
    }
    
    # Add NumPy as reference (which doesn't support threading directly)
    if 'numpy' in FFT_IMPLEMENTATIONS and FFT_IMPLEMENTATIONS['numpy']['available']:
        threading_impls['numpy'] = FFT_IMPLEMENTATIONS['numpy']
    
    for transform in transform_types:
        transform_name = transform["name"]
        func_name = transform["func"]
        shape = transform["shape"]
        is_complex = transform["is_complex"]
        
        logger.info(f"Testing {transform_name} with shape {shape}...")
        
        for dtype_info in dtypes:
            dtype_name = dtype_info["name"]
            real_dtype = dtype_info["real_dtype"]
            complex_dtype = dtype_info["complex_dtype"]
            
            logger.info(f"  Testing dtype: {dtype_name}")
            
            # Test with multiple datasets
            for dataset_idx in range(n_datasets):
                logger.info(f"    Dataset {dataset_idx+1}/{n_datasets}")
                
                # Create test array with appropriate dtype
                try:
                    array = create_test_array(
                        shape, 
                        dtype=complex_dtype if is_complex else real_dtype, 
                        is_complex=is_complex
                    )
                    
                    # Get single-thread reference times for all implementations
                    reference_times = {}
                    
                    for impl_key, impl in threading_impls.items():
                        impl_module = impl["module"]
                        impl_func = getattr(impl_module, func_name)
                        
                        # Setup implementation
                        impl["setup"]()
                        
                        # Get single-thread time (will always be available)
                        kwarg = {"threads": 1} if impl_key != 'numpy' else {}
                        
                        # Warm-up
                        _ = impl_func(array.copy(), **kwarg)
                        
                        # Measure
                        times = []
                        for _ in range(n_runs):
                            array_copy = array.copy()
                            force_gc()
                            
                            start_time = time.time()
                            _ = impl_func(array_copy, **kwarg)
                            end_time = time.time()
                            
                            times.append(end_time - start_time)
                            
                        # Store reference time
                        reference_times[impl_key] = sum(times) / len(times)
                        
                        # Cleanup
                        impl["cleanup"]()
                        force_gc()
                    
                    # Test each thread count
                    for thread_count in thread_counts:
                        logger.info(f"      Threads: {thread_count}")
                        
                        for impl_key, impl in threading_impls.items():
                            # Skip thread scaling for NumPy above 1 thread
                            if impl_key == 'numpy' and thread_count > 1:
                                continue
                                
                            logger.info(f"        Implementation: {impl['name']}")
                            
                            impl_module = impl["module"]
                            impl_func = getattr(impl_module, func_name)
                            
                            # Setup implementation
                            impl["setup"]()
                            
                            # Set thread count (except for NumPy which doesn't support it)
                            kwarg = {"threads": thread_count} if impl_key != 'numpy' else {}
                            
                            # Warm-up
                            _ = impl_func(array.copy(), **kwarg)
                            
                            # Measure performance over multiple runs
                            times = []
                            memory_readings = []
                            
                            for run_idx in range(n_runs):
                                array_copy = array.copy()
                                force_gc()
                                
                                # Check memory usage before
                                mem_before = get_memory_usage()
                                
                                start_time = time.time()
                                _ = impl_func(array_copy, **kwarg)
                                end_time = time.time()
                                
                                times.append(end_time - start_time)
                                
                                # Check memory after
                                mem_after = get_memory_usage()
                                memory_readings.append(mem_after - mem_before)
                            
                            # Calculate statistics
                            avg_time = sum(times) / len(times)
                            std_dev = np.std(times) if len(times) > 1 else 0
                            avg_memory_delta = sum(memory_readings) / len(memory_readings) if memory_readings else 0
                            
                            # Calculate speedup vs. 1 thread
                            reference_time = reference_times[impl_key]
                            speedup = reference_time / avg_time if avg_time > 0 else 0
                            
                            # Calculate efficiency (speedup per thread)
                            efficiency = speedup / thread_count if thread_count > 0 else 0
                            
                            # Record result
                            result = {
                                "implementation": impl["name"],
                                "transform": transform_name,
                                "shape": str(shape),
                                "dtype": dtype_name,
                                "dataset_idx": dataset_idx,
                                "thread_count": thread_count,
                                "avg_time": avg_time,
                                "std_dev": std_dev,
                                "reference_time": reference_time,
                                "speedup": speedup,
                                "efficiency": efficiency,
                                "memory_delta_mb": avg_memory_delta,
                                "theoretical_max_speedup": min(thread_count, array.ndim)  # Theoretical limit based on dimensions
                            }
                            
                            results.append(result)
                            logger.info(f"          Time: {avg_time:.6f}s ±{std_dev:.6f}s, Speedup: {speedup:.2f}x")
                            
                            # Cleanup
                            impl["cleanup"]()
                            force_gc()
                except Exception as e:
                    logger.error(f"Error processing array with shape {shape}: {str(e)}")
                    traceback.print_exc()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if save_results and not df.empty:
        csv_file = os.path.join(RESULTS_DIR, f"thread_scaling_{TIMESTAMP}.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file}")
        
        # Create enhanced visualization
        # Create speedup plot for each transform + dtype combination
        for transform_name in df['transform'].unique():
            for dtype_name in df['dtype'].unique():
                transform_df = df[(df['transform'] == transform_name) & (df['dtype'] == dtype_name)]
                if transform_df.empty:
                    continue
                    
                plt.figure(figsize=(12, 10))
                
                # Plot 1: Speedup vs thread count
                plt.subplot(2, 1, 1)
                for impl_name in transform_df['implementation'].unique():
                    impl_df = transform_df[transform_df['implementation'] == impl_name]
                    if impl_df.empty or len(impl_df['thread_count'].unique()) <= 1:
                        continue
                        
                    # Group by thread count and calculate statistics
                    grouped = impl_df.groupby('thread_count')[['speedup']].agg(['mean', 'std']).reset_index()
                    # Flatten multi-level columns
                    grouped.columns = ['thread_count', 'mean_speedup', 'std_speedup']
                    
                    # Plot with error bars
                    plt.errorbar(
                        grouped['thread_count'],
                        grouped['mean_speedup'],
                        yerr=grouped['std_speedup'],
                        marker='o',
                        label=impl_name
                    )
                
                # Add ideal scaling reference line
                if len(transform_df['thread_count'].unique()) > 1:
                    max_threads_shown = max(transform_df['thread_count'].unique())
                    plt.plot([1, max_threads_shown], [1, max_threads_shown], 'k--', alpha=0.5, label='Ideal Scaling')
                
                plt.title(f'Thread Scaling for {transform_name} ({dtype_name})')
                plt.xlabel('Thread Count')
                plt.ylabel('Speedup vs. Single Thread')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Plot 2: Efficiency vs thread count
                plt.subplot(2, 1, 2)
                for impl_name in transform_df['implementation'].unique():
                    impl_df = transform_df[transform_df['implementation'] == impl_name]
                    if impl_df.empty or len(impl_df['thread_count'].unique()) <= 1:
                        continue
                        
                    # Group by thread count and calculate statistics
                    grouped = impl_df.groupby('thread_count')[['efficiency']].agg(['mean', 'std']).reset_index()
                    # Flatten multi-level columns
                    grouped.columns = ['thread_count', 'mean_efficiency', 'std_efficiency']
                    
                    # Plot with error bars
                    plt.errorbar(
                        grouped['thread_count'],
                        grouped['mean_efficiency'],
                        yerr=grouped['std_efficiency'],
                        marker='o',
                        label=impl_name
                    )
                
                # Add reference line for 100% efficiency
                plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='100% Efficiency')
                
                plt.title(f'Thread Efficiency for {transform_name} ({dtype_name})')
                plt.xlabel('Thread Count')
                plt.ylabel('Efficiency (Speedup / Thread Count)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                plt.tight_layout()
                
                # Save figure
                filename = f"thread_scaling_{transform_name.replace(' ', '_')}_{dtype_name}_{TIMESTAMP}.png"
                plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
                plt.close()
        
        # Create a summary plot: maximum achieved speedup by implementation
        if 'implementation' in df.columns and 'speedup' in df.columns:
            plt.figure(figsize=(12, 8))
            
            # Find max speedup for each implementation and transform
            max_speedup = df.groupby(['implementation', 'transform', 'dtype'])['speedup'].max().reset_index()
            
            # Create a grouped bar chart
            g = sns.catplot(
                data=max_speedup, 
                x='transform', 
                y='speedup', 
                hue='implementation',
                col='dtype',
                kind='bar',
                height=6, aspect=1.5
            )
            
            g.set_xticklabels(rotation=45, ha='right')
            g.set_titles("{col_name}")
            g.fig.suptitle('Maximum Achieved Speedup by Implementation')
            g.set_axis_labels("Transform Type", "Maximum Speedup")
            
            # Add a horizontal line at y=1 (no speedup)
            for ax in g.axes.flat:
                ax.axhline(y=1, color='r', linestyle='--', alpha=0.5)
                ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Adjust for suptitle
            
            plt.savefig(os.path.join(RESULTS_DIR, f"max_speedup_summary_{TIMESTAMP}.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    return df

def benchmark_memory_efficiency(n_runs=3, save_results=True):
    """
    Benchmark how efficiently each implementation uses memory during large transforms.
    
    Args:
        n_runs: Number of runs per test configuration
        save_results: Whether to save results to disk
    """
    logger.info("Starting memory efficiency benchmark...")
    
    results = []
    
    # Test large array sizes that approach system memory limits
    large_sizes = [
        (8192, 8192),      # 2D: 512MB for complex128
        (16384, 16384),    # 2D: 2GB for complex128
        (1024, 1024, 128)  # 3D: 1GB for complex128
    ]
    
    # Data types to test
    dtypes = [
        {"name": "complex64", "dtype": np.complex64},
        {"name": "complex128", "dtype": np.complex128}
    ]
    
    for shape in large_sizes:
        dim = len(shape)
        func_name = "fft2" if dim == 2 else "fftn"
        
        logger.info(f"Testing {dim}D FFT with shape {shape}...")
        
        for dtype_info in dtypes:
            dtype_name = dtype_info["name"]
            dtype = dtype_info["dtype"]
            
            logger.info(f"  Testing dtype: {dtype_name}")
            
            # Only run if we have enough memory for this test (rough estimation)
            array_size_gb = np.prod(shape) * np.dtype(dtype).itemsize / (1024**3)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            if array_size_gb * 3 > available_memory_gb:  # Need ~3x the array size for transforms
                logger.warning(f"  Skipping test due to insufficient memory (need ~{array_size_gb*3:.1f}GB, have {available_memory_gb:.1f}GB)")
                continue
            
            for impl_key, impl in FFT_IMPLEMENTATIONS.items():
                if not impl["available"]:
                    continue
                    
                logger.info(f"    Implementation: {impl['name']}")
                
                # Get the transform function
                try:
                    impl_module = impl["module"]
                    impl_func = getattr(impl_module, func_name)
                    
                    # Measure across multiple runs
                    execution_times = []
                    peak_memories = []
                    
                    for run_idx in range(n_runs):
                        # Setup implementation
                        impl["setup"]()
                        
                        # Create test array - fresh each time to ensure clean state
                        array = create_test_array(shape, dtype=dtype, is_complex=True)
                        
                        # Measure memory before
                        mem_before = get_memory_usage()
                        
                        # Force garbage collection
                        force_gc()
                        
                        # Set up memory monitoring thread
                        memory_readings = []
                        stop_monitoring = False
                        
                        def monitor_memory():
                            process = psutil.Process(os.getpid())
                            while not stop_monitoring:
                                memory_readings.append(process.memory_info().rss / (1024 * 1024))
                                time.sleep(0.01)  # 10ms sampling
                        
                        # Start memory monitoring
                        monitor_thread = threading.Thread(target=monitor_memory)
                        monitor_thread.daemon = True
                        monitor_thread.start()
                        
                        # Perform transform
                        start_time = time.time()
                        result = impl_func(array)
                        end_time = time.time()
                        execution_time = end_time - start_time
                        
                        # Stop monitoring
                        stop_monitoring = True
                        monitor_thread.join()
                        
                        # Calculate peak memory during operation
                        if memory_readings:
                            peak_memory = max(memory_readings)
                            peak_memory_delta = peak_memory - mem_before
                        else:
                            # Fallback if monitoring failed
                            mem_after = get_memory_usage()
                            peak_memory = mem_after
                            peak_memory_delta = mem_after - mem_before
                        
                        # Record stats
                        execution_times.append(execution_time)
                        peak_memories.append(peak_memory_delta)
                        
                        # Cleanup
                        impl["cleanup"]()
                        force_gc()
                        
                        logger.info(f"      Run {run_idx+1}: {execution_time:.3f}s, +{peak_memory_delta:.2f}MB memory")
                    
                    # Calculate statistics
                    avg_time = sum(execution_times) / len(execution_times)
                    avg_peak_memory = sum(peak_memories) / len(peak_memories)
                    
                    # Calculate memory efficiency (throughput per MB)
                    array_size_mb = np.prod(shape) * np.dtype(dtype).itemsize / (1024 * 1024)
                    throughput = array_size_mb / avg_time  # MB/s
                    memory_efficiency = throughput / avg_peak_memory if avg_peak_memory > 0 else 0
                    
                    # Record result
                    result = {
                        "implementation": impl["name"],
                        "shape": str(shape),
                        "dimensions": dim,
                        "dtype": dtype_name,
                        "array_size_mb": array_size_mb,
                        "avg_execution_time": avg_time,
                        "avg_peak_memory_delta_mb": avg_peak_memory,
                        "throughput_mbs": throughput,
                        "memory_efficiency": memory_efficiency
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error with {impl['name']} on {func_name} shape {shape}: {str(e)}")
                    traceback.print_exc()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if save_results and not df.empty:
        csv_file = os.path.join(RESULTS_DIR, f"memory_efficiency_{TIMESTAMP}.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file}")
        
        # Create summary visualization
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Memory usage by implementation
        plt.subplot(2, 2, 1)
        sns.barplot(x='implementation', y='avg_peak_memory_delta_mb', hue='dtype', data=df)
        plt.title('Average Peak Memory Usage')
        plt.xlabel('Implementation')
        plt.ylabel('Peak Memory Usage (MB)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Plot 2: Throughput by implementation
        plt.subplot(2, 2, 2)
        sns.barplot(x='implementation', y='throughput_mbs', hue='dtype', data=df)
        plt.title('Data Throughput')
        plt.xlabel('Implementation')
        plt.ylabel('Throughput (MB/s)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Plot 3: Memory efficiency by implementation
        plt.subplot(2, 1, 2)
        sns.barplot(x='implementation', y='memory_efficiency', hue='dtype', data=df)
        plt.title('Memory Efficiency (Throughput per MB of Memory Used)')
        plt.xlabel('Implementation')
        plt.ylabel('Efficiency (Throughput/Memory)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"memory_efficiency_{TIMESTAMP}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    return df

def benchmark_non_power_of_two(n_runs=5, n_datasets=3, save_results=True):
    """
    Specifically benchmark performance on non-power-of-2 sizes, 
    including prime number dimensions and other challenging cases.
    
    Args:
        n_runs: Number of runs per test configuration
        n_datasets: Number of different random datasets to test
        save_results: Whether to save results to disk
    """
    logger.info("Starting non-power-of-two benchmark...")
    
    results = []
    
    # 1D test sizes - various non-power-of-2 cases
    sizes_1d = [
        1000,     # 2^3 * 5^3
        1331,     # 11^3 (prime cubed)
        1920,     # 2^6 * 3 * 5 (HD width)
        1999,     # Prime number
        2187,     # 3^7 (power of 3)
        10000,    # 2^4 * 5^4
        12289,    # Prime number
        15625,    # 5^6 (power of 5)
        20000,    # 2^5 * 5^4
        32805,    # 3 * 5 * 7 * 313 (multiple prime factors)
        65537,    # Fermat prime (2^16 + 1)
        100000,   # 2^5 * 5^5
    ]
    
    # 2D test sizes - pairs of challenging sizes
    sizes_2d = [
        (720, 480),       # DVD resolution
        (1280, 720),      # 720p
        (1920, 1080),     # 1080p
        (2560, 1440),     # 1440p
        (3840, 2160),     # 4K UHD
        (1331, 1331),     # 11^3 * 11^3 (prime cubed)
        (1024, 1999),     # Power of 2 × prime
        (2187, 2187),     # 3^7 * 3^7 (power of 3)
    ]
    
    # Transform functions to test
    transforms = [
        {"name": "FFT", "func": "fft", "is_real": False},
        {"name": "Real FFT", "func": "rfft", "is_real": True}
    ]
    
    # 1D Transform Tests
    logger.info("Testing 1D non-power-of-2 transforms...")
    for transform in transforms:
        transform_name = transform["name"]
        func_name = transform["func"]
        is_real = transform["is_real"]
        
        logger.info(f"  Testing {transform_name}...")
        
        for size in sizes_1d:
            logger.info(f"    Size: {size}")
            
            # Calculate if this is a power of 2 (for verification)
            is_power_of_2 = (size & (size - 1) == 0)
            if is_power_of_2:
                logger.warning(f"      Warning: {size} is actually a power of 2")
            
            # Find prime factors for labeling
            def get_prime_factors(n):
                factors = []
                d = 2
                while n > 1:
                    while n % d == 0:
                        factors.append(d)
                        n //= d
                    d += 1
                    if d*d > n:
                        if n > 1: 
                            factors.append(n)
                        break
                return factors
            
            prime_factors = get_prime_factors(size)
            factor_str = '*'.join(map(str, prime_factors))
            
            for dataset_idx in range(n_datasets):
                # Create test array
                try:
                    if is_real:
                        array = create_test_array((size,), dtype=np.float64, is_complex=False)
                    else:
                        array = create_test_array((size,), dtype=np.complex128, is_complex=True)
                    
                    for impl_key, impl in FFT_IMPLEMENTATIONS.items():
                        if not impl["available"]:
                            continue
                        
                        logger.info(f"      Implementation: {impl['name']}")
                        
                        # Get the transform function
                        try:
                            impl_module = impl["module"]
                            impl_func = getattr(impl_module, func_name)
                            
                            # Setup implementation
                            impl["setup"]()
                            
                            # First run (includes planning overhead)
                            array_copy = array.copy()
                            _, first_run_time = time_function(impl_func, array_copy)
                            
                            # Measure multiple runs for execution time
                            execution_times = []
                            for _ in range(n_runs):
                                array_copy = array.copy()
                                _, run_time = time_function(impl_func, array_copy)
                                execution_times.append(run_time)
                            
                            # Calculate statistics
                            avg_execution_time = sum(execution_times) / len(execution_times)
                            std_deviation = np.std(execution_times) if len(execution_times) > 1 else 0
                            planning_overhead = first_run_time - avg_execution_time
                            
                            # Record result
                            result = {
                                "implementation": impl["name"],
                                "dimension": 1,
                                "transform": transform_name,
                                "size": size,
                                "prime_factors": factor_str,
                                "dataset_idx": dataset_idx,
                                "first_run_time": first_run_time,
                                "avg_execution_time": avg_execution_time,
                                "std_deviation": std_deviation,
                                "planning_overhead": planning_overhead,
                            }
                            
                            results.append(result)
                            logger.info(f"        Avg Time: {avg_execution_time:.6f}s ±{std_deviation:.6f}s")
                            
                            # Cleanup
                            impl["cleanup"]()
                            force_gc()
                            
                        except Exception as e:
                            logger.error(f"Error with {impl['name']} on {func_name}: {str(e)}")
                            traceback.print_exc()
                            
                except Exception as e:
                    logger.error(f"Error creating array with size {size}: {str(e)}")
                    traceback.print_exc()
    
    # 2D Transform Tests
    logger.info("Testing 2D non-power-of-2 transforms...")
    for transform in transforms:
        transform_name = transform["name"]
        func_name = transform["func"] + "2"  # Use 2D version (fft2, rfft2)
        is_real = transform["is_real"]
        
        logger.info(f"  Testing {transform_name}...")
        
        for shape in sizes_2d:
            logger.info(f"    Shape: {shape}")
            
            # Calculate if dimensions are powers of 2
            is_power_of_2_x = (shape[0] & (shape[0] - 1) == 0)
            is_power_of_2_y = (shape[1] & (shape[1] - 1) == 0)
            
            if is_power_of_2_x and is_power_of_2_y:
                logger.warning(f"      Warning: shape {shape} has all power-of-2 dimensions")
            
            # Find prime factors for labeling
            x_factors = get_prime_factors(shape[0])
            y_factors = get_prime_factors(shape[1])
            factor_str = f"({','.join(map(str, x_factors))}) × ({','.join(map(str, y_factors))})"
            
            for dataset_idx in range(n_datasets):
                # Create test array
                try:
                    if is_real:
                        array = create_test_array(shape, dtype=np.float64, is_complex=False)
                    else:
                        array = create_test_array(shape, dtype=np.complex128, is_complex=True)
                    
                    for impl_key, impl in FFT_IMPLEMENTATIONS.items():
                        if not impl["available"]:
                            continue
                        
                        logger.info(f"      Implementation: {impl['name']}")
                        
                        # Get the transform function
                        try:
                            impl_module = impl["module"]
                            impl_func = getattr(impl_module, func_name)
                            
                            # Setup implementation
                            impl["setup"]()
                            
                            # First run (includes planning overhead)
                            array_copy = array.copy()
                            _, first_run_time = time_function(impl_func, array_copy)
                            
                            # Measure multiple runs for execution time
                            execution_times = []
                            for _ in range(n_runs):
                                array_copy = array.copy()
                                _, run_time = time_function(impl_func, array_copy)
                                execution_times.append(run_time)
                            
                            # Calculate statistics
                            avg_execution_time = sum(execution_times) / len(execution_times)
                            std_deviation = np.std(execution_times) if len(execution_times) > 1 else 0
                            planning_overhead = first_run_time - avg_execution_time
                            
                            # Record result
                            result = {
                                "implementation": impl["name"],
                                "dimension": 2,
                                "transform": transform_name,
                                "shape": str(shape),
                                "size_x": shape[0],
                                "size_y": shape[1],
                                "prime_factors": factor_str,
                                "dataset_idx": dataset_idx,
                                "first_run_time": first_run_time,
                                "avg_execution_time": avg_execution_time,
                                "std_deviation": std_deviation,
                                "planning_overhead": planning_overhead,
                            }
                            
                            results.append(result)
                            logger.info(f"        Avg Time: {avg_execution_time:.6f}s ±{std_deviation:.6f}s")
                            
                            # Cleanup
                            impl["cleanup"]()
                            force_gc()
                            
                        except Exception as e:
                            logger.error(f"Error with {impl['name']} on {func_name}: {str(e)}")
                            traceback.print_exc()
                            
                except Exception as e:
                    logger.error(f"Error creating array with shape {shape}: {str(e)}")
                    traceback.print_exc()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if save_results and not df.empty:
        csv_file = os.path.join(RESULTS_DIR, f"non_power_of_two_{TIMESTAMP}.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file}")
        
        # Create visualizations
        # 1D transform comparison
        plt.figure(figsize=(14, 10))
        
        # Create subplot for each transform type
        for transform_name in df[df['dimension'] == 1]['transform'].unique():
            transform_df = df[(df['dimension'] == 1) & (df['transform'] == transform_name)]
            
            plt.figure(figsize=(14, 8))
            
            # Plot performance vs size
            plt.subplot(2, 1, 1)
            for impl_name in transform_df['implementation'].unique():
                impl_df = transform_df[transform_df['implementation'] == impl_name]
                
                # Group by size and calculate statistics
                grouped = impl_df.groupby('size')[['avg_execution_time']].agg(['mean', 'std']).reset_index()
                # Flatten multi-level columns
                grouped.columns = ['size', 'mean_time', 'std_time']
                
                # Sort by size for proper line plotting
                grouped = grouped.sort_values('size')
                
                plt.errorbar(
                    grouped['size'],
                    grouped['mean_time'],
                    yerr=grouped['std_time'],
                    marker='o',
                    label=impl_name
                )
            
            plt.title(f'1D {transform_name} Performance on Non-Power-of-2 Sizes')
            plt.xlabel('Size')
            plt.ylabel('Execution Time (seconds)')
            plt.xscale('log', base=10)
            plt.yscale('log', base=10)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Plot planning overhead as percentage of execution time
            plt.subplot(2, 1, 2)
            for impl_name in transform_df['implementation'].unique():
                impl_df = transform_df[transform_df['implementation'] == impl_name]
                
                # Group by size and calculate statistics
                grouped = impl_df.groupby('size').agg({
                    'planning_overhead': 'mean',
                    'avg_execution_time': 'mean'
                }).reset_index()
                
                # Calculate overhead percentage
                grouped['overhead_percent'] = grouped['planning_overhead'] / grouped['avg_execution_time'] * 100
                
                # Sort by size for proper line plotting
                grouped = grouped.sort_values('size')
                
                plt.plot(
                    grouped['size'],
                    grouped['overhead_percent'],
                    marker='o',
                    label=impl_name
                )
            
            plt.title(f'Planning Overhead for 1D {transform_name} (Non-Power-of-2)')
            plt.xlabel('Size')
            plt.ylabel('Planning Overhead (% of execution time)')
            plt.xscale('log', base=10)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"non_power_of_two_1d_{transform_name.replace(' ', '_')}_{TIMESTAMP}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2D transform comparison
        for transform_name in df[df['dimension'] == 2]['transform'].unique():
            transform_df = df[(df['dimension'] == 2) & (df['transform'] == transform_name)]
            
            # Create a figure for relative performance
            plt.figure(figsize=(12, 8))
            
            # Calculate relative performance compared to NumPy
            if 'NumPy FFT' in transform_df['implementation'].unique():
                # Create a pivot table with shapes as rows and implementations as columns
                pivot = transform_df.pivot_table(
                    index=['shape', 'dataset_idx'],
                    columns='implementation',
                    values='avg_execution_time'
                )
                
                # Calculate speedup ratios
                for impl in pivot.columns:
                    if impl != 'NumPy FFT':
                        pivot[f"{impl}/NumPy"] = pivot['NumPy FFT'] / pivot[impl]
                
                # Average across datasets and reset index
                speedup_cols = [col for col in pivot.columns if '/NumPy' in col]
                avg_speedup = pivot.groupby('shape')[speedup_cols].mean().reset_index()
                
                # Plot speedups
                ax = avg_speedup.plot(x='shape', kind='bar', figsize=(12, 8))
                plt.title(f'2D {transform_name} Speedup vs NumPy on Non-Power-of-2 Shapes')
                plt.xlabel('Shape')
                plt.ylabel('Speedup Factor')
                plt.grid(axis='y', alpha=0.3)
                plt.axhline(y=1, color='r', linestyle='--', alpha=0.5)
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Implementation')
                plt.tight_layout()
                
                plt.savefig(os.path.join(RESULTS_DIR, f"non_power_of_two_2d_speedup_{transform_name.replace(' ', '_')}_{TIMESTAMP}.png"), dpi=300, bbox_inches='tight')
                plt.close()
    
    return df

def create_enhanced_summary_report(results_dict, save_file=None):
    """Create a comprehensive summary report of all benchmarks with statistical analysis."""
    logger.info("Creating enhanced summary report...")
    
    report = ["# BetterFFTW Enhanced Benchmark Summary Report\n"]
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # System information
    sys_info = {}
    sys_info["os"] = os.name
    sys_info["cpu_count"] = os.cpu_count()
    sys_info["memory_gb"] = psutil.virtual_memory().total / (1024**3)
    sys_info["python_version"] = sys.version
    sys_info["numpy_version"] = np.__version__
    sys_info["scipy_version"] = scipy.__version__
    sys_info["pyfftw_version"] = pyfftw.__version__ if HAVE_PYFFTW else "Not installed"
    sys_info["betterfftw_version"] = betterfftw.__version__
    sys_info["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if NumPy is using MKL
    np_config = str(np.__config__).lower()
    sys_info["numpy_using_mkl"] = "mkl" in np_config or "intel" in np_config
    
    report.append("## System Information\n")
    for key, value in sys_info.items():
        report.append(f"* **{key}**: {value}")
    report.append("\n")
    
    # Direct comparison summary with statistical analysis
    if 'direct_comparison' in results_dict:
        df = results_dict['direct_comparison']
        report.append("## Direct Comparison Summary\n")
        
        # Add statistical analysis
        if 'implementation' in df.columns and 'avg_execution_time' in df.columns:
            # Calculate average speedups vs NumPy with confidence intervals
            pivot = df.pivot_table(
                index=['dimension', 'transform', 'size', 'dataset_idx'],
                columns='implementation',
                values='avg_execution_time'
            )
            
            if 'NumPy FFT' in pivot.columns:
                speedup_stats = {}
                
                for impl in pivot.columns:
                    if impl != 'NumPy FFT':
                        # Calculate speedup ratio
                        speedup = pivot['NumPy FFT'] / pivot[impl]
                        
                        # Get statistics by dimension
                        for dim in df['dimension'].unique():
                            dim_speedup = speedup[speedup.index.get_level_values('dimension') == dim]
                            
                            if not dim_speedup.empty:
                                key = f"{impl}_vs_NumPy_{dim}D"
                                speedup_stats[key] = {
                                    'mean': dim_speedup.mean(),
                                    'median': dim_speedup.median(),
                                    'min': dim_speedup.min(),
                                    'max': dim_speedup.max(),
                                    'std': dim_speedup.std(),
                                    # Calculate 95% confidence interval
                                    'ci_95_low': dim_speedup.mean() - 1.96 * dim_speedup.std() / np.sqrt(len(dim_speedup)),
                                    'ci_95_high': dim_speedup.mean() + 1.96 * dim_speedup.std() / np.sqrt(len(dim_speedup)),
                                }
                
                # Report speedup statistics
                for key, stats in speedup_stats.items():
                    parts = key.split('_')
                    impl = parts[0]
                    dim = parts[-1].replace('D', '')
                    
                    report.append(f"### {impl} vs NumPy FFT ({dim}D transforms)\n")
                    report.append(f"* **Average Speedup**: {stats['mean']:.2f}x (95% CI: {stats['ci_95_low']:.2f}x - {stats['ci_95_high']:.2f}x)")
                    report.append(f"* **Median Speedup**: {stats['median']:.2f}x")
                    report.append(f"* **Range**: {stats['min']:.2f}x - {stats['max']:.2f}x")
                    report.append(f"* **Standard Deviation**: {stats['std']:.2f}\n")
                
                # Add analysis for power-of-2 vs non-power-of-2 sizes
                report.append("### Power-of-2 vs Non-Power-of-2 Size Performance\n")
                
                # Group by implementation, dimension, and whether size is power of 2
                pow2_analysis = df.groupby(['implementation', 'dimension', 'is_power_of_2'])['avg_execution_time'].mean().unstack()
                
                if True in pow2_analysis.columns and False in pow2_analysis.columns:
                    pow2_analysis['non_pow2_slowdown'] = pow2_analysis[False] / pow2_analysis[True]
                    
                    # Convert to a regular DataFrame for easier reporting
                    pow2_df = pow2_analysis.reset_index()
                    
                    for impl in pow2_df['implementation'].unique():
                        impl_data = pow2_df[pow2_df['implementation'] == impl]
                        report.append(f"#### {impl}\n")
                        
                        for dim in impl_data['dimension'].unique():
                            row = impl_data[impl_data['dimension'] == dim]
                            if not row.empty and 'non_pow2_slowdown' in row.columns:
                                slowdown = row['non_pow2_slowdown'].values[0]
                                report.append(f"* **{dim}D transforms**: Non-power-of-2 sizes are {slowdown:.2f}x slower than power-of-2 sizes\n")
    
    # Repeated transform summary with enhanced analysis
    if 'repeated_use' in results_dict:
        df = results_dict['repeated_use']
        report.append("## Repeated Transform Summary\n")
        
        if 'implementation' in df.columns and 'amortization_factor' in df.columns:
            # Analyze the amortization curve - how quickly does the overhead get amortized?
            report.append("### Amortization Analysis\n")
            
            for impl in df['implementation'].unique():
                impl_df = df[df['implementation'] == impl]
                
                # Find the repetition count at which amortization factor exceeds various thresholds
                thresholds = [2, 5, 10]
                amort_report = []
                
                for threshold in thresholds:
                    # For each transform type, find when the threshold is exceeded
                    for transform in impl_df['transform'].unique():
                        transform_df = impl_df[impl_df['transform'] == transform]
                        
                        # Group by repetitions and calculate mean amortization factor
                        grouped = transform_df.groupby('repetitions')['amortization_factor'].mean()
                        
                        # Find the first repetition count where the threshold is exceeded
                        above_threshold = grouped[grouped >= threshold]
                        if not above_threshold.empty:
                            min_reps = above_threshold.index.min()
                            amort_report.append(f"* **{transform}**: {min_reps} repetitions to reach {threshold}x amortization")
                
                if amort_report:
                    report.append(f"#### {impl}\n")
                    report.extend(amort_report)
                    report.append("")
            
            # Maximum amortization analysis
            report.append("### Maximum Amortization Factors\n")
            
            # Get highest repetition count
            max_reps = df['repetitions'].max()
            max_rep_df = df[df['repetitions'] == max_reps]
            
            # Group by implementation and transform, calculate mean amortization
            max_amort = max_rep_df.groupby(['implementation', 'transform'])['amortization_factor'].mean().reset_index()
            
            # Pivot for better presentation
            max_amort_pivot = max_amort.pivot(index='implementation', columns='transform', values='amortization_factor')
            
            # Report the table
            report.append(f"Maximum amortization factors at {max_reps} repetitions:\n")
            report.append(f"```\n{max_amort_pivot.to_string()}\n```\n")
    
    # Real-world scenario summary with detailed analysis
    if 'real_world' in results_dict:
        df = results_dict['real_world']
        report.append("## Real-World Scenario Summary\n")
        
        # Create a table of scenario performance with statistical significance
        if 'implementation' in df.columns and 'avg_time' in df.columns:
            # Calculate average and confidence interval for each implementation/scenario
            stats_df = df.groupby(['implementation', 'scenario']).agg({
                'avg_time': ['mean', 'std', 'count']
            }).reset_index()
            
            # Flatten multi-level columns
            stats_df.columns = ['implementation', 'scenario', 'mean_time', 'std_time', 'sample_count']
            
            # Calculate 95% confidence intervals
            stats_df['ci_95_low'] = stats_df['mean_time'] - 1.96 * stats_df['std_time'] / np.sqrt(stats_df['sample_count'])
            stats_df['ci_95_high'] = stats_df['mean_time'] + 1.96 * stats_df['std_time'] / np.sqrt(stats_df['sample_count'])
            
            # Pivot for better presentation
            pivot = stats_df.pivot(index='scenario', columns='implementation', values=['mean_time', 'ci_95_low', 'ci_95_high'])
            
            # Report the table
            report.append("### Scenario Performance (seconds) with 95% Confidence Intervals\n")
            
            # Format table with confidence intervals
            table_data = []
            
            # Add header
            header = ["Scenario"]
            for impl in pivot['mean_time'].columns:
                header.append(f"{impl} (95% CI)")
            table_data.append(header)
            
            # Add rows
            for scenario in pivot.index:
                row = [scenario]
                for impl in pivot['mean_time'].columns:
                    mean = pivot['mean_time'][impl][scenario]
                    ci_low = pivot['ci_95_low'][impl][scenario]
                    ci_high = pivot['ci_95_high'][impl][scenario]
                    row.append(f"{mean:.3f} ({ci_low:.3f}-{ci_high:.3f})")
                table_data.append(row)
            
            # Convert to a nicely formatted table
            report.append(f"```\n{tabulate(table_data, headers='firstrow', tablefmt='grid')}\n```\n")
            
            # Add speedup analysis
            if 'NumPy FFT' in pivot['mean_time'].columns:
                report.append("### Speedup Analysis vs NumPy\n")
                
                # Calculate speedup of each implementation vs NumPy
                speedup_data = []
                speedup_header = ["Scenario"]
                
                for impl in pivot['mean_time'].columns:
                    if impl != 'NumPy FFT':
                        speedup_header.append(f"{impl} Speedup (95% CI)")
                
                speedup_data.append(speedup_header)
                
                for scenario in pivot.index:
                    row = [scenario]
                    numpy_mean = pivot['mean_time']['NumPy FFT'][scenario]
                    
                    for impl in pivot['mean_time'].columns:
                        if impl != 'NumPy FFT':
                            impl_mean = pivot['mean_time'][impl][scenario]
                            
                            # Calculate speedup with error propagation for confidence interval
                            speedup = numpy_mean / impl_mean
                            
                            # Calculate confidence interval for the ratio using error propagation
                            # This is a simplified approach - a more accurate approach would use bootstrapping
                            numpy_ci_low = pivot['ci_95_low']['NumPy FFT'][scenario]
                            numpy_ci_high = pivot['ci_95_high']['NumPy FFT'][scenario]
                            impl_ci_low = pivot['ci_95_low'][impl][scenario]
                            impl_ci_high = pivot['ci_95_high'][impl][scenario]
                            
                            # Estimate the relative error of the speedup
                            rel_err_numpy = (numpy_ci_high - numpy_ci_low) / (2 * numpy_mean)
                            rel_err_impl = (impl_ci_high - impl_ci_low) / (2 * impl_mean)
                            rel_err_speedup = np.sqrt(rel_err_numpy**2 + rel_err_impl**2)
                            
                            # Calculate speedup confidence interval
                            speedup_ci_low = speedup * (1 - rel_err_speedup)
                            speedup_ci_high = speedup * (1 + rel_err_speedup)
                            
                            row.append(f"{speedup:.2f}x ({speedup_ci_low:.2f}-{speedup_ci_high:.2f})")
                    
                    speedup_data.append(row)
                
                # Add speedup table to report
                report.append(f"```\n{tabulate(speedup_data, headers='firstrow', tablefmt='grid')}\n```\n")
                
                # Add statistical significance testing
                report.append("### Statistical Significance of Performance Differences\n")
                
                signif_table = []
                signif_header = ["Scenario", "Comparison", "p-value", "Significant?"]
                signif_table.append(signif_header)
                
                # For each scenario, compare each implementation against NumPy
                for scenario in df['scenario'].unique():
                    scenario_df = df[df['scenario'] == scenario]
                    
                    numpy_times = scenario_df[scenario_df['implementation'] == 'NumPy FFT']['avg_time'].values
                    
                    for impl in scenario_df['implementation'].unique():
                        if impl != 'NumPy FFT':
                            impl_times = scenario_df[scenario_df['implementation'] == impl]['avg_time'].values
                            
                            # Simple t-test for statistical significance
                            from scipy import stats
                            t_stat, p_value = stats.ttest_ind(numpy_times, impl_times, equal_var=False)
                            
                            row = [
                                scenario,
                                f"NumPy vs {impl}",
                                f"{p_value:.4f}",
                                "Yes" if p_value < 0.05 else "No"
                            ]
                            signif_table.append(row)
                
                # Add significance table to report
                report.append(f"```\n{tabulate(signif_table, headers='firstrow', tablefmt='grid')}\n```\n")
    
    # Thread scaling summary with enhanced analysis
    if 'thread_scaling' in results_dict:
        df = results_dict['thread_scaling']
        report.append("## Thread Scaling Summary\n")
        
        # Report max speedup achieved for each implementation
        if 'implementation' in df.columns and 'speedup' in df.columns:
            report.append("### Parallel Efficiency Analysis\n")
            
            # Group by implementation and transform
            for impl, impl_df in df.groupby('implementation'):
                if impl == 'NumPy FFT':
                    continue  # Skip NumPy which doesn't support threading
                
                report.append(f"#### {impl}\n")
                
                # Find the maximum thread count tested
                max_threads = impl_df['thread_count'].max()
                
                # Get data for maximum thread count
                max_thread_df = impl_df[impl_df['thread_count'] == max_threads]
                
                if not max_thread_df.empty:
                    # Group by transform and calculate average speedup
                    max_speedup = max_thread_df.groupby('transform')['speedup'].mean().reset_index()
                    
                    # Calculate parallel efficiency
                    max_speedup['efficiency'] = max_speedup['speedup'] / max_threads
                    
                    # Report results
                    for _, row in max_speedup.iterrows():
                        transform = row['transform']
                        speedup = row['speedup']
                        efficiency = row['efficiency']
                        
                        report.append(f"* **{transform}**: {speedup:.2f}x speedup with {max_threads} threads ({efficiency:.2f} efficiency)")
                    
                    # Add scaling analysis
                    report.append("\nScaling characteristics:\n")
                    
                    # Calculate linear regression of speedup vs thread count
                    for transform in impl_df['transform'].unique():
                        transform_df = impl_df[impl_df['transform'] == transform]
                        
                        # Need at least 3 data points for meaningful regression
                        if len(transform_df['thread_count'].unique()) >= 3:
                            try:
                                from scipy.stats import linregress
                                
                                # Calculate average speedup for each thread count
                                avg_speedup = transform_df.groupby('thread_count')['speedup'].mean().reset_index()
                                
                                # Linear regression
                                slope, intercept, r_value, p_value, std_err = linregress(
                                    avg_speedup['thread_count'], 
                                    avg_speedup['speedup']
                                )
                                
                                # Calculate Amdahl's Law parameters (very simplified model)
                                # If the relationship follows: speedup = 1 / ((1-p) + p/N)
                                # where N is thread count and p is the parallel portion
                                
                                # Using the speedup at maximum thread count to estimate p
                                S_max = avg_speedup[avg_speedup['thread_count'] == max_threads]['speedup'].values[0]
                                p_est = (max_threads * (S_max - 1)) / (S_max * (max_threads - 1)) if max_threads > 1 and S_max > 1 else 0
                                
                                report.append(f"* **{transform}**: Scaling slope: {slope:.3f} (R²: {r_value**2:.3f}), Est. parallel portion: {p_est*100:.1f}%")
                            except Exception as e:
                                report.append(f"* **{transform}**: Scaling analysis failed: {str(e)}")
                
                report.append("")
    
    # Memory efficiency summary
    if 'memory_efficiency' in results_dict:
        df = results_dict['memory_efficiency']
        report.append("## Memory Efficiency Summary\n")
        
        if 'implementation' in df.columns:
            # Calculate average memory usage and throughput
            summary = df.groupby('implementation').agg({
                'avg_peak_memory_delta_mb': 'mean',
                'throughput_mbs': 'mean',
                'memory_efficiency': 'mean'
            }).reset_index()
            
            # Sort by memory efficiency
            summary = summary.sort_values('memory_efficiency', ascending=False)
            
            # Report the table
            report.append("### Memory Efficiency Ranking\n")
            report.append(f"```\n{summary.to_string(index=False)}\n```\n")
            
            # Add analysis
            most_efficient = summary.iloc[0]['implementation']
            
            report.append(f"**{most_efficient}** is the most memory-efficient implementation, achieving the highest throughput per MB of memory used.\n")
            
            # Compare each implementation to NumPy
            if 'NumPy FFT' in summary['implementation'].values:
                numpy_row = summary[summary['implementation'] == 'NumPy FFT']
                if not numpy_row.empty:
                    numpy_mem = numpy_row['avg_peak_memory_delta_mb'].values[0]
                    numpy_throughput = numpy_row['throughput_mbs'].values[0]
                    
                    report.append("### Comparison to NumPy\n")
                    
                    for _, row in summary.iterrows():
                        if row['implementation'] != 'NumPy FFT':
                            mem_ratio = row['avg_peak_memory_delta_mb'] / numpy_mem
                            throughput_ratio = row['throughput_mbs'] / numpy_throughput
                            
                            report.append(f"* **{row['implementation']}**: Uses {mem_ratio:.2f}x the memory of NumPy, with {throughput_ratio:.2f}x the throughput")
                    
                    report.append("")
    
    # Non-power-of-two summary
    if 'non_power_of_two' in results_dict:
        df = results_dict['non_power_of_two']
        report.append("## Non-Power-of-2 Size Performance Summary\n")
        
        if 'implementation' in df.columns:
            # Calculate average performance relative to NumPy
            if 'NumPy FFT' in df['implementation'].unique():
                # Create a pivot table
                pivot = df.pivot_table(
                    index=['dimension', 'transform', 'size', 'dataset_idx'],
                    columns='implementation',
                    values='avg_execution_time'
                )
                
                # Calculate speedup ratios
                for impl in pivot.columns:
                    if impl != 'NumPy FFT':
                        pivot[f"{impl}_vs_NumPy"] = pivot['NumPy FFT'] / pivot[impl]
                
                # Calculate statistics by dimension and transform
                speedup_stats = {}
                
                for dim in df['dimension'].unique():
                    for transform in df[df['dimension'] == dim]['transform'].unique():
                        dim_transform_indices = (pivot.index.get_level_values('dimension') == dim) & (pivot.index.get_level_values('transform') == transform)
                        
                        for impl in [col for col in pivot.columns if '_vs_NumPy' in col]:
                            impl_name = impl.replace('_vs_NumPy', '')
                            key = f"{impl_name}_{dim}D_{transform}"
                            
                            speedup_values = pivot.loc[dim_transform_indices, impl]
                            
                            if not speedup_values.empty:
                                speedup_stats[key] = {
                                    'mean': speedup_values.mean(),
                                    'median': speedup_values.median(),
                                    'min': speedup_values.min(),
                                    'max': speedup_values.max(),
                                    'std': speedup_values.std()
                                }
                
                # Report speedup statistics
                report.append("### Performance Relative to NumPy on Non-Power-of-2 Sizes\n")
                
                # Create a table
                speedup_table = []
                header = ["Implementation", "Dimension", "Transform", "Mean Speedup", "Median Speedup", "Min-Max Speedup"]
                speedup_table.append(header)
                
                for key, stats in speedup_stats.items():
                    parts = key.split('_')
                    impl = parts[0]
                    dim = parts[1].replace('D', '')
                    transform = ' '.join(parts[2:])
                    
                    row = [
                        impl,
                        dim,
                        transform,
                        f"{stats['mean']:.2f}x",
                        f"{stats['median']:.2f}x",
                        f"{stats['min']:.2f}x - {stats['max']:.2f}x"
                    ]
                    speedup_table.append(row)
                
                # Sort by dimension, transform, then mean speedup (descending)
                speedup_table[1:] = sorted(speedup_table[1:], key=lambda x: (x[1], x[2], -float(x[3].replace('x', ''))))
                
                # Add table to report
                report.append(f"```\n{tabulate(speedup_table, headers='firstrow', tablefmt='grid')}\n```\n")
                
                # Add analysis of prime number sizes
                report.append("### Performance on Prime Number Sizes\n")
                
                # Find rows with prime number sizes (assuming the prime_factors column has this info)
                if 'prime_factors' in df.columns:
                    prime_sizes = df[df['prime_factors'].apply(lambda x: ',' not in x and '*' not in x)]
                    
                    if not prime_sizes.empty:
                        # Calculate relative performance on prime sizes
                        prime_pivot = prime_sizes.pivot_table(
                            index=['dimension', 'size', 'dataset_idx'],
                            columns='implementation',
                            values='avg_execution_time'
                        )
                        
                        if 'NumPy FFT' in prime_pivot.columns:
                            for impl in prime_pivot.columns:
                                if impl != 'NumPy FFT':
                                    prime_pivot[f"{impl}_vs_NumPy"] = prime_pivot['NumPy FFT'] / prime_pivot[impl]
                            
                            # Calculate average speedup on prime sizes
                            prime_speedup = {}
                            
                            for impl in [col for col in prime_pivot.columns if '_vs_NumPy' in col]:
                                impl_name = impl.replace('_vs_NumPy', '')
                                prime_speedup[impl_name] = prime_pivot[impl].mean()
                            
                            # Report findings
                            for impl, speedup in prime_speedup.items():
                                report.append(f"* **{impl}**: {speedup:.2f}x faster than NumPy on prime number sizes\n")
    
    # Overall summary
    report.append("## Overall Analysis and Recommendations\n")
    
    # Identify the best implementation for different scenarios
    best_impls = {}
    
    # For general usage (from direct comparison)
    if 'direct_comparison' in results_dict:
        direct_df = results_dict['direct_comparison']
        
        # Calculate overall average speedup vs NumPy
        pivot = direct_df.pivot_table(
            index=['dimension', 'transform', 'size', 'dataset_idx'],
            columns='implementation',
            values='avg_execution_time'
        )
        
        if 'NumPy FFT' in pivot.columns:
            overall_speedup = {}
            
            for impl in pivot.columns:
                if impl != 'NumPy FFT':
                    speedup = pivot['NumPy FFT'] / pivot[impl]
                    overall_speedup[impl] = speedup.mean()
            
            # Find the best implementation
            if overall_speedup:
                best_general = max(overall_speedup.items(), key=lambda x: x[1])
                best_impls['general'] = best_general
    
    # For repeated transforms (from repeated use)
    if 'repeated_use' in results_dict:
        repeated_df = results_dict['repeated_use']
        
        # Get the maximum repetition count
        max_reps = repeated_df['repetitions'].max()
        max_rep_df = repeated_df[repeated_df['repetitions'] == max_reps]
        
        if not max_rep_df.empty:
            # Calculate average amortization factor
            avg_amort = max_rep_df.groupby('implementation')['amortization_factor'].mean()
            
            # Find the best implementation
            if not avg_amort.empty:
                best_repeated = (avg_amort.idxmax(), avg_amort.max())
                best_impls['repeated'] = best_repeated
    
    # For real-world scenarios
    if 'real_world' in results_dict:
        real_df = results_dict['real_world']
        
        if 'implementation' in real_df.columns and 'avg_time' in real_df.columns:
            # Calculate average time across all scenarios
            avg_time = real_df.groupby('implementation')['avg_time'].mean()
            
            if 'NumPy FFT' in avg_time.index:
                # Calculate speedup vs NumPy
                speedup = avg_time['NumPy FFT'] / avg_time
                
                # Find the best implementation
                best_real_world = (speedup.idxmax(), speedup.max())
                best_impls['real_world'] = best_real_world
    
    # For thread scaling
    if 'thread_scaling' in results_dict:
        thread_df = results_dict['thread_scaling']
        
        if 'implementation' in thread_df.columns and 'speedup' in thread_df.columns:
            # Find the maximum thread count
            max_threads = thread_df['thread_count'].max()
            max_thread_df = thread_df[thread_df['thread_count'] == max_threads]
            
            # Calculate average speedup at max threads
            avg_speedup = max_thread_df.groupby('implementation')['speedup'].mean()
            
            # Find the best implementation
            if not avg_speedup.empty:
                best_threading = (avg_speedup.idxmax(), avg_speedup.max())
                best_impls['threading'] = best_threading
    
    # For non-power-of-2 sizes
    if 'non_power_of_two' in results_dict:
        nonpow2_df = results_dict['non_power_of_two']
        
        if 'implementation' in nonpow2_df.columns and 'avg_execution_time' in nonpow2_df.columns:
            # Calculate relative performance to NumPy
            pivot = nonpow2_df.pivot_table(
                index=['dimension', 'size', 'dataset_idx'],
                columns='implementation',
                values='avg_execution_time'
            )
            
            if 'NumPy FFT' in pivot.columns:
                overall_speedup = {}
                
                for impl in pivot.columns:
                    if impl != 'NumPy FFT':
                        speedup = pivot['NumPy FFT'] / pivot[impl]
                        overall_speedup[impl] = speedup.mean()
                
                # Find the best implementation
                if overall_speedup:
                    best_nonpow2 = max(overall_speedup.items(), key=lambda x: x[1])
                    best_impls['non_power_of_two'] = best_nonpow2
    
    # Report the best implementations
    if best_impls:
        report.append("### Best Performing Implementations\n")
        
        if 'general' in best_impls:
            impl, speedup = best_impls['general']
            report.append(f"* **General Usage**: {impl} ({speedup:.2f}x faster than NumPy on average)\n")
        
        if 'repeated' in best_impls:
            impl, amort = best_impls['repeated']
            report.append(f"* **Repeated Transforms**: {impl} ({amort:.2f}x amortization factor)\n")
        
        if 'real_world' in best_impls:
            impl, speedup = best_impls['real_world']
            report.append(f"* **Real-World Scenarios**: {impl} ({speedup:.2f}x faster than NumPy)\n")
        
        if 'threading' in best_impls:
            impl, speedup = best_impls['threading']
            report.append(f"* **Multi-Threading**: {impl} ({speedup:.2f}x speedup with maximum threads)\n")
        
        if 'non_power_of_two' in best_impls:
            impl, speedup = best_impls['non_power_of_two']
            report.append(f"* **Non-Power-of-2 Sizes**: {impl} ({speedup:.2f}x faster than NumPy)\n")
    
    # Final recommendations
    report.append("### Recommendations\n")
    
    # Look at the results across all benchmarks to make overall recommendations
    all_results = []
    
    # Collect all speedup values
    if 'general' in best_impls:
        all_results.append(('general', best_impls['general'][0], best_impls['general'][1]))
    
    if 'real_world' in best_impls:
        all_results.append(('real_world', best_impls['real_world'][0], best_impls['real_world'][1]))
    
    if 'non_power_of_two' in best_impls:
        all_results.append(('non_power_of_two', best_impls['non_power_of_two'][0], best_impls['non_power_of_two'][1]))
    
    # Count the "wins" for each implementation
    wins = {}
    for _, impl, _ in all_results:
        if impl not in wins:
            wins[impl] = 0
        wins[impl] += 1
    
    # Find the implementation with the most wins
    if wins:
        best_overall = max(wins.items(), key=lambda x: x[1])
        
        report.append(f"Based on the benchmarks, **{best_overall[0]}** appears to be the best overall FFT implementation for Python, winning in {best_overall[1]} out of {len(all_results)} categories.\n")
        
        # Add specific recommendations for different use cases
        report.append("#### Specific Use Case Recommendations:\n")
        
        # For casual/general use
        report.append("**For casual/general use:**\n")
        if 'general' in best_impls:
            impl, speedup = best_impls['general']
            if speedup > 1.5:
                report.append(f"- {impl} is recommended (average {speedup:.2f}x speedup)\n")
            else:
                report.append("- NumPy's built-in FFT is sufficient for most casual usage\n")
        
        # For performance-critical applications
        report.append("**For performance-critical applications:**\n")
        perfs = []
        
        if 'general' in best_impls:
            perfs.append(best_impls['general'])
        if 'real_world' in best_impls:
            perfs.append(best_impls['real_world'])
        
        if perfs:
            # Find the implementation with the highest speedup
            best_perf = max(perfs, key=lambda x: x[1])
            report.append(f"- {best_perf[0]} provides the highest performance ({best_perf[1]:.2f}x speedup)\n")
        
        # For repeated FFT operations
        report.append("**For applications with repeated FFT operations:**\n")
        if 'repeated' in best_impls:
            impl, amort = best_impls['repeated']
            report.append(f"- {impl} provides excellent plan reuse ({amort:.2f}x speedup for repeated transforms)\n")
        
        # For non-power-of-2 sizes
        report.append("**For applications with non-power-of-2 sizes:**\n")
        if 'non_power_of_two' in best_impls:
            impl, speedup = best_impls['non_power_of_two']
            report.append(f"- {impl} handles non-power-of-2 sizes most effectively ({speedup:.2f}x speedup)\n")
        
        # For multi-threaded applications
        report.append("**For multi-threaded applications:**\n")
        if 'threading' in best_impls:
            impl, speedup = best_impls['threading']
            report.append(f"- {impl} provides the best multi-threading scaling ({speedup:.2f}x speedup)\n")
    
    # Save report if requested
    if save_file:
        with open(save_file, 'w') as f:
            f.write('\n'.join(report))
        logger.info(f"Enhanced summary report saved to {save_file}")
    
    return '\n'.join(report)

def run_enhanced_benchmarks(n_runs=5, n_datasets=3):
    """Run all enhanced benchmarks and generate a comprehensive report."""
    # Print system information
    sys_info = {}
    sys_info["os"] = os.name
    sys_info["cpu_count"] = os.cpu_count()
    sys_info["memory_gb"] = psutil.virtual_memory().total / (1024**3)
    sys_info["python_version"] = sys.version
    sys_info["numpy_version"] = np.__version__
    sys_info["scipy_version"] = scipy.__version__
    sys_info["pyfftw_version"] = pyfftw.__version__ if HAVE_PYFFTW else "Not installed"
    sys_info["betterfftw_version"] = betterfftw.__version__
    
    print("\n" + "=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    for key, value in sys_info.items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")
    
    results = {}
    
    # Run benchmarks
    print("\nRunning Enhanced Direct Comparison Benchmark...")
    results['direct_comparison'] = benchmark_direct_comparison(n_runs=n_runs, n_datasets=n_datasets)
    
    print("\nRunning Enhanced Repeated Use Benchmark...")
    results['repeated_use'] = benchmark_repeated_use(n_runs=n_runs, n_datasets=n_datasets)
    
    print("\nRunning Enhanced Real-World Benchmark...")
    results['real_world'] = benchmark_real_world(n_runs=n_runs, n_datasets=n_datasets)
    
    print("\nRunning Thread Scaling Benchmark...")
    results['thread_scaling'] = benchmark_thread_scaling(n_runs=n_runs, n_datasets=n_datasets)
    
    print("\nRunning Memory Efficiency Benchmark...")
    results['memory_efficiency'] = benchmark_memory_efficiency(n_runs=n_runs)
    
    print("\nRunning Non-Power-of-Two Benchmark...")
    results['non_power_of_two'] = benchmark_non_power_of_two(n_runs=n_runs, n_datasets=n_datasets)
    
    # Create and save enhanced summary report
    summary_file = os.path.join(RESULTS_DIR, f"enhanced_summary_report_{TIMESTAMP}.md")
    summary = create_enhanced_summary_report(results, save_file=summary_file)
    
    print(f"\nAll benchmarks completed. Results saved to {RESULTS_DIR}")
    print(f"Enhanced summary report: {summary_file}")
    
    return results, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced BetterFFTW Benchmark Suite")
    parser.add_argument("--runs", type=int, default=5, help="Number of measurement runs per test")
    parser.add_argument("--datasets", type=int, default=3, help="Number of different datasets per test")
    parser.add_argument("--output", type=str, default=RESULTS_DIR, help="Output directory")
    parser.add_argument("--benchmark", type=str, default="all",
                       choices=["all", "direct", "repeated", "realworld", "threads", "memory", "nonpow2"],
                       help="Which benchmark to run")
    
    args = parser.parse_args()
    
    # Update output directory if specified
    if args.output != RESULTS_DIR:
        RESULTS_DIR = args.output
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run selected benchmark
    if args.benchmark == "all":
        run_enhanced_benchmarks(n_runs=args.runs, n_datasets=args.datasets)
    elif args.benchmark == "direct":
        benchmark_direct_comparison(n_runs=args.runs, n_datasets=args.datasets)
    elif args.benchmark == "repeated":
        benchmark_repeated_use(n_runs=args.runs, n_datasets=args.datasets)
    elif args.benchmark == "realworld":
        benchmark_real_world(n_runs=args.runs, n_datasets=args.datasets)
    elif args.benchmark == "threads":
        benchmark_thread_scaling(n_runs=args.runs, n_datasets=args.datasets)
    elif args.benchmark == "memory":
        benchmark_memory_efficiency(n_runs=args.runs)
    elif args.benchmark == "nonpow2":
        benchmark_non_power_of_two(n_runs=args.runs, n_datasets=args.datasets)