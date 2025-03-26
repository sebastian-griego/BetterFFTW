"""
BetterFFTW Benchmark Suite

This script compares BetterFFTW's default settings against NumPy, SciPy, and PyFFTW
across various realistic usage scenarios. The goal is to demonstrate BetterFFTW's
automatic optimization capabilities without requiring user configuration.

Benchmarks:
1. Direct comparison across dimensions and sizes
2. Repeated transform performance with plan caching
3. Real-world application scenarios
4. Drop-in replacement performance
5. Memory usage and thread scaling

Results are presented as both raw data and visualizations.
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
        "cleanup": lambda: betterfftw.clear_cache(),
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
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return result, end_time - start_time

# Format benchmark results as a table
def format_results(results, metric='time', baseline='numpy'):
    """Format benchmark results as a table with speedup ratios."""
    if not results:
        return "No results to display"
    
    # Create a DataFrame from results
    df = pd.DataFrame(results)
    
    # Pivot the data for better presentation
    if 'size' in df.columns and 'implementation' in df.columns:
        pivot = df.pivot_table(
            index='size',
            columns='implementation',
            values=metric,
            aggfunc='mean'
        )
        
        # Calculate speedup compared to baseline
        if baseline in pivot.columns:
            for col in pivot.columns:
                if col != baseline:
                    pivot[f"{col}/numpy speedup"] = pivot[baseline] / pivot[col]
        
        return pivot
    else:
        return df

# Plot benchmark results
def plot_results(results, title, x_col='size', y_col='time', hue_col='implementation', 
                filename=None, log_scale=True):
    """Create a line plot of benchmark results."""
    plt.figure(figsize=(10, 6))
    
    # Use seaborn for better aesthetics
    sns.set_style("whitegrid")
    
    # Create line plot
    ax = sns.lineplot(
        data=results, 
        x=x_col, 
        y=y_col, 
        hue=hue_col,
        marker='o'
    )
    
    # Set log scale if requested
    if log_scale:
        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
    
    # Set labels and title
    plt.xlabel(x_col.capitalize())
    plt.ylabel(y_col.capitalize() + ' (seconds)')
    plt.title(title)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(title=hue_col.capitalize())
    
    # Save figure if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {filename}")
    
    plt.close()

# Get system information
def get_system_info():
    """Collect system information for benchmarking context."""
    info = {
        "os": os.name,
        "cpu_count": os.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pyfftw_version": pyfftw.__version__ if HAVE_PYFFTW else "Not installed",
        "betterfftw_version": betterfftw.__version__,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Check if NumPy is using MKL
    np_config = str(np.__config__).lower()
    info["numpy_using_mkl"] = "mkl" in np_config or "intel" in np_config
    
    return info

# Print system information
def print_system_info():
    """Print system information in a readable format."""
    info = get_system_info()
    print("\n" + "=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")
    return info

def benchmark_direct_comparison(n_runs=5, save_results=True):
    """
    Compare BetterFFTW defaults against other libraries across
    various array sizes and dimensions.
    """
    logger.info("Starting direct comparison benchmark...")
    
    results = []
    
    # Test configurations
    dimensions = [1, 2, 3]  # 1D, 2D, 3D
    
    # Power-of-2 sizes
    power2_sizes = [2**n for n in range(8, 14)]  # 256 to 8192
    
    # Non-power-of-2 sizes (includes prime numbers and mixed factors)
    nonpower2_sizes = [
        300,    # 2^2 * 3 * 5^2
        720,    # 2^4 * 3^2 * 5
        1000,   # 2^3 * 5^3
        1331,   # 11^3 (prime cubed)
        1920,   # 2^6 * 3 * 5 (HD width)
        2187,   # 3^7 (power of 3)
        10000   # 2^4 * 5^4 (large round number)
    ]
    
    # Combined sizes list
    all_sizes = sorted(power2_sizes + nonpower2_sizes)
    
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
                sizes = [s for s in all_sizes if s <= 2048]  # Limit 2D to 2048
            else:  # dim == 3
                sizes = [s for s in all_sizes if s <= 256]   # Limit 3D to 256
            
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
                
                # Create test array
                try:
                    if is_real:
                        # Real input for rfft
                        array = create_test_array(shape, dtype=np.float64, is_complex=False)
                    else:
                        # Complex input for regular fft
                        array = create_test_array(shape, dtype=np.complex128, is_complex=True)
                        
                    logger.info(f"  Size {size} ({'power of 2' if is_power_of_2 else 'non-power of 2'})")
                    
                    # Test each implementation
                    for impl_key, impl in FFT_IMPLEMENTATIONS.items():
                        if not impl["available"]:
                            continue
                        
                        # Get the transform function
                        try:
                            impl_module = impl["module"]
                            impl_func = getattr(impl_module, full_func_name)
                            
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
                            planning_overhead = first_run_time - avg_execution_time
                            
                            # Record memory before and after to estimate memory usage
                            mem_before = get_memory_usage()
                            array_copy = array.copy()
                            _ = impl_func(array_copy)
                            mem_after = get_memory_usage()
                            mem_delta = mem_after - mem_before
                            
                            # Record result
                            result = {
                                "implementation": impl["name"],
                                "dimension": dim,
                                "transform": transform_name,
                                "size": size,
                                "shape": shape,
                                "is_power_of_2": is_power_of_2,
                                "first_run_time": first_run_time,
                                "avg_execution_time": avg_execution_time,
                                "planning_overhead": planning_overhead,
                                "memory_delta_mb": mem_delta
                            }
                            
                            logger.info(f"    {impl['name']}: {avg_execution_time:.6f}s (first: {first_run_time:.6f}s)")
                            results.append(result)
                            
                            # Cleanup
                            impl["cleanup"]()
                            force_gc()
                            
                        except Exception as e:
                            logger.error(f"Error with {impl['name']} on {full_func_name} size {size}: {str(e)}")
                            traceback.print_exc()
                            
                except Exception as e:
                    logger.error(f"Error creating array with shape {shape}: {str(e)}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if save_results:
        csv_file = os.path.join(RESULTS_DIR, f"direct_comparison_{TIMESTAMP}.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file}")
        
        # Create summary plots
        if not df.empty:
            # 1D transforms
            plot_df = df[(df['dimension'] == 1)]
            if not plot_df.empty:
                plot_results(
                    plot_df,
                    'Performance of 1D Transforms',
                    x_col='size',
                    y_col='avg_execution_time',
                    hue_col='implementation',
                    filename=os.path.join(RESULTS_DIR, f"direct_comparison_1d_{TIMESTAMP}.png")
                )
            
            # 2D transforms
            plot_df = df[(df['dimension'] == 2)]
            if not plot_df.empty:
                plot_results(
                    plot_df,
                    'Performance of 2D Transforms',
                    x_col='size',
                    y_col='avg_execution_time',
                    hue_col='implementation',
                    filename=os.path.join(RESULTS_DIR, f"direct_comparison_2d_{TIMESTAMP}.png")
                )
            
            # Planning overhead comparison
            plot_df = df.groupby(['implementation', 'dimension'])['planning_overhead'].mean().reset_index()
            if not plot_df.empty:
                plt.figure(figsize=(10, 6))
                sns.barplot(data=plot_df, x='implementation', y='planning_overhead', hue='dimension')
                plt.title('Average Planning Overhead by Implementation')
                plt.ylabel('Planning Overhead (seconds)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_DIR, f"planning_overhead_{TIMESTAMP}.png"), dpi=300)
                plt.close()

    return df

def benchmark_repeated_use(n_runs=3, save_results=True):
    """
    Measure performance when the same transform is used repeatedly,
    showing the benefits of intelligent plan caching.
    """
    logger.info("Starting repeated use benchmark...")
    
    results = []
    
    # Test configurations
    transforms = [
        {"name": "1D FFT", "func": "fft", "shape": (2048,), "is_complex": True},
        {"name": "2D FFT", "func": "fft2", "shape": (512, 512), "is_complex": True},
        {"name": "1D RFFT", "func": "rfft", "shape": (2048,), "is_complex": False},
        {"name": "2D RFFT", "func": "rfft2", "shape": (512, 512), "is_complex": False}
    ]
    
    # Repetition counts to test
    repetition_counts = [1, 10, 100, 1000]
    
    for transform in transforms:
        transform_name = transform["name"]
        func_name = transform["func"]
        shape = transform["shape"]
        is_complex = transform["is_complex"]
        
        logger.info(f"Testing {transform_name} with shape {shape}...")
        
        # Create test array
        array = create_test_array(shape, dtype=np.complex128 if is_complex else np.float64, is_complex=is_complex)
        
        for impl_key, impl in FFT_IMPLEMENTATIONS.items():
            if not impl["available"]:
                continue
                
            logger.info(f"  Implementation: {impl['name']}")
            
            # Get the transform function
            try:
                impl_module = impl["module"]
                impl_func = getattr(impl_module, func_name)
                
                for repetitions in repetition_counts:
                    logger.info(f"    Repetitions: {repetitions}")
                    
                    # Setup implementation
                    impl["setup"]()
                    
                    # Measure time for fresh (uncached) transform
                    array_copies = [array.copy() for _ in range(repetitions)]
                    
                    # First transform (with planning)
                    start_time = time.time()
                    _ = impl_func(array_copies[0])
                    first_transform_time = time.time() - start_time
                    
                    # Measure total time for all repetitions
                    start_time = time.time()
                    for i in range(repetitions):
                        _ = impl_func(array_copies[i])
                    total_time = time.time() - start_time
                    
                    # Amortized time per transform
                    amortized_time = total_time / repetitions
                    
                    # Record result
                    result = {
                        "implementation": impl["name"],
                        "transform": transform_name,
                        "shape": str(shape),
                        "repetitions": repetitions,
                        "first_transform_time": first_transform_time,
                        "total_time": total_time,
                        "amortized_time": amortized_time,
                        "amortization_factor": first_transform_time / amortized_time if amortized_time > 0 else 0,
                    }
                    
                    results.append(result)
                    logger.info(f"      First: {first_transform_time:.6f}s, Amortized: {amortized_time:.6f}s")
                    
                    # Cleanup
                    impl["cleanup"]()
                    force_gc()
                    
            except Exception as e:
                logger.error(f"Error with {impl['name']} on {func_name}: {str(e)}")
                traceback.print_exc()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if save_results and not df.empty:
        csv_file = os.path.join(RESULTS_DIR, f"repeated_use_{TIMESTAMP}.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file}")
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        
        # One plot per transform type
        for transform_name in df['transform'].unique():
            transform_df = df[df['transform'] == transform_name]
            
            plt.figure(figsize=(10, 6))
            for impl_name in transform_df['implementation'].unique():
                impl_df = transform_df[transform_df['implementation'] == impl_name]
                plt.plot(impl_df['repetitions'], impl_df['amortized_time'], 
                         marker='o', label=impl_name)
            
            plt.xscale('log', base=10)
            plt.yscale('log', base=10)
            plt.title(f'Amortized Time for {transform_name}')
            plt.xlabel('Number of Repetitions')
            plt.ylabel('Amortized Time per Transform (seconds)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.savefig(os.path.join(RESULTS_DIR, f"repeated_use_{transform_name.replace(' ', '_')}_{TIMESTAMP}.png"), 
                      dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create amortization factor comparison
        plt.figure(figsize=(12, 8))
        pivot = df.pivot_table(
            index='repetitions',
            columns=['implementation', 'transform'],
            values='amortization_factor'
        )
        
        # Plot heatmap of amortization factors
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
        plt.title('Amortization Factor (First Transform Time / Amortized Time)')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"amortization_factor_{TIMESTAMP}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    return df

def benchmark_real_world(n_runs=3, save_results=True):
    """
    Test with realistic application scenarios.
    """
    logger.info("Starting real-world scenario benchmark...")
    
    results = []
    
    # Define real-world scenarios
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
            "name": "Signal Analysis",
            "description": "Analyze a signal with FFT, apply window, calculate power spectrum",
            "shape": (8192,),  # 1D signal
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
        }
    ]
    
    for scenario in scenarios:
        scenario_name = scenario["name"]
        shape = scenario["shape"]
        steps = scenario["steps"]
        
        logger.info(f"Testing scenario: {scenario_name} with shape {shape}")
        
        for impl_key, impl in FFT_IMPLEMENTATIONS.items():
            if not impl["available"]:
                continue
                
            impl_module = impl["module"]
            logger.info(f"  Implementation: {impl['name']}")
            
            # Run the scenario multiple times
            scenario_times = []
            
            for run in range(n_runs):
                # Setup implementation
                impl["setup"]()
                
                # Create initial data
                data = create_test_array(shape, dtype=np.float64, is_complex=False)
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
                        
                    else:
                        # Unknown step type
                        logger.warning(f"Unknown step type: {step}")
                        step_time = 0
                        
                    step_times.append(step_time)
                
                # Calculate total scenario time
                total_time = sum(step_times)
                scenario_times.append(total_time)
                
                # Cleanup
                impl["cleanup"]()
                force_gc()
            
            # Calculate average time
            avg_scenario_time = sum(scenario_times) / len(scenario_times)
            
            # Record result
            result = {
                "implementation": impl["name"],
                "scenario": scenario_name,
                "shape": str(shape),
                "avg_time": avg_scenario_time,
                "min_time": min(scenario_times),
                "max_time": max(scenario_times)
            }
            
            results.append(result)
            logger.info(f"    Average time: {avg_scenario_time:.6f}s")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if save_results and not df.empty:
        csv_file = os.path.join(RESULTS_DIR, f"real_world_{TIMESTAMP}.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file}")
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        
        # Create a grouped bar chart
        ax = sns.barplot(x='scenario', y='avg_time', hue='implementation', data=df)
        
        plt.title('Real-World Scenario Performance')
        plt.xlabel('Scenario')
        plt.ylabel('Average Time (seconds)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(RESULTS_DIR, f"real_world_{TIMESTAMP}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create speedup comparison
        # Pivot to calculate speedup vs NumPy
        pivot = df.pivot_table(index='scenario', columns='implementation', values='avg_time')
        
        if 'NumPy FFT' in pivot.columns:
            for col in pivot.columns:
                if col != 'NumPy FFT':
                    pivot[f"{col} Speedup"] = pivot['NumPy FFT'] / pivot[col]
            
            # Plot speedups
            speedup_cols = [col for col in pivot.columns if 'Speedup' in col]
            if speedup_cols:
                plt.figure(figsize=(10, 6))
                pivot[speedup_cols].plot(kind='bar')
                plt.title('Speedup vs NumPy in Real-World Scenarios')
                plt.ylabel('Speedup Factor')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(RESULTS_DIR, f"real_world_speedup_{TIMESTAMP}.png"), dpi=300, bbox_inches='tight')
                plt.close()
    
    return df

def benchmark_drop_in_replacement(n_runs=5, save_results=True):
    """
    Measure the impact of using betterfftw.use_as_default()
    on existing NumPy/SciPy code.
    """
    logger.info("Starting drop-in replacement benchmark...")
    
    results = []
    
    # Define test cases
    test_cases = [
        {
            "name": "1D FFT",
            "code": "np.fft.fft(data)",
            "shape": (4096,),
            "is_complex": False
        },
        {
            "name": "2D FFT",
            "code": "np.fft.fft2(data)",
            "shape": (512, 512),
            "is_complex": False
        },
        {
            "name": "Real FFT",
            "code": "np.fft.rfft(data)",
            "shape": (4096,),
            "is_complex": False
        },
        {
            "name": "Inverse FFT",
            "code": "np.fft.ifft(data)",
            "shape": (4096,),
            "is_complex": True
        },
        {
            "name": "NumPy Batch",
            "code": """
for _ in range(10):
    result = np.fft.fft(data)
    result = np.fft.ifft(result)
            """,
            "shape": (2048,),
            "is_complex": False
        },
        {
            "name": "SciPy FFT",
            "code": "scipy.fft.fft(data)",
            "shape": (4096,),
            "is_complex": False
        },
        {
            "name": "SciPy 2D FFT",
            "code": "scipy.fft.fft2(data)",
            "shape": (512, 512),
            "is_complex": False
        }
    ]
    
    for test_case in test_cases:
        name = test_case["name"]
        code = test_case["code"]
        shape = test_case["shape"]
        is_complex = test_case["is_complex"]
        
        logger.info(f"Testing: {name}")
        
        # Create test data
        data = create_test_array(shape, dtype=np.complex128 if is_complex else np.float64, is_complex=is_complex)
        
        # Test with native NumPy/SciPy
        namespace = {
            'np': np,
            'scipy': scipy,
            'data': data.copy()
        }
        
        # Make sure betterfftw is not active
        if hasattr(betterfftw, 'restore_default'):
            betterfftw.restore_default()
        
        force_gc()
        
        # Warm-up run
        exec(code, namespace)
        
        # Time native execution
        native_times = []
        for _ in range(n_runs):
            namespace['data'] = data.copy()
            force_gc()
            
            start_time = time.time()
            exec(code, namespace)
            end_time = time.time()
            
            native_times.append(end_time - start_time)
        
        native_avg_time = sum(native_times) / len(native_times)
        logger.info(f"  Native: {native_avg_time:.6f}s")
        
        # Now test with BetterFFTW as drop-in replacement
        namespace = {
            'np': np,
            'scipy': scipy,
            'data': data.copy()
        }
        
        # Activate BetterFFTW
        betterfftw.use_as_default()
        
        force_gc()
        
        # Warm-up run
        exec(code, namespace)
        
        # Time BetterFFTW execution
        betterfftw_times = []
        for _ in range(n_runs):
            namespace['data'] = data.copy()
            force_gc()
            
            start_time = time.time()
            exec(code, namespace)
            end_time = time.time()
            
            betterfftw_times.append(end_time - start_time)
        
        betterfftw_avg_time = sum(betterfftw_times) / len(betterfftw_times)
        logger.info(f"  BetterFFTW: {betterfftw_avg_time:.6f}s")
        
        # Calculate speedup
        speedup = native_avg_time / betterfftw_avg_time if betterfftw_avg_time > 0 else 0
        
        # Record result
        result = {
            "test_case": name,
            "shape": str(shape),
            "native_time": native_avg_time,
            "betterfftw_time": betterfftw_avg_time,
            "speedup": speedup
        }
        
        results.append(result)
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        # Restore native NumPy/SciPy
        betterfftw.restore_default()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if save_results and not df.empty:
        csv_file = os.path.join(RESULTS_DIR, f"drop_in_replacement_{TIMESTAMP}.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file}")
        
        # Create bar chart of speedups
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='test_case', y='speedup', data=df)
        
        # Add a horizontal line at y=1 (no speedup)
        plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
        
        # Add text annotations
        for i, row in df.iterrows():
            ax.text(i, row['speedup'] + 0.1, f"{row['speedup']:.2f}x", 
                   ha='center', va='bottom', fontsize=10)
        
        plt.title('BetterFFTW Drop-in Replacement Speedup')
        plt.xlabel('Test Case')
        plt.ylabel('Speedup Factor (Native time / BetterFFTW time)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(os.path.join(RESULTS_DIR, f"drop_in_replacement_{TIMESTAMP}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    return df

def benchmark_thread_scaling(n_runs=3, save_results=True):
    """
    Measure how performance scales with thread count for large arrays.
    """
    logger.info("Starting thread scaling benchmark...")
    
    # Skip if only NumPy/SciPy available (no threading support)
    if not HAVE_PYFFTW and not HAVE_BETTERFFTW:
        logger.warning("Skipping thread scaling benchmark: No threading-capable libraries available")
        return pd.DataFrame()
    
    results = []
    
    # Test configurations
    transform_types = [
        {"name": "2D FFT", "func": "fft2", "shape": (2048, 2048), "is_complex": False},
        {"name": "3D FFT", "func": "fftn", "shape": (128, 128, 128), "is_complex": False}
    ]
    
    # Get available CPU count
    max_threads = os.cpu_count()
    thread_counts = [1]
    
    # Add powers of 2 thread counts
    for i in range(1, int(np.log2(max_threads)) + 1):
        thread_count = 2**i
        if thread_count <= max_threads:
            thread_counts.append(thread_count)
    
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
        
        # Create test array
        array = create_test_array(shape, dtype=np.complex128 if is_complex else np.float64, is_complex=is_complex)
        
        # Get single-thread reference times
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
            logger.info(f"  Threads: {thread_count}")
            
            for impl_key, impl in threading_impls.items():
                # Skip thread scaling for NumPy above 1 thread
                if impl_key == 'numpy' and thread_count > 1:
                    continue
                    
                logger.info(f"    Implementation: {impl['name']}")
                
                impl_module = impl["module"]
                impl_func = getattr(impl_module, func_name)
                
                # Setup implementation
                impl["setup"]()
                
                # Set thread count (except for NumPy which doesn't support it)
                kwarg = {"threads": thread_count} if impl_key != 'numpy' else {}
                
                # Warm-up
                _ = impl_func(array.copy(), **kwarg)
                
                # Measure
                times = []
                memory_before = get_memory_usage()
                
                for _ in range(n_runs):
                    array_copy = array.copy()
                    force_gc()
                    
                    start_time = time.time()
                    _ = impl_func(array_copy, **kwarg)
                    end_time = time.time()
                    
                    times.append(end_time - start_time)
                
                memory_after = get_memory_usage()
                memory_delta = memory_after - memory_before
                
                # Calculate statistics
                avg_time = sum(times) / len(times)
                
                # Calculate speedup vs. 1 thread
                reference_time = reference_times[impl_key]
                speedup = reference_time / avg_time if avg_time > 0 else 0
                
                # Calculate efficiency
                efficiency = speedup / thread_count if thread_count > 0 else 0
                
                # Record result
                result = {
                    "implementation": impl["name"],
                    "transform": transform_name,
                    "shape": str(shape),
                    "thread_count": thread_count,
                    "avg_time": avg_time,
                    "reference_time": reference_time,
                    "speedup": speedup,
                    "efficiency": efficiency,
                    "memory_delta_mb": memory_delta
                }
                
                results.append(result)
                logger.info(f"      Time: {avg_time:.6f}s, Speedup: {speedup:.2f}x, Efficiency: {efficiency:.2f}")
                
                # Cleanup
                impl["cleanup"]()
                force_gc()
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    if save_results and not df.empty:
        csv_file = os.path.join(RESULTS_DIR, f"thread_scaling_{TIMESTAMP}.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file}")
        
        # Create speedup plot
        for transform_name in df['transform'].unique():
            transform_df = df[df['transform'] == transform_name]
            
            # Create a line plot for speedup vs thread count
            plt.figure(figsize=(10, 6))
            for impl_name in transform_df['implementation'].unique():
                impl_df = transform_df[transform_df['implementation'] == impl_name]
                if impl_df.empty:
                    continue
                
                plt.plot(impl_df['thread_count'], impl_df['speedup'], marker='o', label=impl_name)
            
            # Add ideal scaling reference line
            max_threads_shown = max(thread_counts)
            plt.plot([1, max_threads_shown], [1, max_threads_shown], 'k--', alpha=0.5, label='Ideal Scaling')
            
            plt.title(f'Thread Scaling for {transform_name}')
            plt.xlabel('Thread Count')
            plt.ylabel('Speedup vs. Single Thread')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.savefig(os.path.join(RESULTS_DIR, f"thread_scaling_{transform_name.replace(' ', '_')}_{TIMESTAMP}.png"), 
                      dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create efficiency plot
        for transform_name in df['transform'].unique():
            transform_df = df[df['transform'] == transform_name]
            
            # Create a line plot for efficiency vs thread count
            plt.figure(figsize=(10, 6))
            for impl_name in transform_df['implementation'].unique():
                impl_df = transform_df[transform_df['implementation'] == impl_name]
                if impl_df.empty or len(impl_df) <= 1:
                    continue
                
                plt.plot(impl_df['thread_count'], impl_df['efficiency'], marker='o', label=impl_name)
            
            # Add reference line for 100% efficiency
            plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='100% Efficiency')
            
            plt.title(f'Thread Efficiency for {transform_name}')
            plt.xlabel('Thread Count')
            plt.ylabel('Efficiency (Speedup / Thread Count)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.savefig(os.path.join(RESULTS_DIR, f"thread_efficiency_{transform_name.replace(' ', '_')}_{TIMESTAMP}.png"), 
                      dpi=300, bbox_inches='tight')
            plt.close()
    
    return df

def create_summary_report(results_dict, save_file=None):
    """Create a comprehensive summary report of all benchmarks."""
    logger.info("Creating summary report...")
    
    report = ["# BetterFFTW Benchmark Summary Report\n"]
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # System information
    sys_info = get_system_info()
    report.append("## System Information\n")
    for key, value in sys_info.items():
        report.append(f"* **{key}**: {value}")
    report.append("\n")
    
    # Direct comparison summary
    if 'direct_comparison' in results_dict:
        df = results_dict['direct_comparison']
        report.append("## Direct Comparison Summary\n")
        
        # Calculate average speedups vs NumPy
        if 'implementation' in df.columns and 'avg_execution_time' in df.columns:
            # Make a pivot table with implementations as columns
            pivot = df.pivot_table(
                index=['dimension', 'transform', 'size'],
                columns='implementation',
                values='avg_execution_time'
            )
            
            # Calculate speedup ratios
            if 'NumPy FFT' in pivot.columns:
                for impl in pivot.columns:
                    if impl != 'NumPy FFT':
                        speedup = pivot['NumPy FFT'] / pivot[impl]
                        report.append(f"### {impl} vs NumPy FFT\n")
                        report.append(f"* **Average Speedup**: {speedup.mean():.2f}x")
                        report.append(f"* **Maximum Speedup**: {speedup.max():.2f}x")
                        report.append(f"* **Minimum Speedup**: {speedup.min():.2f}x\n")
    
    # Repeated transform summary
    if 'repeated_use' in results_dict:
        df = results_dict['repeated_use']
        report.append("## Repeated Transform Summary\n")
        
        if 'implementation' in df.columns and 'amortization_factor' in df.columns:
            # Group by implementation and maximum repetitions
            max_reps = df['repetitions'].max()
            max_rep_df = df[df['repetitions'] == max_reps]
            
            report.append(f"### Amortization Factor at {max_reps} Repetitions\n")
            for impl, group in max_rep_df.groupby('implementation'):
                avg_factor = group['amortization_factor'].mean()
                report.append(f"* **{impl}**: {avg_factor:.2f}x\n")
    
    # Real-world scenario summary
    if 'real_world' in results_dict:
        df = results_dict['real_world']
        report.append("## Real-World Scenario Summary\n")
        
        # Create a table of scenario performance
        if 'implementation' in df.columns and 'avg_time' in df.columns:
            pivot = df.pivot_table(
                index='scenario',
                columns='implementation',
                values='avg_time'
            )
            
            # Add speedup columns
            if 'NumPy FFT' in pivot.columns:
                for impl in pivot.columns:
                    if impl != 'NumPy FFT':
                        pivot[f"{impl} Speedup"] = pivot['NumPy FFT'] / pivot[impl]
            
            # Report the table
            report.append("### Scenario Performance (seconds)\n")
            report.append(f"```\n{pivot.to_string()}\n```\n")
    
    # Drop-in replacement summary
    if 'drop_in_replacement' in results_dict:
        df = results_dict['drop_in_replacement']
        report.append("## Drop-in Replacement Summary\n")
        
        if 'speedup' in df.columns:
            avg_speedup = df['speedup'].mean()
            max_speedup = df['speedup'].max()
            min_speedup = df['speedup'].min()
            
            report.append(f"* **Average Speedup**: {avg_speedup:.2f}x")
            report.append(f"* **Maximum Speedup**: {max_speedup:.2f}x")
            report.append(f"* **Minimum Speedup**: {min_speedup:.2f}x\n")
            
            # Add table of individual test cases
            report.append("### Test Case Performance\n")
            report.append(f"```\n{df[['test_case', 'speedup']].to_string()}\n```\n")
    
    # Thread scaling summary
    if 'thread_scaling' in results_dict:
        df = results_dict['thread_scaling']
        report.append("## Thread Scaling Summary\n")
        
        # Report max speedup achieved for each implementation
        if 'implementation' in df.columns and 'speedup' in df.columns:
            for impl, group in df.groupby('implementation'):
                max_speedup_row = group.loc[group['speedup'].idxmax()]
                
                report.append(f"### {impl}\n")
                report.append(f"* **Max Speedup**: {max_speedup_row['speedup']:.2f}x with {max_speedup_row['thread_count']} threads")
                report.append(f"* **Efficiency at max threads**: {group[group['thread_count'] == group['thread_count'].max()]['efficiency'].values[0]:.2f}\n")
    
    # Overall summary
    report.append("## Overall Findings\n")
    
    # Combine all benchmarks to determine overall BetterFFTW performance
    overall_speedups = []
    
    for benchmark_name, df in results_dict.items():
        if benchmark_name == 'direct_comparison':
            if 'implementation' in df.columns and 'avg_execution_time' in df.columns:
                pivot = df.pivot_table(
                    index=['dimension', 'transform', 'size'],
                    columns='implementation',
                    values='avg_execution_time'
                )
                
                if 'NumPy FFT' in pivot.columns and 'BetterFFTW' in pivot.columns:
                    speedups = pivot['NumPy FFT'] / pivot['BetterFFTW']
                    overall_speedups.extend(speedups.values)
        
        elif benchmark_name == 'real_world':
            if 'implementation' in df.columns and 'avg_time' in df.columns:
                pivot = df.pivot_table(
                    index='scenario',
                    columns='implementation',
                    values='avg_time'
                )
                
                if 'NumPy FFT' in pivot.columns and 'BetterFFTW' in pivot.columns:
                    speedups = pivot['NumPy FFT'] / pivot['BetterFFTW']
                    overall_speedups.extend(speedups.values)
    
    if overall_speedups:
        avg_overall_speedup = sum(overall_speedups) / len(overall_speedups)
        report.append(f"* **Average Overall Speedup vs NumPy**: {avg_overall_speedup:.2f}x\n")
    
    # Save report if requested
    if save_file:
        with open(save_file, 'w') as f:
            f.write('\n'.join(report))
        logger.info(f"Summary report saved to {save_file}")
    
    return '\n'.join(report)

def run_all_benchmarks(n_runs=3):
    """Run all benchmarks and generate a comprehensive report."""
    print_system_info()
    
    results = {}
    
    # Run benchmarks
    print("\nRunning Direct Comparison Benchmark...")
    results['direct_comparison'] = benchmark_direct_comparison(n_runs=n_runs)
    
    print("\nRunning Repeated Use Benchmark...")
    results['repeated_use'] = benchmark_repeated_use(n_runs=n_runs)
    
    print("\nRunning Real-World Benchmark...")
    results['real_world'] = benchmark_real_world(n_runs=n_runs)
    
    print("\nRunning Drop-in Replacement Benchmark...")
    results['drop_in_replacement'] = benchmark_drop_in_replacement(n_runs=n_runs)
    
    print("\nRunning Thread Scaling Benchmark...")
    results['thread_scaling'] = benchmark_thread_scaling(n_runs=n_runs)
    
    # Create and save summary report
    summary_file = os.path.join(RESULTS_DIR, f"summary_report_{TIMESTAMP}.md")
    summary = create_summary_report(results, save_file=summary_file)
    
    print(f"\nAll benchmarks completed. Results saved to {RESULTS_DIR}")
    print(f"Summary report: {summary_file}")
    
    return results, summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BetterFFTW Benchmark Suite")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test")
    parser.add_argument("--output", type=str, default=RESULTS_DIR, help="Output directory")
    parser.add_argument("--benchmark", type=str, default="all",
                       choices=["all", "direct", "repeated", "realworld", "dropin", "threads"],
                       help="Which benchmark to run")
    
    args = parser.parse_args()
    
    # Update output directory if specified
    if args.output != RESULTS_DIR:
        RESULTS_DIR = args.output
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run selected benchmark
    if args.benchmark == "all":
        run_all_benchmarks(n_runs=args.runs)
    elif args.benchmark == "direct":
        benchmark_direct_comparison(n_runs=args.runs)
    elif args.benchmark == "repeated":
        benchmark_repeated_use(n_runs=args.runs)
    elif args.benchmark == "realworld":
        benchmark_real_world(n_runs=args.runs)
    elif args.benchmark == "dropin":
        benchmark_drop_in_replacement(n_runs=args.runs)
    elif args.benchmark == "threads":
        benchmark_thread_scaling(n_runs=args.runs)