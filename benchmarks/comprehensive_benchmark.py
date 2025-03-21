#!/usr/bin/env python
"""
Comprehensive FFT Benchmarking Suite

A thorough benchmark comparing BetterFFTW with NumPy and SciPy FFT implementations
across various dimensions, sizes, data types, and usage patterns.

Features:
- Tests different array shapes (1D, 2D, 3D, rectangular)
- Multiple data types (float32/64, complex64/128)
- Different transform types (FFT, RFFT, etc.)
- Measures both first-call and subsequent call performance
- Controls for thread count and planning strategies
- Evaluates memory usage
- Statistical analysis with multiple runs
- Mixed workload simulation
- Detailed reporting and visualization

Usage:
    python comprehensive_benchmark.py [options]
    
Options:
    --quick               Run a reduced set of benchmarks (faster)
    --full                Run all benchmarks (can take hours)
    --report FILENAME     Save report to specified file
    --plot                Generate plots
    --output-dir DIR      Directory to save outputs (default: ./benchmark_results)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import gc
import argparse
import statistics
import json
import platform
import psutil
import threading
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# To compare with SciPy's FFT implementation
try:
    import scipy.fft as scipy_fft
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# Global variables for configuration
RUNTIME_LIMIT = 20 * 60  # Maximum runtime for the entire benchmark in seconds (20 minutes by default)
REPEAT_SHORT_TEST = 5   # Number of repetitions for short tests
REPEAT_LONG_TEST = 3    # Number of repetitions for longer tests
REPEAT_MEASURES = 5     # Number of times to measure each test for statistical significance
MIN_RUNTIME = 0.1       # Minimum runtime for a single test in seconds

# ============================================================================================
# Utility Functions
# ============================================================================================

def format_size(size_tuple):
    """Format a size tuple into a string."""
    return 'x'.join(str(s) for s in size_tuple)

def format_duration(seconds):
    """Format a duration in seconds to a human-readable string."""
    if seconds < 0.001:
        return f"{seconds*1000000:.2f} µs"
    elif seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.2f}s"

def get_system_info():
    """Get information about the system."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "total_memory_gb": psutil.virtual_memory().total / (1024**3),
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except:
        info["numpy_version"] = "unknown"
        
    try:
        import scipy
        info["scipy_version"] = scipy.__version__
    except:
        info["scipy_version"] = "not installed"
        
    try:
        import betterfftw
        info["betterfftw_version"] = betterfftw.__version__
    except:
        info["betterfftw_version"] = "unknown"
    
    return info

def measure_memory_usage(func, *args, **kwargs):
    """Measure memory usage of a function."""
    # Force garbage collection before measurement
    gc.collect()
    
    # Get baseline memory usage
    process = psutil.Process(os.getpid())
    baseline = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run the function
    result = func(*args, **kwargs)
    
    # Get peak memory usage
    peak = process.memory_info().rss / 1024 / 1024  # MB
    
    # Return the result and the additional memory used
    return result, peak - baseline

def adaptive_repeat_count(func, *args, duration_target=0.5, max_repeats=1000, **kwargs):
    """
    Adaptively determine how many times to repeat a function to get meaningful timing.
    
    Args:
        func: Function to test
        *args: Arguments to pass to the function
        duration_target: Target duration in seconds
        max_repeats: Maximum number of repeats
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Number of repeats needed to reach target duration
    """
    # Start with a single call to estimate duration
    start = time.time()
    func(*args, **kwargs)
    single_duration = time.time() - start
    
    # Estimate repeats needed
    if single_duration < 1e-6:  # Avoid division by zero
        repeats = max_repeats
    else:
        repeats = int(duration_target / single_duration)
        repeats = max(1, min(repeats, max_repeats))
    
    return repeats


# ============================================================================================
# Core Benchmark Classes
# ============================================================================================

class FFTImplementation:
    """Base class for FFT implementations to benchmark."""
    
    def __init__(self, name):
        self.name = name
        
    def setup(self):
        """Setup before benchmarking."""
        pass
        
    def teardown(self):
        """Teardown after benchmarking."""
        pass
    
    def fft(self, array, **kwargs):
        """Compute 1D FFT."""
        raise NotImplementedError
        
    def ifft(self, array, **kwargs):
        """Compute 1D inverse FFT."""
        raise NotImplementedError
        
    def rfft(self, array, **kwargs):
        """Compute 1D real FFT."""
        raise NotImplementedError
        
    def irfft(self, array, **kwargs):
        """Compute 1D inverse real FFT."""
        raise NotImplementedError
        
    def fft2(self, array, **kwargs):
        """Compute 2D FFT."""
        raise NotImplementedError
        
    def ifft2(self, array, **kwargs):
        """Compute 2D inverse FFT."""
        raise NotImplementedError
        
    def rfft2(self, array, **kwargs):
        """Compute 2D real FFT."""
        raise NotImplementedError
        
    def irfft2(self, array, **kwargs):
        """Compute 2D inverse real FFT."""
        raise NotImplementedError
        
    def fftn(self, array, **kwargs):
        """Compute N-dimensional FFT."""
        raise NotImplementedError
        
    def ifftn(self, array, **kwargs):
        """Compute N-dimensional inverse FFT."""
        raise NotImplementedError
        
    def rfftn(self, array, **kwargs):
        """Compute N-dimensional real FFT."""
        raise NotImplementedError
        
    def irfftn(self, array, **kwargs):
        """Compute N-dimensional inverse real FFT."""
        raise NotImplementedError


class NumPyFFT(FFTImplementation):
    """NumPy FFT implementation."""
    
    def __init__(self):
        super().__init__("NumPy")
        
    def fft(self, array, **kwargs):
        return np.fft.fft(array, **kwargs)
        
    def ifft(self, array, **kwargs):
        return np.fft.ifft(array, **kwargs)
        
    def rfft(self, array, **kwargs):
        return np.fft.rfft(array, **kwargs)
        
    def irfft(self, array, **kwargs):
        return np.fft.irfft(array, **kwargs)
        
    def fft2(self, array, **kwargs):
        return np.fft.fft2(array, **kwargs)
        
    def ifft2(self, array, **kwargs):
        return np.fft.ifft2(array, **kwargs)
        
    def rfft2(self, array, **kwargs):
        return np.fft.rfft2(array, **kwargs)
        
    def irfft2(self, array, **kwargs):
        return np.fft.irfft2(array, **kwargs)
        
    def fftn(self, array, **kwargs):
        return np.fft.fftn(array, **kwargs)
        
    def ifftn(self, array, **kwargs):
        return np.fft.ifftn(array, **kwargs)
        
    def rfftn(self, array, **kwargs):
        return np.fft.rfftn(array, **kwargs)
        
    def irfftn(self, array, **kwargs):
        return np.fft.irfftn(array, **kwargs)


class SciPyFFT(FFTImplementation):
    """SciPy FFT implementation."""
    
    def __init__(self):
        super().__init__("SciPy")
        if not HAVE_SCIPY:
            raise ImportError("SciPy not available")
        
    def fft(self, array, **kwargs):
        return scipy_fft.fft(array, **kwargs)
        
    def ifft(self, array, **kwargs):
        return scipy_fft.ifft(array, **kwargs)
        
    def rfft(self, array, **kwargs):
        return scipy_fft.rfft(array, **kwargs)
        
    def irfft(self, array, **kwargs):
        return scipy_fft.irfft(array, **kwargs)
        
    def fft2(self, array, **kwargs):
        return scipy_fft.fft2(array, **kwargs)
        
    def ifft2(self, array, **kwargs):
        return scipy_fft.ifft2(array, **kwargs)
        
    def rfft2(self, array, **kwargs):
        return scipy_fft.rfft2(array, **kwargs)
        
    def irfft2(self, array, **kwargs):
        return scipy_fft.irfft2(array, **kwargs)
        
    def fftn(self, array, **kwargs):
        return scipy_fft.fftn(array, **kwargs)
        
    def ifftn(self, array, **kwargs):
        return scipy_fft.ifftn(array, **kwargs)
        
    def rfftn(self, array, **kwargs):
        return scipy_fft.rfftn(array, **kwargs)
        
    def irfftn(self, array, **kwargs):
        return scipy_fft.irfftn(array, **kwargs)


class BetterFFTW(FFTImplementation):
    """BetterFFTW implementation."""
    
    def __init__(self, threads=None, planner=None):
        self.threads = threads
        self.planner = planner
        
        # Construct a descriptive name based on configuration
        name_parts = ["BetterFFTW"]
        if threads is not None:
            name_parts.append(f"{threads}threads")
        if planner is not None:
            plan_name = planner.replace("FFTW_", "")
            name_parts.append(plan_name.lower())
            
        super().__init__("-".join(name_parts))
        
        # Import the library
        import betterfftw
        self.bfft = betterfftw
        
    def setup(self):
        # Configure BetterFFTW
        if self.threads is not None:
            self.bfft.set_num_threads(self.threads)
        if self.planner is not None:
            self.bfft.set_planner_effort(self.planner)
        
        # Clear caches for a fair comparison
        self.bfft.clear_cache()
        
    def teardown(self):
        # Restore defaults if needed
        pass
    
    def fft(self, array, **kwargs):
        kwargs_copy = kwargs.copy()
        if 's' in kwargs_copy:
            kwargs_copy['n'] = kwargs_copy.pop('s')
        if 'axes' in kwargs_copy:
            kwargs_copy['axis'] = kwargs_copy.pop('axes')
        return self.bfft.fft(array, **kwargs_copy)
        
    def ifft(self, array, **kwargs):
        kwargs_copy = kwargs.copy()
        if 's' in kwargs_copy:
            kwargs_copy['n'] = kwargs_copy.pop('s')
        if 'axes' in kwargs_copy:
            kwargs_copy['axis'] = kwargs_copy.pop('axes')
        return self.bfft.ifft(array, **kwargs_copy)
        
    def rfft(self, array, **kwargs):
        kwargs_copy = kwargs.copy()
        if 's' in kwargs_copy:
            kwargs_copy['n'] = kwargs_copy.pop('s')
        if 'axes' in kwargs_copy:
            kwargs_copy['axis'] = kwargs_copy.pop('axes')
        return self.bfft.rfft(array, **kwargs_copy)
        
    def irfft(self, array, **kwargs):
        kwargs_copy = kwargs.copy()
        if 's' in kwargs_copy:
            kwargs_copy['n'] = kwargs_copy.pop('s')
        if 'axes' in kwargs_copy:
            kwargs_copy['axis'] = kwargs_copy.pop('axes')
        return self.bfft.irfft(array, **kwargs_copy)
        
    def fft2(self, array, **kwargs):
        return self.bfft.fft2(array, **kwargs)
        
    def ifft2(self, array, **kwargs):
        return self.bfft.ifft2(array, **kwargs)
        
    def rfft2(self, array, **kwargs):
        return self.bfft.rfft2(array, **kwargs)
        
    def irfft2(self, array, **kwargs):
        return self.bfft.irfft2(array, **kwargs)
        
    def fftn(self, array, **kwargs):
        return self.bfft.fftn(array, **kwargs)
        
    def ifftn(self, array, **kwargs):
        return self.bfft.ifftn(array, **kwargs)
        
    def rfftn(self, array, **kwargs):
        return self.bfft.rfftn(array, **kwargs)
        
    def irfftn(self, array, **kwargs):
        return self.bfft.irfftn(array, **kwargs)


class BenchmarkTest:
    """Base class for benchmark tests."""
    
    def __init__(self, name, category):
        self.name = name
        self.category = category
        
    def run(self, implementation, quick_mode=False):
        """Run the benchmark for a specific implementation."""
        raise NotImplementedError
        
    def get_description(self):
        """Get a description of the benchmark."""
        raise NotImplementedError


class FFTBenchmark(BenchmarkTest):
    """Benchmark for FFT operations."""
    
    def __init__(self, transform_type, array_info, dtype=np.float64):
        """
        Initialize an FFT benchmark.
        
        Args:
            transform_type: Type of transform ('fft', 'rfft', 'fft2', etc.)
            array_info: Dictionary with array information:
                - shape: Tuple of dimensions
                - is_pow2: Whether dimensions are powers of 2
            dtype: Data type of the array
        """
        self.transform_type = transform_type
        self.array_info = array_info
        self.shape = array_info['shape']
        self.is_pow2 = array_info.get('is_pow2', False)
        self.dtype = dtype
        
        # Determine dimensionality from transform type
        if transform_type in ('fft', 'ifft', 'rfft', 'irfft'):
            self.ndim = 1
        elif transform_type in ('fft2', 'ifft2', 'rfft2', 'irfft2'):
            self.ndim = 2
        elif transform_type in ('fftn', 'ifftn', 'rfftn', 'irfftn'):
            self.ndim = len(self.shape)
        
        # Determine if it's a real or complex transform
        self.is_real = 'r' in transform_type
        
        # Check if it's an inverse transform
        self.is_inverse = 'i' in transform_type
        
        # Create a name and category
        shape_str = 'x'.join(str(s) for s in self.shape)
        dtype_str = str(dtype).replace('numpy.', '')
        pow2_str = 'pow2' if self.is_pow2 else 'nonpow2'
        
        name = f"{transform_type}_{shape_str}_{dtype_str}_{pow2_str}"
        category = f"{self.ndim}D_{pow2_str}"
        
        super().__init__(name, category)
        
    def get_description(self):
        """Get a human-readable description of the benchmark."""
        shape_str = 'x'.join(str(s) for s in self.shape)
        dtype_str = str(self.dtype).replace('numpy.', '')
        pow2_str = 'power-of-2' if self.is_pow2 else 'non-power-of-2'
        
        return f"{self.transform_type} on {shape_str} {dtype_str} array ({pow2_str})"
    
    def _create_input_array(self):
        """Create an appropriate input array based on transform type."""
        # For real transforms, input must be real
        if not self.is_inverse and self.is_real:
            if issubclass(self.dtype, np.complexfloating):
                dtype = np.float64 if self.dtype == np.complex128 else np.float32
            else:
                dtype = self.dtype
            return np.random.random(self.shape).astype(dtype)
        
        # For inverse real transforms, input has special shape and must be complex
        elif self.is_inverse and self.is_real:
            # For irfft, shape of input is different
            if self.ndim == 1:
                shape = list(self.shape)
                shape[-1] = shape[-1] // 2 + 1
                if issubclass(self.dtype, np.complexfloating):
                    dtype = self.dtype
                else:
                    dtype = np.complex128 if self.dtype == np.float64 else np.complex64
                return np.random.random(shape).astype(dtype) + 1j * np.random.random(shape).astype(dtype)
            else:
                # For irfft2/irfftn, shape of last dimension is halved+1
                shape = list(self.shape)
                shape[-1] = shape[-1] // 2 + 1
                if issubclass(self.dtype, np.complexfloating):
                    dtype = self.dtype
                else:
                    dtype = np.complex128 if self.dtype == np.float64 else np.complex64
                return np.random.random(shape).astype(dtype) + 1j * np.random.random(shape).astype(dtype)
        
        # For complex transforms, input can be complex
        else:
            if issubclass(self.dtype, np.complexfloating):
                dtype = self.dtype
            else:
                dtype = np.complex128 if self.dtype == np.float64 else np.complex64
            return np.random.random(self.shape).astype(dtype) + 1j * np.random.random(self.shape).astype(dtype)
    
    def run(self, implementation, quick_mode=False):
        """
        Run the benchmark for a specific implementation.
        
        Args:
            implementation: FFT implementation to benchmark
            quick_mode: Whether to run a reduced set of tests
            
        Returns:
            Dictionary with benchmark results
        """
        # Create the input array
        input_array = self._create_input_array()
        
        # Get the transform function
        transform_func = getattr(implementation, self.transform_type)
        
        # Prepare results
        results = {
            'name': self.name,
            'description': self.get_description(),
            'implementation': implementation.name,
            'first_call': {},
            'subsequent_calls': {},
            'memory_usage': None,
        }
        
        # Set up the implementation
        implementation.setup()
        
        try:
            # Measure first call (includes planning)
            start_time = time.time()
            output = transform_func(input_array.copy())
            first_call_duration = time.time() - start_time
            
            # Force computation by accessing a value
            _ = output[0]
            
            results['first_call']['duration'] = first_call_duration
            
            # Determine how many times to repeat for meaningful timing
            repeats = adaptive_repeat_count(
                transform_func, input_array.copy(), 
                duration_target=MIN_RUNTIME
            )
            
            # Reduce repeats in quick mode
            if quick_mode:
                repeats = min(repeats, 3)
            
            # Measure subsequent calls
            durations = []
            for i in range(REPEAT_MEASURES):
                # Run multiple times and measure
                start_time = time.time()
                for _ in range(repeats):
                    output = transform_func(input_array.copy())
                    # Force computation by accessing a value
                    _ = output[0]
                duration = (time.time() - start_time) / repeats
                durations.append(duration)
            
            # Calculate statistics
            mean_duration = statistics.mean(durations)
            if len(durations) > 1:
                stdev_duration = statistics.stdev(durations)
                results['subsequent_calls']['stdev'] = stdev_duration
                results['subsequent_calls']['stdev_percent'] = (stdev_duration / mean_duration) * 100
            
            results['subsequent_calls']['duration'] = mean_duration
            results['subsequent_calls']['repeats'] = repeats
            
            # Calculate overhead ratio
            results['overhead_ratio'] = first_call_duration / mean_duration
            
            # Measure memory usage (only if not in quick mode)
            if not quick_mode:
                _, memory_usage = measure_memory_usage(transform_func, input_array.copy())
                results['memory_usage'] = memory_usage
            
        finally:
            # Tear down the implementation
            implementation.teardown()
        
        return results


class MixedWorkloadBenchmark(BenchmarkTest):
    """Benchmark simulating a mixed workload of FFT operations."""
    
    def __init__(self, workload_type, size_range=(256, 4096), n_transforms=100):
        """
        Initialize a mixed workload benchmark.
        
        Args:
            workload_type: Type of workload ('random', 'increasing', 'scientific')
            size_range: Tuple of (min_size, max_size)
            n_transforms: Number of transforms to perform
        """
        self.workload_type = workload_type
        self.size_range = size_range
        self.n_transforms = n_transforms
        
        name = f"mixed_{workload_type}_{size_range[0]}_{size_range[1]}_{n_transforms}"
        category = "mixed_workload"
        
        super().__init__(name, category)
    
    def get_description(self):
        """Get a human-readable description of the benchmark."""
        return (f"Mixed {self.workload_type} workload with {self.n_transforms} transforms "
                f"in range {self.size_range[0]}-{self.size_range[1]}")
    
    def _generate_workload(self):
        """Generate a sequence of transforms to perform."""
        min_size, max_size = self.size_range
        workload = []
        
        # Generate transform specifications based on workload type
        if self.workload_type == 'random':
            # Random mixture of transform types and sizes
            for _ in range(self.n_transforms):
                # Randomly choose dimension
                ndim = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
                
                # Randomly choose transform type
                if ndim == 1:
                    transform_type = np.random.choice(['fft', 'ifft', 'rfft', 'irfft'])
                elif ndim == 2:
                    transform_type = np.random.choice(['fft2', 'ifft2', 'rfft2', 'irfft2'])
                else:
                    transform_type = np.random.choice(['fftn', 'ifftn', 'rfftn', 'irfftn'])
                
                # Randomly choose size
                if np.random.random() < 0.5:
                    # Power of 2 size
                    log_min = np.log2(min_size)
                    log_max = np.log2(max_size)
                    log_size = np.random.uniform(log_min, log_max)
                    size = 2 ** int(log_size)
                    is_pow2 = True
                else:
                    # Non-power of 2 size
                    size = np.random.randint(min_size, max_size)
                    is_pow2 = False
                
                # Create a shape based on dimension
                if ndim == 1:
                    shape = (size,)
                elif ndim == 2:
                    if np.random.random() < 0.7:
                        # Square array
                        shape = (size, size)
                    else:
                        # Rectangular array
                        shape = (size, size // 2)
                else:
                    if np.random.random() < 0.7:
                        # Cubic array
                        small_size = max(32, size // 8)  # Limit 3D size to avoid excessive memory
                        shape = (small_size, small_size, small_size)
                    else:
                        # Non-cubic array
                        small_size = max(32, size // 8)
                        shape = (small_size, small_size, small_size // 2)
                
                # Choose data type
                dtype = np.random.choice([np.float32, np.float64, np.complex64, np.complex128])
                
                # Add to workload
                workload.append({
                    'transform_type': transform_type,
                    'shape': shape,
                    'is_pow2': is_pow2,
                    'dtype': dtype,
                })
                
        elif self.workload_type == 'increasing':
            # Gradually increasing sizes with mix of transform types
            transforms = ['fft', 'fft2', 'fftn', 'rfft', 'rfft2', 'rfftn']
            sizes = np.linspace(min_size, max_size, self.n_transforms).astype(int)
            
            for i, size in enumerate(sizes):
                # Cycle through transform types
                transform_type = transforms[i % len(transforms)]
                
                # Create shape based on transform type
                if transform_type in ('fft', 'ifft', 'rfft', 'irfft'):
                    shape = (size,)
                elif transform_type in ('fft2', 'ifft2', 'rfft2', 'irfft2'):
                    shape = (size, size)
                else:
                    # Limit 3D size to avoid excessive memory
                    small_size = max(32, size // 8)
                    shape = (small_size, small_size, small_size)
                
                # Alternate between power-of-2 and non-power-of-2
                is_pow2 = (i % 2 == 0)
                if is_pow2:
                    # Find nearest power of 2
                    log_size = np.log2(size)
                    size = 2 ** int(log_size)
                    shape = tuple(2 ** int(np.log2(s)) for s in shape)
                
                # Choose data type (alternate)
                if i % 4 == 0:
                    dtype = np.float32
                elif i % 4 == 1:
                    dtype = np.float64
                elif i % 4 == 2:
                    dtype = np.complex64
                else:
                    dtype = np.complex128
                
                # Add to workload
                workload.append({
                    'transform_type': transform_type,
                    'shape': shape,
                    'is_pow2': is_pow2,
                    'dtype': dtype,
                })
                
        elif self.workload_type == 'scientific':
            # Simulate a scientific computing workload
            # Common patterns: repeated FFTs of same size, occasional large FFTs
            
            # First, add some small 1D FFTs (time series analysis)
            small_sizes = [256, 512, 1024]
            for _ in range(20):
                size = np.random.choice(small_sizes)
                workload.append({
                    'transform_type': 'fft',
                    'shape': (size,),
                    'is_pow2': True,
                    'dtype': np.complex128,
                })
            
            # Add some 2D FFTs (image processing)
            medium_sizes = [256, 512, 1024]
            for _ in range(50):
                size = np.random.choice(medium_sizes)
                workload.append({
                    'transform_type': 'fft2',
                    'shape': (size, size),
                    'is_pow2': True,
                    'dtype': np.complex128,
                })
            
            # Add a few large 2D FFTs (high-res images)
            large_sizes = [2048, 4096]
            for _ in range(5):
                size = np.random.choice(large_sizes)
                workload.append({
                    'transform_type': 'fft2',
                    'shape': (size, size),
                    'is_pow2': True,
                    'dtype': np.complex128,
                })
            
            # Add some 3D FFTs (volumetric data)
            for _ in range(10):
                size = np.random.choice([64, 128, 256])
                workload.append({
                    'transform_type': 'fftn',
                    'shape': (size, size, size),
                    'is_pow2': True,
                    'dtype': np.complex128,
                })
            
            # Add some real transforms (common in scientific computing)
            for _ in range(15):
                size = np.random.choice([512, 1024, 2048])
                workload.append({
                    'transform_type': 'rfft2',
                    'shape': (size, size),
                    'is_pow2': True,
                    'dtype': np.float64,
                })
            
        return workload
    
    def run(self, implementation, quick_mode=False):
        """
        Run the benchmark for a specific implementation.
        
        Args:
            implementation: FFT implementation to benchmark
            quick_mode: Whether to run a reduced set of tests
            
        Returns:
            Dictionary with benchmark results
        """
        # Generate the workload
        workload = self._generate_workload()
        
        # Adjust workload size for quick mode
        if quick_mode:
            workload = workload[:min(20, len(workload))]
        
        # Prepare results
        results = {
            'name': self.name,
            'description': self.get_description(),
            'implementation': implementation.name,
            'n_transforms': len(workload),
            'durations': [],
            'total_duration': 0,
            'details': [],
        }
        
        # Set up the implementation
        implementation.setup()
        
        try:
            # Run each transform in the workload
            start_time = time.time()
            for i, transform_spec in enumerate(workload):
                # Create input array
                transform_type = transform_spec['transform_type']
                shape = transform_spec['shape']
                dtype = transform_spec['dtype']
                
                # Create benchmark for this transform
                benchmark = FFTBenchmark(transform_type, {
                    'shape': shape,
                    'is_pow2': transform_spec['is_pow2'],
                }, dtype)
                
                # Run the benchmark
                transform_result = benchmark.run(implementation, quick_mode=True)
                
                # Add to results
                results['durations'].append(transform_result['subsequent_calls']['duration'])
                results['details'].append({
                    'transform_type': transform_type,
                    'shape': shape,
                    'dtype': str(dtype),
                    'is_pow2': transform_spec['is_pow2'],
                    'duration': transform_result['subsequent_calls']['duration'],
                })
            
            # Calculate total duration
            results['total_duration'] = time.time() - start_time
            
            # Calculate statistics
            if results['durations']:
                results['mean_duration'] = statistics.mean(results['durations'])
                if len(results['durations']) > 1:
                    results['stdev_duration'] = statistics.stdev(results['durations'])
        
        finally:
            # Tear down the implementation
            implementation.teardown()
        
        return results


# ============================================================================================
# Benchmark Suite Configuration
# ============================================================================================

def create_benchmark_suite(quick_mode=False):
    """Create a suite of benchmarks to run."""
    benchmarks = []
    
    # Define array shapes to test
    if quick_mode:
        # Reduced set for quick testing
        pow2_1d = [(256,), (1024,), (4096,)]
        pow2_2d = [(64, 64), (512, 512)]
        pow2_3d = [(64, 64, 64)]
        
        nonpow2_1d = [(384,), (768,), (3072,)]
        nonpow2_2d = [(96, 96), (768, 768)]
        nonpow2_3d = [(48, 48, 48)]
        
        # Rectangular arrays
        rect_2d = [(256, 512), (512, 256)]
    else:
        # Full set of test cases
        pow2_1d = [(256,), (1024,), (4096,), (16384,), (65536,)]
        pow2_2d = [(64, 64), (256, 256), (512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
        pow2_3d = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]
        
        nonpow2_1d = [(384,), (768,), (1536,), (3072,), (6144,), (12288,)]
        nonpow2_2d = [(96, 96), (384, 384), (768, 768), (1536, 1536), (3072, 3072)]
        nonpow2_3d = [(48, 48, 48), (96, 96, 96), (192, 192, 192)]
        
        # Rectangular arrays
        rect_2d = [(256, 512), (512, 256), (1024, 512), (512, 1024)]
    
    # Data types to test
    dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    
    # Create benchmarks for 1D transforms
    transform_types_1d = ['fft', 'ifft', 'rfft', 'irfft']
    for shape in pow2_1d:
        for transform_type in transform_types_1d:
            for dtype in dtypes:
                # Skip complex dtype for real input transforms
                if 'r' in transform_type and not 'i' in transform_type and issubclass(dtype, np.complexfloating):
                    continue
                benchmarks.append(FFTBenchmark(transform_type, {
                    'shape': shape,
                    'is_pow2': True,
                }, dtype))
    
    for shape in nonpow2_1d:
        for transform_type in transform_types_1d:
            # Only test float64 and complex128 for non-power-of-2 to reduce test time
            for dtype in [np.float64, np.complex128]:
                # Skip complex dtype for real input transforms
                if 'r' in transform_type and not 'i' in transform_type and issubclass(dtype, np.complexfloating):
                    continue
                benchmarks.append(FFTBenchmark(transform_type, {
                    'shape': shape,
                    'is_pow2': False,
                }, dtype))
    
    # Create benchmarks for 2D transforms
    transform_types_2d = ['fft2', 'ifft2', 'rfft2', 'irfft2']
    for shape in pow2_2d:
        for transform_type in transform_types_2d:
            # Only test float64 and complex128 for 2D to reduce test time
            for dtype in [np.float64, np.complex128]:
                # Skip complex dtype for real input transforms
                if 'r' in transform_type and not 'i' in transform_type and issubclass(dtype, np.complexfloating):
                    continue
                benchmarks.append(FFTBenchmark(transform_type, {
                    'shape': shape,
                    'is_pow2': True,
                }, dtype))
    
    for shape in nonpow2_2d:
        for transform_type in transform_types_2d:
            # Only test float64 for non-power-of-2 2D to reduce test time
            benchmarks.append(FFTBenchmark(transform_type, {
                'shape': shape,
                'is_pow2': False,
            }, np.float64))
    
    # Create benchmarks for rectangular arrays
    for shape in rect_2d:
        for transform_type in ['fft2', 'rfft2']:
            benchmarks.append(FFTBenchmark(transform_type, {
                'shape': shape,
                'is_pow2': all((d & (d-1) == 0) for d in shape),
            }, np.float64))
    
    # Create benchmarks for 3D transforms
    transform_types_3d = ['fftn', 'ifftn', 'rfftn', 'irfftn']
    for shape in pow2_3d:
        for transform_type in transform_types_3d:
            # Only test float64 for 3D to reduce test time
            benchmarks.append(FFTBenchmark(transform_type, {
                'shape': shape,
                'is_pow2': True,
            }, np.float64))
    
    for shape in nonpow2_3d:
        for transform_type in transform_types_3d:
            # Only test float64 for non-power-of-2 3D to reduce test time
            benchmarks.append(FFTBenchmark(transform_type, {
                'shape': shape,
                'is_pow2': False,
            }, np.float64))
    
    # Add mixed workload benchmarks
    if quick_mode:
        # Only add one mixed workload in quick mode
        benchmarks.append(MixedWorkloadBenchmark('scientific', (256, 2048), 20))
    else:
        # Add various mixed workloads
        for workload_type in ['random', 'increasing', 'scientific']:
            benchmarks.append(MixedWorkloadBenchmark(workload_type, (256, 4096), 100))
    
    return benchmarks


def create_implementations(test_betterfftw_configs=True):
    """Create implementations to benchmark."""
    implementations = [NumPyFFT()]
    
    if HAVE_SCIPY:
        implementations.append(SciPyFFT())
    
    # Add BetterFFTW with different configurations
    implementations.append(BetterFFTW())  # Default configuration
    
    if test_betterfftw_configs:
        # Test different thread counts
        cpu_count = os.cpu_count() or 4
        if cpu_count >= 4:
            implementations.append(BetterFFTW(threads=1))
            implementations.append(BetterFFTW(threads=cpu_count//2))
            implementations.append(BetterFFTW(threads=cpu_count))
        
        # Test different planner strategies
        implementations.append(BetterFFTW(planner='FFTW_ESTIMATE'))
        implementations.append(BetterFFTW(planner='FFTW_MEASURE'))
        implementations.append(BetterFFTW(planner='FFTW_PATIENT'))
    
    return implementations


# ============================================================================================
# Benchmark Execution and Reporting
# ============================================================================================

def run_benchmark_suite(benchmarks, implementations, quick_mode=False, output_dir="benchmark_results"):
    """
    Run all benchmarks with all implementations.
    
    Args:
        benchmarks: List of benchmarks to run
        implementations: List of implementations to benchmark
        quick_mode: Whether to run a reduced set of tests
        output_dir: Directory to save results
    
    Returns:
        Dictionary with all results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare results dictionary
    results = {
        'system_info': get_system_info(),
        'quick_mode': quick_mode,
        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'benchmarks': {},
    }
    
    # Track overall progress
    total_benchmarks = len(benchmarks) * len(implementations)
    completed_benchmarks = 0
    start_time = time.time()
    
    # Run each benchmark with each implementation
    for benchmark in benchmarks:
        print(f"\nRunning benchmark: {benchmark.name}")
        results['benchmarks'][benchmark.name] = {
            'description': benchmark.get_description(),
            'category': benchmark.category,
            'implementations': {},
        }
        
        for implementation in implementations:
            # Check if we've exceeded the runtime limit
            if time.time() - start_time > RUNTIME_LIMIT:
                print(f"Runtime limit of {RUNTIME_LIMIT} seconds exceeded. Stopping benchmark.")
                break
                
            print(f"  Testing implementation: {implementation.name}")
            try:
                # Run the benchmark for this implementation
                result = benchmark.run(implementation, quick_mode=quick_mode)
                
                # Store the result
                results['benchmarks'][benchmark.name]['implementations'][implementation.name] = result
                
                # Print quick summary
                if 'subsequent_calls' in result and 'duration' in result['subsequent_calls']:
                    duration = result['subsequent_calls']['duration']
                    print(f"    Duration: {format_duration(duration)}")
                    
                    # If we have results from NumPy, calculate speedup
                    if 'NumPy' in results['benchmarks'][benchmark.name]['implementations']:
                        numpy_result = results['benchmarks'][benchmark.name]['implementations']['NumPy']
                        if 'subsequent_calls' in numpy_result and 'duration' in numpy_result['subsequent_calls']:
                            numpy_duration = numpy_result['subsequent_calls']['duration']
                            speedup = numpy_duration / duration
                            print(f"    Speedup vs NumPy: {speedup:.2f}x")
                elif 'total_duration' in result:
                    duration = result['total_duration']
                    print(f"    Total duration: {format_duration(duration)}")
                
            except Exception as e:
                print(f"    Error: {str(e)}")
                results['benchmarks'][benchmark.name]['implementations'][implementation.name] = {
                    'error': str(e),
                }
            
            # Update progress
            completed_benchmarks += 1
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / completed_benchmarks) * total_benchmarks
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"  Progress: {completed_benchmarks}/{total_benchmarks} ({completed_benchmarks/total_benchmarks*100:.1f}%)")
            print(f"  Elapsed: {format_duration(elapsed_time)}, Remaining: {format_duration(remaining_time)}")
            
            # Periodically save results
            if completed_benchmarks % 10 == 0:
                results['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                results['elapsed_time'] = elapsed_time
                with open(os.path.join(output_dir, "benchmark_results_partial.json"), 'w') as f:
                    json.dump(results, f, indent=2)
    
    # Final timing information
    end_time = time.time()
    total_duration = end_time - start_time
    results['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results['elapsed_time'] = total_duration
    
    # Save full results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(output_dir, f"benchmark_results_{timestamp}.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save as latest
    with open(os.path.join(output_dir, "benchmark_results_latest.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark completed in {format_duration(total_duration)}")
    print(f"Results saved to {output_dir}")
    
    return results


def analyze_results(results):
    """
    Analyze benchmark results.
    
    Args:
        results: Results dictionary from run_benchmark_suite
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'system_info': results['system_info'],
        'overall_summary': {},
        'categories': {},
        'implementations': {},
        'detailed_benchmarks': {},
    }
    
    # Extract implementations
    all_implementations = set()
    numpy_name = 'NumPy'  # Default reference implementation
    
    for benchmark_name, benchmark_results in results['benchmarks'].items():
        for implementation_name in benchmark_results['implementations'].keys():
            all_implementations.add(implementation_name)
            
    all_implementations = sorted(list(all_implementations))
    
    # Initialize data structures
    for implementation in all_implementations:
        analysis['implementations'][implementation] = {
            'total_benchmarks': 0,
            'successful_benchmarks': 0,
            'speedups': [],
            'speedups_by_category': {},
            'categories': {},
        }
    
    # Process each benchmark
    for benchmark_name, benchmark_results in results['benchmarks'].items():
        category = benchmark_results['category']
        
        # Initialize category if needed
        if category not in analysis['categories']:
            analysis['categories'][category] = {
                'benchmarks': 0,
                'implementations': {},
            }
            
            for implementation in all_implementations:
                analysis['categories'][category]['implementations'][implementation] = {
                    'total_benchmarks': 0,
                    'successful_benchmarks': 0,
                    'speedups': [],
                }
                analysis['implementations'][implementation]['categories'][category] = {
                    'speedups': [],
                }
        
        # Initialize benchmark analysis
        analysis['detailed_benchmarks'][benchmark_name] = {
            'category': category,
            'description': benchmark_results['description'],
            'implementations': {},
        }
        
        # Track if NumPy result is available for this benchmark
        numpy_result = None
        if numpy_name in benchmark_results['implementations']:
            impl_result = benchmark_results['implementations'][numpy_name]
            if ('subsequent_calls' in impl_result and 
                'duration' in impl_result['subsequent_calls'] and
                not impl_result.get('error')):
                numpy_result = impl_result['subsequent_calls']['duration']
        
        # Process each implementation
        for implementation, impl_result in benchmark_results['implementations'].items():
            # Track total benchmarks
            analysis['implementations'][implementation]['total_benchmarks'] += 1
            analysis['categories'][category]['implementations'][implementation]['total_benchmarks'] += 1
            analysis['categories'][category]['benchmarks'] += 1 / len(benchmark_results['implementations'])
            
            # Check if the benchmark was successful
            success = False
            duration = None
            memory = None
            speedup = None
            
            if 'error' not in impl_result:
                if 'subsequent_calls' in impl_result and 'duration' in impl_result['subsequent_calls']:
                    success = True
                    duration = impl_result['subsequent_calls']['duration']
                    
                    if 'memory_usage' in impl_result and impl_result['memory_usage'] is not None:
                        memory = impl_result['memory_usage']
                        
                    # Calculate speedup if NumPy result is available
                    if numpy_result is not None and implementation != numpy_name:
                        speedup = numpy_result / duration
                        
                        # Add to speedup lists
                        analysis['implementations'][implementation]['speedups'].append(speedup)
                        analysis['categories'][category]['implementations'][implementation]['speedups'].append(speedup)
                        analysis['implementations'][implementation]['categories'][category]['speedups'].append(speedup)
                        
                elif 'total_duration' in impl_result:
                    # Mixed workload benchmark
                    success = True
                    duration = impl_result['total_duration']
                    
                    # Calculate speedup for mixed workload
                    if numpy_name in benchmark_results['implementations']:
                        numpy_mixed = benchmark_results['implementations'][numpy_name]
                        if 'total_duration' in numpy_mixed and implementation != numpy_name:
                            speedup = numpy_mixed['total_duration'] / duration
                            
                            # Add to speedup lists
                            analysis['implementations'][implementation]['speedups'].append(speedup)
                            analysis['categories'][category]['implementations'][implementation]['speedups'].append(speedup)
                            analysis['implementations'][implementation]['categories'][category]['speedups'].append(speedup)
            
            # Track successful benchmarks
            if success:
                analysis['implementations'][implementation]['successful_benchmarks'] += 1
                analysis['categories'][category]['implementations'][implementation]['successful_benchmarks'] += 1
            
            # Store implementation details
            analysis['detailed_benchmarks'][benchmark_name]['implementations'][implementation] = {
                'success': success,
                'duration': duration,
                'memory': memory,
                'speedup': speedup,
                'error': impl_result.get('error'),
            }
    
    # Calculate overall summary
    for implementation in all_implementations:
        impl_analysis = analysis['implementations'][implementation]
        speedups = impl_analysis['speedups']
        
        if implementation != numpy_name and speedups:
            impl_analysis['mean_speedup'] = sum(speedups) / len(speedups)
            impl_analysis['min_speedup'] = min(speedups)
            impl_analysis['max_speedup'] = max(speedups)
            
            if len(speedups) > 1:
                impl_analysis['median_speedup'] = statistics.median(speedups)
                impl_analysis['stdev_speedup'] = statistics.stdev(speedups)
            else:
                impl_analysis['median_speedup'] = speedups[0]
                impl_analysis['stdev_speedup'] = 0
        
        # Calculate success rate
        if impl_analysis['total_benchmarks'] > 0:
            impl_analysis['success_rate'] = (impl_analysis['successful_benchmarks'] / 
                                           impl_analysis['total_benchmarks'] * 100)
        else:
            impl_analysis['success_rate'] = 0
            
        # Calculate category summaries
        for category, cat_data in impl_analysis['categories'].items():
            speedups = cat_data['speedups']
            
            if implementation != numpy_name and speedups:
                cat_data['mean_speedup'] = sum(speedups) / len(speedups)
                cat_data['min_speedup'] = min(speedups)
                cat_data['max_speedup'] = max(speedups)
                
                if len(speedups) > 1:
                    cat_data['median_speedup'] = statistics.median(speedups)
                    cat_data['stdev_speedup'] = statistics.stdev(speedups)
                else:
                    cat_data['median_speedup'] = speedups[0]
                    cat_data['stdev_speedup'] = 0
    
    # Calculate category summaries
    for category, cat_data in analysis['categories'].items():
        for implementation, impl_data in cat_data['implementations'].items():
            speedups = impl_data['speedups']
            
            if implementation != numpy_name and speedups:
                impl_data['mean_speedup'] = sum(speedups) / len(speedups)
                impl_data['min_speedup'] = min(speedups)
                impl_data['max_speedup'] = max(speedups)
                
                if len(speedups) > 1:
                    impl_data['median_speedup'] = statistics.median(speedups)
                    impl_data['stdev_speedup'] = statistics.stdev(speedups)
                else:
                    impl_data['median_speedup'] = speedups[0]
                    impl_data['stdev_speedup'] = 0
            
            # Calculate success rate
            if impl_data['total_benchmarks'] > 0:
                impl_data['success_rate'] = (impl_data['successful_benchmarks'] / 
                                           impl_data['total_benchmarks'] * 100)
            else:
                impl_data['success_rate'] = 0
    
    # Overall summary
    analysis['overall_summary']['total_benchmarks'] = len(results['benchmarks'])
    analysis['overall_summary']['total_implementations'] = len(all_implementations)
    
    best_mean_speedup = 0
    best_implementation = None
    
    for implementation in all_implementations:
        if implementation != numpy_name:
            impl_analysis = analysis['implementations'][implementation]
            if 'mean_speedup' in impl_analysis and impl_analysis['mean_speedup'] > best_mean_speedup:
                best_mean_speedup = impl_analysis['mean_speedup']
                best_implementation = implementation
    
    if best_implementation:
        analysis['overall_summary']['best_implementation'] = best_implementation
        analysis['overall_summary']['best_mean_speedup'] = best_mean_speedup
    
    return analysis


def generate_report(analysis, output_dir="benchmark_results"):
    """
    Generate a human-readable report from the analysis results.
    
    Args:
        analysis: Analysis dictionary from analyze_results
        output_dir: Directory to save the report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare report
    report = []
    report.append("# FFT BENCHMARK REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # System information
    report.append("## System Information")
    report.append(f"Platform: {analysis['system_info']['platform']}")
    report.append(f"CPU: {analysis['system_info']['processor']}")
    report.append(f"CPU Count: {analysis['system_info']['cpu_count']} (Physical cores: {analysis['system_info']['physical_cores']})")
    report.append(f"Memory: {analysis['system_info']['total_memory_gb']:.2f} GB")
    report.append(f"Python: {analysis['system_info']['python_version']}")
    report.append(f"NumPy: {analysis['system_info']['numpy_version']}")
    report.append(f"SciPy: {analysis['system_info']['scipy_version']}")
    report.append(f"BetterFFTW: {analysis['system_info']['betterfftw_version']}")
    report.append("")
    
    # Overall summary
    report.append("## Overall Summary")
    report.append(f"Total benchmarks: {analysis['overall_summary']['total_benchmarks']}")
    report.append(f"Total implementations: {analysis['overall_summary']['total_implementations']}")
    
    if 'best_implementation' in analysis['overall_summary']:
        report.append(f"Best overall implementation: {analysis['overall_summary']['best_implementation']} " +
                    f"(Mean speedup: {analysis['overall_summary']['best_mean_speedup']:.2f}x)")
    report.append("")
    
    # Implementation summary
    report.append("## Implementation Summary")
    for implementation, impl_data in analysis['implementations'].items():
        if implementation == 'NumPy':
            report.append(f"### {implementation} (Reference)")
        else:
            report.append(f"### {implementation}")
            
        report.append(f"Success rate: {impl_data['success_rate']:.1f}% " +
                    f"({impl_data['successful_benchmarks']}/{impl_data['total_benchmarks']})")
        
        if implementation != 'NumPy' and 'mean_speedup' in impl_data:
            report.append(f"Mean speedup: {impl_data['mean_speedup']:.2f}x")
            report.append(f"Median speedup: {impl_data['median_speedup']:.2f}x")
            report.append(f"Min speedup: {impl_data['min_speedup']:.2f}x")
            report.append(f"Max speedup: {impl_data['max_speedup']:.2f}x")
            if 'stdev_speedup' in impl_data:
                report.append(f"Std. dev.: {impl_data['stdev_speedup']:.2f}")
        
        report.append("")
        
        # Implementation category summary
        if 'categories' in impl_data:
            report.append("#### Category Performance")
            for category, cat_data in impl_data['categories'].items():
                report.append(f"**{category}**:")
                
                if implementation != 'NumPy' and 'mean_speedup' in cat_data:
                    report.append(f"Mean speedup: {cat_data['mean_speedup']:.2f}x, " +
                                f"Range: {cat_data['min_speedup']:.2f}x - {cat_data['max_speedup']:.2f}x")
            
            report.append("")
    
    # Category summary
    report.append("## Category Summary")
    for category, cat_data in analysis['categories'].items():
        report.append(f"### {category}")
        report.append(f"Total benchmarks: {int(cat_data['benchmarks'])}")
        report.append("")
        
        # Add implementation details for this category
        for implementation, impl_data in cat_data['implementations'].items():
            if implementation == 'NumPy':
                continue  # Skip NumPy as it's the reference
                
            report.append(f"**{implementation}**:")
            
            if 'mean_speedup' in impl_data:
                report.append(f"Mean speedup: {impl_data['mean_speedup']:.2f}x, " +
                            f"Median: {impl_data['median_speedup']:.2f}x, " +
                            f"Range: {impl_data['min_speedup']:.2f}x - {impl_data['max_speedup']:.2f}x")
        
        report.append("")
    
    # Key insights
    report.append("## Key Insights")
    
    # Determine the best implementation for each category
    category_winners = {}
    for category in analysis['categories']:
        best_speedup = 0
        best_impl = None
        
        for implementation, impl_data in analysis['implementations'].items():
            if implementation == 'NumPy':
                continue
                
            if category in impl_data['categories'] and 'mean_speedup' in impl_data['categories'][category]:
                cat_speedup = impl_data['categories'][category]['mean_speedup']
                if cat_speedup > best_speedup:
                    best_speedup = cat_speedup
                    best_impl = implementation
        
        if best_impl:
            category_winners[category] = (best_impl, best_speedup)
    
    # Add insights
    for category, (impl, speedup) in category_winners.items():
        report.append(f"- For {category} transforms, {impl} is the fastest with {speedup:.2f}x mean speedup over NumPy")
    
    # Significant findings
    significant_speedups = []
    slowdowns = []
    
    for benchmark_name, benchmark_data in analysis['detailed_benchmarks'].items():
        for implementation, impl_data in benchmark_data['implementations'].items():
            if implementation == 'NumPy':
                continue
                
            if impl_data['speedup']:
                if impl_data['speedup'] > 2.0:
                    significant_speedups.append((benchmark_name, implementation, impl_data['speedup']))
                elif impl_data['speedup'] < 0.8:
                    slowdowns.append((benchmark_name, implementation, impl_data['speedup']))
    
    # Sort by speedup
    significant_speedups.sort(key=lambda x: x[2], reverse=True)
    slowdowns.sort(key=lambda x: x[2])
    
    if significant_speedups:
        report.append("")
        report.append("### Top 5 Speedups")
        for i, (benchmark, impl, speedup) in enumerate(significant_speedups[:5]):
            benchmark_desc = analysis['detailed_benchmarks'][benchmark]['description']
            report.append(f"{i+1}. {impl} on {benchmark_desc}: {speedup:.2f}x speedup")
    
    if slowdowns:
        report.append("")
        report.append("### Significant Slowdowns")
        for i, (benchmark, impl, speedup) in enumerate(slowdowns[:5]):
            benchmark_desc = analysis['detailed_benchmarks'][benchmark]['description']
            report.append(f"{i+1}. {impl} on {benchmark_desc}: {speedup:.2f}x speedup (slowdown)")
    
    # Save report
    report_str = '\n'.join(report)
    with open(os.path.join(output_dir, f"benchmark_report_{timestamp}.md"), 'w') as f:
        f.write(report_str)
    
    # Also save as latest
    with open(os.path.join(output_dir, "benchmark_report_latest.md"), 'w') as f:
        f.write(report_str)
    
    print(f"Report saved to {output_dir}")
    
    return report_str


def generate_plots(analysis, output_dir="benchmark_results"):
    """
    Generate plots from the analysis results.
    
    Args:
        analysis: Analysis dictionary from analyze_results
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get implementations (excluding NumPy)
    implementations = [impl for impl in analysis['implementations'] if impl != 'NumPy']
    
    # Get categories
    categories = list(analysis['categories'].keys())
    
    # Colors for implementations
    colors = plt.cm.tab10(np.linspace(0, 1, len(implementations)))
    color_map = {impl: colors[i] for i, impl in enumerate(implementations)}
    
    # Timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Overall speedup comparison
    plt.figure(figsize=(10, 6))
    
    # Collect mean speedups
    mean_speedups = []
    labels = []
    colors_list = []
    
    for implementation in implementations:
        if 'mean_speedup' in analysis['implementations'][implementation]:
            mean_speedups.append(analysis['implementations'][implementation]['mean_speedup'])
            labels.append(implementation)
            colors_list.append(color_map[implementation])
    
    # Sort by speedup
    sorted_indices = np.argsort(mean_speedups)[::-1]
    mean_speedups = [mean_speedups[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    colors_list = [colors_list[i] for i in sorted_indices]
    
    # Plot
    bars = plt.bar(range(len(mean_speedups)), mean_speedups, color=colors_list)
    
    # Add horizontal line at 1.0 (NumPy baseline)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='NumPy baseline')
    
    # Customize
    plt.xlabel('Implementation')
    plt.ylabel('Mean Speedup vs NumPy (higher is better)')
    plt.title('Overall Performance Comparison')
    plt.xticks(range(len(mean_speedups)), labels, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Add speedup values on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{mean_speedups[i]:.2f}x', ha='center', va='bottom')
    
    # Save
    plt.savefig(os.path.join(output_dir, f"overall_speedup_{timestamp}.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "overall_speedup_latest.png"), dpi=150)
    plt.close()
    
    # 2. Category comparison
    plt.figure(figsize=(12, 8))
    
    # Collect data
    category_data = []
    
    for category in categories:
        cat_data = {'category': category, 'implementations': {}}
        
        for implementation in implementations:
            if (category in analysis['implementations'][implementation]['categories'] and
                'mean_speedup' in analysis['implementations'][implementation]['categories'][category]):
                cat_data['implementations'][implementation] = analysis['implementations'][implementation]['categories'][category]['mean_speedup']
        
        if cat_data['implementations']:
            category_data.append(cat_data)
    
    # Arrange data for plotting
    x = np.arange(len(category_data))
    width = 0.8 / len(implementations)
    offsets = np.linspace(-(len(implementations)-1)/2*width, (len(implementations)-1)/2*width, len(implementations))
    
    # Plot each implementation as a group of bars
    for i, implementation in enumerate(implementations):
        values = []
        valid_categories = []
        indices = []
        
        for j, cat_data in enumerate(category_data):
            if implementation in cat_data['implementations']:
                values.append(cat_data['implementations'][implementation])
                valid_categories.append(cat_data['category'])
                indices.append(j)
        
        if values:
            bars = plt.bar(x[indices] + offsets[i], values, width, label=implementation, color=color_map[implementation])
    
    # Add horizontal line at 1.0 (NumPy baseline)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='NumPy baseline')
    
    # Customize
    plt.xlabel('Category')
    plt.ylabel('Mean Speedup vs NumPy (higher is better)')
    plt.title('Performance by Category')
    plt.xticks(x, [data['category'] for data in category_data], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(output_dir, f"category_comparison_{timestamp}.png"), dpi=150)
    plt.savefig(os.path.join(output_dir, "category_comparison_latest.png"), dpi=150)
    plt.close()
    
    # 3. Size trend analysis
    # For 2D power-of-2 and non-power-of-2, see how performance varies with size
    for is_pow2 in [True, False]:
        pow2_str = "pow2" if is_pow2 else "nonpow2"
        
        # Collect data for 2D transforms
        size_data = {}
        
        for benchmark_name, benchmark_data in analysis['detailed_benchmarks'].items():
            if benchmark_data['category'] == f"2D_{pow2_str}":
                # Extract size from description (e.g., "fft2 on 512x512 float64 array (power-of-2)")
                desc = benchmark_data['description']
                size_str = desc.split(' on ')[1].split(' ')[0]
                
                if 'x' in size_str:
                    size = int(size_str.split('x')[0])  # Just use first dimension for square arrays
                    
                    if size not in size_data:
                        size_data[size] = {}
                    
                    for implementation, impl_data in benchmark_data['implementations'].items():
                        if implementation != 'NumPy' and impl_data['speedup']:
                            if implementation not in size_data[size]:
                                size_data[size][implementation] = []
                            
                            size_data[size][implementation].append(impl_data['speedup'])
        
        # Only plot if we have data
        if size_data:
            plt.figure(figsize=(10, 6))
            
            # Calculate average speedup for each size
            sizes = sorted(size_data.keys())
            
            for implementation in implementations:
                speedups = []
                valid_sizes = []
                
                for size in sizes:
                    if implementation in size_data[size]:
                        mean_speedup = sum(size_data[size][implementation]) / len(size_data[size][implementation])
                        speedups.append(mean_speedup)
                        valid_sizes.append(size)
                
                if speedups:
                    plt.plot(valid_sizes, speedups, 'o-', label=implementation, color=color_map[implementation])
            
            # Add horizontal line at 1.0 (NumPy baseline)
            plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='NumPy baseline')
            
            # Customize
            plt.xlabel('Array Size (NxN)')
            plt.ylabel('Mean Speedup vs NumPy (higher is better)')
            plt.title(f'Performance Trend for 2D {pow2_str} Arrays')
            plt.xscale('log', base=2)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend()
            plt.tight_layout()
            
            # Save
            plt.savefig(os.path.join(output_dir, f"size_trend_2D_{pow2_str}_{timestamp}.png"), dpi=150)
            plt.savefig(os.path.join(output_dir, f"size_trend_2D_{pow2_str}_latest.png"), dpi=150)
            plt.close()
    
    # 4. Thread count analysis (if available)
    # Check if we have results for different thread counts
    thread_implementations = [impl for impl in implementations if "threads" in impl.lower()]
    
    if thread_implementations:
        plt.figure(figsize=(10, 6))
        
        # Group by thread count
        thread_counts = []
        thread_speedups = []
        
        for implementation in thread_implementations:
            # Extract thread count from name
            if "threads" in implementation.lower():
                thread_count = int(implementation.lower().split("threads")[0].strip("-"))
                
                if 'mean_speedup' in analysis['implementations'][implementation]:
                    thread_counts.append(thread_count)
                    thread_speedups.append(analysis['implementations'][implementation]['mean_speedup'])
        
        # Sort by thread count
        sorted_indices = np.argsort(thread_counts)
        thread_counts = [thread_counts[i] for i in sorted_indices]
        thread_speedups = [thread_speedups[i] for i in sorted_indices]
        
        # Plot
        plt.plot(thread_counts, thread_speedups, 'o-', color='blue')
        
        # Add horizontal line at 1.0 (NumPy baseline)
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='NumPy baseline')
        
        # Customize
        plt.xlabel('Thread Count')
        plt.ylabel('Mean Speedup vs NumPy (higher is better)')
        plt.title('Performance vs Thread Count')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(thread_counts)
        plt.tight_layout()
        
        # Save
        plt.savefig(os.path.join(output_dir, f"thread_analysis_{timestamp}.png"), dpi=150)
        plt.savefig(os.path.join(output_dir, "thread_analysis_latest.png"), dpi=150)
        plt.close()
    
    # 5. Planning strategy analysis (if available)
    # Check if we have results for different planning strategies
    planner_implementations = [impl for impl in implementations if any(p in impl.lower() for p in ['estimate', 'measure', 'patient'])]
    
    if planner_implementations:
        plt.figure(figsize=(10, 6))
        
        # Group by planning strategy
        planners = []
        planner_names = []
        planner_speedups = []
        
        for implementation in planner_implementations:
            # Extract planner from name
            if "estimate" in implementation.lower():
                planner = 1
                planner_name = "ESTIMATE"
            elif "measure" in implementation.lower():
                planner = 2
                planner_name = "MEASURE"
            elif "patient" in implementation.lower():
                planner = 3
                planner_name = "PATIENT"
            elif "exhaustive" in implementation.lower():
                planner = 4
                planner_name = "EXHAUSTIVE"
            else:
                continue
                
            if 'mean_speedup' in analysis['implementations'][implementation]:
                planners.append(planner)
                planner_names.append(planner_name)
                planner_speedups.append(analysis['implementations'][implementation]['mean_speedup'])
        
        # Sort by planner effort
        sorted_indices = np.argsort(planners)
        planner_names = [planner_names[i] for i in sorted_indices]
        planner_speedups = [planner_speedups[i] for i in sorted_indices]
        
        # Plot
        bars = plt.bar(range(len(planner_names)), planner_speedups, color='green')
        
        # Add horizontal line at 1.0 (NumPy baseline)
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='NumPy baseline')
        
        # Customize
        plt.xlabel('Planning Strategy')
        plt.ylabel('Mean Speedup vs NumPy (higher is better)')
        plt.title('Performance vs Planning Strategy')
        plt.xticks(range(len(planner_names)), planner_names)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Add speedup values on top of bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{planner_speedups[i]:.2f}x', ha='center', va='bottom')
        
        # Save
        plt.savefig(os.path.join(output_dir, f"planner_analysis_{timestamp}.png"), dpi=150)
        plt.savefig(os.path.join(output_dir, "planner_analysis_latest.png"), dpi=150)
        plt.close()
    
    print(f"Plots saved to {output_dir}")


# ============================================================================================
# Main Function and Argument Parsing
# ============================================================================================

def main():
    """Main function to run the benchmarks."""
    parser = argparse.ArgumentParser(description='Comprehensive FFT benchmarking suite')
    parser.add_argument('--quick', action='store_true', help='Run a reduced set of benchmarks (faster)')
    parser.add_argument('--full', action='store_true', help='Run all benchmarks (can take hours)')
    parser.add_argument('--report', metavar='FILENAME', help='Save report to specified file')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output-dir', metavar='DIR', default='benchmark_results', help='Directory to save outputs')
    parser.add_argument('--runtime-limit', type=int, default=20*60, help='Maximum runtime in seconds (default: 20 minutes)')
    
    args = parser.parse_args()
    
    # Set runtime limit
    global RUNTIME_LIMIT
    RUNTIME_LIMIT = args.runtime_limit
    
    # Use full mode unless quick mode is explicitly specified
    quick_mode = not args.full if args.quick or args.full else True
    
    print("Starting FFT benchmark suite...")
    print(f"Mode: {'Quick' if quick_mode else 'Full'}")
    print(f"Runtime limit: {format_duration(RUNTIME_LIMIT)}")
    print(f"Output directory: {args.output_dir}")
    
    # Create benchmark suite
    benchmarks = create_benchmark_suite(quick_mode=quick_mode)
    print(f"Created {len(benchmarks)} benchmarks")
    
    # Create implementations
    implementations = create_implementations(test_betterfftw_configs=not quick_mode)
    print(f"Testing {len(implementations)} implementations")
    
    # Run benchmarks
    results = run_benchmark_suite(
        benchmarks, implementations, 
        quick_mode=quick_mode, 
        output_dir=args.output_dir
    )
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Generate report
    generate_report(analysis, output_dir=args.output_dir)
    
    # Generate plots if requested
    if args.plot:
        generate_plots(analysis, output_dir=args.output_dir)
    
    print("Benchmark completed!")


if __name__ == "__main__":
    main()