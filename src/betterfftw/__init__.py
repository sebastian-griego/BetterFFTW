"""
BetterFFTW: A high-performance, user-friendly wrapper for FFTW in Python.

this package provides drop-in replacements for NumPy and SciPy FFT functions
with transparent acceleration using FFTW. it handles all the complexity of
plan caching, wisdom management, and thread optimization automatically.

Basic usage:
    import betterfftw

    # Make all NumPy and SciPy FFT calls use BetterFFTW
    betterfftw.use_as_default()

    # Now standard NumPy/SciPy code will use FFTW under the hood
    import numpy as np
    x = np.random.random(1024)
    y = np.fft.fft(x)  # Uses BetterFFTW automatically!

Advanced usage:
    # Direct access to core functionality with explicit parameters
    import betterfftw as bfft
    
    # Perform FFT with specific parameters
    y = bfft.fft(x, threads=4, planner='FFTW_PATIENT')
    
    # Create aligned arrays for optimal performance
    aligned_array = bfft.empty_aligned((1024, 1024), dtype=np.complex128)
"""

__version__ = '0.1.0'
# Import core functionality
from .core import (
    # Main FFT functions
    fft, ifft, rfft, irfft,
    fft2, ifft2, rfft2, irfft2,
    fftn, ifftn, rfftn, irfftn,
    
    # Utility functions
    empty_aligned, empty_aligned_like, byte_align,
    import_wisdom, export_wisdom,
    
    # Configuration functions
    set_num_threads, set_planner_effort,
    get_stats, clear_cache,
    set_threading_thresholds, set_threading_limits,
    set_max_cache_size,
    
    # Core class for advanced users
    SmartFFTW
)

# Import interface functionality for NumPy/SciPy compatibility
from .interface import (
    # Registration functions
    use_as_default_fft, restore_default_fft,
    register_numpy_fft, unregister_numpy_fft,
    register_scipy_fft, unregister_scipy_fft,
    
    # Utility functions that were missing
    fftfreq, rfftfreq, fftshift, ifftshift,
    hfft, ihfft
)

# Import planning module for advanced users
from .planning import (
    # Planning optimization functions
    get_optimal_planner, get_optimal_threads,
    optimize_transform_shape, optimal_transform_size,
    benchmark_planners
)

# Define package-level constants (planning efforts)
PLANNER_ESTIMATE = 'FFTW_ESTIMATE'
PLANNER_MEASURE = 'FFTW_MEASURE'
PLANNER_PATIENT = 'FFTW_PATIENT'
PLANNER_EXHAUSTIVE = 'FFTW_EXHAUSTIVE'
# Main convenience function to make BetterFFTW the default FFT implementation
def use_as_default(register_scipy=True):
    """
    Make BetterFFTW the default FFT implementation for both NumPy and SciPy.
    
    Based on extensive benchmarking, this implementation provides 2-3x speedup
    over NumPy FFT across a range of transform sizes and types. The library 
    intelligently selects between planning strategies (ESTIMATE for speed, 
    MEASURE for efficiency) and thread counts based on your specific workload.
    
    Key features:
    - For non-power-of-2 sizes, the library automatically switches to optimized
      planning after just a few repeats to ensure consistent speedups
    - Multi-threading is applied intelligently: small FFTs use a single thread to 
      avoid overhead, large multi-dimensional FFTs leverage multiple cores
    - Adaptive fallback: In rare cases where NumPy might outperform FFTW for a 
      specific size, the library can detect this and automatically use NumPy
    
    Args:
        register_scipy: Whether to also register for SciPy's FFT functions.
                       Set to False to avoid SciPy-related warnings if you're
                       not using SciPy's FFT functions.
    """
    return use_as_default_fft(register_scipy)

def restore_default(unregister_scipy=True):
    """
    Restore the original NumPy and SciPy FFT implementations.
    
    Args:
        unregister_scipy: Whether to also unregister from SciPy's FFT functions.
    """
    return restore_default_fft(unregister_scipy)

# Try to import wisdom at package initialization time
try:
    import_wisdom()
except Exception:
    pass  # Silently continue if wisdom import fails

# Default configuration - load wisdom, but don't replace NumPy/SciPy FFT yet
# This makes the package ready to use but non-invasive by default