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
    
    # Core class for advanced users
    SmartFFTW
)

# Import interface functionality for NumPy/SciPy compatibility
from .interface import (
    # Registration functions
    use_as_default_fft, restore_default_fft,
    register_numpy_fft, unregister_numpy_fft,
    register_scipy_fft, unregister_scipy_fft
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
def use_as_default():
    """
    Make BetterFFTW the default FFT implementation for both NumPy and SciPy.
    
    After calling this function, any code that uses numpy.fft or scipy.fft
    will automatically use BetterFFTW's accelerated implementation instead.
    
    This is the recommended way to use BetterFFTW for most users.
    
    Example:
        import betterfftw
        betterfftw.use_as_default()
        
        # Now all FFT calls use BetterFFTW automatically
        import numpy as np
        x = np.random.random(1024)
        y = np.fft.fft(x)  # Uses BetterFFTW!
        
    Returns:
        bool: True if registration was successful, False otherwise
    """
    return use_as_default_fft()

def restore_default():
    """
    Restore the original NumPy and SciPy FFT implementations.
    
    This undoes the changes made by use_as_default().
    
    Example:
        import betterfftw
        betterfftw.use_as_default()
        
        # Do some FFT operations with BetterFFTW
        
        # Switch back to original implementations
        betterfftw.restore_default()
        
    Returns:
        bool: True if restoration was successful, False otherwise
    """
    return restore_default_fft()

# Try to import wisdom at package initialization time
try:
    import_wisdom()
except Exception:
    pass  # Silently continue if wisdom import fails

# Default configuration - load wisdom, but don't replace NumPy/SciPy FFT yet
# This makes the package ready to use but non-invasive by default