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

import os
import logging


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

# Configuration system
_config = {
    # Default configuration
    'cache': {
        'max_size': 1000,
        'timeout': 300,  # seconds
        'cleaning_interval': 900,  # seconds
    },
    'planning': {
        'default_strategy': 'FFTW_ESTIMATE',
        'auto_upgrade': True,
        'min_repeat_for_upgrade': 5,
    },
    'threading': {
        'default_threads': min(os.cpu_count(), 4),
        'small_threshold': 16384,
        'medium_threshold': 65536,
        'large_threshold': 262144,
        'small_max_threads': 1,
        'medium_max_threads': 2,
        'multi_dim_max_threads': 4,
    },
    'fallback': {
        'use_numpy_for_non_power_of_two': True,
        'allow_runtime_selection': True,
    },
    'logging': {
        'level': 'WARNING',
    }
}

def _update_nested_dict(d, u):
    """Update nested dictionary recursively."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _update_nested_dict(d[k], v)
        else:
            d[k] = v

def configure(config_dict=None, **kwargs):
    """
    Configure BetterFFTW global settings.
    
    Args:
        config_dict: Dictionary with configuration settings
        **kwargs: Configuration settings as keyword arguments
    
    Examples:
        # Configure with a dictionary
        betterfftw.configure({
            'cache': {'max_size': 2000},
            'threading': {'default_threads': 8}
        })
        
        # Or with keyword arguments
        betterfftw.configure(
            planning_default_strategy='FFTW_MEASURE',
            threading_default_threads=8
        )
    """
    global _config
    
    if config_dict:
        # Update nested dictionary recursively
        _update_nested_dict(_config, config_dict)
    
    # Process kwargs (flattened config)
    for key, value in kwargs.items():
        if '_' in key:
            # Handle nested keys like 'cache_max_size'
            parts = key.split('_', 1)
            if parts[0] in _config:
                if parts[1] in _config[parts[0]]:
                    _config[parts[0]][parts[1]] = value
        else:
            # Handle top-level keys
            if key in _config:
                _config[key] = value
    
    # Apply configuration
    _apply_configuration()
    
    return _config.copy()  # Return a copy of the current config

def _load_env_config():
    """Load configuration from environment variables."""
    import os
    
    # Environment variable prefix
    prefix = "BETTERFFTW_"
    
    # Find all relevant environment variables
    for key, value in os.environ.items():
        if key.startswith(prefix):
            # Remove prefix and convert to lowercase
            config_key = key[len(prefix):].lower()
            
            # Try to convert value to appropriate type
            if value.isdigit():
                value = int(value)
            elif value.lower() in ('true', 'yes', 'on'):
                value = True
            elif value.lower() in ('false', 'no', 'off'):
                value = False
            elif value.replace('.', '', 1).isdigit():
                value = float(value)
            
            # Apply setting using configure
            configure(**{config_key: value})

# Load environment config at startup
_load_env_config()




def _apply_configuration():
    """Apply configuration settings to module components."""
    # Apply to core module
    core.MAX_CACHE_SIZE = _config['cache']['max_size']
    core.DEFAULT_CACHE_TIMEOUT = _config['cache']['timeout']
    core._cache_cleaning_interval = _config['cache']['cleaning_interval']
    
    core.DEFAULT_PLANNER = _config['planning']['default_strategy']
    core.MIN_REPEAT_FOR_MEASURE = _config['planning']['min_repeat_for_upgrade']
    
    core.DEFAULT_THREADS = _config['threading']['default_threads']
    core.THREADING_SMALL_THRESHOLD = _config['threading']['small_threshold']
    core.THREADING_MEDIUM_THRESHOLD = _config['threading']['medium_threshold']
    core.THREADING_LARGE_THRESHOLD = _config['threading']['large_threshold']
    core.THREADING_MAX_SMALL = _config['threading']['small_max_threads']
    core.THREADING_MAX_MEDIUM = _config['threading']['medium_max_threads']
    core.THREADING_MAX_MULTI_DIM = _config['threading']['multi_dim_max_threads']
    
    core.USE_NUMPY_FOR_NON_POWER_OF_TWO = _config['fallback']['use_numpy_for_non_power_of_two']
    
    # Configure logging
    logging.getLogger("betterfftw").setLevel(getattr(logging, _config['logging']['level']))

# Initialize logging
def _setup_logging():
    """Set up default logging configuration."""
    logger = logging.getLogger("betterfftw")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, _config['logging']['level']))
        # Don't propagate to root logger
        logger.propagate = False

_setup_logging()

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

# Expose the configure function at package level
from . import configure

# Default configuration - load wisdom, but don't replace NumPy/SciPy FFT yet
# This makes the package ready to use but non-invasive by default