"""
Smart planning strategies for optimizing FFTW performance.

this module provides algorithms to determine the optimal FFT planning strategy
based on array characteristics, usage patterns, and system capabilities.
"""

import numpy as np
import time
import os
import platform
import psutil
import logging
from typing import Dict, Tuple, Optional, Union, Any, List

# set up logging
logger = logging.getLogger("betterfftw.planning")

# Constants for planning strategies
PLANNER_ESTIMATE = 'FFTW_ESTIMATE'  # quick planning, good for one-off FFTs
PLANNER_MEASURE = 'FFTW_MEASURE'    # thorough planning, good for repeated FFTs
PLANNER_PATIENT = 'FFTW_PATIENT'    # very thorough planning, for critical performance
PLANNER_EXHAUSTIVE = 'FFTW_EXHAUSTIVE'  # maximum optimization, can be very slow

# Size thresholds based on empirical testing
SMALL_ARRAY_THRESHOLD = 256 * 256    # arrays smaller than this can use ESTIMATE
MEDIUM_ARRAY_THRESHOLD = 1024 * 1024  # medium-sized arrays
LARGE_ARRAY_THRESHOLD = 2048 * 2048   # large arrays might benefit from PATIENT

# Usage pattern thresholds
LOW_USAGE_THRESHOLD = 2     # used just a few times
MEDIUM_USAGE_THRESHOLD = 10  # used several times
HIGH_USAGE_THRESHOLD = 50    # used many times

# Memory thresholds (in GB)
LOW_MEMORY_THRESHOLD = 4
HIGH_MEMORY_THRESHOLD = 16

# Cache for system info to avoid repeated calls
_system_info_cache = {}
_array_stats_cache = {}


def get_system_info() -> Dict[str, Any]:
    """
    Get information about the system that's relevant for planning.
    
    Returns:
        Dict with system information like CPU count, memory, etc.
    """
    # use cached info if available
    if _system_info_cache:
        return _system_info_cache
    
    info = {
        'cpu_count': os.cpu_count() or 1,
        'physical_cores': psutil.cpu_count(logical=False) or 1,
        'total_memory_gb': psutil.virtual_memory().total / (1024**3),
        'available_memory_gb': psutil.virtual_memory().available / (1024**3),
        'platform': platform.system(),
        'processor': platform.processor(),
        'python_bits': 64 if platform.architecture()[0] == '64bit' else 32,
    }
    
    # cache for future use - don't need global declaration here
    _system_info_cache.update(info)
    
    return info


def clear_cache():
    """Clear cached information to force re-calculation."""
    _system_info_cache.clear()
    _array_stats_cache.clear()


def analyze_array(array: np.ndarray) -> Dict[str, Any]:
    """
    Analyze an array to determine its characteristics for planning.
    
    Args:
        array: NumPy array to analyze
        
    Returns:
        Dict with array characteristics
    """
    # use array shape as cache key
    cache_key = (array.shape, array.dtype)
    if cache_key in _array_stats_cache:
        return _array_stats_cache[cache_key]
    
    # calculate basic stats
    size = np.prod(array.shape)
    memory_usage = array.nbytes / (1024**2)  # in MB
    
    # check if dimensions are powers of 2 (good for FFT)
    is_power_of_two = all((d & (d-1) == 0) and d != 0 for d in array.shape)
    
    # check if dimensions have small prime factors (also good)
    has_small_primes = True
    for d in array.shape:
        # factorize the dimension
        n = d
        largest_factor = 1
        for i in range(2, int(np.sqrt(n)) + 1):
            while n % i == 0:
                largest_factor = i
                n //= i
        if n > 1:
            largest_factor = n
        
        # if largest prime factor is > 7, it's not "small primes"
        if largest_factor > 7:
            has_small_primes = False
    
    stats = {
        'size': size,
        'memory_mb': memory_usage,
        'is_power_of_two': is_power_of_two,
        'has_small_primes': has_small_primes,
        'dimensionality': len(array.shape),
        'is_complex': np.issubdtype(array.dtype, np.complexfloating),
        'largest_dim': max(array.shape),
    }
    
    # cache for future use with the same shape
    _array_stats_cache[cache_key] = stats
    
    return stats
def get_optimal_planner(array: np.ndarray, 
                       usage_count: int = 1, 
                       time_critical: bool = False,
                       prefer_speed: bool = True) -> str:
    """
    Determine the optimal planning strategy based on array and usage pattern.
    
    Args:
        array: The array that will be transformed
        usage_count: How many times this transform will be used
        time_critical: Whether this is in a time-sensitive context
        prefer_speed: Whether to prioritize speed over memory usage
        
    Returns:
        The recommended FFTW planner strategy
    """
    # Get array and system info
    array_info = analyze_array(array)
    size = array_info['size']
    is_power_of_two = array_info['is_power_of_two']
    dimensions = len(array.shape)
    
    # For time-critical applications, always use ESTIMATE
    if time_critical:
        return PLANNER_ESTIMATE
    
    # For non-power-of-2 sizes that will be used repeatedly,
    # MEASURE often performs better according to benchmarks
    if not is_power_of_two and usage_count >= 3 and size >= 8192:
        return PLANNER_MEASURE
    
    # For multi-dimensional arrays used repeatedly, consider MEASURE
    if dimensions >= 2 and usage_count >= 5 and size >= 32768:
        return PLANNER_MEASURE
    
    # For extremely frequent usage of large arrays, MEASURE pays off
    if usage_count >= 10 and size >= 65536:
        return PLANNER_MEASURE
    
    # Default to ESTIMATE for everything else
    return PLANNER_ESTIMATE


def get_optimal_threads(array: np.ndarray) -> int:
    """
    Determine the optimal number of threads for an FFT operation.
    
    Args:
        array: The array that will be transformed
        
    Returns:
        The recommended thread count
    """
    sys_info = get_system_info()
    array_info = analyze_array(array)
    
    # single-thread for small arrays - threading overhead isn't worth it
    if array_info['size'] < SMALL_ARRAY_THRESHOLD:
        return 1
    
    # for medium arrays, use a fraction of available cores
    if array_info['size'] < LARGE_ARRAY_THRESHOLD:
        # use physical cores for medium arrays
        return max(1, sys_info['physical_cores'] // 2)
    
    # for large arrays, use all physical cores
    return sys_info['physical_cores']


def benchmark_planners(array: np.ndarray, 
                      repeats: int = 5,
                      strategies: List[str] = None) -> Dict[str, float]:
    """
    Benchmark different planning strategies on an array to find the fastest.
    
    Args:
        array: Array to benchmark with
        repeats: Number of FFT operations to perform for each strategy
        strategies: Planner strategies to test (defaults to all)
        
    Returns:
        Dict mapping strategies to their execution times
    """
    if strategies is None:
        strategies = [PLANNER_ESTIMATE, PLANNER_MEASURE, PLANNER_PATIENT]
    
    from .core import SmartFFTW
    
    results = {}
    
    for strategy in strategies:
        # clear the cache to ensure fair testing
        SmartFFTW.clear_cache()
        
        # measure the planning time
        start_time = time.time()
        
        # create a new plan with this strategy
        plan = SmartFFTW._create_plan(
            array=array.copy(), 
            builder_func=getattr(SmartFFTW, '_create_plan'),
            planner=strategy
        )
        
        planning_time = time.time() - start_time
        
        # measure execution time (average of repeats)
        execution_times = []
        for _ in range(repeats):
            array_copy = array.copy()
            start_time = time.time()
            result = plan(array_copy)
            execution_times.append(time.time() - start_time)
        
        avg_execution = sum(execution_times) / len(execution_times)
        
        # calculate total time for all repeats
        total_time = planning_time + (avg_execution * repeats)
        
        results[strategy] = {
            'planning_time': planning_time,
            'avg_execution': avg_execution,
            'total_time': total_time,
        }
    
    return results

def optimal_transform_size(target_size: int, 
                         fast_sizes: bool = True,
                         max_increase: float = 0.2) -> int:
    """
    Find the optimal transform size near a target size.
    
    FFTW performs better with certain sizes, especially powers of 2
    or sizes with small prime factors. This function finds a nearby
    size that would perform better.
    
    Args:
        target_size: Target size for the transform
        fast_sizes: Whether to prefer fast sizes (powers of 2, etc.)
        max_increase: Maximum allowed size increase as a fraction
        
    Returns:
        Optimal size for the transform
    """
    if not fast_sizes:
        return target_size
    
    # check if target_size is already a power of 2
    if target_size & (target_size - 1) == 0:
        return target_size
    
    # find next power of 2
    next_pow2 = 1
    while next_pow2 < target_size:
        next_pow2 *= 2
    
    # find previous power of 2
    prev_pow2 = next_pow2 // 2
    
    # find next size with small prime factors (2,3,5,7)
    def has_only_small_primes(n):
        # check if n has only 2,3,5,7 as prime factors
        for p in [2, 3, 5, 7]:
            while n % p == 0:
                n //= p
        return n == 1
    
    # try to find a good size with small primes near the target
    good_size = target_size
    max_size = int(target_size * (1 + max_increase))
    
    for size in range(target_size, max_size + 1):
        if has_only_small_primes(size):
            good_size = size
            break
    
    # choose the best size
    candidates = [
        (target_size, 0),  # original size
        (prev_pow2, (target_size - prev_pow2) / target_size),  # previous power of 2
        (next_pow2, (next_pow2 - target_size) / target_size),  # next power of 2
        (good_size, (good_size - target_size) / target_size),  # size with small primes
    ]
    
    # filter out candidates that exceed max_increase OR are smaller than target_size
    valid_candidates = [(size, diff) for size, diff in candidates 
                      if diff <= max_increase and size >= target_size]
    
    if not valid_candidates:
        return target_size
    
    # prefer powers of 2, then small primes
    if (next_pow2, (next_pow2 - target_size) / target_size) in valid_candidates:
        return next_pow2
    if (good_size, (good_size - target_size) / target_size) in valid_candidates:
        return good_size
    
    # fallback to original size
    return target_size
def optimize_transform_shape(shape: Tuple[int, ...], 
                           max_increase: float = 0.2) -> Tuple[int, ...]:
    """
    Find the optimal shape for a multidimensional transform.
    
    Args:
        shape: Original shape tuple
        max_increase: Maximum allowed size increase per dimension
        
    Returns:
        Optimized shape tuple
    """
    return tuple(optimal_transform_size(dim, max_increase=max_increase) 
               for dim in shape)


def get_memory_limit() -> int:
    """
    Determine a safe memory limit for planning in bytes.
    
    Returns:
        Memory limit in bytes
    """
    sys_info = get_system_info()
    available_gb = sys_info['available_memory_gb']
    
    # limit to a fraction of available memory to avoid swapping
    if available_gb < LOW_MEMORY_THRESHOLD:
        # on low-memory systems, use at most 25% of available memory
        return int(available_gb * 0.25 * 1024**3)
    elif available_gb < HIGH_MEMORY_THRESHOLD:
        # on medium-memory systems, use at most 50% of available memory
        return int(available_gb * 0.5 * 1024**3)
    else:
        # on high-memory systems, use at most 75% of available memory
        return int(available_gb * 0.75 * 1024**3)


def is_planning_feasible(array: np.ndarray, planner: str) -> bool:
    """
    Check if using the given planner is feasible for this array.
    
    Some planning strategies use a lot of memory for large arrays,
    so this function checks if it's safe to use the strategy.
    
    Args:
        array: Array to be transformed
        planner: Planning strategy
        
    Returns:
        True if planning is feasible, False otherwise
    """
    # ESTIMATE is always feasible
    if planner == PLANNER_ESTIMATE:
        return True
    
    array_info = analyze_array(array)
    memory_limit = get_memory_limit()
    
    # rough estimation of planning memory requirements
    planning_factor = {
        PLANNER_MEASURE: 2,      # typically needs ~2x array size
        PLANNER_PATIENT: 4,      # can use up to ~4x array size
        PLANNER_EXHAUSTIVE: 8,   # can use up to ~8x array size
    }.get(planner, 1)
    
    estimated_memory = array_info['memory_mb'] * planning_factor * 1024**2  # convert MB to bytes
    
    # check if estimated memory is within limit
    return estimated_memory <= memory_limit


def adapt_planner_to_resources(array: np.ndarray, 
                             desired_planner: str) -> str:
    """
    Adapt the planner to available system resources.
    
    If the desired planner would use too much memory, fall back to a simpler one.
    
    Args:
        array: Array to be transformed
        desired_planner: Desired planning strategy
        
    Returns:
        Feasible planning strategy
    """
    if is_planning_feasible(array, desired_planner):
        return desired_planner
    
    # fall back to simpler planners
    fallback_order = [PLANNER_PATIENT, PLANNER_MEASURE, PLANNER_ESTIMATE]
    for planner in fallback_order:
        if planner == desired_planner:
            continue  # skip the desired planner as we already checked it
        if is_planning_feasible(array, planner):
            logger.warning(f"Falling back to {planner} from {desired_planner} due to resource constraints")
            return planner
    
    # ultimate fallback
    logger.warning(f"All planners exceed resource limits, using {PLANNER_ESTIMATE}")
    return PLANNER_ESTIMATE