"""
Core FFTW wrapper functionality with smart defaults and optimizations.

this module provides the core functionality for the BetterFFTW package,
including plan caching, thread management, and auto-optimization.
"""

import os
import time
import multiprocessing
import threading
import numpy as np
import concurrent.futures
import pyfftw
import atexit
import logging
from typing import Dict, Tuple, Optional, Union, Any, List, Callable
# enable the PyFFTW cache
pyfftw.interfaces.cache.enable()

logger = logging.getLogger("betterfftw.core")

# configuration constants with smart defaults
DEFAULT_THREADS = min(multiprocessing.cpu_count(), 4)  # reasonable default
DEFAULT_PLANNER = 'FFTW_ESTIMATE'  # Fastest in latest benchmark
MEASURE_PLANNER = 'FFTW_MEASURE'  # thorough planning for repeated use
PATIENCE_PLANNER = 'FFTW_PATIENT'  # thorough planning for critical performance
DEFAULT_CACHE_TIMEOUT = 300  # seconds to keep plans in cache (from 60)

# thresholds for auto-configuration
SIZE_THREADING_THRESHOLD = 1024 * 1024  # When to use multi-threading - increased from 32768
MIN_REPEAT_FOR_MEASURE = 5  # trigger MEASURE planning after this many repeats
AUTO_ALIGN = True  # auto-align arrays for FFTW
USE_NUMPY_FOR_NON_POWER_OF_TWO = True  # Set True to use NumPy for non-power-of-2 sizes (typically 2x faster)

# threading thresholds for different array sizes
THREADING_SMALL_THRESHOLD = 16384    # 16K elements
THREADING_MEDIUM_THRESHOLD = 65536   # 64K elements
THREADING_LARGE_THRESHOLD = 262144   # 256K elements
THREADING_MAX_SMALL = 1              # Max threads for small arrays
THREADING_MAX_MEDIUM = 2             # Max threads for medium arrays
THREADING_MAX_MULTI_DIM = 4          # Max threads for multi-dimensional arrays
MAX_CACHE_SIZE = 1000                # Maximum number of plans to keep in cache

# we'll use a timer to periodically clean the cache
_cache_cleaning_interval = 900  # 15 minutes (from 5 minutes)
_cache_lock = threading.RLock()  # prevent race conditions
_optimization_queue = []  # queue for background optimizations
_optimization_lock = threading.RLock()  # lock for the optimization queue
_background_optimizing = False  # flag for background optimization

# Add to the top of the file:
_optimization_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=2,  # Start with a small number
    thread_name_prefix="betterfftw_opt"
)
_optimization_futures = {}  # Track running optimizations


# wisdom file for persistent planning
WISDOM_FILE = os.path.expanduser("~/.betterfftw_wisdom")

def _exit_handler():
    """Save wisdom and clean up resources when exiting."""
    try:
        SmartFFTW.export_wisdom(WISDOM_FILE)
        _optimization_executor.shutdown(wait=False)
    except Exception as e:
        logger.warning(f"Exit handler error: {str(e)}")


# register the exit handler
atexit.register(_exit_handler)

class SmartFFTW:
    """
    Smart FFTW wrapper with automatic optimization.
    
    handles plan caching, thread selection, and progressive optimization
    to make FFTW both convenient and high-performance.
    """
    # static caches
    _plan_cache: Dict[Tuple, Any] = {}  # cache for FFTW plan objects
    _call_count: Dict[Tuple, int] = {}  # track call frequency per shape
    _last_used: Dict[Tuple, float] = {}  # track when plans were last used
    _plan_quality: Dict[Tuple, str] = {}  # track current plan quality
    _numpy_fallback_keys: set = set()  # keys where NumPy outperforms FFTW
    _numpy_fallback_counts: Dict[Tuple, int] = {}  # track consistent performance differences
    
    def __init__(self):
        """Initialize SmartFFTW instance."""
        # optional instance-specific settings can go here
        pass
    @classmethod
    def clear_cache(cls, older_than: Optional[float] = None):
        """
        Clear cached plans to free memory.
        
        Args:
            older_than: Clear plans unused for this many seconds (None = clear all)
        """
        with _cache_lock:
            # If clearing everything, also clear fallback caches
            if older_than is None:
                if hasattr(cls, '_numpy_fallback_keys'):
                    cls._numpy_fallback_keys.clear()
                if hasattr(cls, '_numpy_fallback_counts'):
                    cls._numpy_fallback_counts.clear()
                
            # Check if cache size exceeds maximum
            if len(cls._plan_cache) > MAX_CACHE_SIZE:
                # Sort by combination of usage count and recency
                keys_by_importance = []
                now = time.time()
                
                for key in cls._plan_cache.keys():
                    count = cls._call_count.get(key, 0)
                    last_used = cls._last_used.get(key, 0)
                    age = now - last_used
                    
                    # Higher count and lower age = more important
                    importance = count / (age + 1)  # Add 1 to avoid division by zero
                    keys_by_importance.append((key, importance))
                
                # Sort by importance (higher value = more important)
                keys_by_importance.sort(key=lambda x: x[1], reverse=True)
                
                # Keep the most important half
                keys_to_keep = [k for k, _ in keys_by_importance[:MAX_CACHE_SIZE//2]]
                
                # Remove keys not in the keep list
                for cache_dict in [cls._plan_cache, cls._call_count, cls._plan_quality, cls._last_used]:
                    keys_to_remove = [k for k in cache_dict.keys() if k not in keys_to_keep]
                    for key in keys_to_remove:
                        del cache_dict[key]
                
                return
            
            # Standard age-based cleanup
            if older_than is None:
                # Clear everything
                cls._plan_cache.clear()
                cls._call_count.clear()
                cls._last_used.clear()
                cls._plan_quality.clear()
                return
            
            # Selectively clear based on age and usage
            now = time.time()
            keys_to_remove = []
            
            for key, last_used in cls._last_used.items():
                age = now - last_used
                
                if age > older_than:
                    count = cls._call_count.get(key, 0)
                    quality = cls._plan_quality.get(key, DEFAULT_PLANNER)
                    
                    # More frequently used plans get longer retention
                    if count > MIN_REPEAT_FOR_MEASURE and quality != DEFAULT_PLANNER:
                        # High-quality plans for frequently used transforms get the longest life
                        if age > older_than * 3:
                            keys_to_remove.append(key)
                    elif count > MIN_REPEAT_FOR_MEASURE:
                        # Frequently used plans get longer life
                        if age > older_than * 2:
                            keys_to_remove.append(key)
                    else:
                        # Standard expiration
                        keys_to_remove.append(key)
            
            # Remove identified keys
            for key in keys_to_remove:
                if key in cls._plan_cache:
                    del cls._plan_cache[key]
                if key in cls._call_count:
                    del cls._call_count[key]
                if key in cls._last_used:
                    del cls._last_used[key]
                if key in cls._plan_quality:
                    del cls._plan_quality[key]
    @classmethod
    def _get_cache_key(cls, array: np.ndarray, n: Optional[Union[int, Tuple[int, ...]]] = None, 
                     axis: Union[int, Tuple[int, ...]] = -1, 
                     norm: Optional[str] = None,
                     transform_type: str = 'fft') -> Tuple:
        """Generate a unique key for caching plans based on transform parameters."""
        shape = array.shape
        dtype = array.dtype
        # include transform type in key to differentiate between fft, ifft, rfft, etc.
        return (transform_type, shape, dtype, n, axis, norm)
    
    @classmethod
    def _is_non_power_of_two(cls, dimensions):
        """Check if any dimension is not a power of 2."""
        if not dimensions:
            return False
        return any((dim & (dim - 1)) != 0 for dim in dimensions if dim > 0)
    
    @classmethod
    def _select_threads(cls, array: np.ndarray) -> int:
        """
        Auto-select optimal thread count based on array size and dimensionality.
        
        Uses more threads for large multi-dimensional arrays while staying
        conservative for smaller transforms where threading overhead isn't worth it.
        """
        size = np.prod(array.shape)
        dims = array.ndim
        
        # For small arrays, use a single thread.
        if size < 262144:  # 256K elements
            return 1
        
        # For large 2D or 3D arrays, use up to 4 threads (but do not exceed DEFAULT_THREADS).
        if dims >= 2 and size >= 1048576:  # 1M elements or more
            return min(DEFAULT_THREADS, 4)
        
        # For very large 1D arrays, consider using 2 threads.
        if dims == 1 and size >= 2097152:  # 2M elements or more
            return min(DEFAULT_THREADS, 2)
        
        # Otherwise, default to a single thread.
        return 1
    @classmethod
    def _should_upgrade_plan(cls, key: Tuple) -> bool:
        """
        Determine if we should upgrade to a more thorough planning strategy.
        
        For non-power-of-two sizes, we upgrade after fewer repeats since these
        benefit more from MEASURE planning than power-of-2 sizes.
        """
        # Get usage info
        count = cls._call_count.get(key, 0)
        current_quality = cls._plan_quality.get(key, DEFAULT_PLANNER)
        
        # If already upgraded, don't upgrade again
        if current_quality != DEFAULT_PLANNER:
            return False

        # Unpack key information
        transform_type, shape, dtype, n, axis, norm = key
        # Ensure dimensions is a tuple
        dimensions = shape if isinstance(shape, tuple) else (shape,)
        size = np.prod(dimensions)
        
        # Check if any dimension is non-power-of-two.
        non_power_of_two = any((d & (d - 1)) != 0 for d in dimensions if d > 0)
        
        # For non-power-of-two sizes, upgrade after MIN_REPEAT_FOR_MEASURE calls.
        if non_power_of_two and count >= MIN_REPEAT_FOR_MEASURE:
            return True

        # For very large transforms (mostly power-of-2) upgrade later.
        if size > 65536 and count >= MIN_REPEAT_FOR_MEASURE * 2:
            return True

        return False
    @classmethod
    def _create_plan(cls, array: np.ndarray, builder_func: Callable, 
                    n: Optional[Union[int, Tuple[int, ...]]] = None, 
                    axis: Union[int, Tuple[int, ...]] = -1, 
                    norm: Optional[str] = None, 
                    threads: Optional[int] = None, 
                    planner: Optional[str] = None,
                    **kwargs) -> Any:
        """Create a new FFTW plan with specified parameters."""
        if threads is None:
            threads = cls._select_threads(array)
            
        if planner is None:
            planner = DEFAULT_PLANNER
        
        # Check the function name to determine correct parameter names and arguments
        func_name = builder_func.__name__ if hasattr(builder_func, '__name__') else str(builder_func)
        
        # Determine if this is an inverse real transform (irfft, irfft2, irfftn)
        is_inv_real = func_name in ['irfft', 'irfft2', 'irfftn']
        
        # Common keyword arguments - some functions may not support all of these
        common_kwargs = {
            'threads': threads,
            'planner_effort': planner,
            'auto_align_input': AUTO_ALIGN,
            'auto_contiguous': True,
        }
        
        # Only add overwrite_input for functions that support it
        # irfft-family functions don't support overwrite_input in PyFFTW
        if not is_inv_real:
            common_kwargs['overwrite_input'] = False
        
        # Combine with any additional keyword arguments
        all_kwargs = {**common_kwargs, **kwargs}
        
        # Different parameter names for different FFT types
        if 'fft2' in func_name or 'fftn' in func_name:
            # 2D and ND transforms use 's' instead of 'n' and 'axes' instead of 'axis'
            fft_obj = builder_func(
                array,
                s=n,  # 's' is used instead of 'n'
                axes=axis,  # 'axes' is used instead of 'axis'
                **all_kwargs
            )
        else:
            # Regular 1D transforms
            fft_obj = builder_func(
                array,
                n=n,
                axis=axis,
                **all_kwargs
            )
        
        return fft_obj
    
    @classmethod
    def _schedule_optimization(cls, key: Tuple, array: np.ndarray, builder_func: Callable,
                             n: Optional[Union[int, Tuple[int, ...]]] = None,
                             axis: Union[int, Tuple[int, ...]] = -1,
                             norm: Optional[str] = None,
                             **kwargs):
        """Schedule a plan for optimization in the background using thread pool."""
        with _optimization_lock:
            # Don't reoptimize if already in progress or completed
            if key in _optimization_futures:
                future = _optimization_futures[key]
                if not future.done():
                    return  # Already being optimized
            
            # Make a copy of the array to avoid holding references
            array_copy = np.array(array, copy=True)
            
            # Submit optimization task to thread pool
            future = _optimization_executor.submit(
                cls._optimize_plan, 
                key, array_copy, builder_func, n, axis, norm, kwargs
            )
            _optimization_futures[key] = future
            
            # Optional: Add a callback to handle completion
            future.add_done_callback(lambda f: cls._optimization_completed(key, f))

    @classmethod
    def _optimization_completed(cls, key, future):
        """Handle optimization completion."""
        try:
            # Check if optimization succeeded
            if future.exception() is None:
                logger.debug(f"Plan optimization completed successfully for {key[0]}")
            else:
                logger.warning(f"Plan optimization failed for {key[0]}: {future.exception()}")
        except concurrent.futures.CancelledError:
            logger.debug(f"Plan optimization cancelled for {key[0]}")

    @classmethod
    def get_memory_usage(cls):
        """Get current memory usage of the plan cache."""
        with _cache_lock:
            # Estimate memory usage of cached plans
            total_bytes = 0
            plan_count = len(cls._plan_cache)
            
            # Sample a few plans to estimate average size
            sample_size = min(10, plan_count)
            if sample_size > 0:
                sampled_keys = list(cls._plan_cache.keys())[:sample_size]
                for key in sampled_keys:
                    plan = cls._plan_cache[key]
                    # PyFFTW plan objects have input and output arrays
                    if hasattr(plan, 'input_array') and hasattr(plan, 'output_array'):
                        total_bytes += plan.input_array.nbytes
                        total_bytes += plan.output_array.nbytes
                
                # Extrapolate to estimate total
                avg_bytes_per_plan = total_bytes / sample_size
                estimated_total = avg_bytes_per_plan * plan_count
            else:
                estimated_total = 0
            
            return {
                'plan_count': plan_count,
                'estimated_bytes': estimated_total,
                'estimated_mb': estimated_total / (1024 * 1024)
            }

    # Initialize performance metrics
    _performance_metrics = {
        'calls': 0,
        'cache_hits': 0, 
        'cache_misses': 0,
        'plan_upgrades': 0,
        'numpy_fallbacks': 0,
        'execution_time': 0.0,
        'planning_time': 0.0,
    }

    @classmethod
    def reset_metrics(cls):
        """Reset performance metrics."""
        for key in cls._performance_metrics:
            cls._performance_metrics[key] = 0 if isinstance(cls._performance_metrics[key], int) else 0.0

    @classmethod
    def get_metrics(cls):
        """Get performance metrics."""
        with _cache_lock:
            metrics = cls._performance_metrics.copy()
            
            # Add derived metrics
            if metrics['calls'] > 0:
                metrics['cache_hit_rate'] = metrics['cache_hits'] / metrics['calls']
                metrics['avg_execution_time'] = metrics['execution_time'] / metrics['calls']
            
            # Add cache stats
            metrics['cache_size'] = len(cls._plan_cache)
            metrics['unique_shapes'] = len(set(key[1] for key in cls._plan_cache.keys()))
            
            return metrics

    @classmethod
    def _optimize_plan(cls, key, array, builder_func, n, axis, norm, kwargs):
        """Perform actual plan optimization."""
        # Check if this plan is still in use
        with _cache_lock:
            if key not in cls._last_used:
                logger.debug(f"Skipping optimization for unused plan {key[0]}")
                return  # Plan was removed, skip it
            
            # Create optimized plan
            try:
                optimized_plan = cls._create_plan(
                    array, builder_func, n, axis, norm,
                    threads=cls._select_threads(array),
                    planner=MEASURE_PLANNER,
                    **kwargs
                )
                # Update cache with optimized plan
                cls._plan_cache[key] = optimized_plan
                cls._plan_quality[key] = MEASURE_PLANNER
                logger.debug(f"Optimized plan for {key[0]} with shape {array.shape}")
            except Exception as e:
                logger.warning(f"Plan optimization failed: {str(e)}")
                raise  # Re-raise for the future to handle

    @classmethod
    def _background_optimize(cls):
        """Background thread to optimize plans."""
        global _background_optimizing
        try:
            while True:
                # get next item to optimize
                with _optimization_lock:
                    if not _optimization_queue:
                        _background_optimizing = False
                        return  # exit if no more work
                    
                    key, array, builder_func, n, axis, norm, kwargs = _optimization_queue.pop(0)
                
                # check if this plan is still in use
                with _cache_lock:
                    if key not in cls._last_used:
                        continue  # plan was removed, skip it
                    
                    # create optimized plan
                    try:
                        optimized_plan = cls._create_plan(
                            array, builder_func, n, axis, norm,
                            threads=cls._select_threads(array),
                            planner=MEASURE_PLANNER,
                            **kwargs
                        )
                        
                        # update cache with optimized plan
                        cls._plan_cache[key] = optimized_plan
                        cls._plan_quality[key] = MEASURE_PLANNER
                    except Exception:
                        # if optimization fails, keep the old plan
                        pass
        finally:
            with _optimization_lock:
                _background_optimizing = False
        
    @classmethod
    def fft(cls, array: np.ndarray, 
           n: Optional[int] = None, 
           axis: int = -1, 
           norm: Optional[str] = None, 
           threads: Optional[int] = None, 
           planner: Optional[str] = None) -> np.ndarray:
        """
        Compute FFT of input array with smart optimization.

        This method automatically:
        - Uses NumPy for non-power-of-2 sizes when enabled
        - Reuses cached plans for the same array shape
        - Selects optimal thread count based on array size
        - Progressively optimizes plans for frequently used shapes
        - Adaptively falls back to NumPy for specific sizes where it outperforms FFTW

        Args:
            array: Input array
            n: Length of transformed axis
            axis: Axis to transform
            norm: Normalization mode
            threads: Number of threads (None = auto-select)
            planner: FFTW planning strategy (None = auto-select)
            
        Returns:
            Transformed array
        """
        # Fallback: if configured to use NumPy for non-power-of-two inputs, do so.
        if USE_NUMPY_FOR_NON_POWER_OF_TWO:
            dimensions = [array.shape[axis] if n is None else n]
            if cls._is_non_power_of_two(dimensions):
                return np.fft.fft(array, n=n, axis=axis, norm=norm)
        
        # generate cache key for this transform
        key = cls._get_cache_key(array, n, axis, norm, 'fft')
        
        # Check if this transform is known to perform better with NumPy
        if hasattr(cls, '_numpy_fallback_keys') and key in cls._numpy_fallback_keys:
            # Use NumPy instead of FFTW for this specific case
            return np.fft.fft(array, n=n, axis=axis, norm=norm)
        
        with _cache_lock:
            # track call frequency for this shape
            cls._call_count[key] = cls._call_count.get(key, 0) + 1
            cls._last_used[key] = time.time()
            count = cls._call_count[key]
            
            # get or create plan
            if key in cls._plan_cache:
                # plan exists in cache
                fft_obj = cls._plan_cache[key]
                
                # check if we should upgrade the plan
                if cls._should_upgrade_plan(key):
                    # schedule optimization
                    cls._schedule_optimization(key, array, pyfftw.builders.fft, n, axis, norm)
                    cls._plan_quality[key] = MEASURE_PLANNER
            else:
                # create new plan
                if planner is None:
                    # use cached planner or default
                    planner = cls._plan_quality.get(key, DEFAULT_PLANNER)
                    
                fft_obj = cls._create_plan(array, pyfftw.builders.fft, n, axis, norm, threads, planner)
                
                # cache the plan
                cls._plan_cache[key] = fft_obj
                cls._plan_quality[key] = planner
            
            # time the FFTW execution 
            start_time = time.time()
            result = fft_obj(array)
            exec_time = time.time() - start_time
            
            # Check performance of non-power-of-2 transforms that should have been optimized
            transform_type, shape, dtype, n, axis, norm = key
            dimensions = shape if isinstance(shape, tuple) else (shape,)
            non_power_of_two = any((dim & (dim - 1)) != 0 for dim in dimensions)
            size = np.prod(dimensions)
            
            # If this is a non-power-of-2 transform that's been optimized,
            # check if NumPy might be faster
            if (non_power_of_two and 
                count > MIN_REPEAT_FOR_MEASURE and 
                cls._plan_quality.get(key) == MEASURE_PLANNER and
                size > 8192):
                
                # Only benchmark every few calls to avoid constant overhead
                if count % 5 == 0 and not hasattr(cls, '_numpy_fallback_keys'):
                    cls._numpy_fallback_keys = set()
                    
                if hasattr(cls, '_numpy_fallback_keys') and count % 5 == 0:
                    # Time NumPy's FFT on a copy of the array
                    array_copy = np.array(array, copy=True)
                    numpy_start = time.time()
                    np.fft.fft(array_copy, n=n, axis=axis, norm=norm)
                    numpy_time = time.time() - numpy_start
                    
                    # If NumPy is consistently faster by a significant margin,
                    # add this key to our fallback list
                    if numpy_time < exec_time * 0.9:  # NumPy is >10% faster
                        if not hasattr(cls, '_numpy_fallback_counts'):
                            cls._numpy_fallback_counts = {}
                        
                        cls._numpy_fallback_counts[key] = cls._numpy_fallback_counts.get(key, 0) + 1
                        
                        # If NumPy is consistently faster over multiple checks,
                        # switch to using NumPy for this transform
                        if cls._numpy_fallback_counts.get(key, 0) >= 3:
                            # Add to fallback set
                            cls._numpy_fallback_keys.add(key)
                            # Log the fallback decision
                            logger.info(f"BetterFFTW: Using NumPy for size {shape} - better performance detected")
            
            # PyFFTW doesn't handle normalization exactly like NumPy, so we need to handle it manually
            if norm == 'ortho':
                # Calculate the normalization factor correctly
                if n is None:
                    n = array.shape[axis]
                scale = 1.0 / np.sqrt(n)
                result *= scale
            
            return result
    
    @classmethod
    def ifft(cls, array: np.ndarray, 
            n: Optional[int] = None, 
            axis: int = -1, 
            norm: Optional[str] = None, 
            threads: Optional[int] = None, 
            planner: Optional[str] = None) -> np.ndarray:
        """
        Compute inverse FFT of input array with smart optimization.
        
        Args:
            array: Input array
            n: Length of transformed axis
            axis: Axis to transform
            norm: Normalization mode
            threads: Number of threads (None = auto-select)
            planner: FFTW planning strategy (None = auto-select)
            
        Returns:
            Inverse-transformed array
        """
        # Fallback: if configured to use NumPy for non-power-of-two inputs, do so.
        if USE_NUMPY_FOR_NON_POWER_OF_TWO:
            dimensions = [array.shape[axis] if n is None else n]
            if cls._is_non_power_of_two(dimensions):
                return np.fft.ifft(array, n=n, axis=axis, norm=norm)
                
        # generate cache key for this transform
        key = cls._get_cache_key(array, n, axis, norm, 'ifft')
        
        with _cache_lock:
            # track call frequency for this shape
            cls._call_count[key] = cls._call_count.get(key, 0) + 1
            cls._last_used[key] = time.time()
            
            # get or create plan
            if key in cls._plan_cache:
                # plan exists in cache
                ifft_obj = cls._plan_cache[key]
                
                # check if we should upgrade the plan
                if cls._should_upgrade_plan(key):
                    # schedule optimization
                    cls._schedule_optimization(key, array, pyfftw.builders.ifft, n, axis, norm)
                    cls._plan_quality[key] = MEASURE_PLANNER
            else:
                # create new plan
                if planner is None:
                    # use cached planner or default
                    planner = cls._plan_quality.get(key, DEFAULT_PLANNER)
                    
                ifft_obj = cls._create_plan(array, pyfftw.builders.ifft, n, axis, norm, threads, planner)
                
                # cache the plan
                cls._plan_cache[key] = ifft_obj
                cls._plan_quality[key] = planner
            
            # Execute the transform
            result = ifft_obj(array)
            
            # PyFFTW doesn't handle normalization exactly like NumPy, so we need to handle it manually
            if norm == 'ortho':
                # Calculate the normalization factor correctly
                if n is None:
                    n = array.shape[axis]
                scale = 1.0 / np.sqrt(n)
                result *= scale
            
            return result
    
    @classmethod
    def rfft(cls, array: np.ndarray, 
            n: Optional[int] = None, 
            axis: int = -1, 
            norm: Optional[str] = None, 
            threads: Optional[int] = None, 
            planner: Optional[str] = None) -> np.ndarray:
        """
        Compute real FFT of input array with smart optimization.
        
        Args:
            array: Real input array
            n: Length of transformed axis
            axis: Axis to transform
            norm: Normalization mode
            threads: Number of threads (None = auto-select)
            planner: FFTW planning strategy (None = auto-select)
            
        Returns:
            Transformed array
        """
        
        # Fallback: if configured to use NumPy for non-power-of-two inputs, do so.
        if USE_NUMPY_FOR_NON_POWER_OF_TWO:
            dimensions = [array.shape[axis] if n is None else n]
            if cls._is_non_power_of_two(dimensions):
                return np.fft.rfft(array, n=n, axis=axis, norm=norm)
        
        # generate cache key for this transform
        key = cls._get_cache_key(array, n, axis, norm, 'rfft')
        
        with _cache_lock:
            # track call frequency for this shape
            cls._call_count[key] = cls._call_count.get(key, 0) + 1
            cls._last_used[key] = time.time()
            
            # get or create plan
            if key in cls._plan_cache:
                # plan exists in cache
                rfft_obj = cls._plan_cache[key]
                
                # check if we should upgrade the plan
                if cls._should_upgrade_plan(key):
                    # schedule optimization
                    cls._schedule_optimization(key, array, pyfftw.builders.rfft, n, axis, norm)
                    cls._plan_quality[key] = MEASURE_PLANNER
            else:
                # create new plan
                if planner is None:
                    # use cached planner or default
                    planner = cls._plan_quality.get(key, DEFAULT_PLANNER)
                    
                rfft_obj = cls._create_plan(array, pyfftw.builders.rfft, n, axis, norm, threads, planner)
                
                # cache the plan
                cls._plan_cache[key] = rfft_obj
                cls._plan_quality[key] = planner
            
            # Execute the transform
            result = rfft_obj(array)
            
            # PyFFTW doesn't handle normalization exactly like NumPy, so we need to handle it manually
            if norm == 'ortho':
                # Calculate the normalization factor correctly
                if n is None:
                    n = array.shape[axis]
                scale = 1.0 / np.sqrt(n)
                result *= scale
            
            return result
    
    @classmethod
    def irfft(cls, array: np.ndarray, 
             n: Optional[int] = None, 
             axis: int = -1, 
             norm: Optional[str] = None, 
             threads: Optional[int] = None, 
             planner: Optional[str] = None) -> np.ndarray:
        """
        Compute inverse real FFT of input array with smart optimization.
        
        Args:
            array: Input array with Hermitian symmetry
            n: Length of transformed axis
            axis: Axis to transform
            norm: Normalization mode
            threads: Number of threads (None = auto-select)
            planner: FFTW planning strategy (None = auto-select)
            
        Returns:
            Inverse-transformed real array
        """
        # Fallback: if configured to use NumPy for non-power-of-two inputs, do so.
        if USE_NUMPY_FOR_NON_POWER_OF_TWO:  
            dimensions = [array.shape[axis] if n is None else n]
            if cls._is_non_power_of_two(dimensions):
                return np.fft.irfft(array, n=n, axis=axis, norm=norm)
        
        # generate cache key for this transform
        key = cls._get_cache_key(array, n, axis, norm, 'irfft')
        
        with _cache_lock:
            # track call frequency for this shape
            cls._call_count[key] = cls._call_count.get(key, 0) + 1
            cls._last_used[key] = time.time()
            
            # get or create plan
            if key in cls._plan_cache:
                # plan exists in cache
                irfft_obj = cls._plan_cache[key]
                
                # check if we should upgrade the plan
                if cls._should_upgrade_plan(key):
                    # schedule optimization
                    cls._schedule_optimization(key, array, pyfftw.builders.irfft, n, axis, norm)
                    cls._plan_quality[key] = MEASURE_PLANNER
            else:
                # create new plan
                if planner is None:
                    # use cached planner or default
                    planner = cls._plan_quality.get(key, DEFAULT_PLANNER)
                    
                irfft_obj = cls._create_plan(array, pyfftw.builders.irfft, n, axis, norm, threads, planner)
                
                # cache the plan
                cls._plan_cache[key] = irfft_obj
                cls._plan_quality[key] = planner
            
            # Execute the transform
            result = irfft_obj(array)
            
            # PyFFTW doesn't handle normalization exactly like NumPy, so we need to handle it manually
            if norm == 'ortho':
                # Calculate the normalization factor correctly
                if n is None:
                    n = array.shape[axis]
                scale = 1.0 / np.sqrt(n)
                result *= scale

            return result
    
    @classmethod
    def fft2(cls, array: np.ndarray, 
            s: Optional[Tuple[int, int]] = None, 
            axes: Tuple[int, int] = (-2, -1), 
            norm: Optional[str] = None, 
            threads: Optional[int] = None, 
            planner: Optional[str] = None) -> np.ndarray:
        """
        Compute 2D FFT of input array with smart optimization.
        
        Args:
            array: Input array
            s: Shape of transformed axes
            axes: Axes to transform
            norm: Normalization mode
            threads: Number of threads (None = auto-select)
            planner: FFTW planning strategy (None = auto-select)
            
        Returns:
            Transformed array
        """
        
        # Fallback: if configured to use NumPy for non-power-of-two inputs, do so.
        if USE_NUMPY_FOR_NON_POWER_OF_TWO:
            if s is not None:
                dimensions = s
            else:
                dimensions = [array.shape[ax] for ax in axes]
    
            if cls._is_non_power_of_two(dimensions):
                return np.fft.fft2(array, s=s, axes=axes, norm=norm)        
        
        # generate cache key for this transform
        key = cls._get_cache_key(array, s, axes, norm, 'fft2')
        
        with _cache_lock:
            # track call frequency for this shape
            cls._call_count[key] = cls._call_count.get(key, 0) + 1
            cls._last_used[key] = time.time()
            
            # get or create plan
            if key in cls._plan_cache:
                # plan exists in cache
                fft2_obj = cls._plan_cache[key]
                
                # check if we should upgrade the plan
                if cls._should_upgrade_plan(key):
                    # schedule optimization
                    cls._schedule_optimization(key, array, pyfftw.builders.fft2, s, axes, norm)
                    cls._plan_quality[key] = MEASURE_PLANNER
            else:
                # create new plan
                if planner is None:
                    # use cached planner or default
                    planner = cls._plan_quality.get(key, DEFAULT_PLANNER)
                    
                fft2_obj = cls._create_plan(array, pyfftw.builders.fft2, s, axes, norm, threads, planner)
                
                # cache the plan
                cls._plan_cache[key] = fft2_obj
                cls._plan_quality[key] = planner
        
            # Execute the transform
            result = fft2_obj(array)
            
            # PyFFTW doesn't handle normalization exactly like NumPy, so we need to handle it manually
            if norm == 'ortho':
                # Calculate the normalization factor correctly
                if s is None:
                    # If s is None, use the array shape along the transformed axes
                    dims = [array.shape[ax] for ax in axes]
                else:
                    # Otherwise use the specified shape
                    dims = s
                scale = 1.0 / np.sqrt(np.prod(dims))
                result *= scale
            
            return result
    
    @classmethod
    def ifft2(cls, array: np.ndarray, 
             s: Optional[Tuple[int, int]] = None, 
             axes: Tuple[int, int] = (-2, -1), 
             norm: Optional[str] = None, 
             threads: Optional[int] = None, 
             planner: Optional[str] = None) -> np.ndarray:
        """
        Compute inverse 2D FFT of input array with smart optimization.
        
        Args:
            array: Input array
            s: Shape of transformed axes
            axes: Axes to transform
            norm: Normalization mode
            threads: Number of threads (None = auto-select)
            planner: FFTW planning strategy (None = auto-select)
            
        Returns:
            Inverse-transformed array
        """
        # Fallback: if configured to use NumPy for non-power-of-two inputs, do so.
        if USE_NUMPY_FOR_NON_POWER_OF_TWO:
            if s is not None:
                dimensions = s
            else:
                dimensions = [array.shape[ax] for ax in axes]
    
            if cls._is_non_power_of_two(dimensions):
                return np.fft.ifft2(array, s=s, axes=axes, norm=norm)
        # generate cache key for this transform
        key = cls._get_cache_key(array, s, axes, norm, 'ifft2')
        with _cache_lock:
            # track call frequency for this shape
            cls._call_count[key] = cls._call_count.get(key, 0) + 1
            cls._last_used[key] = time.time()
            
            # get or create plan
            if key in cls._plan_cache:
                # plan exists in cache
                ifft2_obj = cls._plan_cache[key]
                
                # check if we should upgrade the plan
                if cls._should_upgrade_plan(key):
                    # schedule optimization
                    cls._schedule_optimization(key, array, pyfftw.builders.ifft2, s, axes, norm)
                    cls._plan_quality[key] = MEASURE_PLANNER
            else:
                # create new plan
                if planner is None:
                    # use cached planner or default
                    planner = cls._plan_quality.get(key, DEFAULT_PLANNER)
                    
                ifft2_obj = cls._create_plan(array, pyfftw.builders.ifft2, s, axes, norm, threads, planner)
                
                # cache the plan
                cls._plan_cache[key] = ifft2_obj
                cls._plan_quality[key] = planner
        
            # Execute the transform
            result = ifft2_obj(array)
            
            # PyFFTW doesn't handle normalization exactly like NumPy, so we need to handle it manually
            if norm == 'ortho':
                # Calculate the normalization factor correctly
                if s is None:
                    # If s is None, use the array shape along the transformed axes
                    dims = [array.shape[ax] for ax in axes]
                else:
                    # Otherwise use the specified shape
                    dims = s
                scale = 1.0 / np.sqrt(np.prod(dims))
                result *= scale
            
            return result
    
    @classmethod
    def rfft2(cls, array: np.ndarray, 
             s: Optional[Tuple[int, int]] = None, 
             axes: Tuple[int, int] = (-2, -1), 
             norm: Optional[str] = None, 
             threads: Optional[int] = None, 
             planner: Optional[str] = None) -> np.ndarray:
        """
        Compute 2D real FFT of input array with smart optimization.
        
        Args:
            array: Real input array
            s: Shape of transformed axes
            axes: Axes to transform
            norm: Normalization mode
            threads: Number of threads (None = auto-select)
            planner: FFTW planning strategy (None = auto-select)
            
        Returns:
            Transformed array
        """
        # Fallback: if configured to use NumPy for non-power-of-two inputs, do so.
        if USE_NUMPY_FOR_NON_POWER_OF_TWO:
            if s is not None:
                dimensions = s
            else:
                dimensions = [array.shape[ax] for ax in axes]
            
            if cls._is_non_power_of_two(dimensions):
                return np.fft.rfft2(array, s=s, axes=axes, norm=norm)
        # generate cache key for this transform
        key = cls._get_cache_key(array, s, axes, norm, 'rfft2')
        
        with _cache_lock:
            # track call frequency for this shape
            cls._call_count[key] = cls._call_count.get(key, 0) + 1
            cls._last_used[key] = time.time()
            
            # get or create plan
            if key in cls._plan_cache:
                # plan exists in cache
                rfft2_obj = cls._plan_cache[key]
                
                # check if we should upgrade the plan
                if cls._should_upgrade_plan(key):
                    # schedule optimization
                    cls._schedule_optimization(key, array, pyfftw.builders.rfft2, s, axes, norm)
                    cls._plan_quality[key] = MEASURE_PLANNER
            else:
                # create new plan
                if planner is None:
                    # use cached planner or default
                    planner = cls._plan_quality.get(key, DEFAULT_PLANNER)
                    
                rfft2_obj = cls._create_plan(array, pyfftw.builders.rfft2, s, axes, norm, threads, planner)
                
                # cache the plan
                cls._plan_cache[key] = rfft2_obj
                cls._plan_quality[key] = planner
        
            # Execute the transform
            result = rfft2_obj(array)
            
            # PyFFTW doesn't handle normalization exactly like NumPy, so we need to handle it manually
            if norm == 'ortho':
                # Calculate the normalization factor correctly
                if s is None:
                    # If s is None, use the array shape along the transformed axes
                    dims = [array.shape[ax] for ax in axes]
                else:
                    # Otherwise use the specified shape
                    dims = s
                scale = 1.0 / np.sqrt(np.prod(dims))
                result *= scale
            
            return result
    
    @classmethod
    def irfft2(cls, array: np.ndarray, 
              s: Optional[Tuple[int, int]] = None, 
              axes: Tuple[int, int] = (-2, -1), 
              norm: Optional[str] = None, 
              threads: Optional[int] = None, 
              planner: Optional[str] = None) -> np.ndarray:
        """
        Compute inverse 2D real FFT of input array with smart optimization.
        
        Args:
            array: Input array with Hermitian symmetry
            s: Shape of transformed axes
            axes: Axes to transform
            norm: Normalization mode
            threads: Number of threads (None = auto-select)
            planner: FFTW planning strategy (None = auto-select)
            
        Returns:
            Inverse-transformed real array
        """
        # Fallback: if configured to use NumPy for non-power-of-two inputs, do so.
        if USE_NUMPY_FOR_NON_POWER_OF_TWO:
            if s is not None:
                dimensions = s
            else:
                dimensions = [array.shape[ax] for ax in axes]
            
            if cls._is_non_power_of_two(dimensions):
                return np.fft.irfft2(array, s=s, axes=axes, norm=norm)
        # generate cache key for this transform
        key = cls._get_cache_key(array, s, axes, norm, 'irfft2')
        
        with _cache_lock:
            # track call frequency for this shape
            cls._call_count[key] = cls._call_count.get(key, 0) + 1
            cls._last_used[key] = time.time()
            
            # get or create plan
            if key in cls._plan_cache:
                # plan exists in cache
                irfft2_obj = cls._plan_cache[key]
                
                # check if we should upgrade the plan
                if cls._should_upgrade_plan(key):
                    # schedule optimization
                    cls._schedule_optimization(key, array, pyfftw.builders.irfft2, s, axes, norm)
                    cls._plan_quality[key] = MEASURE_PLANNER
            else:
                # create new plan
                if planner is None:
                    # use cached planner or default
                    planner = cls._plan_quality.get(key, DEFAULT_PLANNER)
                    
                irfft2_obj = cls._create_plan(array, pyfftw.builders.irfft2, s, axes, norm, threads, planner)
                
                # cache the plan
                cls._plan_cache[key] = irfft2_obj
                cls._plan_quality[key] = planner
        
            # Execute the transform
            result = irfft2_obj(array)
            
            # PyFFTW doesn't handle normalization exactly like NumPy, so we need to handle it manually
            if norm == 'ortho':
                # Calculate the normalization factor correctly
                if s is None:
                    # If s is None, use the array shape along the transformed axes
                    dims = [array.shape[ax] for ax in axes]
                else:
                    # Otherwise use the specified shape
                    dims = s
                scale = 1.0 / np.sqrt(np.prod(dims))
                result *= scale
            
            return result
    
    @classmethod
    def fftn(cls, array: np.ndarray, 
            s: Optional[Tuple[int, ...]] = None, 
            axes: Optional[Tuple[int, ...]] = None, 
            norm: Optional[str] = None, 
            threads: Optional[int] = None, 
            planner: Optional[str] = None) -> np.ndarray:
        """
        Compute n-dimensional FFT of input array with smart optimization.
        
        Args:
            array: Input array
            s: Shape of transformed axes
            axes: Axes to transform
            norm: Normalization mode
            threads: Number of threads (None = auto-select)
            planner: FFTW planning strategy (None = auto-select)
            
        Returns:
            Transformed array
        """
        # Fallback: if configured to use NumPy for non-power-of-two inputs, do so.
        if USE_NUMPY_FOR_NON_POWER_OF_TWO:
            if s is not None:
                dimensions = s
            elif axes is not None:
                dimensions = [array.shape[ax] for ax in axes]
            else:
                dimensions = array.shape
            
            if cls._is_non_power_of_two(dimensions):
                return np.fft.fftn(array, s=s, axes=axes, norm=norm)
        # generate cache key for this transform
        key = cls._get_cache_key(array, s, axes, norm, 'fftn')
        
        with _cache_lock:
            # track call frequency for this shape
            cls._call_count[key] = cls._call_count.get(key, 0) + 1
            cls._last_used[key] = time.time()
            
            # get or create plan
            if key in cls._plan_cache:
                # plan exists in cache
                fftn_obj = cls._plan_cache[key]
                
                # check if we should upgrade the plan
                if cls._should_upgrade_plan(key):
                    # schedule optimization
                    cls._schedule_optimization(key, array, pyfftw.builders.fftn, s, axes, norm)
                    cls._plan_quality[key] = MEASURE_PLANNER
            else:
                # create new plan
                if planner is None:
                    # use cached planner or default
                    planner = cls._plan_quality.get(key, DEFAULT_PLANNER)
                    
                fftn_obj = cls._create_plan(array, pyfftw.builders.fftn, s, axes, norm, threads, planner)
                
                # cache the plan
                cls._plan_cache[key] = fftn_obj
                cls._plan_quality[key] = planner
        
            # Execute the transform
            result = fftn_obj(array)
            
            # PyFFTW doesn't handle normalization exactly like NumPy, so we need to handle it manually
            if norm == 'ortho':
                # Calculate the normalization factor correctly
                if s is None:
                    # If s is None, use the array shape along the transformed axes
                    if axes is None:
                        # If axes is None, all dimensions are transformed
                        dims = array.shape
                    else:
                        # Otherwise use the specified axes
                        dims = [array.shape[ax] for ax in axes]
                else:
                    # Otherwise use the specified shape
                    dims = s
                scale = 1.0 / np.sqrt(np.prod(dims))
                result *= scale
            
            return result
    
    @classmethod
    def ifftn(cls, array: np.ndarray, 
             s: Optional[Tuple[int, ...]] = None, 
             axes: Optional[Tuple[int, ...]] = None, 
             norm: Optional[str] = None, 
             threads: Optional[int] = None, 
             planner: Optional[str] = None) -> np.ndarray:
        """
        Compute inverse n-dimensional FFT of input array with smart optimization.
        
        Args:
            array: Input array
            s: Shape of transformed axes
            axes: Axes to transform
            norm: Normalization mode
            threads: Number of threads (None = auto-select)
            planner: FFTW planning strategy (None = auto-select)
            
        Returns:
            Inverse-transformed array
        """
        # Fallback: if configured to use NumPy for non-power-of-two inputs, do so.
        if USE_NUMPY_FOR_NON_POWER_OF_TWO:
            if s is not None:
                dimensions = s
            elif axes is not None:
                dimensions = [array.shape[ax] for ax in axes]
            else:
                dimensions = array.shape
            
            if cls._is_non_power_of_two(dimensions):
                return np.fft.ifftn(array, s=s, axes=axes, norm=norm)
        # generate cache key for this transform
        key = cls._get_cache_key(array, s, axes, norm, 'ifftn')
        
        with _cache_lock:
            # track call frequency for this shape
            cls._call_count[key] = cls._call_count.get(key, 0) + 1
            cls._last_used[key] = time.time()
            
            # get or create plan
            if key in cls._plan_cache:
                # plan exists in cache
                ifftn_obj = cls._plan_cache[key]
                
                # check if we should upgrade the plan
                if cls._should_upgrade_plan(key):
                    # schedule optimization
                    cls._schedule_optimization(key, array, pyfftw.builders.ifftn, s, axes, norm)
                    cls._plan_quality[key] = MEASURE_PLANNER
            else:
                # create new plan
                if planner is None:
                    # use cached planner or default
                    planner = cls._plan_quality.get(key, DEFAULT_PLANNER)
                    
                ifftn_obj = cls._create_plan(array, pyfftw.builders.ifftn, s, axes, norm, threads, planner)
                
                # cache the plan
                cls._plan_cache[key] = ifftn_obj
                cls._plan_quality[key] = planner
        
            # Execute the transform
            result = ifftn_obj(array)
            
            # PyFFTW doesn't handle normalization exactly like NumPy, so we need to handle it manually
            if norm == 'ortho':
                # Calculate the normalization factor correctly
                if s is None:
                    # If s is None, use the array shape along the transformed axes
                    if axes is None:
                        # If axes is None, all dimensions are transformed
                        dims = array.shape
                    else:
                        # Otherwise use the specified axes
                        dims = [array.shape[ax] for ax in axes]
                else:
                    # Otherwise use the specified shape
                    dims = s
                scale = 1.0 / np.sqrt(np.prod(dims))
                result *= scale
            
            return result
    
    @classmethod
    def rfftn(cls, array: np.ndarray, 
             s: Optional[Tuple[int, ...]] = None, 
             axes: Optional[Tuple[int, ...]] = None, 
             norm: Optional[str] = None, 
             threads: Optional[int] = None, 
             planner: Optional[str] = None) -> np.ndarray:
        """
        Compute n-dimensional real FFT of input array with smart optimization.
        
        Args:
            array: Real input array
            s: Shape of transformed axes
            axes: Axes to transform
            norm: Normalization mode
            threads: Number of threads (None = auto-select)
            planner: FFTW planning strategy (None = auto-select)
            
        Returns:
            Transformed array
        """
        
        # Fallback: if configured to use NumPy for non-power-of-two inputs, do so.
        if USE_NUMPY_FOR_NON_POWER_OF_TWO:
            if s is not None:
                dimensions = s
            elif axes is not None:
                dimensions = [array.shape[ax] for ax in axes]
            else:
                dimensions = array.shape
            
            if cls._is_non_power_of_two(dimensions):
                return np.fft.rfftn(array, s=s, axes=axes, norm=norm)
        # generate cache key for this transform
        key = cls._get_cache_key(array, s, axes, norm, 'rfftn')
        
        with _cache_lock:
            # track call frequency for this shape
            cls._call_count[key] = cls._call_count.get(key, 0) + 1
            cls._last_used[key] = time.time()
            
            # get or create plan
            if key in cls._plan_cache:
                # plan exists in cache
                rfftn_obj = cls._plan_cache[key]
                
                # check if we should upgrade the plan
                if cls._should_upgrade_plan(key):
                    # schedule optimization
                    cls._schedule_optimization(key, array, pyfftw.builders.rfftn, s, axes, norm)
                    cls._plan_quality[key] = MEASURE_PLANNER
            else:
                # create new plan
                if planner is None:
                    # use cached planner or default
                    planner = cls._plan_quality.get(key, DEFAULT_PLANNER)
                    
                rfftn_obj = cls._create_plan(array, pyfftw.builders.rfftn, s, axes, norm, threads, planner)
                
                # cache the plan
                cls._plan_cache[key] = rfftn_obj
                cls._plan_quality[key] = planner
        
            # Execute the transform
            result = rfftn_obj(array)
            
            # PyFFTW doesn't handle normalization exactly like NumPy, so we need to handle it manually
            if norm == 'ortho':
                # Calculate the normalization factor correctly
                if s is None:
                    # If s is None, use the array shape along the transformed axes
                    if axes is None:
                        # If axes is None, all dimensions are transformed
                        dims = array.shape
                    else:
                        # Otherwise use the specified axes
                        dims = [array.shape[ax] for ax in axes]
                else:
                    # Otherwise use the specified shape
                    dims = s
                scale = 1.0 / np.sqrt(np.prod(dims))
                result *= scale
            
            return result
    
    @classmethod
    def irfftn(cls, array: np.ndarray, 
              s: Optional[Tuple[int, ...]] = None, 
              axes: Optional[Tuple[int, ...]] = None, 
              norm: Optional[str] = None, 
              threads: Optional[int] = None, 
              planner: Optional[str] = None) -> np.ndarray:
        """
        Compute inverse n-dimensional real FFT of input array with smart optimization.
        
        Args:
            array: Input array with Hermitian symmetry
            s: Shape of transformed axes
            axes: Axes to transform
            norm: Normalization mode
            threads: Number of threads (None = auto-select)
            planner: FFTW planning strategy (None = auto-select)
            
        Returns:
            Inverse-transformed real array
        """
        # Fallback: if configured to use NumPy for non-power-of-two inputs, do so.
        if USE_NUMPY_FOR_NON_POWER_OF_TWO:
            if s is not None:
                dimensions = s
            elif axes is not None:
                dimensions = [array.shape[ax] for ax in axes]
            else:
                dimensions = array.shape
            
            if cls._is_non_power_of_two(dimensions):
                return np.fft.irfftn(array, s=s, axes=axes, norm=norm)
        # generate cache key for this transform
        key = cls._get_cache_key(array, s, axes, norm, 'irfftn')
        
        with _cache_lock:
            # track call frequency for this shape
            cls._call_count[key] = cls._call_count.get(key, 0) + 1
            cls._last_used[key] = time.time()
            
            # get or create plan
            if key in cls._plan_cache:
                # plan exists in cache
                irfftn_obj = cls._plan_cache[key]
                
                # check if we should upgrade the plan
                if cls._should_upgrade_plan(key):
                    # schedule optimization
                    cls._schedule_optimization(key, array, pyfftw.builders.irfftn, s, axes, norm)
                    cls._plan_quality[key] = MEASURE_PLANNER
            else:
                # create new plan
                if planner is None:
                    # use cached planner or default
                    planner = cls._plan_quality.get(key, DEFAULT_PLANNER)
                    
                irfftn_obj = cls._create_plan(array, pyfftw.builders.irfftn, s, axes, norm, threads, planner)
                
                # cache the plan
                cls._plan_cache[key] = irfftn_obj
                cls._plan_quality[key] = planner
        
            # Execute the transform
            result = irfftn_obj(array)
            
            # PyFFTW doesn't handle normalization exactly like NumPy, so we need to handle it manually
            if norm == 'ortho':
                # Calculate the normalization factor correctly
                if s is None:
                    # If s is None, use the array shape along the transformed axes
                    if axes is None:
                        # If axes is None, all dimensions are transformed
                        dims = array.shape
                    else:
                        # Otherwise use the specified axes
                        dims = [array.shape[ax] for ax in axes]
                else:
                    # Otherwise use the specified shape
                    dims = s
                scale = 1.0 / np.sqrt(np.prod(dims))
                result *= scale
            
            return result
    @classmethod
    def import_wisdom(cls, filename: str = None) -> bool:
        """
        Import FFTW wisdom from a file.
        
        Args:
            filename: Path to wisdom file (None = use default path)
            
        Returns:
            True if wisdom was successfully imported, False otherwise
        """
        if filename is None:
            filename = WISDOM_FILE
            
        try:
            # Check if file exists first
            if not os.path.exists(filename):
                logger.warning(f"Wisdom file not found: {filename}")
                return False
                
            # Read the file
            with open(filename, 'rb') as f:
                wisdom_data = f.read()
                
            # Handle empty file case
            if not wisdom_data:
                logger.warning(f"Wisdom file is empty: {filename}")
                return False
                
            # Import the wisdom - PyFFTW expects a tuple of (bytes, bytes, bytes)
            # We need to create a proper wisdom tuple
            wisdom_tuple = (wisdom_data, b'', b'')  # Format: (double, single, long double)
            result = pyfftw.import_wisdom(wisdom_tuple)
            
            return True
        except Exception as e:
            logger.warning(f"Error importing wisdom: {str(e)}")
            return False

    @classmethod
    def export_wisdom(cls, filename: str = None) -> bool:
        """
        Export FFTW wisdom to a file.
        
        Args:
            filename: Path to wisdom file (None = use default path)
            
        Returns:
            True if wisdom was successfully exported, False otherwise
        """
        if filename is None:
            filename = WISDOM_FILE
            
        try:
            # Generate some wisdom if none exists by running a transform
            # This ensures we have something to export
            dummy_arr = np.ones((16, 16), dtype=np.complex128)
            pyfftw.interfaces.numpy_fft.fft2(dummy_arr)
            
            # Get the wisdom
            wisdom = pyfftw.export_wisdom()
            if not wisdom or len(wisdom) < 1:
                logger.warning("No wisdom available to export")
                return False
                
            # Create directory if it doesn't exist
            directory = os.path.dirname(filename)
            if directory:
                os.makedirs(directory, exist_ok=True)
                
            # Write wisdom to file - only write the double precision wisdom (first element)
            wisdom_bytes = wisdom[0]
            if not isinstance(wisdom_bytes, bytes):
                logger.warning(f"Expected bytes, got {type(wisdom_bytes)}")
                return False
                
            with open(filename, 'wb') as f:
                f.write(wisdom_bytes)
                
            return True
        except Exception as e:
            logger.warning(f"Error exporting wisdom: {str(e)}")
            return False
    @classmethod
    def set_default_threads(cls, threads: int):
        """Set the default thread count for all transforms."""
        global DEFAULT_THREADS
        DEFAULT_THREADS = threads
        
    @classmethod
    def set_default_planner(cls, planner: str):
        """Set the default planning strategy."""
        global DEFAULT_PLANNER
        if planner not in ('FFTW_ESTIMATE', 'FFTW_MEASURE', 'FFTW_PATIENT', 'FFTW_EXHAUSTIVE'):
            raise ValueError(f"Invalid planner: {planner}")
        DEFAULT_PLANNER = planner
        
    @classmethod
    def set_optimization_threshold(cls, count: int):
        """Set the number of calls before upgrading plans."""
        global MIN_REPEAT_FOR_MEASURE
        MIN_REPEAT_FOR_MEASURE = count
    @classmethod
    def set_threading_thresholds(cls, small=16384, medium=65536, large=262144):
        """
        Configure array size thresholds for threading decisions.
        
        Args:
            small: Arrays smaller than this use single thread
            medium: Arrays smaller than this use limited threads
            large: Arrays smaller than this have thread count based on dimensionality
        """
        global THREADING_SMALL_THRESHOLD, THREADING_MEDIUM_THRESHOLD, THREADING_LARGE_THRESHOLD
        THREADING_SMALL_THRESHOLD = small
        THREADING_MEDIUM_THRESHOLD = medium
        THREADING_LARGE_THRESHOLD = large
        
    @classmethod
    def set_threading_limits(cls, small_max=1, medium_max=2, multi_dim_max=4):
        """
        Configure maximum thread counts for different array sizes.
        
        Args:
            small_max: Maximum threads for small arrays
            medium_max: Maximum threads for medium arrays
            multi_dim_max: Maximum threads for multi-dimensional arrays
        """
        global THREADING_MAX_SMALL, THREADING_MAX_MEDIUM, THREADING_MAX_MULTI_DIM
        THREADING_MAX_SMALL = small_max
        THREADING_MAX_MEDIUM = medium_max
        THREADING_MAX_MULTI_DIM = multi_dim_max
        
    @classmethod
    def set_max_cache_size(cls, size=1000):
        """
        Set the maximum number of plans to keep in cache.
        
        When the cache exceeds this size, older/less used plans will be removed.
        
        Args:
            size: Maximum number of plans to keep in cache
        """
        global MAX_CACHE_SIZE
        MAX_CACHE_SIZE = size
    @classmethod
    def get_stats(cls) -> Dict:
        """Get statistics about the plan cache."""
        with _cache_lock:
            stats = {
                'total_plans': len(cls._plan_cache),
                'estimated_plans': sum(1 for q in cls._plan_quality.values() if q == 'FFTW_ESTIMATE'),
                'measured_plans': sum(1 for q in cls._plan_quality.values() if q == 'FFTW_MEASURE'),
                'patient_plans': sum(1 for q in cls._plan_quality.values() if q == 'FFTW_PATIENT'),
                'total_calls': sum(cls._call_count.values()),
                'unique_shapes': len(set(key[1] for key in cls._call_count.keys())),
                'pending_optimizations': len(_optimization_queue),
                'background_optimizing': _background_optimizing,
            }
            return stats


# Attempt to import wisdom at module load time
SmartFFTW.import_wisdom()

# Start a periodic timer to clean up the cache
def _clean_cache():
    """Periodically clean the cache of unused plans."""
    SmartFFTW.clear_cache(older_than=DEFAULT_CACHE_TIMEOUT)
    threading.Timer(_cache_cleaning_interval, _clean_cache).start()
    
# Start the cache cleaning timer
cleaning_timer = threading.Timer(_cache_cleaning_interval, _clean_cache)
cleaning_timer.daemon = True  # allow the program to exit without waiting for timer
cleaning_timer.start()

# Create a singleton instance for simple access
fftw = SmartFFTW()

# Simplified function interfaces
def fft(array, n=None, axis=-1, norm=None, threads=None, planner=None):
    """Smart FFT function with auto-optimization."""
    return SmartFFTW.fft(array, n, axis, norm, threads, planner)

def ifft(array, n=None, axis=-1, norm=None, threads=None, planner=None):
    """Smart inverse FFT function with auto-optimization."""
    return SmartFFTW.ifft(array, n, axis, norm, threads, planner)

def rfft(array, n=None, axis=-1, norm=None, threads=None, planner=None):
    """Smart real FFT function with auto-optimization."""
    return SmartFFTW.rfft(array, n, axis, norm, threads, planner)

def irfft(array, n=None, axis=-1, norm=None, threads=None, planner=None):
    """Smart inverse real FFT function with auto-optimization."""
    return SmartFFTW.irfft(array, n, axis, norm, threads, planner)

def fft2(array, s=None, axes=(-2, -1), norm=None, threads=None, planner=None):
    """Smart 2D FFT function with auto-optimization."""
    return SmartFFTW.fft2(array, s, axes, norm, threads, planner)

def ifft2(array, s=None, axes=(-2, -1), norm=None, threads=None, planner=None):
    """Smart inverse 2D FFT function with auto-optimization."""
    return SmartFFTW.ifft2(array, s, axes, norm, threads, planner)

def rfft2(array, s=None, axes=(-2, -1), norm=None, threads=None, planner=None):
    """Smart 2D real FFT function with auto-optimization."""
    return SmartFFTW.rfft2(array, s, axes, norm, threads, planner)

def irfft2(array, s=None, axes=(-2, -1), norm=None, threads=None, planner=None):
    """Smart inverse 2D real FFT function with auto-optimization."""
    return SmartFFTW.irfft2(array, s, axes, norm, threads, planner)

def fftn(array, s=None, axes=None, norm=None, threads=None, planner=None):
    """Smart n-dimensional FFT function with auto-optimization."""
    return SmartFFTW.fftn(array, s, axes, norm, threads, planner)

def ifftn(array, s=None, axes=None, norm=None, threads=None, planner=None):
    """Smart inverse n-dimensional FFT function with auto-optimization."""
    return SmartFFTW.ifftn(array, s, axes, norm, threads, planner)

def rfftn(array, s=None, axes=None, norm=None, threads=None, planner=None):
    """Smart n-dimensional real FFT function with auto-optimization."""
    return SmartFFTW.rfftn(array, s, axes, norm, threads, planner)

def irfftn(array, s=None, axes=None, norm=None, threads=None, planner=None):
    """Smart inverse n-dimensional real FFT function with auto-optimization."""
    return SmartFFTW.irfftn(array, s, axes, norm, threads, planner)

def set_threading_thresholds(small=16384, medium=65536, large=262144):
    """
    Configure array size thresholds for threading decisions.
    
    Args:
        small: Arrays smaller than this use single thread
        medium: Arrays smaller than this use limited threads
        large: Arrays smaller than this have thread count based on dimensionality
    """
    global THREADING_SMALL_THRESHOLD, THREADING_MEDIUM_THRESHOLD, THREADING_LARGE_THRESHOLD
    THREADING_SMALL_THRESHOLD = small
    THREADING_MEDIUM_THRESHOLD = medium
    THREADING_LARGE_THRESHOLD = large

def set_threading_limits(small_max=1, medium_max=2, multi_dim_max=4):
    """
    Configure maximum thread counts for different array sizes.
    
    Args:
        small_max: Maximum threads for small arrays
        medium_max: Maximum threads for medium arrays
        multi_dim_max: Maximum threads for multi-dimensional arrays
    """
    global THREADING_MAX_SMALL, THREADING_MAX_MEDIUM, THREADING_MAX_MULTI_DIM
    THREADING_MAX_SMALL = small_max
    THREADING_MAX_MEDIUM = medium_max
    THREADING_MAX_MULTI_DIM = multi_dim_max

def set_max_cache_size(size=1000):
    """
    Set the maximum number of plans to keep in cache.
    
    When the cache exceeds this size, older/less used plans will be removed.
    
    Args:
        size: Maximum number of plans to keep in cache
    """
    global MAX_CACHE_SIZE
    MAX_CACHE_SIZE = size

# Utility functions
def empty_aligned(shape, dtype=np.complex128, n=None):
    """Create an empty aligned array for optimal FFTW performance."""
    return pyfftw.empty_aligned(shape, dtype, n)

def empty_aligned_like(array, n=None):
    """Create an empty aligned array like another array."""
    return pyfftw.empty_aligned(array.shape, array.dtype, n)

def byte_align(array, n=None, copy=True):
    """
    Align an existing array to memory boundary for optimal FFTW performance.
    
    Args:
        array: Array to align
        n: Byte boundary to align to (None = FFTW default)
        copy: Whether to copy the array if it's already aligned
        
    Returns:
        Aligned array
    """
    try:
        # Try using keyword arguments first
        return pyfftw.byte_align(array, n=n, copy=copy)
    except TypeError:
        # Fall back to positional arguments
        if n is None:
            return pyfftw.byte_align(array, copy)
        else:
            return pyfftw.byte_align(array, n, copy)
    except Exception as e:
        # Last resort - just return the original array
        logger.warning(f"byte_align failed: {str(e)}")
        return np.array(array, copy=copy)

def import_wisdom(filename=None):
    """Import FFTW wisdom from a file."""
    return SmartFFTW.import_wisdom(filename)

def export_wisdom(filename=None):
    """Export FFTW wisdom to a file."""
    return SmartFFTW.export_wisdom(filename)

def set_num_threads(threads):
    """Set the default number of threads for FFT computations."""
    SmartFFTW.set_default_threads(threads)

def set_planner_effort(planner):
    """Set the default planning strategy."""
    SmartFFTW.set_default_planner(planner)

def get_stats():
    """Get statistics about the plan cache."""
    return SmartFFTW.get_stats()

def clear_cache(older_than=None):
    """Clear the plan cache."""
    SmartFFTW.clear_cache(older_than)