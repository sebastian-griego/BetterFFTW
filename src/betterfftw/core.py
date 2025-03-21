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
import pyfftw
import atexit
from typing import Dict, Tuple, Optional, Union, Any, List, Callable

# enable the PyFFTW cache
pyfftw.interfaces.cache.enable()

# configuration constants with smart defaults
DEFAULT_THREADS = min(multiprocessing.cpu_count(), 4)  # reasonable default
DEFAULT_PLANNER = 'FFTW_ESTIMATE'  # fast planning initially - changed from MEASURE
MEASURE_PLANNER = 'FFTW_MEASURE'  # thorough planning for repeated use
PATIENCE_PLANNER = 'FFTW_PATIENT'  # thorough planning for critical performance
DEFAULT_CACHE_TIMEOUT = 60  # seconds to keep plans in cache

# thresholds for auto-configuration
SIZE_THREADING_THRESHOLD = 1024 * 1024  # When to use multi-threading - increased from 32768
MIN_REPEAT_FOR_MEASURE = 10  # number of uses before upgrading plan - increased from 3
AUTO_ALIGN = True  # auto-align arrays for FFTW

# we'll use a timer to periodically clean the cache
_cache_cleaning_interval = 300  # 5 minutes
_cache_lock = threading.RLock()  # prevent race conditions
_optimization_queue = []  # queue for background optimizations
_optimization_lock = threading.RLock()  # lock for the optimization queue
_background_optimizing = False  # flag for background optimization

# wisdom file for persistent planning
WISDOM_FILE = os.path.expanduser("~/.betterfftw_wisdom")


def _exit_handler():
    """Save wisdom and clean up resources when exiting."""
    try:
        SmartFFTW.export_wisdom(WISDOM_FILE)
    except Exception:
        pass  # don't crash if we can't save wisdom


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
            now = time.time()
            keys_to_remove = []
            
            if older_than is None:
                # clear everything
                cls._plan_cache.clear()
                cls._call_count.clear()
                cls._last_used.clear()
                cls._plan_quality.clear()
                return
                
            # selectively clear based on age
            for key, last_used in cls._last_used.items():
                if now - last_used > older_than:
                    keys_to_remove.append(key)
                    
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
    def _select_threads(cls, array: np.ndarray) -> int:
        """
        Auto-select optimal thread count based on array size.
        
        small arrays use 1 thread to avoid overhead, large arrays use multiple.
        """
        size = np.prod(array.shape)
        if size < SIZE_THREADING_THRESHOLD:
            return 1
        return DEFAULT_THREADS
    
    @classmethod
    def _should_upgrade_plan(cls, key: Tuple) -> bool:
        """Determine if we should upgrade to a more thorough planning strategy."""
        # if we've called this size multiple times, it's worth optimizing
        count = cls._call_count.get(key, 0)
        current_quality = cls._plan_quality.get(key, DEFAULT_PLANNER)
        
        if count >= MIN_REPEAT_FOR_MEASURE and current_quality == DEFAULT_PLANNER:
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
        
        # Check the function name to determine correct parameter names
        func_name = builder_func.__name__ if hasattr(builder_func, '__name__') else str(builder_func)
        
        # Different parameter names for different FFT types
        if 'fft2' in func_name or 'fftn' in func_name:
            # 2D and ND transforms use 's' instead of 'n' and 'axes' instead of 'axis'
            fft_obj = builder_func(
                array,
                s=n,  # 's' is used instead of 'n'
                axes=axis,  # 'axes' is used instead of 'axis'
                overwrite_input=False,
                threads=threads,
                planner_effort=planner,
                auto_align_input=AUTO_ALIGN,
                auto_contiguous=True,
                **kwargs
            )
        else:
            # Regular 1D transforms
            fft_obj = builder_func(
                array,
                n=n,
                axis=axis,
                overwrite_input=False,
                threads=threads,
                planner_effort=planner,
                auto_align_input=AUTO_ALIGN,
                auto_contiguous=True,
                **kwargs
            )
        
        return fft_obj
    
    @classmethod
    def _schedule_optimization(cls, key: Tuple, array: np.ndarray, builder_func: Callable,
                             n: Optional[Union[int, Tuple[int, ...]]] = None,
                             axis: Union[int, Tuple[int, ...]] = -1,
                             norm: Optional[str] = None,
                             **kwargs):
        """Schedule a plan for optimization in the background."""
        with _optimization_lock:
            # add to optimization queue if not already there
            for item in _optimization_queue:
                if item[0] == key:
                    return  # already scheduled
            
            # make a copy of the array to avoid holding references
            array_copy = np.array(array, copy=True)
            
            # add to queue
            _optimization_queue.append((key, array_copy, builder_func, n, axis, norm, kwargs))
            
            # start optimization thread if not running
            global _background_optimizing
            if not _background_optimizing:
                _background_optimizing = True
                threading.Thread(target=cls._background_optimize, daemon=True).start()
    
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
        - Reuses cached plans for the same array shape
        - Selects optimal thread count based on array size
        - Progressively optimizes plans for frequently used shapes
        
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
        # generate cache key for this transform
        key = cls._get_cache_key(array, n, axis, norm, 'fft')
        
        with _cache_lock:
            # track call frequency for this shape
            cls._call_count[key] = cls._call_count.get(key, 0) + 1
            cls._last_used[key] = time.time()
            
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
            
            # Execute the transform
            result = fft_obj(array)
            
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
            
            return result
    
    @classmethod
    def import_wisdom(cls, filename: str = None) -> bool:
        """
        Import FFTW wisdom from a file or default location.
        
        Args:
            filename: Path to wisdom file (None = use default path)
            
        Returns:
            True if wisdom was successfully imported, False otherwise
        """
        if filename is None:
            filename = WISDOM_FILE
            
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    wisdom = f.read()
                return pyfftw.import_wisdom(wisdom)
            return False
        except Exception:
            return False
    
    @classmethod
    def export_wisdom(cls, filename: str = None) -> bool:
        """
        Export FFTW wisdom to a file or default location.
        
        Args:
            filename: Path to wisdom file (None = use default path)
            
        Returns:
            True if wisdom was successfully exported, False otherwise
        """
        if filename is None:
            filename = WISDOM_FILE
            
        try:
            wisdom = pyfftw.export_wisdom()
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'wb') as f:
                f.write(wisdom[0])
            return True
        except Exception:
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
    def set_threading_threshold(cls, size: int):
        """Set the array size threshold for using multiple threads."""
        global SIZE_THREADING_THRESHOLD
        SIZE_THREADING_THRESHOLD = size
        
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
# Utility functions
def empty_aligned(shape, dtype=np.complex128, n=None):
    """Create an empty aligned array for optimal FFTW performance."""
    return pyfftw.empty_aligned(shape, dtype, n)

def empty_aligned_like(array, n=None):
    """Create an empty aligned array like another array."""
    return pyfftw.empty_aligned(array.shape, array.dtype, n)

def byte_align(array, n=None, copy=True):
    """Align an existing array to memory boundary for optimal FFTW performance."""
    return pyfftw.byte_align(array, n, copy)

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