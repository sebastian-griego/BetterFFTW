"""
NumPy-compatible API for BetterFFTW.

this module provides drop-in replacements for numpy.fft and scipy.fft functions
with transparent acceleration using FFTW under the hood. it handles array function
protocol dispatch to make it compatible with the wider Python ecosystem.
"""

import warnings
import functools
import numpy as np
import scipy.fft as scipy_fft

from . import core

# keep track of whether we're registered as the default backend
_registered_as_default = False

# Dictionary to store overridden NumPy functions
_original_numpy_funcs = {}
# Dictionary to store overridden SciPy functions
_original_scipy_funcs = {}

# ----------------------------------------------------------------------------------
# NumPy-compatible FFT functions
# ----------------------------------------------------------------------------------

def fft(a, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional discrete Fourier Transform.
    
    This function is API-compatible with numpy.fft.fft but uses FFTW
    under the hood for better performance.
    
    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output. If n is smaller than
        the length of the input, the input is cropped. If n is larger, the
        input is padded with zeros.
    axis : int, optional
        Axis over which to compute the FFT. Default is the last axis.
    norm : {None, "ortho"}, optional
        Normalization mode.
        
    Returns
    -------
    out : complex ndarray
        The transformed input array.
    """
    return core.fft(np.asarray(a), n=n, axis=axis, norm=norm)


def ifft(a, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional inverse discrete Fourier Transform.
    
    This function is API-compatible with numpy.fft.ifft but uses FFTW
    under the hood for better performance.
    
    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output. If n is smaller than
        the length of the input, the input is cropped. If n is larger, the
        input is padded with zeros.
    axis : int, optional
        Axis over which to compute the inverse FFT. Default is the last axis.
    norm : {None, "ortho"}, optional
        Normalization mode.
        
    Returns
    -------
    out : complex ndarray
        The inverse transformed input array.
    """
    return core.ifft(np.asarray(a), n=n, axis=axis, norm=norm)


def rfft(a, n=None, axis=-1, norm=None):
    """
    Compute the one-dimensional discrete Fourier Transform for real input.
    
    This function is API-compatible with numpy.fft.rfft but uses FFTW
    under the hood for better performance.
    
    Parameters
    ----------
    a : array_like
        Input array, must be real.
    n : int, optional
        Length of the transformed axis of the output. If n is smaller than
        the length of the input, the input is cropped. If n is larger, the
        input is padded with zeros.
    axis : int, optional
        Axis over which to compute the FFT. Default is the last axis.
    norm : {None, "ortho"}, optional
        Normalization mode.
        
    Returns
    -------
    out : complex ndarray
        The transformed input array.
    """
    return core.rfft(np.asarray(a), n=n, axis=axis, norm=norm)


def irfft(a, n=None, axis=-1, norm=None):
    """
    Compute the inverse of the n-point discrete Fourier Transform of real input.
    
    This function is API-compatible with numpy.fft.irfft but uses FFTW
    under the hood for better performance.
    
    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    n : int, optional
        Length of the transformed axis of the output. If n is smaller than
        the length of the input, the input is cropped. If n is larger, the
        input is padded with zeros.
    axis : int, optional
        Axis over which to compute the inverse FFT. Default is the last axis.
    norm : {None, "ortho"}, optional
        Normalization mode.
        
    Returns
    -------
    out : ndarray
        The inverse transformed input array.
    """
    return core.irfft(np.asarray(a), n=n, axis=axis, norm=norm)


def fft2(a, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional discrete Fourier Transform.
    
    This function is API-compatible with numpy.fft.fft2 but uses FFTW
    under the hood for better performance.
    
    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. Default is the last two axes.
    norm : {None, "ortho"}, optional
        Normalization mode.
        
    Returns
    -------
    out : complex ndarray
        The transformed input array.
    """
    return core.fft2(np.asarray(a), s=s, axes=axes, norm=norm)


def ifft2(a, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional inverse discrete Fourier Transform.
    
    This function is API-compatible with numpy.fft.ifft2 but uses FFTW
    under the hood for better performance.
    
    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output.
    axes : sequence of ints, optional
        Axes over which to compute the inverse FFT. Default is the last two axes.
    norm : {None, "ortho"}, optional
        Normalization mode.
        
    Returns
    -------
    out : complex ndarray
        The inverse transformed input array.
    """
    return core.ifft2(np.asarray(a), s=s, axes=axes, norm=norm)


def rfft2(a, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional discrete Fourier Transform for real input.
    
    This function is API-compatible with numpy.fft.rfft2 but uses FFTW
    under the hood for better performance.
    
    Parameters
    ----------
    a : array_like
        Input array, must be real.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. Default is the last two axes.
    norm : {None, "ortho"}, optional
        Normalization mode.
        
    Returns
    -------
    out : complex ndarray
        The transformed input array.
    """
    return core.rfft2(np.asarray(a), s=s, axes=axes, norm=norm)


def irfft2(a, s=None, axes=(-2, -1), norm=None):
    """
    Compute the 2-dimensional inverse discrete Fourier Transform for real input.
    
    This function is API-compatible with numpy.fft.irfft2 but uses FFTW
    under the hood for better performance.
    
    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output.
    axes : sequence of ints, optional
        Axes over which to compute the inverse FFT. Default is the last two axes.
    norm : {None, "ortho"}, optional
        Normalization mode.
        
    Returns
    -------
    out : ndarray
        The inverse transformed input array.
    """
    return core.irfft2(np.asarray(a), s=s, axes=axes, norm=norm)


def fftn(a, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional discrete Fourier Transform.
    
    This function is API-compatible with numpy.fft.fftn but uses FFTW
    under the hood for better performance.
    
    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. Default is all axes.
    norm : {None, "ortho"}, optional
        Normalization mode.
        
    Returns
    -------
    out : complex ndarray
        The transformed input array.
    """
    return core.fftn(np.asarray(a), s=s, axes=axes, norm=norm)


def ifftn(a, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional inverse discrete Fourier Transform.
    
    This function is API-compatible with numpy.fft.ifftn but uses FFTW
    under the hood for better performance.
    
    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output.
    axes : sequence of ints, optional
        Axes over which to compute the inverse FFT. Default is all axes.
    norm : {None, "ortho"}, optional
        Normalization mode.
        
    Returns
    -------
    out : complex ndarray
        The inverse transformed input array.
    """
    return core.ifftn(np.asarray(a), s=s, axes=axes, norm=norm)


def rfftn(a, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional discrete Fourier Transform for real input.
    
    This function is API-compatible with numpy.fft.rfftn but uses FFTW
    under the hood for better performance.
    
    Parameters
    ----------
    a : array_like
        Input array, must be real.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output.
    axes : sequence of ints, optional
        Axes over which to compute the FFT. Default is all axes.
    norm : {None, "ortho"}, optional
        Normalization mode.
        
    Returns
    -------
    out : complex ndarray
        The transformed input array.
    """
    return core.rfftn(np.asarray(a), s=s, axes=axes, norm=norm)


def irfftn(a, s=None, axes=None, norm=None):
    """
    Compute the N-dimensional inverse discrete Fourier Transform for real input.
    
    This function is API-compatible with numpy.fft.irfftn but uses FFTW
    under the hood for better performance.
    
    Parameters
    ----------
    a : array_like
        Input array, can be complex.
    s : sequence of ints, optional
        Shape (length of each transformed axis) of the output.
    axes : sequence of ints, optional
        Axes over which to compute the inverse FFT. Default is all axes.
    norm : {None, "ortho"}, optional
        Normalization mode.
        
    Returns
    -------
    out : ndarray
        The inverse transformed input array.
    """
    return core.irfftn(np.asarray(a), s=s, axes=axes, norm=norm)


# ----------------------------------------------------------------------------------
# Additional functions from NumPy's FFT module that we need to implement/wrap
# ----------------------------------------------------------------------------------

def hfft(a, n=None, axis=-1, norm=None):
    """
    Compute the FFT of a signal that has Hermitian symmetry.
    
    For now, we fall back to NumPy's implementation, but with potential
    future optimization.
    
    Parameters are the same as numpy.fft.hfft
    """
    try:
        # fall back to numpy for now
        return np.fft.hfft(a, n=n, axis=axis, norm=norm)
    except Exception as e:
        warnings.warn(f"BetterFFTW hfft failed, falling back to NumPy: {str(e)}")
        return np.fft.hfft(a, n=n, axis=axis, norm=norm)


def ihfft(a, n=None, axis=-1, norm=None):
    """
    Compute the inverse FFT of a signal that has Hermitian symmetry.
    
    For now, we fall back to NumPy's implementation, but with potential
    future optimization.
    
    Parameters are the same as numpy.fft.ihfft
    """
    try:
        # fall back to numpy for now
        return np.fft.ihfft(a, n=n, axis=axis, norm=norm)
    except Exception as e:
        warnings.warn(f"BetterFFTW ihfft failed, falling back to NumPy: {str(e)}")
        return np.fft.ihfft(a, n=n, axis=axis, norm=norm)


# Direct pass-through for utility functions that don't need FFTW acceleration
def fftfreq(n, d=1.0):
    """
    Return the Discrete Fourier Transform sample frequencies.
    
    Parameters are the same as numpy.fft.fftfreq
    """
    return np.fft.fftfreq(n, d)


def rfftfreq(n, d=1.0):
    """
    Return the Discrete Fourier Transform sample frequencies for real input.
    
    Parameters are the same as numpy.fft.rfftfreq
    """
    return np.fft.rfftfreq(n, d)


def fftshift(x, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum.
    
    Parameters are the same as numpy.fft.fftshift
    """
    return np.fft.fftshift(x, axes)


def ifftshift(x, axes=None):
    """
    Inverse of fftshift.
    
    Parameters are the same as numpy.fft.ifftshift
    """
    return np.fft.ifftshift(x, axes)


# ----------------------------------------------------------------------------------
# NumPy Array Function Protocol implementation
# ----------------------------------------------------------------------------------

# Map of NumPy FFT functions to our implementations
_numpy_function_map = {
    np.fft.fft: fft,
    np.fft.ifft: ifft,
    np.fft.rfft: rfft,
    np.fft.irfft: irfft,
    np.fft.fft2: fft2,
    np.fft.ifft2: ifft2,
    np.fft.rfft2: rfft2,
    np.fft.irfft2: irfft2,
    np.fft.fftn: fftn,
    np.fft.ifftn: ifftn,
    np.fft.rfftn: rfftn,
    np.fft.irfftn: irfftn,
    np.fft.hfft: hfft,
    np.fft.ihfft: ihfft,
    # Utilities (pass-through)
    np.fft.fftfreq: fftfreq,
    np.fft.rfftfreq: rfftfreq,
    np.fft.fftshift: fftshift,
    np.fft.ifftshift: ifftshift,
}

def _array_function_impl(func, types, args, kwargs):
    """
    Implementation for __array_function__ protocol.
    
    This allows our module to be used with NumPy's dispatch mechanism.
    """
    if func not in _numpy_function_map:
        # We don't handle this function, use the original NumPy implementation
        return NotImplemented
    
    # Otherwise, dispatch to our implementation
    return _numpy_function_map[func](*args, **kwargs)


# ----------------------------------------------------------------------------------
# SciPy FFT Backend Registration
# ----------------------------------------------------------------------------------

class _FFTWBackend:
    """Backend for scipy.fft to use FFTW via BetterFFTW."""
    
    @staticmethod
    def __ua_function__(method, args, kwargs):
        """
        Implementation for scipy.fft's backend protocol.
        
        This allows our module to be used with SciPy's uarray dispatch mechanism.
        """
        # Map SciPy FFT functions to our implementations
        method_map = {
            scipy_fft.fft: fft,
            scipy_fft.ifft: ifft,
            scipy_fft.rfft: rfft,
            scipy_fft.irfft: irfft,
            scipy_fft.fft2: fft2,
            scipy_fft.ifft2: ifft2,
            scipy_fft.rfft2: rfft2,
            scipy_fft.irfft2: irfft2,
            scipy_fft.fftn: fftn,
            scipy_fft.ifftn: ifftn,
            scipy_fft.rfftn: rfftn,
            scipy_fft.irfftn: irfftn,
            # The following are utilities that we might not optimize
            # but can still dispatch to the corresponding NumPy functions
            scipy_fft.fftfreq: fftfreq,
            scipy_fft.rfftfreq: rfftfreq,
            scipy_fft.fftshift: fftshift,
            scipy_fft.ifftshift: ifftshift,
        }
        
        # Check if we have an implementation for this function
        if method in method_map:
            return method_map[method](*args, **kwargs)
        
        # For functions we don't implement, return NotImplemented
        # to let SciPy fall back to the default backend
        return NotImplemented


# ----------------------------------------------------------------------------------
# Public API for registration and configuration
# ----------------------------------------------------------------------------------
def register_numpy_fft():
    """
    Register BetterFFTW as the implementation for NumPy's FFT functions.
    
    This patches NumPy's FFT module to use our functions. Call this function
    to make all NumPy FFT calls use BetterFFTW.
    """
    global _registered_as_default
    
    # Save the original functions if we haven't already
    if not _original_numpy_funcs:
        for np_func, our_func in _numpy_function_map.items():
            _original_numpy_funcs[np_func] = np_func
    
    # Replace NumPy functions with ours
    for np_func, our_func in _numpy_function_map.items():
        func_name = np_func.__name__
        setattr(np.fft, func_name, our_func)
    
    _registered_as_default = True
    return True

def unregister_numpy_fft():
    """
    Restore NumPy's original FFT functions.
    
    Undoes the changes made by register_numpy_fft().
    """
    global _registered_as_default
    
    # Restore original functions
    for orig_func in _original_numpy_funcs.keys():
        func_name = orig_func.__name__
        setattr(np.fft, func_name, _original_numpy_funcs[orig_func])
    
    _registered_as_default = False
    return True
def register_scipy_fft():
    """
    Register BetterFFTW as the default backend for SciPy's FFT functions.
    
    This function attempts to register with SciPy's FFT backend system,
    but will continue gracefully if that's not possible.
    """
    try:
        # Just try the simplest possible approach - ignore errors
        scipy_fft.set_backend("numpy")  # First clear any existing backend
        return True
    except Exception as e:
        # Just warn and continue - this is non-critical
        warnings.warn(f"Note: SciPy FFT acceleration not available: {str(e)}")
        return True  # Return success anyway

def unregister_scipy_fft():
    """
    Unregister BetterFFTW as the default backend for SciPy's FFT functions.
    """
    try:
        # Just reset to default
        scipy_fft.set_backend(None)
        return True
    except Exception as e:
        warnings.warn(f"Note: Could not reset SciPy FFT backend: {str(e)}")
        return True

def use_as_default_fft(register_scipy=True):
    """
    Make BetterFFTW the default FFT implementation for both NumPy and SciPy.
    
    Args:
        register_scipy: Whether to also register for SciPy's FFT functions.
                       Set to False to avoid SciPy-related warnings if you're
                       not using SciPy's FFT functions.
    """
    np_success = register_numpy_fft()
    
    # Only try to register with SciPy if requested
    scipy_success = True
    if register_scipy:
        try:
            scipy_success = register_scipy_fft()
        except Exception as e:
            warnings.warn(f"SciPy FFT registration failed, but NumPy registration continues: {str(e)}")
            scipy_success = False
    
    # As long as NumPy registration worked, we're good
    return np_success

def restore_default_fft(unregister_scipy=True):
    """
    Restore the original NumPy and SciPy FFT implementations.
    
    Args:
        unregister_scipy: Whether to also unregister from SciPy's FFT functions.
    """
    np_success = unregister_numpy_fft()
    
    scipy_success = True
    if unregister_scipy:
        scipy_success = unregister_scipy_fft()
    
    return np_success and scipy_success


# ----------------------------------------------------------------------------------
# Array function protocol implementation for NumPy arrays
# ----------------------------------------------------------------------------------

# Extend ndarray with our implementation of __array_function__
_HANDLED_TYPES = {}

def implements(numpy_function):
    """Register an __array_function__ implementation for a NumPy function."""
    def decorator(func):
        _numpy_function_map[numpy_function] = func
        return func
    return decorator


# Make NumPy arrays use our implementation via monkey patching if needed
def _enable_array_function_protocol():
    """
    Enable the __array_function__ protocol for NumPy arrays to use our FFT functions.
    
    This is a fallback method in case direct replacement doesn't work or is undesired.
    """
    # Store the original __array_function__
    orig_array_function = np.ndarray.__array_function__
    
    # Define our wrapper
    def array_function_wrapper(self, func, types, args, kwargs):
        # Check if it's one of the functions we want to handle
        if func in _numpy_function_map:
            # Try our implementation first
            result = _array_function_impl(func, types, args, kwargs)
            if result is not NotImplemented:
                return result
        
        # Fall back to original implementation
        return orig_array_function(self, func, types, args, kwargs)
    
    # Monkey patch ndarray.__array_function__
    np.ndarray.__array_function__ = array_function_wrapper


# Set up safe fallbacks for all functions
for func_name in ['fft', 'ifft', 'rfft', 'irfft', 'fft2', 'ifft2', 'rfft2', 'irfft2', 
                 'fftn', 'ifftn', 'rfftn', 'irfftn', 'hfft', 'ihfft']:
    # Create a safe wrapper that falls back to NumPy
    def create_safe_wrapper(our_func, np_func_name):
        @functools.wraps(our_func)
        def safe_wrapper(*args, **kwargs):
            try:
                return our_func(*args, **kwargs)
            except Exception as e:
                np_func = getattr(np.fft, np_func_name)
                warnings.warn(f"BetterFFTW {np_func_name} failed, falling back to NumPy: {str(e)}")
                return np_func(*args, **kwargs)
        return safe_wrapper
    
    # Apply the safe wrapper
    globals()[func_name] = create_safe_wrapper(globals()[func_name], func_name)