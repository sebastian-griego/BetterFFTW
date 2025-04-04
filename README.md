# BetterFFTW

A high-performance, user-friendly wrapper for FFTW in Python, optimized for academic and research use.

## About

BetterFFTW builds on the excellent [PyFFTW](https://github.com/pyFFTW/pyFFTW) library, providing a more intuitive interface with smart defaults. This library aims to make the power of FFTW more accessible to Python users while addressing some of the usability challenges of PyFFTW.

## Features

- **Drop-in replacement** for NumPy and SciPy FFT functions
- **Significantly faster** for multi-dimensional transforms (4-6x for 2D, 6-14x for 3D transforms)
- **Automatic multi-threading** that intelligently scales with array size
- **Smart plan caching** with progressive optimization
- **Seamless integration** with the Python scientific ecosystem

## Strengths and Limitations

### Strengths
- **Multi-dimensional transforms**: 96-100% win rate vs. NumPy/SciPy/PyFFTW for 2D and 3D FFTs
- **Non-power-of-2 sizes**: 62% win rate vs. other implementations
- **Thread scaling**: Effective parallelization for large arrays

### Limitations
- **Higher memory usage**: Uses approximately 2x more memory than SciPy FFT in some cases
- **First-transform overhead**: For one-off small 1D FFTs, NumPy or PyFFTW may be faster
- **Memory allocation strategy**: When creating new arrays for each transform, sometimes less efficient than PyFFTW's approach

## Installation

```bash
pip install betterfftw
```

### Prerequisites

BetterFFTW depends on PyFFTW, which requires the FFTW C library:

- **Linux**: `sudo apt-get install libfftw3-dev` (Ubuntu/Debian) or `sudo yum install fftw-devel` (Fedora/RHEL)
- **macOS**: `brew install fftw` (using Homebrew)
- **Windows**: Pre-compiled binaries available via conda: `conda install -c conda-forge pyfftw`

## Usage

Basic usage as a drop-in replacement:

```python
import betterfftw

# Make all NumPy and SciPy FFT calls use BetterFFTW
betterfftw.use_as_default()

# Now standard NumPy code will use FFTW under the hood
import numpy as np
x = np.random.random(1024)
y = np.fft.fft(x)  # Uses BetterFFTW automatically!
```

Advanced usage with explicit control:

```python
import betterfftw as bfft
import numpy as np

# Create input data
x = np.random.random((1024, 1024))

# Perform 2D FFT with specific parameters
y = bfft.fft2(x, threads=4, planner='FFTW_PATIENT')
```

## When to Use BetterFFTW

BetterFFTW is most beneficial for:
- Multi-dimensional transforms (2D images, 3D volumes)
- Repeatedly performing the same transform in loops
- Non-power-of-2 sized arrays
- Batch processing of large datasets

Consider sticking with NumPy/SciPy for:
- Simple one-off 1D transforms
- Memory-constrained environments
- Cases where simplicity is valued over maximum performance

## Credits

BetterFFTW is built on top of [PyFFTW](https://github.com/pyFFTW/pyFFTW), which provides Python bindings to the [FFTW C library](http://www.fftw.org/). I am grateful to the maintainers of these excellent projects.

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.

[![PyPI version](https://badge.fury.io/py/betterfftw.svg)](https://badge.fury.io/py/betterfftw)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)