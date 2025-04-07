[![PyPI version](https://badge.fury.io/py/betterfftw.svg)](https://badge.fury.io/py/betterfftw)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# BetterFFTW

A high-performance wrapper around pyFFTW with optimized defaults and automatic performance tuning.

## About

BetterFFTW is a drop‑in replacement for NumPy and SciPy FFT routines built on top of [pyFFTW](https://github.com/pyFFTW/pyFFTW). It automatically configures optimal performance settings based on your array’s characteristics—making it significantly faster than NumPy or SciPy in most use cases, especially when handling large sizes and multi‑dimensional data.

## Key Improvements

- **Auto-tuned Threading:**  
  Dynamically selects the optimal number of threads based on array size and dimensionality, ensuring minimal overhead on small arrays and maximum speed on large ones.

- **Progressive Optimization:**  
  Automatically upgrades the planning strategy (from fast ESTIMATE to more thorough MEASURE) for transforms that are repeated or computationally intensive.

- **Enhanced Non-Power-of-2 Handling:**  
  Special strategies are applied for arrays with non‑power‑of‑2 dimensions—areas where standard FFT implementations can struggle.

- **Performance-Based Fallback:**  
  Smartly detects cases where NumPy’s FFT might be faster and falls back accordingly, so you always benefit from the best available performance.

## Installation

Install via pip:

```bash
pip install betterfftw
```

## Usage

BetterFFTW is designed to work seamlessly as the default FFT backend. Simply register it and enjoy the speed boost:

```python
import betterfftw

# Register BetterFFTW as the default FFT implementation
betterfftw.use_as_default()

import numpy as np
x = np.random.random(1024)
y = np.fft.fft(x)  # Uses BetterFFTW under the hood for superior performance!
```

For specialized use cases, you can access explicit controls, but in most scenarios, the default settings are optimized for best performance.

## Performance

BetterFFTW leverages advanced heuristics to optimize FFT performance:
- **Faster Transforms:**  
  Benchmarks show that BetterFFTW is a lot faster than the default NumPy or SciPy FFT routines, particularly for large arrays and high-dimensional transforms.
- **Adaptive Tuning:**  
  It automatically adapts planning strategies and thread counts based on your specific workload and system capabilities.
- **Efficient Caching:**  
  FFT plans are cached and intelligently upgraded, cutting down on redundant computation and ensuring continuous performance gains with repeated use.

## Credits

BetterFFTW builds on [pyFFTW](https://github.com/pyFFTW/pyFFTW) and the [FFTW C library](http://www.fftw.org/). The project leverages these proven technologies, focusing on unlocking even greater performance through smart configuration and automatic tuning.

## License

This project is licensed under the GPL-3.0 License—see the LICENSE file for details.