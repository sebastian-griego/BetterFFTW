[![PyPI version](https://badge.fury.io/py/betterfftw.svg)](https://badge.fury.io/py/betterfftw)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# BetterFFTW

BetterFFTW is a NumPy-style wrapper around [pyFFTW](https://github.com/pyFFTW/pyFFTW).
It provides convenience functions for common FFT transforms, caches FFTW plans, and
can optionally register itself as the implementation behind `numpy.fft` calls.

## Installation

```bash
pip install betterfftw
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Usage

Call BetterFFTW directly when you want explicit control over FFTW planning:

```python
import numpy as np
import betterfftw as bfft

x = np.random.random(1024)
y = bfft.fft(x, threads=2, planner=bfft.PLANNER_ESTIMATE)
```

You can also register it as a drop-in implementation for NumPy FFT calls:

```python
import betterfftw

betterfftw.use_as_default(register_scipy=False)

import numpy as np

x = np.random.random(1024)
y = np.fft.fft(x)

betterfftw.restore_default()
```

For scoped use in tests, notebooks, or larger applications, prefer the context
manager so NumPy is restored automatically:

```python
import numpy as np
import betterfftw

with betterfftw.as_default(register_scipy=False):
    y = np.fft.fft(np.ones(1024))
```

## What It Optimizes

- Caches FFTW plans by transform type, shape, dtype, transform parameters, thread
  count, and planner strategy.
- Selects a default thread count and planner strategy based on transform shape.
- Can upgrade frequently reused plans to a more thorough FFTW planning strategy.
- Falls back to the original NumPy FFT implementation if a registered BetterFFTW
  wrapper fails at runtime.
- Provides helpers for FFTW wisdom import/export and aligned array allocation.

## Performance Notes

FFTW planning can be expensive. BetterFFTW is most useful when you repeat the same
transform configuration enough times to amortize planning cost. For one-off
transforms, NumPy may be as fast or faster because it avoids explicit FFTW plan
construction.

Use `betterfftw.benchmark_planners()` to compare FFTW planner strategies for a
specific workload:

```python
import numpy as np
import betterfftw

x = np.random.random(4096)
timings = betterfftw.benchmark_planners(
    x,
    repeats=10,
    strategies=[betterfftw.PLANNER_ESTIMATE, betterfftw.PLANNER_MEASURE],
)
```

Each result includes planning time, average execution time, and estimated total
time for the requested repeat count.

## Development

Run the test suite with:

```bash
python -m pytest
```

The GitHub Actions workflow runs the same suite on Python 3.10, 3.11, and 3.12.

## Credits

BetterFFTW builds on [pyFFTW](https://github.com/pyFFTW/pyFFTW) and the
[FFTW C library](http://www.fftw.org/).

## License

This project is licensed under GPL-3.0-or-later. See [LICENSE](LICENSE) for
details.
