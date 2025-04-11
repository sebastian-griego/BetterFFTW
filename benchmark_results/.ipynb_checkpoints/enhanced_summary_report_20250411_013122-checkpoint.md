# BetterFFTW Enhanced Benchmark Summary Report

Generated: 2025-04-11 01:31:22

## System Information

* **os**: posix
* **cpu_count**: 64
* **memory_gb**: 755.3984298706055
* **python_version**: 3.11.9 | packaged by conda-forge | (main, Apr 19 2024, 18:36:13) [GCC 12.3.0]
* **numpy_version**: 1.26.4
* **scipy_version**: 1.14.0
* **pyfftw_version**: 0.15.0
* **betterfftw_version**: 0.1.0
* **timestamp**: 2025-04-11 01:31:22
* **numpy_using_mkl**: False


## Direct Comparison Summary

### BetterFFTW vs NumPy FFT (1D transforms)

* **Average Speedup**: 1.95x (95% CI: 1.47x - 2.42x)
* **Median Speedup**: 1.23x
* **Range**: 0.20x - 7.72x
* **Standard Deviation**: 1.75

### BetterFFTW vs NumPy FFT (2D transforms)

* **Average Speedup**: 6.12x (95% CI: 4.84x - 7.40x)
* **Median Speedup**: 6.64x
* **Range**: 2.17x - 10.48x
* **Standard Deviation**: 2.78

### BetterFFTW vs NumPy FFT (3D transforms)

* **Average Speedup**: 12.19x (95% CI: 10.71x - 13.68x)
* **Median Speedup**: 11.75x
* **Range**: 9.63x - 15.75x
* **Standard Deviation**: 2.14

### PyFFTW vs NumPy FFT (1D transforms)

* **Average Speedup**: 1.25x (95% CI: 0.97x - 1.54x)
* **Median Speedup**: 0.99x
* **Range**: 0.28x - 4.36x
* **Standard Deviation**: 1.04

### PyFFTW vs NumPy FFT (2D transforms)

* **Average Speedup**: 1.48x (95% CI: 1.33x - 1.64x)
* **Median Speedup**: 1.54x
* **Range**: 0.84x - 2.03x
* **Standard Deviation**: 0.34

### PyFFTW vs NumPy FFT (3D transforms)

* **Average Speedup**: 1.08x (95% CI: 0.83x - 1.33x)
* **Median Speedup**: 1.15x
* **Range**: 0.44x - 1.47x
* **Standard Deviation**: 0.36

### SciPy FFT vs NumPy FFT (1D transforms)

* **Average Speedup**: 1.05x (95% CI: 0.88x - 1.21x)
* **Median Speedup**: 0.96x
* **Range**: 0.38x - 2.95x
* **Standard Deviation**: 0.61

### SciPy FFT vs NumPy FFT (2D transforms)

* **Average Speedup**: 1.36x (95% CI: 1.26x - 1.47x)
* **Median Speedup**: 1.34x
* **Range**: 0.98x - 1.73x
* **Standard Deviation**: 0.23

### SciPy FFT vs NumPy FFT (3D transforms)

* **Average Speedup**: 2.01x (95% CI: 1.55x - 2.47x)
* **Median Speedup**: 1.82x
* **Range**: 1.40x - 2.88x
* **Standard Deviation**: 0.66

### Power-of-2 vs Non-Power-of-2 Size Performance

#### BetterFFTW

* **1D transforms**: Non-power-of-2 sizes are 3.90x slower than power-of-2 sizes

* **2D transforms**: Non-power-of-2 sizes are 1.38x slower than power-of-2 sizes

* **3D transforms**: Non-power-of-2 sizes are 2.89x slower than power-of-2 sizes

#### NumPy FFT

* **1D transforms**: Non-power-of-2 sizes are 3.61x slower than power-of-2 sizes

* **2D transforms**: Non-power-of-2 sizes are 0.93x slower than power-of-2 sizes

* **3D transforms**: Non-power-of-2 sizes are 2.50x slower than power-of-2 sizes

#### PyFFTW

* **1D transforms**: Non-power-of-2 sizes are 3.66x slower than power-of-2 sizes

* **2D transforms**: Non-power-of-2 sizes are 1.00x slower than power-of-2 sizes

* **3D transforms**: Non-power-of-2 sizes are 1.38x slower than power-of-2 sizes

#### SciPy FFT

* **1D transforms**: Non-power-of-2 sizes are 2.41x slower than power-of-2 sizes

* **2D transforms**: Non-power-of-2 sizes are 1.01x slower than power-of-2 sizes

* **3D transforms**: Non-power-of-2 sizes are 2.41x slower than power-of-2 sizes

## Repeated Transform Summary

### Amortization Analysis

#### NumPy FFT

* **1D FFT**: 1 repetitions to reach 2x amortization
* **1D RFFT**: 1 repetitions to reach 2x amortization

#### SciPy FFT

* **1D FFT**: 1 repetitions to reach 2x amortization
* **1D RFFT**: 1 repetitions to reach 2x amortization
* **1D RFFT**: 100 repetitions to reach 5x amortization

#### PyFFTW

* **1D FFT**: 1 repetitions to reach 2x amortization
* **1D FFT (Large)**: 1 repetitions to reach 2x amortization
* **1D FFT (Very Large)**: 500 repetitions to reach 2x amortization
* **1D RFFT**: 1 repetitions to reach 2x amortization
* **1D RFFT (Large)**: 1 repetitions to reach 2x amortization
* **1D FFT**: 5 repetitions to reach 5x amortization
* **1D RFFT**: 5 repetitions to reach 5x amortization

#### BetterFFTW

* **1D FFT**: 1 repetitions to reach 2x amortization
* **1D FFT (Large)**: 1 repetitions to reach 2x amortization
* **1D FFT (Very Large)**: 1 repetitions to reach 2x amortization
* **2D FFT**: 1 repetitions to reach 2x amortization
* **1D RFFT**: 1 repetitions to reach 2x amortization
* **1D RFFT (Large)**: 1000 repetitions to reach 2x amortization
* **2D RFFT**: 1 repetitions to reach 2x amortization
* **3D FFT**: 1 repetitions to reach 2x amortization
* **1D FFT**: 5 repetitions to reach 5x amortization
* **1D FFT (Large)**: 1 repetitions to reach 5x amortization
* **1D FFT (Very Large)**: 1 repetitions to reach 5x amortization
* **2D FFT**: 1 repetitions to reach 5x amortization
* **1D RFFT**: 5 repetitions to reach 5x amortization
* **1D RFFT (Large)**: 1000 repetitions to reach 5x amortization
* **2D RFFT**: 1 repetitions to reach 5x amortization
* **3D FFT**: 1 repetitions to reach 5x amortization
* **1D FFT (Large)**: 1 repetitions to reach 10x amortization
* **1D FFT (Very Large)**: 1 repetitions to reach 10x amortization
* **2D FFT**: 1 repetitions to reach 10x amortization
* **2D RFFT**: 1 repetitions to reach 10x amortization
* **3D FFT**: 1 repetitions to reach 10x amortization

### Maximum Amortization Factors

Maximum amortization factors at 1000 repetitions:

```
transform         1D FFT  1D FFT (Large)  1D FFT (Very Large)   1D RFFT  1D RFFT (Large)     2D FFT  2D FFT (Large)    2D RFFT     3D FFT  3D FFT (Large)
implementation                                                                                                                                           
BetterFFTW      6.227634       18.240950            17.543508  6.186085         8.792947  13.676288        1.078153  15.164221  11.467616        1.252612
NumPy FFT       2.737222        1.524529             1.073426  4.045041         1.419848   1.078905        1.009905   1.202913   1.050592        1.001213
PyFFTW          7.760273        2.798213             2.634448  8.774059         2.416510   1.707352        1.526605   1.569051   1.649468        1.478181
SciPy FFT       4.569830        1.678916             1.103775  6.025573         1.491184   1.298226        1.003593   1.186100   1.253072        0.999503
```

## Thread Scaling Summary

### Parallel Efficiency Analysis

#### BetterFFTW

* **2D FFT**: 1.00x speedup with 64 threads (0.02 efficiency)
* **2D FFT (Large)**: 1.00x speedup with 64 threads (0.02 efficiency)
* **2D FFT (Very Large)**: 1.00x speedup with 64 threads (0.02 efficiency)
* **3D FFT**: 1.01x speedup with 64 threads (0.02 efficiency)
* **3D FFT (Large)**: 1.00x speedup with 64 threads (0.02 efficiency)

Scaling characteristics:

* **2D FFT**: Scaling slope: 0.000 (R²: 0.151), Est. parallel portion: 0.4%
* **2D FFT (Large)**: Scaling slope: 0.000 (R²: 0.027), Est. parallel portion: 0.3%
* **2D FFT (Very Large)**: Scaling slope: 0.000 (R²: 0.001), Est. parallel portion: 0.1%
* **3D FFT**: Scaling slope: 0.000 (R²: 0.008), Est. parallel portion: 0.9%
* **3D FFT (Large)**: Scaling slope: -0.000 (R²: 0.464), Est. parallel portion: 0.0%

#### PyFFTW

* **2D FFT**: 4.59x speedup with 64 threads (0.07 efficiency)
* **2D FFT (Large)**: 4.25x speedup with 64 threads (0.07 efficiency)
* **2D FFT (Very Large)**: 2.62x speedup with 64 threads (0.04 efficiency)
* **3D FFT**: 3.23x speedup with 64 threads (0.05 efficiency)
* **3D FFT (Large)**: 5.81x speedup with 64 threads (0.09 efficiency)

Scaling characteristics:

* **2D FFT**: Scaling slope: 0.038 (R²: 0.341), Est. parallel portion: 79.4%
* **2D FFT (Large)**: Scaling slope: 0.046 (R²: 0.625), Est. parallel portion: 77.7%
* **2D FFT (Very Large)**: Scaling slope: 0.017 (R²: 0.422), Est. parallel portion: 62.8%
* **3D FFT**: Scaling slope: 0.028 (R²: 0.423), Est. parallel portion: 70.1%
* **3D FFT (Large)**: Scaling slope: 0.062 (R²: 0.375), Est. parallel portion: 84.1%

## Memory Efficiency Summary

### Memory Efficiency Ranking

```
implementation  avg_peak_memory_delta_mb  throughput_mbs  memory_efficiency
     SciPy FFT               1791.952865      352.566098           0.301795
        PyFFTW               5375.998698      200.036396           0.079903
     NumPy FFT               4778.617448      260.358545           0.075072
    BetterFFTW               -597.327083      935.800756           0.000000
```

**SciPy FFT** is the most memory-efficient implementation, achieving the highest throughput per MB of memory used.

### Comparison to NumPy

* **SciPy FFT**: Uses 0.37x the memory of NumPy, with 1.35x the throughput
* **PyFFTW**: Uses 1.13x the memory of NumPy, with 0.77x the throughput
* **BetterFFTW**: Uses -0.12x the memory of NumPy, with 3.59x the throughput

## Non-Power-of-2 Size Performance Summary

### Performance Relative to NumPy on Non-Power-of-2 Sizes

```
+------------------+-------------+-------------+----------------+------------------+-------------------+
| Implementation   |   Dimension | Transform   | Mean Speedup   | Median Speedup   | Min-Max Speedup   |
+==================+=============+=============+================+==================+===================+
| PyFFTW           |           1 | FFT         | 1.12x          | 0.91x            | 0.34x - 3.22x     |
+------------------+-------------+-------------+----------------+------------------+-------------------+
| BetterFFTW       |           1 | FFT         | 1.00x          | 0.91x            | 0.44x - 1.92x     |
+------------------+-------------+-------------+----------------+------------------+-------------------+
| SciPy FFT        |           1 | FFT         | 1.00x          | 1.05x            | 0.53x - 1.70x     |
+------------------+-------------+-------------+----------------+------------------+-------------------+
| BetterFFTW       |           1 | Real FFT    | 1.63x          | 1.44x            | 0.87x - 3.23x     |
+------------------+-------------+-------------+----------------+------------------+-------------------+
| PyFFTW           |           1 | Real FFT    | 1.03x          | 0.84x            | 0.27x - 3.16x     |
+------------------+-------------+-------------+----------------+------------------+-------------------+
| SciPy FFT        |           1 | Real FFT    | 0.93x          | 0.91x            | 0.50x - 1.65x     |
+------------------+-------------+-------------+----------------+------------------+-------------------+
```

### Performance on Prime Number Sizes

* **BetterFFTW**: 1.78x faster than NumPy on prime number sizes

* **PyFFTW**: 1.94x faster than NumPy on prime number sizes

* **SciPy FFT**: 1.40x faster than NumPy on prime number sizes

## Overall Analysis and Recommendations

### Best Performing Implementations

* **General Usage**: BetterFFTW (3.96x faster than NumPy on average)

* **Repeated Transforms**: BetterFFTW (9.96x amortization factor)

* **Multi-Threading**: PyFFTW (4.10x speedup with maximum threads)

### Recommendations

Based on the benchmarks, **BetterFFTW** appears to be the best overall FFT implementation for Python, winning in 1 out of 1 categories.

#### Specific Use Case Recommendations:

**For casual/general use:**

- BetterFFTW is recommended (average 3.96x speedup)

**For performance-critical applications:**

- BetterFFTW provides the highest performance (3.96x speedup)

**For applications with repeated FFT operations:**

- BetterFFTW provides excellent plan reuse (9.96x speedup for repeated transforms)

**For applications with non-power-of-2 sizes:**

**For multi-threaded applications:**

- PyFFTW provides the best multi-threading scaling (4.10x speedup)
