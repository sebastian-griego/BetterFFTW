# FFT BENCHMARK REPORT
Generated: 2025-03-21 10:45:16

## System Information
Platform: Windows-10-10.0.26100-SP0
CPU: Intel64 Family 6 Model 186 Stepping 3, GenuineIntel
CPU Count: 12 (Physical cores: 10)
Memory: 15.69 GB
Python: 3.11.9
NumPy: 1.26.3
SciPy: 1.13.1
BetterFFTW: 0.1.0

## Overall Summary
Total benchmarks: 98
Total implementations: 3
Best overall implementation: SciPy (Mean speedup: 1.65x)

## Implementation Summary
### BetterFFTW
Success rate: 63.3% (62/98)
Mean speedup: 0.51x
Median speedup: 0.37x
Min speedup: 0.01x
Max speedup: 2.43x
Std. dev.: 0.60

#### Category Performance
**1D_pow2**:
Mean speedup: 0.03x, Range: 0.03x - 0.03x
**1D_nonpow2**:
**2D_pow2**:
Mean speedup: 0.66x, Range: 0.11x - 2.43x
**2D_nonpow2**:
Mean speedup: 0.29x, Range: 0.01x - 0.51x
**3D_pow2**:
Mean speedup: 0.86x, Range: 0.65x - 1.02x
**3D_nonpow2**:
Mean speedup: 0.22x, Range: 0.14x - 0.35x
**mixed_workload**:

### NumPy (Reference)
Success rate: 28.6% (28/98)

#### Category Performance
**1D_pow2**:
**1D_nonpow2**:
**2D_pow2**:
**2D_nonpow2**:
**3D_pow2**:
**3D_nonpow2**:
**mixed_workload**:

### SciPy
Success rate: 26.5% (26/98)
Mean speedup: 1.65x
Median speedup: 1.44x
Min speedup: 0.93x
Max speedup: 3.19x
Std. dev.: 0.64

#### Category Performance
**1D_pow2**:
**1D_nonpow2**:
**2D_pow2**:
Mean speedup: 1.49x, Range: 0.93x - 2.61x
**2D_nonpow2**:
Mean speedup: 1.57x, Range: 1.04x - 2.33x
**3D_pow2**:
Mean speedup: 1.57x, Range: 1.15x - 1.95x
**3D_nonpow2**:
Mean speedup: 2.45x, Range: 1.50x - 3.19x
**mixed_workload**:

## Category Summary
### 1D_pow2
Total benchmarks: 42

**BetterFFTW**:
Mean speedup: 0.03x, Median: 0.03x, Range: 0.03x - 0.03x
**SciPy**:

### 1D_nonpow2
Total benchmarks: 20

**BetterFFTW**:
**SciPy**:

### 2D_pow2
Total benchmarks: 18

**BetterFFTW**:
Mean speedup: 0.66x, Median: 0.37x, Range: 0.11x - 2.43x
**SciPy**:
Mean speedup: 1.49x, Median: 1.27x, Range: 0.93x - 2.61x

### 2D_nonpow2
Total benchmarks: 7

**BetterFFTW**:
Mean speedup: 0.29x, Median: 0.39x, Range: 0.01x - 0.51x
**SciPy**:
Mean speedup: 1.57x, Median: 1.35x, Range: 1.04x - 2.33x

### 3D_pow2
Total benchmarks: 4

**BetterFFTW**:
Mean speedup: 0.86x, Median: 0.90x, Range: 0.65x - 1.02x
**SciPy**:
Mean speedup: 1.57x, Median: 1.59x, Range: 1.15x - 1.95x

### 3D_nonpow2
Total benchmarks: 4

**BetterFFTW**:
Mean speedup: 0.22x, Median: 0.16x, Range: 0.14x - 0.35x
**SciPy**:
Mean speedup: 2.45x, Median: 2.65x, Range: 1.50x - 3.19x

### mixed_workload
Total benchmarks: 1

**BetterFFTW**:
**SciPy**:

## Key Insights
- For 1D_pow2 transforms, BetterFFTW is the fastest with 0.03x mean speedup over NumPy
- For 2D_pow2 transforms, SciPy is the fastest with 1.49x mean speedup over NumPy
- For 2D_nonpow2 transforms, SciPy is the fastest with 1.57x mean speedup over NumPy
- For 3D_pow2 transforms, SciPy is the fastest with 1.57x mean speedup over NumPy
- For 3D_nonpow2 transforms, SciPy is the fastest with 2.45x mean speedup over NumPy

### Top 5 Speedups
1. SciPy on ifftn on 48x48x48 <class 'float64'> array (non-power-of-2): 3.19x speedup
2. SciPy on irfftn on 48x48x48 <class 'float64'> array (non-power-of-2): 2.65x speedup
3. SciPy on irfft2 on 512x512 <class 'float64'> array (power-of-2): 2.61x speedup
4. SciPy on rfft2 on 512x512 <class 'float64'> array (power-of-2): 2.52x speedup
5. BetterFFTW on ifft2 on 512x512 <class 'complex128'> array (power-of-2): 2.43x speedup

### Significant Slowdowns
1. BetterFFTW on fft2 on 96x96 <class 'float64'> array (non-power-of-2): 0.01x speedup (slowdown)
2. BetterFFTW on rfft2 on 96x96 <class 'float64'> array (non-power-of-2): 0.01x speedup (slowdown)
3. BetterFFTW on fft on 1024 <class 'float64'> array (power-of-2): 0.03x speedup (slowdown)
4. BetterFFTW on fft2 on 64x64 <class 'float64'> array (power-of-2): 0.11x speedup (slowdown)
5. BetterFFTW on rfftn on 48x48x48 <class 'float64'> array (non-power-of-2): 0.14x speedup (slowdown)