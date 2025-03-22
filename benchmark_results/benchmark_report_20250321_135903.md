# FFT BENCHMARK REPORT
Generated: 2025-03-21 13:59:03

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
Best overall implementation: SciPy (Mean speedup: 2.17x)

## Implementation Summary
### BetterFFTW
Success rate: 83.7% (82/98)
Mean speedup: 0.54x
Median speedup: 0.38x
Min speedup: 0.00x
Max speedup: 3.09x
Std. dev.: 0.73

#### Category Performance
**1D_pow2**:
Mean speedup: 0.30x, Range: 0.00x - 1.33x
**1D_nonpow2**:
Mean speedup: 0.43x, Range: 0.01x - 2.08x
**2D_pow2**:
Mean speedup: 0.85x, Range: 0.10x - 3.09x
**2D_nonpow2**:
Mean speedup: 0.37x, Range: 0.02x - 0.59x
**3D_pow2**:
Mean speedup: 0.88x, Range: 0.74x - 0.98x
**3D_nonpow2**:
Mean speedup: 0.34x, Range: 0.21x - 0.44x
**mixed_workload**:
Mean speedup: 0.60x, Range: 0.60x - 0.60x

### NumPy (Reference)
Success rate: 48.0% (47/98)

#### Category Performance
**1D_pow2**:
**1D_nonpow2**:
**2D_pow2**:
**2D_nonpow2**:
**3D_pow2**:
**3D_nonpow2**:
**mixed_workload**:

### SciPy
Success rate: 91.8% (90/98)
Mean speedup: 2.17x
Median speedup: 1.54x
Min speedup: 0.62x
Max speedup: 12.91x
Std. dev.: 2.04

#### Category Performance
**1D_pow2**:
Mean speedup: 1.38x, Range: 0.62x - 2.46x
**1D_nonpow2**:
Mean speedup: 2.25x, Range: 1.33x - 4.13x
**2D_pow2**:
Mean speedup: 2.64x, Range: 1.03x - 12.91x
**2D_nonpow2**:
Mean speedup: 2.61x, Range: 1.10x - 6.44x
**3D_pow2**:
Mean speedup: 1.88x, Range: 1.36x - 2.53x
**3D_nonpow2**:
Mean speedup: 2.04x, Range: 1.20x - 3.42x
**mixed_workload**:
Mean speedup: 1.00x, Range: 1.00x - 1.00x

## Category Summary
### 1D_pow2
Total benchmarks: 42

**BetterFFTW**:
Mean speedup: 0.30x, Median: 0.02x, Range: 0.00x - 1.33x
**SciPy**:
Mean speedup: 1.38x, Median: 1.26x, Range: 0.62x - 2.46x

### 1D_nonpow2
Total benchmarks: 20

**BetterFFTW**:
Mean speedup: 0.43x, Median: 0.01x, Range: 0.01x - 2.08x
**SciPy**:
Mean speedup: 2.25x, Median: 1.93x, Range: 1.33x - 4.13x

### 2D_pow2
Total benchmarks: 18

**BetterFFTW**:
Mean speedup: 0.85x, Median: 0.40x, Range: 0.10x - 3.09x
**SciPy**:
Mean speedup: 2.64x, Median: 1.38x, Range: 1.03x - 12.91x

### 2D_nonpow2
Total benchmarks: 7

**BetterFFTW**:
Mean speedup: 0.37x, Median: 0.43x, Range: 0.02x - 0.59x
**SciPy**:
Mean speedup: 2.61x, Median: 2.06x, Range: 1.10x - 6.44x

### 3D_pow2
Total benchmarks: 4

**BetterFFTW**:
Mean speedup: 0.88x, Median: 0.93x, Range: 0.74x - 0.98x
**SciPy**:
Mean speedup: 1.88x, Median: 1.82x, Range: 1.36x - 2.53x

### 3D_nonpow2
Total benchmarks: 4

**BetterFFTW**:
Mean speedup: 0.34x, Median: 0.38x, Range: 0.21x - 0.44x
**SciPy**:
Mean speedup: 2.04x, Median: 1.76x, Range: 1.20x - 3.42x

### mixed_workload
Total benchmarks: 1

**BetterFFTW**:
Mean speedup: 0.60x, Median: 0.60x, Range: 0.60x - 0.60x
**SciPy**:
Mean speedup: 1.00x, Median: 1.00x, Range: 1.00x - 1.00x

## Key Insights
- For 1D_pow2 transforms, SciPy is the fastest with 1.38x mean speedup over NumPy
- For 1D_nonpow2 transforms, SciPy is the fastest with 2.25x mean speedup over NumPy
- For 2D_pow2 transforms, SciPy is the fastest with 2.64x mean speedup over NumPy
- For 2D_nonpow2 transforms, SciPy is the fastest with 2.61x mean speedup over NumPy
- For 3D_pow2 transforms, SciPy is the fastest with 1.88x mean speedup over NumPy
- For 3D_nonpow2 transforms, SciPy is the fastest with 2.04x mean speedup over NumPy
- For mixed_workload transforms, SciPy is the fastest with 1.00x mean speedup over NumPy

### Top 5 Speedups
1. SciPy on rfft2 on 512x256 <class 'float64'> array (power-of-2): 12.91x speedup
2. SciPy on irfft2 on 96x96 <class 'float64'> array (non-power-of-2): 6.44x speedup
3. SciPy on ifft on 3072 <class 'complex128'> array (non-power-of-2): 4.13x speedup
4. SciPy on irfftn on 48x48x48 <class 'float64'> array (non-power-of-2): 3.42x speedup
5. BetterFFTW on ifft2 on 512x512 <class 'complex128'> array (power-of-2): 3.09x speedup

### Significant Slowdowns
1. BetterFFTW on irfft on 4096 <class 'float32'> array (power-of-2): 0.00x speedup (slowdown)
2. BetterFFTW on irfft on 4096 <class 'float64'> array (power-of-2): 0.01x speedup (slowdown)
3. BetterFFTW on irfft on 768 <class 'float64'> array (non-power-of-2): 0.01x speedup (slowdown)
4. BetterFFTW on rfft on 4096 <class 'float64'> array (power-of-2): 0.01x speedup (slowdown)
5. BetterFFTW on ifft on 3072 <class 'float64'> array (non-power-of-2): 0.01x speedup (slowdown)