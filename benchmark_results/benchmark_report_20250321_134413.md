# FFT BENCHMARK REPORT
Generated: 2025-03-21 13:44:13

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
Best overall implementation: SciPy (Mean speedup: 1.82x)

## Implementation Summary
### BetterFFTW
Success rate: 72.4% (71/98)
Mean speedup: 0.45x
Median speedup: 0.26x
Min speedup: 0.00x
Max speedup: 3.01x
Std. dev.: 0.64

#### Category Performance
**1D_pow2**:
Mean speedup: 0.16x, Range: 0.00x - 1.31x
**1D_nonpow2**:
Mean speedup: 0.01x, Range: 0.01x - 0.01x
**2D_pow2**:
Mean speedup: 0.78x, Range: 0.13x - 3.01x
**2D_nonpow2**:
Mean speedup: 0.26x, Range: 0.01x - 0.76x
**3D_pow2**:
Mean speedup: 0.85x, Range: 0.75x - 0.97x
**3D_nonpow2**:
Mean speedup: 0.89x, Range: 0.40x - 1.72x
**mixed_workload**:

### NumPy (Reference)
Success rate: 45.9% (45/98)

#### Category Performance
**1D_pow2**:
**1D_nonpow2**:
**2D_pow2**:
**2D_nonpow2**:
**3D_pow2**:
**3D_nonpow2**:
**mixed_workload**:

### SciPy
Success rate: 44.9% (44/98)
Mean speedup: 1.82x
Median speedup: 1.60x
Min speedup: 0.75x
Max speedup: 5.49x
Std. dev.: 0.96

#### Category Performance
**1D_pow2**:
Mean speedup: 1.87x, Range: 0.75x - 5.49x
**1D_nonpow2**:
Mean speedup: 2.41x, Range: 1.55x - 3.27x
**2D_pow2**:
Mean speedup: 1.47x, Range: 0.99x - 2.17x
**2D_nonpow2**:
Mean speedup: 2.19x, Range: 1.26x - 3.38x
**3D_pow2**:
Mean speedup: 1.57x, Range: 1.06x - 2.02x
**3D_nonpow2**:
Mean speedup: 2.08x, Range: 1.11x - 3.25x
**mixed_workload**:

## Category Summary
### 1D_pow2
Total benchmarks: 42

**BetterFFTW**:
Mean speedup: 0.16x, Median: 0.01x, Range: 0.00x - 1.31x
**SciPy**:
Mean speedup: 1.87x, Median: 1.27x, Range: 0.75x - 5.49x

### 1D_nonpow2
Total benchmarks: 20

**BetterFFTW**:
Mean speedup: 0.01x, Median: 0.01x, Range: 0.01x - 0.01x
**SciPy**:
Mean speedup: 2.41x, Median: 2.41x, Range: 1.55x - 3.27x

### 2D_pow2
Total benchmarks: 18

**BetterFFTW**:
Mean speedup: 0.78x, Median: 0.36x, Range: 0.13x - 3.01x
**SciPy**:
Mean speedup: 1.47x, Median: 1.49x, Range: 0.99x - 2.17x

### 2D_nonpow2
Total benchmarks: 7

**BetterFFTW**:
Mean speedup: 0.26x, Median: 0.19x, Range: 0.01x - 0.76x
**SciPy**:
Mean speedup: 2.19x, Median: 1.99x, Range: 1.26x - 3.38x

### 3D_pow2
Total benchmarks: 4

**BetterFFTW**:
Mean speedup: 0.85x, Median: 0.84x, Range: 0.75x - 0.97x
**SciPy**:
Mean speedup: 1.57x, Median: 1.60x, Range: 1.06x - 2.02x

### 3D_nonpow2
Total benchmarks: 4

**BetterFFTW**:
Mean speedup: 0.89x, Median: 0.53x, Range: 0.40x - 1.72x
**SciPy**:
Mean speedup: 2.08x, Median: 1.97x, Range: 1.11x - 3.25x

### mixed_workload
Total benchmarks: 1

**BetterFFTW**:
**SciPy**:

## Key Insights
- For 1D_pow2 transforms, SciPy is the fastest with 1.87x mean speedup over NumPy
- For 1D_nonpow2 transforms, SciPy is the fastest with 2.41x mean speedup over NumPy
- For 2D_pow2 transforms, SciPy is the fastest with 1.47x mean speedup over NumPy
- For 2D_nonpow2 transforms, SciPy is the fastest with 2.19x mean speedup over NumPy
- For 3D_pow2 transforms, SciPy is the fastest with 1.57x mean speedup over NumPy
- For 3D_nonpow2 transforms, SciPy is the fastest with 2.08x mean speedup over NumPy

### Top 5 Speedups
1. SciPy on ifft on 4096 <class 'complex128'> array (power-of-2): 5.49x speedup
2. SciPy on fft2 on 96x96 <class 'float64'> array (non-power-of-2): 3.38x speedup
3. SciPy on fft on 3072 <class 'complex128'> array (non-power-of-2): 3.27x speedup
4. SciPy on irfftn on 48x48x48 <class 'float64'> array (non-power-of-2): 3.25x speedup
5. SciPy on ifft2 on 96x96 <class 'float64'> array (non-power-of-2): 3.07x speedup

### Significant Slowdowns
1. BetterFFTW on rfft on 4096 <class 'float64'> array (power-of-2): 0.00x speedup (slowdown)
2. BetterFFTW on fft on 3072 <class 'float64'> array (non-power-of-2): 0.01x speedup (slowdown)
3. BetterFFTW on fft on 4096 <class 'float32'> array (power-of-2): 0.01x speedup (slowdown)
4. BetterFFTW on ifft on 4096 <class 'float64'> array (power-of-2): 0.01x speedup (slowdown)
5. BetterFFTW on irfft on 4096 <class 'float64'> array (power-of-2): 0.01x speedup (slowdown)