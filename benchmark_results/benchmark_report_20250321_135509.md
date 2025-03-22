# FFT BENCHMARK REPORT
Generated: 2025-03-21 13:55:09

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
Best overall implementation: SciPy (Mean speedup: 1.75x)

## Implementation Summary
### BetterFFTW
Success rate: 54.1% (53/98)
Mean speedup: 0.54x
Median speedup: 0.36x
Min speedup: 0.00x
Max speedup: 2.82x
Std. dev.: 0.71

#### Category Performance
**1D_pow2**:
Mean speedup: 0.02x, Range: 0.00x - 0.04x
**1D_nonpow2**:
**2D_pow2**:
Mean speedup: 0.74x, Range: 0.07x - 2.82x
**2D_nonpow2**:
Mean speedup: 0.38x, Range: 0.02x - 0.65x
**3D_pow2**:
Mean speedup: 0.91x, Range: 0.58x - 1.48x
**3D_nonpow2**:
Mean speedup: 0.45x, Range: 0.20x - 0.61x
**mixed_workload**:

### NumPy (Reference)
Success rate: 36.7% (36/98)

#### Category Performance
**1D_pow2**:
**1D_nonpow2**:
**2D_pow2**:
**2D_nonpow2**:
**3D_pow2**:
**3D_nonpow2**:
**mixed_workload**:

### SciPy
Success rate: 34.7% (34/98)
Mean speedup: 1.75x
Median speedup: 1.58x
Min speedup: 0.82x
Max speedup: 3.68x
Std. dev.: 0.71

#### Category Performance
**1D_pow2**:
Mean speedup: 3.13x, Range: 3.13x - 3.13x
**1D_nonpow2**:
Mean speedup: 1.93x, Range: 1.93x - 1.93x
**2D_pow2**:
Mean speedup: 1.44x, Range: 0.82x - 2.12x
**2D_nonpow2**:
Mean speedup: 1.63x, Range: 1.11x - 2.45x
**3D_pow2**:
Mean speedup: 2.10x, Range: 1.31x - 3.68x
**3D_nonpow2**:
Mean speedup: 1.97x, Range: 1.05x - 2.85x
**mixed_workload**:

## Category Summary
### 1D_pow2
Total benchmarks: 42

**BetterFFTW**:
Mean speedup: 0.02x, Median: 0.01x, Range: 0.00x - 0.04x
**SciPy**:
Mean speedup: 3.13x, Median: 3.13x, Range: 3.13x - 3.13x

### 1D_nonpow2
Total benchmarks: 20

**BetterFFTW**:
**SciPy**:
Mean speedup: 1.93x, Median: 1.93x, Range: 1.93x - 1.93x

### 2D_pow2
Total benchmarks: 18

**BetterFFTW**:
Mean speedup: 0.74x, Median: 0.35x, Range: 0.07x - 2.82x
**SciPy**:
Mean speedup: 1.44x, Median: 1.49x, Range: 0.82x - 2.12x

### 2D_nonpow2
Total benchmarks: 7

**BetterFFTW**:
Mean speedup: 0.38x, Median: 0.42x, Range: 0.02x - 0.65x
**SciPy**:
Mean speedup: 1.63x, Median: 1.48x, Range: 1.11x - 2.45x

### 3D_pow2
Total benchmarks: 4

**BetterFFTW**:
Mean speedup: 0.91x, Median: 0.68x, Range: 0.58x - 1.48x
**SciPy**:
Mean speedup: 2.10x, Median: 1.71x, Range: 1.31x - 3.68x

### 3D_nonpow2
Total benchmarks: 4

**BetterFFTW**:
Mean speedup: 0.45x, Median: 0.55x, Range: 0.20x - 0.61x
**SciPy**:
Mean speedup: 1.97x, Median: 1.98x, Range: 1.05x - 2.85x

### mixed_workload
Total benchmarks: 1

**BetterFFTW**:
**SciPy**:

## Key Insights
- For 1D_pow2 transforms, SciPy is the fastest with 3.13x mean speedup over NumPy
- For 1D_nonpow2 transforms, SciPy is the fastest with 1.93x mean speedup over NumPy
- For 2D_pow2 transforms, SciPy is the fastest with 1.44x mean speedup over NumPy
- For 2D_nonpow2 transforms, SciPy is the fastest with 1.63x mean speedup over NumPy
- For 3D_pow2 transforms, SciPy is the fastest with 2.10x mean speedup over NumPy
- For 3D_nonpow2 transforms, SciPy is the fastest with 1.97x mean speedup over NumPy

### Top 5 Speedups
1. SciPy on rfftn on 64x64x64 <class 'float64'> array (power-of-2): 3.68x speedup
2. SciPy on irfft on 1024 <class 'float32'> array (power-of-2): 3.13x speedup
3. SciPy on ifftn on 48x48x48 <class 'float64'> array (non-power-of-2): 2.85x speedup
4. BetterFFTW on fft2 on 512x512 <class 'complex128'> array (power-of-2): 2.82x speedup
5. SciPy on irfftn on 48x48x48 <class 'float64'> array (non-power-of-2): 2.47x speedup

### Significant Slowdowns
1. BetterFFTW on irfft on 1024 <class 'float64'> array (power-of-2): 0.00x speedup (slowdown)
2. BetterFFTW on irfft on 4096 <class 'float32'> array (power-of-2): 0.01x speedup (slowdown)
3. BetterFFTW on irfft on 1024 <class 'float32'> array (power-of-2): 0.02x speedup (slowdown)
4. BetterFFTW on fft2 on 96x96 <class 'float64'> array (non-power-of-2): 0.02x speedup (slowdown)
5. BetterFFTW on ifft on 4096 <class 'float64'> array (power-of-2): 0.04x speedup (slowdown)