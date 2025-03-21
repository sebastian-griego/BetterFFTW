"""
Simple test script for BetterFFTW
"""

# Add the src directory to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import time

def main():
    # Create sample data
    print("Creating test data...")
    size = 512
    data = np.random.random((size, size)).astype(np.float64)
    
    # Run with NumPy
    print("\nTesting NumPy FFT...")
    start = time.time()
    np_result = np.fft.fft2(data)
    numpy_time = time.time() - start
    print(f"NumPy FFT: {numpy_time:.4f} seconds")
    
    # Import BetterFFTW
    print("\nImporting BetterFFTW...")
    import betterfftw
    
    # Try basic FFT without replacing NumPy
    print("\nTesting BetterFFTW direct call...")
    start = time.time()
    bfft_result = betterfftw.fft2(data)
    bfft_time = time.time() - start
    print(f"BetterFFTW direct: {bfft_time:.4f} seconds")
    
    # Verify results match
    print(f"\nResults match: {np.allclose(np_result, bfft_result)}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()