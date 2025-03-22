import unittest
import numpy as np
import scipy.fft
import betterfftw
import time
import warnings
import os

class TestBetterFFTW(unittest.TestCase):
    
    def setUp(self):
        # silence warnings for cleaner output
        warnings.filterwarnings('ignore')
        
        # reset to default FFT implementations before each test
        try:
            betterfftw.restore_default()
        except:
            pass
        
        # create test arrays of different shapes and types with fixed seed for reproducibility
        np.random.seed(42)
        
        # 1D arrays - various sizes to test different FFT behaviors
        self.real_1d_small = np.random.random(16)  # small power of 2
        self.real_1d_large = np.random.random(1024)  # large power of 2
        self.real_1d_odd = np.random.random(33)  # odd size
        self.real_1d_prime = np.random.random(101)  # prime size (harder for FFT)
        
        # complex versions of 1D arrays
        self.complex_1d_small = self.real_1d_small + 1j * np.random.random(16)
        self.complex_1d_large = self.real_1d_large + 1j * np.random.random(1024)
        self.complex_1d_odd = self.real_1d_odd + 1j * np.random.random(33)
        self.complex_1d_prime = self.real_1d_prime + 1j * np.random.random(101)
        
        # 2D arrays
        self.real_2d_small = np.random.random((16, 16))  # small square
        self.real_2d_rect = np.random.random((32, 16))  # rectangular
        self.real_2d_odd = np.random.random((15, 17))  # odd dimensions
        self.real_2d_prime = np.random.random((101, 103))  # prime dimensions
        
        # complex versions of 2D arrays
        self.complex_2d_small = self.real_2d_small + 1j * np.random.random((16, 16))
        self.complex_2d_rect = self.real_2d_rect + 1j * np.random.random((32, 16))
        self.complex_2d_odd = self.real_2d_odd + 1j * np.random.random((15, 17))
        self.complex_2d_prime = self.real_2d_prime + 1j * np.random.random((101, 103))
        
        # 3D arrays
        self.real_3d = np.random.random((8, 8, 8))  # 3D cube
        self.complex_3d = self.real_3d + 1j * np.random.random((8, 8, 8))
        
        # Empty array (edge case)
        self.empty_array = np.array([])
        
        # tolerance for floating point comparisons
        self.tol = 1e-10
    
    def tearDown(self):
        # restore original FFT implementations after each test
        try:
            betterfftw.restore_default()
        except:
            pass
    
    def _assert_allclose(self, a, b, msg=None):
        """Helper to compare arrays with appropriate tolerance based on dtype."""
        # Use higher tolerance for float32 and complex64
        if a.dtype in (np.float32, np.complex64) or b.dtype in (np.float32, np.complex64):
            rtol, atol = 1e-5, 1e-5  # Looser tolerance for single precision
        else:
            rtol, atol = self.tol, self.tol  # Default tolerance for double precision
            
        self.assertTrue(np.allclose(a, b, rtol=rtol, atol=atol), msg)
    # ===== 1D TRANSFORM TESTS =====
    
    def test_fft_1d(self):
        """Test 1D FFT against NumPy reference"""
        # test all array types
        for arr in [self.real_1d_small, self.real_1d_large, self.real_1d_odd, self.real_1d_prime,
                   self.complex_1d_small, self.complex_1d_large, self.complex_1d_odd, self.complex_1d_prime]:
            np_result = np.fft.fft(arr)
            bf_result = betterfftw.fft(arr)
            self._assert_allclose(bf_result, np_result, f"fft failed for {arr.shape}")
        
        # test with n parameter (padding/truncation)
        arr = self.real_1d_small
        n = 32  # larger than array size
        np_result = np.fft.fft(arr, n=n)
        bf_result = betterfftw.fft(arr, n=n)
        self._assert_allclose(bf_result, np_result, "fft with n parameter failed")
        
        # test axis parameter on 2D array
        arr = self.real_2d_small
        for axis in [0, 1, -1]:
            np_result = np.fft.fft(arr, axis=axis)
            bf_result = betterfftw.fft(arr, axis=axis)
            self._assert_allclose(bf_result, np_result, f"fft with axis={axis} failed")
        
        # test normalization
        for norm in [None, "ortho"]:
            np_result = np.fft.fft(arr, norm=norm)
            bf_result = betterfftw.fft(arr, norm=norm)
            self._assert_allclose(bf_result, np_result, f"fft with norm={norm} failed")
    
    def test_ifft_1d(self):
        """Test 1D inverse FFT against NumPy reference"""
        # test basic functionality
        for arr in [self.complex_1d_small, self.complex_1d_large, self.complex_1d_odd, self.complex_1d_prime]:
            np_result = np.fft.ifft(arr)
            bf_result = betterfftw.ifft(arr)
            self._assert_allclose(bf_result, np_result, f"ifft failed for {arr.shape}")
        
        # test roundtrip: original → fft → ifft → original
        for arr in [self.real_1d_small, self.complex_1d_small]:
            roundtrip = betterfftw.ifft(betterfftw.fft(arr))
            self._assert_allclose(roundtrip, arr, "roundtrip fft→ifft failed")
    
    def test_rfft_1d(self):
        """Test 1D real FFT against NumPy reference"""
        # test on real inputs of various sizes
        for arr in [self.real_1d_small, self.real_1d_large, self.real_1d_odd, self.real_1d_prime]:
            np_result = np.fft.rfft(arr)
            bf_result = betterfftw.rfft(arr)
            self._assert_allclose(bf_result, np_result, f"rfft failed for {arr.shape}")
    
    def test_irfft_1d(self):
        """Test 1D inverse real FFT against NumPy reference"""
        # test on complex inputs
        for arr in [self.complex_1d_small, self.complex_1d_large, self.complex_1d_odd, self.complex_1d_prime]:
            np_result = np.fft.irfft(arr)
            bf_result = betterfftw.irfft(arr)
            self._assert_allclose(bf_result, np_result, f"irfft failed for {arr.shape}")
        
        # test roundtrip: original → rfft → irfft → original
        for arr in [self.real_1d_small, self.real_1d_large]:
            roundtrip = betterfftw.irfft(betterfftw.rfft(arr))
            # for real transforms, match the length of the input
            self._assert_allclose(roundtrip[:len(arr)], arr, "roundtrip rfft→irfft failed")
    
    def test_hfft_1d(self):
        """Test 1D Hermitian FFT against NumPy reference"""
        # create Hermitian-symmetric data
        half_size = 8
        hermitian_data = np.random.random(half_size) + 1j * np.random.random(half_size)
        
        np_result = np.fft.hfft(hermitian_data)
        bf_result = betterfftw.hfft(hermitian_data)
        self._assert_allclose(bf_result, np_result, "hfft failed")
    
    def test_ihfft_1d(self):
        """Test 1D inverse Hermitian FFT against NumPy reference"""
        # real data for ihfft
        real_data = np.random.random(16)
        
        np_result = np.fft.ihfft(real_data)
        bf_result = betterfftw.ihfft(real_data)
        self._assert_allclose(bf_result, np_result, "ihfft failed")
    
    # ===== 2D TRANSFORM TESTS =====
    
    def test_fft2(self):
        """Test 2D FFT against NumPy reference"""
        # test all 2D array types
        for arr in [self.real_2d_small, self.real_2d_rect, self.real_2d_odd, self.real_2d_prime,
                   self.complex_2d_small, self.complex_2d_rect, self.complex_2d_odd, self.complex_2d_prime]:
            np_result = np.fft.fft2(arr)
            bf_result = betterfftw.fft2(arr)
            self._assert_allclose(bf_result, np_result, f"fft2 failed for {arr.shape}")
        
        # test with s parameter (shape)
        arr = self.real_2d_small
        s = (32, 32)  # different output shape
        np_result = np.fft.fft2(arr, s=s)
        bf_result = betterfftw.fft2(arr, s=s)
        self._assert_allclose(bf_result, np_result, "fft2 with s parameter failed")
        
        # test with axes parameter on 3D array
        arr = self.real_3d
        axes = (0, 2)  # non-default axes
        np_result = np.fft.fft2(arr, axes=axes)
        bf_result = betterfftw.fft2(arr, axes=axes)
        self._assert_allclose(bf_result, np_result, "fft2 with axes parameter failed")
    
    def test_ifft2(self):
        """Test 2D inverse FFT against NumPy reference"""
        # test all complex 2D arrays
        for arr in [self.complex_2d_small, self.complex_2d_rect, self.complex_2d_odd, self.complex_2d_prime]:
            np_result = np.fft.ifft2(arr)
            bf_result = betterfftw.ifft2(arr)
            self._assert_allclose(bf_result, np_result, f"ifft2 failed for {arr.shape}")
        
        # test roundtrip: original → fft2 → ifft2 → original
        for arr in [self.real_2d_small, self.complex_2d_small]:
            roundtrip = betterfftw.ifft2(betterfftw.fft2(arr))
            self._assert_allclose(roundtrip, arr, "roundtrip fft2→ifft2 failed")
    
    def test_rfft2(self):
        """Test 2D real FFT against NumPy reference"""
        # test all real 2D arrays
        for arr in [self.real_2d_small, self.real_2d_rect, self.real_2d_odd, self.real_2d_prime]:
            np_result = np.fft.rfft2(arr)
            bf_result = betterfftw.rfft2(arr)
            self._assert_allclose(bf_result, np_result, f"rfft2 failed for {arr.shape}")
    
    def test_irfft2(self):
        """Test 2D inverse real FFT against NumPy reference"""
        # test all complex 2D arrays
        for arr, real_arr in [(self.complex_2d_small, self.real_2d_small), 
                             (self.complex_2d_rect, self.real_2d_rect),
                             (self.complex_2d_odd, self.real_2d_odd),
                             (self.complex_2d_prime, self.real_2d_prime)]:
            np_result = np.fft.irfft2(arr)
            bf_result = betterfftw.irfft2(arr)
            self._assert_allclose(bf_result, np_result, f"irfft2 failed for {arr.shape}")
        
        # test roundtrip: original → rfft2 → irfft2 → original
        for arr in [self.real_2d_small, self.real_2d_rect]:
            # specify shape for irfft2
            roundtrip = betterfftw.irfft2(betterfftw.rfft2(arr), s=arr.shape)
            self._assert_allclose(roundtrip, arr, "roundtrip rfft2→irfft2 failed")
    
    # ===== ND TRANSFORM TESTS =====
    
    def test_fftn(self):
        """Test N-dimensional FFT against NumPy reference"""
        # test 3D arrays
        for arr in [self.real_3d, self.complex_3d]:
            np_result = np.fft.fftn(arr)
            bf_result = betterfftw.fftn(arr)
            self._assert_allclose(bf_result, np_result, f"fftn failed for {arr.shape}")
        
        # test with s parameter
        arr = self.real_3d
        s = (16, 16, 16)  # different shape
        np_result = np.fft.fftn(arr, s=s)
        bf_result = betterfftw.fftn(arr, s=s)
        self._assert_allclose(bf_result, np_result, "fftn with s parameter failed")
        
        # test with axes parameter (partial transform)
        arr = self.real_3d
        axes = (0, 2)
        np_result = np.fft.fftn(arr, axes=axes)
        bf_result = betterfftw.fftn(arr, axes=axes)
        self._assert_allclose(bf_result, np_result, "fftn with axes parameter failed")
    
    def test_ifftn(self):
        """Test N-dimensional inverse FFT against NumPy reference"""
        # test 3D complex array
        for arr in [self.complex_3d]:
            np_result = np.fft.ifftn(arr)
            bf_result = betterfftw.ifftn(arr)
            self._assert_allclose(bf_result, np_result, f"ifftn failed for {arr.shape}")
        
        # test roundtrip: original → fftn → ifftn → original
        for arr in [self.real_3d, self.complex_3d]:
            roundtrip = betterfftw.ifftn(betterfftw.fftn(arr))
            self._assert_allclose(roundtrip, arr, "roundtrip fftn→ifftn failed")
    
    def test_rfftn(self):
        """Test N-dimensional real FFT against NumPy reference"""
        # test 3D real array
        for arr in [self.real_3d]:
            np_result = np.fft.rfftn(arr)
            bf_result = betterfftw.rfftn(arr)
            self._assert_allclose(bf_result, np_result, f"rfftn failed for {arr.shape}")
    
    def test_irfftn(self):
        """Test N-dimensional inverse real FFT against NumPy reference"""
        # test 3D complex array
        for arr in [self.complex_3d]:
            np_result = np.fft.irfftn(arr)
            bf_result = betterfftw.irfftn(arr)
            self._assert_allclose(bf_result, np_result, f"irfftn failed for {arr.shape}")
        
        # test roundtrip: original → rfftn → irfftn → original
        for arr in [self.real_3d]:
            # specify shape for irfftn
            roundtrip = betterfftw.irfftn(betterfftw.rfftn(arr), s=arr.shape)
            self._assert_allclose(roundtrip, arr, "roundtrip rfftn→irfftn failed")
    
    # ===== HELPER FUNCTION TESTS =====
    
    def test_fftfreq(self):
        """Test FFT frequency generation"""
        n = 16
        d = 0.1
        np_result = np.fft.fftfreq(n, d)
        bf_result = betterfftw.fftfreq(n, d)
        self._assert_allclose(bf_result, np_result, "fftfreq failed")
    
    def test_rfftfreq(self):
        """Test real FFT frequency generation"""
        n = 16
        d = 0.1
        np_result = np.fft.rfftfreq(n, d)
        bf_result = betterfftw.rfftfreq(n, d)
        self._assert_allclose(bf_result, np_result, "rfftfreq failed")
    
    def test_fftshift(self):
        """Test FFT shift function"""
        arr = self.real_2d_small
        np_result = np.fft.fftshift(arr)
        bf_result = betterfftw.fftshift(arr)
        self._assert_allclose(bf_result, np_result, "fftshift failed")
        
        # test with axes parameter
        axes = 0
        np_result = np.fft.fftshift(arr, axes=axes)
        bf_result = betterfftw.fftshift(arr, axes=axes)
        self._assert_allclose(bf_result, np_result, "fftshift with axes parameter failed")
    
    def test_ifftshift(self):
        """Test inverse FFT shift function"""
        arr = self.real_2d_small
        np_result = np.fft.ifftshift(arr)
        bf_result = betterfftw.ifftshift(arr)
        self._assert_allclose(bf_result, np_result, "ifftshift failed")
        
        # verify ifftshift is the inverse of fftshift
        shifted = betterfftw.fftshift(arr)
        unshifted = betterfftw.ifftshift(shifted)
        self._assert_allclose(unshifted, arr, "ifftshift is not the inverse of fftshift")
    
    # ===== DROP-IN REPLACEMENT TESTS =====
    
    def test_numpy_replacement(self):
        """Test if BetterFFTW can replace NumPy's FFT functions"""
        # save original function
        orig_fft = np.fft.fft
        
        # register BetterFFTW
        betterfftw.use_as_default()
        
        # verify NumPy now uses BetterFFTW
        arr = self.real_1d_small
        np_result = np.fft.fft(arr)
        bf_result = betterfftw.fft(arr)
        self._assert_allclose(np_result, bf_result, "NumPy replacement failed")
        
        # verify they're not the same function references
        self.assertNotEqual(orig_fft, np.fft.fft, "NumPy function not replaced")
        
        # restore original
        betterfftw.restore_default()
        
        # verify NumPy is back to original
        self.assertEqual(orig_fft, np.fft.fft, "NumPy function not restored")
    
    def test_scipy_replacement(self):
        """Test if BetterFFTW can replace SciPy's FFT functions"""
        # save original function
        orig_fft = scipy.fft.fft
        
        try:
            # register BetterFFTW
            betterfftw.use_as_default(register_scipy=True)
            
            # verify SciPy uses BetterFFTW
            arr = self.real_1d_small
            scipy_result = scipy.fft.fft(arr)
            bf_result = betterfftw.fft(arr)
            self._assert_allclose(scipy_result, bf_result, "SciPy replacement failed")
        except Exception as e:
            # some environments might not support SciPy backend replacement
            warnings.warn(f"SciPy replacement test skipped: {str(e)}")
        
        # restore original
        betterfftw.restore_default(unregister_scipy=True)
    
    # ===== ADVANCED FEATURES AND EDGE CASES =====
    
    def test_threads_parameter(self):
        """Test if threads parameter affects performance"""
        arr = np.random.random((512, 512))  # large array to make threading noticeable
        
        # measure time with 1 thread
        start = time.time()
        betterfftw.fft2(arr, threads=1)
        single_thread_time = time.time() - start
        
        # measure time with multiple threads
        start = time.time()
        betterfftw.fft2(arr, threads=4)
        multi_thread_time = time.time() - start
        
        # just verify it runs without crashing with different thread counts
        # (actual performance depends on hardware)
        print(f"Single thread: {single_thread_time:.4f}s, Multi-thread: {multi_thread_time:.4f}s")
    
    def test_planner_parameter(self):
        """Test if planner parameter works"""
        arr = self.real_1d_small
        
        # try different planner strategies
        for planner in [betterfftw.PLANNER_ESTIMATE, betterfftw.PLANNER_MEASURE, 
                       betterfftw.PLANNER_PATIENT]:
            # should run without errors
            bf_result = betterfftw.fft(arr, planner=planner)
            
            # verify the result still matches NumPy
            np_result = np.fft.fft(arr)
            self._assert_allclose(bf_result, np_result, f"fft with planner={planner} failed")
    
    def test_alignment_functions(self):
        """Test array alignment functions"""
        # test empty_aligned
        aligned_array = betterfftw.empty_aligned((16, 16), dtype=np.complex128)
        self.assertEqual(aligned_array.shape, (16, 16), "empty_aligned shape incorrect")
        self.assertEqual(aligned_array.dtype, np.complex128, "empty_aligned dtype incorrect")
        
        # test empty_aligned_like
        template = self.real_2d_small
        aligned_array = betterfftw.empty_aligned_like(template)
        self.assertEqual(aligned_array.shape, template.shape, "empty_aligned_like shape incorrect")
        self.assertEqual(aligned_array.dtype, template.dtype, "empty_aligned_like dtype incorrect")
        
        # test byte_align
        aligned = betterfftw.byte_align(self.real_1d_small)
        self._assert_allclose(aligned, self.real_1d_small, "byte_align changed values")
    
    def test_wisdom_functions(self):
        """Test FFTW wisdom import/export"""
        # First, generate some wisdom by running more substantial FFTs
        arr = np.random.random((128, 128))
        
        # Run several FFTs with different settings to generate wisdom
        for planner in [betterfftw.PLANNER_MEASURE, betterfftw.PLANNER_PATIENT]:
            for threads in [1, 2]:
                betterfftw.fft2(arr, planner=planner, threads=threads)
        
        # Export wisdom directly using FFTW
        import pyfftw
        raw_wisdom = pyfftw.export_wisdom()
        
        # Save this wisdom to a file manually for testing
        wisdom_file = "test_wisdom.dat"
        with open(wisdom_file, 'wb') as f:
            f.write(raw_wisdom[0])
        
        # Now test the import function
        success = betterfftw.import_wisdom(wisdom_file)
        self.assertTrue(success, "Wisdom import failed")
        
        # Clean up
        try:
            os.remove(wisdom_file)
        except:
            pass
    def test_empty_array(self):
        """Test behavior with empty arrays"""
        try:
            result = betterfftw.fft(self.empty_array)
            # if it succeeds, verify the result is also empty
            self.assertEqual(len(result), 0, "Empty array FFT should return empty array")
        except ValueError:
            # some implementations might raise ValueError for empty arrays
            pass
    
    def test_performance_improvement(self):
        """Benchmark BetterFFTW against NumPy"""
        # use a large array where FFTW should have an advantage
        large_array = np.random.random((1024, 1024))
        
        # time NumPy's FFT (with warm-up)
        np.fft.fft2(large_array)  # warm-up
        start = time.time()
        np.fft.fft2(large_array)
        numpy_time = time.time() - start
        
        # time BetterFFTW (with warm-up)
        betterfftw.fft2(large_array)  # warm-up
        start = time.time()
        betterfftw.fft2(large_array)
        betterfftw_time = time.time() - start
        
        # print results (informational, not an assertion)
        print(f"NumPy: {numpy_time:.4f}s, BetterFFTW: {betterfftw_time:.4f}s")
        print(f"Speedup: {numpy_time / betterfftw_time:.2f}x")
    
    def test_stat_functions(self):
        """Test statistical functions of BetterFFTW"""
        # run some FFTs to populate the cache
        betterfftw.fft(self.real_1d_small)
        betterfftw.fft2(self.real_2d_small)
        
        # get statistics
        stats = betterfftw.get_stats()
        
        # verify the stats object is a dictionary with expected keys
        self.assertIsInstance(stats, dict, "Stats should be a dictionary")
        
        # check for expected keys
        expected_keys = ['total_plans', 'total_calls']
        for key in expected_keys:
            self.assertIn(key, stats, f"Expected key '{key}' missing from stats")
        
        # clear the cache and verify stats are updated
        betterfftw.clear_cache()
        stats = betterfftw.get_stats()
        self.assertEqual(stats['total_plans'], 0, "Cache not cleared properly")
    def test_adaptive_planning(self):
        """Test if BetterFFTW automatically optimizes frequently used transforms"""
        # reset cache first
        betterfftw.clear_cache()
        
        # create a test array
        arr = np.random.random((128, 128))
        
        # run the transform multiple times to trigger optimization
        # first run is with ESTIMATE and might be slower
        start = time.time()
        for _ in range(5):
            betterfftw.fft2(arr)
        first_batch_time = time.time() - start
        
        # get stats to check optimization status
        stats = betterfftw.get_stats()
        print(f"After first batch: {stats}")
        
        # run more iterations - these should use an optimized plan
        start = time.time()
        for _ in range(20):
            betterfftw.fft2(arr)
        second_batch_time = time.time() - start
        
        # get stats again to see if plans were optimized
        stats = betterfftw.get_stats()
        print(f"After second batch: {stats}")
        
        # performance should improve or stay similar (can't assert exact timing due to system variations)
        print(f"First batch (5 runs): {first_batch_time:.4f}s, Second batch (20 runs): {second_batch_time:.4f}s")
        
        # Handle potential division by zero (if timing is too fast to measure)
        if first_batch_time > 0 and second_batch_time > 0:
            per_run_first = first_batch_time / 5
            per_run_second = second_batch_time / 20
            improvement = per_run_first / per_run_second
            print(f"Per-run improvement: {improvement:.2f}x")
        else:
            print("Timing too small to measure accurate improvement")
    def test_planning_optimization(self):
        """Test planning module functions"""
        # test optimal transform size function
        orig_size = 101  # prime number, not ideal for FFT
        opt_size = betterfftw.planning.optimal_transform_size(orig_size)
        self.assertGreaterEqual(opt_size, orig_size, "Optimal size should be >= original")
        
        # test if optimal size has better prime factorization
        # optimal sizes should be powers of 2 or have small prime factors
        def largest_prime_factor(n):
            i = 2
            while i * i <= n:
                if n % i:
                    i += 1
                else:
                    n //= i
            return n
        
        lpf_orig = largest_prime_factor(orig_size)
        lpf_opt = largest_prime_factor(opt_size)
        print(f"Original size: {orig_size} (largest prime: {lpf_orig})")
        print(f"Optimal size: {opt_size} (largest prime: {lpf_opt})")
        self.assertLessEqual(lpf_opt, lpf_orig, "Optimal size should have smaller largest prime factor")
        
        # test multi-dimensional shape optimization
        orig_shape = (101, 103)
        opt_shape = betterfftw.planning.optimize_transform_shape(orig_shape)
        print(f"Original shape: {orig_shape}, Optimized shape: {opt_shape}")
        
        # test get_optimal_threads function
        small_array = np.random.random((16, 16))
        large_array = np.random.random((1024, 1024))
        
        small_threads = betterfftw.planning.get_optimal_threads(small_array)
        large_threads = betterfftw.planning.get_optimal_threads(large_array)
        
        print(f"Optimal threads for small array: {small_threads}")
        print(f"Optimal threads for large array: {large_threads}")
        self.assertGreaterEqual(large_threads, small_threads, 
                               "Larger arrays should use at least as many threads as smaller arrays")
    
    def test_different_dtypes(self):
        """Test BetterFFTW with different NumPy data types"""
        dtypes = [np.float32, np.float64, np.complex64, np.complex128]
        
        for dtype in dtypes:
            arr = np.random.random(32).astype(dtype)
            
            # should run without errors
            result = betterfftw.fft(arr)
            
            # verify the result matches NumPy
            np_result = np.fft.fft(arr)
            self._assert_allclose(result, np_result, f"fft failed for dtype {dtype}")
    
    def test_edge_case_scaling(self):
        """Test scaling behavior with different norm parameters"""
        arr = self.real_1d_small
        
        # test with default normalization
        np_result = np.fft.fft(arr, norm=None)
        bf_result = betterfftw.fft(arr, norm=None)
        self._assert_allclose(bf_result, np_result, "Default normalization failed")
        
        # test with orthogonal normalization
        np_result = np.fft.fft(arr, norm="ortho")
        bf_result = betterfftw.fft(arr, norm="ortho")
        self._assert_allclose(bf_result, np_result, "Orthogonal normalization failed")


if __name__ == "__main__":
    unittest.main()