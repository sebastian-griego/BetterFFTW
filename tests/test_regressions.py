import numpy as np
import pytest

import betterfftw
from betterfftw import planning
from betterfftw.core import SmartFFTW
import betterfftw.interface as interface


def test_registered_numpy_fft_falls_back_to_original_numpy(monkeypatch):
    betterfftw.restore_default()
    values = np.arange(16.0)
    expected = interface._ORIGINAL_NUMPY_BY_NAME["fft"](values)

    def broken_fft(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(interface.core, "fft", broken_fft)

    try:
        betterfftw.use_as_default(register_scipy=False)
        with pytest.warns(UserWarning, match="falling back to NumPy"):
            result = np.fft.fft(values)
    finally:
        betterfftw.restore_default()

    np.testing.assert_allclose(result, expected)


def test_as_default_context_restores_numpy_fft_after_exit():
    betterfftw.restore_default()
    original_fft = np.fft.fft
    values = np.arange(16.0)

    with betterfftw.as_default(register_scipy=False):
        assert np.fft.fft is not original_fft
        np.testing.assert_allclose(np.fft.fft(values), betterfftw.fft(values))

    assert np.fft.fft is original_fft


def test_as_default_context_restores_numpy_fft_after_exception():
    betterfftw.restore_default()
    original_fft = np.fft.fft

    with pytest.raises(RuntimeError, match="forced"):
        with betterfftw.as_default(register_scipy=False):
            assert np.fft.fft is not original_fft
            raise RuntimeError("forced")

    assert np.fft.fft is original_fft


def test_as_default_context_preserves_existing_registration():
    betterfftw.restore_default()
    original_fft = np.fft.fft

    try:
        betterfftw.use_as_default(register_scipy=False)
        registered_fft = np.fft.fft
        assert registered_fft is not original_fft

        with betterfftw.as_default(register_scipy=False):
            assert np.fft.fft is registered_fft

        assert np.fft.fft is registered_fft
    finally:
        betterfftw.restore_default()

    assert np.fft.fft is original_fft


def test_cache_key_separates_explicit_thread_counts():
    SmartFFTW.clear_cache()
    values = np.arange(64.0)

    SmartFFTW.fft(values, threads=1, planner=betterfftw.PLANNER_ESTIMATE)
    SmartFFTW.fft(values, threads=2, planner=betterfftw.PLANNER_ESTIMATE)

    keys = list(SmartFFTW._plan_cache)
    assert len(keys) == 2
    assert {key[-2] for key in keys} == {1, 2}
    assert {key[-1] for key in keys} == {betterfftw.PLANNER_ESTIMATE}


def test_benchmark_planners_builds_real_fftw_plans():
    values = np.linspace(0.0, 1.0, 32)

    results = planning.benchmark_planners(
        values,
        repeats=2,
        strategies=[planning.PLANNER_ESTIMATE],
        threads=1,
    )

    assert set(results) == {planning.PLANNER_ESTIMATE}
    metrics = results[planning.PLANNER_ESTIMATE]
    assert metrics["planning_time"] >= 0
    assert metrics["avg_execution"] >= 0
    assert metrics["total_time"] >= metrics["planning_time"]


def test_benchmark_planners_rejects_empty_repeat_count():
    with pytest.raises(ValueError, match="repeats"):
        planning.benchmark_planners(np.arange(8.0), repeats=0)
