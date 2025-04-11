---
title: "BetterFFTW: A High-Performance, User-Friendly Wrapper for FFTW in Python"
tags:
  - FFT
  - FFTW
  - PyFFTW
  - Python
  - Scientific Computing
authors:
  - name: "Sebastian Griego"
    orcid: "0009-0001-3195-8575"
    affiliation: "1"
affiliations:
  - name: "San Diego State University"
    index: 1
date: 7 April 2025
bibliography: paper.bib
---

# Summary

BetterFFTW is a GPL-3.0–licensed Python library that offers a high-level interface to the FFTW C library by wrapping pyFFTW. It provides drop-in replacements for NumPy and SciPy FFT functions with optimized defaults. By abstracting away FFTW's often complex configuration (such as planner flag management and thread selection), BetterFFTW makes high-performance FFT computation accessible to non-expert users.

# Statement of Need

Fourier transforms are essential for many scientific and engineering applications. While Python's NumPy and SciPy FFT functions are easy to use, they typically lack the performance benefits afforded by FFTW—a library renowned for its speed and efficiency when compared with traditional algorithms [@fftw]. Although pyFFTW exposes FFTW's capabilities to Python, its default settings are often suboptimal. BetterFFTW fills this gap by automating plan caching, thread management, and planner flag selection (e.g. using FFTW_ESTIMATE for one-off calls and upgrading to FFTW_MEASURE or FFTW_PATIENT for repeated transforms). This approach ensures that while the initial call may incur FFTW's planning overhead, subsequent FFTs benefit from high-amortized efficiency—particularly for multi-dimensional transforms where speedups can exceed 10×.

# Implementation

BetterFFTW is implemented as a thin, lightweight layer over pyFFTW [@pyfftw]. It mirrors the APIs of NumPy's and SciPy's FFT modules and leverages Python's uarray protocol for seamless integration. Major implementation details include:

- **Plan Caching and Upgrade:** FFT plans are cached based on transform parameters. An initial plan is generated using a fast planner flag (FFTW_ESTIMATE); if the same transform is used repeatedly, the library schedules a background upgrade to a more optimized plan (using FFTW_MEASURE or FFTW_PATIENT) to improve performance.
- **Automatic Thread Selection:** The library inspects input array sizes and dimensionality to automatically determine optimal thread counts, mitigating the overhead of excessive threading on small arrays.
- **Fallback Mechanism:** When testing indicates that NumPy's FFT may be faster (especially for one-off or very small transforms), BetterFFTW will automatically revert to the default implementation.
- **Planner Flags:** Users benefit from improved defaults without manual tuning; BetterFFTW uses FFTW's planner flags (e.g., FFTW_ESTIMATE, FFTW_MEASURE, FFTW_PATIENT, FFTW_EXHAUSTIVE) internally to balance planning time versus runtime performance.

The design of BetterFFTW adheres to current JOSS criteria by providing a transparent, reproducible interface with full documentation, unit tests, and an accompanying repository (https://github.com/sebastian-griego/BetterFFTW).

# Performance

Benchmarks show that BetterFFTW is, on average, nearly 4× faster than NumPy's FFT functions for common use cases. For multi-dimensional transforms and scenarios where the same transform is applied repeatedly, speedups can exceed 10× once the planning overhead is amortized. These improvements arise from FFTW's efficient algorithms and the automatic reuse of highly optimized FFT plans. BetterFFTW's performance approach is similar to that of FFTW itself, and while a one-off transform may bear a planning cost, the benefits become evident in iterative computations—exactly the context where FFTW (and thus BetterFFTW) excels [@fftw; @pyfftw].

# Acknowledgements

The development of BetterFFTW was inspired by FFTW's renowned efficiency and by pyFFTW's efforts to bring FFTW to Python. Many thanks to Matteo Frigo and Steven G. Johnson for creating FFTW [@fftw] and to the maintainers of pyFFTW [@pyfftw]. BetterFFTW also benefits from the robust ecosystems of NumPy and SciPy. Feedback from early adopters has been invaluable for refining its default settings and usability.

# References