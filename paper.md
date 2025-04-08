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

`BetterFFTW` is a Python library that leverages the highly optimized FFTW C library for Fourier transforms, providing a user-friendly interface and intelligent plan caching. This paper summarizes the core features, design principles, and performance benefits of `BetterFFTW`.

# Statement of Need

Fourier transforms are fundamental in many fields, including signal processing, computational physics, and other domains needing fast and repeated FFTs. While `pyFFTW` provides Python bindings for FFTW, users still need to manage caching, planner selection, and thread optimization. `BetterFFTW` simplifies and automates these tasks, thus lowering the barrier for scientists and researchers seeking peak performance.

# Key Features

1. **Automatic Plan Caching & Upgrades**  
   Blah blah…

2. **Threading and Planner Auto-Selection**  
   Blah blah…

3. **Integration with NumPy & SciPy**  
   Blah blah…

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References