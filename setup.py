from setuptools import setup, find_packages

setup(
    name="betterfftw",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.17.0",
        "scipy>=1.3.0",
        "pyfftw>=0.12.0",
        "psutil>=5.6.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
        "bench": ["matplotlib", "seaborn", "pandas", "tabulate"],
    },
    python_requires=">=3.7",
    author="Sebastian Griego",
    author_email="sebastianngriego@gmail.com",
    description="A high-performance, user-friendly wrapper for FFTW in Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sebastian-griego/betterfftw",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GPL v3 License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
)