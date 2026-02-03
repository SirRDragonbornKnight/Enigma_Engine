"""
Setup script for compiling Cython kernels.

Usage:
    cd forge_ai/builtin
    python setup_cython.py build_ext --inplace

Or for development:
    pip install cython numpy
    cythonize -i cython_kernels.pyx
"""

from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False
    print("Cython not installed. Install with: pip install cython")

if USE_CYTHON:
    extensions = [
        Extension(
            "cython_kernels",
            ["cython_kernels.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3", "-ffast-math", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
    ]
    
    setup(
        name="forge_ai_cython_kernels",
        ext_modules=cythonize(
            extensions,
            compiler_directives={
                "language_level": 3,
                "boundscheck": False,
                "wraparound": False,
                "cdivision": True,
            }
        ),
    )
else:
    print("Cannot compile without Cython. Install it first:")
    print("  pip install cython numpy")
