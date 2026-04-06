from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="ekf_functions",
        sources=["ekf_functions.pyx", "simd_acceleration_functions.c"],
        include_dirs=[np.get_include()],
    )
]

setup(name="ekf_functions", ext_modules=cythonize(extensions))