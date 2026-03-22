from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="quantize_helpers",
    ext_modules=cythonize("quantize_helpers.pyx", compiler_directives={"language_level": "3"}),
    include_dirs=[np.get_include()]
)