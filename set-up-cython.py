from setuptools import setup
from Cython.Build import cythonize
import numpy as np
import os
import shutil

# Create cython folder if not exists
if not os.path.exists('cython'):
    os.makedirs('cython')

# Move .pyx to cython folder
pyx_path = 'quantize_helpers.pyx'
cython_pyx_path = 'cython/quantize_helpers.pyx'
if os.path.exists(pyx_path) and not os.path.exists(cython_pyx_path):
    shutil.move(pyx_path, cython_pyx_path)

# Build the extension
setup(
    name="quantize_helpers",
    ext_modules=cythonize(cython_pyx_path, compiler_directives={"language_level": "3"}),
    include_dirs=[np.get_include()]
)

# Move the compiled .pyd back to root for import
for file in os.listdir('cython'):
    if file.startswith('quantize_helpers') and file.endswith('.pyd'):
        pyd_path = os.path.join('cython', file)
        shutil.move(pyd_path, '.')
        break
