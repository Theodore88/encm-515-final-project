Steps to run:
1. Run python setup.py build_ext --inplace to get any changes to SIMD C or Cython implementation
2. Run everything else normally

Files:
- simd_kernels.h/c: The actual raw and accelerated implementation
- setup.py: To compile the C code
- ekf_compare.c: The layer between the C and Python code