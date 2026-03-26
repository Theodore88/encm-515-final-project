import numpy as np
import copy
cimport numpy as cnp
from libc.stdint cimport int64_t

BITS = 64
Q_BITS = 32

# Explicit domain scales
X_SCALE = 12  # state values
P_SCALE = 12  # covariance values (larger dynamic range)
H_SCALE = 12  # measurement matrix values (usually 1, but we use same unit)
Z_SCALE = 12  # measurement values

def quantize(double value, double scale = Q_BITS) -> int:
    """Quantize a float to fixed-point int."""
    cdef double q_copy = value
    cdef double scale_amount = 2**scale
    cdef double x_scaled = q_copy * scale_amount
    cdef int64_t q_min = -(2**(BITS - 1))
    cdef int64_t q_max = (2**(BITS - 1) - 1)

    if x_scaled < q_min:
        return int(q_min)
    elif x_scaled > q_max:
        return int(q_max)

    return int(x_scaled)

def dequantize(int q_value, double scale = Q_BITS) -> float:
    """Dequantize a fixed-point int to float."""
    cdef double scale_amount = 2.0**(scale)
    cdef double x_decode = q_value / scale_amount
    return float(x_decode)

def quantize_array(cnp.ndarray arr, int scale = Q_BITS):
    """Quantize a float array to fixed-point int array."""
    cdef cnp.ndarray x_input = np.asarray(copy.deepcopy(arr), dtype=np.float64)
    cdef double scale_amount = 2**scale
    cdef cnp.ndarray x_scaled = np.multiply(x_input, scale_amount)
    cdef int64_t q_min = -(2**(BITS - 1))
    cdef int64_t q_max = (2**(BITS - 1) - 1)
    
    return np.clip(x_scaled, q_min, q_max).astype(np.int64)

def dequantize_array(cnp.ndarray q_arr, int scale = Q_BITS):
    """Dequantize a fixed-point int array to float array."""
    cdef cnp.ndarray x_input = np.asarray(copy.deepcopy(q_arr))
    cdef double scale_amount = 2**scale
    cdef cnp.ndarray x_decode = np.divide(x_input, scale_amount)
    return x_decode.astype(np.float64)

def q_mul(a_q, b_q, int a_scale = Q_BITS,
          int b_scale = Q_BITS, int out_scale = Q_BITS):
    cdef int64_t q_min = -(2**(BITS - 1))
    cdef int64_t q_max = (2**(BITS - 1) - 1)
    cdef int shift = a_scale + b_scale - out_scale
    cdef object product
    cdef int64_t shifted

    cdef cnp.ndarray A_in = np.asarray(a_q, dtype=np.int64)
    cdef cnp.ndarray B_in = np.asarray(b_q, dtype=np.int64)

    product = np.multiply(A_in, B_in)
    shift = a_scale + b_scale - out_scale
    result = product >> shift
    q_min = -(2**(BITS - 1))
    q_max = (2**(BITS - 1) - 1)
    return np.clip(result, q_min, q_max).astype(np.int64)


def q_mat_mul(cnp.ndarray A, cnp.ndarray B, int a_scale = Q_BITS, 
              int b_scale = Q_BITS, int out_scale = Q_BITS):
    cdef cnp.ndarray result = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    cdef int i, j, k, shift
    cdef int64_t acc
    cdef cnp.ndarray A_in = np.asarray(A, dtype=np.int64)
    cdef cnp.ndarray B_in = np.asarray(B, dtype=np.int64)

    # Vectorized matrix multiply using BLAS
    product = A_in @ B_in
    shift = a_scale + b_scale - out_scale
    result = product >> shift
    q_min = -(2**(BITS - 1))
    q_max = (2**(BITS - 1) - 1)
    return np.clip(result, q_min, q_max).astype(np.int64)