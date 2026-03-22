import numpy as np
import copy
cimport numpy as cnp
from libc.stdint cimport int64_t

BITS = 64
Q_BITS = 32

# Explicit domain scales
X_SCALE = 24  # state values
P_SCALE = 20  # covariance values (larger dynamic range)
H_SCALE = 24  # measurement matrix values (usually 1, but we use same unit)
Z_SCALE = 24  # measurement values

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
    cdef cnp.ndarray x_icnput = np.asarray(copy.deepcopy(arr), dtype=np.float64)
    cdef double scale_amount = 2**scale
    cdef cnp.ndarray x_scaled = np.round(x_icnput * scale_amount)
    cdef int64_t q_min = -(2**(BITS - 1))
    cdef int64_t q_max = (2**(BITS - 1) - 1)
    
    return np.clip(x_scaled, q_min, q_max).astype(np.int64)

def dequantize_array(cnp.ndarray q_arr, int scale = Q_BITS):
    """Dequantize a fixed-point int array to float array."""
    cdef cnp.ndarray x_icnput = np.asarray(copy.deepcopy(q_arr))
    cdef double scale_amount = 2**scale
    cdef cnp.ndarray x_decode = x_icnput / scale_amount
    return x_decode.astype(np.float64)

def q_mul(a_q, b_q, int a_scale = Q_BITS,
          int b_scale = Q_BITS, int out_scale = Q_BITS):
    cdef int64_t q_min = -(2**(BITS - 1))
    cdef int64_t q_max = (2**(BITS - 1) - 1)
    cdef int shift = a_scale + b_scale - out_scale
    cdef object product
    cdef int64_t shifted
    if np.isscalar(a_q) and np.isscalar(b_q):
        product = int(a_q) * int(b_q)
        shifted = product >> shift
        if shifted < q_min:
            shifted = q_min
        elif shifted > q_max:
            shifted = q_max
        return np.int64(shifted)

    cdef cnp.ndarray[cnp.int64_t] a_q_int = np.asarray(a_q, dtype=np.int64)
    cdef object b_q_int = np.asarray(b_q, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t] product_q = a_q_int * b_q_int
    cdef cnp.ndarray[cnp.int64_t] shifted_q = product_q >> shift
    return np.clip(shifted_q, q_min, q_max).astype(np.int64)

def q_mat_mul(cnp.ndarray A, cnp.ndarray B, int a_scale = Q_BITS, 
              int b_scale = Q_BITS, int out_scale = Q_BITS):
    cdef cnp.ndarray result = np.zeros((A.shape[0], B.shape[1]), dtype=np.int64)
    cdef int i, j, k, shift
    cdef int64_t acc
    cdef int64_t q_min = -(2**(BITS - 1))
    cdef int64_t q_max = (2**(BITS - 1) - 1)
    
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            acc = 0
            for k in range(A.shape[1]):
                acc += A[i, k] * B[k, j]
            shift = a_scale + b_scale - out_scale
            result[i, j] = acc >> shift
    
    return np.clip(result, q_min, q_max).astype("int64")