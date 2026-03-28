import numpy as np
BITS = 64
Q_BITS = 32

# Explicit domain scales
X_SCALE = 10 # state values
P_SCALE = 14  # covariance values (larger dynamic range)
H_SCALE = 10 # measurement matrix values (usually 1, but we use same unit)
Z_SCALE = 10  # measurement values


q_min = -(2**(BITS - 1))
q_max = (2**(BITS - 1) - 1)

def quantize(value: float, scale: float = Q_BITS) -> int:
    """Quantize a float to fixed-point int."""
    q_copy = value

    scale_amount = 1 << scale
    x_scaled = q_copy * scale_amount

    if x_scaled < q_min:
        return int(q_min)
    elif x_scaled > q_max:
        return int(q_max)

    return int(x_scaled)

def dequantize(q_value: int, scale: float = Q_BITS) -> float:
    """Dequantize a fixed-point int to float."""
    q_copy = np.asarray(q_value, dtype=np.int64)
    scale_amount = 1 << scale

    x_decode = (q_copy/scale_amount)
    return float(x_decode)

def quantize_array(arr: np.ndarray, scale: int = Q_BITS) -> np.ndarray:
    """Quantize a float array to fixed-point int array."""
    x_input = np.asarray(arr, dtype=np.float64)

    scale_amount = 1 << scale

    clip = np.clip(np.round(np.multiply(x_input, scale_amount)), q_min, q_max)

    return clip.astype(np.int64)

def dequantize_array(q_arr: np.ndarray, scale: int = Q_BITS) -> np.ndarray:
  """Dequantize a fixed-point int array to float array."""
  x_input = np.asarray((q_arr))

  scale_amount = 1 << scale

  x_decode = np.divide(x_input, scale_amount)

  return x_decode.astype(np.float64)

def q_mat_mul(A: np.ndarray, B: np.ndarray, a_scale: int = Q_BITS, b_scale: int = Q_BITS,
              out_scale: int = Q_BITS) -> np.ndarray:
    A_in = np.asarray(A, dtype=np.int64)
    B_in = np.asarray(B, dtype=np.int64)

    # Vectorized matrix multiply using BLAS
    product = A_in @ B_in
    shift = a_scale + b_scale - out_scale 

    rounding_factor = 1 << (shift - 1)  
    result = (product + rounding_factor) >> (shift)

    return np.clip(result, q_min, q_max).astype(np.int64)

def q_mul(a_q: np.ndarray, b_q: np.ndarray, a_scale: int = Q_BITS,
          b_scale: int = Q_BITS, out_scale: int = Q_BITS) -> np.ndarray:
    # Vectorized element-wise multiply
    A_in = np.asarray(a_q, dtype=np.int64)
    B_in = np.asarray(b_q, dtype=np.int64)
    product = np.multiply(A_in, B_in)
    shift = a_scale + b_scale - out_scale 
    rounding_factor = 1 << (shift - 1)
    result = (product + rounding_factor) >> (shift)

    return np.clip(result, q_min, q_max).astype(np.int64)
