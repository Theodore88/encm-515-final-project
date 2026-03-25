import numpy as np
cimport numpy as cnp

cnp.import_array()

cdef extern from "simd_scalar_functions.h":
    void c_matrix_transpose "matrix_transpose"(const double* matrix, double* transpose_matrix)
    void c_matrix_add_scalar "matrix_add_scalar"(const double* matrix_A, const double* matrix_B, double* total_sum)
    void c_matrix_multiply_scalar "matrix_multiply_scalar"(const double* matrix_A, const double* matrix_B, double* product)
    void c_covariance_predict_scalar "covariance_predict_scalar"(const double* F, const double* P, const double* Q, double* resulting_matrix)

    void c_matrix_add_simd "matrix_add_simd"(const double* matrix_A, const double* matrix_B, double* total_sum)
    void c_matrix_multiply_simd "matrix_multiply_simd"(const double* matrix_A, const double* matrix_B, double* product)
    void c_covariance_predict_simd "covariance_predict_simd"(const double* F, const double* P, const double* Q, double* resulting_matrix)

def matrix_transpose(cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] matrix):
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] transpose_matrix = np.zeros((9, 9), dtype=np.float64) # Define the return type (populate with zeros)
    c_matrix_transpose(<double*>matrix.data, <double*>transpose_matrix.data) # Call the C function
    return transpose_matrix

def matrix_add_scalar(cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] matrix_A,
                      cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] matrix_B):
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] total_sum = np.zeros((9, 9), dtype=np.float64)
    c_matrix_add_scalar(<double*>matrix_A.data, <double*>matrix_B.data, <double*>total_sum.data)
    return total_sum


def matrix_add_simd(cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] matrix_A,
                    cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] matrix_B):
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] total_sum = np.zeros((9, 9), dtype=np.float64)
    c_matrix_add_simd(<double*>matrix_A.data, <double*>matrix_B.data, <double*>total_sum.data)
    return total_sum


def matrix_multiply_scalar(cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] matrix_A,
                           cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] matrix_B):
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] product = np.zeros((9, 9), dtype=np.float64)
    c_matrix_multiply_scalar(<double*>matrix_A.data, <double*>matrix_B.data, <double*>product.data)
    return product


def matrix_multiply_simd(cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] matrix_A,
                         cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] matrix_B):
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] product = np.zeros((9, 9), dtype=np.float64)
    c_matrix_multiply_simd(<double*>matrix_A.data, <double*>matrix_B.data, <double*>product.data)
    return product


def covariance_predict_scalar(cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] F,
                              cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] P,
                              cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] Q):
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] resulting_matrix = np.zeros((9, 9), dtype=np.float64)
    c_covariance_predict_scalar(<double*>F.data, <double*>P.data, <double*>Q.data, <double*>resulting_matrix.data)
    return resulting_matrix


def covariance_predict_simd(cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] F,
                            cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] P,
                            cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] Q):
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] resulting_matrix = np.zeros((9, 9), dtype=np.float64)
    c_covariance_predict_simd(<double*>F.data, <double*>P.data, <double*>Q.data, <double*>resulting_matrix.data)
    return resulting_matrix