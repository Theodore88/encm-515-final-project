#include "simd_scalar_functions.h"
#include <immintrin.h>

// Use AVX/AVX2-style 256-bit vectors from immintrin to do 4 doubles at one time (because 256 / 64 bit doubles = 4)

/**
 * _mm256_loadu_pd: Load 256-bits (4 sets of 64-bit doubles from memory) - https://doc.rust-lang.org/beta/core/arch/x86_64/fn._mm256_loadu_pd.html
 * _mm256_storeu_pd: Store 256-bits - https://doc.rust-lang.org/beta/core/arch/x86_64/fn._mm256_storeu_pd.html
 * _mm256_add_pd: Add two vector registers at one time (https://acl.inf.ethz.ch/teaching/fastcode/2022/slides/07-simd-avx.pdf, https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_add_pd&ig_expand=119)
 * _mm256_fmadd_pd: Multiplies the first two arguments and then adds it to the third argument (https://doc.rust-lang.org/beta/core/arch/x86_64/fn._mm256_fmadd_pd.html)
 */

static inline double hsum256_pd(__m256d vector) {
    __m128d high_half = _mm256_extractf128_pd(vector, 1);
    __m128d low_half = _mm256_castpd256_pd128(vector);
    __m128d pairwise_sum = _mm_add_pd(low_half, high_half);
    __m128d swapped = _mm_unpackhi_pd(pairwise_sum, pairwise_sum);
    __m128d total_sum = _mm_add_sd(pairwise_sum, swapped);
    return _mm_cvtsd_f64(total_sum);
}

void matrix_transpose(const double* matrix, double* transpose_matrix) {
    for (int row = 0; row < 9; ++row) {
        for (int column = 0; column < 9; ++column) {
            transpose_matrix[column * 9 + row] = matrix[row * 9 + column];
        }
    }
}

void matrix_add_scalar(const double* matrix_A, const double* matrix_B, double* total_sum) {
    for (int index = 0; index < 81; index++) {
        total_sum[index] = matrix_A[index] + matrix_B[index];
    }
}

void matrix_multiply_scalar(const double* matrix_A, const double* matrix_B, double* product) {
    for (int row = 0; row < 9; row++) {
        for (int column = 0; column < 9; column++) {
            double total_sum = 0.0;
            for (int element = 0; element < 9; element++) {
                total_sum += matrix_A[row * 9 + element] * matrix_B[element * 9 + column];
            }
            product[row * 9 + column] = total_sum;
        }
    }
}

void covariance_predict_scalar(const double* F, const double* P, const double* Q, double* resulting_matrix) {
    double F_transpose[81];
    double propagated_covariance[81];
    double propagated_covariance_times_transpose[81];

    matrix_transpose(F, F_transpose);
    matrix_multiply_scalar(F, P, propagated_covariance);
    matrix_multiply_scalar(propagated_covariance, F_transpose, propagated_covariance_times_transpose);
    matrix_add_scalar(propagated_covariance_times_transpose, Q, resulting_matrix);
}

void matrix_add_simd(const double* matrix_A, const double* matrix_B, double* total_sum)
{    
    int index = 0;

    while (index + 4 <= 81) // 9x9 matrix means 81 elements, 4 completed at a time with SIMD
    {
        __m256d matrix_A_values = _mm256_loadu_pd(&matrix_A[index]);
        __m256d matrix_B_values = _mm256_loadu_pd(&matrix_B[index]);
        __m256d summed_values = _mm256_add_pd(matrix_A_values, matrix_B_values);
        _mm256_storeu_pd(&total_sum[index], summed_values);
        index += 4;
    }

    while (index < 81) { // 81 isn't divisible by 4, so get the remaining elements
        total_sum[index] = matrix_A[index] + matrix_B[index];
        index++;
    }
}

void matrix_multiply_simd(const double* matrix_A, const double* matrix_B, double* product) {
    double matrix_B_transpose[81];
    matrix_transpose(matrix_B, matrix_B_transpose);

    // Taking the transpose makes it easier lateer because product[row][column] = dot_product(matrix_A[row], matrix_B[column])

    for (int row = 0; row < 9; ++row) {
        const double* matrix_A_row = &matrix_A[row * 9]; // Pointer to the current row from matrix A

        for (int column = 0; column < 9; ++column) {
            const double* matrix_B_transpose_row = &matrix_B_transpose[column * 9]; // Pointer to the current column from matrix B

            __m256d accumulated_sum = _mm256_setzero_pd(); // Reset/initialize a vector register filled with 0s (4 values)

            // Compue for elements 0..3
            __m256d matrix_A_values_0 = _mm256_loadu_pd(&matrix_A_row[0]); // Load 4 elements from current row of matrix A
            __m256d matrix_B_values_0 = _mm256_loadu_pd(&matrix_B_transpose_row[0]); // Load 4 elements from the transposed matrix B
            accumulated_sum = _mm256_fmadd_pd(matrix_A_values_0, matrix_B_values_0, accumulated_sum); // Compute the MAC

            // Compute for elements 4..7
            __m256d matrix_A_values_1 = _mm256_loadu_pd(&matrix_A_row[4]); // Load the next 4 elements from current row of matrix A
            __m256d matrix_B_values_1 = _mm256_loadu_pd(&matrix_B_transpose_row[4]); // Load the corresponding 4 elements from transpose matrix B
            accumulated_sum = _mm256_fmadd_pd(matrix_A_values_1, matrix_B_values_1, accumulated_sum); // Add to the current MAC

            double total_sum = hsum256_pd(accumulated_sum);
            
            // Add the element at index 8 (the 9th element... for yaw)
            total_sum += matrix_A_row[8] * matrix_B_transpose_row[8];

            product[row * 9 + column] = total_sum;
        }
    }
}

void covariance_predict_simd(const double* F, const double* P, const double* Q, double* resulting_matrix) {
    double F_transpose[81];
    double propagated_covariance[81];
    double propagated_covariance_times_transpose[81];

    matrix_transpose(F, F_transpose); // No big SIMD optimizations for the transpose
    matrix_multiply_simd(F, P, propagated_covariance);
    matrix_multiply_simd(propagated_covariance, F_transpose, propagated_covariance_times_transpose);
    matrix_add_simd(propagated_covariance_times_transpose, Q, resulting_matrix);
}