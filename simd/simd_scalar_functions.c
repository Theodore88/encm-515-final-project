#include "simd_scalar_functions.h"
#include <immintrin.h>
#include <stdlib.h>

// Use AVX/AVX2-style 256-bit vectors from immintrin to do 4 doubles at one time (because 256 / 64 bit doubles = 4)

/**
 * _mm256_loadu_pd: Load 256-bits (4 sets of 64-bit doubles from memory) - https://doc.rust-lang.org/beta/core/arch/x86_64/fn._mm256_loadu_pd.html
 * _mm256_storeu_pd: Store 256-bits - https://doc.rust-lang.org/beta/core/arch/x86_64/fn._mm256_storeu_pd.html
 * _mm256_add_pd: Add two vector registers at one time (https://acl.inf.ethz.ch/teaching/fastcode/2022/slides/07-simd-avx.pdf, https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_add_pd&ig_expand=119)
 * _mm256_fmadd_pd: Multiplies the first two arguments and then adds it to the third argument (https://doc.rust-lang.org/beta/core/arch/x86_64/fn._mm256_fmadd_pd.html)
 */

static inline double hsum256_pd(__m256d vector) {
    __m128d top_half_sum = _mm256_extractf128_pd(vector, 1);
    __m128d bottom_half_sum = _mm256_castpd256_pd128(vector);
    __m128d pair_sum = _mm_add_pd(bottom_half_sum, top_half_sum);
    __m128d swapped = _mm_unpackhi_pd(pair_sum, pair_sum);
    __m128d total_sum = _mm_add_sd(pair_sum, swapped);
    return _mm_cvtsd_f64(total_sum);
}

void matrix_transpose(const double* matrix, double* transpose_matrix, int N) {
    for (int row = 0; row < N; ++row) {
        for (int column = 0; column < N; ++column) {
            transpose_matrix[column * N + row] = matrix[row * N + column];
        }
    }
}

void matrix_add_scalar(const double* matrix_A, const double* matrix_B, double* total_sum, int N) {
    int total_elements = N * N;
    for (int index = 0; index < total_elements; index++) {
        total_sum[index] = matrix_A[index] + matrix_B[index];
    }
}

void matrix_multiply_scalar(const double* matrix_A, const double* matrix_B, double* product, int N) {
    for (int row = 0; row < N; row++) {
        for (int column = 0; column < N; column++) {
            double total_sum = 0.0;
            for (int element = 0; element < N; element++) {
                total_sum += matrix_A[row * N + element] * matrix_B[element * N + column];
            }
            product[row * N + column] = total_sum;
        }
    }
}

void covariance_predict_scalar(const double* F, const double* P, const double* Q, double* resulting_matrix, int N) {
    int total_elements = N * N;

    double* F_transpose = (double*)malloc(total_elements * sizeof(double));
    double* propagated_covariance = (double*)malloc(total_elements * sizeof(double));
    double* propagated_covariance_times_transpose = (double*)malloc(total_elements * sizeof(double));

    if (F_transpose == NULL || propagated_covariance == NULL || propagated_covariance_times_transpose == NULL) {
        free(F_transpose);
        free(propagated_covariance);
        free(propagated_covariance_times_transpose);
        return;
    }

    matrix_transpose(F, F_transpose, N);
    matrix_multiply_scalar(F, P, propagated_covariance, N);
    matrix_multiply_scalar(propagated_covariance, F_transpose, propagated_covariance_times_transpose, N);
    matrix_add_scalar(propagated_covariance_times_transpose, Q, resulting_matrix, N);

    free(F_transpose);
    free(propagated_covariance);
    free(propagated_covariance_times_transpose);
}

void matrix_add_simd(const double* matrix_A, const double* matrix_B, double* total_sum, int N)
{    
    int index = 0;
    int total_elements = N * N;

    while (index + 4 <= total_elements) // 4 completed at a time with SIMD
    {
        __m256d matrix_A_values = _mm256_loadu_pd(&matrix_A[index]);
        __m256d matrix_B_values = _mm256_loadu_pd(&matrix_B[index]);
        __m256d summed_values = _mm256_add_pd(matrix_A_values, matrix_B_values);
        _mm256_storeu_pd(&total_sum[index], summed_values);
        index += 4;
    }

    while (index < total_elements) {
        total_sum[index] = matrix_A[index] + matrix_B[index];
        index++;
    }
}

void matrix_multiply_simd(const double* matrix_A, const double* matrix_B, double* product, int N) {
    int total_elements = N * N;
    double* matrix_B_transpose = (double*)malloc(total_elements * sizeof(double));

    if (matrix_B_transpose == NULL) {
        return;
    }

    matrix_transpose(matrix_B, matrix_B_transpose, N);

    // Taking the transpose makes it easier lateer because product[row][column] = dot_product(matrix_A[row], matrix_B[column])

    for (int row = 0; row < N; ++row) {
        const double* matrix_A_row = &matrix_A[row * N]; // Pointer to the current row from matrix A

        for (int column = 0; column < N; ++column) {
            const double* matrix_B_transpose_row = &matrix_B_transpose[column * N]; // Pointer to the current column from matrix B

            __m256d accumulated_sum = _mm256_setzero_pd(); // Reset/initialize a vector register filled with 0s (4 values)

            int element = 0;

            while (element + 4 <= N) {
                __m256d matrix_A_values = _mm256_loadu_pd(&matrix_A_row[element]); // Load 4 elements from current row of matrix A
                __m256d matrix_B_values = _mm256_loadu_pd(&matrix_B_transpose_row[element]); // Load the corresponding 4 elements from transpose matrix B
                accumulated_sum = _mm256_fmadd_pd(matrix_A_values, matrix_B_values, accumulated_sum); // Compute the MAC
                element += 4;
            }

            // Sum all 4 values in accumulated_sum vector (get final dot product for current row and column)
            __m128d top_half_sum = _mm256_extractf128_pd(accumulated_sum, 1);
            __m128d bottom_half_sum = _mm256_castpd256_pd128(accumulated_sum);
            __m128d pair_sum = _mm_add_pd(bottom_half_sum, top_half_sum); // Add the top and bottom halves of the vector together [a, b] + [c, d] = [a + c, b + d]
            __m128d swapped = _mm_unpackhi_pd(pair_sum, pair_sum); // Get b + d into the lower elements of the register
            __m128d reduced_sum = _mm_add_sd(pair_sum, swapped); // add it all (only adds bottom 2 which is why we swapped in the previous step)
            double total_sum = _mm_cvtsd_f64(reduced_sum); // extract into double

            while (element < N) {
                total_sum += matrix_A_row[element] * matrix_B_transpose_row[element];
                element++;
            }

            product[row * N + column] = total_sum;
        }
    }

    free(matrix_B_transpose);
}

void covariance_predict_simd(const double* F, const double* P, const double* Q, double* resulting_matrix, int N) {
    int total_elements = N * N;

    double* F_transpose = (double*)malloc(total_elements * sizeof(double));
    double* propagated_covariance = (double*)malloc(total_elements * sizeof(double));
    double* propagated_covariance_times_transpose = (double*)malloc(total_elements * sizeof(double));

    if (F_transpose == NULL || propagated_covariance == NULL || propagated_covariance_times_transpose == NULL) {
        free(F_transpose);
        free(propagated_covariance);
        free(propagated_covariance_times_transpose);
        return;
    }

    matrix_transpose(F, F_transpose, N); // No big SIMD optimizations for the transpose
    matrix_multiply_simd(F, P, propagated_covariance, N);
    matrix_multiply_simd(propagated_covariance, F_transpose, propagated_covariance_times_transpose, N);
    matrix_add_simd(propagated_covariance_times_transpose, Q, resulting_matrix, N);

    free(F_transpose);
    free(propagated_covariance);
    free(propagated_covariance_times_transpose);
}