void matrix_transpose(const double* matrix, double* transpose_matrix);

void matrix_add_scalar(const double* matrix_A, const double* matrix_B, double* total_sum);
void matrix_multiply_scalar(const double* matrix_A, const double* matrix_B, double* product);
void covariance_predict_scalar(const double* F, const double* P, const double* Q, double* resulting_matrix);

void matrix_add_simd(const double* matrix_A, const double* matrix_B, double* total_sum);
void matrix_multiply_simd(const double* matrix_A, const double* matrix_B, double* product);
void covariance_predict_simd(const double* F, const double* P, const double* Q, double* resulting_matrix);