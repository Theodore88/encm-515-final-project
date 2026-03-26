void matrix_transpose(const double* matrix, double* transpose_matrix, int N);

void matrix_add_scalar(const double* matrix_A, const double* matrix_B, double* total_sum, int N);
void matrix_multiply_scalar(const double* matrix_A, const double* matrix_B, double* product, int N);
void covariance_predict_scalar(const double* F, const double* P, const double* Q, double* resulting_matrix, int N);

void matrix_add_simd(const double* matrix_A, const double* matrix_B, double* total_sum, int N);
void matrix_multiply_simd(const double* matrix_A, const double* matrix_B, double* product, int N);
void covariance_predict_simd(const double* F, const double* P, const double* Q, double* resulting_matrix, int N);