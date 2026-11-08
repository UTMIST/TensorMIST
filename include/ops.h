#ifndef TENSORMIST_OPS_H
#define TENSORMIST_OPS_H

#include <stddef.h>

// macro for accessing contiguous row-major matrix: 
#define AT(data, row, col, cols) ((data)[(row) * (cols) + (col)])

// a[m, k] b[k, n] out[m, n]
void tm_matmul(const float *a, const float *b, float *out, size_t m, size_t k, size_t n);
void tm_matmul_backward(const float *a, const float *b, const float *grad_out,
                        float *grad_a, float *grad_b, size_t m, size_t k, size_t n);

// copies values into a separate output, does not create a view (we will do this later)
void tm_transpose(const float *a, float *out, size_t rows, size_t cols);
void tm_transpose_backward(const float *grad_out, float *grad_a, size_t rows, size_t cols);

// elementwise add, each buffer has m * n elements
void tm_add(const float *a, const float *b, float *out, size_t m, size_t n);
void tm_add_backward(const float *grad_out, float *grad_a, float *grad_b, size_t m, size_t n);

void tm_subtract(const float *a, const float *b, float *out, size_t m, size_t n);
void tm_subtract_backward(const float *grad_out, float *grad_a, float *grad_b, size_t m, size_t n);

// elementwise
void tm_multiply(const float *a, const float *b, float *out, size_t m, size_t n);
void tm_multiply_backward(const float *a, const float *b, const float *grad_out,
                          float *grad_a, float *grad_b, size_t m, size_t n);

// scalar mul
void tm_scalar_multiply(const float *a, float scalar, float *out, size_t m, size_t n);
void tm_scalar_multiply_backward(const float *grad_out, float scalar,
                                 float *grad_a, size_t m, size_t n);

// sum reductions
void tm_sum(const float *a, float *out, size_t n);
void tm_sum_rows(const float *a, float *out, size_t m, size_t n);
void tm_sum_cols(const float *a, float *out, size_t m, size_t n);

void tm_sum_backward(const float *grad_out, float *grad_a, size_t n);
void tm_sum_rows_backward(const float *grad_out, float *grad_a, size_t m, size_t n);
void tm_sum_cols_backward(const float *grad_out, float *grad_a, size_t m, size_t n);

// bias add 
void tm_bias_add(const float *a, const float *bias, float *out,
                 size_t m, size_t n);
void tm_bias_add_backward(const float *grad_out, float *grad_a, float *grad_bias,
                          size_t m, size_t n);

// ReLU
void tm_relu(const float *a, float *out, size_t n);
void tm_relu_backward(const float *a, const float *grad_out,
                      float *grad_in, size_t n);

#endif