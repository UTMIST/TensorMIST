#ifndef TENSORMIST_OPS_H
#define TENSORMIST_OPS_H

#include <stddef.h>

// macro for accessing contiguous row-major matrix: 
#define AT(data, row, col, cols) ((data)[(row) * (cols) + (col)])

#ifdef __cplusplus
extern "C" {
#endif

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

// elementwise sub
void tm_subtract(const float *a, const float *b, float *out, size_t m, size_t n);
void tm_subtract_backward(const float *grad_out, float *grad_a, float *grad_b, size_t m, size_t n);

// elementwise mul
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
void tm_relu(const float *a, float *out, size_t m, size_t n);
void tm_relu_backward(const float *a, const float *grad_out,
                      float *grad_in, size_t m, size_t n);

// Softmax
void tm_softmax(const float *input, float *out, size_t m, size_t n);
void tm_softmax_backward(const float *out, const float *grad_out, float *grad_input, size_t m, size_t n);

// Sigmoid
void tm_sigmoid(const float *input, float *out, size_t m, size_t n);
void tm_sigmoid_backward(const float *out, const float *grad_out, float *grad_input, size_t m, size_t n);

// Cross entropy
void tm_cross_entropy(const float *logits, const size_t *target, float *out, size_t m, size_t n);
void tm_cross_entropy_backward(const float *logits, const size_t *target, const float *grad_out,
                               float *grad_logits, size_t m, size_t n);

// Tanh, backward takes the forward output like sigmoid
void tm_tanh(const float *input, float *out, size_t m, size_t n);
void tm_tanh_backward(const float *out, const float *grad_out, float *grad_input, size_t m, size_t n);

// Binary cross entropy, pred holds probabilities and target has the same shape
// writes the mean over all m * n elements to out[0], backward reads grad_out[0]
// logs are clamped at -100 like pytorch
void tm_binary_cross_entropy(const float *pred, const float *target, float *out, size_t m, size_t n);
void tm_binary_cross_entropy_backward(const float *pred, const float *target, const float *grad_out,
                                      float *grad_pred, size_t m, size_t n);

// Mean squared error, target has the same shape as pred
// writes the mean over all m * n elements to out[0], backward reads grad_out[0]
void tm_mse_loss(const float *pred, const float *target, float *out, size_t m, size_t n);
void tm_mse_loss_backward(const float *pred, const float *target, const float *grad_out,
                          float *grad_pred, size_t m, size_t n);

#ifdef __cplusplus
}
#endif

#endif
