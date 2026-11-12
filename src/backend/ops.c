#include "ops.h"

void tm_multiply(const float *a, const float *b, float *out, size_t m, size_t n) {
    for (size_t i = 0; i < m * n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void tm_multiply_backward(const float *a, const float *b, const float *grad_out,
                          float *grad_a, float *grad_b, size_t m, size_t n) {
    for (size_t i = 0; i < m * n; ++i) {
        // Chain rule for each element: d(a * b)/da = b, d(a * b)/db = a.
        grad_a[i] = grad_out[i] * b[i];
        grad_b[i] = grad_out[i] * a[i];
    }
}
