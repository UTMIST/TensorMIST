#include "ops.h"
// (Hadamard) multiplication Impplemenation
void tm_multiply(const float *a, const float *b,
                 float *out, size_t m, size_t n) {

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            AT(out, i, j, n) =
                AT(a, i, j, n) * AT(b, i, j, n);
        }
    }
}
// Backward operations Implementations
void tm_multiply_backward(const float *a, const float *b,
                          const float *grad_out,
                          float *grad_a, float *grad_b,
                          size_t m, size_t n) {

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {

            AT(grad_a, i, j, n) =
                AT(grad_out, i, j, n) * AT(b, i, j, n);

            AT(grad_b, i, j, n) =
                AT(grad_out, i, j, n) * AT(a, i, j, n);
        }
    }
}
// hi