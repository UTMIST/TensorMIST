#include "ops.h"

void tm_add(const float *a, const float *b, float *out, size_t m, size_t n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            AT(out, i, j, n) = AT(a, i, j, n) + AT(b, i, j, n);
        }
    }
}

void tm_add_backward(const float *grad_out, float *grad_a, float *grad_b, size_t m, size_t n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            AT(grad_a, i, j, n) = AT(grad_out, i, j, n);
            AT(grad_b, i, j, n) = AT(grad_out, i, j, n);
        }
    }
}
