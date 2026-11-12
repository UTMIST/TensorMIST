#include "ops.h"

void tm_subtract(const float *a, const float *b, float *out, size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            AT(out, i, j, n) = AT(a, i, j, n) - AT(b, i, j, n);
        }
    }
}

void tm_subtract_backward(const float *grad_out, float *grad_a, float *grad_b, size_t m, size_t n) {
    // d(a - b)/da = 1 and d(a - b)/db = -1, so grad_b is grad_out negated.
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            AT(grad_a, i, j, n) = AT(grad_out, i, j, n);
            AT(grad_b, i, j, n) = -AT(grad_out, i, j, n);
        }
    }
}
