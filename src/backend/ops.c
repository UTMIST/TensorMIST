#include "ops.h"

void tm_matmul(const float *a, const float *b, float *out, size_t m, size_t k, size_t n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      for (int l = 0; l < k; l++) {
		AT(out, j, i, n) += AT(a, j, l, k) * AT(b, l, i, n);
      }
    }
  }
}

void tm_matmul_backward(const float *a, const float *b, const float *grad_out,
                        float *grad_a, float *grad_b, size_t m, size_t k, size_t n) {
  // dL/dA = (dL/d(AB))B^T
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < m; j++) {
      for (int l = 0; l < n; l++) {
		AT(grad_a, j, i, k) += AT(grad_out, j, l, n) * AT(b, l, i, k);
      }
    }
  }
  // dL/dB = A^T(dL/d(AB))
  for (int i = 0; i < k; i++) {
    for (int l = 0; l < n; l++) {
      for (int j = 0; j < m; j++) {
		AT(grad_b, i, l, n) += AT(a, i, j, m) * AT(grad_out, j, l, n);
      }
    }
  }
}
