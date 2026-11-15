#include "ops.h"

void tm_relu(const float *a, float *out, size_t m, size_t n){
    float x;
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            x = AT(a, i, j, n);
            AT(out, i, j, n) = (x > 0) ? x : 0;
        }
        
    }
}

void tm_relu_backward(const float *a, const float *grad_out, float *grad_in, size_t m, size_t n){
    int relu_grad;
    for (int i = 0; i < m; i++){ 
        for (int j = 0; j < n; j++){ 
            relu_grad = (AT(a, i, j, n) > 0) ? 1 : 0;
            AT(grad_in, i, j, n) = relu_grad * AT(grad_out, i, j, n);
        }
        
    }
}
