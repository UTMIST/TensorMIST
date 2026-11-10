#include "ops.h"

void tm_relu(const float *a, float *out, size_t m, size_t n){
    float x;
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            x = AT(a, i, j, m);
            AT(out, i, j, m) = (x > 0) ? x : 0;
        }
        
    }
}

void tm_relu_backward(const float *a, const float *grad_out, float *grad_in, size_t m, size_t n){
    int relu_grad;
    for (int i = 0; i < m; i++){ 
        for (int j = 0; j < n; j++){ 
            relu_grad = (AT(a, i, j, m) > 0) ? 1 : 0;
            AT(grad_in, i, j, m) = relu_grad * AT(grad_out, i, j, m);
        }
        
    }
}
