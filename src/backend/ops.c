#include "ops.h"

void tm_transpose(const float *a, float *out, size_t rows, size_t cols){
    for(size_t i = 0; i<rows; i++){
        for(size_t j = 0; j<cols; j++){
            AT(out, j, i, rows) = AT(a, i, j, cols);
        }
    }  
}

void tm_transpose_backward(const float *grad_out, float *grad_a, size_t rows, size_t cols){
    for(size_t i = 0; i<rows; i++){
        for(size_t j = 0; j<cols; j++){
            AT(grad_a, i, j, cols) = AT(grad_out, j, i, rows);
        }
    }
}
