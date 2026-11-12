#include "ops.h"


void tm_sum(const float *a, float *out, size_t n){

    float sum = 0.0f;

    for(size_t i = 0; i < n; i++){
        sum += a[i];
    }

    out[0] = sum;

}

void tm_sum_rows(const float *a, float *out, size_t m, size_t n){

    for(size_t i = 0; i < m ; i++){
        float row_sum = 0.0f;

        for(size_t j = 0; j < n; j++){
            row_sum += AT(a, i, j, n);
        }

        out[i] = row_sum;
    }

}
void tm_sum_cols(const float *a, float *out, size_t m, size_t n){

    for(size_t i = 0; i < n ; i++){
        float col_sum = 0.0f;

        for(size_t j = 0; j < m; j++){
            col_sum += AT(a, j, i, n);
        }

        out[i] = col_sum;
    }

}

void tm_sum_backward(const float *grad_out, float *grad_a, size_t n){

    for(size_t i = 0; i < n; i++){
        grad_a[i] = grad_out[0];
    }

}

void tm_sum_rows_backward(const float *grad_out, float *grad_a, size_t m, size_t n){

    for(size_t i = 0; i < m; i++){
        for(size_t j = 0; j < n; j++){
            AT(grad_a, i, j, n) = grad_out[i];
        }
    }

}
void tm_sum_cols_backward(const float *grad_out, float *grad_a, size_t m, size_t n){

    for(size_t i = 0; i < n; i++){
        for(size_t j = 0; j < m; j++){
            AT(grad_a, j, i, n) = grad_out[i];
        }
    }

}
