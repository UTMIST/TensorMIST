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

void tm_bias_add(const float *a, const float *bias, float *out, size_t m, size_t n)
{
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++)
		{
			AT(out, i, j, n) = AT(a, i, j, n) + AT(bias, j, 0, 1);
		}
	}
}
void tm_bias_add_backward(const float *grad_out, float *grad_a, float *grad_bias, size_t m, size_t n)
{
	for (int j = 0; j < n; j++)
	{
		float running_bias_grad = 0;
		for (int i = 0; i < m; i++)
		{
			AT(grad_a, i, j, n) = AT(grad_out, i, j, n);
			running_bias_grad += AT(grad_out, i, j, n);
		}
		AT(grad_bias, j, 0, 1) = running_bias_grad;
	}
}
