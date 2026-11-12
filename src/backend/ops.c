#include "ops.h"

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