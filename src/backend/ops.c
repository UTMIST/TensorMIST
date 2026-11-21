#include "ops.h"
#include <math.h>

void tm_softmax(const float *input, float *out, size_t m, size_t n) {
    float total;
    float max_val;
    for (int i = 0; i < m; i++) {
        total = 0;
        max_val = -INFINITY;

        for (int j = 0; j < n; j++) {
            if (AT(input, i, j, n) > max_val) {
                max_val = AT(input, i, j, n);
            }
        }

        for (int j = 0; j < n; j++) {
            total += expf(AT(input, i, j, n) - max_val);
        }

        for (int j = 0; j < n; j++) {
            AT(out, i, j, n) = expf(AT(input, i, j, n) - max_val) / total; 
        }
    }
}

void tm_softmax_backward(const float *out, const float *grad_out, float *grad_input, size_t m, size_t n) {
    float total;
    for (int i = 0; i < m; i++){
        total = 0;
        for (int j = 0; j < n; j++){
            total += AT(grad_out, i, j, n) * AT(out, i, j, n);
        }

        for (int j = 0; j < n; j++){
            AT(grad_input, i, j, n) = AT(out, i, j, n) * (AT(grad_out, i, j, n) - total);
        }
    }

}

void tm_sigmoid(const float *input, float *out,size_t m, size_t n){
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            AT(out, i, j, n) = 1 / (expf(-AT(input, i, j, n)) + 1);
        }
    }
}

void tm_sigmoid_backward(const float *out, const float *grad_out, float *grad_input, size_t m, size_t n) {
    float x;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            x = AT(out, i, j, n);
            AT(grad_input, i, j, n) = AT(grad_out, i, j, n) * (x) * (1 - x);
        }
    }
}

void tm_cross_entropy(const float *logits, const size_t *target, float *out, size_t m, size_t n) {
    float total;
    float max_val; 
    for (int i = 0; i < m; i++) {
        total = 0;
        max_val = -INFINITY;
        for (int j = 0; j < n; j++) {
            if (AT(logits, i, j, n) > max_val) {
                max_val = AT(logits, i, j, n);
            }
        }

        for (int j = 0; j < n; j++) {
            total += expf(AT(logits, i, j, n) - max_val);
        }

        out[i] = logf(total) + max_val - AT(logits, i, target[i], n);
    }
}

void tm_cross_entropy_backward(const float *logits, const size_t *target, const float *grad_out, float *grad_logits, size_t m, size_t n) {
    float max_val;
    float total;

    for (int i = 0; i < m; i++) {
        max_val = -INFINITY;
        total = 0.0f;

        for (int j = 0; j < n; j++) {
            if (AT(logits, i, j, n) > max_val) {
                max_val = AT(logits, i, j, n);
            }
        }

        for (int j = 0; j < n; j++) {
            total += expf(AT(logits, i, j, n) - max_val);
        }

        for (int j = 0; j < n; j++) {
            float prob = expf(AT(logits, i, j, n) - max_val) / total;

            float grad = prob - (j == target[i] ? 1.0f : 0.0f);

            AT(grad_logits, i, j, n) = grad * grad_out[i];
        }
    }
}

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

void tm_multiply(const float *a, const float *b, float *out, size_t m, size_t n) {
    for (size_t i = 0; i < m * n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void tm_multiply_backward(const float *a, const float *b, const float *grad_out,
                          float *grad_a, float *grad_b, size_t m, size_t n) {
    for (size_t i = 0; i < m * n; ++i) {
        // Chain rule for each element: d(a * b)/da = b, d(a * b)/db = a.
        grad_a[i] = grad_out[i] * b[i];
        grad_b[i] = grad_out[i] * a[i];
    }
}

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

void tm_matmul(const float *a, const float *b, float *out, size_t m, size_t k, size_t n) {
    for (size_t j = 0; j < m; j++) {
        for (size_t i = 0; i < n; i++) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; l++) {
                sum += AT(a, j, l, k) * AT(b, l, i, n);
            }
            AT(out, j, i, n) = sum;
        }
    }
}

void tm_matmul_backward(const float *a, const float *b, const float *grad_out,
                        float *grad_a, float *grad_b, size_t m, size_t k, size_t n) {
    // dL/dA[j,i] = sum_l grad_out[j,l] * B[i,l]
    for (size_t j = 0; j < m; j++) {
        for (size_t i = 0; i < k; i++) {
            float sum = 0.0f;
            for (size_t l = 0; l < n; l++) {
                sum += AT(grad_out, j, l, n) * AT(b, i, l, n);
            }
            AT(grad_a, j, i, k) = sum;
        }
    }

    // dL/dB[i,l] = sum_j A[j,i] * grad_out[j,l]
    for (size_t i = 0; i < k; i++) {
        for (size_t l = 0; l < n; l++) {
            float sum = 0.0f;
            for (size_t j = 0; j < m; j++) {
                sum += AT(a, j, i, k) * AT(grad_out, j, l, n);
            }
            AT(grad_b, i, l, n) = sum;
        }
    }
}

void tm_scalar_multiply(const float *a, float scalar,
                         float *out, size_t m, size_t n) {
    for (size_t i = 0; i < m * n; i++) {
        out[i] = a[i] * scalar;
    }
}

void tm_scalar_multiply_backward(const float *grad_out, float scalar,
                                 float *grad_a, size_t m, size_t n) {
    for (size_t i = 0; i < m * n; i++) {
        grad_a[i] = grad_out[i] * scalar;
    }
}

void tm_tanh(const float *input, float *out, size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            AT(out, i, j, n) = tanhf(AT(input, i, j, n));
        }
    }
}

void tm_tanh_backward(const float *out, const float *grad_out, float *grad_input,
                      size_t m, size_t n) {
    // d tanh(x)/dx = 1 - tanh(x)^2, so the forward output is all this needs.
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float y = AT(out, i, j, n);
            AT(grad_input, i, j, n) = AT(grad_out, i, j, n) * (1.0f - y * y);
        }
    }
}

void tm_binary_cross_entropy(const float *pred, const float *target, float *out,
                             size_t m, size_t n) {
    if (m == 0 || n == 0) {
        out[0] = NAN; // nothing to average, same as pytorch's mean of an empty tensor
        return;
    }
    double sum = 0.0; // double so the mean stays accurate over many elements
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float p = AT(pred, i, j, n);
            float y = AT(target, i, j, n);
            float log_p = logf(p);
            float log_1mp = log1pf(-p);
            // Clamp both logs at -100 like pytorch, so a pred of exactly 0 or 1 stays finite.
            if (log_p < -100.0f) {
                log_p = -100.0f;
            }
            if (log_1mp < -100.0f) {
                log_1mp = -100.0f;
            }
            sum += -(y * log_p + (1.0f - y) * log_1mp);
        }
    }
    out[0] = (float)(sum / ((double)m * (double)n));
}

void tm_binary_cross_entropy_backward(const float *pred, const float *target, const float *grad_out,
                                      float *grad_pred, size_t m, size_t n) {
    if (m == 0 || n == 0) {
        return; // grad_pred is empty, and m * n below would be 0
    }
    // dL/dp = (p - y) / (p * (1 - p)), and taking the mean divides each one by m * n.
    // The denominator is clamped at 1e-12 like pytorch, so a pred of 0 or 1 never divides by 0.
    float scale = grad_out[0] / (float)(m * n);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float p = AT(pred, i, j, n);
            float y = AT(target, i, j, n);
            float denom = p * (1.0f - p);
            if (denom < 1e-12f) {
                denom = 1e-12f;
            }
            AT(grad_pred, i, j, n) = scale * (p - y) / denom;
        }
    }
}

void tm_mse_loss(const float *pred, const float *target, float *out, size_t m, size_t n) {
    if (m == 0 || n == 0) {
        out[0] = NAN; // nothing to average, same as pytorch's mean of an empty tensor
        return;
    }
    double sum = 0.0;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float diff = AT(pred, i, j, n) - AT(target, i, j, n);
            sum += diff * diff;
        }
    }
    out[0] = (float)(sum / ((double)m * (double)n));
}

void tm_mse_loss_backward(const float *pred, const float *target, const float *grad_out,
                          float *grad_pred, size_t m, size_t n) {
    if (m == 0 || n == 0) {
        return; // grad_pred is empty, and m * n below would be 0
    }
    // No 1/2 in front of the square here (same as pytorch), so its derivative keeps the 2.
    float scale = 2.0f * grad_out[0] / (float)(m * n);
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            AT(grad_pred, i, j, n) = scale * (AT(pred, i, j, n) - AT(target, i, j, n));
        }
    }
}
