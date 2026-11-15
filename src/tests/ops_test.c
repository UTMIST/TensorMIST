#include "ops.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

static void test_tm_add() {
    float a[6] = {1, 2, 3, 4, 5, 6};
    float b[6] = {-1, -2, -3, -4, -5 , -6};
    float out[6];
    float expected[6] = {0};
    int m = 2; 
    int n = 3;

    tm_add(a, b, out, m, n);
}
    
static void test_tm_bias_add()
{
    float a[6] = {1, 2, 3, 1, 2, 3};
    float bias[3] = {3, 2, 1};
    float expected[6] = {4, 4, 4, 4, 4, 4};
    float out[6] = {0};
    int m = 2;
    int n = 3;

    tm_bias_add(a, bias, out, m, n);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            assert(AT(expected, i, j, n) == AT(out, i, j, n));
        }
    }
}


void test_sum() {
    float a[] = {1, 2, 3, 4, 5, 6};
    float out;

    tm_sum(a, &out, 6);

    assert(fabsf(out - 21.0f) < 1e-5f);
    printf("tm_sum passed\n");
}

void test_sum_rows() {
    // [1, 2, 3]
    // [4, 5, 6]
    float a[] = {1, 2, 3, 4, 5, 6};
    float out[2];

    tm_sum_rows(a, out, 2, 3);

    assert(fabsf(out[0] - 6.0f) < 1e-5f);
    assert(fabsf(out[1] - 15.0f) < 1e-5f);
    printf("tm_sum_rows passed\n");
}

void test_sum_cols() {
    // [1 2 3]
    // [4 5 6]
    float a[] = {1, 2, 3, 4, 5, 6};
    float out[3];

    tm_sum_cols(a, out, 2, 3);

    assert(fabsf(out[0] - 5.0f) < 1e-5f);
    assert(fabsf(out[1] - 7.0f) < 1e-5f);
    assert(fabsf(out[2] - 9.0f) < 1e-5f);
    printf("tm_sum_cols passed\n");
}

static void test_transpose(void) {
    // A 3x2 matrix becomes a 2x3 matrix, both stored row-major.
    const float a[] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f};
    const float expected[] = {1.1f, 3.3f, 5.5f, 2.2f, 4.4f, 6.6f};
    float out[6] = {0};
    tm_transpose(a, out, 3, 2);

    for (size_t i = 0; i < 6; i++) {
        assert(out[i] == expected[i]);
    }
}

static void test_multiply(void) {
    float a[4] = {1, -2, 0, 4}; // 2x2
    float b[4] = {3, 4, 5, -2}; // 2x2
    float out[4];

    tm_multiply(a, b, out, 2, 2);
    assert(out[0] == 3 && out[1] == -8);
    assert(out[2] == 0 && out[3] == -8);

    printf("Multiply sanity check passed.\n");
}

static void test_subtract(void) {
    // 2x3 rather than square, so mixing up m and n in the indexing gets caught.
    const float a[6] = {5.0f, 0.5f, -1.0f, 2.0f, 8.0f, 3.0f};
    const float b[6] = {2.0f, 1.5f, -3.0f, 6.0f, -0.25f, 4.0f};
    const float expected[6] = {3.0f, -1.0f, 2.0f, -4.0f, 8.25f, -1.0f};
    float out[6] = {7, 7, 7, 7, 7, 7}; // not 0, so out has to be overwritten

    tm_subtract(a, b, out, 2, 3);
    for (size_t i = 0; i < 6; i++) {
        assert(out[i] == expected[i]);
    }
    printf("tm_subtract passed\n");
}

static void test_subtract_backward(void) {
    const float grad_out[6] = {1.0f, -2.0f, 0.5f, 3.0f, -0.75f, 4.0f};
    const float expected_grad_b[6] = {-1.0f, 2.0f, -0.5f, -3.0f, 0.75f, -4.0f};
    float grad_a[6] = {7, 7, 7, 7, 7, 7}; // overwritten too, not added to
    float grad_b[6] = {7, 7, 7, 7, 7, 7};

    tm_subtract_backward(grad_out, grad_a, grad_b, 2, 3);
    for (size_t i = 0; i < 6; i++) {
        assert(grad_a[i] == grad_out[i]);
        assert(grad_b[i] == expected_grad_b[i]);
    }
    printf("tm_subtract_backward passed\n");
}

void test_matmul() {
  float a[] = {1, 2, 3, 4, 5, 6}; // 2 x 3
  float b[] = {4, 1, 3, 5, 6, 2}; // 3 x 2
  float out[4] = {0};

  tm_matmul(a, b, out, 2, 3, 2);
  
  assert(out[0] == 28 && out[1] == 17);
  assert(out[2] == 67 && out[3] == 41);

  printf("Matrix multiplication test passed!\n");
}

int main() {
    printf("Hello TensorMIST\n");
    
    float a[100] = {0}; // 10x10 
    float b[100]; // 10x10
    tm_relu(a, b, 10, 10);

    test_sum();
    test_sum_rows();
    test_sum_cols();

    test_tm_bias_add();
    test_tm_add();
    test_transpose();
    test_multiply();
    
    test_subtract();
    test_subtract_backward();
    test_matmul();
}