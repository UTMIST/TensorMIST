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

static void test_scalar_multiply(void) {
    // 2x3 matrix test
    const float a[6] = {1.0f, -2.0f, 0.0f, 4.0f, -5.0f, 3.0f};
    const float scalar = 2.5f;
    const float expected[6] = {2.5f, -5.0f, 0.0f, 10.0f, -12.5f, 7.5f};
    float out[6] = {0};

    tm_scalar_multiply(a, scalar, out, 2, 3);

    for (size_t i = 0; i < 6; i++) {
        assert(fabsf(out[i] - expected[i]) < 1e-5f);
    }

    printf("tm_scalar_multiply passed\n");
}


static void test_scalar_multiply_backward(void) {
    // Incoming gradients test
    const float grad_out[6] = {1.0f, -2.0f, 0.5f, 3.0f, -0.75f, 4.0f};
    const float scalar = 3.0f;

    const float expected_grad_a[6] = {3.0f, -6.0f, 1.5f, 9.0f, -2.25f, 12.0f};
    float grad_a[6] = {0};

    tm_scalar_multiply_backward(grad_out, scalar, grad_a, 2, 3);

    for (size_t i = 0; i < 6; i++) {
        assert(fabsf(grad_a[i] - expected_grad_a[i]) < 1e-5f);
    }

    printf("tm_scalar_multiply_backward passed\n");
}

static void test_sigmoid(void) {
    const float input[6] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 10.0f};

    const float expected[6] = {
        0.119202923f, 0.268941421f, 0.5f,
        0.731058579f, 0.880797078f, 0.999954602f
    };

    float out[6] = {7, 7, 7, 7, 7, 7};

    tm_sigmoid(input, out, 2, 3);

    for (size_t i = 0; i < 6; i++) {
        assert(fabsf(out[i] - expected[i]) < 1e-6f);
    }

    printf("tm_sigmoid passed\n");
}

static void test_sigmoid_backward(void) {
    // Using exact binary fractions for sigmoid outputs.
    const float out[6] = {
        0.5f, 0.25f, 0.75f,
        0.125f, 0.875f, 0.625f
    };

    const float grad_out[6] = {
        1.0f, 2.0f, -1.0f,
        4.0f, -2.0f, 0.5f
    };

    const float expected[6] = {
        0.25f, 0.375f, -0.1875f,
        0.4375f, -0.21875f, 0.1171875f
    };

    float grad_input[6] = {7, 7, 7, 7, 7, 7};

    tm_sigmoid_backward(out, grad_out, grad_input, 2, 3);

    for (size_t i = 0; i < 6; i++) {
        assert(grad_input[i] == expected[i]);
    }

    printf("tm_sigmoid_backward passed\n");
}

static void test_softmax(void) {
    const float input[6] = {
        1.0f, 2.0f, 3.0f,
        0.0f, 0.0f, 0.0f
    };

    const float expected[6] = {
        0.090030573f, 0.244728472f, 0.665240955f,
        0.333333333f, 0.333333333f, 0.333333333f
    };

    float out[6] = {7, 7, 7, 7, 7, 7};

    tm_softmax(input, out, 2, 3);

    for (size_t i = 0; i < 6; i++) {
        assert(fabsf(out[i] - expected[i]) < 1e-6f);
    }

    // Each row's probabilities should sum to 1.
    for (size_t i = 0; i < 2; i++) {
        float sum = 0.0f;

        for (size_t j = 0; j < 3; j++) {
            sum += out[i * 3 + j];
        }

        assert(fabsf(sum - 1.0f) < 1e-6f);
    }

    printf("tm_softmax passed\n");
}

static void test_softmax_backward(void) {
    const float out[6] = {
        0.5f, 0.25f, 0.25f,
        0.2f, 0.3f, 0.5f
    };

    const float grad_out[6] = {
        1.0f, 2.0f, -1.0f,
        1.0f, -2.0f, 3.0f
    };

    const float expected[6] = {
        0.125f, 0.3125f, -0.4375f,
        -0.02f, -0.93f, 0.95f
    };

    float grad_input[6] = {7, 7, 7, 7, 7, 7};

    tm_softmax_backward(out, grad_out, grad_input, 2, 3);

    for (size_t i = 0; i < 6; i++) {
        assert(fabsf(grad_input[i] - expected[i]) < 1e-6f);
    }

    printf("tm_softmax_backward passed\n");
}

static void test_cross_entropy(void) {
    const float logits[6] = {
        2.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };

    const size_t target[2] = {1, 2};

    const float expected[2] = {
        1.407605964f,
        1.098612289f
    };

    float out[2] = {7, 7};

    tm_cross_entropy(logits, target, out, 2, 3);

    for (size_t i = 0; i < 2; i++) {
        assert(fabsf(out[i] - expected[i]) < 1e-5f);
    }

    printf("tm_cross_entropy passed\n");
}

static void test_cross_entropy_backward(void) {
    const float logits[6] = {
        2.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };

    const size_t target[2] = {1, 2};

    const float grad_out[2] = {
        1.0f, 0.5f
    };

    const float expected[6] = {
        0.665240955f, -0.755271528f, 0.090030573f,
        0.166666667f, 0.166666667f, -0.333333333f
    };

    float grad_logits[6] = {7, 7, 7, 7, 7, 7};

    tm_cross_entropy_backward(
        logits, target, grad_out, grad_logits, 2, 3
    );

    for (size_t i = 0; i < 6; i++) {
        assert(fabsf(grad_logits[i] - expected[i]) < 1e-6f);
    }

    printf("tm_cross_entropy_backward passed\n");
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

    test_scalar_multiply();
    test_scalar_multiply_backward();

    test_sigmoid();
    test_sigmoid_backward();

    test_softmax();
    test_softmax_backward();

    test_cross_entropy();
    test_cross_entropy_backward();
}