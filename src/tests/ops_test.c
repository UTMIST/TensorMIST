#include "ops.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>



static void test_multiply(void) {

    // Input matrices (2x2)
    float a[] = {1, 2, 3, 4};
    float b[] = {5, 6, 7, 8};
 float out[4];

    // Expected Hadamard product
    float expected[] = {5, 12, 21, 32};

    tm_multiply(a, b, out, 2, 2);

    // Check forward operation
    for (size_t i = 0; i < 4; i++) {
        assert(fabsf(out[i] - expected[i]) < 1e-5f);
    }

    printf("tm_multiply passed\n");
}

static void test_multiply_backward(void) {

    float a[] = {1, 2, 3, 4};
    float b[] = {5, 6, 7, 8};

    float grad_out[] = {1, 2, 3, 4};

    float grad_a[4];
    float grad_b[4];

    float expected_grad_a[] = {5, 12, 21, 32};
    float expected_grad_b[] = {1, 4, 9, 16};

    tm_multiply_backward(a, b, grad_out,
                         grad_a, grad_b, 2, 2);

    // Check gradient with respect to A
    for (size_t i = 0; i < 4; i++) {
        assert(fabsf(grad_a[i] - expected_grad_a[i]) < 1e-5f);
    }

    // Check gradient with respect to B
    for (size_t i = 0; i < 4; i++) {
        assert(fabsf(grad_b[i] - expected_grad_b[i]) < 1e-5f);
    }

    printf("tm_multiply_backward passed\n");
}


int main() {
    test_multiply();
    test_multiply_backward();

    printf("All tests passed!\n");
    return 0;
}

