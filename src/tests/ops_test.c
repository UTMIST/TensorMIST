#include "ops.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

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

int main(void) {
    test_subtract();
    test_subtract_backward();
    return 0;
}
