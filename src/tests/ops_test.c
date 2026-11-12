#include "ops.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>


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


int main() {
    printf("Hello TensorMIST\n");

    test_sum();
    test_sum_rows();
    test_sum_cols();

    printf("All sum reduction forward sanity tests passed. \n");
    
    return 0;
}
