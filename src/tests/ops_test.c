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

    return 0;
}
