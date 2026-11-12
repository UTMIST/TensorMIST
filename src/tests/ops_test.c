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
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            assert(AT(expected, i, j, n) == AT(out, i, j, n));
        }
    }
}

int main() {
    printf("Hello TensorMIST");
    
    test_tm_add();
    return 0;
}