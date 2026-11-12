#include "ops.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

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

int main()
{
    test_tm_bias_add();
    return 0;
}