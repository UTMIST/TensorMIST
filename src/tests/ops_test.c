#include "ops.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

static void test_multiply(void) {
    float a[4] = {1, -2, 0, 4}; // 2x2
    float b[4] = {3, 4, 5, -2}; // 2x2
    float out[4];

    tm_multiply(a, b, out, 2, 2);
    assert(out[0] == 3 && out[1] == -8);
    assert(out[2] == 0 && out[3] == -8);

    printf("Multiply sanity check passed.\n");
}

int main(void) {
    test_multiply();
    return 0;
}
