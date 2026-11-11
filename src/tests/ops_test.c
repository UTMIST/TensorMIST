#include "ops.h"
#include <assert.h>
#include <stdio.h>

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

int main(void) {
    test_transpose();
    return 0;
}
