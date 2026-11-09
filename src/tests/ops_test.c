#include "ops.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

int main() {
    printf("Hello TensorMIST");
    float a[1000] = {0}; // 100x100 
    float b[1000]; // 100x100
    tm_relu(a, b, 100, 100);
    return 0;
}