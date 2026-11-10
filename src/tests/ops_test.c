#include "ops.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

int main() {
    printf("Hello TensorMIST");
    float a[100] = {0}; // 10x10 
    float b[100]; // 10x10
    tm_relu(a, b, 10, 10);
    return 0;
}