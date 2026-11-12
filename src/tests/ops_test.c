#include "ops.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

void test_matmul() {
  float a[] = {1, 2, 3, 4, 5, 6}; // 2 x 3
  float b[] = {4, 1, 3, 5, 6, 2}; // 3 x 2
  float out[4] = {0};

  tm_matmul(a, b, out, 2, 3, 2);
  
  assert(out[0] == 28 && out[1] == 17);
  assert(out[2] == 67 && out[3] == 41);

  printf("Matrix multiplication test passed!\n");
}

int main() {
  test_matmul();
    
  return 0;
}
