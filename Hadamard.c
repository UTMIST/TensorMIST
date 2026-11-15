#define ROWS 2
#define COLS 3
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


int main() {

    // Example 1 in Decimal/Float
    float A[ROWS][COLS] = {
        {1.2, 2.5, 3.0},
        {4.1, 5.5, 6.2}
    };
    float B[ROWS][COLS] = {
        {2.0, 3.0, 4.0},
        {1.5, 2.0, 3.0}
    };

    float C[ROWS][COLS]; //Output Matrix here

    // CHECK 1: Ensure matrix dimensions are valid
    if (ROWS <= 0 || COLS <= 0) {
        printf("Error: Invalid matrix dimensions.\n");
    }

    // CHECK 2: Validate input values
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {

            if (!isfinite(A[i][j]) || !isfinite(B[i][j])) {
                printf("Error: Invalid input at [%d][%d]\n", i, j);
            }
        }
    }

    // CHECK 3: Perform Hadamard multiplication
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {

            C[i][j] = A[i][j] * B[i][j];

            // CHECK 4: Detect overflow or invalid results
            if (!isfinite(C[i][j])) {
                printf("Error: Invalid result at [%d][%d]\n", i, j);
            }
        }
    }

    // CHECK 5: Display results
    printf("Hadamard Product:\n");

    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%.2f ", C[i][j]);
        }
        printf("\n");
    }

    printf("Hadamard multiplication completed.\n");

}
