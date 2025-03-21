#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 1000000

int main()
{
    double *a = malloc(SIZE * sizeof(double));
    double *b = malloc(SIZE * sizeof(double));
    double *c = malloc(SIZE * sizeof(double));

    // Initialize vectors
    for (int i = 0; i < SIZE; i++)
    {
        a[i] = i;
        b[i] = 2 * i;
    }

    double start = omp_get_wtime();

#pragma omp parallel for
    for (int i = 0; i < SIZE; i++)
    {
        c[i] = a[i] + b[i];
    }

    double end = omp_get_wtime();

    printf("Parallel Time: %f seconds\n", end - start);
    printf("c[100] = %f (expected 300.0)\n", c[100]);

    free(a);
    free(b);
    free(c);
    return 0;
}