#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1000

int main()
{
    double (*a)[N] = malloc(N * N * sizeof(double));
    double (*b)[N] = malloc(N * N * sizeof(double));
    double (*c)[N] = calloc(N * N, sizeof(double));

    // Initialize matrices
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i][j] = i + j;
            b[i][j] = i - j;
        }
    }

    double start = omp_get_wtime();

#pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < N; k++)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    double end = omp_get_wtime();

    printf("Matrix Multiply Time: %f seconds\n", end - start);
    printf("c[100][100] = %f\n", c[100][100]);

    free(a);
    free(b);
    free(c);
    return 0;
}