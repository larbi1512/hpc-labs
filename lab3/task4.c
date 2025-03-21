#include <stdio.h>
#include <omp.h>

#define ITERATIONS 1000000

int main()
{
    int counter = 0;

// Without synchronization
#pragma omp parallel num_threads(4)
    {
        for (int i = 0; i < ITERATIONS / omp_get_num_threads(); i++)
        {
            counter++;
        }
    }
    printf("Without Sync: %d (Expected %d)\n", counter, ITERATIONS);

    counter = 0;
// With synchronization
#pragma omp parallel num_threads(4)
    {
        for (int i = 0; i < ITERATIONS / omp_get_num_threads(); i++)
        {
#pragma omp atomic
            counter++;
        }
    }
    printf("With Sync: %d (Expected %d)\n", counter, ITERATIONS);

    return 0;
}