#include <omp.h>
#include <stdio.h>

int main() {
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        printf("Thread %d executing the parallel region.\n", thread_id);
    }
    return 0;
}