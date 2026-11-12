/* The same search in C (affine-gap Smith-Waterman, the same recurrences and scores), one
   sequence after another, on all CPU cores with OpenMP. cc -O3 -march=native -fopenmp compare.c */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define Q 128
#define L 256
static float max2(float a, float b) { return a > b ? a : b; }
static float align(const unsigned char *q, const unsigned char *d) {
    float h[Q] = {0}, e[Q] = {0}, best = 0;
    for (int j = 0; j < L; j++) {
        float diag = 0, up = 0, f = 0;
        for (int i = 0; i < Q; i++) {
            float ei = max2(h[i] - 3, e[i] - 1);
            f = max2(up - 3, f - 1);
            float s = diag + (q[i] == d[j] ? 2 : -1);
            float hi = max2(max2(0, s), max2(ei, f));
            diag = h[i]; h[i] = hi; e[i] = ei; up = hi;
            best = max2(best, hi);
        }
    }
    return best;
}
int main(int argc, char **argv) {
    int n = 65536;
    unsigned char *q = malloc(Q), *db = malloc((size_t)n * L);
    float *sc = malloc(n * sizeof(float));
    srand(42);
    for (int i = 0; i < Q; i++) q[i] = rand() % 4;
    for (long i = 0; i < (long)n * L; i++) db[i] = rand() % 4;
    int cores = omp_get_max_threads();  /* read once: omp_set_num_threads changes it */
    int counts[2] = {1, cores};
    for (int c = 0; c < (cores > 1 ? 2 : 1); c++) {
        int t = counts[c];
        omp_set_num_threads(t);
        double t0 = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic, 64)
        for (int k = 0; k < n; k++) sc[k] = align(q, db + (long)k * L);
        double s = omp_get_wtime() - t0;
        printf("C, %d thread(s): %.0f ms, %.2f GCUPS\n", t, s * 1000, (double)n * L * Q / s / 1e9);
    }
    return 0;
}
