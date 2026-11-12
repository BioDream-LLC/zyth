# DNA sequence search (Smith-Waterman)

One 128-base query aligned against 65,536 database sequences of 256 bases with Smith-Waterman
(local alignment, affine gaps), one GPU thread per sequence, in one `kernel fn` (`main.zyth`).
Five sequences carry a mutated copy of the query; the search ranks them first, and their scores
are checked against a plain Zyth version on the CPU.

```
zyth showcase dna-align        # or: zyth run main.zyth  (--cpu for the CPU)
```

## Measured

2.1 billion cells (65,536 x 256 x 128), f32 scores, 2026-10-29:

| | time | GCUPS |
|---|---|---|
| Zyth `kernel fn`, RTX 3090 at 100 W (desktop in use) | 75.7 ms | 28.4 |
| C (gcc 9.4 -O3 -march=native, OpenMP), 16 threads | 490 ms | 4.4 |
| C, 1 thread | 7,269 ms | 0.30 |

`compare.c` is the C version (`cc -O3 -march=native -fopenmp compare.c`). The C code is a
straightforward one-sequence-per-thread loop. Tuned CPU tools (SIMD striped alignment, as in
SSW or parasail) are several times faster than it, and tuned GPU tools (CUDASW++) faster than
this kernel; neither was measured here.
