# Monte Carlo option pricing

A European and an Asian call option priced from 4.2 million simulated stock-price paths of 64
steps each, in one `kernel fn` (`main.zyth`). Every GPU thread follows 64 paths with its own
random numbers. The European price is checked against the Black-Scholes formula.

```
zyth showcase monte-carlo      # or, in this directory: zyth run main.zyth  (--cpu for the CPU)
```

## Measured

RTX 3090 (power limited to 100 W, a desktop also in use, so the times vary), 4,194,304 paths x 64
steps, f32, 2026-10-28:

| | time | European call (exact 8.0214) |
|---|---|---|
| Zyth `kernel fn` (CUDA) | 10.7-17.7 ms | 8.0206 |
| PyTorch 2.4.1, vectorized over paths (CUDA) | 35.8 ms | 8.0133 |
| NumPy 1.24.4, vectorized over paths (CPU) | 3,066 ms | 8.0289 |

`python3 compare.py` reruns the comparison. PyTorch's version launches several kernels per time
step and keeps every path's price in GPU memory; the Zyth kernel keeps a path in registers from
start to end, which is why one kernel is 2-3x faster here. The random numbers differ between the
three (a 31-bit generator in the kernel, PCG64 in NumPy, Philox in PyTorch), so the prices differ
by the Monte Carlo error, about 0.01.

Not measured: CuPy (not installed here), an H100, the browser (WebGPU) run.
