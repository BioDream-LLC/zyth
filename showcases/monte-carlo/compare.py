"""The same pricing in NumPy (CPU) and PyTorch (GPU), vectorized over paths, for the README."""
import math, sys, time
import numpy as np

S0, K, R, V, T, STEPS = 100.0, 105.0, 0.05, 0.2, 1.0, 64
dt = T / STEPS
drift, shock = (R - 0.5 * V * V) * dt, V * math.sqrt(dt)

def numpy_price(n):
    rng = np.random.default_rng(1)
    s = np.full(n, S0, np.float32)
    avg = np.zeros(n, np.float32)
    for _ in range(STEPS):
        s *= np.exp(drift + shock * rng.standard_normal(n, dtype=np.float32))
        avg += s
    avg /= STEPS
    d = math.exp(-R * T)
    return d * np.maximum(s - K, 0).mean(), d * np.maximum(avg - K, 0).mean()

def torch_price(n):
    import torch
    g = torch.Generator(device="cuda").manual_seed(1)
    s = torch.full((n,), S0, device="cuda")
    avg = torch.zeros(n, device="cuda")
    for _ in range(STEPS):
        s *= torch.exp(drift + shock * torch.randn(n, device="cuda", generator=g))
        avg += s
    avg /= STEPS
    d = math.exp(-R * T)
    return d * (s - K).clamp(min=0).mean().item(), d * (avg - K).clamp(min=0).mean().item()

n = 1 << 22
t = time.time(); eu, asian = numpy_price(n); np_ms = (time.time() - t) * 1000
print(f"NumPy {np.__version__}, {n} paths: {np_ms:.0f} ms (European {eu:.4f}, Asian {asian:.4f})")
try:
    import torch
    torch_price(n); torch.cuda.synchronize()
    t = time.time()
    for _ in range(5):
        eu, asian = torch_price(n)
    torch.cuda.synchronize()
    print(f"PyTorch {torch.__version__} on {torch.cuda.get_device_name()}, {n} paths: {(time.time() - t) * 200:.1f} ms (European {eu:.4f}, Asian {asian:.4f})")
except ImportError:
    print("PyTorch not installed: GPU comparison skipped")
