# Zyth Language Specification — v0.1

> This specification is open source (Apache-2.0 or MIT; see [LICENSING.md](LICENSING.md)). Where it names files such as `STATUS.md`, `DECISIONS.md`, `tests/`, `scripts/`, `eval/` or `bench/`, those belong to the compiler's sources, which are not published.

**Status:** Draft — authoritative implementation spec
**Date:** October 20, 2026
**Author:** BioDream Developer
**Companion documents:**
- `ZYTH-STRATEGY.md` — the original, long-term vision (includes deferred features) and the positioning and adoption plan

---

## Table of Contents

1. [Mission & The Master Plan](#1-mission--the-master-plan)
2. [Design Principles](#2-design-principles)
3. [Architecture](#3-architecture)
4. [Syntax](#4-syntax)
5. [Type System](#5-type-system)
6. [Tensors & Shape Checking](#6-tensors--shape-checking)
7. [Automatic Differentiation](#7-automatic-differentiation)
8. [Models](#8-models)
9. [Memory Model](#9-memory-model)
10. [Concurrency & Parallelism](#10-concurrency--parallelism)
11. [Compile-Time Features & Annotations](#11-compile-time-features--annotations)
12. [Script Mode, REPL & Debugging](#12-script-mode-repl--debugging)
13. [Targets & Deployment](#13-targets--deployment)
14. [Interop: C, Python, npm & Models](#14-interop-c-python-npm--models)
15. [Package Security & Permissions](#15-package-security--permissions)
16. [Built for AI Coding Assistants](#16-built-for-ai-coding-assistants)
17. [Developer Experience: Debugging, Testing, Profiling & Experiments](#17-developer-experience-debugging-testing-profiling--experiments)
18. [Stability & Editions](#18-stability--editions)
19. [Standard Library](#19-standard-library)
20. [The Flagship Demos](#20-the-flagship-demos)
21. [Tooling & CLI](#21-tooling--cli)
22. [Implementation Roadmap](#22-implementation-roadmap)
23. [Out of Scope for v0.1](#23-out-of-scope-for-v01)
24. [Decisions and Open Questions](#24-decisions-and-open-questions)
25. [Decision Points (AI Inside Programs)](#25-decision-points-ai-inside-programs)
26. [Appendix: Reserved Keywords](#appendix-reserved-keywords)

---

## 1. Mission & The Master Plan

**Mission:** Build the most advanced AI programming language — a small, powerful core with a comprehensive standard library and access to existing ecosystems.

**The master plan — four pillars, all in v0.1:**

| Pillar | v0.1 Commitment |
|---|---|
| **Runs in browsers** | Same Zyth source runs in the browser via Wasm + WebGPU |
| **Train a GPT in ~12 lines** | 12 lines of logic using the standard library (training time depends on hardware and model size) |
| **Uses npm and Python libraries** | Any npm package installs; pure-JS packages run everywhere, Node-API packages run natively after M13. Every Python library runs in native builds; Python does not run in the browser (Pyodide was dropped) |
| **Compiles to native programs** | Single self-contained executable; true ahead-of-time (AOT) compilation follows the interpreter |

**The headline differentiator:** tensor shape, dtype, and device errors are caught at compile time — before a training run starts.

---

## 2. Design Principles

1. **If a beginner can't guess what a line does, the syntax is wrong.**
2. **Small core, big library.** A feature belongs in the language only if it cannot be expressed well as a library. Training methods, servers, RAG, and agents are libraries.
3. **AI types are first-class.** Tensors, gradients, and devices are understood by the type checker.
4. **Safe by default, fast by default, escape hatches are loud.**
5. **Absorb ecosystems, don't fight them.** Use PyPI, npm, and C libraries instead of rewriting them.
6. **Honest claims.** No performance numbers in documentation until they are benchmarked.

---

## 3. Architecture

```
            Zyth source (.zyth)
                   │
        ┌──────────┴──────────┐
        │  Front end (Rust)   │  lexer → parser → AST → type & shape checker
        └──────────┬──────────┘
                   │  typed IR
     ┌─────────────┼──────────────────┐
     │             │                  │
 Interpreter   Wasm build of      AOT compiler (Cranelift):
               the interpreter    compiled functions call the
     │             │              interpreter's built-ins
     │        Browser runtime      Native binary on zyth-rt
     │             │
     └──────┬──────┘
            │  tensor ops
     ┌──────┴──────────────────────────┐
     │  Tensor runtime (Rust)          │
     │  CPU: SIMD kernels / BLAS       │
     │  GPU: wgpu (Vulkan/Metal/DX12   │
     │       natively, WebGPU in web)  │
     │  CUDA fast path: cuBLAS/cuDNN   │
     │       (optional, later)         │
     └─────────────────────────────────┘
```

**Key implementation decisions:**

- **Implementation language: Rust.** Strong LLVM bindings (`inkwell`), Wasm support, parser and diagnostics crates (`logos`, `chumsky`, `ariadne`), and memory safety for the compiler itself.
- **Interpreter first.** The fastest path to a usable language and a stable design. The AOT compiler (Cranelift) came next; it compiles what it can and hands the rest to the interpreter, so results are identical either way.
- **The browser target is the interpreter compiled to Wasm.** Because the interpreter and tensor runtime are Rust, they compile to Wasm with little extra work. This delivers the browser pillar early.
- **One GPU backend for v0.1: `wgpu`.** It runs on Vulkan, Metal, and DX12 natively and on WebGPU in browsers, so the same kernels serve both targets. A CUDA fast path (cuBLAS/cuDNN) is added later for peak NVIDIA performance.
- **Lazy graph capture, not one GPU call per op.** WebGPU validates every dispatch, and per-dispatch overhead is the main bottleneck for browser inference. The tensor runtime records operations lazily, fuses compatible element-wise chains (e.g. `x * w + b |> gelu`) into one kernel, and submits batches of work. Execution is forced only when a value is read (`print`, `.item()`, control flow on a tensor value) or at an explicit `sync()`. User code is unchanged; `zyth profile` shows dispatch counts. General-purpose fusion (matmul epilogues, attention) remains a v2 goal.
- **Drawing shares the GPU layer with tensors.** The GPU code that runs tensor math also draws: render pipelines, vertex and index buffers, textures and presenting to a window or a browser canvas sit next to compute, on the same device and the same buffers. 3D scenes (cameras, meshes, lighting, controls) are a library on top, not part of Zyth (Section 19.3).
- **Single-binary native output before AOT.** `zyth build` initially packages the runtime plus compiled bytecode into one executable (the approach used by `deno compile` and `bun build --compile`). True AOT replaces this later without changing user code.

### File Extensions

| Extension | Purpose |
|---|---|
| `.zyth` | Zyth source |
| `.zyth.model` | Serialized model (safetensors-based container) |
| `.zyi` | Type definitions for Python/npm packages (Section 14.6) |
| `zyth.toml` | Project manifest |

---

## 4. Syntax

### 4.1 Variables

```zyth
let name = "Zyth"              // not changed later, type inferred
let count: i32 = 42            // explicit type
var loss = 1.0                 // changes
const LEARNING_RATE = 3e-4     // compile-time constant
a, b = b, a                    // several at once (the right side first)
```

A `let` binding that the same function changes later (`let xs = []` then `xs.push(1)`, the habit of JavaScript and Python) is a `var`: the program runs, `zyth fmt` and `zyth fix` write `var`, and strict mode reports it (E0205). `let INF = 1_000_000` (a capitalized name) binds the name.

### 4.2 Functions

```zyth
fn add(a: f32, b: f32) -> f32 {
    return a + b
}

fn square(x: f32) -> f32 = x * x          // single-expression form

fn split(x: f32) -> (f32, f32) {          // tuple return
    return (x.floor(), x.fract())
}

fn dense(input: i32, output: i32, bias: bool = true) -> Linear { ... }

let layer = dense(784, 256)                // positional
let layer = dense(input: 784, output: 256) // named
```

### 4.3 Trailing Blocks

A function whose last parameter is a closure can take it as a trailing block. This lets the standard library provide constructs that read like keywords without growing the grammar:

```zyth
reproducible(seed: 42) {
    model.train(data)
}

server(port: 8080) { app ->
    app.get("/health") { _ -> { status: "ok" } }
}
```

### 4.4 Control Flow

```zyth
if loss < 0.01 {
    print("converged")
} else {
    print("training")
}

let status = if loss < 0.01 { "done" } else { "training" }   // if as expression

match activation {
    ReLU         => relu(x)
    LeakyReLU(s) => leaky_relu(x, slope: s)
    _            => x
}

for i in 0..100 { ... }        // exclusive range
for i in 0..=100 { ... }       // inclusive range
for (i, batch) in data.enumerate() { ... }
while loss > 0.01 { ... }
loop { ... }                   // infinite loop; exit with break
```

**Generators.** A `fn` whose body uses `yield` is a generator: calling it runs nothing yet and returns a lazy `stream<T>` (T is the type of the yielded values, checked against a declared `-> stream<T>`). Each request for a value runs the body to its next `yield`; `return` (without a value) or the end of the body ends the stream. Streams work in `for`, and with `.next()`, `.take(n)`, `.skip(n)`, `.map(f)`, `.filter(f)`, `.enumerate()`, `.batch(n)` and `.collect()`, so a generator can be infinite. A stream dropped early (a `break` out of its `for`) stops its generator. `yield` is allowed only directly in a `fn`, not in a closure or `go` block (E0217).

```zyth
fn batches(path: str, size: i64) -> stream<str[]> {
    var batch = []
    for line in fs.read_lines(path)? {
        batch.push(line)
        if batch.len() == size {
            yield batch
            batch = []
        }
    }
    if batch.len() > 0 { yield batch }
}
for b in batches("data.txt", 32) { train_step(b) }
```

Natively the body runs on its own thread, one value at a time, so memory stays flat. In the browser build (no threads) a generator runs to its end on the first request and its values are kept, so it must be finite there. Generators are interpreted under `--aot`; the functions that use them are compiled.

### 4.5 Strings

```zyth
let msg = "epoch {epoch}: loss {loss:.4f}"     // interpolation with format spec
let prompt = """
    You are a helpful assistant.
    Question: {question}
    """                                         // multi-line, indentation-trimmed
```

### 4.6 Operators

| Operator | Meaning |
|---|---|
| `+ - * /` | Arithmetic (element-wise on tensors, with broadcasting); `/` on integers truncates |
| `//` | Whole-number division rounding down, `a.floor_div(b)` (Python's `//`), when it comes between two operands on one line; otherwise `//` starts a comment |
| `??` | Default for a missing value; binds tighter than comparisons and looser than arithmetic (Swift's order): `xs.pop() ?? 0 == 1` is `(xs.pop() ?? 0) == 1` |
| `@` | Matrix multiplication |
| `**` | Power |
| `\|>` | Pipe: `x \|> f(a)` means `f(x, a)` |
| `?` | Propagate `Err` / `None` to caller |
| `== != < > <= >=` | Comparison |
| `&& \|\| !` | Logical |
| `& \| ^ << >> ~` | Bitwise on integers: and, or, exclusive or, shifts, and `~x` (every bit flipped, `-x - 1`) |
| `.. ..=` | Ranges |
| `∇` | Optional alias for `grad` |

Note: the vision document used separate `.*` and `.+` element-wise operators. v0.1 drops them — `*` and `+` are element-wise on tensors, and `@` is the only matrix product.

### 4.7 Statements & Comments

- Newline ends a statement. No semicolons.
- An expression continues to the next line when the line ends with an operator, comma, or open bracket, or the next line starts with `|>` or `.`.
- `//` line comments, `/* */` block comments, `///` doc comments.

---

## 5. Type System

### 5.1 Scalars

| Types | Notes |
|---|---|
| `i8 i16 i32 i64`, `u8 u16 u32 u64` | Integers |
| `f16 bf16 f32 f64` | Floats |
| `f8` | Storage/inference only; arithmetic upcasts |
| `bool`, `char`, `str` | `str` is UTF-8 |

No implicit numeric coercion. `"1" + 1` is a compile error; `x as f32` converts explicitly.

Defaults: an integer literal is `i64`; a float literal is `f64` (as in Python, Mojo and Julia, so `0.1 + 0.2` prints `0.30000000000000004` everywhere); tensor elements default to `f32` (as in PyTorch and JAX), so `tensor.from([1.0, 2.0])` is `f32`. A literal takes the type its context asks for: `let y: f32 = 0.1`, `w * 0.5` on an `f32` tensor. A single number declared `f32`, `f16` or `bf16` is checked as that type but computed and printed as `f64` (`let y: f32 = 0.1` prints `0.1`); its precision applies when it meets a tensor, whose element type it takes. This is how JAX treats Python numbers ("weak types"); it keeps compiled and interpreted results identical and printed numbers readable.

### 5.2 Compound Types

```zyth
let t: (i32, str) = (1, "a")          // tuple
let a: i32[4] = [1, 2, 3, 4]          // fixed-size array
let v: i32[] = [1, 2, 3]              // growable list
let m: map<str, i32> = { "a": 1 }
let s: set<str> = set(["a", "b"])
```

### 5.3 Structs, Enums, Traits

```zyth
struct Point {
    x: f32
    y: f32
}

enum Activation {
    ReLU
    GELU
    LeakyReLU(slope: f32)
}

trait Trainable {
    fn forward(self, input: tensor) -> tensor
    fn params(self) -> tensor[]

    // Default methods
    fn param_count(self) -> i64 = self.params().map(|p| p.size()).sum()
}

impl Trainable for MyModel {
    fn forward(self, input: tensor) -> tensor { ... }
    fn params(self) -> tensor[] { ... }
}
```

No classes, no inheritance. Composition plus traits.

### 5.4 Generics

```zyth
fn normalize<T: Float>(x: tensor<T>) -> tensor<T> = (x - x.mean()) / x.std()

fn train<M: Trainable>(model: mut borrow M, data: stream<Batch>) { ... }

fn fit<M, O>(model: mut borrow M, opt: O) where M: Trainable, O: Optimizer { ... }

struct Pair<A, B> {
    first: A
    second: B
}
```

Supported in v0.1: type parameters, trait bounds, `where` clauses, associated types, const generics (dimensions — see Section 6), type aliases.

Not supported (by design): template metaprogramming, specialization, higher-kinded types, user-written variance annotations.

### 5.5 Optionals & Results

```zyth
let maybe: ?tensor = cache.get(key)             // no null
let value = maybe ?? tensor.zeros(10)           // default

fn load(path: str) -> Result<Model, IOError> {
    let bytes = fs.read(path)?                  // propagate error
    return Ok(Model.decode(bytes)?)
}

match load("m.zyth.model") {
    Ok(m)  => m.run(x)
    Err(e) => print("failed: {e}")
}
```

No exceptions. Unrecoverable bugs call `panic(...)`, which terminates the current task.

An optional compares with a plain value of its kind: `Some(3) == 3` is true and `None == 3` false (Swift's rule), so `xs.pop() != "("` means what it says. Where a plain value is needed (arithmetic and orderings, `as`, an index, a field or method the optional does not have, unpacking with `let (a, b) = ...`, a function result or argument of a plain type), an optional or Result is unwrapped, and the program stops with an error if there is no value, as Python's exceptions do. In a function whose written result type is a plain value (`-> i64`, `-> str[]`, or none), `x?` stops with the error (as `x.unwrap()`). Strict mode wants each of these written (`?`, `.unwrap()`, `??`): E0223.

### 5.6 Type Aliases

```zyth
type Image = tensor<f32, [3, 224, 224]>
type Batch<B> = tensor<f32, [B, 3, 224, 224]>
type Matrix<T, M, N> = tensor<T, [M, N]>
type Vector<T, N> = tensor<T, [N]>
```

`Matrix` and `Vector` are aliases in the prelude, not separate types.

---

## 6. Tensors & Shape Checking

Zyth's headline feature.

### 6.1 The Tensor Type

```
tensor<DType, Shape> on Device
```

- **DType:** any float or integer scalar type.
- **Shape:** a list of dimensions. Each dimension is one of:
  - a literal: `784`
  - a symbolic dimension: `B`, `T` (unified across a function signature)
  - a runtime dimension: `_` (checked when the value is created or crosses a typed boundary)
- **Device:** `cpu`, `gpu`, `gpu(n)`, and `npu` (inference only; roadmap M12). Omitted means "inferred / any".

```zyth
let w: tensor<f32, [784, 256]> = tensor.randn(784, 256)
let x: tensor<f32, [B, 784]>                       // B is symbolic
let y: tensor<f32, [_, 784]> = data.load()         // runtime-sized
let g = tensor.zeros(1024, 1024) on gpu
let t = tensor<f32>([[1, 2], [3, 4]])              // shape [2, 2] inferred from literal
```

### 6.2 Shape Rules

The type checker knows the shape rules for every built-in operation:

| Operation | Rule |
|---|---|
| `a @ b` | `[..., M, K] @ [..., K, N] → [..., M, N]`; batch dims broadcast |
| `a + b`, `a * b` | NumPy broadcasting rules |
| `x.T` | Swaps the last two dims |
| `x.reshape(...)` | Element count must match (statically if known, else at runtime) |
| `x[0:3, :]` | Slicing computes the result shape |
| `concat([a, b], axis: n)` | All other dims must match |
| reductions (`sum`, `mean`, ...) | Remove or keep the reduced axis |

```zyth
let a = tensor<f32, [3, 4]>.randn()
let b = tensor<f32, [5, 6]>.randn()
let c = a @ b
// error[E0401]: matmul dimension mismatch
//   a: [3, 4]
//   b: [5, 6]
//        ^ expected 4, found 5
```

### 6.3 Symbolic Dimensions

```zyth
fn forward(self, x: tensor<f32, [B, 784]>) -> tensor<f32, [B, 10]> {
    return relu(x @ self.w1) @ self.w2
}
// B is inferred at each call site; the checker proves the output is [B, 10].
```

### 6.4 Runtime Dimensions

When a shape cannot be known statically (data loaded from disk, user input), use `_`. The runtime checks it at the first typed boundary and reports where the mismatch came from:

```zyth
let batch: tensor<f32, [_, 784]> = data.next()
model.forward(batch)          // OK: [_, 784] matches [B, 784] with B = runtime value
```

Design rule: static checking must never reject a correct program. When a shape can't be proven, the checker inserts a runtime check — it does not raise an error.

### 6.5 Dtype & Device Checking

```zyth
let w: tensor<f16, [1024, 1024]> on gpu = ...
let x: tensor<f32, [1024]> on cpu = ...
let y = w @ x
// error[E0402]: dtype mismatch: f16 @ f32 — use x.to(f16)
// error[E0403]: device mismatch: gpu @ cpu — use x.to(gpu)
```

A model has a device too: a new one is on the CPU, and `m on gpu`, `m.to(d)`, `m.gpu()` and `m.cpu()` move its weights. A float tensor given to a model (calling it, `forward`, or its own methods) must be on the model's device, and what it returns is there (E0403 before running, with the move as a fix). Integer tensors (token ids, indexes) may stay on the CPU: the weights they pick are moved to them. A model whose own fields put weights on several devices (tensor- and expert-parallel layers) has no single device and is not checked this way. Batches from `zyth.data` (`stream.batches(n)`, text data) are stacked on the device their items were on, which for data read from files is the CPU.

Element-wise operations and comparisons on two different dtypes convert the narrower side (bool, then integers, then floats; narrower then wider), as NumPy and PyTorch do: `(p > 0.5) == y` with a float `y` compares in f32. `@`, `concat` and `where` need one dtype, and strict mode wants every conversion written: `x.to(bf16)`, or the `@mixed_precision` annotation (Section 11). Numbers of two kinds (`f32 * f64`) give the wider one; strict mode wants a cast.

### 6.6 Shape Error Provenance

Every shape error explains where each conflicting dimension came from:

```
error[E0401]: matmul dimension mismatch
  --> train.zyth:14:22
   |
14 |     let logits = h @ self.head
   |                  ^^^^^^^^^^^^^ [64, 512] @ [256, 10]
   |
   = note: 512 comes from `Linear(784 -> 512)`        at model.zyth:4
   = note: 256 comes from `Linear(256 -> 10)`         at model.zyth:6
   = note: 64  comes from `data.batch(64)`            at train.zyth:3
   = help: change line 6 to `Linear(512 -> 10)`
```

The same provenance is available to the editor (Section 17.1) and to AI tools (Section 16).

### 6.7 Tensor Operations (Prelude)

Creation: `zeros ones full randn rand arange linspace eye from`
Math: `exp log sqrt sin cos tanh abs clamp pow`
Reductions: `sum mean std var min max argmin argmax`
Shape: `reshape view transpose permute squeeze unsqueeze flatten concat stack split`
Linear algebra: `@ matmul dot outer` (more in `zyth.math.linalg`)
NN functions: `relu gelu silu sigmoid softmax log_softmax layer_norm dropout cross_entropy mse attention`. `attention(qkv, heads: h, causal: true)` is multi-head scaled dot-product attention on a packed `[B, T, 3D]` projection, giving `[B, T, D]` (on an NVIDIA GPU of compute capability 8.0 or newer, a bf16 `qkv` with heads of 32, 64, 96, 128 or 256 values and no gradient wanted runs as one FlashAttention-2 style kernel: bf16 tensor cores, f32 softmax, the scores never written to memory; also an f32 `qkv` inside `@mixed_precision(compute: bf16)`, with an f32 result. On an H100 SXM it takes 0.074 ms for B 8, T 1,024, 12 heads of 64, causal, against 0.077 ms for FlashAttention-2 in PyTorch 2.4.1 on the same layout; across heads of 64 and 128, T 1,024-4,096, with and without the mask, 0.90-1.06x FlashAttention-2's speed; heads of 256 (B 2 T 4,096 8 heads) 0.521 / 0.838 ms causal / not against FlashAttention-2's 0.542 / 0.855, B 8 T 1,024 3 heads 0.079 / 0.102 against 0.081 / 0.096. With gradients (heads of 32 to 128 or 256 values) a FlashAttention-2 style backward kernel computes them: forward plus backward on the H100, B 8 T 1,024 12x64 causal 0.66 ms against 0.91 ms for PyTorch's FlashAttention-2, B 4 T 4,096 8x64 1.64 / 2.41 ms (causal / not) against 1.91 / 2.57, 8x128 3.13 / 4.71 against 3.39 / 4.58, B 2 T 4,096 8x256 3.36 / 5.11 against 3.63 / 5.09; B 8 T 1,024 3x256 0.84 / 1.97 ms not causal; on GPUs with 99 KB of shared memory a block (8.6, 8.9) heads of 256 use a 32 x 32 tile: B 4 T 1,024 8x256 on an RTX 3090 at 100 W 29 ms against 771 ms for the composite formula); `x.linear(w, bias: b)` is `x @ w + b` as one operation. Both are defined by their composite formulas; implementations may fuse them.
Element-wise chains are fused: on NVIDIA GPUs into one generated kernel (Section 3), and on the CPU a chain of `+ - * /` over f32 tensors of one shape (64K elements or more) and numbers, written with names and literals, runs as one pass over blocks of 2,048 elements; `y = x * 2.0 + b` writes into `y`'s memory when nothing else holds it. Each element gets the same f32 operations in the same order, so the values are those of the separate operations (measured on the M4, 16M elements: 3.66 → 2.56 ms; Mojo's hand-fused loop 2.05).
Named axes: `einsum("bhqd,bhkd->bhqk", q, k)` multiplies and sums over axes labeled by letters (a letter repeated in one operand takes its diagonal; without `->` the output is the letters used once, in alphabetical order; no `...`). The checker checks that each letter has one size wherever it is used (E0401), that each operand has one letter per axis (E0403), and gives the result's shape. It is built from `permute`, `reshape`, `matmul` and `sum`, so it has gradients and runs on GPUs.
Gather and scatter: `x.gather(axis, index)` (the result has the index's shape), `x.scatter(axis, index, src)`, `x.scatter_add(axis, index, src)`, `x.index_add(ids, rows)` (`out[ids[k]] += rows[k]`) and `x.index_select(ids)`, each with a gradient; together they route tokens to experts and back.

### 6.8 Sparse Tensors

`tensor.sparse_coo(indices, values, shape: [m, n])` (`indices` is `[2, nnz]`: rows, then columns), `tensor.sparse_csr(row_ptr, cols, values, shape:)`, and `x.to_sparse()` (the nonzero entries of a 2-D tensor) store only the nonzero entries. `s @ dense` (a vector or a matrix) and `s.to_dense()` are built from `index_select` and `index_add`, so they have gradients with respect to the values and the dense operand and run on GPUs; also `s.t()`, `s.csr()` (row pointers, columns and values sorted by row), `s.nnz`, `s.shape`, `s.values`, `s.indices`, `s.mul(k)`.

### 6.9 Low-Precision Types and Quantization

Besides `f16`, `bf16`, `f32`, `f64` and the integer types, tensors can be `fp8_e4m3` (also `f8`: 4 exponent and 3 mantissa bits, finite, up to 448), `fp8_e5m2` (5 and 2 bits, up to 57344; values past it saturate) and `i4` (also `int4`: -8 to 7; `int8` is `i8`). `x.to(fp8_e5m2)` rounds to the nearest value the type can hold. The fp8 types are storage types: arithmetic on them happens in f32, and the checker types the result as f32 (so `x.to(f8) * 2.0` is a `tensor<f32, ...>`). The exception is the product of two fp8 tensors (either format on each side, so E4M3 activations can multiply E5M2 gradients), which multiplies the fp8 values themselves and gives f32: on NVIDIA GPUs of compute capability 8.9 or newer (Ada, Hopper) on fp8 tensor cores through cuBLASLt, except E5M2 by E5M2, which those do not take; that one, and every fp8 product on older GPUs, runs on bf16 tensor cores (fp8 values are exact in bf16); on the CPU in f32. On the GPU both fp8 formats take one byte a value and are rounded there as on the CPU; slices and transposes stay on the GPU. Hopper's fp8 tensor cores add up with fewer bits than f32 (about 14), so products differ from f32 sums of the same values by up to about 3e-4 of the largest. Measured on an H100 SXM: 8192 x 8192 products at 1,276-1,323 TFLOP/s in fp8 E4M3 (PyTorch 2.4's `_scaled_mm`: 1,125), 1,341 for E4M3 x E5M2, 649 for E5M2 x E5M2 (on bf16 tensor cores), against 782-804 in bf16 and 52 in f32.

`x.quantize(int4 | int8 | fp8_e4m3 | fp8_e5m2, block: 32)` stores the tensor as values of that type plus one scale per run of `block` values along the last axis (the largest of them maps to the type's largest value; scales are kept in f16, as GGUF does). The result has `dequantize()`, `bytes` (the packed size: int4 two values a byte), `values`, `scales`, `block`; `x @ q` multiplies by the dequantized matrix, on the device of `x`. On an NVIDIA GPU, int4 weights `[k, n]` stay packed there (two values a byte, made at the first product) and multiply an f32 `x` of up to 16 rows without expanding them (no gradient through this path): a block of 256 threads reads 128 contiguous bytes of weights per row and adds its partial sums once. Measured with `[k, n] = [8192, 8192]` weights (37.7 MB in int4, 268 MB in f32), against the f32 product of the expanded weights already on the GPU: on an H100, 1 row 0.063 against 0.100 ms, 4 rows 0.069 against 0.101, 16 rows 0.206 against 0.142; on the RTX 3090, 0.56 against 1.01, 0.67 against 0.81, 1.87 against 1.99. Expanding the weights at each product, the way before, took 174 ms for `[4, 4096] @ [4096, 4096]`. Measured on 64 × 256 normal values, block 32: int4 9,216 bytes (f32: 65,536), relative error 9.6%; int8 17,408 bytes, 0.54%; fp8_e4m3 2.4%; fp8_e5m2 4.8%.

`@mixed_precision(compute: fp8)` rounds the inputs of f32 matrix products inside the function to fp8_e4m3 with one scale per tensor (so the largest value is 448) and adds up in f32; gradients pass straight through the rounding. On a GPU, products that need no gradient run on fp8 tensor cores (scaling and rounding on the GPU each time: 383 TFLOP/s at 8192 on the H100, 7x f32). Training products (`[..., K] @ [K, N]` with gradients, as in `Linear`, on an NVIDIA GPU) are trained in fp8 as torchao's float8 does: the forward product of E4M3 values, the backward products of the gradient in E5M2 with the saved E4M3 inputs, each tensor with its own scale from its largest value, found and applied on the GPU (on fp8 tensor cores where present, else the rounded values multiplied in f32); the backward pass keeps the fp8 copies (1 byte a value). `ZYTH_FP8_TRAIN=0` goes back to straight-through rounding. Other products with gradients, and the CPU, multiply the rounded values in f32 (the numbers of FP8 tensor cores, not their speed). Measured on an H100 SXM (feed-forward block 4096 -> 16384 -> 4096, batch 8,192): 11.9 ms a training step in fp8, 13.0 in bf16, 12.7 for torchao float8 with `torch.compile`; loss after 20 steps 0.40324 against bf16's 0.40268. `compute: bf16` uses bf16 tensor cores on GPUs that have them.

### 6.10 GPU Kernels (`kernel fn`)

A `kernel fn` is a GPU function written in Zyth. It runs once per thread, and `launch` starts it (`kernel` is a keyword only before `fn`):

```zyth
kernel fn total(x: tensor<f32>, out: mut tensor<f32>) {
    var buf = shared(f32, 256)          // memory shared by the threads of a block
    let t = thread.x
    let i = thread.global               // block.x * block_dim.x + thread.x
    buf[t] = if i < x.len() { x[i] } else { 0.0 }
    sync()                              // wait for every thread of the block
    var step = 128
    while step > 0 {
        if t < step { buf[t] += buf[t + step] }
        sync()
        step = step / 2
    }
    if t == 0 { atomic_add(out, 0, buf[0]) }
}
total.launch(x, out, threads: x.len(), block: 256)     // or grid: (gx, gy), block: (bx, by)
```

- **Parameters** are tensors (`mut` to write them; f32 or i64 elements), integers, floats and booleans. Tensors are passed whole: `x[i]` reads element i of the data, `x[i, j]` indexes by the tensor's shape at launch, `x.len()` is its number of elements. Writes change the tensor in place.
- **The body** uses `let`/`var`, arithmetic, comparisons, `as`, `if` (also as a value), `for i in a..b`, `while`, `loop`, `break`, `continue`, `return`; `thread`, `block`, `block_dim`, `grid_dim` (`.x`, `.y`, `.z`; `thread.global`, `.global_y`, `.global_z`); `shared(f32 | i64, n)`; `local(f32 | i64, n)` (a thread's own array, up to 256 elements, starting at zeros: in registers on a GPU when indexed by the variables of loops with written bounds of up to 64 steps, which are unrolled); `sync()`; `atomic_add(t, i, v)`; and `exp log sqrt abs min max pow floor ceil sin cos tanh`. Anything else (text, collections, closures, calls of other functions, decisions) is E0250, reported by the checker before the program runs.
- **Where it runs.** On `gpu` tensors with an NVIDIA GPU it is compiled to CUDA C and run through NVRTC (compiled once per signature, and cached on disk like the fused kernels); in browsers and `--features webgpu` builds it is compiled to WGSL; otherwise it runs on the CPU, thread after thread (with `sync()`, one OS thread per thread of a block), with bounds checks (E0411). On an NVIDIA GPU, tensor indices are checked too: the first index outside its tensor stops the launch's reads and writes there and is reported after the launch as E0411, pointing at the index (`launch(..., checked: false)` or `ZYTH_KERNEL_CHECKS=0` removes the checks, which cost nothing measurable in a 16M-element saxpy and about 50% in the FlashAttention example). `shared` and `local` arrays are not checked on a GPU, and WebGPU clamps out-of-range indices into the buffer. `k.cuda(args...)` and `k.wgsl(args...)` return the generated code.
- **Measured** (RTX 3090 capped at 100 W, 16M-element `saxpy`): 0.44 ms per launch, 460 GB/s; the same `x * 2.0 + y` with built-in operations takes 3.9 ms, because it makes new tensors. Checked: the generated WGSL of the test kernels validates with naga, and runs in headless Chromium on SwiftShader's WebGPU with the same output as CUDA and the CPU.
- **Tensor cores.** A warp (32 threads, lane `thread.lane`) multiplies 16 x 16 tiles together: `let acc = mma_acc()` is its f32 accumulator (8 elements a thread, starting at zeros); `mma(acc, a, a_off, lda, b, b_off, ldb)` adds A @ B, where A is the 16 x 16 tile of `a` at flat offset `a_off` with rows `lda` apart (B likewise), with inputs rounded to tf32; `mma_bf16(...)` rounds them to bf16 instead (twice the speed on NVIDIA GPUs); `let qa = mma_a(a, off, ld)` (or `mma_a_bf16`) keeps an A tile in registers for `mma(acc, qa, b, b_off, ldb)`; `mma_store(acc, out, off, ld)` writes the tile; `acc[e]` is the element at row `mma_row(e)`, column `mma_col(e)` of the tile (the layout of NVIDIA's `mma.sync`), so a softmax or a scale can work on the accumulator. `a`, `b` and `out` are f32 tensor parameters or f32 shared or local arrays. Every thread of the warp takes part (blocks of a multiple of 32 threads). On NVIDIA GPUs of compute capability 8.0 or newer these are `mma.sync` instructions; elsewhere each thread computes its own 8 elements with the same roundings, so the values agree up to the order of the additions. Measured (`examples/showcase/flash_attention_tc.zyth`, the attention below on tensor cores, 96 heads x 1,024 tokens, causal): on an H100 SXM 0.30 ms with tf32 and 0.32 ms with bf16, against 0.70 ms for the f32 kernel below, 0.62 ms for PyTorch 2.4's default f32 attention and 0.09 ms for its FlashAttention-2 in bf16 (written with `ldmatrix`, asynchronous copies and register-resident tiles, which `kernel fn` does not have); on the RTX 3090 (100 W, a busy desktop GPU, so ±20%) 3.1-6.0 ms with tf32 and 4.5-4.9 ms with bf16, against 8.2-10 ms for the f32 kernel.
- **bf16 data, asynchronous copies, shuffles** (what FlashAttention-2's speed needs). Tensors may be `bf16` (on NVIDIA GPUs their 16 bits; read as f32, written rounded to nearest even) and so may shared arrays (`shared(bf16, n)`). `copy_async(dst, dst_offset, src, src_offset, n)` copies `n` elements of a tensor into a shared array (zeros past the tensor's end); the copies are done once `copy_wait()` has run (`copy_commit()` closes a group, `copy_wait(k)` waits for all but the newest `k` groups, so the next tile can load during work on this one); on NVIDIA GPUs 8 bf16 or 4 f32 values are one `cp.async` of 16 bytes, on the CPU and WebGPU the copy happens at once. `mma_bf16` reads its operands from bf16 shared arrays with `ldmatrix` (offsets and row strides multiples of 8: checked kernels fall back to element reads otherwise), `mma_bf16_t(acc, a, ..., bt, offset, row_stride)` takes B transposed (rows of `bt` are B's columns: keys for `Q K^T`), `let p = mma_a_bf16(acc)` makes an accumulator the A operand of the next product (its layout already is; no data moves on the GPU), `let y = shfl_xor(x, m)` is the value of `x` in lane `thread.lane ^ m`, and `exp2(x)` is one instruction. `mma_a_bf16(acc)` and `shfl_xor` are run by all threads of the block together, like `sync()` (the CPU and WebGPU exchange the values through block memory). Example (`examples/showcase/flash_attention_fa2.zyth`): FlashAttention-2's forward pass in 200 lines of Zyth, the same values on the CPU and the GPU: on an H100 SXM 0.072 ms for 96 heads x 1,024 tokens of 64 (causal), against 0.073 ms for the built-in `attention` kernel written in CUDA C, 0.075-0.079 ms for FlashAttention-2 in PyTorch 2.4.1, and 0.30 ms for the tensor-core version without these operations.
- **Gradients.** A kernel writes in place and has no gradient of its own; a function that launches one gets one with `@custom_grad` (Section 7.1), typically a second kernel.
- **Example** (`examples/showcase/flash_attention.zyth`): FlashAttention's forward pass (causal, head size 64) in 110 lines: tiles of 64 queries and 32 keys in shared memory, each thread holding 8 rows x 4 dimensions of the output and the running softmax max and sum in `local` arrays. Measured on the RTX 3090 at 100 W, f32, 96 heads of 1,024 tokens: 8.2-9.3 ms, against 7.8 ms for PyTorch 2.4's default f32 attention, 11.1 ms for its memory-efficient kernel, 18.6 ms for its plain one, and 21.8 ms for the same attention from Zyth's built-in operations; FlashAttention-2 in fp16 on tensor cores takes 1.9 ms (kernels cannot use tensor cores yet).

### 6.11 Sharded Tensors

```zyth
let xs: tensor<f32, [B, 512]> on gpus(8) shard [B, _] = x.shard(gpus(8), axis: 0)
let ys = xs @ w                         // each piece times the whole w: still split on B
let loss = xs.map_shards(|p| model(p)).sum()      // data parallelism; gradients reach `model`
let whole = ys.gather()
```

A sharded tensor is split along one axis into pieces, one per device. Element-wise operations and products work piece by piece; a whole tensor on the other side is used by every piece (split the same way when it has the split axis in full). Where the pieces' results must be combined, they are added up: `sum()` and `mean()` over everything or over the split axis, and a product that contracts the split axis. `map_shards(f)` applies a function of one piece to every piece; `gather()` puts the whole tensor back; `shards` are the pieces. The checker follows the layout through operations: two tensors split differently (`gpus(4)` against `gpus(2)`, or other axes) are E0403, a reduction over the split axis gives a whole tensor, and `on gpus(n) shard [B, _]` names the split axis (one per tensor; the others are `_`).

**Several GPUs.** Piece `i` lives on `gpu(i % n)` for the `n` NVIDIA GPUs the computer has (`ZYTH_GPUS=k` uses the first k); with one GPU, or none, all pieces are on it (or on the CPU), so a program written for eight GPUs runs, and computes the same numbers, on one. Work on different GPUs runs at the same time; a whole tensor used by every piece is copied to each GPU once while it is unchanged (directly between GPUs, over NVLink where there is one). Sums of the pieces' results go through NCCL when it is installed (`libnccl.so.2`, or the `nvidia-nccl-cu12` package PyTorch brings) and no gradient is needed; otherwise the results are copied and added. `s.all_reduce()` replaces every piece of a sharded tensor whose pieces have one shape (per-GPU partial results such as gradients) by their sum, each on its own GPU; `s.devices()` lists where the pieces are. Any tensor can be put on one GPU with `x on gpu(1)` or `x.gpu(1)`; operations need their operands on the same device (E0403). Measured on two H100 SXM GPUs (NVLink): `x [16384, 8192] @ w [8192, 8192]` in f32 with x split in two, 21.2 ms against 42.1 ms on one GPU; `all_reduce` of 256 MB, 0.81 ms (a bus bandwidth of 332 GB/s as nccl-tests counts it). One process drives all the GPUs of one machine; several machines are not supported.

---

## 7. Automatic Differentiation

### 7.1 API

```zyth
let loss = model.forward(x) |> cross_entropy(y)
let grads = grad(loss, wrt: model.params)     // or ∇(loss, wrt: model.params)
opt.step(grads)
```

`grad` returns a gradient value with the same structure (shapes, dtypes) as `wrt`. The checker verifies that `loss` is a scalar.

**Custom gradients.** `@custom_grad(g)` gives a function its own gradient, instead of the one of the operations inside it (for numerical stability, straight-through estimators, gradient clipping):

```zyth
fn softplus_grad(x: tensor<f32>, g: tensor<f32>) -> tensor<f32> = g * sigmoid(x)
@custom_grad(softplus_grad)
fn softplus(x: tensor<f32>) -> tensor<f32> = log(1.0 + exp(x))
```

The function runs without recording its operations, and its result is one step of the graph. In the backward pass `g` is called with the function's arguments, optionally its result, then the gradient of the result, and returns the gradient of each tensor argument: one tensor, or a tuple with one per tensor argument (`none` for one without). The checker checks that `g` exists and takes that many parameters (E0201); a gradient of the wrong shape is E0403 when it is used. The function returns one tensor; tensors it uses that are not arguments (a model's parameters, say) get no gradient through it.

### 7.2 Implementation (v0.1)

- **Reverse-mode autodiff using a tape.** Operations on tensors that require gradients are recorded; `grad` walks the tape backward. This is the proven approach (PyTorch, JAX's eager mode).
- Gradients are defined for every built-in differentiable op. User-defined ops can provide a custom gradient with `@custom_grad`.
- `no_grad { ... }` disables recording (inference, evaluation).
- Frozen parameters: `p.frozen = true` excludes a tensor from `grad`.
- Gradients of gradients (Hessian-vector products) are supported by recording the backward pass.

Compiling the gradient computation ahead of time (instead of building a tape at runtime) is a v2 optimization. User code doesn't change.

### 7.3 Function Transforms

- `vmap(f, in_axes: [0, None], out_axis: 0)`: `f` written for one example works on a batch: each argument with an axis is split along it (the others are passed to every call), `f` runs once per example (each call checks its arguments against `f`'s parameter types), and the results are stacked; tuples of results are stacked element by element. It runs the examples one after another (no fusion).
- `jacobian(f, x)`: the derivative of every element of `f(x)` with respect to every element of `x`, shape `f(x).shape + x.shape` (checked from `f`'s declared result).
- `hessian(f, x)`: the second derivatives of a function with one number as its result, shape `x.shape + x.shape`. Second derivatives go through indexing too: the gradient of a slice is itself differentiable.
- `@checkpoint` on a function: when gradients are recorded, only its result is kept; its activations are freed when it returns and computed again during the backward pass (the inputs of that step are the function's tensor arguments and every parameter it used). The gradients are the same as without it. The function returns one tensor.


---

## 8. Models

`model` is syntax sugar for a struct that automatically implements `Trainable`, collects its parameters, and supports save/load.

```zyth
model Classifier {
    fc1 = Linear(784 -> 256)
    fc2 = Linear(256 -> 10)

    fn forward(self, x: tensor<f32, [B, 784]>) -> tensor<f32, [B, 10]> {
        return x |> self.fc1 |> relu |> self.fc2
    }
}

let m = Classifier() on gpu
m.save("classifier.zyth.model")
let m2 = Classifier.load("classifier.zyth.model")?
```

- `784 -> 256` is layer shorthand meaning "in-features to out-features".
- Layers are callable: `self.fc1(x)` and `x |> self.fc1` are equivalent.
- If `forward` is omitted, the fields are applied in order (sequential model).

---

## 9. Memory Model

### 9.1 Ownership & Borrowing

- Every value has one owner. Assignment of non-copy types moves: `let b = move a`.
- `borrow T` is a shared reference; `mut borrow T` is an exclusive one.
- A parameter declared `mut T` (or `mut borrow T`) changes the caller's value. A variable passed to it by a function of the same file is lent as if written `mut borrow x`, except a `let` binding (which cannot change) and a variable passed in two arguments of the call (an exclusive borrow is the only access). Where the value is copied instead, so the function's changes are lost, the checker warns (W0109) with an edit: a variable copied into a `mut` parameter (a `let`, or a function of another module, where `mut borrow` must be written), and a function that returns nothing and only changes a parameter that is not `mut` (the edit makes it `mut`).
- The compiler rejects use-after-move and aliasing a mutable borrow.
- **Lifetimes are inferred.** A function that returns a borrow says nothing about where it comes from; its body does. The compiler collects, over every `return` and branch (and through calls of other functions, recursive ones included), which borrowed parameters the result can borrow, and a call borrows exactly those arguments:

  ```zyth
  fn longer(a: borrow str[], b: borrow str[]) -> borrow str[] {
      if a.len() >= b.len() { borrow a } else { borrow b }
  }
  fn second(a: borrow i64[], b: borrow i64[]) -> borrow i64[] = borrow b

  let l = longer(borrow xs, borrow ys)   // l borrows xs and ys
  let s = second(borrow p, borrow q)     // s borrows q only: p can change
  ```

  A borrow is live until the last use of what holds it: a variable, a function's result, a struct field, a list or tuple element, or a collection it was pushed into. Changing the value while it is live is E0303, moving it E0304, returning a borrow of a local or by-value parameter E0305. Annotations remain valid and are checked (`fn first<'a>(x: borrow 'a T[], y: borrow T[]) -> borrow 'a T`); they are required only on signatures without a body (trait methods, extern functions), where there is nothing to infer from (E0306). Unlike Rust, an inferred signature can change when the body changes; `zyth doc` and hovers show the inferred origin.

### 9.2 Tensor Storage

Tensors are the exception that keeps ML code ergonomic:

- A tensor value is a handle to **reference-counted storage**.
- Passing a tensor to a function never copies its data.
- Mutation is copy-on-write: writing to a tensor whose storage is shared first copies it.
- In-place ops (`x.add_(y)`) require `mut borrow` and fail at compile time if the storage could be shared.

This means typical ML code never needs `move` or `borrow` on tensors, while non-tensor code gets full ownership checking.

### 9.3 Deterministic Cleanup

```zyth
struct GpuBuffer {
    handle: ptr<void>

    drop {
        gpu.free(self.handle)        // runs when the value goes out of scope
    }
}

region {
    let scratch = tensor.zeros(4096, 4096)
    ...
}   // everything allocated in the region is freed at once
```

### 9.4 Escape Hatch

```zyth
danger {
    let p = ptr.alloc<f32>(1000)
    p[500] = 3.14
    ptr.free(p)
}
```

Raw pointers, FFI calls, and unchecked indexing are only allowed inside `danger` blocks or `danger fn`. The compiler and `zyth lint` report every `danger` site.

**Inline assembly (2026-10-29).** For a hot loop nothing else makes fast enough, an `asm` block inside `danger` holds machine instructions for one processor:

```zyth
fn popcount(x: i64) -> i64 {
    var r = 0
    danger {
        asm "x86_64" {
            "popcnt {r}, {x}"
            in(reg) x
            out(reg) r
            clobber("cc")
        }
        asm "aarch64" {
            "fmov d0, {x}"
            "cnt v0.8b, v0.8b"
            "addv b0, v0.8b"
            "fmov {r}, d0"
            in(reg) x
            out(reg) r
            clobber("v0")
        }
    }
    return r
}
```

- **Operands** as in Rust's `asm!`: `in(reg) x`, `out(reg) r`, `inout(reg) acc` name local variables (`reg` for `i64`, `freg` for `f64`; outputs are changed, so they are `var`s); `{name}` in an instruction is that operand's register, `{{` and `}}` are braces. `clobber("cc", "memory", "rax")` lists what else the code changes. On x86-64 the instructions use Intel syntax, as in Rust.
- **Per processor.** `"x86_64"` or `"aarch64"`; a block for another processor than the one the code is for does nothing, so a function carries one block per processor it supports.
- **Only in release code.** The interpreter cannot run assembly and neither can Cranelift, so an `asm` block runs only in code LLVM makes (`zyth build --aot --release`, with zig or clang); anywhere else it stops with E0503. A release build whose LLVM step fails is not used for a program with assembly. Checked before running: `asm` outside `danger` (E0501), operands that are not `i64`/`f64` variables (E0201), and `{name}` that is not an operand (E0101). The assembly itself is not checked: it can do anything, which is why it needs `danger`, and `zyth lint` lists it with the other `danger` sites.
- Measured on Linux x86-64: `popcount` above and an `lea`-based multiply-add, called 1,000 times each, give the same results as the plain-code versions; the instructions appear unchanged in the machine code.

---

## 10. Concurrency & Parallelism

### 10.1 Tasks & Channels

```zyth
let ch = channel<Batch>(capacity: 8)

go {
    for batch in loader {
        ch.send(batch)
    }
    ch.close()
}

for batch in ch {
    model.step(batch)
}
```

- `go` starts a lightweight task on a work-stealing scheduler (no global interpreter lock).
- Data sent across a channel is moved; the borrow checker prevents data races.
- `async`/`await` is not part of v0.1; blocking calls inside `go` tasks yield to the scheduler.

### 10.2 Structured Concurrency (`scope`)

Every task belongs to a scope, so tasks can't leak and errors can't be lost:

```zyth
let (images, labels) = scope {
    let a = go { load_images("train/") }
    let b = go { load_labels("train.csv") }
    (a.join()?, b.join()?)
}
```

Rules:
- Each `go` belongs to the nearest enclosing `scope`. A function body is an implicit scope, so a function never returns while tasks it started are still running.
- If any task in a scope fails, the other tasks in that scope are cancelled and the error propagates out of the scope.
- Cancellation is cooperative: tasks are cancelled at their next blocking point (channel operation, I/O, `sync()`).
- `scope(timeout: 30.seconds) { ... }` cancels everything still running when the timeout hits.
- Tasks that must outlive their creator use `task.detach { ... }` from the standard library; `zyth lint` reports every detached task.

### 10.3 `parallel for`

```zyth
let results = parallel for img in images {
    img |> preprocess |> model.forward
}
```

v0.1 semantics: iterations run concurrently on the CPU thread pool, and tensor operations inside them are batched and dispatched to the selected device. Generating custom GPU kernels from arbitrary loop bodies is deferred to v2.

### 10.4 `pipeline`

```zyth
pipeline {
    load:    loader.batches(64)
    prep:    load |> augment
    train:   prep |> model.step
}
```

Each stage runs as its own task, connected by bounded channels. Stages form one scope, so a failure in any stage stops the whole pipeline. Single machine in v0.1; multi-node stays a library concern (`zyth.dist`).

---

## 11. Compile-Time Features & Annotations

### 11.1 `comptime`

```zyth
const SIN_TABLE: f32[256] = comptime {
    var t: f32[256]
    for i in 0..256 { t[i] = sin(i as f32 * PI / 128.0) }
    t
}
```

`comptime` blocks run in the interpreter during compilation. Inside an expression they compute a value and can't do I/O.

A `comptime` block written as a statement at the top of a file **makes declarations**: it runs when the file is read, before it is checked, and the Zyth source text it gives (a `str`, or a list of them) is added to the file. A library can so turn a database schema, a migration folder or an OpenAPI document into structs and functions the checker sees like hand-written ones:

```zyth
use sqltypes                       // a library: CREATE TABLE statements -> Zyth structs

comptime {
    sqltypes.structs("schema/orders.sql")    // "pub struct Order { id: i64  customer: str ... }"
}

let o = Order { id: 7, customer: "Ada", total: 12.5 }    // checked against the schema
```

- **Permission.** Such code may read files, use the network or run programs only as the list `comptime = [...]` in zyth.toml's `[permissions]` grants, with the capabilities of `allow` (§15.2): `comptime = ["fs:read:./schema", "net:localhost"]`. Without it, it can do nothing outside the program (E0236). The program's own `allow` does not apply to it, and a `[permissions]` table with only `comptime` does not limit the program.
- **What runs.** The file's imports and items are made for it (its other top-level code does not run); relative paths are the project's, wherever the command runs from; each block gets 10 seconds. It must give text that parses as declarations only (`fn`, `struct`, `enum`, `const`, `use` and the like): E0114 otherwise.
- **Where the declarations go.** The block is replaced by spaces (every other line and column stays where it was) and the declarations follow the file's last line, under a comment naming the block's line, so errors in them are shown there. `zyth check`, `run`, `test`, `build`, the language server and imports of the file all see the same text; a built program carries it and never runs the block again.

### 11.2 Contracts (Debug Checks)

```zyth
fn softmax(x: tensor<f32, [N]>) -> tensor<f32, [N]>
    requires N > 0
    ensures (result.sum() - 1.0).abs() < 1e-5
{
    let e = exp(x - x.max())
    return e / e.sum()
}
```

`requires` and `ensures` are checked at runtime in debug builds and removed in release builds. They are not formal proofs.

### 11.3 Annotations (v0.1 Set)

| Annotation | Effect |
|---|---|
| `@test` | Test function run by `zyth test` |
| `@bench` | Benchmark run by `zyth bench` |
| `@target(web)` / `@target(native)` | Include item only for that target |
| `@mixed_precision(compute: bf16)` | Run tensor ops in lower precision, accumulate in f32 (implemented for matrix products, the costly part: bf16 inputs, f32 accumulation and outputs, also in their gradients); `compute: fp8` gives FP8 products, on fp8 tensor cores on GPUs that have them (Section 6.9) |
| `@checkpoint` | Recompute the function's activations in the backward pass instead of keeping them (Section 7.3) |
| `@tool`, `@resource("uri")`, `@prompt` | Offer the function to AI agents through `mcp.serve()` (Section 16.6) |
| `@custom_grad(fn)` | Provide a gradient for a function (Section 7.1) |
| `@export` | Expose a function or model to Python and npm builds (Section 14.5) |
| `@deprecated("msg")` | Warn on use |
| `@inline` / `@noinline` | Hints for compiled code: calls of an `@inline` function are replaced by its body; small functions (up to 40 instructions) are inlined unless `@noinline`. Results never change, only speed (Section 13.5) |

**User-defined annotations.** A library declares its own annotations; they are data on items, not macros (they never rewrite code):

```zyth
/// Marks a function as the handler of a web route.
pub annotation route(path: str, method: str = "GET") on fn

@route("/items", method: "POST")
fn add_item() -> str = "added"

for r in annotated(route) {       // [{name, kind, owner, item, args: {path, method}}], in definition order
    let handler = r.item
    print(r.args.method, r.args.path, handler())
}
```

- `annotation NAME(params) on KINDS` declares one: its parameters are written like a function's (types, defaults), and `on` lists what it may be put on: items (`fn`, `struct`, `enum`, `trait`, `model`, `const`, `type`, `impl`), and inside them `method` (a function in a struct, enum, model, trait or impl), `field` (a struct's or model's field, on the lines above it) and `param` (a function's or method's parameter, in front of it: `fn show(@from_query("v") verbose: bool)`); without `on`, any. `fn` means top-level functions only, so an annotation meant for both says `on fn, method`. It is an item: `pub`, imported with `use`, documented by `zyth doc`.
- `zyth check` checks every `@NAME(...)` like a call (missing, unknown and mistyped arguments) and against `on` (E0111). An `@NAME` that is neither built in (the table above) nor declared is a warning, W0110, with a suggestion (`@tset`: did you mean `@test`?).
- `annotated(NAME)` lists what carries it: the items, methods, fields and parameters of the program and of every module it imports, in the order they are defined (an item, then its parameters, fields and methods), with the arguments evaluated (defaults filled in). `kind` says which (`fn`, `method`, `field`, `param`, ...), `owner` is the type a method or field is in or the function a parameter is of (`User.show` for a method's parameter; empty for items), and `item` is the function, the type (for a field), the function or method (for a parameter), or the method, which takes its receiver first (`r.item(user, true)`, as `User.show(user, true)` does). The checker knows the record's type, so `r.args.path` is a `str`. The list is filled while the items are defined, as `@test` functions are collected: there is no scan of files or packages and no reflection on values.

### 11.4 Effects (`uses`)

A function header may say what the function does besides computing its result:

```zyth
fn load(path: str) -> str uses files, fails {
    return fs.read(path).unwrap()
}
fn triage(email: str) -> Label uses ai(dollars: 0.01, decisions: 2) { ... }
fn area(r: f64) -> f64 uses nothing = PI * r * r
```

| Effect | The function (or anything it calls) |
|---|---|
| `net` | uses the network |
| `files` | reads or writes files |
| `gpu` | puts data on the GPU |
| `ai` | makes a decision (`decide`, `ai`, `ai.chat`, ...) |
| `process` | starts other programs |
| `ffi` | calls Python, npm, or C code |
| `fails` | can stop with an error on purpose: `panic`, `assert*`, `.unwrap()`, `.expect()` (not `?`, which returns the error, and not out-of-range indexing or division by zero) |

- **Rules.** The clause comes after the result type and before `requires`/`ensures`. `uses nothing` declares no effects. Names are written in the order above; `zyth fmt` reorders them and rewrites other spellings (`fs`, `network`, `panics`, `pure`, ...).
- **Checking.** The checker works out each function's effects from its body and the functions it calls; a callee with its own `uses` clause counts as what its clause says. A function with a clause that lacks one of its effects is E0230; the note shows the chain of calls (`count_words → load (main.zyth:8) → .unwrap() ... (main.zyth:4)`), and the fix adds the effect. An unknown name is E0231. An effect declared but certainly never done (every call in the body is known) is warning W0106, with a fix that removes it.
- **Functions without a clause** are not checked, so quick scripts need none. With `zyth check --effects`, and in strict mode, a missing clause means `uses nothing` (E0232, with a fix that writes the clause). `zyth fix --effects FILE` writes every clause.
- **While running.** Effects the checker cannot see (a function value passed in, Python code) are checked when they happen: a file, network, GPU, process, foreign or AI use inside a running function whose clause does not allow it is E0233. Tasks started with `go` inside such a function keep its clause. A decider's own work (a cloud decider's network use, a local model's files) belongs to `ai`.
- **AI limits per call.** `ai(dollars: D, decisions: N, seconds: S)` caps what one call of the function may spend on decisions (estimated cost, number of decisions, time since the call started). They are checked before each decision, like `ai.limits` (§25.7), and a reached limit is E0804.

### 11.5 What the Checker Tells Libraries (`zyth.check`)

A library can add its own warnings and can learn which fields of what it returns a caller reads. Both are ordinary Zyth in the library (module `zyth.check`); nothing about any one library is built into the compiler.

```zyth
// db.zyth (a library)
use zyth.check.{lint, fields_used, Call}

pub fn query(table: str, @fields_used fields: ?[str] = None) -> [Row] { ... }

/// A query inside a loop runs once per item.
@lint(calls: "query")
fn query_in_loop(c: Call) -> ?str {
    if c.in_loop { return Some("query inside a loop (line {c.loop_line}): use `include`") }
    return None
}
```

- **Library warnings.** `@lint(calls: "query")` (or `"Db.query"` for a method) makes a function a warning of its library: `zyth check`, `zyth lint` and the language server (as you type) call it once for every call of `query` in the program's other files, and a `Some(message)` is warning W0111 at that call, with a note naming the lint (`db.query_in_loop`; the JSON form has `"lint"`). They run only when the program has no errors, never while it runs.
- **What a lint is told.** A `Call`: `name`, `file`, `line`, `col`; `in_fn` (the function or method the call is in, `""` at the top level); `in_loop` and `loop_line` (inside a `for`, `while` or `loop` of that function, or a closure given to a list's `map`, `filter`, `for_each` and the like); `args` (as written, named ones as `name: text`); `fields` (below).
- **Limits.** A lint takes one `Call` and returns `?str` (E0112 otherwise), and does nothing outside the program: one that reads files, uses the network, a GPU, an AI, other programs or foreign code is E0112, worked out like `uses` (§11.4). Only the library's items are made for it (its top-level code does not run), and all of a lint's calls together get 2 seconds. One that fails or runs out of time is a W0111 saying so; the check result is otherwise the same.
- **Fields read.** A parameter marked `@fields_used`, of type `?[str]` with a default (E0112 otherwise), is filled by `zyth run` when the caller leaves it out: the fields the code around the call reads from the result, sorted (`for u in users() { print(u.name) }` gives `Some(["name"])`; `users().len()` gives `Some([])`). The checker follows the result in the function the call is in: kept in a variable, looped over, indexed, given to `filter`, `sorted_by_key`, `take`, `first` and the like, and closures given to `map`, `for_each`, `any`, `fold` and the like, and fields read on the items. Any other use (passed to a function, returned, printed, put in a list, a method of an item called) means it cannot tell, and the parameter keeps its default (`None`): a library then fetches everything. The same set is `Call.fields` for a lint. `zyth run`, `zyth test`, the REPL and notebooks (for each input, also when the library was imported by an earlier one), built programs (`zyth build` records the calls) and JavaScript output (`--target npm-jsx`, for functions of the same file) fill it.

---

## 12. Script Mode, REPL & Debugging

### 12.1 Script Mode

Top-level statements run in order. `fn main()` is optional.

```zyth
// hello.zyth
print("Hello, Zyth!")
```

```bash
$ zyth run hello.zyth
```

### 12.2 REPL

```
$ zyth repl
zyth> let t = tensor.randn(3, 4)
zyth> t.shape
[3, 4]
zyth> t @ t.T
tensor<f32, [3, 3]> ...
```

The REPL runs the full type and shape checker on each input.

### 12.3 Notebooks

`zyth notebook` starts a local notebook server (cells, inline tensor/plot output). A Jupyter kernel (`zyth-kernel`) is also provided so existing Jupyter users can use Zyth without switching tools.

### 12.4 `inspect`

```zyth
inspect(w)
// tensor<f32, [784, 256]> on gpu
//   mean 0.0012  std 1.001  min -3.45  max 3.72
//   NaN 0  Inf 0  grad: yes  memory: 784 KB
```

### 12.5 Reproducibility

```zyth
reproducible(seed: 42) {
    let model = Classifier()
    model.fit(data, epochs: 3)
}
```

`reproducible` (a standard-library function with a trailing block) seeds every random source and selects deterministic kernels inside the block. Results are reproducible on the same hardware and backend; bit-identical results across different GPUs are not guaranteed.

---

## 13. Targets & Deployment

### 13.1 Targets

| Target | Command | Output |
|---|---|---|
| Native (Linux, macOS, Windows; x64 and arm64) | `zyth build --target <triple>` | Single executable |
| Web | `zyth build --target web` | `.wasm` + JS loader + `index.html` |
| Python module | `zyth build --target python` | Installable wheel (`pip install`) — Section 14.5 |
| npm package | `zyth build --target npm` | Package with `.d.ts` types, on the shared runtime package — Section 14.5 |
| React components | `zyth build --target npm-jsx` | Zyth compiled to JavaScript: components, `.d.ts` — Section 14.5 |

GPU code runs through `wgpu` on both targets: Vulkan/Metal/DX12 natively, WebGPU in the browser. When no GPU is available, tensors fall back to the CPU. On macOS the Metal device is also used directly (2026-10-30): tensor buffers are allocated in shared memory, which the CPU reads without a copy, and matrix products go to Metal Performance Shaders; `ZYTH_METAL=0` turns this off and `ZYTH_METAL_GEMM=0` only the products.

### 13.2 What a Native Binary Needs

| You use | Binary contains | Target machine needs |
|---|---|---|
| Pure Zyth | Runtime + your program | Nothing |
| Zyth on GPU | Same | GPU driver (Vulkan, Metal, or DX12) |
| Zyth + npm (pure JS) | Embedded JS engine + bundled packages | Nothing |
| Zyth + C libraries | Statically linked where licenses allow | Shared libraries if dynamically linked |
| Zyth + Python | Python bridge | The project's Python environment — bundled with `--bundle-python` or provisioned on first run (Section 14.2) |

### 13.3 What the Web Build Supports

| Feature | Web |
|---|---|
| Core language, tensors, autodiff | ✅ |
| GPU via WebGPU | ✅ (in browsers with WebGPU; CPU fallback otherwise) |
| npm packages that run in browsers | ✅ |
| DOM / browser APIs (`zyth.web`) | ✅ |
| Python interop | ❌ (Pyodide was dropped: `docs/DECISIONS.md`); web builds use Zyth and npm |
| C FFI | ❌ (unless the C library is compiled to Wasm) |
| File system, raw sockets | ❌ (browser sandbox) |

### 13.4 Browser Model Delivery

Downloading and fitting the model is often the hardest part of browser AI: a 4-bit 1B-parameter LLM is roughly a 700 MB download and needs close to 900 MB of GPU memory. `zyth.web` handles this so every app doesn't reinvent it:

```zyth
@target(web)
use zyth.web.{models, dom}

let model = models.load("tiny-gpt", variants: [
        { file: "tiny-gpt-q4.zyth.model", min_gpu_mb: 600 },
        { file: "tiny-gpt-q8.zyth.model", min_gpu_mb: 1200 },
    ],
    on_progress: { p -> dom.set_text("#status", "loading {p.percent}%") }
)?
```

- **Persistent cache:** model files are stored in the Origin Private File System (OPFS), falling back to the Cache API, so they download once per origin.
- **Resumable downloads:** files are fetched in chunks; an interrupted download resumes where it stopped.
- **Integrity:** files are verified against content hashes from the build manifest.
- **Fit check:** the runtime queries WebGPU adapter limits and picks the largest variant that fits, or falls back to CPU (or fails with a clear error) when none do.
- **Quota check:** storage quota is checked before downloading; when it's too low, the app gets an error it can show the user instead of a silent failure.
- **Weights load as they arrive:** layers become usable as their weights finish downloading, so the app can show progress instead of a blank page.

Implemented so far, for the data and model files a `zyth build --target web` page uses: the persistent cache (files stored under their content hash from the build manifest), the quota check, persistent storage asked for, integrity (every file, downloaded or kept, checked against its SHA-256 in the manifest with `crypto.subtle`; a damaged kept copy is dropped and downloaded again), and resumable downloads (files over 8 MB come in 8 MB Range requests into an OPFS part file closed after each chunk; after an interruption the next page load asks only for the rest; servers without Range support get a whole download; progress as `zyth-progress` events on the document; checked in headless Chromium with a server that fails after one chunk), and since 2026-10-28 `models.load` (`zyth.web.models`, also in native programs): the first argument is the model type or a name (the type then comes from the file's metadata); variant files (records with `file:`) are left out of the page's start-up download and only the chosen one is fetched when `load` runs; the fit check takes the largest variant whose `min_gpu_mb` fits and loads it onto the GPU, else the smallest onto the CPU (`require_gpu: true` makes that an error); `models.choose(variants)` returns the pick. GPU memory: the driver's total on NVIDIA and AMD; in browsers, which do not tell it, an estimate (the adapter's largest buffer ×4, at most half of `navigator.deviceMemory`), set by `?zyth_gpu_mb=N` in the page's address or `gpu_mb:` in the call. `on_progress` gets `{ file, loaded, total, percent, layers_ready, layers_total, done }`, where a layer (tensors grouped by name up to their first number, as `blocks.3`) is ready once all its bytes are in. What "weights load as they arrive" means here: progress per layer while the file comes in; the model itself is returned when the whole file is in (running layers before the rest arrives is not done). Checked in headless Chromium with a 20.6 MB and a 1.5 KB variant: with `?zyth_gpu_mb=8000` only the large file is requested (three Range chunks, layers 1/3, 2/3, 3/3), with 100 only the small one.

### 13.5 Compiled Functions (`--aot`)

`zyth run --aot app.zyth` and `zyth build --aot app.zyth` compile functions to machine code (Cranelift; x86-64 and arm64 on Linux, macOS and Windows; Android phones as Linux arm64). The results are the same as interpreted, including error messages and the "in `f` called at" notes; this is checked by running every `tests/run` program both ways and by `scripts/aot_fuzz.py`, which compares random programs. Parts of a compiled function that the compiler does not handle run in the interpreter, with the variables they use passed back and forth. `--aot-report` lists the functions that stay interpreted, and inside compiled ones each line the interpreter still runs, with the reason:

```
aot: compiled 4 of 5 function(s) to native code (x86_64-sse4.2)
  native       area
  native       largest
  native       row_sums
  native       main
  interpreted  squares: parallel `for` is not compiled
```

When the summary says "all of their code is native", no line of the compiled functions is run by the interpreter (they still call its built-in functions and methods). Compiled: closures (as functions, or as loops when written in place in `xs.map(|x| ...)`, `filter`, `for_each`, `any`, `all`, `sorted_by_key`), closures that capture a variable the function changes (the variable then lives in one shared slot that the function, the closure and interpreted parts all use), `mut self` methods, generic functions (a parameter of type `T` is any value; type parameters check nothing when the program runs), and functions whose types name dimensions (`tensor<f32, [N, D]>`: the caller checks the shapes as interpreted and passes `N` and `D` in). Also compiled: nested functions (lifted to top-level functions; one may call another declared after it in the same run of declarations, as recursive-descent parsers do; one that calls a function declared after other statements stays interpreted), `x |> f(a)`, `&mut` arguments (the borrowed variable lives in a shared slot), and tensor element reads and writes with several indices or slices (`x[i, j]`, `x[i, :]`, `x[i, j] += v`). What stays interpreted: functions with contracts, threads and `parallel for`, and decision points (`decide`, which always runs through the decision runtime). The top level of a program always runs in the interpreter; it is usually a few calls. Number code gains the most (`fib(30)`: 1.1 s interpreted, 16 ms compiled); code that works on text, lists and maps runs about as fast as CPython to 1.4 times faster on this machine (`docs/STATUS.md` has the measurements). Text gets some help: lists of text (from `split` and the like, or declared `str[]`) give elements known to be text, `for` over a value declared as a list, map, set or text runs without the interpreter's loop (a value of another type there is an error), and `split`, `len` and `parse_int() ?? d` on text are native.

**Data in place.** Local lists of numbers (`var xs = [0; n]`, `[1.5, 2.0]`, `var b: bool[] = []`) that are only indexed, measured, grown with `push` and iterated are kept by compiled code as plain memory (8 bytes an element, 1 for bools; any other use reads a copy as a value); `for x in xs` over a list reads its items where they are (the loop keeps the list); struct fields that are numbers are read in place, and a struct literal giving every field is made in one call. Measured: the sieve benchmark 213 → 15 ms, the structs benchmark 60 → 33 ms, a loop summing a field of 1M structs 74 → 10 ns an item.

**Release code.** `zyth build --aot --release` (and `zyth run --aot --release`) has LLVM optimize the compiled functions: their final Cranelift IR is translated to LLVM IR, optimized with `-O2` and linked into the same kind of position-independent block, so the runtime and results are the same. LLVM is used as an installed program (`zig`, or `clang` and `ld.lld`, on PATH; `ZYTH_LLVM_CC`/`ZYTH_LLVM_LD` name others); without it, or for targets other than Linux x86-64 for now, the build notes this and keeps Cranelift's code. Measured: sieve 11.4 → 6.6 ms, fib 6.7 → 5.9 ms; programs whose time goes to values and built-in functions change little.

**Reuse in place.** A function whose only use of a list parameter is `return xs.map(|x| ...)` takes the caller's list when the caller lets go of it (`xs = f(xs, ...)`); a `map` over a list that nothing else refers to then writes its results over the items, and a struct literal it returns overwrites the item (same type, nothing else refers to it) instead of making a new struct. This is the reuse of reference-counting compilers such as Koka's (Perceus); values that are shared are copied as before, so results do not change. Parameters and results with declared types are converted at the call, as the interpreter does, without going through it.

**Checks left out.** No overflow check where `if`/`while` conditions on an integer bound the operands so that the result fits; none for `x += 1` on a variable declared once as `var x = k` with |k| <= 2^62 and changed only that way (reaching an overflow would take more than 2^62 additions, over a century at one per nanosecond). List indexes are checked with one comparison; negative indexes take a slower path.

**Inlining.** Calls of compiled functions marked `@inline`, and of other compiled functions of up to 40 Cranelift instructions unless they are marked `@noinline`, are replaced by the callee's code (Cranelift's inliner; callees are compiled first, recursive calls stay calls). Errors inside inlined code report the callee's line. Measured: a loop of 30M iterations calling three small helpers takes 185 ms inlined and 278 ms with `@noinline` on them.

A built program starts with `zyth-rt`, the runtime without the compiler, checker and tools (11 MB; the whole `zyth` is 20 MB), when it sits next to `zyth`, or with `zyth-rt-lite` (7 MB: no JavaScript engine, HTTPS kept) when the program imports no npm package; `ZYTH_RUNTIME=path` names another. The runtime still contains the interpreter's library of values, built-in functions and methods: compiled code calls it for everything but arithmetic, comparisons and control flow, which is what keeps its results and error messages identical to interpreted runs.

### 13.6 Deployment

```bash
$ zyth build --release --target linux-x64 app.zyth
$ scp app server:/usr/local/bin/ && ssh server ./app

$ zyth build --target web app.zyth
$ npx netlify deploy --dir dist     # any static host works
```

---

## 14. Interop: C, Python, npm & Models

### 14.1 C FFI

```zyth
extern "C" from "sqlite3" {
    fn sqlite3_libversion() -> ptr<u8>
}

danger {
    print(str.from_c(sqlite3_libversion()))
}
```

- FFI calls require `danger`.
- The standard library wraps common C libraries safely where useful (SQLite, zlib).
- Native targets only (unless the library is compiled to Wasm): Linux, macOS and Windows (x86-64 and ARM64), and Android with the `aarch64-linux-android` build (a static musl build has no `dlopen`). Without `from`, the functions come from the running program on Unix and from the C runtime (`ucrtbase.dll`) on Windows; `from "m"` and `from "c"` name the C runtime on Windows.
- Arguments and results are integers, floats, `bool`, strings, `ptr<T>`, and structs by value: a Zyth `struct` whose fields are C scalars is laid out as C lays it out. There is no limit on the number of parameters (calls go through libffi).
- Variadic functions end their parameter list with `...`; the extra arguments take C's default promotions (integers as `int` when they fit in 32 bits, else `long long`; floats as `double`; strings and pointers as pointers):

```zyth
struct DivT { quot: i32, rem: i32 }
extern "C" {
    fn div(n: i32, d: i32) -> DivT
    fn snprintf(buf: ptr<u8>, size: u64, fmt: str, ...) -> i32
}
```

- Structs may hold structs by value, fixed-size arrays (`scores: i32[3]`, a list of exactly that many values; read back as a list) and unions. A union is a struct marked `@c_union` whose fields are all optional: exactly one is set when it is passed to C, and a union returned by C has every field set, each reading the same bytes (it can be passed back unchanged; its bytes are kept):

```zyth
@c_union
struct Num { i: ?i64, d: ?f64 }
struct Tagged { kind: i32, value: Num }
struct Player { name: u8[12], scores: i32[3], level: u8 }
```

- A parameter of type `fn(A, B) -> C` (scalars and pointers) takes a Zyth function, which C may call while the call runs (`qsort`'s comparison, a numerical integrator's integrand). It is invalid after the call returns: C must not keep it. An error inside it cannot unwind through C: the callback returns zero to C, later calls of it during the same C call return zero at once, and the error is raised when the C call returns.
- `ptr` values read and write C memory: `p.read_i32(at)`, `p.write_f64(at, v)` (`at` in bytes; `i8` to `u64`, `f32`, `f64`, `bool`, `ptr`), `p.offset(n)`, `p.is_null()`, `p.addr()`. Nothing is checked, as in C; it needs the `ffi` permission.
- Not supported: bit fields, packed structs, `long double`, structs inside a union's or callback's parameters, and callbacks C keeps after the call.

### 14.2 Python

```zyth
use python.pandas as pd
use python.matplotlib.pyplot as plt

let df = pd.read_csv("data.csv")
let x: tensor<f32, [_, 3]> = tensor.from(df[["age", "income", "score"]].to_numpy())

plt.plot(losses)
plt.savefig("loss.png")
```

**Native builds (v0.1):** every Python library works, including packages with compiled extensions (NumPy, pandas, PyTorch, scikit-learn, OpenCV).

- Implemented by embedding CPython (via PyO3).
- Python values are dynamically typed in Zyth (type `pyobject`); converting to Zyth types is explicit (`tensor.from`, `as str`, ...).
- NumPy arrays convert to tensors without copying where layouts allow.
- Python calls hold the Python GIL; Zyth tasks keep running in parallel outside Python calls.

#### Managed Python Environments

Zyth manages Python for each project, so users never create or activate a virtual environment:

```bash
$ zyth add py:pandas            # resolves, installs, and pins pandas
$ zyth add py:torch@2.5
```

```toml
# zyth.toml
[python]
version = "3.12"                # Zyth downloads this interpreter if it isn't installed

[dependencies]
pandas = { py = "2.2.3", allow = ["fs:read:./data"] }
torch  = { py = "2.5.1", allow = ["gpu"] }
```

- Each project gets its own environment in `.zyth/python/`, created and updated automatically by `zyth run`, `zyth test`, and `zyth build`.
- Built on `uv` (a fast Python package and interpreter manager): Zyth can download the requested Python version, so a Python installation isn't a prerequisite on development machines.
- Python packages are pinned with hashes in `zyth.lock`, alongside npm and Zyth packages — one lockfile for everything.
- `zyth run` on a fresh checkout recreates the exact environment from the lockfile.
- Deployed native binaries that use Python need the environment on the target machine: `zyth build --bundle-python` ships the interpreter and packages next to the binary (a directory, not a single file); otherwise the binary provisions it on first run.

#### Python in the Browser (dropped)

Web builds do not include Python. Pyodide (CPython compiled to WebAssembly) was planned as an opt-in and dropped: it adds 6-7 MB of CPython plus packages and takes seconds to start (`docs/DECISIONS.md`). In the browser, Zyth's own tensors and npm packages that run in browsers cover what Python would have been used for.

### 14.3 npm

```zyth
use npm.marked
use npm.zod as z
use npm."date-fns".{format}

print(marked.parse("# Hello"))
```

**v0.1 scope:**
- **Web target:** any npm package that runs in a browser. Zyth's Wasm module and JS packages are bundled together.
- **Native target:** packages run in one of two engines:
  - **The embedded engine**: QuickJS-ng 0.16 (through `rquickjs` 0.14) inside the Zyth binary, with Zyth's Node.js core modules (M13; Express runs). It is the only engine that holds each package to its own `allow` list (Section 15) and limits the network, so it is used whenever the project has `[permissions]` or per-package limits, and by built programs (`zyth build`), which then behave the same on every computer.
  - **The computer's own Node.js** (22 or newer, on the `PATH` or in `ZYTH_NODE`), as a separate process: everything else in `zyth run`. Calls, callbacks into Zyth, handles and printing cross a pipe (one JSON message per line); Node's event loop runs between calls, and a call's ticks and promise jobs run before it returns, as when Node calls a callback.
  - The choice: `ZYTH_NPM_ENGINE` (`auto`, `node`, `embedded`), else `[npm] engine = "..."` in `zyth.toml`, else `auto` (as above). `node` in a project with limits falls back to the embedded engine with a note.
  - Measured (this machine, `bench` of six pure-JavaScript loops, ms): QuickJS 0.11 → QuickJS-ng 0.16 → Node 24: recursion 88 → 85 → 9, JSON 265 → 219 → 46, sort 228 → 234 → 104, string building 500 → 30 → 8, regular expressions 51 → 17 → 5, `Map` 170 → 122 → 91. Starting Node adds about 55 ms to a run (27 → 81 ms for `tests/npm/app.zyth`).
- **Not in v0.1:** native addons in the embedded engine.

**Types:** if a package ships TypeScript declarations (or has `@types/*`), Zyth uses them to type-check calls. Otherwise its values have type `jsvalue` and the compiler emits a warning.

**Install:** `zyth add npm:marked` records the dependency in `zyth.toml` and a lockfile.

### 14.4 Models

```zyth
let m = hub.load("gpt2")                              // Hugging Face Hub
let m = hub.load("meta-llama/Llama-3.2-1B", dtype: bf16)
let m = import.safetensors("model.safetensors", arch: Llama)
let m = import.onnx("model.onnx")
let m = import.gguf("model.gguf")
```

- Native readers for safetensors, GGUF, and ONNX (weights + supported ops).
- PyTorch `.pt` files are pickled Python objects, so they load through the Python bridge (native only). Prefer safetensors.
- Architectures supported out of the box are those in `zyth.nn.arch` (GPT-2, Llama, BERT, ViT, Whisper encoder/decoder in v0.1). Others need their architecture written in Zyth.

**Export (no lock-in):**

```zyth
model.export("model.safetensors")
model.export("model.onnx")              // supported ops only; errors list the unsupported ones
model.export("model.gguf", quant: q4_k) // for llama.cpp / Ollama
model.export("model.mlir", input: x)     // StableHLO: XLA (TPUs, through PJRT) and IREE
model.export("model.mlir", input: x, weights: "args")   // weights as arguments, in model.weights.safetensors
```

StableHLO export (2026-10-31) lowers the same operation trace as ONNX into StableHLO's MLIR text, with the example input's static shapes: the input that Google's TPUs take (XLA through PJRT: JAX, PyTorch/XLA, Cloud TPU VMs), and that IREE compiles for CPUs and other accelerators. Covered: element-wise and comparison operations with NumPy broadcasting, matrix products (batched and broadcast), reductions, softmax and log-softmax, layer norm, gather (embeddings), slices, concatenation, masks, convolutions and pooling. Checked by running the exports of an MLP, a CNN, Zyth's Transformer, Llama and GPT-2 in XLA's CPU compiler and in IREE (`scripts/stablehlo_check.py`): the largest difference from Zyth's own outputs is 8.3e-7. Not done: running on a TPU (none here), training steps (the export is of the forward pass), and AWS Trainium (its compiler, Neuron, also takes StableHLO; not tried).

### 14.5 Exporting Zyth to Python and npm

Interop works both ways. Zyth code can be packaged as a Python module or an npm package, so Python and JavaScript developers can use Zyth without switching languages. This is the main adoption path: write one fast, shape-safe component in Zyth and drop it into an existing project.

```zyth
// fastops.zyth
@export
fn rms_norm(x: tensor<f32, [B, D]>, weight: tensor<f32, [D]>) -> tensor<f32, [B, D]> {
    return x / (x ** 2).mean(axis: -1, keep: true).sqrt() * weight
}

@export
model TinyClassifier {
    fc = Linear(784 -> 10)
}
```

**Python:**

```bash
$ zyth build --target python fastops.zyth     # produces fastops-0.1.0-<platform>.whl
$ pip install dist/fastops-0.1.0-*.whl
```

```python
import numpy as np, fastops
y = fastops.rms_norm(np.random.randn(64, 512).astype("float32"),
                     np.ones(512, dtype="float32"))
# Wrong shapes raise a ZythShapeError naming the parameter and expected shape
```

- NumPy arrays and PyTorch CPU tensors convert without copying where layouts allow.
- Type stubs (`.pyi`) are generated so editors show shapes and types.

**npm:**

```bash
$ zyth build --target npm fastops.zyth        # produces a package with .d.ts
$ zyth build --target npm-runtime             # the shared runtime, dist/zyth-runtime (once per app)
$ npm install ./dist/fastops ./dist/zyth-runtime
```

```typescript
import { rmsNorm } from "fastops";
const y = await rmsNorm(x, weight);   // Float32Array or zyth Tensor; runs on WebGPU when available
```

- TypeScript declarations are generated from Zyth signatures.
- Works in browsers, Node.js, Deno, and Bun.
- **One runtime per app.** A package holds its Zyth program and its wrappers, not zyth.wasm: the runtime is the package `@biodream-llc/zyth-runtime` (npm scope of BioDream LLC), a peer dependency at exactly the version of `zyth` that built the package. It carries the full zyth.wasm and the light one; one instance runs the programs of all the app's Zyth packages (each loaded once, at its first call), so the app downloads one of the two once. It is the light one unless a package registered by the first call uses `zyth.gfx` or `zyth.media` or mentions ONNX, StableHLO or GGUF files; a package needing the full one that arrives later (a lazy import) starts a full instance beside it. Tensors and errors are the same classes in every package; a model belongs to the package that made it (passing it to another is E0101). Calls into the runtime run one at a time.
- **Checked bytecode.** A package carries its program as checked bytecode (the syntax tree `zyth build` checked, in a compact binary form, `zyth_syntax::bin`) beside its source (kept for messages), so the runtime neither parses nor checks it again. The light runtime has no parser, no checker, no ONNX reader, no ONNX/StableHLO/GGUF writers and no loader for compiled native code; it runs bytecode from the same version of Zyth only (another version's is refused: build the package again). Its standard modules written in Zyth (`zyth.nn`, `zyth.train`, ...) are inside it as bytecode too. GGUF and safetensors weights still load in it.

**React components (`--target npm-jsx`).** Zyth compiled to JavaScript, so a component is a plain JavaScript function and renders never cross into WebAssembly. `@export` functions whose name is capitalized and that return `zyth.react.Element` become components; their parameters are the props (camelCase in JavaScript), checked by `zyth check` where the component is used. Elements are tag calls (all of HTML's elements but the document's own, and SVG's, `linear_gradient` for linearGradient) with props as named arguments and a trailing block of children; `for` and `if` in the block add children, and a component's `children: Element` parameter gets its block. React's hooks are there (`use_state`, `use_reducer` with a dispatch checked against the reducer's action type, `use_effect`, `use_layout_effect`, `use_memo`, `use_callback`, `use_ref`, `use_id`, `use_transition`, `use_deferred_value`), contexts (`create_context`, `provider(ctx, value:) { }`, `use_context`, typed by the default) and react-redux's `use_selector` and `use_dispatch` (react-redux becomes a peer dependency); `tests/jsxreact` renders them with React 19 and a Redux store:

```zyth
use zyth.react.{div, h1, button, ul, li, use_state, Element}

@export
fn Counter(label: str, start: i64 = 0) -> Element {
    let (count, set_count) = use_state(start)
    return div(class: "counter") {
        h1 { "{label}: {count}" }
        button(on_click: || set_count(count + 1)) { "+1" }
    }
}
```

The package is an ES module with `index.d.ts` and React as a peer dependency. Values: integers are JavaScript numbers, exact to 2^53 (docs/DECISIONS.md); lists and tuples are arrays, structs and records objects, `None` null. Lists and structs stay values: `xs.push(x)`, `p.x = 1` and `xs[i] = v` make a new value and store it back, as React state needs. Programs print what `zyth run` prints (`tests/jsx`). **Mixed packages.** Functions JavaScript output cannot write (tensors, models, modules other than `zyth.react`, anything else refused), and functions that call them, run in WebAssembly: the build lists each with the reason and writes `wasm.js`, which runs them on the shared runtime package (above; the full runtime when the source uses `zyth.gfx` or `zyth.media`), loaded at the first call. Their answers take time, so components call them in event handlers (which become `async`) or through `use_async(|| f(x), deps: [x])`, which is `None` until the value arrives; a call while rendering is E0130. Structs, enums (with or without fields), tuples, lists and options cross both ways (arguments are decoded by the parameter's type; tuples coming back keep their kind). A model a WebAssembly function returns is a handle JavaScript can keep and give back; its methods are called the same way as WebAssembly functions (in handlers or `use_async`), and it is freed when JavaScript lets go of it. State setters take the state's type. Top-level values JavaScript cannot hold, and top-level statements of a mixed package, run in WebAssembly when it loads. Tested by `tests/jsxmix` (rendered in Node with a React stand-in). Components themselves always run in JavaScript. **3D:** `gfx.window(..., host: "id")` puts the window's canvas into the page element with that id (elsewhere `host:` is ignored); a component renders `div(id: "id")` and calls a drawing function from `use_frame(|t| draw(t))`, which runs once per animation frame, the next only after the last finished. Calls into a package's WebAssembly run one at a time (a frame waiting for the screen is never entered by a click's call).

Not in JavaScript output yet, reported at build time (E0130) or moved to WebAssembly as above: `?`, `mut borrow`, sets, loops that change their items in place, generators, `go`, `parallel`, `danger`, AI blocks, local modules, and methods not mapped yet.

### 14.6 Type Definitions for Python & npm Packages (`zyth-types`)

Calls into Python and npm are dynamically typed by default (`pyobject`, `jsvalue`), so they lose Zyth's shape checking. Type definition files fix that for the most-used libraries — the same approach TypeScript took with DefinitelyTyped, extended with tensor shapes.

#### Definition Files (`.zyi`)

```zyth
// zyth-types/python/numpy/2.1.zyi
declare module python.numpy {
    type ndarray<T, S> = tensor<T, S> on cpu       // NumPy arrays map to CPU tensors

    fn matmul<T: Numeric, M, K, N>(a: ndarray<T, [M, K]>, b: ndarray<T, [K, N]>) -> ndarray<T, [M, N]>
    fn zeros<T = f64>(shape: Shape) -> ndarray<T, shape>
    fn concatenate<T, A, B, R>(arrays: (ndarray<T, [A, R]>, ndarray<T, [B, R]>), axis: 0) -> ndarray<T, [A + B, R]>
}
```

```zyth
// zyth-types/npm/zod/3.23.zyi
declare module npm.zod {
    fn string() -> ZodString
    fn object<S: Schema>(shape: S) -> ZodObject<S>
}
```

`declare` is only valid in `.zyi` files. Definitions describe existing code; they contain no implementation.

#### Where Types Come From (in Order)

1. Definitions shipped inside the package itself.
2. The `zyth-types` community registry (https://types.zyth.io; sources in `registry/types` of the Zyth repository), matched to the installed package version: the same major.minor, else the newest earlier one with the same major version.
3. Auto-converted existing types: Python `.pyi` stubs and type hints, TypeScript `.d.ts` files. These provide argument and return types but no tensor shapes.
4. Otherwise dynamic (`pyobject` / `jsvalue`), with a compiler warning.

`zyth add py:<package>` and `zyth add npm:<package>` fetch the registry's definitions into the project's `zyth-types/` (checked against the SHA-256 in the registry's index and parsed before they are kept; a file the project wrote itself is never replaced), and `zyth types fetch` does the same for packages already installed. `ZYTH_TYPES_URL` points elsewhere; `off` stops fetching. `zyth types status` shows the type coverage of every dependency. `zyth types gen py:<package>` and `zyth types gen npm:<package>` write `zyth-types/python.<package>.zyi` / `zyth-types/npm.<package>.zyi` from the package's stubs, annotations or TypeScript declarations (step 3), to use as they are or as a start for hand-written shapes. TypeScript's `number` becomes `float`; classes are an opaque type plus a constructor (`Cls(args, new: true)`).

Named arguments to an npm function become one options object in the next parameter's place, as when the program runs (`marked.parse(src, gfm: true)` passes `{ gfm: true }`); a named argument where the declaration has a plain value (`count(text, unit: "chars")` with `unit: str`) is an error that says to pass it by position.

#### Keeping Definitions Honest

Hand-written definitions can be wrong. Zyth checks them:

- **Debug builds** validate values at the interop boundary against the declared types and shapes. A mismatch is reported as a definition bug (with the file and line of the `.zyi`), not as an error in the user's code.
- **Release builds** skip the boundary checks for speed.
- Registry definitions are versioned per package version and tested in CI against the real package.

#### Seeding the Registry

The project maintains first-party definitions for the libraries that matter most to Zyth's users, and the community covers the long tail:

| Ecosystem | First-party definitions (initial set) |
|---|---|
| Python | numpy, pandas, scikit-learn, torch (tensor-facing API), transformers (model I/O), matplotlib |
| npm | three, d3, zod, marked, date-fns |

Starter definitions are generated from existing `.pyi` / `.d.ts` files, so contributors mostly add shapes instead of writing definitions from scratch.

---

## 15. Package Security & Permissions

Zyth pulls code from npm, PyPI, and Git repositories of Zyth packages, so it inherits their supply-chain risks (OpenSSF reported 1.8 million malicious packages flagged in 2026 alone). Every dependency runs with explicit permissions.

### 15.1 Declaring Permissions

```toml
# zyth.toml
[permissions]                       # what YOUR program may do
allow = ["net:api.example.com", "fs:read:./data", "gpu"]

[dependencies]
marked   = { npm = "12.0.2", allow = [] }                     # pure compute
requests = { py = "2.32.3", allow = ["net"] }
tokenizer_utils = { zyth = "0.4.1", git = "https://github.com/someone/tokenizer_utils", allow = ["fs"] }
```

A Zyth package's `allow` is checked when the program is checked (Section 15.3), so it names capabilities, not paths: `fs` (or any `fs:read:`/`fs:write:` entry) for files, `net`, `gpu`, `subprocess`, `ffi`, `ai` (also given by `net`). Paths and hosts are still limited by the program's `[permissions]` while it runs.

### 15.2 Capabilities

| Capability | Grants |
|---|---|
| `net`, `net:<host>` | Network access (all hosts or one) |
| `fs:read:<path>`, `fs:write:<path>` | File access, scoped to a path |
| `env` | Environment variables |
| `gpu` | GPU device access |
| `subprocess` | Starting processes |
| `ffi` | Calling C code — this is full trust, and `zyth lint` flags it |

Missing permissions fail with a clear error at the call site: `permission denied: package `quiet` attempted reading `data/a.txt` without the `fs:read:./data` permission; add "fs:read:./data" to `allow` of `quiet` under [dependencies] in zyth.toml`.

For npm packages (enforced now): a dependency's `allow` also covers the packages it depends on (from their `package.json`; a package reached from two dependencies may do what either allows), and a package may always read installed packages' files. A dependency without `allow` is limited only by the program's `[permissions]`. Each file, socket and process call checks every limited package whose code is on the JavaScript stack, so a package cannot use another package's functions to do what it may not; callbacks a package schedules (timers, `process.nextTick`, `queueMicrotask`, promise reactions) keep its limits. This guards against careless or malicious updates; it is not a sandbox for hostile code, which could still reach the engine's internal functions.

### 15.3 How Each Kind of Package Is Enforced

| Package kind | Enforcement | Strength |
|---|---|---|
| Zyth packages | The checker infers the effects of all of a package's code (its functions, and its top-level code, which runs when it is imported), counting calls into other packages, a module function taken as a value (`let f = fs.read`) and a module passed as an argument; one its dependency's `allow` does not grant is error E0235 in `zyth check`, `run`, `build` and `test`. Not traced: calls through values whose type the checker does not know | Strong for direct calls |
| npm (embedded JS engine) | The engine has no built-in I/O; Zyth only injects the APIs a package is allowed to use | Strong |
| npm (web target) | Browser sandbox, plus the same injected-API rule | Strong |
| Python | Restricted packages run in a separate, OS-sandboxed process (Landlock/seccomp on Linux, sandbox profiles on macOS). In-process mode is faster but means full trust and is flagged as such | Medium — process isolation costs speed |

**Python modes.** In a project with `[permissions]` that does not grant `ffi`, `use python.X` starts a worker process (the project's `.venv/bin/python` or `.venv\Scripts\python.exe`, else `python3` or `python.exe`) and locks it in before any package is imported. It may read Python's own files and what the package's `allow` list under `[dependencies]` gives (`numpy = { py = "2.1.0", allow = ["fs:read:./data"] }`), write only a private temporary directory and its `fs:write` paths, and nothing else: on Linux Landlock (files, TCP ports, abstract sockets, signals) and seccomp (no ptrace, namespaces, kernel modules, BPF; no new programs without `subprocess`; no internet sockets without `net`), on macOS a sandbox-exec profile with the same rules, on Windows an AppContainer (the files Zyth names in access lists only; the network only with `net`, never this machine's loopback address) and a job object (no new programs without `subprocess`), one container per allow list. On Windows, Python's own folder must be one this user may grant access to (a Python installed by uv or for this user); a Python under `C:\` or `Program Files` needs a one-time grant from an administrator prompt, and the error gives the `icacls` command. Each `allow` entry must also be allowed to the program. The OS cannot limit `net:<host>` to that host: it allows TCP to ports 80 and 443 (or the port given). The environment is cleared unless the package has `env`. Python objects stay in the worker (`pyobject` handles; `obj.method(...)` is one round trip), and tensors travel as raw bytes. Without `[permissions]`, or with `ffi`, Python runs in-process (builds with `--features python`) or in a worker without the sandbox. `ZYTH_PYTHON=sandbox|process|inprocess` overrides the choice. Measured cost (`bench/python`, October 2026): the sandbox adds nothing measurable to a separate process; the process adds 12 µs per call on an Apple M4 and 65 µs on Linux under WSL2, about 2-4x the time of calls passing 64x64 to 256x256 tensors, and 0-2x for 1024x1024 tensors; calls that leave arrays in Python and compute on them cost the same in both modes. So the sandbox is the default only where the project restricts permissions.
| C FFI | Cannot be sandboxed; requires `ffi` | Full trust |

### 15.4 Supply-Chain Defaults

- Lockfile (`zyth.lock`) pins every package by version and content hash. For Zyth packages it holds the commit and a SHA-256 of the files; every `zyth run`, `check`, `build` and `test` compares the installed copies with it (a changed file is an error; `zyth install` restores it), a moved Git tag is reported and the locked commit used, and downloaded files that differ from the hash are refused.
- npm install scripts (`postinstall`) never run.
- `zyth audit` checks dependencies against public vulnerability advisories.
- New dependencies require `zyth add`, which shows the permissions requested before installing.
- **Minimum release age.** Most malicious versions are found and removed within hours of publishing, so Zyth does not pick a version younger than `[packages] min_age` (default `"1d"`; `"0"` turns it off; `ZYTH_MIN_AGE` overrides it; `--allow-new` on `zyth add` and `zyth update`), as pnpm 11 does by default. npm versions by the registry's publish time (for every package resolved, not only the one named), PyPI versions through `uv --exclude-newer` or pip 26's `--uploaded-prior-to` (older pip: a note, no limit), Zyth packages by the latest of three times: the tag's date (which its author writes), when this computer first saw the tag, and when the checksum log (Section 15.5) first saw the version, so a release that claims an old date is still counted from when it appeared, even on a computer reading the repository for the first time. The message names which of the three it counted from. A version asked for by number that is too young is refused with the reason. Without the log (unreachable, or `ZYTH_SUMDB=off`) a first install has only the tag's date; minimal version selection (Section 15.5) already never moves to a version nobody asked for. The log records a version when someone first asks for it, so a version nobody has asked the log about waits out the minimum age from that first request.
- **Names that do not exist.** AI models name packages that do not exist (open models in about 22% of samples, commercial ones in about 5%, in a 2025 study of 576,000 samples), and attackers register those names ("slopsquatting"). `zyth add npm:`/`py:` first asks the registry: a package that does not exist is an error that lists the existing names closest to it (npm's search; PyPI's 15,000 most downloaded projects), and one first published under 30 days ago, downloaded under 100 times last week, with a single release, or replaced by npm's security placeholder needs confirmation (`--yes`). `zyth vet <npm:|py:|git:...>` and the MCP tool `check_package` give the same verdict to agents before they write the import; `zyth make` reports a made-up or risky npm import to the model as a problem to fix. Offline, the check is skipped with a note.
- **What changed on update.** `zyth update` shows, for each package that moves, what the new version can do that the old one could not: for Zyth packages the effects the checker infers (exact for direct calls), with the function and line; for npm and Python packages a scan of their files for the modules and calls that reach the network, files, processes, environment variables, native code and code built from text (it reads names, so it can miss code and flag harmless code; it is meant for the difference). New permissions for a Zyth package are asked for and then written to its `allow` (`--yes` gives them); a declined npm or Python update puts the old version back.

### 15.5 Zyth Packages

A Zyth package is a Git repository with a `zyth.toml` whose `[project] name` is the package's name; its versions are its tags (`v0.4.1` or `0.4.1`). The registry (registry.zyth.io) maps names to repositories, so `zyth add geom` works without a URL; the code stays in Git.

```bash
$ zyth add git:github.com/someone/geom          # the newest version old enough, or @0.4.1
$ zyth install                                  # a fresh checkout: exactly what zyth.lock pins
$ zyth update [geom] [--major] [--yes]          # newer versions, with what they can now do
```

- **Versions are chosen as Go does (minimal version selection).** The version in `[dependencies]` is a minimum; a package's own `zyth.toml` lists its dependencies the same way. Of all the versions the project and its packages ask for, Zyth takes the highest for each package and never one nobody asked for, so a new release changes nothing until someone asks for it and the lock only adds the commit and hash. Two packages with the same name from different repositories, or two major versions (1.0 and later) of one package, are an error: a new major version is a new package with a new name (`geom2`).
- **Where they live.** Downloads go to a cache shared by projects (`~/.cache/zyth/git/`, `ZYTH_CACHE_DIR`) and are copied into `.zyth/pkg/<name>/` of the project. `use geom` loads the package's `lib.zyth` (or `src/lib.zyth`), `use geom.shapes` its `shapes.zyth`. Package names are unique in a project, so a package's files import each other by the package's name (`use geom.util`), and a project can import its own files by its `[project]` name (how a package is written and tested in its own repository). A package that imports a project file, or a project folder with a package's name, is error E0109. `zyth build` bundles the packages a program imports.
- **Versions follow the API (as Elm's package tools enforce).** `zyth api` lists the public items of the project's modules with their types. `zyth bump` compares them with the newest version tag and sets `version` in zyth.toml to the smallest version the changes allow: a major version when an item, parameter, field or variant was removed or changed, a required parameter or a field without a default was added, or a variant was added (a `match` that lists every variant breaks); a minor version for additions; a patch otherwise. Before 1.0 the minor number counts as the major one. `zyth publish` refuses a version that is not new or is smaller than the changes need, uncommitted changes and code that does not check, then tags the commit `v<version>`; it does not push.
- **Updates that fix your code.** A package that renames an item keeps the old name for a release, marked `@deprecated("...", replace: "new_name")`; that is an addition, not a breaking change. When `zyth update` moves a project to that release it runs `zyth migrate`, which rewrites the project's uses of the old name. A following major release can then drop it.
- **Docs at the installed version.** `zyth doc --deps [package]` and the MCP tool `package_docs` print a compact reference for the packages a project has installed, at their versions: Zyth packages from their code and `///` comments, npm packages from their TypeScript declarations (or `zyth-types/npm.<name>.zyi`), Python packages from their `.zyi` definitions. A model's memory of a library is often of another version; this is the one the project runs.
- **Single-file scripts.** A script can list what it needs in a header, as PEP 723 does for Python; `zyth run`, `check`, `build` and `test` then make its project in the cache (packages, npm packages, a Python environment), keep its lock next to it (`tool.zyth.lock`), and use that project (`ZYTH_PROJECT`):

```zyth
// /// zyth
// [dependencies]
// geom = { zyth = "0.4.1", git = "https://github.com/someone/geom", allow = [] }
// marked = { npm = "18.0.14" }
// [permissions]
// allow = ["net"]
// ///
use geom
```

- **The registry.** `zyth register` (after `zyth publish` and pushing the tag) puts the project's name in the registry for its repository: the registry reads the newest version tag itself and takes the name only if that version's `zyth.toml` says it, and a name belongs to the publisher who first registered it (`zyth login --registry URL --token T`; tokens come from the registry's operator for now). `zyth add <name>[@version]` asks the registry for the repository, `zyth search <words>` searches names and descriptions. `ZYTH_REGISTRY` names another registry; `zyth registry-server` runs one.
- **The checksum log.** sum.zyth.io keeps a public, append-only log of every version's content hash, as Go's checksum database does, so a version's files cannot change unseen, even for a project installing it for the first time. The log computes each hash itself from the package's Git origin (never from a client, a proxy or the registry, which it does not share code paths or data with: the independence Go's database lacked in 2026), records `<repository> <version> <tag> <commit> <hash> <first seen>`, and signs the tree head with Ed25519 (the key is built into Zyth). Its tree is RFC 6962's Merkle tree: every lookup comes with an inclusion proof, and the client asks for a consistency proof from the last tree head it saw, so a log that rewrites its past is caught. `zyth add`, `install` and `update` check each version not yet in `zyth.lock` against it and refuse one whose commit or files differ (a moved tag), and the time the log first saw a version counts toward the minimum release age (Section 15.4); when the log cannot be reached they say so and rely on the lock. `ZYTH_SUMDB` names another log (`off`: no checks), `ZYTH_SUMDB_KEY` its key; `zyth sum-server` runs one. **Auditors and witnesses.** Anyone can check the log without trusting it: `zyth sum-audit --log URL --dir DIR` reads every record (`/records`), rebuilds the tree and compares it with each signed head (a log that changed its past, or signs a tree it does not show, is caught), refuses two hashes for one version, and works out each new version's hash from its Git origin itself (a log that lied about a hash is reported as a finding). Each head it has checked it cosigns with its own Ed25519 key and serves at `/cosigned` (findings at `/findings`), as a witness. A client given witnesses (`ZYTH_SUMDB_WITNESSES=URL=KEY,...`) requires the log's tree to extend, or be extended by, each witness's latest cosigned head, so the log cannot show one history to it and another to its auditors.

---

## 16. Built for AI Coding Assistants

Most code is now written with AI help. Zyth is designed so AI assistants write correct Zyth, and the compiler catches their mistakes — especially tensor shape bugs — before code runs.

### 16.1 Machine-Readable Diagnostics

```bash
$ zyth check --json train.zyth
```

```json
{
  "code": "E0401",
  "message": "matmul dimension mismatch",
  "file": "train.zyth", "line": 14, "col": 22,
  "shapes": { "left": [64, 512], "right": [256, 10] },
  "provenance": [
    { "dim": 512, "from": "Linear(784 -> 512)", "file": "model.zyth", "line": 4 }
  ],
  "fixes": [
    { "file": "model.zyth", "line": 6,
      "replace": "Linear(256 -> 10)", "with": "Linear(512 -> 10)" }
  ]
}
```

Every error that has an unambiguous fix includes it as an edit an agent can apply directly.

### 16.2 `zyth mcp` — Compiler as an MCP Server

`zyth mcp` runs a Model Context Protocol server so AI agents can call the toolchain directly:

| Tool | Purpose |
|---|---|
| `check` | Type and shape check a file or project |
| `infer_shapes` | Return the inferred shape of every tensor expression in a range |
| `run_tests` | Run tests and return structured results |
| `explain` | Explain an error code with examples |
| `search_docs` | Search the standard library documentation |
| `type_at` | The type of the expression at a line and column, where its name is defined, and a function's header and effects |
| `find_uses` | The definition of a name and every place that uses it |
| `edit_impact` | Before an edit is made: the problems it adds and removes, and the callers of each function whose header or effects it changes |
| `fix` | Apply every machine-applicable fix, re-checking until none is left; returns the new source and what is left |

The same questions are on the command line as JSON (`zyth query type FILE:LINE:COL`, `zyth query uses FILE:LINE:COL`, `zyth query impact FILE --edit L:C-L:C TEXT` or `--new OTHER.zyth`, `zyth fix FILE`), and in the editor (`zyth lsp`: hover shows the type at any point, references, go to definition, quick fixes). Where the checker's own edits stop, `zyth fix --ai --decider SPEC [--run]` asks an AI to repair what is left (given the program, the reference and the problems; its answer is checked again, for up to three rounds), so an agent's loop gets a program that checks (and with `--run` runs) without writing the prompt itself. Every answer reports how long the check took (`ms`); a 70-line program checks in about 5 ms and a 560-line one in about 11 ms (measured with `zyth query`; the parse of the standard library is part of each check).

**Fixes.** Errors whose correction is clear carry an edit (`fixes` in the JSON): `let` that must be `var`, `?` in a function that cannot return an error (`.unwrap()`), casting a maybe-absent value (`.unwrap()`), comparing a maybe-absent value with a plain one (`Some(...)`), a result that is thrown away (`sort` for `sorted`), `//` used as division, a missing or wrong `uses` clause, misspelled names and methods, and tensor shape mismatches. On the programs AI models wrote while being tested (§16.5), 50% of the errors left have such an edit.

**Other spellings.** The parser also accepts these forms from other languages, and `zyth fmt` writes them the Zyth way: Rust macros `println!("{} {:?}", a, b)`, `format!`, `panic!`, `assert!`, `assert_eq!` (the `{}` holes take the arguments in order); `x!` (unwrap); `[]T`, `vec<T>`; `f(mut x)` and `mut T` parameters (a `mut borrow`); `var m = {}: map<str, i64>`; number suffixes (`10.0_f64`, `1u8`); a lone `{` in text (`"{"`); `{` on the line after `for`/`if`/`fn`; a line starting with `.method()`, `and`, `or`, `&&` or `||` continues the one above; `-> T, uses ...`.

**Habits accepted with one meaning (2026-10-28).** From the mistakes left in the 1,380 programs AI models wrote in the eval rounds (§16.5), these now run with the meaning the writer had, and `zyth fmt` writes the Zyth form: a `let` the code changes later (a `var`); `a, b = b, a` and `(a, b) = (b, a)`; `n // 2` (`n.floor_div(2)`); `xs.pop() ?? "" != "("` (Swift's `??` order); comparing an optional with a plain value; an optional or Result used as a plain value (unwrapped, §5.5); `x?` in a function with a plain result; `let INF = ...`; `i64::MAX`, `f64::INFINITY`, `u8::MAX`; `|&a, &b|` closure parameters; `x ^ 2` on floats (a power); character ranges (`'a'..='z'`, `c in '0'..='9'`); `[][]` (an empty list); `xs |> map(f)` (`xs.map(f)`); `find_map`, `filter_map`, `to_digit`, `is_ascii`, `x.str()`, `.unwrap()` on a value that is always there, `cmp(...).reverse()`, `x[i].cmp(x[j])` on single tensor elements; a list of strings returned where text is declared (joined); sizes computed in a tensor type (`tensor<i64, [n + 1, cap as usize + 1]>.zeros()`, also `::zeros()`); mixed tensor dtypes (§6.5); NumPy index lists (`probs[rows, targets]`) and new axes (`a[:, None]`); `x //= k`, `x >>= 1`, `x |= m` and the other bitwise assignments; `[]i64`, `[][]str`, `[]str::new()`, `map<str, i64>{}`, `set<T>()` (empty collections); `null`, `nil`, `undefined` (`None`); `m[k] == None` (whether `m` has no `k`, as `m.get(k) == None`); `for i, x in xs.enumerate()`; `assert x == y, "why"` as a statement (also `requires c` inside a body); `f64(i)`, `i64("42")` (conversions); `let r = xs.reverse()` and `a + b.sort()` (a used result means the reversed or sorted copy; as a statement they change the list); `reduce(init, f)`, `names.find("x")` (a position), `retain`, `split_at`, `transpose` of a list of lists, `argsort`; `-f32.inf`, `i64.min_value()`; `Some(var p)`; and Markdown code fences pasted with the code. Re-checked against those programs with the current `zyth`, 84.2% now pass (70.7% before), and none that passed fails. One of these was a silent error before: `x //= 10` read as `x` and a comment, so loops never ended. Strict mode turns the ones that hide a decision (a changed `let`, an unwrap, a dtype or number conversion) back into errors. Added 2026-10-29 from the first tries the eval re-judges (`eval/failures.py`): a loop variable changed in place (`for g in groups { g.sort() }`, also `for (k, v) in m.items()` and `.enumerate()`) is stored back into the collection after each pass, as in Python (strict mode: E0205); a priority queue orders structs and enums by their own `cmp`/`partial_cmp`/`compare` (Rust's `impl Ord`); `x?` in a function whose result is a struct or enum of the same file unwraps, and `x?` on a value that is always there is the value; `"row {}: {}".format(i, x)` and `"{1} {0}".format(a, b)`; `chars.collect::<String>()` and `String::from_iter(chars)` (the joined text); record types `{ title: str, day: str }` (also for `ai` answers); `mut` closure parameters; `var` as a name where no declaration can follow (`let var = x.var()`); `&mut x` on a `let` (a `var`); `xs.reverse().join("")` (the reversed copy) and `xs.sorted_by(f)` as a statement (sorts `xs`, except as a block's last expression, its value); a method taking `self` that changes it (directly or through a `mut self` method of the same type) is a `mut self` method; uppercase names in `let` patterns bind (`let (T, D) = x.shape()`); `decide` answers written as text (`"spam" => ...`; `"not spam"` is the answer `not_spam`); `if let v = optional` and `while let v = optional` (Swift) bind the value inside and skip `None`; prose around Markdown fences when the text as written does not parse; the tensor methods as functions of their first argument (`tensor.argsort(x, descending: true)`); `tensor<f32, [n, n]>.from_fn(|i, j| ...)` and `.fill(v)`; `pad_start`/`padStart`/`rjust`, `pad_end`/`ljust`, `is_alnum`, `as_str`. `a ?? b + c` is now an error in every mode (E0221, with an edit to `(a ?? b) + c`): all 11 eval programs with it meant the default first. Added 2026-10-30: `f(pos)` where `f` declares `pos: mut i64` lends `pos` (§9.1), and `~x` (bitwise not); blocks given by indentation alone (Python, with or without the trailing `:`) get their braces when the file then parses, and otherwise one error with an edit; two statements on one line whose newline was lost (`n += x     seen += 1`) are two statements; `inf` (as `INF`); `s.trim_start_matches(p)`, `trim_end_matches`, `trim_matches`, `replace_first`, `replacen`, `c.ord()`; `xs.concat(ys)` (a new list), `xs.resize(n, v)`; `.is_none()` / `.is_some()` on a value that is always there (false / true); `for i in a..b step k` (`(a..b).step_by(k)`).

### 16.3 Documentation for Language Models

- `zyth.io/llms.txt` and `zyth docs --llm` provide a compact language reference sized for an LLM context window.
- `zyth doc --deps` and the MCP tool `package_docs` give the API of the project's packages at the versions it has installed; `zyth vet` and `check_package` tell whether a package exists before an agent imports it (Section 15.4).
- The reference is generated from the same source as the human docs, so it never goes stale.
- `zyth explain E0401` prints the explanation offline.

### 16.4 Why This Matters

A deterministic formatter keeps AI diffs clean, and compile-time shape checking means AI-generated tensor code is verified before it runs. The pitch: **the language where AI-written ML code is checked before it trains.**

### 16.5 Measuring How Well AI Writes Zyth

`eval/run.py` asks AI models through OpenRouter to write 62 small programs (text, lists, maps, parsing, graphs, tensors, and decision points), checks and runs each one against expected results, and feeds the problems back for up to three more tries; the same tasks in Python are the comparison. `eval/compare.py` summarizes runs. Language changes are kept when they raise the share of programs right on the first try, and the results are in `docs/STATUS.md`.

### 16.6 MCP Servers and Clients in Zyth (`zyth.mcp`)

A Zyth program becomes an MCP server by marking functions; `zyth new mcp-server <dir>` starts one.

```zyth
use zyth.mcp

/// Add two numbers.
/// - a: the first number
@tool
fn add(a: i64, b: i64) -> i64 uses nothing = a + b

/// A note by id.
@resource("notes://{id}")
fn note(id: str) -> str = notes[id] ?? "no such note"

/// Ask for a code review.
@prompt
fn review(code: str) -> str = "Review this code:\n{code}"

mcp.serve(name: "notes")          // stdio; `http: 8080` for streamable HTTP at /mcp
```

- **Tools.** A `@tool` function's parameter types are its input schema (structs are objects, enums without fields are string choices, `?T` and parameters with defaults are optional, defaults are listed), its doc comment is the description (lines `- name: text` describe parameters), and its result type is the output schema (`structuredContent`; anything but an object is wrapped as `{"result": ...}`). An `Err` result or a run-time error is a tool error (`isError`) the model sees. `@tool("name", title:, read_only:, destructive:, idempotent:, open_world:)` sets the name and hints; without them the hints come from the `uses` clause (no `files`, `process`, `ffi` or `net`: read-only; `net` or `ai`: open world).
- **Resources and prompts.** `@resource("uri")` functions are resources; `{name}` in the URI is a template variable passed as the parameter of that name. `@prompt` functions are prompts: text is one user message; a list of `{role:, text:}` records is several.
- **Permissions.** The project's `[permissions]` are sent in the `initialize` result (`_meta["zyth/permissions"]`, `null` without a table) and each tool lists its `uses` clause (`_meta["zyth/uses"]`), so a client sees what the server may touch before it calls anything; the limits are enforced as for any program (Section 15).
- **Output.** Standard output carries the protocol, so in a program with `@tool`, `@resource` or `@prompt` functions `print` writes to standard error.
- **Protocol.** JSON-RPC 2.0, protocol versions 2025-06-18, 2025-03-26 and 2024-11-05 (the client's when it is one of them). Over HTTP the server answers `POST /mcp` with JSON (no server-sent events), checks the `Origin` header (local origins only), and listens on 127.0.0.1 unless `host:` is given. `requests: n` stops after n messages (tests).
- **One file.** `zyth build main.zyth` makes a single executable that is the server.

**Using other servers.** `use mcp."npx -y @modelcontextprotocol/server-everything" as ev` starts the command (it needs the `subprocess` permission) or, for an `http(s)://` URL, connects to it (`net`), and lists its tools. Tools are called like methods, with named or positional arguments (in schema order): `ev.echo(message: "hi")`; `get_sum` finds a tool named `get-sum`. A result with `structuredContent` comes back as a value (unwrapped from `{"result": ...}`), otherwise as text; a tool error is a run-time error. Also: `ev.tools()`, `ev.call(name, ...)`, `ev.read(uri)`, `ev.resources()`, `ev.prompt(name, ...)`, `ev.server()`, `ev.permissions()`; `mcp.connect(command_or_url)` does the same at run time. The server process ends with the program.

`zyth types gen mcp:ev "<command or URL>"` writes `zyth-types/mcp.ev.zyi` from the server's schemas; with it, the checker checks every call's argument types and reports tools the server does not have (E0102) before the program runs. Without it, calls are checked when they run.

### 16.7 Agents (`zyth.agent`)

An agent is a model that calls tools until it can answer; `zyth new agent <dir>` starts one.

```zyth
use zyth.agent
use mcp."npx -y @modelcontextprotocol/server-filesystem ./docs" as files

/// The weather in a city.
/// - city: the city's name
fn weather(city: str) -> str uses net { ... }

struct Trip { city: str, pack_umbrella: bool }

let planner = agent.new(
    model: "openrouter:openai/gpt-4o-mini",      // or any decider: ai.local(...), ai.scripted([...])
    instructions: "You plan trips.",
    tools: [weather, files],                     // functions, MCP servers, other agents
    max_steps: 10, max_seconds: 60, max_dollars: 0.05,
    approve: [weather],                          // a person allows each call (--approve all|none)
)
let r = planner.run("Plan a day in Lisbon", output: Trip)
print(r.value.pack_umbrella, r.steps.len(), r.dollars)
```

- **Models.** Cloud models with tool calling (OpenRouter, OpenAI-compatible servers such as Ollama and vLLM, Anthropic) get the tools natively, several calls per turn. Any other decider (`ai.local`, Bedrock, `ai.scripted` for tests, `ai.rules`, `ai.human()`) is asked step by step: which tool or `answer` (a choice), then the tool's arguments or the answer as typed JSON, which a local model writes under a grammar so it is always valid.
- **Tools.** A named `fn` is described by its name, doc comment and parameter types, as an MCP tool is (closures are refused: they have no name or description). An MCP server gives all of its tools. Another agent is a tool taking a `task`. `sandbox.tool(...)` is the tool `run_zyth` (below). A tool's error, or a call with wrong arguments, goes back to the model as text (`error: ...`) so it can correct itself.
- **Results.** `run(task, output: T)` returns a record: `text`, `value` (a `T` when `output:` is given, else the text), `ok`, `stopped` (`answer`, `max_steps`, `max_seconds` or `max_dollars`), `steps` (each with `n`, `text`, `calls` (`tool`, `args`, `result`, `error`, `approved`), `by`, `dollars`, `ms`, `final`), `tool_calls`, `dollars`. `ask(task)` returns just the value, and is an error (E0809) if the agent stopped before answering. `stream(task)` returns the steps as they happen.
- **Memory.** An agent keeps its conversation across `run`s; `reset()` forgets it, `history()` shows it.
- **Diary and replay.** With `--diary`, each model turn is written before its tools run (`kind: "agent"`), and what the tools returned after (`agent_tools`). `--replay` takes the model's turns from the file and runs the tools again, so a run can be reproduced without the model; `zyth diary` counts turns, tool calls, failures and cost per agent.
- **Limits.** Each step needs the `ai` effect, and counts against `ai.limits(...)`, `--ai-max-dollars` and `uses ai(dollars:)` as decisions do.

**Agent to agent (A2A).** `planner.serve(port: 9000)` offers the agent over the A2A protocol (JSON-RPC 2.0 over HTTP): the agent card at `/.well-known/agent-card.json`, `SendMessage` (A2A 1.0) and `message/send` (0.3) at `/`, answered with a message (a failed task when the agent stops early); each conversation (`contextId`) has its own history. `agent.remote("http://host:9000")` reads a card and returns an agent with `ask(text)` that keeps the conversation, and that can be another agent's tool. Checked against the official A2A Python SDK 1.2.3 in both directions. Streaming, push notifications and long-running tasks are not supported.

### 16.8 Running Code an AI Wrote (`zyth.sandbox`)

```zyth
use zyth.sandbox
let r = sandbox.run(code, allow: ["fs:read:./data"], seconds: 5, memory_mb: 512)
```

The code is checked first; if it has errors they come back (`r.errors`: `code`, `message`, `line`, `col`, `help`, `fixes`, the same edits `zyth check --json` gives) and nothing runs. Otherwise it runs as a separate `zyth` process in a project whose `[permissions]` are exactly `allow` (each entry must also be allowed to the calling program, so code cannot get more than its caller), with no standard input, an environment cleared to `PATH`, `HOME` and the locale (unless `env` is allowed), output capped at 1 MB, a time limit (default 10 s) and a memory limit (default 1024 MB, on Linux and macOS: the process is stopped when its resident memory passes it). The result is a record: `ok`, `stage` (`check`, `run`, `timeout` or `memory`), `output`, `errors`, `ms`. Run-time errors are read from the error report (code, message, line). A built program finds `zyth` through `ZYTH_BIN` or the `PATH`.

`sandbox.tool(allow:, seconds:, memory_mb:)` is the same as an agent tool named `run_zyth`, whose description tells the model what the code may use; an agent writes a program, sees the checker's errors and fixes, corrects it and runs it again in one loop.

---

## 17. Developer Experience: Debugging, Testing, Profiling & Experiments

### 17.1 Shape-Aware Editor

Through `zyth lsp`, every tensor expression shows its inferred shape inline as you type, and hovering a dimension shows its provenance (Section 6.6).

```zyth
let h = x @ self.w1          // ▸ tensor<f32, [64, 512]> on gpu
let y = relu(h) @ self.w2    // ▸ tensor<f32, [64, 10]> on gpu
```

### 17.2 NaN/Inf Hunting

```bash
$ zyth run --nan-check train.zyth
```

```
NaN detected at step 41,233
  first produced by: log(x)            at model.zyth:22
  input stats: min 0.0  max 1.0  zeros 12 (log(0) = -inf, then -inf * 0 = NaN)
  hint: use log(x.clamp(min: 1e-8)) or log_softmax
```

Off by default (it slows training); also available as `nan_check { ... }` around part of a program.

### 17.3 ML-Aware Testing

```zyth
@test
fn softmax_sums_to_one() {
    let y = softmax(tensor.randn(10))
    assert_close(y.sum(), 1.0, tol: 1e-5)
}

@test
fn attention_gradients_are_correct() {
    gradcheck(attention, inputs: [q, k, v], tol: 1e-3)   // compares autodiff to finite differences
}
```

Standard-library helpers: `assert_close`, `assert_shape`, `gradcheck`, seeded random inputs (tests are reproducible by default).

**Contract tests.** `zyth test` also calls every top-level function that has `requires` or `ensures` with generated inputs (100 by default, `--cases N`; `--seed S`; `--no-contracts` turns them off):

- **Inputs** come from the parameter types: numbers (small values and edge cases such as 0, ±1, ±1000), text, lists, sets, maps, optionals, tuples, tensors of the declared dtype and shape (a symbolic dimension such as `N` has one size across all parameters of a call), structs of these, and enums without fields. Inputs grow over the run. Functions with other parameter types (functions, models, generics) are skipped with the reason.
- **Inputs that break the function's own `requires`** are skipped (an input is retried once with all numbers made non-negative). If more than half are skipped, a note says so.
- **A failure** is a false `ensures`, any runtime error (an index out of range, a division by zero, a `panic`, a `requires` of a function it calls), or a call that does not finish within 2 seconds. The failing input is shrunk (numbers toward 0, shorter text and lists, `None`) while it fails with the same error, and printed as a call with a ready `@test` that repeats it.
- **Effects.** Functions that use the network, files, the GPU, an AI decider, processes or foreign code (§11.4) are not called with made-up inputs; `fails` is allowed.
- `requires`/`ensures` failures name the condition, and `ensures` failures show the result: `` contract violated: `ensures` in `clamp01` is false: `result >= 0.0` with result = -0.301 ``.

### 17.4 Profiler

```bash
$ zyth profile train.zyth --steps 100
```

Reports time per operation (CPU and GPU), host↔device transfers, GPU dispatch counts, and a GPU memory timeline. Output is viewable in the terminal or exported to the Chrome trace format (`--trace out.json`).

The profiler also gives **suggestions**, ranked by estimated time saved:

```
Top suggestions (estimated savings per step):
  1. host↔device copy every step          train.zyth:30   ~41 ms
     `batch.to(gpu)` runs inside the loop on data already on the GPU; remove it
  2. matmul is memory-bound               model.zyth:12   ~18 ms
     inputs are f32; @mixed_precision(compute: bf16) would halve memory traffic
  3. 212 small GPU dispatches per step    model.zyth:20-34
     ops are not being fused because `.item()` at line 27 forces execution; move it after the loop
```

Suggestions are heuristics, labeled as estimates, and never applied automatically.

### 17.5 Built-In Experiment Tracking

Every `zyth run` records a run locally in `.zyth/runs/`: the config, git commit, dependency lockfile hash, hardware, logged metrics, and loss curves.

```zyth
track("loss", loss, step: step)        // one call records a metric
```

```bash
$ zyth runs                    # list runs
$ zyth runs show 15            # metrics, config, charts in the terminal
$ zyth runs compare 12 15      # diff configs and overlay curves
$ zyth runs ui                 # local web dashboard
```

Runs are stored only on your machine. Nothing is uploaded unless you explicitly export it.

**Data versioning:** each run also records a content hash of every dataset file it read (via `zyth.data` and `zyth.fs`), plus the source of remote data (URL or Hub revision). `zyth runs compare` flags when two runs used different data, so "the model got worse" can be traced to a data change instead of a code change. Large files are hashed once and cached by path, size, and modification time.

### 17.6 Toolchain Speed Targets

Fast feedback is a design goal, measured and tracked in CI:

| Operation | Target |
|---|---|
| REPL input → result (type + shape check included) | < 100 ms |
| `zyth check` on a 10,000-line project, incremental | < 1 s |
| Interpreter startup | < 50 ms |

The type checker is incremental (query-based, e.g. the `salsa` crate), so editing one function only rechecks what depends on it.

---

## 18. Stability & Editions

Mojo spent three years breaking users' code before reaching 1.0. Zyth makes its stability promise explicit from the start.

- **Before 1.0:** breaking changes are allowed, but each release ships migration notes and `zyth migrate` applies the changes automatically where possible.
- **From 1.0:** Rust-style **editions**. Each project declares an edition (`edition = "2028"` in `zyth.toml`). Code written for an old edition keeps compiling forever, and editions interoperate, so a 2028-edition project can use a 2030-edition library.
- **Standard library:** semantic versioning. Stable APIs are marked; experimental APIs live under `zyth.experimental` and are excluded from the promise.
- **Model files:** the `.zyth.model` format is versioned, and newer Zyth versions always read older files.

---

## 19. Standard Library

Everything below is ordinary Zyth code. Libraries can evolve faster than the language.

| Module | Contents |
|---|---|
| prelude | tensor, scalars, `print`, `inspect`, `grad`, common NN functions |
| `zyth.nn` | Linear, Conv1d/2d, Embedding, LayerNorm, RMSNorm, Dropout, MultiHeadAttention, TransformerBlock, Transformer, LSTM/GRU |
| `zyth.nn.arch` | GPT-2, Llama, BERT, ViT, Whisper |
| `zyth.optim` | SGD, Adam, AdamW, Lion; LR schedules; gradient clipping; quantization (int8/int4) |
| `zyth.data` | `stream`, batching, shuffling, `text` tokenizers (BPE, SentencePiece), CSV/JSONL/Parquet readers |
| `zyth.train` | `fit` loop, checkpointing, mixed precision, LoRA, **SFT, DPO, GRPO**, distillation |
| `zyth.media` | `image`, `audio`, `video` types; decode/encode; resize, crop, spectrogram, mel features. Implemented: `media.image(path)` (PNG, JPEG) with `width`, `height`, `channels`, `resize`, `crop`, `gray`, `rgb`, `flip`, `pixel`, `tensor()` (`[H, W, C]` or `layout: "chw"`, 0..1), `save` (PNG); `media.image_from(t)`; `media.audio(path)` (WAV) with `rate`, `channels`, `frames`, `duration`, `mono`, `resample`, `slice`, `tensor()` (`[C, N]`), `save` (16-bit WAV); `media.audio_from(t, rate:)`; `media.spectrogram`, `media.mel`, `media.log_mel` (Whisper's features); `media.video_frames(path, fps:, width:, height:)` through ffmpeg |
| `zyth.ai` | Text generation (sampling, KV cache), embeddings, vector index, RAG helpers, eval harness |
| `zyth.net` | HTTP client and server, WebSocket, model serving (implemented; Section 19.2) |
| `zyth.dist` | Processes that train together on one or many machines (Section 19.4): `rank`, `world`, `all_reduce`, `average`, `broadcast`, `barrier`, `part`, `share_dir`; federated averaging (`dist.fedavg(models, weights:)`) |
| `zyth.check` | What the checker tells libraries (Section 11.5): `@lint(calls:)` library warnings given a `Call`, `@fields_used` parameters |
| `zyth.inject` | Components wired together (Section 19.5): `@component` on functions (they provide what they return) and structs (their fields are filled in), `container(replace:)`, `.get(Type)`, `.call(handler)`, `.request()`, `.explain()` |
| `zyth.web` | DOM, canvas, camera, microphone (web target) |
| `zyth.math` | linalg, stats, random, fft |
| `zyth.fs`, `zyth.json`, `zyth.time` | Files, JSON, time |

### 19.1 Training Methods as Library Calls

```zyth
use zyth.train.{sft, dpo, lora}

let base  = hub.load("meta-llama/Llama-3.2-1B")
let tuned = sft(base,
    data: jsonl("instructions.jsonl", prompt: "instruction", response: "output"),
    adapter: lora(rank: 16, targets: [q_proj, v_proj]),
    lr: 2e-5, epochs: 3)

let aligned = dpo(tuned, reference: base,
    data: jsonl("prefs.jsonl", prompt: "q", chosen: "good", rejected: "bad"),
    beta: 0.1)

aligned.save("aligned.zyth.model")
```

### 19.2 Serving

```zyth
use zyth.net.{server}

let model = Classifier.load("classifier.zyth.model")?

server(port: 8080) { app ->
    app.get("/health") { _ -> { status: "ok" } }
    app.post("/predict") { req ->
        let x = tensor.from(req.json()["pixels"])
        { class: model.forward(x).argmax() }
    }
}
```

`zyth.net.server` has no authentication by default. Add `app.use(auth.bearer(...))` or put it behind a gateway before exposing it.

What `zyth.net` has (implemented): `net.get/post/put/delete(url, headers:, body:, json:, timeout:)` and `net.request(method, url, ...)` give a response (`status`, `ok`, `headers`, `url`; `text()`, `json()`, `bytes()`), following redirects; `server(port:, host: "127.0.0.1", requests:, background:) { app -> ... }` with `app.get/post/put/delete/patch/any(path) { req -> ... }` (`:name` and `*rest` path parameters; `req.method`, `path`, `query`, `params`, `headers`, `body`, `json()`, `form()`, `header(name)`), `app.use(auth.bearer(token))` or `app.use { req -> ... }` (a non-`()` result answers instead of the route), `app.ws(path) { ws -> ... }`, and `net.response(status:, body:, json:, headers:, content_type:)`; `net.websocket(url)` (`send`, `recv() -> ?str`, `close`). Each connection runs on its own thread with its own interpreter, like `go`. A handler's map, list or number is sent as JSON, text as `text/plain`, `()` as 204; an error is a 500 with `{"error": message}`. The server listens on 127.0.0.1 unless `host:` says otherwise, and needs the `net:listen:<port>` permission (or `net`) when `[permissions]` is set. Not yet: HTTP/2, TLS for the server (put it behind a proxy), streaming bodies.

### 19.4 Training on many processes and machines, and recovering from failures

`zyth launch` runs a program as one process per GPU (or `--nproc N` processes) on each machine, and starts them all again when one fails:

```
$ zyth launch train.zyth                                   # one machine, one process per GPU
$ zyth launch --nnodes 2 --node-rank 0 --master 10.0.0.5:29500 --max-restarts 3 train.zyth   # machine 0
$ zyth launch --nnodes 2 --node-rank 1 --master 10.0.0.5:29500 --max-restarts 3 train.zyth   # machine 1
$ srun --ntasks-per-node=1 zyth launch train.zyth          # Slurm: machines and machine 0 from its environment
```

- **Inside the program** nothing changes: `fit(model, batches, loss, checkpoint: "ck", resume: true)` sees `dist.world() > 1` and trains data-parallel across the processes. Each process takes its share of every batch (`dist.part`, cut along the first dimension, weighted by its rows), the gradients and the loss are averaged in one exchange (`dist.average`), and every process takes the same step, so the losses are those of one process training on the whole batches (checked to 5 decimal places for 1, 2, 3 and 4 processes, on one machine, two Slurm nodes and two Kubernetes pods). The weights start as rank 0's. Only rank 0 logs and writes checkpoints.
- **Collectives:** `dist.rank()`, `dist.world()`, `dist.local_rank()`, `dist.round()` (how many restarts so far), `dist.all_reduce(t, op: "sum" | "mean")`, `dist.average(tensors, weight:)`, `dist.broadcast(x, from: 0)` (a tensor, a list of tensors, a string or a number), `dist.barrier()`, `dist.backend()` (`"nccl"`, `"tcp"`, `"none"`). GPU tensors are summed by NCCL when every process has its own NVIDIA GPU (the launcher gives each process one through `CUDA_VISIBLE_DEVICES`; processes sharing a GPU use TCP); everything else goes over a ring of TCP connections (reduce-scatter then all-gather, the same result on every process). Without `zyth launch` a program is rank 0 of 1 and every collective returns its input.
- **Rendezvous:** the launcher of machine 0 runs a small key-value store on `--master`'s port; the launchers and the processes meet through it (addresses, the NCCL id, heartbeats). In each pair of neighbours in the ring the higher rank connects, so the machines given the highest ranks need not accept connections (machines behind NAT); `ZYTH_DIST_ADDR` sets the address a process gives the others.
- **Slurm and Kubernetes:** under Slurm the machine count, this machine's number and machine 0 (the first host of `SLURM_JOB_NODELIST`) come from Slurm's environment; in a Kubernetes Indexed Job the machine number is `JOB_COMPLETION_INDEX` and `--master` names pod 0 through a headless service (`train-0.svc:29500`). torchrun's `MASTER_ADDR`, `MASTER_PORT`, `NNODES` and `NODE_RANK` are read too.
- **Recovery:** a run goes in rounds. When a process exits with an error or is killed, or a machine sends no heartbeat for `--heartbeat` seconds (30 by default), every launcher stops its processes and, while `--max-restarts` allows, all start again in a new round; `fit(..., resume: true)` continues from the latest checkpoint. A machine that was lost rejoins by starting its launcher again (Slurm's requeue, a Job's replacement pod). A machine without rank 0's checkpoint (local disks) gets a copy over the ring (`dist.share_dir`). Not handled: the loss of machine 0 itself (its store holds the run), and changing the number of machines during a run.
- **Checkpoints that survive a crash:** `fit`, `fit_sharded` and `fit_pipeline` take `checkpoint:`, `every:`, `resume:` and `keep:` (2). A checkpoint is written into `ck/.step-N.tmp`, synced to disk, renamed to `ck/step-N` and then named in `ck/latest` (written and renamed the same way), so a crash at any moment leaves the previous checkpoint or the new one; it holds the weights, the optimizer state, the step and the random generator's state, so a resumed run gives the same losses as one that never stopped.

### 19.5 Components (`zyth.inject`)

A small container, written as a library on user-defined annotations (Section 11.3). Components are matched by type: the last name with its type arguments (`Pool<Db>` and `Pool<Cache>` are different; `app.Settings` and `Settings` are the same). The wiring is checked by `zyth check` before the program runs.

```zyth
use zyth.inject.{component, container}

@component
fn settings() -> Settings = Settings { url: "sqlite::memory:" }

@component
struct Store { settings: Settings }            // fields of component types are filled in

@component(scope: "request")
fn session(store: Store) -> Session = ...      // a new one in every request

fn handler(store: Store, s: Session) -> str = ...

let app = container()
app.get(Store)                                  // made once for the app, in dependency order
app.request().call(handler)                     // the handler's parameters filled in
container(replace: { Mailer: FakeMailer() })    // tests replace pieces
container(test: true)                           // or use the components marked scope: "test"
```

- Before running: a need no component provides (E0820), a type several components provide without one marked `primary: true` (E0821), a cycle with its chain (E0822), an app component that needs a request one (E0823). The same checks apply where the program asks: `app.get(T)` and `app.call(handler)` (each of the handler's parameters), and `app.get(T)` has type `T`, so `app.get(Settings).port` is caught too. A container made with `replace:` may provide more than the components do, so for it a missing component waits until the program runs. `zyth explain components app.zyth` prints who gets what from where.
- Scopes: `"app"` (the default: one per container), `"request"` (one per `app.request()`) and `"test"` (used instead of the others for its type, only in `container(test: true)`; ignored by other containers). A request component asked of the app container is an error. Fields and parameters of other types need defaults.
- Left out on purpose: XML, profiles, proxies and aspects, post-processors, scanning of packages, automatic configuration. Components come only from the program and the modules it imports.

### 19.3 Graphics: Zyth draws, a library builds scenes

**Requirement (2026-10-29).** 3D graphics is two jobs, kept apart the way the web keeps them apart (the browser provides WebGPU; three.js, a library, builds scenes on it):

1. **Talking to the graphics card: in Zyth.** Only the runtime can reach the GPU (a program in a browser cannot reach it at all), and the code differs on macOS (Metal), Windows (DX12 or Vulkan), Linux (Vulkan) and browsers (WebGPU). Zyth provides it as a small built-in module next to the tensor GPU code it extends (`wgpu` natively, WebGPU in browsers):
   - a surface to draw on: a window in native programs, a `<canvas>` in `zyth build --target web` pages, or an off-screen texture (for tests, servers and image output);
   - GPU buffers (vertex, index, uniform) and textures, where **a tensor can be a buffer without a copy**: the GPU computes the numbers and draws them in place;
   - render pipelines with shaders (WGSL first; vertex and fragment shaders written in Zyth like `kernel` functions are a later step), depth testing and blending;
   - draw calls, render passes, and presenting a frame; a frame loop with timing;
   - input events from the surface: mouse, keyboard, touch, resize.
   
   It is kept small and stable: what the hardware does, no opinions about how scenes look. It follows Zyth's rules: errors before running where the types allow (buffer layouts against pipeline inputs), the `gpu` permission (Section 15), and the same program on every platform.
2. **The 3D itself: a separate library.** Cameras, meshes and shapes, lighting and materials, orbit and mouse controls, scene graphs, and helpers such as plotting a graph in 3D are ordinary Zyth code in a package on top of item 1, not part of the language or the standard library. It can change and be released on its own schedule.

Why the split: Zyth's purpose is AI and the 3D layer is a side line that must not grow the core; drawing next to the AI math lets one GPU compute and draw without moving data back and forth; and a library can improve quickly where the language must stay stable.

**Implemented (2026-10-30): `zyth.gfx` and the `scene3d` package.**

```zyth
use zyth.gfx

let s = gfx.window(title: "points", width: 640, height: 640)   // or gfx.surface(width:, height:) off screen
let shader = gfx.shader(r"""
@vertex fn vs(@location(0) p: vec2f) -> @builtin(position) vec4f { return vec4f(p, 0.0, 1.0); }
@fragment fn fs() -> @location(0) vec4f { return vec4f(1.0, 0.6, 0.2, 1.0); }
""")
let pipe = s.pipeline(shader, buffers: [["f32x2"]], topology: "points")
let pos = tensor.randn(4096, 2) * 0.3 on gpu          // drawn where it is: no copy
s.run { f -> f.draw(pipe, vertices: [pos]) }            // every frame until the window closes
```

- **Surfaces:** `gfx.surface(width:, height:)` (off screen), `gfx.window(title:, width:, height:, vsync:)` (a native window; a `<canvas>` on the page in `--target web` builds). `s.frame(clear:) { f -> ... }` draws one frame; `s.run(frames:, clear:) { f -> ... }` draws frames until the window closes, `f.stop()`, or `frames:`; in the browser each frame is shown at the next animation frame, the page running in between. `f.time`, `f.dt`, `f.index`, `f.width`, `f.height`, `f.aspect`, `f.events` (records: `mouse_move`, `mouse_down`/`mouse_up` with `button`, `wheel` with `dx`/`dy`, `key_down`/`key_up` with `key`, `resize`, `close`). `s.read()` gives the last frame as `tensor<u8, [H, W, 4]>`, `s.save("f.png")` writes it.
- **Resources:** `gfx.shader(wgsl)` (errors carry the WGSL compiler's message, E0420), `s.pipeline(shader, vertex: "vs", fragment: "fs", buffers: [[formats...], {attrs: [...], instance: true}], topology: "triangles" | "triangle_strip" | "lines" | "line_strip" | "points", blend: "none" | "alpha" | "add", depth:, cull:)`, `gfx.buffer(data, type: "f32" | "u32" | "u16" | "i32")` with `b.write(data, offset:)`, `gfx.texture(image or [H, W, 4] tensor)`, `gfx.sampler(filter:, wrap:)`. `f.draw(pipe, vertices: [...], indices:, count:, instances:, bind: [...])` binds group 0 in order (buffers, tensors as storage, textures, samplers). Raw strings `r"""..."""` hold shader code as written.
- **Text and multisampling (2026-10-30):** `f.text("score {n}", x: 10, y: 10, size: 24, color: [1, 1, 1, 1], font: gfx.font("f.ttf"))` draws text with its top left at (x, y) pixels, over what the frame drew before it; `gfx.text_size(text, size:, font:)` gives its width and height. Glyphs are rasterized on the CPU into an atlas (whole-pixel positions), so text is the same on every system; the built-in font is a subset of DejaVu Sans (Latin, Greek, punctuation, arrows, math symbols, box drawing). `gfx.surface(..., samples: 4)` and `gfx.window(..., samples: 4)` draw with 4 samples per pixel (multisampling; 1 or 4, which every WebGPU device supports), resolved into the surface's image.
- **Tensors without copies:** buffers of `gpu` tensors are made usable as vertex, index and storage buffers, so the GPU draws what it computed. Checked: on Windows with an RTX 3090 (Vulkan) the drawn tensors are not copied (`ZYTH_GPU_DEBUG=1` reports any copy); where tensors stay on the CPU (a GPU that draws but cannot compute) they are copied once per frame.
- **Where it runs:** the WebGPU builds (macOS and Windows by default, `zyth-webgpu` on Linux, to which `zyth run` hands programs that use `zyth.gfx`) and browsers. Checked on 2026-10-30: the same triangle gives the same pixels on Linux (GL on D3D12, WSL), Windows (Vulkan, RTX 3090), macOS (Metal, M4) and headless Chromium (SwiftShader WebGPU, off screen and on a canvas); a window on Windows and macOS showed and read back the frames (`examples/gfx/points.zyth`).
- **The 3D library** (`packages/scene3d`, Zyth code on `zyth.gfx`): matrices (`perspective`, `look_at`, `translate`, `rotate_y`, ...), meshes (`cube`, `sphere`, `plane`), lit scenes (`Scene.new(surface)`, `scene.add(mesh, color:, at:, size:)`, `scene.points(positions, colors)` for 3D plots of tensors), and `Orbit` (drag and wheel). Its demo renders the same image on Linux and Windows (0 pixels differ).
- Text is put in display order by the Unicode bidirectional algorithm and shaped (rustybuzz, a port of HarfBuzz): right-to-left lines, joined and combining scripts, ligatures and kerning come out as the font defines them, when the font has the glyphs (the built-in font covers Latin; give another with `font: gfx.font(path)`).
- Render to texture: a surface can be bound like a texture (`f.draw(p, bind: [other_surface, gfx.sampler()])`); the draw samples that surface's last frame. A surface cannot be drawn from while it is drawn into (E0420).
- **Android:** the `aarch64-linux-android` build with `--features webgpu` draws off-screen surfaces and computes on the phone's GPU (Vulkan); it has no windows (`gfx.window` fails with E0420 and says to use surfaces or a web build), since an Android window belongs to an app's Activity. Pages built with `--target web` draw in Chrome on Android (WebGPU).
- Not done yet: compute passes written in Zyth for drawing (`kernel fn` writes into a tensor that is then drawn, which works today), and Android and iOS apps with a window.

---

## 20. The Flagship Demos

### 20.1 Train a GPT in ~12 Lines

```zyth
use zyth.nn.Transformer
use zyth.data.text
use zyth.optim.AdamW

let tok   = text.tokenizer("gpt2")
let data  = text.stream("tinyshakespeare.txt", tokenizer: tok, context: 256).batch(32)
let model = Transformer(layers: 6, heads: 6, dim: 384, vocab: tok.vocab_size, context: 256) on gpu
let opt   = AdamW(model.params, lr: 3e-4)

for (step, batch) in data.take(5000).enumerate() {
    let loss = model(batch.x) |> cross_entropy(batch.y)
    opt.step(grad(loss, wrt: model.params))
    if step % 500 == 0 { print("step {step}: loss {loss:.3f}") }
}
model.save("tiny-gpt.zyth.model")
```

Twelve lines of logic plus three imports. This trains a GPT with ~10M parameters in its transformer blocks (~49M in total, most of them in the 50,257-token embedding and output layers); how long it takes depends on the GPU.

### 20.2 Run It in the Browser

```zyth
@target(web)
use zyth.web.{dom}
use zyth.nn.Transformer
use zyth.data.text

let tok   = text.tokenizer("gpt2")
let model = Transformer.load("tiny-gpt.zyth.model")?      // runs on WebGPU

dom.on("#go", "click") { _ ->
    let prompt = dom.value("#prompt")
    dom.set_text("#output", model.generate(tok.encode(prompt), max_tokens: 200) |> tok.decode)
}
```

```bash
$ zyth build --target web chat.zyth && npx serve dist
```

Training on a server and running in the browser use the same model definition and the same file format.

---

## 21. Tooling & CLI

| Command | Purpose |
|---|---|
| `zyth run file.zyth` | Run a script (records an experiment run — Section 17.5) |
| `zyth run --nan-check` | Stop at the first operation that produces NaN/Inf |
| `zyth repl` | Interactive REPL |
| `zyth build [--release] [--target t]` | Build a native binary, web bundle, Python wheel, or npm package |
| `zyth check [--json]` | Type and shape check; `--json` for tools and AI agents |
| `zyth explain <code>` | Explain an error code |
| `zyth test` / `zyth bench` | Run `@test` / `@bench` functions |
| `zyth profile` | Per-op CPU/GPU timing and memory timeline |
| `zyth runs [show\|compare\|ui]` | Browse and compare experiment runs |
| `zyth fmt` / `zyth lint` | Format / lint (lint reports every `danger` and `ffi` site) |
| `zyth add git:<repo>` / `npm:<pkg>` / `py:<pkg>` | Add a dependency (checks the name exists, shows what it can do and the permissions it needs) |
| `zyth install` / `zyth update` | Install what zyth.lock pins / move to newer versions, showing new permissions |
| `zyth vet <pkg>` | Does a package exist; is it new or rarely used |
| `zyth api` / `zyth bump` / `zyth publish` | A package's API; the version its changes need; tag a release |
| `zyth audit` | Check dependencies against vulnerability advisories |
| `zyth types status` | Show type-definition coverage for every Python/npm dependency |
| `zyth migrate` | Apply automatic fixes for breaking changes / edition upgrades |
| `zyth share` | Publish a web build to `<name>.zyth.io` |
| `zyth notebook` | Local notebook server |
| `zyth lsp` | Language server for editors (inline shape hints) |
| `zyth mcp` | MCP server for AI agents |
| `zyth doc [--llm]` | Generate docs from `///` comments; `--llm` for a compact LLM reference |

Error messages are a feature: every error has a code, points at the source, explains where shapes came from, and suggests a fix (Sections 6.6 and 16.1).

### 21.1 `zyth share`

```bash
$ zyth login
$ zyth share chat.zyth
  built web target (2.1 MB + model 38 MB)
  published: https://yourname-chat.zyth.io  (unlisted — add --public to list it)
```

- Shares are static web builds: the model runs in the visitor's browser, so there is no server to secure or pay for per request.
- Unlisted by default. Size limits and abuse policy apply (to be defined before launch).

---

## 22. Implementation Roadmap

Milestones are ordered so each one produces something demoable. Durations are rough estimates for a solo developer and will change. The status of each milestone, and which parts of the implementation it lives in, is in `docs/STATUS.md`; decisions on what was dropped or changed are in `docs/DECISIONS.md`.

| Milestone | Deliverable | Pillar unlocked | Rough effort |
|---|---|---|---|
| **M1 — Core interpreter** | Lexer, parser, type checker (no tensors), tree-walking interpreter, REPL, `zyth run`, `zyth check --json`, incremental checking, `go`/`scope`/channels | — | 2–3 months |
| **M2 — Tensors + shape checking** | CPU tensor runtime, static/symbolic/runtime dims, E04xx diagnostics with provenance, LSP inline shape hints | Headline feature | 2–3 months |
| **M3 — Autodiff + `zyth.nn`** | Tape autodiff, Linear/Transformer, AdamW, MNIST and tiny-GPT on CPU, `assert_close`, `gradcheck`, `--nan-check` | 12-line GPT (CPU) | 2–3 months |
| **M4 — Browser** | Interpreter + runtime compiled to Wasm; web playground; `zyth build --target web`; browser model cache and variant selection; `zyth share` | Runs in browsers | 1–2 months |
| **M5 — GPU** | `wgpu` backend for core ops, native and WebGPU; lazy graph capture and element-wise fusion; `zyth profile` with suggestions | GPT on GPU, in browser | 3–4 months |
| **M6 — Single binary + C FFI** | `zyth build` produces one executable; `extern "C"`; `zyth mcp` | Native programs | 1–2 months |
| **M7 — Python both ways + model I/O** | PyO3 bridge, managed Python environments (uv), `.zyi` type definitions + first-party numpy/pandas/torch definitions, `--target python` wheels, safetensors/GGUF/ONNX import and export, `hub.load` | Python libraries | 2–3 months |
| **M8 — npm both ways + permissions** | Pure-JS packages (web + embedded QuickJS), TypeScript declarations → types, `zyth-types` registry, `--target npm`, permission system, lockfile, `zyth audit` | npm libraries | 3–4 months |
| **M9 — Training library + experiments** | `zyth.train`: checkpointing, LoRA, SFT, DPO, GRPO; experiment tracking (`zyth runs`) with dataset hashing | — | 2–3 months |
| **M10 — AOT compiler** | LLVM or Cranelift backend replacing bytecode in native builds | Faster native code | 4–6 months |
| **M11 — Mobile** | iOS and Android targets via `wgpu` (Metal / Vulkan) | On-device AI on phones | 2–3 months |
| **M12 — NPU inference** | `npu` device for inference: WebNN in browsers (once enabled by default), Core ML and Windows ML natively | Laptop/phone NPUs | 2–3 months |
| **M13 — Node compatibility** | Node built-ins (`fs`, `net`, `http`, `Buffer`, `process`, `stream`) for the native target, so packages like Express run | Full npm (native) | 4–6 months for the common APIs |

Realistic total to all four pillars (M1–M8): roughly 1.5–2 years solo. Speed claims wait until there are benchmarks against PyTorch. The stability policy (Section 18) applies from the first public release.

---

## 23. Out of Scope for v0.1

Kept in `ZYTH-STRATEGY.md`; may return in v2 if there is demand.

| Feature | Status |
|---|---|
| ZythScript transpiler layer | Cut — Zyth's syntax already fills that role |
| Separate `matrix`/`vector`/`graph` types | Cut — aliases of `tensor` |
| `chain`, `mesh`, `events` keywords | Cut — library patterns on channels and `zyth.net` |
| `pretrain`/`sft`/`align`/`serve`/`rag`/`agent`/`data` keywords | Moved to the standard library |
| Effect system | Partly done: optional `uses` clauses on functions, checked (Section 11.4), and package permissions checked against the effects of package code (Section 15.3); anything beyond that is deferred to v2 |
| Fully inferred lifetimes | Done (Section 9.1): inferred from function bodies; annotations only on signatures without a body |
| GPU kernel generation from arbitrary loops / general kernel fusion | Deferred to v2 (v0.1 fuses element-wise chains via lazy graph capture — Section 3; GPU code is written by hand with `kernel fn`, Section 6.10) |
| NPU training | Not planned; NPUs are inference-only (M12) |
| CUDA/ROCm native code generation | CUDA C generated for `kernel fn` (Section 6.10); ROCm deferred |
| npm packages needing Node.js built-ins (native target) | Done: the Node compatibility layer (M13), or the computer's own Node.js (Section 14.3) |
| `federation` as a language construct | Cut (2026-10-25): federated learning becomes a `zyth.dist` library (docs/DECISIONS.md, after the roadmap) |
| Formal proofs | Cut — contracts are runtime checks |
| Bundled generative models (image, video, music) | Use `hub.load` with open models |

---

## 24. Decisions and Open Questions

Decided in October 2026, each after looking at what comparable projects do:

1. **Braces, not indentation.** Indentation errors are one of the three error classes LlmFix (arXiv 2409.00676) had to repair in LLM-written Python, and models handle code without layout as well as with it (arXiv 2508.13666). Braces keep a program's structure in its tokens. Revisit only if user tests show a real problem.
2. **Number defaults:** integer literals are `i64`, float literals `f64`, tensor elements `f32` (Section 5.1). Mojo, Julia and Python use 64-bit floats for scalars; PyTorch and JAX use 32-bit floats for tensors to save memory and time. This is what the interpreter already did.
3. **Embedded JS engine: QuickJS** (through `rquickjs`), already used for npm. Boa passes about 90-96% of test262 against QuickJS's about 97% of the tests it runs, and Boa has had only patch releases since 0.21 (October 2025).
4. **Model files: plain safetensors.** A `.zyth.model` is a safetensors file whose Zyth configuration is stored as strings in the format's reserved `__metadata__` field (the same place Stability's ModelSpec and Kohya put theirs), so any safetensors reader opens it. No extra Zyth header.
5. **Packages: Git URLs first, a registry later.** As in Go and Gleam's Git dependencies, a package is a Git URL and a tag; the lockfile records the commit and a content hash. A hosted index (a mirror of those repositories, like Go's module proxy) comes when there are enough packages to need search. Built in October 2026 (Section 15.5), with Go's minimal version selection rather than a solver: the build is decided by what was asked for, needs no lockfile to be reproducible, and never moves to a new release by itself, which also keeps freshly published (possibly malicious) versions out.
6. **License (changed 2026-11-12):** the parts programs and tools build on are open source under Apache-2.0 OR MIT, the same as Rust (Apache's patent grant, and MIT for projects that cannot take Apache-2.0): this specification, the standard library, the type definitions, the package tools, editor support and the examples. The compiler, checker, interpreter and runtimes are proprietary to BioDream LLC and free to use (the Zyth Binary License); programs made with them belong to their authors. Mojo splits its parts the same way. `LICENSING.md` lists which part is which.
7. **`zyth-types` support window:** follow Scientific Python SPEC 0. Type definitions cover every minor release of numpy, pandas and torch from the last 24 months, and at least the three newest.
8. **`zyth share` hosting:** object storage without download fees (Cloudflare R2 charges nothing for egress), a 100 MB limit per shared app, and no model files stored: apps fetch models from Hugging Face with `hub.load`, so bandwidth for large models is not ours. Abuse: a report link on every page and takedown by name. Not deployed yet; it needs an account and an operator.
9. **Sandboxed Python by default only in restricted projects** (Section 15.3). Measured: the OS sandbox itself costs nothing beyond running Python in a separate process, but the process costs 12 µs (Apple M4) to 65 µs (Linux under WSL2) per call, 2-4x on calls that pass small tensors; NumPy's own guide warns that IPC overhead from serializing and moving data can be significant. Projects without `[permissions]` already trust their code fully, so they keep in-process Python.

10. **Generators run on their own thread** (October 2026): a `fn` that uses `yield` returns a `stream<T>`, the existing lazy type, so every stream method works on it. A tree-walking interpreter cannot suspend a function in the middle without rewriting it, and Python's and JavaScript's generators are coroutines; here the body runs on a thread that waits at each `yield`. It costs a thread switch per value (fine for data loading and token streams, the uses that motivated it) and keeps generators out of compiled code. In the browser (no threads) they run to their end first.
11. **`kernel fn` is SIMT, one call per thread, CUDA's model** rather than Triton's tile-level one (October 2026): it maps one to one onto CUDA C and WGSL, which every GPU programmer and model has seen, and needs no tile compiler. A tile-level form for FlashAttention-class kernels stays with Part 4. `kernel` is a keyword only before `fn`, so existing names stay valid.
12. **fp8 is a storage type** (October 2026): `fp8_e4m3` and `fp8_e5m2` round values when stored and compute in f32, as FP8 training does between its matrix products (cuBLASLt and Transformer Engine keep scales and accumulate in f32). `i4` is an integer type; block-quantized weights are a value of their own (`quantize`), not a dtype, because their scales travel with them.
13. **A sharded tensor names one split axis** (October 2026): `on gpus(n) shard [B, _]`, as JAX's sharding annotations name mesh axes per tensor axis. One axis covers data parallelism and tensor parallelism of a layer; several at once wait for a use.
14. **Agents are a library, MCP and A2A are protocols in it** (October 2026), following the principle "small core, big library": the only new syntax for agents is `@tool`, `@resource` and `@prompt`, which are annotations. Node.js runs npm packages only when nothing limits them, since the per-package permissions (Section 15) are enforced inside the embedded engine.

Still open:

1. **Name:** `zyth.io` is registered. A search in October 2026 found ZYTH at zyth.net, a terminal AI agent for cloud security teams (also on LinkedIn), and Zythsoft, a software outsourcing company. Both are software, so a conflict is possible. A USPTO wordmark search on 2026-10-25 found three filings containing "zyth", none for software: ZYTH (live, registered, class 4 fuels, serial 99246738), ZYTHBOX (live, class 20 furniture) and REVENGE OF THE ZYTH (dead, class 32 beer); none from zyth.net or Zythsoft, and nothing in class 9 or 42. Not checked: EU, UK, China and WIPO registers, and unregistered use. Decided (2026-10-25): no attorney's clearance before launch; Zyth launches under its name, accepting the risk of a rename or a dispute later.

---

## 25. Decision Points (AI Inside Programs)

A *decision point* is a place where a program stops and lets a decider choose what happens next. The programmer lists every possible outcome; the decider only picks among them (`decide`) or fills in a value of a declared type (`ai`). Who decides is an ordinary value chosen in the program (§25.4): a model in the cloud, a model on the machine, plain code, a person, or a combination. The same program can therefore take different paths under different deciders, and every path is one the programmer wrote.

This section is normative. The design rationale is in `ZYTH-STRATEGY.md` Section 33.

### 25.1 Syntax

```
decide_expr  = "decide" ai_spec "{" arm { sep arm } [ sep ] "}"
arm          = [ "approve" ] label [ STRING ] "=>" expr
label        = IDENT                       // `unsure` is a label with a fixed meaning
ai_expr      = "ai" ai_spec [ "as" type ]
ai_spec      = STRING { "from" unary | "using" unary | "within" "(" limit { "," limit } ")" }   // each clause at most once, any order
limit        = ( "dollars" | "seconds" ) ":" NUMBER
using_expr   = "using" expr block
decider_stmt = "decider" expr
sep          = NEWLINE | ","
```

- **Contextual words.** None of these words is reserved. `decide` and `ai` start a decision only when the next token is a string literal; `using` and `decider` only when the next token is a name or `(`; `from`, `using` and `as` only inside an `ai_spec`; `approve` only when another name follows it in an arm. Everywhere else they are ordinary names (`let ai = 3`, `fn decide()` and the `ai` module all still work).
- **The question** is a string literal and may contain interpolation (`"Is {name} spam?"`).
- **Inside an `ai_spec` before `{`**, the `from` and `using` operands are parsed without struct literals, as in `if` conditions; write `from (Point { x: 1 })` to pass one.
- **Arms.** A `decide` has at least two arms other than `unsure` (error E0010), and each label appears once.
  - A description, if present, is a plain string literal without interpolation. It must not change between runs, so the decider always sees the same text.
  - Each arm's body is one expression, usually on the same line as `=>`; use `{ ... }` for several statements.
- **Types for `ai`.** The type is given by `as T`, or by a `let` annotation directly on it (`let e: Event = ai "..." from text`). Without either, the type is `str`.

```zyth
let action = decide "What should happen to this email?" from email using triage {
    reply "a person wrote it and expects an answer" => draft_reply(email)
    archive "newsletters, receipts and notifications" => archive(email)
    approve delete "spam or scams" => delete(email)
    unsure => flag_for_me(email)
}
let when: ?Date = ai "When is the meeting?" from email
```

### 25.2 Static Rules (Checked Before Running)

| Rule | Error |
|---|---|
| A `decide` with an `approve` arm must have an `unsure` arm, which runs when the person says no | E0810 |
| `unsure` cannot be marked `approve` | E0810 |
| `ai` cannot fill in tensors, models, functions, maps with non-`str` keys, or enums whose variants have fields | E0811 |

**Fillable types:** `str`, `char`, `bool`, every integer and float type, `T[]`, `set<T>`, `?T`, tuples, records, `map<str, T>`, structs whose fields are fillable, and enums whose variants have no fields, all nested freely. A field without a type is filled with any JSON value.

**Types of the expressions:**
- A `decide` has the joined type of its arm bodies: the common type if they agree, otherwise a dynamic value.
- An `ai` has its declared type, or `str`.
- A `using` block has the type of its block, and `decider d` has `()`.

**Moves:** the checker treats the arms like `match` arms. A value moved in one arm counts as moved after the `decide`.

### 25.3 Evaluating a Decision

Each time a `decide` or `ai` is evaluated:

1. **Gather.** The question is evaluated, then the `from` input if present, then the decision's own `using` operand if present. All three are evaluated every time, also when replaying (§25.6).
2. **Render the input** as text, the same way every run:
   - a `str` as it is;
   - no input or `()` as the empty string;
   - any other value as indented JSON in field order.

   The decider receives the rendered text, the original value (§25.4), the question, and the labels and descriptions, or the requested shape.
3. **Check the limits** (§25.7). A reached limit stops with E0804 before the decider is asked.
4. **Get the answer:** from the diary being replayed (§25.6), otherwise from the decider chosen by §25.4.
5. **Validate it.**
   - A `decide` answer must be one of the labels or `unsure`; any other label is E0803.
   - An `ai` answer must convert to the declared type; otherwise E0803. Enum variants match by name, ignoring case. Under `?T`, `null` means `None`.
6. **Ask for approval** if the chosen arm is marked `approve` (§25.8). A "no" turns the answer into `unsure`.
7. **Record** the decision in the diary, if one is being written (§25.6).
8. **Continue.**
   - `decide` evaluates the body of the chosen arm.
   - An `unsure` answer evaluates the `unsure` arm. Without one it is error E0802.
   - `ai` returns the converted value. An unsure fill returns `None` for `?T`, and is E0802 for any other type.

Decisions are numbered from 1 in the order they are evaluated in a run, across all tasks. A decision's *site* is `file:line:column` of the `decide` or `ai`, with the directories of an absolute path removed, so diaries stay valid on other machines.

### 25.4 Choosing the Decider

The decider for a decision is the first of:

1. the decision's own `using d`;
2. the innermost enclosing `using d { ... }` block;
3. `--decider SPEC` on the command line (`ZYTH_DECIDER`);
4. the last `decider d` statement executed.

If none applies, the error is E0800.

- **`using` is dynamic.** Its decider applies to everything evaluated inside the block, including functions it calls and tasks started with `go` inside it. Those tasks keep the decider even after the block ends.
- **`decider d` is a statement, not a declaration.** It takes effect when it runs, for the rest of the run and in every task.
- **`--decider` replaces `decider d`, but not `using`.** So a program can be run under another decider without editing it, while blocks that name their decider keep it.

A decider is a value made by the `ai` module. A plain function `|request| answer` used where a decider is expected acts like `ai.rules(f)`.

| Decider | Behavior |
|---|---|
| `ai.local(model, device:, system:, tokenizer:)` | A language model on this machine: a Hub name, a local folder, a saved `.zyth.model` (needs `tokenizer:`), or a model value. `decide` scores each label by probability instead of generating text, so the answer is always a listed label and has a confidence. `ai` writes JSON restricted to the requested shape. Models are loaded on first use. |
| `ai.cloud("provider:model", key:, url:)` | A model behind a web API: `openrouter`, `openai` (and compatible servers through `url:`; `ollama` defaults to localhost), `anthropic`, `bedrock`. Keys come from the provider's usual environment variable (OpenRouter also reads `~/.config/zyth/openrouter.key`). A fill-in that does not convert is asked once more, with the problem explained, before E0803. Failures are E0808. Cost is estimated from a price table. |
| `ai.rules(f, name:)` | Plain code. `f` receives a record `{kind, question, choices, descriptions, input, text, shape, site}` and returns a label (or the value to fill in), `None`/`()` for unsure, or `ai.answer(x, confidence:, why:)`. |
| `ai.human()` | Asks the person at the terminal; end of input means unsure. |
| `ai.scripted(answers)` | Gives the listed answers in order, for tests; running out is E0806. `null`/`()` means unsure. |
| `ai.first(deciders, below: 0.7)` | Asks each decider in turn and takes the first answer that is not unsure and whose confidence is unknown or at least `below`. If none qualifies, the last answer is used. |
| `ai.remember(d)` | Reuses the answer to an identical request made earlier in the run: same site, question, rendered input and shape. Unsure answers are not remembered. |
| `ai.review(d)` | Shows `d`'s answer to the person at the terminal, who may accept it or type a correction. A correction is recorded with the original answer (§25.6). Without a terminal the answer is accepted as it is. |
| `ai.verified(proposer, check, tries: 2)` | The proposer answers and the check judges it. The check is a function `|request, answer|` returning `true` (or `()`) to accept, or `false` or a reason to reject; or an `ai.argue` decider, which accepts an answer whose argument holds; or any other decider, which must give the same answer. A rejected answer is removed from the choices, and the proposer is asked again with the reason in a note, up to `tries` times; then the answer is unsure. The diary records the rejections (`verified`). |
| `ai.argue(framework, semantics: "grounded", mode: "skeptical", solver: "auto")` | Decides by argumentation. `framework(request)` returns `{arguments: [...], attacks: [(a, b), ...], choices: {label: argument}}` (without `choices`, labels name arguments). The answer is the one label whose argument holds: in the grounded extension, or under `"preferred"` or `"stable"` semantics in every extension (`mode: "credulous"`: in some). None or several holding is unsure, with the reason. Small frameworks (up to 16 arguments) are solved inside Zyth; larger ones by an external solver with the ICCMA 2023 command line (`solver:` its path, or the `ZYTH_ARGUE_SOLVER` setting; `"builtin"` never uses one), and E0809 without one or when its answer is not YES or NO. |
| `ai.cheaper(main, cheap, after: 20, agree: 0.95, check: 0.1, memory: path)` | Asks both at each decision site and uses `main`'s answer until `cheap` has agreed on at least `agree` of the last `after` decisions there; then asks only `cheap`, and also `main` on a `check` fraction of decisions (using `main`'s answer when they differ). Agreement falling below `agree` switches back. `memory:` keeps the counts in a JSON file between runs. |

Command-line specs: `human`, `scripted:a,b`, `local:MODEL[@gpu]`, `openrouter:MODEL`, `openai:MODEL`, `anthropic:MODEL`, `bedrock:MODEL`, `first:A|B`, `remember:SPEC`, `review:SPEC`, `cheaper:MAIN|CHEAP`.

**One decision's budget.** `within(dollars: D, seconds: S)` limits a single decision:
- before a cloud decider is asked, its cost is estimated from the prompt; if it is more than what is left of `D` (other deciders of an `ai.first` may already have spent some), it is not asked and its answer is unsure, with the reason;
- cloud requests time out when `S` runs out, a decider is not asked once it has, and an answer that arrives after `S` counts as unsure;
- the diary records the budget. With `ai.first([rules, model])`, rules answer what they can and the model is asked only within the budget.

Cost is the provider's reported cost where it reports one (OpenRouter), otherwise the estimate from a price table.

### 25.5 Determinism

Deciders are not required to be deterministic, and a program's output may depend on which decider ran. This is the purpose of the feature. Everything else is deterministic:

- the evaluation order of §25.3;
- the rendering of inputs;
- the numbering of decisions in single-task code;
- the prompts sent to local and cloud deciders (steady text first, the changing input last, so a model can reuse what it has already read).

Local deciders use greedy scoring and cloud requests use temperature 0, but neither is guaranteed to give the same answer twice. For repeatable runs, use a diary (§25.6).

### 25.6 Diaries and Replay

- **Writing a diary.** `zyth run --diary run.jsonl` (`ZYTH_DIARY`) writes one JSON line per decision, in order, even when the answer was then rejected (E0803). The fields are:
  - `n`, `site`, `kind` (`choose` or `fill`), `question`, `input` (the rendered text);
  - `choices`, `descriptions` and `unsure` for `decide`; `shape` and `shape_text` for `ai`;
  - `answer` (a label, a JSON value, or `null` for unsure), `confidence`, `why`, `by` (which decider answered);
  - `ms`, `fresh`, `reused`, `cost`;
  - `approved` if approval was asked, and `ai_answer` with `corrected: true` if a person corrected the answer.
- **Replaying.** `zyth run --replay run.jsonl` (`ZYTH_REPLAY`) answers decision `n` with record `n` instead of asking a decider, and repeats the recorded approval.
  - The record must have the same site and question; otherwise E0805. The input is not compared, so a replay repeats the recorded choices even if the data changed.
  - A run that makes more decisions than the diary records stops with E0805.
  - Replay needs no decider. Programs whose tasks make decisions concurrently may be numbered differently between runs, and are not guaranteed to replay.
- **Lessons.** `ai.lessons(path, corrections_only:)` reads a diary's decided, non-unsure records as lessons for `ai.teach` (§25.9).

### 25.7 Limits and Cost

- **Setting limits.** `ai.limits(decisions:, seconds:, dollars:)` and the `--ai-max-decisions/seconds/dollars` flags cap:
  - the number of decisions in a run;
  - the time from the first decision on;
  - the estimated cost.
- **When they apply.** Limits are checked before each decision (E0804), so a run can exceed a cost limit by at most one decision. A limit set in the program replaces one from the command line.
- **Counters.** `ai.decisions()` and `ai.spent()` return the counts so far.

### 25.8 Approval

An `approve` arm runs only after a person allows it. The person is asked at the terminal (`[y/N]`).

- **No terminal** (for example, input redirected from a file) means *no*, with a note on the error stream.
- **Flags.** `--approve all` and `--approve none` (`ZYTH_APPROVE`) answer every approval without asking.
- **Replays** reuse the recorded approval.
- **A "no"** runs the `unsure` arm, which §25.2 guarantees exists.

### 25.9 Teaching a Local Decider

`ai.teach(local_decider, lessons, epochs:, lr:, rank:, batch:, max_len:)` fine-tunes a copy of a local decider's model with a LoRA adapter (§19.1) and returns a new local decider.

- **Same layout as at run time.** Every lesson is written in exactly the layout the decider sees when it runs: the same system text, the same question and answers block, the same input rendering. Only the answer tokens are trained on.
- **The lessons:**
  - diary records from `ai.lessons`;
  - `{kind: "chat", question, answer, system}` records, for teaching a model facts it can recall in `ai.chat`.
- **Saving.** `d.model()` returns the model, for example to save it as a `.zyth.model` and load it later with `ai.local(path, tokenizer:)`.
- **Command line.** `zyth teach` drives this from a diary or a folder of notes (§21).

### 25.10 Interaction with Other Features

- **AOT compilation.** Functions that contain decision points may be compiled. The decision itself always runs through the runtime's decision machinery, with the same results as interpreted code.
- **Browser builds.** These have no network client, so cloud deciders report E0808 there. `ai.rules` and `ai.scripted` work. Local deciders need their model files shipped with the page (§13.4), and have not been tested in a browser yet.
- **Permissions (§15).** In a project with `[permissions]`, a cloud decider needs `net:<host>` for its service (for example `net:openrouter.ai`), and a local decider needs `net:huggingface.co` to download a model. A refusal is E0701.

### 25.11 Tooling

- `zyth compare FILE --decider A --decider B [--approve]` runs a program under two deciders and shows where their decisions and outputs differ.
- `zyth diary run.jsonl... [--json]` summarizes a diary per decision site (answers, deciders, cost, time) and suggests code where a decider is not needed: a site that gave one answer at least 95% of at least 10 times, and rules (a word in the input text, or a field's value in a record input) that give the recorded answer on every decision they cover, written as an `ai.rules` decider to put first in `ai.first([...])`. Sites that cost money with no such rule get a suggestion to try `ai.cheaper`.
- `zyth make TASK -o FILE --decider SPEC` writes a program. Each round's checker errors and run errors are fed back, for up to `--rounds` rounds.
- `zyth convert prog.py|prog.js --decider SPEC` rewrites a program in Zyth. The result is accepted only when it prints exactly what the original prints.
- `zyth teach` fine-tunes a local model (§25.9).
- The `scripted-files:a,b` spec answers these tools with whole files, for tests.
- `zyth explain E0800` through `E0811` describe every error of this section.

| Code | Meaning |
|---|---|
| E0800 | No decider, or not a valid decider |
| E0802 | The decider was unsure, and there is no `unsure` arm or optional type |
| E0803 | An answer not among the labels, or not of the requested type |
| E0804 | A limit was reached |
| E0805 | The replay does not match the program |
| E0806 | A scripted decider ran out of answers |
| E0807 | The diary file cannot be written or read |
| E0808 | A cloud request failed |
| E0809 | A local model problem |
| E0810 | `approve` without `unsure`, or `approve unsure` |
| E0811 | A type an AI cannot fill in |

---

## Appendix: Reserved Keywords

```
let var const fn return if else match for in while loop break continue
struct enum trait impl type model use as pub module where
move borrow mut drop region danger extern
go scope parallel pipeline on comptime requires ensures no_grad
true false self Self
```

Contextual keyword (only in `.zyi` definition files): `declare`.

Contextual words of decision points (§25.1; ordinary names elsewhere): `decide`, `ai`, `using`, `decider`, `from`, `approve`, `unsure`.

Everything else in the original vision (`pretrain`, `sft`, `align`, `serve`, `server`, `worker`, `schedule`, `chain`, `federation`, `mesh`, `events`, `emit`, `rag`, `agent`, `eval`, `data`, `with`) is either a standard-library function or deferred.
