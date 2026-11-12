# Zyth samples

Small programs that exercise the language end to end. The results below were
measured on the development machine (WSL2, RTX 3090 power-limited to 100 W)
on 2026-10-22, with the commit that added this directory.

| Sample | What it shows | Result |
|---|---|---|
| `01_shapes.zyth` | Shape error caught by `zyth check` before running | E0406 with where both dimensions came from, and the fix (`Linear(256 -> 10)`) |
| `02_tiny_gpt.zyth` | Character GPT on Shakespeare, GPU | 826K parameters, 1,500 steps in 40 s, loss 1.98 at step 600 → 1.59 at step 1,200, Shakespeare-like text |
| `03_finetune.zyth` | LoRA fine-tune of SmolLM2-135M (`zyth.train.sft`) | Response loss 2.49 → 0.03 in 12 steps; the tuned model answers "The Berlin Zyth meetup is at the old train station, every Thursday." for a city it never saw; the base model is unchanged |
| `04_grpo.zyth` | GRPO from a reward function | Mean reward 0.73 → 0.99 in 40 steps; greedy answer becomes `aaa` |
| `05_express.zyth` | Express server with Zyth closures as routes | `/`, `/square/12.5`, JSON `POST /stats`, and `/stop` (closes the server; the program then exits) all respond correctly |
| `06_aot.zyth` | Same program interpreted vs `--aot` | Collatz search 11,487 ms → 74 ms; map word count 22 → 18 ms; identical output |
| `07_concurrency.zyth` | Tasks, channels, `scope` | 78,498 primes below 10⁶ with 8 workers: 3,689 ms interpreted, 261 ms with `--aot` |
| `08_web.zyth` | `zyth build --target web` | Trains `y = 3x − 1` in headless Chromium: learned w = 2.99, b = −1.00 |
| `09_python.zyth` | NumPy from Zyth (`--features python` build) | `66 2.0`, and the array as a tensor |
| `10_export.zyth` | Save / load / safetensors / ONNX | 99.0% accuracy; reload is exact; onnxruntime 1.19 gives the same classes |
| `11_language.zyth` | Enums, traits, `match`, `Result` and `?`, closures, maps | Expected output, including `Err("unknown shape \`hexagon\`")` |

Run any of them with `zyth run <file>` from this directory (`02` needs
`examples/data/tinyshakespeare.txt`, `03` downloads SmolLM2 from the Hugging
Face Hub, `05` needs `zyth add npm:express`).
