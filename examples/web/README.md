# Tiny GPT in the browser (spec Section 20.2)

`chat.zyth` is the spec's browser demo. It needs a trained `tiny-gpt.zyth.model`
next to it, for example from the GPU demo:

```bash
cd examples && zyth run gpt_gpu.zyth          # writes tiny-gpt.zyth.model
mv tiny-gpt.zyth.model web/
zyth build --target web web/chat.zyth -o web/dist
python3 -m http.server -d web/dist 8000       # open http://localhost:8000
```

`zyth build --target web` copies the runtime (`zyth.wasm`), the program, the
page (`index.html` here, with the loader added), the model file, and the GPT-2
tokenizer files into `dist/`. This demo's model runs on the CPU inside the page
(WebAssembly, one thread). Tensors a program puts `on gpu` use WebGPU in
browsers that have it (and JSPI), and the CPU elsewhere.
