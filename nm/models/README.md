# models/

Q4 GGUF checkpoints the engine is tested against. **Gitignored** — run
`./download.sh` to fetch them (~20 GB total, from the `unsloth` mirrors).

| file | family | notes |
|---|---|---|
| `gemma-4-12b-it-Q4_K_M.gguf` | Gemma 4 12B Unified | 48 layers, 40 sliding + 8 global unified-K/V |
| `gemma-4-E4B-it-Q4_K_M.gguf` | Gemma 4 E4B | 42 layers, PLE, shared KV on the last 18 |
| `Qwen3.5-4B-Q4_K_M.gguf` | Qwen 3.5 4B | hybrid: 24 gated-DeltaNet + 8 gated attention |
| `Qwen3.5-9B-Q4_K_M.gguf` | Qwen 3.5 9B | same shape, wider, untied embeddings |

Both Qwen files mix Q5_K tensors into the Q4_K_M quant, which is why the engine
reads Q5_K as well as Q4_K/Q6_K.

## Verifying

```sh
./build/gemma4_check models/gemma-4-12b-it-Q4_K_M.gguf     # schema + assembly
./build/gemma4_text  models/gemma-4-12b-it-Q4_K_M.gguf "The capital of France is" 12
./build/qwen35_check models/Qwen3.5-4B-Q4_K_M.gguf 4b "The capital of France is" 10
```

Greedy parity against the same file, using llama.cpp as the oracle:

```sh
LC=~/projects/detective-english/sm/llama.cpp/build-arm64-apple-clang+cpu-release/bin
$LC/llama-completion -m models/<file>.gguf -p "The capital of France is" \
  -n 12 --temp 0 --top-k 1 --no-warmup \
  --in-prefix "" --in-suffix "" --chat-template "" < /dev/null
```

The empty `--chat-template` matters: without it llama-completion wraps the
prompt in the model's chat template and you are comparing different inputs.
