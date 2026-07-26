# PLAN: a modern-transformer inference engine (pure C++, interactive chat)

Goal: one transformer type system, pure CPU, covering the **Gemma 4** and
**Qwen 3.5** families, built so that the matrix/tensor operations sit behind a
narrow seam that a hardware matrix accelerator can take over later.

Older families (Llama, GPT-2-style dense stacks) have been **removed**. They
served their purpose — they are how the storage layer, the tokenizer, the quant
kernels and the parity methodology were derived — but carrying them forward
costs abstraction clarity and buys nothing for the target checkpoints.

Supported GGUF storage formats: F32/F16/BF16, Q8_0, Q4_0, Q4_K, Q6_K.

## 0. Ground rules (non-negotiable)

1. FIDELITY FIRST. The checkpoint dictates anatomy; the engine is a faithful
   executor. Every milestone ends with a verification gate against a reference
   oracle. No perf work before greedy-token parity on the same weights.
2. KEEP THE FIVE STRATA:
     0 typed storage (`Vec`/`Matrix`/`MatT`/`KVCache`) — the ONLY layout owners
     1 algebra (dot, softmax, gelu/silu, rms_norm, `par_for`/`par_map`)
     2 components (`Linear`, norms, gated MLPs, `MoE`) — config-free
     2 token mixing (`attention.hpp`: cache, mask, GQA reduction, RoPE)
     3 transformer + family specifications (weights, layers, cache policy)
     4 application tools (input history, sampling, chat)
   New code must state which stratum it belongs to.
3. Components take dimensions; assemblies take architectures. Hyperparameters
   are compile-time template parameters; weights are runtime data; sequence
   length T is the only runtime dimension.
4. Shared-by-construction beats copied-per-family. If two families need the
   same reduction, it goes in `attention.hpp` with a neutral name and the
   difference becomes a parameter (window width, score scale, rotary fraction).
5. Executable property tests stay in the build and grow.

## 1. Target checkpoints

### Gemma 4 — implemented: E4B
```
Gemma4E4BTextConfig: V=262144 D=2560 L=42 Hq=8 Hkv=2
  LOCAL_HEAD_DIM=256 (sliding, window 512, rope base 10000, full rotation)
  GLOBAL_HEAD_DIM=512 (full attention, rope base 1e6, 1/4 rotation)
  attention_kind(l) = Full if l % 6 == 5 else Sliding
  FF=10240 (GELU-gated), PLE=256, CTX=131072
  KV sharing: layers 24..41 own no K/V; they read layer 22 (local) / 23 (full)
  logit softcap 30.0, tied embeddings, embedding scale = bf16(sqrt(D))
```
Vision (mmproj) is implemented and joins at `EmbeddingSegment<D>`.

### Gemma 4 12B Unified — IMPLEMENTED (text)
Config pinned from `google/gemma-4-12B` `config.json` (`gemma4_unified_text`):
```
V=262144 D=3840 L=48 Hq=16 FF=15360 (gelu_pytorch_tanh) CTX=262144
head_dim=256 (sliding, Hkv=8), global_head_dim=512 (full, Hkv=1)
layer_types: full at l % 6 == 5 — the same period-6 rule as E4B, and the
             final layer (47) is global, as the model card requires
sliding_window=1024
rope: sliding = default, theta 1e4; full = proportional, theta 1e6,
      partial_rotary_factor 0.25   → RotaryEmbedding<512, 1000000, 1, 4>
rms_norm_eps=1e-6  tie_word_embeddings=true  final_logit_softcapping=30.0
num_kv_shared_layers=0            → NO KV sharing
hidden_size_per_layer_input=0     → NO PLE
attention_k_eq_v=true             → V IS K on layers with no v_proj tensor
enable_moe_block=false, use_double_wide_mlp=false
```
So 12B is a *simpler* decoder than E4B — no PLE tail, no shared-KV layers —
but it added two things the code did not express, and both became types:
1. **Per-attention-kind KV head count.** Sliding layers have `Hkv=8`, global
   layers `Hkv=1`, so `Hkv` moved from a config constant into the attention
   weight type (`Attention::KV_HEADS`) and the cache follows it.
2. **Unified K/V.** `GemmaKVKind` replaced the `OwnsKV` bool with three cases —
   `Owned`, `Unified`, `Shared`. On a unified layer the VALUE is the raw `W_k`
   output taken *before* the key's learned norm and *before* RoPE, then given
   the scale-free RMS the ordinary value path uses. `key_and_value()` exists so
   that projection runs once.
The PLE tail also became a template parameter (`GemmaPerLayerTail` for E4B,
`GemmaNoTail` for 12B) rather than a "has PLE" bool, and `[[no_unique_address]]`
makes its absence free. E4B's numerics and tests were unchanged by all of this.
Its multimodality is also encoder-free: raw image patches and audio are
projected straight into the decoder embedding space (`patch_size=16`,
`num_soft_tokens=280`, `mm_embed_dim=3840`), so there is no ViT to port —
`EmbeddingSegment<D>` is still the seam, with a much smaller producer.
Note the local llama.cpp checkout predates this checkpoint (its size switch
knows 30/35/42/60, not 48), so read the HF config and a `gguf_dump`, not that
switch, when pinning the config.

### Gemma 4 — 26B-A4B (MoE)
The reference identifies Gemma 4 sizes by layer count:
`30 → 26B-A4B`, `35 → E2B`, `42 → E4B`, `60 → 31B`.
What actually changes for **26B-A4B** (from the reference tensor schema):
- The FFN becomes MoE, but *not* by swapping the channel mixer alone. An expert
  layer carries the dense FFN **as a shared expert**, a router with its own
  scale vector (`ffn_gate_inp.scale`), routed experts (either fused
  `ffn_gate_up_exps` or separate `ffn_gate_exps`/`ffn_up_exps`, plus
  `ffn_down_exps` and a per-expert scale), and **three extra norms**
  (`ffn_pre_norm_2`, `ffn_post_norm_1`, `ffn_post_norm_2`).
  So the MoE layer is a distinct layer type with a second normalized branch,
  not `GemmaDenseDecoderLayer` with `MoE` substituted for `GeluGatedMLP`.
- PLE is optional (`n_embd_per_layer > 0`); a checkpoint without it must not
  require the PLE tail.
- `LAYER_OUT_SCALE` is optional per layer.
`EmbeddingSegment<D>` stays the modality/decoder seam and does not change.

### Qwen 3.5 — 4B and 9B
Config pinned from `Qwen/Qwen3.5-4B` and `Qwen/Qwen3.5-9B` `config.json`.
Both are `L=32`, so the loader must key on width as well as depth:
```
             4B                       9B
D            2560                     4096
FF           9216                     12288
V            248320                   248320
tied         true                     false
shared: Hq=16 Hkv=4 head_dim=256, rope_theta=1e7, partial_rotary_factor=0.25
        full_attention_interval=4, rms_norm_eps=1e-6, CTX=262144
        linear: key_head_dim=128 value_head_dim=128
                num_key_heads=16 num_value_heads=32 conv_kernel_dim=4
```
Note `Hq*head_dim = 4096 != D` on the 4B — the same convention break Gemma
makes, which is why the four attention LAWS never assume it.

Qwen 3.5 is a **hybrid** stack, and this is the interesting part of the work:
- Layer schedule: `is_recurrent(l) = (l + 1) % 4 != 0`. Three of every four
  layers are **gated DeltaNet linear attention**; every fourth is **gated
  full attention**.
- Residual topology is plain pre-norm — despite the tensor being named
  `attn_post_norm`, it is the FFN's *input* norm:
  `h = x + mix(rms_1(x))`, `y = h + ffn(rms_2(h))`. That is exactly the
  canonical `TransformerBlock` with two `ResidualBranch`es and no post
  transform, so Qwen reuses it rather than getting a nominal layer type.
- Full-attention layers are **gated**: one Q projection emits `2*Hq*head_dim`,
  laid out per head as `[q | gate]`; per-head Q/K RMSNorm; MRoPE; conventional
  `1/sqrt(head_dim)` score scale; then `out *= sigmoid(gate)` before `Wo`.
  For text-only positions all MRoPE sections carry the same position, so
  interleaved MRoPE reduces exactly to NEOX-ordered partial RoPE —
  `RotaryEmbedding<256, 10000000, 1, 4>` is bit-for-bit correct there.
- Linear-attention layers: a different token mixer entirely, with a
  **fixed-size recurrent state instead of a growing K/V cache**.
- MTP / NextN is an optional extra block beyond the main stack; not needed for
  correct single-token decoding, skipped in the first pass.
- Channel mixer is SwiGLU (`GatedMLP`); `qwen35moe` routes SwiGLU experts plus
  a gated shared expert.

The gated delta rule, derived from the reference autoregressive graph, per
value head `h` with state `S ∈ R^{Dk × Dv}`:
```
qkv = conv1d_causal(x·W_qkv, width 4) -> silu       split as [q | k | v]
q,k = l2_normalize per head;  q *= 1/sqrt(Dk)
b   = sigmoid(x·W_beta)                              per value head
g   = A * softplus(x·W_alpha + dt_bias)              per value head, A <= 0
S   <- S * exp(g)                                    gated decay
d   = (v - Sᵀk) * b                                  the delta
S   <- S + k ⊗ d                                     rank-1 update
o   = Sᵀq
out = (rmsnorm(o) * silu(x·W_z)) · W_out             gated output norm
```
State cost is `num_value_heads * Dv * Dk` floats per layer — 2 MB fp32 per
layer at 32×128×128, constant in T. That is the whole point of the hybrid:
24 of 32 layers stop paying the `[GROWS-T]` cache tax.

## 2. Milestones

### G1 — Gemma 4 12B Unified   [DONE — greedy parity verified]
Config, architecture, loader, `gemma4_check`/`gemma4_text` support, a chat
adapter, and unit tests (schedule, per-kind KV heads, cache topology, parameter
count, and unified-K/V semantics) are in. Vision/audio is NOT done: 12B is
encoder-free, projecting raw patches and audio straight into the decoder
embedding space, so it needs a much smaller producer behind the same
`EmbeddingSegment<D>` seam.
GATE G1 **MET**: `gemma4_check` validates the real `gemma-4-12b-it-Q4_K_M.gguf`
(667 tensors), and `gemma4_text` reproduces llama.cpp's greedy continuation
token-for-token. The checkpoint confirmed the derived anatomy and corrected one
omission: every layer carries a scalar `layer_output_scale`, so a "no tail"
layer would have silently dropped it — 12B's tail is scale-only, not absent.
Also confirmed: partial RoPE is encoded as a `rope_freqs` tensor (1.0 for
planes 0-63, 1e30 for 64-255), which is exactly what
`RotaryEmbedding<512, 1e6, 1, 4>` computes.

### G2 — Gemma 4 26B-A4B (MoE)
New layer type with the router + shared-expert + extra-norm topology above,
reusing `MoE<..., Expert>` for the routed part. Derive the routing arithmetic
(softmax vs sigmoid gate, renormalization, the router scale vector) from the
reference graph and check it on paper before coding.
GATE G2: greedy parity with llama.cpp on the same file, plus a unit test that
routing with TOPK == NE and uniform scores reduces to the dense FFN.

### Q1 — Qwen 3.5 architecture and loader   [DONE]
`src/recurrent.hpp` (model-neutral: causal conv1d state, delta-net state, the
gated delta rule), `src/qwen35.hpp` (4B/9B configs, both mixers, the hybrid
schedule, the architecture policy), `src/qwen35_loader.hpp`, `tools/qwen35_check`,
a ChatML adapter in the shared REPL, and `tests/qwen35_tests.cpp`.
The design bet paid off: both layer kinds bind the SAME canonical
`TransformerBlock` and differ only in the mixer type, so the residual stream,
the prefix-cache contract, and the channel mixer are untouched.
Verified without a checkpoint: the schedule (24 + 8), conv1d against its own
equation, the delta rule as an associative memory (write at k, read at k
returns v; an orthogonal query returns nothing; exp(gate) decays it), the
gated-attention head split, schedule enforcement at assembly time, and —
the one that matters most for a sequential mixer — **token-by-token decode
equals one-shot prefill**.

### Q2 — Qwen 3.5 parity against a real checkpoint   [DONE]
GATE Q2 **MET**: `Qwen3.5-4B-Q4_K_M.gguf` and `Qwen3.5-9B-Q4_K_M.gguf` both
load, and 4B reproduces llama.cpp's greedy continuation token-for-token
("The capital of France is" -> " Paris.\nA. True\nB. False").

The one genuinely ambiguous decision was settled BY EXPERIMENT rather than by
reading: `QwenGatedDeltaNet::key_head_of` TILES key heads across value heads
(`h % Hk`) rather than grouping them GQA-style (`h / (Hv/Hk)`). Flipping that
single line on the real checkpoint turns top-1 " Paris" (logprob -0.66) into
" a" (-2.05) and coherent text into gibberish. Both mappings are shape-legal,
so no amount of reading could decide it — only weights.

Still open: MRoPE reduces to partial RoPE only because text positions are equal
across all four sections, so a vision/video checkpoint needs the sectioned form.

These files also forced **Q5_K** support: the Qwen Q4_K_M mixes Q5_K tensors.
Q5_K is Q4_K plus a high-bit plane; it has no int8-activation kernel yet and
falls back to the fp32-activation reference path, which is correct but slower.

### Q3 — Qwen 3.5/3.6 MoE
`qwen35moe` routes SwiGLU experts plus a gated shared expert; reuse `MoE<...,
Expert>` and the G2 routing work. Qwen 3.6 shares the 3.5 arch in the
reference (`qwen35`/`qwen35moe` handle both), so it should be a config, not a
new family.

### P1 — Matrix/tensor acceleration (only after the parity gates)
The seam is already narrow: `Weight::matvec`/`matmul`, the norms and
activations, and the attention reduction. Decode is bandwidth-bound and prefill
is compute-bound, so they want different treatments:
  1. Prefill: batch the token dimension into real GEMM against a
     weight-stationary tile; this is what an AMX/SME/tensor-core backend wants.
  2. Decode: keep the int8-activation path; optimize bytes moved, not FLOPs.
  3. KV cache: Q8_0 cache storage — `KVCache::append`/`key`/`value` is the
     pre-built seam.
GATE P1: parity suite still green after every kernel change.

## 3. Verification harnesses

- `tools/gguf_dump model.gguf` — metadata + tensor table; the source of truth
  for any new config. Never write a config from a model card.
- `tools/gemma4_check` / `gemma4_vision_check` — schema validation only.
- `tools/gemma4_text model.gguf "prompt" N` — top-5 logits with logprobs plus
  an N-token greedy continuation. **This is the logit-parity harness**; rerun it
  after any numerics-touching change.
- `tools/tokenizer_parity.py build/tokenizer_test <llama-tokenize> vocab.gguf
  corpus.txt` — token-id parity against the reference tokenizer.
- `ctest` — property tests, quant round-trips, Gemma unit tests, multimodal
  tests, chat-session tests, and tokenizer round-trips for whichever vocab
  GGUFs are present.

## 4. Testing strategy

- Property (in-binary, every build): RoPE offset identity; RMSNorm scale
  invariance; attention against its own equation, including the sliding window
  and the "no visible key is an error" rule; quant block round-trips;
  `PrefixCache` purity (pure, extended, repeated, divergent evaluations agree).
- Golden: tokenizer id parity per vocab; top-5 logit dumps; greedy transcripts
  per quant format.
- E2E: multi-turn chat sanity; image turns for Gemma; `/stats` vs the
  bandwidth model.
- Rule: any numerics-touching change reruns the logit harness before merge.

## 5. Trap checklist (each has bitten someone; check explicitly)

  T1 ggml `[in,out]` tensor layout vs a row-major `[out,in]` assumption —
     decided once in the loader, documented in `core.hpp` (`MatT`)
  T2 RoPE pairing (half-split vs interleaved) and partial-rotation fraction
  T3 pre-tokenizer fidelity — a hand-rolled scanner over unicode categories,
     never an approximated `std::regex`
  T4 special tokens matched whole BEFORE BPE
  T5 tied unembedding: no `output.weight`; logits = h · token_embd rows
  T6 fp32 accumulation everywhere; fp16/int8 are storage formats
  T7 exact chat-template whitespace and control-token spelling
  T8 GGUF alignment: tensor offsets are relative to the ALIGNED data start
  T9 metadata key names vary by converter version — trust the dump
  T10 absolute positions are continuous across turns (RoPE needs the true
     absolute position, not a per-turn offset)

## 6. Explicitly out of scope (this phase)

GPU/HIP backends beyond the kernel seam, speculative decoding and the Qwen MTP
head, multi-session batching, training.

## 7. Definition of done

`./build/chat model.gguf` starts quickly, holds a coherent multi-turn
conversation with streaming output, `/stats` reports context and speeds, and
the parity suite is green against llama.cpp on the same files for both target
families.
