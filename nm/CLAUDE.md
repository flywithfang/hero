# nm — a modern-transformer inference engine (from scratch, C++)

## What this project is
A pure-C++, pure-CPU inference engine for **modern** transformer decoders.
Target families: **Gemma 4** (E4B implemented incl. vision; 12B Unified and
26B-A4B next) and **Qwen 3.5** (4B/9B: architecture, loader and chat
implemented, awaiting a checkpoint for the parity gate). Older families have
been removed: the point of the codebase is the core math and the transformer
abstractions, not breadth of model support.

It is built by the owner (Winston) as a derive-then-implement learning project.
Every architectural component was derived by hand with small traceable numbers
before implementation. `INFERENCE_PLAN.md` holds the plan; this file is the
standing context.

## Working style (important)
- Numbers first, then code. When a new concept appears, show a tiny worked
  example before implementing. Precision over hand-waving; push back on
  imprecision.
- FIDELITY FIRST: the checkpoint dictates anatomy. No perf work before greedy
  parity with llama.cpp / transformers on the same weights.
- Any numerics-touching change reruns the parity harness (`tools/gemma4_text`
  for logits/greedy, `tools/tokenizer_parity.py` for token ids).

## Where the code is
- `src/core.hpp` — storage + algebra. `MatT<In,Out>` is the ggml `[in,out]`
  weight layout, zero-copy over mmap; `par_for`/`par_map` are the parallel seam.
- `src/quant.hpp`, `src/quant_i8.hpp` — GGUF block formats, `Weight<In,Out>`
  matvec/matmul dispatch, int8-activation kernels.
- `src/components.hpp` — config-free weight-carrying components: `Linear`,
  `ClippedLinear`, `RMSNorm`, `RMSNormNoScale`, `GatedMLP` (SwiGLU),
  `GeluGatedMLP`, `MoE` (expert body is a template parameter).
- `src/attention.hpp` — **the model-neutral token-mixing core**: `KVCache`
  (full-retention or sliding ring), `key_is_visible` (the mask, defined once),
  `attend`/`attend_head`/`attend_and_cache` (GQA as gather-and-reduce),
  `RotaryEmbedding` (half-split pairing, optional partial rotation),
  `rotate_heads`, `PerHeadNorm`. Score scale is a parameter, not a constant:
  families that fold 1/sqrt(HeadDim) into a learned Q norm pass 1.0.
- `src/recurrent.hpp` — **the other token mixer**: `CausalConv1dState`,
  `DeltaNetState`, `gated_delta_step`, `softplus`, `l2_normalize`. Attention
  remembers a cache that grows with T; a delta network remembers a fixed-size
  state matrix. Both are "communication across tokens".
- `src/transformer.hpp` — the architecture-neutral spine: one decoder-width
  residual stream, immutable callable `Transformer`, explicit prefix-indexed
  `PrefixCache`. There is deliberately NO generic weights container: each
  family's weights are its own flat struct of per-shape layer vectors.
- `src/qwen35.hpp` + `src/qwen35_loader.hpp` — Qwen 3.5: 4B/9B configs, gated
  attention, the gated delta network, and the 3:1 hybrid layer schedule. Both
  layer kinds are the SAME flat `Qwen35Block`; only the mixer type differs.
  This is the payoff for calling it a token mixer.
- `src/gemma4.hpp` + `src/gemma4_vision.hpp` — Gemma-specific anatomy only:
  configuration, the two attention shapes, KV sharing, PLE, logit softcap, the
  decoder layer equation, and the ViT.
- `src/tokenizer.{hpp,cpp}` — BPE in two dialects (see below).
- `tools/` — the chat application (`chat.cpp` with `chat_session.hpp` and
  `chat_models.hpp`: app code, not engine, which is why it lives here and not in
  `src/`), `gguf_dump`, `gemma4_*` inspection and parity tools, plus two python
  helpers: `tokenizer_parity.py` (ids vs `llama-tokenize`) and
  `gen_unicode_data.py` (regenerates the category tables).

## Design rules (do not violate)
1. Five strata: storage / algebra / components / transformer+specifications /
   application runtime. `Transformer` is called as `T(input, prefix_cache)`.
   `PrefixCache` is explicit derived work for a verified complete-input prefix;
   `Architecture::PrefixState` is its model-specific inner storage. It can
   change cost, never semantics. Conversation history and sampling live outside
   the transformer. Layout knowledge lives ONLY in storage types
   (`Matrix`, `Mat::operator()`, `KVCache`). `Matrix<C>` is a SEQUENCE OF ROWS
   built by `append()`ing whole vectors — never sized-then-poked, and it hands
   out no base pointer. A kernel whose rows are independent uses
   `par_map_rows`/`par_map_heads`; the one kernel that produces columns rather
   than rows (`Weight::matmul`) writes a complete buffer and hands it over with
   `from_row_major`. Rows change after the fact only through `transform_rows`.
2. Abstraction is BOTTOM-UP only. Reusable math components take dimensions
   (template params) and have clear definitions. A layer is a concrete struct
   of tensors named after the checkpoint (`WQ`, `q_norm`, `WO`, ...) plus the
   `attention()` and `forward()` methods that explain how those tensors are
   used; a model's weights are plain vectors, one per physical layer shape;
   `forward_layer` is an if/else on the schedule (`Config::position_in_kind`).
   KV anatomy remains physical: a shared-KV layer has no `WK`/`WV`, while its
   method explicitly attends to an existing cache. Do not infer semantics from
   member-existence `requires` expressions, kind flags, or wrapper variants.
   NO top-down machinery: no
   layer variants/visitors, no schedule-validator types, no generic weight
   containers, no Architecture/Transformer split. A model is ONE entity: its
   tensors and its math, immutable and non-copyable (`Gemma4E4BModel`). It
   exposes exactly TWO operations — `embed(ids, pos)` and
   `forward(state, input) -> logits` — and nothing else. How it runs (layer
   count, order, what it precomputes once per pass) is private; `transformer.hpp`
   must never dictate a pipeline. `PrefixCache` is bound to its model at
   construction and `evaluate(input)` is a MEMBER: it is about CACHING, not
   math, and reaches up to the model rather than the model fitting into it. It
   keeps no token count of its own — `PrefixState::tokens()` is the count, and
   that one function is the whole interface a state owes the outside world. A model's embedding table is its OWN tensor, named and
   scaled in the model that uses it (Gemma ties it and scales by sqrt(D); Qwen
   4B ties it, 9B does not) — there is no token-I/O wrapper. Hyperparameters
   compile-time; weights runtime data; T is the only runtime dimension.
   Share lower mathematical components when they are genuinely identical, but
   keep each physical layer's short equation local to its type. E4B methods
   explicitly apply PLE; 12B methods explicitly do not. Gemma and Qwen remain
   separate: Gemma is sandwich-norm (a norm after each branch) plus a
   per-layer scale, Qwen is
   plain pre-norm with neither, and their mixers take different arguments.
   Merging those buys ~10 lines for four feature tests plus a mixer callback
   punched through the middle — the wrong trade. Duplication that is really
   two different equations is not duplication.
3. Strong types where confusion is plausible AND type-detectable
   (`TokenId`/`Position`, `QHead`/`KvHead`). No role-typed vectors where
   dimensions already discriminate.
4. `par_map` = pure map (partition in, value out, by-value view captures,
   placement owned by the algorithm). `par_for` only for non-map loops
   (batched attention, MoE reduction) with a comment. This is the thread-pool
   seam, and the future hardware-matrix-accelerator seam.
5. The four attention LAWS: q-dim==k-dim (Dqk); v-dim free (Dv);
   Wo in = Hq*Dv; Wo out = D. Everything else is designer freedom.
   Hq*Dqk == D is a convention some models break (Gemma) — never assume it.
6. Anything genuinely shared by modern decoders belongs in `attention.hpp` or
   `components.hpp` under a neutral name. A `Gemma`/`Qwen` prefix means "this
   really is family anatomy", not "this is where I happened to write it".
7. Executable property tests stay in the build: RoPE offset identity, RMSNorm
   scale invariance, attention against its own equation, PrefixCache purity.

## Established facts (derived and verified; don't re-litigate)
- params ~= V*D*(tied?1:2) + L*(D*QW + D*KW + D*VW + OW*D + MLP_MATS*D*FF);
  MoE: (NE+SHARED)*ff + router.
- KV cache/token = L*(KW+VW) floats = L*Hkv*(Dqk+Dv). This is why Gemma 4
  shares K/V across its last layers and windows most of the rest.
- Prefill compute-bound (~T^2/2 attention); decode bandwidth-bound (re-reads
  all weights + whole cache per token). Optimize bytes, not FLOPs, in decode.
- RoPE pairing is a storage convention, not a model property: half-split
  (HF/NEOX) vs interleaved depends on whether conversion permuted Q/K.
  `RotaryEmbedding` implements half-split, which is what Gemma 4 GGUFs need.
  k is cached pre-rotated; v is never rotated; theta_i = base^(-2i/HeadDim).
- Qwen 3.5 is 3:1 gated-DeltaNet to gated-attention. Its recurrent layers carry
  Hv*Dk*Dv floats per layer CONSTANT in T (2 MB at 32x128x128) instead of a
  cache that grows — that is the whole point of the hybrid.
- Two Qwen decisions are derived from llama.cpp but unconfirmed against
  weights, both deliberately one-liners: key heads are TILED across value heads
  (`h % Hk`, not `h / group`), and MRoPE collapses to partial RoPE only because
  text positions are equal across sections. See INFERENCE_PLAN.md gate Q2.
- Known traps T1–T10 are listed in INFERENCE_PLAN.md section 5.

## Conventions
- C++20, no third-party deps (mmap and macOS system frameworks are ok).
- FORMATTING: a parameter list is NEVER broken across lines. Every function,
  method and constructor keeps all its parameters on one line, however long.
  `.clang-format` enforces this with `ColumnLimit: 1000`; write new signatures
  that way rather than relying on the format pass. (`ColumnLimit: 0` is NOT the
  setting for this — it preserves existing breaks instead of joining them.)
  Generated tables are fenced with `// clang-format off`.
- Scalar accumulation fp32 always; fp16/quant are storage formats.
- Comments carry cost tags [COMPUTE][BANDWIDTH][GROWS-T].
- Reference oracles: llama.cpp (greedy tokens, tokenizer, dequant bytes) and HF
  transformers (logit dumps) on the SAME gguf/weights. The user's llama.cpp
  checkout is at `~/projects/detective-english/sm/llama.cpp` and already
  implements `gemma4` and `qwen35`/`qwen35moe` — read it when deriving a new
  family's anatomy.
