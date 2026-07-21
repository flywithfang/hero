# LLM Inference Engine (from scratch, C++) — uniform Llama/Gemma/Qwen inference

## What this project is
A pure-C++, pure-CPU inference engine. Llama-3.2 and Gemma 4 E4B GGUF inference
(including image-text) are implemented; Gemma MoE and Qwen 3.5/3.6 are the
next target families. It is built by the owner (Winston) as a
derive-then-implement learning project. Every architectural component was
derived by hand with small traceable numbers before implementation. Follow
INFERENCE_PLAN.md for milestones; this file is the standing context.

## Working style (important)
- Numbers first, then code. When a new concept appears, show a tiny worked
  example before implementing. Precision over hand-waving; push back on
  imprecision.
- FIDELITY FIRST: the checkpoint dictates anatomy. No perf work before
  greedy parity with llama.cpp / transformers on the same weights.
- Any numerics-touching change reruns the logit_diff harness.

## Current state
- `src/transformer.hpp` is the model-neutral architecture spine derived from
  `General_Multimodal_Transformer_Architecture_Wide.pdf`: one decoder-width
  residual stream, typed token-mixer/channel-mixer residual branches, optional layer
  tail, immutable `TransformerModel`, and mutable `TransformerSession`.
  Llama and Gemma E4B both instantiate it; family files contain specifications
  and cache/layer policy rather than separate forward runtimes.
- Gemma 4 E4B text and vision inference are implemented in immutable
  assemblies. The official Q4_0 text and Q8_0 mmproj fixtures match llama.cpp
  greedy image-text output. `EmbeddingSegment<D>` is the modality/decoder seam;
  A4B MoE should reuse it and replace only the decoder FFN assembly and loader.
- `build/chat` is model-neutral: `ChatModel` adapters are selected from GGUF
  architecture/anatomy. Llama and Gemma E4B share one REPL; Gemma optionally
  owns its vision adapter and supports staged images across multi-turn chat.
- `gpt_pipeline.cpp` is historical concept code, not the production type
  system. Production code lives in `src/transformer.hpp`, family specifications,
  and family loaders; do not copy new models into another standalone pipeline.
- INFERENCE_PLAN.md: the implementation plan (M0 GGUF dump -> M1 fp16
  fidelity -> M2 tokenizer+chat -> M3 Q8_0/Q4_0 -> M4 K-quants -> M5 perf
  -> M6 polish). Follow its gates in order.

## Design rules (do not violate)
1. Five strata: storage / algebra / components / transformer+specifications /
   runtime. `TransformerModel` is immutable; `TransformerSession` exclusively
   owns mutable cache state. Layout knowledge lives ONLY in storage types
   (Mat::operator(), PastView).
2. Components take dimensions (template params); specifications satisfy the
   `TransformerArchitecture` contract. Token I/O satisfies `TokenInputOutput`.
   Model differences are expressed as component/policy types and layer
   schedules, not flat anatomy booleans.
   Hyperparameters compile-time; weights runtime data; T is the only runtime
   dimension.
3. Strong types where confusion is plausible AND type-detectable
   (TokenId/Position, QHead/KvHead + group() as sole constructor). No
   role-typed vectors where dimensions already discriminate.
4. par_map = pure map (partition in, value out, by-value view captures,
   placement owned by the algorithm). par_for only for non-map loops
   (e.g. MoE reduction) with a comment. This is the thread-pool seam.
5. The four attention LAWS: q-dim==k-dim (Dqk); v-dim free (Dv);
   Wo in = Hq*Dv; Wo out = D. Everything else is designer freedom.
   Hq*Dqk == D is a convention some models break (Gemma) — never assume it.
6. Executable property tests stay in the build: RoPE offset identity,
   RMSNorm scale invariance; extend with tokenizer/quant round-trips.

## Established facts (derived and verified; don't re-litigate)
- params ~= V*D*(tied?1:2) + L*(D*QW + D*KW + D*VW + OW*D + MLP_MATS*D*FF);
  MoE: (NE+SHARED)*ff + router. Verified vs advertised sizes in static_assert.
- KV cache/token = L*(KW+VW) floats = L*Hkv*(Dqk+Dv). Llama32-1B: 32 KB fp16.
- Prefill compute-bound (~T^2/2 attention); decode bandwidth-bound
  (re-reads all weights + whole cache per token). Optimize bytes, not FLOPs,
  in decode.
- RoPE pairing is model-specific (Llama GGUF uses interleaved/NORM pairing);
  k is cached pre-rotated and v is never rotated;
  theta_i = base^(-i/(Dqk/2)); Llama-3.2 adds frequency rescaling
  (factor 32, orig ctx 8192) applied before table build.
- Known traps T1-T10 listed in INFERENCE_PLAN.md section 4 (transposes,
  rope pairing, tokenizer regex, special tokens, tied unembed, fp32
  accum, template whitespace, GGUF alignment, metadata names, absolute
  positions across turns).

## Conventions
- C++20, no third-party deps (mmap and macOS system frameworks are ok). Keep property tests
  green in main/tests. Comments carry cost tags [COMPUTE][BANDWIDTH][GROWS-T].
- Scalar accumulation fp32 always; fp16/quant are storage formats.
- Reference oracles: llama.cpp (greedy tokens, tokenizer) and HF
  transformers (logit dumps) on the SAME gguf/weights.

## Roadmap after this phase (context, not tasks)
Gemma 4 MoE via a concrete MoE channel mixer; Qwen 3.5/3.6 via heterogeneous
token-mixer and channel-mixer layer schedules; Q8_0 KV/cache-state storage;
then accelerated matvec kernels and training experiments.
