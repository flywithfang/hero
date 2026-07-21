# PLAN: Uniform Transformer Inference Engine (pure C++, interactive chat)

Goal: a single-binary C++ program with one transformer type system and runtime
for Llama, Gemma 4, and Qwen 3.5/3.6 families. Llama-3.2 and Gemma 4 E4B text
and vision are implemented; Gemma MoE and Qwen heterogeneous layers are next.
The engine runs pure-CPU interactive chat and supports common GGUF storage
formats (F32/F16/BF16, Q8_0, Q4_0, Q4_K, Q6_K).

## 0. Ground rules (non-negotiable, from the design history)

1. FIDELITY FIRST. The checkpoint dictates anatomy; the engine is a faithful
   executor. Every milestone ends with a verification gate. No perf work
   before greedy-token parity with a reference implementation.
2. KEEP THE FIVE STRATA:
     0 typed storage (Vec/Mat/RowStore/PastView)  — the ONLY layout owners
     1 algebra (operator*, dot, softmax, silu, rmsnorm kernels)
     2 components (Linear, RMSNorm, GatedMLP, Attention, Rope) — config-free
     3 transformer + family specifications (model, layers, state, policies)
     4 runtime (AutoregressiveRuntime: prefill/step/generate; chat; sampling)
   New code must state which stratum it belongs to.
3. Components take dimensions; assemblies take architectures. Hyperparameters
   are compile-time template parameters; weights are runtime data; sequence
   length T is the only runtime dimension.
4. Invariants live in types where possible: existing static_asserts and
   model-family parameter-count checks stay green. Extend, don't bypass.
5. Executable property tests stay in the build (RoPE offset identity,
   RMSNorm scale invariance) and grow (tokenizer round-trip, quant
   round-trip, logit diff).

## 1. Target configs (already in v10, verified by static_assert)

Llama32_1B: V=128256 D=2048 L=16 Hq=32 Hkv=8 Dqk=Dv=64 FF=8192
            CTX=131072 RMS=true ROPE=true base=500000 TIED=true GatedMLP
            params ~1.24B, KV cache 32 KB/token fp16
Llama32_3B: V=128256 D=3072 L=28 Hq=24 Hkv=8 Dqk=Dv=128 FF=8192
            TIED=true, otherwise same anatomy. params ~3.21B.
Loader must VALIDATE GGUF metadata against the compiled config and abort
with a clear diff on mismatch (wrong file for this binary is a user error,
not a crash).

## 2. Repository layout

  src/
    core.hpp          strata 0-1 (from v10, split out)
    components.hpp    stratum 2  (Linear, norms, MLPs, Attention, Rope)
    transformer.hpp   stratum 3  (residual branches/block, model/session core)
    llama.hpp         stratum 3  (Llama config, layer/cache architecture policy)
    gemma4.hpp        stratum 3  (Gemma config, layer/cache architecture policy)
    runtime.hpp       stratum 4  (AutoregressiveRuntime, samplers)
    tensor_loader.hpp shared shape-safe GGUF tensor construction
    llama_loader.hpp  Llama GGUF schema validation and immutable assembly
    gguf.hpp/.cpp     GGUF parsing + mmap + tensor registry
    quant.hpp         block formats + dequant + quantized matvec kernels
    tokenizer.hpp/.cpp  byte-level BPE + chat template
    chat.cpp          main(): CLI chat REPL
  tools/
    gguf_dump.cpp     M0 deliverable: print metadata + tensor table
    logit_diff.cpp    M1 harness: dump/compare logits for fixed token ids
  tests/              property + golden tests (assert-based, no framework
                      needed initially)
  CLAUDE.md           project memory (updated version provided)

## 3. Milestones

### M0 — GGUF reader + dump tool
Parse: magic "GGUF" (0x46554747 LE), version (expect 3), tensor_count,
metadata_kv_count; metadata KV store (typed values incl. arrays and strings);
tensor infos (name, n_dims, dims[], ggml_type, offset); alignment from
`general.alignment` (default 32); data section offset = aligned end of
header. mmap the file; tensor data pointer = data_start + offset.
Deliverable: `gguf_dump model.gguf` prints every metadata key/value and a
tensor table (name, shape, type, bytes, offset).
GATE M0: dump of the real Llama-3.2-1B GGUF matches `gguf_dump.py` /
llama.cpp's own listing; all expected keys below are present and recorded
in a committed metadata.txt for reference.

Metadata keys to read (names as written by llama.cpp convert script —
VERIFY against the M0 dump, do not trust this list blindly):
  general.architecture            = "llama"
  llama.block_count, llama.embedding_length, llama.feed_forward_length
  llama.attention.head_count, llama.attention.head_count_kv
  llama.attention.layer_norm_rms_epsilon
  llama.rope.freq_base            (500000)
  llama.rope.dimension_count      (=Dqk)
  llama.context_length
  llama.rope.scaling.* / original_context_length   (Llama-3.2 long-ctx
    scaling params; exact key names TO BE CONFIRMED from the M0 dump —
    expected semantics: factor=32, low_freq_factor=1, high_freq_factor=4,
    original_max_position_embeddings=8192)
  tokenizer.ggml.model            = "gpt2"  (means: byte-level BPE)
  tokenizer.ggml.pre              = "llama-bpe"
  tokenizer.ggml.tokens[], tokenizer.ggml.merges[], tokenizer.ggml.token_type[]
  tokenizer.ggml.bos_token_id (128000), .eos_token_id
  tokenizer.chat_template         (Jinja text; we hardcode the known
                                   Llama-3 template but keep this for ref)

### M1 — Weight loading (F32/F16) + forward fidelity   [THE BIG GATE]
Tensor name map (GGUF -> our Model fields):
  token_embd.weight                 -> embed.wte          [V x D]
  blk.N.attn_norm.weight            -> block[N].ln1.gamma [D]
  blk.N.attn_q.weight               -> attn.q_proj.W      (D -> Hq*Dqk)
  blk.N.attn_k.weight               -> attn.k_proj.W      (D -> Hkv*Dqk)
  blk.N.attn_v.weight               -> attn.v_proj.W      (D -> Hkv*Dv)
  blk.N.attn_output.weight          -> attn.o_proj.W      (Hq*Dv -> D)
  blk.N.ffn_norm.weight             -> block[N].ln2.gamma [D]
  blk.N.ffn_gate.weight             -> ff.gate.W          (D -> FF)
  blk.N.ffn_up.weight               -> ff.up.W            (D -> FF)
  blk.N.ffn_down.weight             -> ff.down.W          (FF -> D)
  output_norm.weight                -> ln_f.gamma         [D]
  output.weight                     -> ABSENT in 1B/3B: tied; unembed = wte
No biases anywhere (Llama). Loader asserts exactly this tensor set.

TRAP T1 (transposes): ggml stores 2-D tensors with dims [ne0, ne1] where
ne0 is the CONTIGUOUS (row) dimension, and llama.cpp computes y = W·x with
W rows contiguous per OUTPUT. Our Mat<In,Out> computes y = x·W with the
In dimension outer. Decide the mapping ONCE in the loader: for each tensor,
either copy-transpose into Mat<In,Out>, or add a Mat variant with the other
layout and a second matvec kernel (preferred for perf later: llama.cpp's
layout is dot-product-friendly per output). RECOMMENDATION: introduce
MatT<In,Out> (row = one OUTPUT's weights, contiguous) + matvec_T kernel
computing Out dot-products of length In; load GGUF 2-D tensors zero-copy
into MatT views over the mmap. This avoids both the transpose copy and the
cache-hostile stride. Document the decision in core.hpp.

RoPE completion:
  - frequency scaling (Llama-3.2): adjust theta_i before table build:
      wavelen = 2*pi/theta
      if wavelen < orig_ctx/high_factor: keep theta
      elif wavelen > orig_ctx/low_factor: theta /= factor
      else: linear blend between the two by
            s = (orig_ctx/wavelen - low)/(high - low)
            theta = (1-s)*(theta/factor) + s*theta
    Constants from metadata. ~10 lines in Rope's constructor.
  - pairing: interleaved GGML NORM pairs `(v[2i], v[2i+1])`; the GGUF
    conversion has already permuted Llama Q/K weights for this convention.
    Keep the executable identity test.
TRAP T2: rope pairing convention. If M1 logits mismatch with correct
weights, this and T1 are the first two suspects.

Numerics: all accumulations in fp32 (already true). F16 tensors: dequant to
fp32 on load initially (1B fp32 resident ~5 GB — acceptable dev mode), or
keep F16 and widen in the kernel (do later with quant work).

logit_diff harness: feed a FIXED token id sequence (no tokenizer needed),
dump top-20 logits per position to a file. Reference: (a) llama.cpp
llama-cli with temp 0 for greedy token parity, and (b) a 10-line Python
transformers script (provided in tools/reference_logits.py) dumping logits
for the same ids.
GATE M1: max |logit delta| vs transformers fp32 within ~1e-2 per position
on 64 positions AND greedy argmax identical for 200 consecutive tokens vs
llama.cpp (temp 0). Bitwise equality is NOT expected (summation order).

### M2 — Tokenizer + chat
Byte-level BPE:
  - byte<->unicode printable remap table (GPT-2 style, e.g. space -> Ġ)
  - pre-tokenizer regex for "llama-bpe": copy VERBATIM from llama.cpp
    (llm_tokenizer_bpe, pre = LLM_TOKENIZER_PRE_LLAMA3). Std::regex cannot
    express \p{L} classes reliably -> implement the pattern as a small
    hand-rolled scanner over unicode categories (llama.cpp does the same).
    TRAP T3: an approximated regex changes tokenization silently and breaks
    greedy parity for non-engine reasons.
  - encode per chunk: greedy lowest-merge-rank fusion (ranks from
    tokenizer.ggml.merges); special tokens matched whole BEFORE BPE.
  - decode: concat token bytes, reverse byte remap.
GATE M2a: tokenizer parity with llama.cpp `llama-tokenize` on a corpus of
tricky cases (unicode, numbers, code, emoji, leading spaces) — 100% id match.

Chat protocol (Instruct model):
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
  {system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n
  {msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
  Stop token: <|eot_id|> (128009). Also honor eos list from metadata.
Multi-turn = incremental prefill: `AutoregressiveRuntime` keeps architecture
state (including the KV cache) across turns;
each user turn appends template-wrapped tokens via prefill(delta), then
stream step() until eot. Never re-prefill history.
Chat REPL: streaming token print (decode incrementally; flush per token),
commands: /reset (new cache), /stats (ctx used, tok/s prefill+decode),
/quit. Context-full policy v1: warn at 90%, refuse at 100% with /reset hint.
GATE M2: interactive chat produces coherent multi-turn conversation;
greedy outputs match llama.cpp for identical prompts.

### M3 — Quantization: Q8_0 and Q4_0
Block formats (all little-endian, block = 32 weights):
  Q8_0: { fp16 d; int8 q[32]; }              34 B   w = d * q
  Q4_0: { fp16 d; uint8 q[16]; }             18 B   two nibbles/byte,
                                                    w = d * (nib - 8)
Implementation: quantized MatT variants (QMatT8, QMatT4) as zero-copy views
over the mmap + specialized matvec kernels that dequantize block-by-block
INSIDE the dot product (fp32 accumulate). Dequant-at-load remains as a
debug path. Embedding rows and norms are typically F32/F16 in these files;
handle mixed per-tensor types via a small variant/dispatch in the loader.
GATE M3: Q8_0 greedy parity with llama.cpp Q8_0 file over 200 tokens;
Q4_0 parity with llama.cpp Q4_0 (quality differs from fp16 — compare
same-quant to same-quant). Memory: 1B Q8_0 resident ~1.3 GB.

### M4 — K-quants: Q4_K_M, Q6_K (the formats people actually download)
Superblock = 256 weights:
  Q6_K: { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; fp16 d; }
        w = d * scales[g] * (q - 32), q from 4 low bits + 2 high bits
  Q4_K: { fp16 d, dmin; uint8 scales[12]; uint8 q[128]; }
        8 sub-blocks of 32; 6-bit scale+min packed in scales[];
        w = d*sc[g]*nib - dmin*m[g]
Copy the exact bit-unpacking from ggml's reference dequant (it is fiddly;
verify with a round-trip test: dequant our way vs ggml's table for one
block of known bytes).
GATE M4: Q4_K_M greedy parity vs llama.cpp same file; quant round-trip
property test in tests/.

### M5 — Performance (only after all parity gates)
Order of attack (decode is BANDWIDTH-bound; measure before/after each):
  1. Threads: par_for/par_map -> persistent thread pool. Parallelize the
     matvec over output rows (MatT layout makes rows independent dots) and
     heads via par_map. Prefill: parallelize over token rows (phase
     structure in forward already marks what is parallel).
  2. SIMD in the kernels: AVX2 fp32/f16 dot; integer-dot paths for Q8/Q4
     (unpack nibbles with shifts/masks, madd). Keep scalar kernels as the
     reference; select at build or runtime.
  3. KV cache access: ensure per-head key scan is contiguous (consider
     storing cache per-head-major if profiling shows strided reads).
  4. Optional: Q8_0 KV cache (quantize on deposit, dequant in attend) —
     the KVCache deposit/past API is the pre-built seam.
Perf model (set expectations, verify with /stats):
  bytes/token ~ weights(active) + KV cache(T). 1B Q8 ~1.1 GB + 32KB*T.
  Dual-channel DDR5 ~60-90 GB/s => ceiling ~55-80 tok/s; naive scalar
  single-thread will land ~5-15; threaded+SIMD should reach a meaningful
  fraction of ceiling. Prefill is compute-bound: report tok/s separately.
GATE M5: >= 20 tok/s decode on 1B Q8_0 on the dev machine at 2K context,
parity tests still green (run the full gate suite after every optimization).

### M6 — Polish
Samplers: temperature, top-k, top-p (host-side, on logits; greedy remains
the test mode). Repetition penalty optional. /save-/load session (dump KV
cache + token history) optional. 3B config enablement (new struct + the
same loader). Sliding context policy (drop-oldest with re-prefill, or
refuse — document the choice; do NOT silently truncate mid-turn).

## 4. Trap checklist (each has bitten someone; check explicitly)
  T1 ggml tensor layout vs Mat<In,Out> — decide once, in loader, documented
  T2 RoPE pairing (rotate_half vs interleaved) + scaling constants
  T3 pre-tokenizer regex fidelity (no std::regex approximations)
  T4 special tokens matched before BPE; correct stop token (<|eot_id|>,
     NOT eos 128001, for Instruct chat)
  T5 tied unembedding: no output.weight tensor; logits = h · wte rows
  T6 fp32 accumulation everywhere; fp16 only as storage
  T7 chat template exact whitespace (\n\n after headers)
  T8 GGUF alignment: tensor offsets are relative to the ALIGNED data start
  T9 metadata key names vary by converter version — trust the M0 dump,
     not this document
  T10 incremental prefill positions: absolute position = cache.tokens(),
     continuous across turns (RoPE needs the true absolute position)

## 5. Testing strategy
  Property (in-binary, every build): RoPE offset identity; RMSNorm scale
    invariance; quant block round-trips; softmax sums to 1.
  Golden (tests/, scripts): tokenizer parity corpus; logit_diff vs
    committed reference dumps; 200-token greedy transcripts per quant.
  E2E (manual then scripted): multi-turn chat sanity; needle-in-haystack
    at 4K/16K once perf allows; /stats numbers vs perf model.
  Rule: any numerics-touching change reruns logit_diff before merge.

## 6. Explicitly out of scope (this phase)
GPU/HIP (M5's kernel seams are the future port surface), speculative
decoding, batching multiple sessions, Qwen3/QK-norm (next architecture:
config + one norm call site — the design bet to validate AFTER Llama
ships), MoE models, sliding-window attention.

## 7. Definition of done
`./chat model-Q4_K_M.gguf` starts in <5 s, holds a coherent multi-turn
conversation with streaming output at >=20 tok/s on the dev machine,
/stats reports context and speeds, and the full parity suite (F16, Q8_0,
Q4_0, Q4_K_M) is green against llama.cpp on the same files.
