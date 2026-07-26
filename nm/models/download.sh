#!/usr/bin/env bash
# Fetch the Q4 GGUFs this engine targets, into nm/models/ (gitignored).
set -uo pipefail
cd "$(dirname "$0")"
log=download.log
: > "$log"
get() {  # repo file
  echo "=== $1 / $2 ===" | tee -a "$log"
  hf download "$1" "$2" --local-dir . >>"$log" 2>&1 \
    && echo "OK   $2" | tee -a "$log" \
    || echo "FAIL $2" | tee -a "$log"
}
get unsloth/gemma-4-12b-it-GGUF  gemma-4-12b-it-Q4_K_M.gguf
get unsloth/Qwen3.5-4B-GGUF      Qwen3.5-4B-Q4_K_M.gguf
get unsloth/gemma-4-E4B-it-GGUF  gemma-4-E4B-it-Q4_K_M.gguf
get unsloth/Qwen3.5-9B-GGUF      Qwen3.5-9B-Q4_K_M.gguf
echo "ALL DONE" | tee -a "$log"
