#!/usr/bin/env python3
"""Token-id parity check against llama.cpp's llama-tokenize on the same vocab.

The in-build tokenizer test only proves decode(encode(x)) == x. Parity is the
real gate: the same vocab must produce the SAME ids as the reference tokenizer,
line for line. Run this after any pre-tokenizer or BPE change.

    ./tools/tokenizer_parity.py build/tokenizer_test /path/to/llama-tokenize \\
        vocab.gguf corpus.txt
"""
import subprocess
import sys


def reference_ids(llama_tokenize, vocab, line):
    out = subprocess.run(
        [llama_tokenize, "-m", vocab, "-p", line, "--ids", "--no-bos",
         "--no-escape", "--log-disable"],
        capture_output=True, text=True, check=True).stdout.strip()
    return [int(x) for x in out.strip("[]").split(",") if x.strip()]


def main(tokenizer_test, llama_tokenize, vocab, corpus):
    lines = [l.rstrip("\n") for l in open(corpus) if l.strip()]
    ours = subprocess.run([tokenizer_test, vocab], input="\n".join(lines),
                          capture_output=True, text=True,
                          check=True).stdout.strip().split("\n")
    mismatches = 0
    for i, line in enumerate(lines):
        want = reference_ids(llama_tokenize, vocab, line)
        got = [int(x) for x in ours[i].split()] if i < len(ours) and ours[i] else []
        if want != got:
            mismatches += 1
            print("MISMATCH line %d: %r\n  reference: %s\n  ours:      %s"
                  % (i, line[:100], want[:24], got[:24]))
    print("%d/%d lines match %s" % (len(lines) - mismatches, len(lines), vocab))
    return 1 if mismatches else 0


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
