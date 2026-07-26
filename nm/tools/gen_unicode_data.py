#!/usr/bin/env python3
"""Regenerate src/unicode_data.hpp from llama.cpp's unicode tables.

The pre-tokenizer scanner must agree with the reference tokenizer on unicode
categories byte for byte, so the category tables are taken from llama.cpp's own
generated data rather than from a system unicode library.

    ./tools/gen_unicode_data.py /path/to/llama.cpp > src/unicode_data.hpp
"""
import re
import sys

FLAGS = {"letter": 0x0004, "number": 0x0002, "mark": 0x0010}
MAX_CODEPOINT = 0x110000


def ranges(pairs, bit):
    """Merge llama.cpp's (start, flags) table into [start, end) ranges."""
    out = []
    for i, (start, flags) in enumerate(pairs):
        end = pairs[i + 1][0] if i + 1 < len(pairs) else MAX_CODEPOINT
        if not flags & bit:
            continue
        if out and out[-1][1] == start:
            out[-1][1] = end
        else:
            out.append([start, end])
    return out


def table(name, rows, per_line=7):
    body = "\n".join(
        "  " + ",".join("{0x%X,0x%X}" % (a, b) for a, b in rows[i:i + per_line]) + ","
        for i in range(0, len(rows), per_line))
    return ("inline constexpr uint32_t nm_%s_ranges[][2] = {\n%s\n};\n"
            "inline constexpr size_t nm_%s_ranges_n = %d;\n" % (name, body, name, len(rows)))


def main(llama_cpp_root):
    source = open(llama_cpp_root + "/src/unicode-data.cpp").read()
    block = source.split("unicode_ranges_flags = {", 1)[1].split("};", 1)[0]
    pairs = [(int(a, 16), int(b, 16))
             for a, b in re.findall(r"\{0x([0-9A-Fa-f]+),\s*0x([0-9A-Fa-f]+)\}", block)]

    whitespace = source.split("unicode_set_whitespace = {", 1)[1].split("};", 1)[0]
    spaces = [int(v, 16) for v in re.findall(r"0x([0-9A-Fa-f]+)", whitespace)]

    print("// unicode_data.hpp — GENERATED from llama.cpp src/unicode-data.cpp")
    print("// (unicode_ranges_flags + unicode_set_whitespace). Letter/Number/Mark")
    print("// codepoint ranges and the whitespace set, used by the byte-level BPE")
    print("// pre-tokenizer scanner. Regenerate with tools/gen_unicode_data.py.")
    print("// Using llama.cpp's OWN tables guarantees byte-exact category parity with")
    print("// the libllama tokenizer oracle. Do not edit by hand.")
    print("#pragma once")
    print("#include <cstdint>")
    print("#include <cstddef>")
    # These are packed data tables, not code; let them keep their own layout.
    print("// clang-format off")
    for name, bit in FLAGS.items():
        print()
        print(table(name, ranges(pairs, bit)), end="")
    print()
    print("inline constexpr uint32_t nm_whitespace_set[] = {")
    print("  " + ",".join("0x%X" % cp for cp in spaces))
    print("};")
    print("inline constexpr size_t nm_whitespace_n = %d;" % len(spaces))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1].rstrip("/"))
