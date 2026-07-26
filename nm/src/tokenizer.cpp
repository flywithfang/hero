#include "tokenizer.hpp"
#include "unicode_data.hpp"
#include <algorithm>
#include <array>

// ================= unicode helpers =========================================
namespace {

// GPT-2 byte<->unicode remap. Each byte maps to a printable codepoint so BPE
// operates on text with no raw control bytes; the vocab stores these remapped
// forms. (space 0x20 -> U+0120 'Ġ', etc.)
struct ByteRemap {
    std::array<uint32_t, 256> byte_to_cp;
    std::unordered_map<uint32_t, uint8_t> cp_to_byte;
    ByteRemap() {
        std::vector<int> bs;
        for (int b = int('!'); b <= int('~'); ++b) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
        std::array<bool, 256> in{};
        for (int b : bs) in[b] = true;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            uint32_t cp;
            if (in[b])
                cp = uint32_t(b);
            else
                cp = uint32_t(256 + n++);
            byte_to_cp[b] = cp;
            cp_to_byte[cp] = uint8_t(b);
        }
    }
};
const ByteRemap& remap() {
    static ByteRemap r;
    return r;
}

void utf8_append(std::string& s, uint32_t cp) {
    if (cp < 0x80)
        s += char(cp);
    else if (cp < 0x800) {
        s += char(0xC0 | (cp >> 6));
        s += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += char(0xE0 | (cp >> 12));
        s += char(0x80 | ((cp >> 6) & 0x3F));
        s += char(0x80 | (cp & 0x3F));
    } else {
        s += char(0xF0 | (cp >> 18));
        s += char(0x80 | ((cp >> 12) & 0x3F));
        s += char(0x80 | ((cp >> 6) & 0x3F));
        s += char(0x80 | (cp & 0x3F));
    }
}

// decode UTF-8 into codepoints, tracking the byte offset where each begins.
void utf8_decode(const std::string& s, std::vector<uint32_t>& cps, std::vector<size_t>& off) {
    size_t i = 0, n = s.size();
    while (i < n) {
        off.push_back(i);
        uint8_t c = s[i];
        uint32_t cp;
        size_t len;
        if (c < 0x80) {
            cp = c;
            len = 1;
        } else if ((c >> 5) == 0x6) {
            cp = c & 0x1F;
            len = 2;
        } else if ((c >> 4) == 0xE) {
            cp = c & 0x0F;
            len = 3;
        } else if ((c >> 3) == 0x1E) {
            cp = c & 0x07;
            len = 4;
        } else {
            cp = 0xFFFD;
            len = 1;
        }
        for (size_t k = 1; k < len && i + k < n; ++k) cp = (cp << 6) | (uint8_t(s[i + k]) & 0x3F);
        cps.push_back(cp);
        i += len;
    }
    off.push_back(n);  // sentinel end
}

bool in_ranges(uint32_t cp, const uint32_t (*r)[2], size_t n) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t m = (lo + hi) / 2;
        if (cp < r[m][0])
            hi = m;
        else if (cp >= r[m][1])
            lo = m + 1;
        else
            return true;
    }
    return false;
}
bool is_letter(uint32_t cp) { return in_ranges(cp, nm_letter_ranges, nm_letter_ranges_n); }
bool is_number(uint32_t cp) { return in_ranges(cp, nm_number_ranges, nm_number_ranges_n); }
bool is_mark(uint32_t cp) { return in_ranges(cp, nm_mark_ranges, nm_mark_ranges_n); }
bool is_ws(uint32_t cp) {
    for (size_t i = 0; i < nm_whitespace_n; ++i)
        if (nm_whitespace_set[i] == cp) return true;
    return false;
}
uint32_t ascii_lower(uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

}  // namespace

// ================= construction ============================================
Tokenizer::Tokenizer(const GGUF& g) {
    const std::string tokenizer_model = g.get_str("tokenizer.ggml.model", "");
    const std::string tokenizer_pre = g.get_str("tokenizer.ggml.pre", "");
    if (tokenizer_model == "gemma4" || tokenizer_pre == "gemma4") {
        pretokenizer_ = Pretokenizer::Sentence;
        byte_encode_ = false;
        ignore_merges_ = false;
    } else if (tokenizer_pre == "qwen35") {
        // Qwen 3.5 splits every digit into its own piece and lets combining
        // marks continue a word; everything else matches the GPT-2 scanner.
        word_split_.max_digit_run = 1;
        word_split_.marks_join_letters = true;
    }
    if (g.has("tokenizer.ggml.ignore_merges")) ignore_merges_ = g.get_int("tokenizer.ggml.ignore_merges", ignore_merges_) != 0;

    const auto& toks = g.get("tokenizer.ggml.tokens").arr_str;
    id_to_tok_ = toks;
    tok_to_id_.reserve(toks.size() * 2);
    for (int32_t i = 0; i < (int32_t)toks.size(); ++i) tok_to_id_[toks[i]] = i;

    if (g.has("tokenizer.ggml.token_type")) {
        for (double d : g.get("tokenizer.ggml.token_type").arr_num) token_type_.push_back((int32_t)d);
    }
    // merges: "A B" -> rank
    const auto& merges = g.get("tokenizer.ggml.merges").arr_str;
    merge_rank_.reserve(merges.size() * 2);
    for (int32_t r = 0; r < (int32_t)merges.size(); ++r) {
        const std::string& m = merges[r];
        size_t sp = m.find(' ', 1);
        if (sp == std::string::npos) continue;
        int32_t a = id_of(m.substr(0, sp)), b = id_of(m.substr(sp + 1));
        if (a >= 0 && b >= 0) merge_rank_[(uint64_t(uint32_t(a)) << 32) | uint32_t(b)] = r;
    }
    // special (CONTROL=3 / USER_DEFINED=4) tokens: surface string -> id
    for (int32_t i = 0; i < (int32_t)toks.size(); ++i) {
        int32_t ty = i < (int32_t)token_type_.size() ? token_type_[i] : 1;
        if (ty == 3 || ty == 4) special_[toks[i]] = i;
    }
    bos_ = (int32_t)g.get_int("tokenizer.ggml.bos_token_id", -1);
    eos_ = (int32_t)g.get_int("tokenizer.ggml.eos_token_id", -1);
}

bool Tokenizer::is_eog(int32_t id) const {
    if (id == eos_) return true;
    // Text-surface fallbacks mirror llama.cpp's EOG classification: a family
    // may terminate a turn with a token that is not the GGUF's eos id. Gemma 4
    // uses <turn|> and a tool-response terminator, Qwen uses <|im_end|>.
    for (const char* s : {"<eos>", "<turn|>", "<|tool_response>", "<|im_end|>", "<|endoftext|>"}) {
        auto it = special_.find(s);
        if (it != special_.end() && it->second == id) return true;
    }
    return false;
}

// ============ pre-tokenizer (hand-rolled scanner, trap T3) =================
std::vector<std::string> Tokenizer::pretokenize(const std::string& text) const {
    if (pretokenizer_ == Pretokenizer::Sentence) {
        // Sentence dialect: spaces become U+2581, then merges operate over
        // each maximal newline/non-newline run with no word-level splitting.
        std::string escaped;
        escaped.reserve(text.size());
        for (char ch : text) {
            if (ch == ' ')
                escaped += "\xE2\x96\x81";
            else
                escaped += ch;
        }
        std::vector<std::string> pieces;
        size_t begin = 0;
        while (begin < escaped.size()) {
            const bool newline = escaped[begin] == '\n';
            size_t end = begin + 1;
            while (end < escaped.size() && (escaped[end] == '\n') == newline) ++end;
            pieces.push_back(escaped.substr(begin, end - begin));
            begin = end;
        }
        return pieces;
    }

    std::vector<uint32_t> cp;
    std::vector<size_t> off;
    utf8_decode(text, cp, off);
    const size_t N = cp.size();
    std::vector<std::pair<size_t, size_t>> spans;  // [ini,end) in codepoints

    const bool marks_join = word_split_.marks_join_letters;
    const size_t max_digits = word_split_.max_digit_run;

    auto in = [&](size_t p) { return p < N; };
    // L is the "word continues" class: \p{L}, or [\p{L}\p{M}] on a vocab whose
    // regex lets combining marks stay attached to their base letter.
    auto L = [&](size_t p) { return in(p) && (is_letter(cp[p]) || (marks_join && is_mark(cp[p]))); };
    auto Nn = [&](size_t p) { return in(p) && is_number(cp[p]); };
    auto W = [&](size_t p) { return in(p) && is_ws(cp[p]); };
    auto C = [&](size_t p) -> uint32_t { return in(p) ? cp[p] : 0xFFFFFFFF; };

    size_t prev = 0;
    auto emit = [&](size_t end) {
        if (end > prev) spans.push_back({prev, end});
        prev = end;
    };

    size_t pos = 0;
    while (pos < N) {
        uint32_t c = cp[pos];
        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c == '\'' && pos + 1 < N) {
            uint32_t n1 = ascii_lower(cp[pos + 1]);
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                pos += 2;
                emit(pos);
                continue;
            }
            if (pos + 2 < N) {
                uint32_t n2 = ascii_lower(cp[pos + 2]);
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                    pos += 3;
                    emit(pos);
                    continue;
                }
            }
        }
        // [^\r\n\p{L}\p{N}]?\p{L}+   (L may include \p{M}; the optional prefix
        // character is excluded by letter/number only, in either dialect)
        if (!(c == '\r' || c == '\n' || is_number(c))) {
            if (L(pos) || L(pos + 1)) {
                pos++;
                while (L(pos)) pos++;
                emit(pos);
                continue;
            }
        }
        // \p{N}{1,max_digits}
        if (is_number(c)) {
            size_t ini = pos;
            while (Nn(pos)) {
                if (++pos - ini >= max_digits) {
                    emit(pos);
                    ini = pos;
                }
            }
            emit(pos);
            continue;
        }
        // <space>?[^\s\p{L}\p{N}]+[\r\n]*
        bool space = (c == ' ');
        {
            size_t q = space ? pos + 1 : pos;
            bool other = in(q) && !(W(q) || L(q) || Nn(q));
            if (other && in(pos)) {
                pos += space ? 1 : 0;
                while (in(pos) && !(W(pos) || L(pos) || Nn(pos))) pos++;
                while (C(pos) == '\r' || C(pos) == '\n') pos++;
                emit(pos);
                continue;
            }
        }
        // whitespace runs: \s*[\r\n]+ | \s+(?!\S) | \s+
        size_t nw = 0, last_rn = 0;
        while (W(pos + nw)) {
            uint32_t c2 = cp[pos + nw];
            if (c2 == '\r' || c2 == '\n') last_rn = pos + nw + 1;
            nw++;
        }
        if (last_rn > 0) {
            pos = last_rn;
            emit(pos);
            continue;
        }
        if (nw > 1 && in(pos + nw)) {
            pos += nw - 1;
            emit(pos);
            continue;
        }
        if (nw > 0) {
            pos += nw;
            emit(pos);
            continue;
        }
        // no match: single codepoint
        pos++;
        emit(pos);
    }

    // codepoint spans -> byte substrings
    std::vector<std::string> pieces;
    pieces.reserve(spans.size());
    for (auto [a, b] : spans) pieces.push_back(text.substr(off[a], off[b] - off[a]));
    return pieces;
}

// ================= BPE on one pre-token ====================================
std::vector<int32_t> Tokenizer::bpe_encode_chunk(const std::string& piece) const {
    std::vector<std::string> sym;
    if (byte_encode_) {
        // GPT-2 mode: each byte becomes its printable remapped codepoint.
        sym.reserve(piece.size());
        for (unsigned char b : piece) {
            std::string s;
            utf8_append(s, remap().byte_to_cp[b]);
            sym.push_back(std::move(s));
        }
    } else {
        // Sentence mode: initial symbols are raw UTF-8 codepoints.
        std::vector<uint32_t> cps;
        std::vector<size_t> offsets;
        utf8_decode(piece, cps, offsets);
        sym.reserve(cps.size());
        for (size_t i = 0; i < cps.size(); ++i) sym.push_back(piece.substr(offsets[i], offsets[i + 1] - offsets[i]));
    }
    if (sym.empty()) return {};

    // ignore_merges: a pre-token already present in the vocab wins directly,
    // without ever consulting the merge table.
    std::string whole;
    for (auto& s : sym) whole += s;
    if (ignore_merges_)
        if (int32_t w = id_of(whole); w >= 0) return {w};

    // greedy lowest-rank merges over adjacent symbol pairs.
    for (;;) {
        int best_rank = INT32_MAX;
        size_t best_i = 0;
        bool found = false;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            int32_t a = id_of(sym[i]), b = id_of(sym[i + 1]);
            if (a < 0 || b < 0) continue;
            auto it = merge_rank_.find((uint64_t(uint32_t(a)) << 32) | uint32_t(b));
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
                found = true;
            }
        }
        if (!found) break;
        sym[best_i] += sym[best_i + 1];
        sym.erase(sym.begin() + best_i + 1);
    }

    std::vector<int32_t> ids;
    ids.reserve(sym.size());
    for (auto& s : sym) {
        int32_t id = id_of(s);
        if (id >= 0)
            ids.push_back(id);
        else
            for (unsigned char ch : s) {
                std::string fallback;
                if (byte_encode_) {
                    fallback.assign(1, char(ch));
                } else {
                    static constexpr char hex[] = "0123456789ABCDEF";
                    fallback = {'<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>'};
                }
                int32_t bid = id_of(fallback);
                if (bid >= 0) ids.push_back(bid);
            }
    }
    return ids;
}

// ================= encode / decode =========================================
std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_bos, bool parse_special) const {
    std::vector<int32_t> out;
    if (add_bos && bos_ >= 0) out.push_back(bos_);

    // split on special-token surfaces (longest match) when parse_special.
    auto tokenize_normal = [&](const std::string& s) {
        for (const auto& piece : pretokenize(s))
            for (int32_t id : bpe_encode_chunk(piece)) out.push_back(id);
    };

    if (!parse_special || special_.empty()) {
        tokenize_normal(text);
        return out;
    }

    size_t i = 0, n = text.size(), run = 0;
    while (i < n) {
        // find longest special surface starting at i
        int32_t sid = -1;
        size_t slen = 0;
        for (const auto& [surf, id] : special_) {
            if (surf.size() > slen && i + surf.size() <= n && text.compare(i, surf.size(), surf) == 0) {
                sid = id;
                slen = surf.size();
            }
        }
        if (sid >= 0) {
            if (i > run) tokenize_normal(text.substr(run, i - run));
            out.push_back(sid);
            i += slen;
            run = i;
        } else
            ++i;
    }
    if (run < n) tokenize_normal(text.substr(run));
    return out;
}

std::string Tokenizer::decode1(int32_t id) const {
    if (id < 0 || id >= (int32_t)id_to_tok_.size()) return "";
    int32_t ty = id < (int32_t)token_type_.size() ? token_type_[id] : 1;
    if (ty == 3) return "";  // control tokens render as nothing
    const std::string& t = id_to_tok_[id];
    if (!byte_encode_) {
        // Sentence-dialect byte fallbacks have literal <0xXX> surfaces.
        if (t.size() == 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x' && t[5] == '>') {
            auto nibble = [](char c) -> int {
                if ('0' <= c && c <= '9') return c - '0';
                if ('A' <= c && c <= 'F') return c - 'A' + 10;
                if ('a' <= c && c <= 'f') return c - 'a' + 10;
                return -1;
            };
            const int hi = nibble(t[3]), lo = nibble(t[4]);
            if (hi >= 0 && lo >= 0) return std::string(1, char((hi << 4) | lo));
        }
        std::string out = t;
        const std::string escaped_space = "\xE2\x96\x81";
        for (size_t at = 0; (at = out.find(escaped_space, at)) != std::string::npos;) {
            out.replace(at, escaped_space.size(), " ");
            ++at;
        }
        return out;
    }

    // reverse GPT-2 byte remap: token codepoints -> raw bytes.
    std::vector<uint32_t> cp;
    std::vector<size_t> off;
    utf8_decode(t, cp, off);
    std::string out;
    for (uint32_t c : cp) {
        auto it = remap().cp_to_byte.find(c);
        if (it != remap().cp_to_byte.end())
            out += char(it->second);
        else
            utf8_append(out, c);
    }
    return out;
}
std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) out += decode1(id);
    return out;
}
