// tokenizer.hpp — BPE over a GGUF's tokenizer.ggml.* metadata, in the two
// dialects the target families use:
//
//   ByteLevel  GPT-2 byte remap + a word-splitting pre-tokenizer scanner.
//              Qwen 3.5 and every other byte-level BPE vocab.
//   Sentence   raw UTF-8 symbols, U+2581 for space, no word splitting.
//              Gemma 4.
//
// Parity target: llama.cpp's llama_tokenize on the same vocab (trap T3 — the
// pre-tokenizer is a hand-rolled scanner over unicode categories, NOT
// std::regex, and T4 — special tokens split before BPE).
#pragma once
#include "gguf.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
public:
    explicit Tokenizer(const GGUF& g);

    // encode: text -> token ids. add_bos prepends bos; parse_special makes
    // control tokens match as whole tokens (chat needs this).
    std::vector<int32_t> encode(const std::string& text, bool add_bos, bool parse_special) const;
    std::string          decode(const std::vector<int32_t>& ids) const;
    std::string          decode1(int32_t id) const;   // single token's bytes

    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }
    size_t  vocab_size() const { return id_to_tok_.size(); }
    bool    is_eog(int32_t id) const;                 // any end-of-generation id

private:
    enum class Pretokenizer : uint8_t { ByteLevel, Sentence };

    // The byte-level pre-tokenizer differs between vocabularies in exactly two
    // places, both visible in the reference regex. Encoding them as data keeps
    // one scanner instead of one per family:
    //
    //   \p{N}{1,MaxDigitRun}    digits per piece: 3 for GPT-2/Llama-family
    //                           vocabs, 1 for Qwen 3.5.
    //   [\p{L}\p{M}]+           whether combining marks continue a word.
    //                           Qwen 3.5 says yes; older vocabs say no.
    struct WordSplit {
        size_t max_digit_run = 3;
        bool marks_join_letters = false;
    };

    std::vector<std::string>              id_to_tok_;  // remapped-unicode strings
    std::unordered_map<std::string, int32_t> tok_to_id_;
    std::unordered_map<std::string, int32_t> special_;  // control-token text->id
    std::unordered_map<uint64_t, int32_t> merge_rank_;  // (a<<32|b) -> rank
    std::vector<int32_t> token_type_;
    int32_t bos_ = -1, eos_ = -1;
    Pretokenizer pretokenizer_ = Pretokenizer::ByteLevel;
    WordSplit word_split_;
    bool byte_encode_ = true;
    bool ignore_merges_ = true;

    std::vector<std::string> pretokenize(const std::string& text) const;
    std::vector<int32_t>     bpe_encode_chunk(const std::string& piece) const;
    int32_t id_of(const std::string& t) const {
        auto it = tok_to_id_.find(t); return it == tok_to_id_.end() ? -1 : it->second;
    }
};
