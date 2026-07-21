// tokenizer.hpp — byte-level BPE (GPT-2 style) with the llama-bpe pre-tokenizer,
// special-token handling, and the Llama-3 chat template. Loaded from a GGUF's
// tokenizer.ggml.* metadata. Parity target: llama.cpp's llama_tokenize on the
// same vocab (trap T3 — the pre-tokenizer is a hand-rolled scanner over unicode
// categories, NOT std::regex, and T4 — special tokens split before BPE).
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
    // <|...|> control tokens match as whole tokens (chat needs this).
    std::vector<int32_t> encode(const std::string& text, bool add_bos, bool parse_special) const;
    std::string          decode(const std::vector<int32_t>& ids) const;
    std::string          decode1(int32_t id) const;   // single token's bytes

    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }
    int32_t eot() const { return eot_; }              // <|eot_id|> (chat stop)
    size_t  vocab_size() const { return id_to_tok_.size(); }
    bool    is_eog(int32_t id) const;                 // any end-of-generation id

    // Llama-3 chat: render one exchange increment. Returns the token ids to
    // prefill for a user turn (with the assistant header opened).
    std::vector<int32_t> chat_user_turn(const std::string& user, bool first,
                                        const std::string& system = "") const;

private:
    enum class Pretokenizer : uint8_t { Llama3, Gemma4 };
    std::vector<std::string>              id_to_tok_;  // remapped-unicode strings
    std::unordered_map<std::string, int32_t> tok_to_id_;
    std::unordered_map<std::string, int32_t> special_;  // control-token text->id
    std::unordered_map<uint64_t, int32_t> merge_rank_;  // (a<<32|b) -> rank
    std::vector<int32_t> token_type_;
    int32_t bos_ = -1, eos_ = -1, eot_ = -1;
    Pretokenizer pretokenizer_ = Pretokenizer::Llama3;
    bool byte_encode_ = true;
    bool ignore_merges_ = true;

    std::vector<std::string> pretokenize(const std::string& text) const;  // llama3
    std::vector<int32_t>     bpe_encode_chunk(const std::string& piece) const;
    int32_t id_of(const std::string& t) const {
        auto it = tok_to_id_.find(t); return it == tok_to_id_.end() ? -1 : it->second;
    }
};
