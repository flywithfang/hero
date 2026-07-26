// multimodal.hpp — modality-neutral input and embedding composition.
//
// Encoders turn owned prompt inputs into EmbeddingSegment<D> values. The text
// decoder consumes only EmbeddedSequence<D>; it never knows how pixels or audio
// became soft tokens. This is the seam shared by dense Gemma 4 E4B and the
// later Gemma 4 A4B MoE decoder.
#pragma once
#include "core.hpp"
#include <atomic>
#include <variant>

enum class Modality : uint8_t { Text, Image, Audio };

class RGBImage {
public:
    RGBImage(size_t width, size_t height, std::vector<uint8_t> pixels)
        : width_(width), height_(height), pixels_(std::move(pixels)) {
        if (width_ == 0 || height_ == 0)
            throw std::invalid_argument("RGBImage: dimensions must be non-zero");
        if (width_ > std::numeric_limits<size_t>::max() / height_ ||
            width_ * height_ > std::numeric_limits<size_t>::max() / 3 ||
            pixels_.size() != width_ * height_ * 3)
            throw std::invalid_argument("RGBImage: pixel count does not match width*height*3");
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    std::span<const uint8_t> pixels() const { return pixels_; }

private:
    size_t width_;
    size_t height_;
    std::vector<uint8_t> pixels_; // interleaved RGB, row-major
};

class TextPrompt {
public:
    explicit TextPrompt(std::vector<TokenId> tokens) : tokens_(std::move(tokens)) {
        if (tokens_.empty()) throw std::invalid_argument("TextPrompt: empty token sequence");
    }
    std::span<const TokenId> tokens() const { return tokens_; }

private:
    std::vector<TokenId> tokens_;
};

using PromptPart = std::variant<TextPrompt, RGBImage>;

class MultimodalPrompt {
public:
    explicit MultimodalPrompt(std::vector<PromptPart> parts) : parts_(std::move(parts)) {
        if (parts_.empty()) throw std::invalid_argument("MultimodalPrompt: no parts");
    }
    std::span<const PromptPart> parts() const { return parts_; }

private:
    std::vector<PromptPart> parts_;
};

class SequenceSpan {
public:
    SequenceSpan(Modality modality, size_t begin, size_t end)
        : modality_(modality), begin_(begin), end_(end) {
        if (begin_ >= end_) throw std::invalid_argument("SequenceSpan: empty or reversed span");
    }

    Modality modality() const { return modality_; }
    size_t begin() const { return begin_; }
    size_t end() const { return end_; }
    bool contains(size_t i) const { return begin_ <= i && i < end_; }

private:
    Modality modality_;
    size_t begin_;
    size_t end_; // half-open
};

template <size_t D>
class EmbeddingSegment {
public:
    EmbeddingSegment(TokenMatrix<D> embeddings, std::vector<TokenId> token_identities)
        : modality_(Modality::Text), embeddings_(std::move(embeddings)),
          token_identities_(std::move(token_identities)) {
        if (embeddings_.rows() == 0)
            throw std::invalid_argument("EmbeddingSegment: empty segment");
        if (token_identities_.size() != embeddings_.rows())
            throw std::invalid_argument("EmbeddingSegment: text identity count mismatch");
    }

    EmbeddingSegment(Modality modality, TokenMatrix<D> embeddings)
        : modality_(modality), embeddings_(std::move(embeddings)) {
        if (modality_ == Modality::Text)
            throw std::invalid_argument("EmbeddingSegment: text requires token identities");
        if (embeddings_.rows() == 0)
            throw std::invalid_argument("EmbeddingSegment: empty segment");
    }

    Modality modality() const { return modality_; }
    const TokenMatrix<D>& embeddings() const { return embeddings_; }
    std::span<const TokenId> token_identities() const { return token_identities_; }

private:
    Modality modality_;
    TokenMatrix<D> embeddings_;
    std::vector<TokenId> token_identities_; // populated only for text
};

template <size_t D>
class EmbeddedSequence {
public:
    EmbeddedSequence(TokenMatrix<D> embeddings, std::vector<SequenceSpan> spans,
                     std::vector<TokenId> ple_token_identities, size_t first_position)
        : embeddings_(std::move(embeddings)), spans_(std::move(spans)),
          ple_token_identities_(std::move(ple_token_identities)),
          first_position_(first_position) {
        size_t cursor = 0;
        for (const SequenceSpan& span : spans_) {
            if (span.begin() != cursor)
                throw std::invalid_argument("EmbeddedSequence: spans must be contiguous");
            cursor = span.end();
        }
        if (cursor != embeddings_.rows())
            throw std::invalid_argument("EmbeddedSequence: spans do not cover embeddings");
        if (ple_token_identities_.size() != embeddings_.rows())
            throw std::invalid_argument("EmbeddedSequence: PLE identity count mismatch");
    }

    size_t tokens() const { return embeddings_.rows(); }
    size_t first_position() const { return first_position_; }
    uint64_t sequence_id() const { return sequence_id_; }
    MatrixView<D> matrix() const { return embeddings_.view(); }
    VecView<D> embedding(size_t i) const { return embeddings_.row(i); }
    Position position(size_t i) const {
        if (i >= tokens()) throw std::out_of_range("EmbeddedSequence: token out of range");
        return Position{first_position_ + i};
    }
    std::span<const SequenceSpan> spans() const { return spans_; }
    TokenId ple_token_identity(size_t i) const {
        if (i >= tokens()) throw std::out_of_range("EmbeddedSequence: token out of range");
        return ple_token_identities_[i];
    }
    std::span<const TokenId> ple_token_identities() const {
        return ple_token_identities_;
    }
    Modality modality(size_t i) const {
        if (i >= tokens())
            throw std::out_of_range(
                "EmbeddedSequence: token out of range");
        for (const SequenceSpan& span : spans_)
            if (span.contains(i)) return span.modality();
        throw std::logic_error(
            "EmbeddedSequence: token is outside every span");
    }

    EmbeddedSequence suffix(size_t first_token) const {
        if (first_token >= tokens())
            throw std::out_of_range(
                "EmbeddedSequence: empty suffix");

        Matrix<D> embeddings(tokens() - first_token);
        std::copy(
            embeddings_.data() + first_token * D,
            embeddings_.data() + tokens() * D,
            embeddings.data());

        std::vector<SequenceSpan> spans;
        for (const SequenceSpan& span : spans_) {
            if (span.end() <= first_token) continue;
            const size_t begin =
                std::max(span.begin(), first_token) - first_token;
            const size_t end = span.end() - first_token;
            spans.emplace_back(span.modality(), begin, end);
        }

        std::vector<TokenId> identities(
            ple_token_identities_.begin() + first_token,
            ple_token_identities_.end());
        return EmbeddedSequence(
            std::move(embeddings), std::move(spans),
            std::move(identities),
            first_position_ + first_token);
    }

    void append(EmbeddingSegment<D> segment,
                TokenId non_text_ple_identity = TokenId{0}) {
        const size_t appended = segment.embeddings().rows();
        if (appended >
            std::numeric_limits<size_t>::max() - tokens())
            throw std::length_error(
                "EmbeddedSequence: token count overflows");

        const size_t begin = tokens();
        Matrix<D> combined(begin + appended);
        std::copy(
            embeddings_.data(), embeddings_.data() + begin * D,
            combined.data());
        std::copy(
            segment.embeddings().data(),
            segment.embeddings().data() + appended * D,
            combined.data() + begin * D);
        std::vector<SequenceSpan> spans = spans_;
        spans.emplace_back(
            segment.modality(), begin, begin + appended);
        std::vector<TokenId> identities =
            ple_token_identities_;
        if (segment.modality() == Modality::Text) {
            identities.insert(
                identities.end(),
                segment.token_identities().begin(),
                segment.token_identities().end());
        } else {
            identities.insert(
                identities.end(),
                appended, non_text_ple_identity);
        }

        embeddings_ = std::move(combined);
        spans_ = std::move(spans);
        ple_token_identities_ = std::move(identities);
    }

private:
    static uint64_t next_sequence_id() {
        static std::atomic<uint64_t> next{1};
        return next.fetch_add(1, std::memory_order_relaxed);
    }

    TokenMatrix<D> embeddings_;
    std::vector<SequenceSpan> spans_;
    std::vector<TokenId> ple_token_identities_;
    size_t first_position_;
    uint64_t sequence_id_ = next_sequence_id();
};

template <size_t D>
EmbeddedSequence<D> compose_embeddings(std::vector<EmbeddingSegment<D>> segments,
                                       size_t first_position = 0,
                                       TokenId non_text_ple_identity = TokenId{0}) {
    size_t total = 0;
    for (const auto& segment : segments) {
        if (segment.embeddings().rows() > std::numeric_limits<size_t>::max() - total)
            throw std::length_error("compose_embeddings: token count overflows");
        total += segment.embeddings().rows();
    }
    if (total == 0) throw std::invalid_argument("compose_embeddings: no segments");

    TokenMatrix<D> combined(total);
    std::vector<SequenceSpan> spans;
    std::vector<TokenId> identities;
    spans.reserve(segments.size());
    identities.reserve(total);
    size_t cursor = 0;
    for (const auto& segment : segments) {
        const size_t begin = cursor;
        for (size_t r = 0; r < segment.embeddings().rows(); ++r) {
            combined.set_row(cursor++, segment.embeddings().row(r));
            identities.push_back(segment.modality() == Modality::Text
                ? segment.token_identities()[r] : non_text_ple_identity);
        }
        spans.emplace_back(segment.modality(), begin, cursor);
    }
    return EmbeddedSequence<D>(std::move(combined), std::move(spans),
                               std::move(identities), first_position);
}

// Compact visibility policy: causal by default, with optional spans whose
// members may see each other bidirectionally (Gemma 4 A4B uses this for image
// soft tokens; E4B can construct the purely causal form).
class DecoderVisibility {
public:
    explicit DecoderVisibility(size_t tokens, std::vector<SequenceSpan> bidirectional = {})
        : tokens_(tokens), bidirectional_(std::move(bidirectional)) {
        for (const auto& span : bidirectional_)
            if (span.end() > tokens_)
                throw std::invalid_argument("DecoderVisibility: span exceeds sequence");
    }

    bool allows(size_t query, size_t key) const {
        if (query >= tokens_ || key >= tokens_)
            throw std::out_of_range("DecoderVisibility: token out of range");
        if (key <= query) return true;
        for (const auto& span : bidirectional_)
            if (span.contains(query) && span.contains(key)) return true;
        return false;
    }

    // Cache positions are absolute while spans describe the current prefill
    // batch. Earlier cached tokens remain causally visible; positions in this
    // batch use the exact span policy.
    bool allows_absolute(size_t query_position, size_t key_position,
                         size_t first_position) const {
        if (query_position < first_position || key_position < first_position)
            return key_position <= query_position;
        const size_t query = query_position - first_position;
        const size_t key = key_position - first_position;
        if (query >= tokens_ || key >= tokens_)
            return key_position <= query_position;
        return allows(query, key);
    }

private:
    size_t tokens_;
    std::vector<SequenceSpan> bidirectional_;
};
