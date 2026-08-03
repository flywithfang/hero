// multimodal.hpp — the decoder's input, however its rows were made.
//
// An encoder produces decoder-width rows: the model's embedding table turns
// ids into them, a ViT turns pixels into them. Either way they go into the one
// EmbeddedSequence<D> the decoder consumes, which never knows which was which.
// This is the seam shared by dense Gemma 4 E4B and the later A4B MoE decoder.
#pragma once
#include "core.hpp"
#include <atomic>

class RGBImage {
public:
    RGBImage(size_t width, size_t height, std::vector<uint8_t> pixels) : width_(width), height_(height), pixels_(std::move(pixels)) {
        if (width_ == 0 || height_ == 0) throw std::invalid_argument("RGBImage: dimensions must be non-zero");
        if (width_ > std::numeric_limits<size_t>::max() / height_ || width_ * height_ > std::numeric_limits<size_t>::max() / 3 || pixels_.size() != width_ * height_ * 3) throw std::invalid_argument("RGBImage: pixel count does not match width*height*3");
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    std::span<const uint8_t> pixels() const { return pixels_; }

private:
    size_t width_;
    size_t height_;
    std::vector<uint8_t> pixels_;  // interleaved RGB, row-major
};

// A row an encoder invented has no vocabulary id, but E4B's per-layer table is
// indexed by id and needs one per row, so those rows carry this placeholder.
inline constexpr TokenId SOFT_TOKEN_ID{0};

// What a decoder actually receives: some rows, the ids they came from, and
// where the first of them sits in the conversation. This is a VIEW — the
// sequence that owns the rows outlives any call taking one — and it is how a
// cache hands over the rows it has not computed yet.
template <size_t D>
class EmbeddedRows {
public:
    EmbeddedRows(MatrixView<D> rows, std::span<const TokenId> token_ids, Position conversation_position) : rows_(rows), token_ids_(token_ids), conversation_position_(conversation_position) {
        if (token_ids_.size() != rows_.rows()) throw std::invalid_argument("EmbeddedRows: one id per row");
    }

    size_t tokens() const { return rows_.rows(); }
    MatrixView<D> matrix() const { return rows_; }
    VecView<D> embedding(size_t i) const { return rows_.row(i); }
    // Where row 0 sits in the CONVERSATION, not in this view. Rows are
    // consecutive from there, so this one number is the whole positional story:
    // RoPE angles, sliding-window range, and where the cache resumes all count
    // up from it, and not one of them wants it a row at a time.
    Position conversation_position() const { return conversation_position_; }
    // The vocabulary id each row came from. Rows an encoder invented carry
    // SOFT_TOKEN_ID; Gemma E4B is the one model that reads any of this, to key
    // its per-layer embedding table.
    std::span<const TokenId> token_ids() const { return token_ids_; }
    TokenId token_id(size_t i) const {
        if (i >= tokens()) throw std::out_of_range("EmbeddedRows: token out of range");
        return token_ids_[i];
    }

private:
    MatrixView<D> rows_;
    std::span<const TokenId> token_ids_;
    Position conversation_position_;
};

// The decoder's whole input: every row of the conversation so far, and the id
// each one came from. It starts empty and grows by appending rows, exactly like
// the Matrix beneath it. Rows arrive from one of two places, and the difference
// is in the call rather than in a tag stored on the data: append() for rows the
// vocabulary produced, which bring their ids with them, and append_soft_tokens()
// for rows an encoder invented, which have none.
//
// Row i is at conversation position i. There is nothing to manage: the only
// thing that ever starts elsewhere is a VIEW of the tail, and it knows where it
// was cut from.
template <size_t D>
class EmbeddedSequence {
public:
    EmbeddedSequence() = default;

    size_t tokens() const { return embeddings_.rows(); }
    uint64_t sequence_id() const { return sequence_id_; }

    // Every row, and the rows from `first` on. The second is what a cache hands
    // the model — the rows it has not seen — and it costs nothing, because
    // neither one copies.
    EmbeddedRows<D> view() const { return from_row(0); }
    EmbeddedRows<D> from_row(size_t first) const {
        if (first > tokens()) throw std::out_of_range("EmbeddedSequence: row out of range");
        return EmbeddedRows<D>(embeddings_.view().from_row(first), std::span<const TokenId>(token_ids_).subspan(first), Position{first});
    }

    // Rows the model made from these ids, which stay with them.
    void append(Matrix<D> rows, std::span<const TokenId> ids) {
        if (ids.size() != rows.rows()) throw std::invalid_argument("EmbeddedSequence: one id per row or none at all");
        append_rows(std::move(rows), ids);
    }
    // Rows an encoder invented — image patches, and audio later. They have no
    // vocabulary identity, so every one of them carries the placeholder.
    void append_soft_tokens(Matrix<D> rows) {
        const std::vector<TokenId> borrowed(rows.rows(), SOFT_TOKEN_ID);
        append_rows(std::move(rows), borrowed);
    }

private:
    // A growing sequence is literally an append: the new rows go onto the end
    // of the ones already there, and their ids onto the end of theirs.
    void append_rows(Matrix<D> rows, std::span<const TokenId> ids) {
        if (rows.rows() == 0) throw std::invalid_argument("EmbeddedSequence: empty append");
        if (rows.rows() > std::numeric_limits<size_t>::max() - tokens()) throw std::length_error("EmbeddedSequence: token count overflows");
        embeddings_.append(rows.view());
        token_ids_.insert(token_ids_.end(), ids.begin(), ids.end());
    }

    static uint64_t next_sequence_id() {
        static std::atomic<uint64_t> next{1};
        return next.fetch_add(1, std::memory_order_relaxed);
    }

    Matrix<D> embeddings_;
    std::vector<TokenId> token_ids_;
    uint64_t sequence_id_ = next_sequence_id();
};
