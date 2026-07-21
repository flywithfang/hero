// gemma4_vision.hpp — Gemma 4 E4B image encoder.
//
// The encoder owns image-specific policy and returns decoder-width soft
// tokens.  The text model sees only EmbeddingSegment<D>, which is also the
// boundary used by a future MoE decoder.
#pragma once
#include "gemma4.hpp"

struct Gemma4E4BVisionConfig {
    static constexpr size_t D = 768;
    static constexpr size_t OUT = Gemma4E4BTextConfig::D;
    static constexpr size_t L = 16;
    static constexpr size_t H = 12;
    static constexpr size_t HEAD_DIM = 64;
    static constexpr size_t FF = 3072;
    static constexpr size_t PATCH = 16;
    static constexpr size_t POOL = 3;
    static constexpr size_t POSITION_ROWS = 10240;
    static constexpr size_t MIN_TOKENS = 40;
    static constexpr size_t MAX_TOKENS = 280;
    static constexpr Scalar EPS = 1e-6f;
    static constexpr Scalar ROPE_BASE = 100.f;
};

template <size_t D, size_t Rows>
class PositionTable2D {
public:
    PositionTable2D(const Scalar* values, std::shared_ptr<const void> keepalive)
        : values_(values), keepalive_(std::move(keepalive)) {
        if (!values_) throw std::invalid_argument("PositionTable2D: null table");
    }

    Vec<D> at(size_t x, size_t y) const {
        if (x >= Rows || y >= Rows)
            throw std::out_of_range("PositionTable2D: position exceeds table");
        Vec<D> result;
        const Scalar* px = values_ + x * D;
        const Scalar* py = values_ + (Rows + y) * D;
        for (size_t i = 0; i < D; ++i) result[i] = px[i] + py[i];
        return result;
    }

private:
    const Scalar* values_;
    std::shared_ptr<const void> keepalive_;
};

template <size_t HeadDim, size_t Base>
class VisionRope2D {
    static_assert(HeadDim % 4 == 0);
    static constexpr size_t AxisDim = HeadDim / 2;
    static constexpr size_t Planes = AxisDim / 2;

public:
    void apply(MutVecView<HeadDim> value, size_t x, size_t y) const {
        apply_axis(value.begin(), x);
        apply_axis(value.begin() + AxisDim, y);
    }

private:
    static void apply_axis(Scalar* value, size_t position) {
        for (size_t i = 0; i < Planes; ++i) {
            const Scalar frequency = Scalar(std::pow(double(Base),
                -double(2 * i) / double(AxisDim)));
            const Scalar angle = Scalar(position) * frequency;
            const Scalar c = std::cos(angle), s = std::sin(angle);
            const Scalar a = value[i], b = value[i + Planes];
            value[i] = a * c - b * s;
            value[i + Planes] = b * c + a * s;
        }
    }
};

template <size_t D, size_t H, size_t HeadDim, size_t FF>
class GemmaVisionLayer {
    static_assert(D == H * HeadDim);

public:
    using DLinear = ClippedLinear<D, D>;

    GemmaVisionLayer(RMSNorm<D> input_norm, DLinear query, DLinear key,
                     DLinear value, DLinear output,
                     PerHeadNorm<H, HeadDim, RMSNorm<HeadDim>> query_norm,
                     PerHeadNorm<H, HeadDim, RMSNorm<HeadDim>> key_norm,
                     RMSNormNoScale<HeadDim> value_norm,
                     RMSNorm<D> attention_post_norm, RMSNorm<D> ffn_norm,
                     ClippedLinear<D, FF> ffn_gate,
                     ClippedLinear<D, FF> ffn_up,
                     ClippedLinear<FF, D> ffn_down,
                     RMSNorm<D> ffn_post_norm)
        : input_norm_(std::move(input_norm)), query_(std::move(query)),
          key_(std::move(key)), value_(std::move(value)), output_(std::move(output)),
          query_norm_(std::move(query_norm)), key_norm_(std::move(key_norm)),
          value_norm_(std::move(value_norm)),
          attention_post_norm_(std::move(attention_post_norm)),
          ffn_norm_(std::move(ffn_norm)), ffn_gate_(std::move(ffn_gate)),
          ffn_up_(std::move(ffn_up)), ffn_down_(std::move(ffn_down)),
          ffn_post_norm_(std::move(ffn_post_norm)) {}

    TokenMatrix<D> operator()(const TokenMatrix<D>& input, size_t patches_x,
                              size_t patches_y) const {
        if (input.rows() != patches_x * patches_y)
            throw std::invalid_argument("GemmaVisionLayer: patch grid mismatch");
        const size_t tokens = input.rows();
        TokenMatrix<D> queries(tokens), keys(tokens), values(tokens);
        VisionRope2D<HeadDim, 100> rope;

        for (size_t t = 0; t < tokens; ++t) {
            Vec<D> normalized = input_norm_(input.row(t));
            Vec<D> q = query_(normalized);
            Vec<D> k = key_(normalized);
            Vec<D> v = value_(normalized);
            q = query_norm_(q);
            k = key_norm_(k);
            for (size_t h = 0; h < H; ++h) {
                rope.apply(slice_mut<HeadDim>(q, h * HeadDim), t % patches_x, t / patches_x);
                rope.apply(slice_mut<HeadDim>(k, h * HeadDim), t % patches_x, t / patches_x);
                Vec<HeadDim> vn = value_norm_(slice<HeadDim>(VecView<D>(v), h * HeadDim));
                std::copy(vn.begin(), vn.end(), v.begin() + h * HeadDim);
            }
            queries.set_row(t, q); keys.set_row(t, k); values.set_row(t, v);
        }

        TokenMatrix<D> attention(tokens);
        for (size_t qpos = 0; qpos < tokens; ++qpos) {
            Vec<D> joined;
            for (size_t h = 0; h < H; ++h) {
                std::vector<Scalar> scores(tokens);
                for (size_t kpos = 0; kpos < tokens; ++kpos)
                    scores[kpos] = dot(slice<HeadDim>(queries.row(qpos), h * HeadDim),
                                       slice<HeadDim>(keys.row(kpos), h * HeadDim));
                softmax(scores);
                Vec<HeadDim> head;
                for (size_t kpos = 0; kpos < tokens; ++kpos)
                    axpy(scores[kpos], slice<HeadDim>(values.row(kpos), h * HeadDim), head);
                std::copy(head.begin(), head.end(), joined.begin() + h * HeadDim);
            }
            Vec<D> branch = output_(joined);
            branch = attention_post_norm_(branch);
            branch += input.row(qpos);

            Vec<D> ff_input = ffn_norm_(branch);
            Vec<FF> gate = ffn_gate_(ff_input);
            gelu(gate);
            Vec<FF> up = ffn_up_(ff_input);
            gate *= up;
            Vec<D> ff = ffn_down_(gate);
            ff = ffn_post_norm_(ff);
            ff += branch;
            attention.set_row(qpos, ff);
        }
        return attention;
    }

private:
    RMSNorm<D> input_norm_;
    DLinear query_, key_, value_, output_;
    PerHeadNorm<H, HeadDim, RMSNorm<HeadDim>> query_norm_, key_norm_;
    RMSNormNoScale<HeadDim> value_norm_;
    RMSNorm<D> attention_post_norm_, ffn_norm_;
    ClippedLinear<D, FF> ffn_gate_, ffn_up_;
    ClippedLinear<FF, D> ffn_down_;
    RMSNorm<D> ffn_post_norm_;
};

class Gemma4E4BVisionEncoder {
    using C = Gemma4E4BVisionConfig;

public:
    using Layer = GemmaVisionLayer<C::D, C::H, C::HEAD_DIM, C::FF>;

    Gemma4E4BVisionEncoder(Weight<C::PATCH * C::PATCH * 3, C::D> patch_embedding,
                           PositionTable2D<C::D, C::POSITION_ROWS> positions,
                           std::vector<Layer> layers, Linear<C::D, C::OUT> projection)
        : patch_embedding_(std::move(patch_embedding)), positions_(std::move(positions)),
          layers_(std::move(layers)), projection_(std::move(projection)) {
        if (layers_.size() != C::L)
            throw std::invalid_argument("Gemma4E4BVisionEncoder: wrong layer count");
    }

    EmbeddingSegment<C::OUT> encode(const RGBImage& image) const {
        const Prepared prepared = prepare(image);
        const size_t px = prepared.width / C::PATCH;
        const size_t py = prepared.height / C::PATCH;
        TokenMatrix<C::D> hidden(px * py);
        for (size_t y = 0; y < py; ++y) {
            for (size_t x = 0; x < px; ++x) {
                Vec<C::PATCH * C::PATCH * 3> patch;
                for (size_t channel = 0; channel < 3; ++channel)
                    for (size_t dy = 0; dy < C::PATCH; ++dy)
                        for (size_t dx = 0; dx < C::PATCH; ++dx) {
                            const size_t source = ((y * C::PATCH + dy) * prepared.width +
                                                   x * C::PATCH + dx) * 3 + channel;
                            const size_t target = dx + C::PATCH * (dy + C::PATCH * channel);
                            patch[target] = 2.f * (Scalar(prepared.pixels[source]) / 255.f) - 1.f;
                        }
                Vec<C::D> token = patch_embedding_.matvec(patch);
                Vec<C::D> position = positions_.at(x, y);
                token += position;
                hidden.set_row(y * px + x, token);
            }
        }
        for (const Layer& layer : layers_) hidden = layer(hidden, px, py);

        const size_t out_x = px / C::POOL, out_y = py / C::POOL;
        if (out_x == 0 || out_y == 0)
            throw std::invalid_argument("Gemma4E4BVisionEncoder: image produces no pooled tokens");
        TokenMatrix<C::OUT> output(out_x * out_y);
        RMSNormNoScale<C::D> pre_projection(C::EPS);
        for (size_t oy = 0; oy < out_y; ++oy) {
            for (size_t ox = 0; ox < out_x; ++ox) {
                Vec<C::D> pooled;
                for (size_t dy = 0; dy < C::POOL; ++dy)
                    for (size_t dx = 0; dx < C::POOL; ++dx)
                        axpy(1.f / Scalar(C::POOL * C::POOL),
                             hidden.row((oy * C::POOL + dy) * px + ox * C::POOL + dx), pooled);
                for (size_t i = 0; i < C::D; ++i) pooled[i] *= std::sqrt(Scalar(C::D));
                Vec<C::D> normalized = pre_projection(pooled);
                output.set_row(oy * out_x + ox, projection_(normalized));
            }
        }
        return EmbeddingSegment<C::OUT>(Modality::Image, std::move(output));
    }

private:
    struct Prepared { size_t width, height; std::vector<uint8_t> pixels; };

    static Prepared prepare(const RGBImage& image) {
        constexpr size_t Align = C::PATCH * C::POOL;
        constexpr size_t MinPixels = C::MIN_TOKENS * Align * Align;
        constexpr size_t MaxPixels = C::MAX_TOKENS * Align * Align;
        auto round_to = [](Scalar value) { return size_t(std::round(value / Align)) * Align; };
        auto ceil_to = [](Scalar value) { return size_t(std::ceil(value / Align)) * Align; };
        auto floor_to = [](Scalar value) { return size_t(std::floor(value / Align)) * Align; };
        size_t height = std::max(Align, round_to(Scalar(image.height())));
        size_t width = std::max(Align, round_to(Scalar(image.width())));
        if (height * width > MaxPixels) {
            const Scalar beta = std::sqrt(Scalar(image.height() * image.width()) / Scalar(MaxPixels));
            height = std::max(Align, floor_to(Scalar(image.height()) / beta));
            width = std::max(Align, floor_to(Scalar(image.width()) / beta));
        } else if (height * width < MinPixels) {
            const Scalar beta = std::sqrt(Scalar(MinPixels) / Scalar(image.height() * image.width()));
            height = ceil_to(Scalar(image.height()) * beta);
            width = ceil_to(Scalar(image.width()) * beta);
        }

        Prepared result{width, height, std::vector<uint8_t>(width * height * 3)};
        const auto source = image.pixels();
        const Scalar xr = width > 1 ? Scalar(image.width() - 1) / Scalar(width - 1) : 0.f;
        const Scalar yr = height > 1 ? Scalar(image.height() - 1) / Scalar(height - 1) : 0.f;
        for (size_t y = 0; y < height; ++y) for (size_t x = 0; x < width; ++x) {
            const Scalar sx = Scalar(x) * xr, sy = Scalar(y) * yr;
            const size_t x0 = std::min(size_t(sx), image.width() - 1);
            const size_t y0 = std::min(size_t(sy), image.height() - 1);
            const size_t x1 = std::min(x0 + 1, image.width() - 1);
            const size_t y1 = std::min(y0 + 1, image.height() - 1);
            const Scalar xf = sx - Scalar(x0), yf = sy - Scalar(y0);
            for (size_t c = 0; c < 3; ++c) {
                const Scalar top = Scalar(source[(y0 * image.width() + x0) * 3 + c]) * (1 - xf) +
                                   Scalar(source[(y0 * image.width() + x1) * 3 + c]) * xf;
                const Scalar bottom = Scalar(source[(y1 * image.width() + x0) * 3 + c]) * (1 - xf) +
                                      Scalar(source[(y1 * image.width() + x1) * 3 + c]) * xf;
                result.pixels[(y * width + x) * 3 + c] = uint8_t(top * (1 - yf) + bottom * yf);
            }
        }
        return result;
    }

    Weight<C::PATCH * C::PATCH * 3, C::D> patch_embedding_;
    PositionTable2D<C::D, C::POSITION_ROWS> positions_;
    std::vector<Layer> layers_;
    Linear<C::D, C::OUT> projection_;
};
