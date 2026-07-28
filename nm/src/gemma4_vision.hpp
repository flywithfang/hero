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
    PositionTable2D(const Scalar* values, std::shared_ptr<const void> keepalive) : values_(values), keepalive_(std::move(keepalive)) {
        if (!values_) throw std::invalid_argument("PositionTable2D: null table");
    }

    Vec<D> at(size_t x, size_t y) const {
        if (x >= Rows || y >= Rows) throw std::out_of_range("PositionTable2D: position exceeds table");
        Vec<D> result;
        const Scalar* px = values_ + x * D;
        const Scalar* py = values_ + (Rows + y) * D;
        for (size_t i = 0; i < D; ++i) result[i] = px[i] + py[i];
        return result;
    }

    Matrix<D> grid(size_t width, size_t height) const {
        if (width > Rows || height > Rows) throw std::out_of_range("PositionTable2D: grid exceeds table");
        Matrix<D> result(width * height);
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x) result.set_row(y * width + x, at(x, y));
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
            const Scalar frequency = Scalar(std::pow(double(Base), -double(2 * i) / double(AxisDim)));
            const Scalar angle = Scalar(position) * frequency;
            const Scalar c = std::cos(angle), s = std::sin(angle);
            const Scalar a = value[i], b = value[i + Planes];
            value[i] = a * c - b * s;
            value[i + Planes] = b * c + a * s;
        }
    }
};

template <size_t Heads, size_t HeadDim, size_t Base>
void apply_vision_rope_2d(Matrix<Heads * HeadDim>& queries, Matrix<Heads * HeadDim>& keys, size_t patches_x, size_t patches_y) {
    if (queries.rows() != patches_x * patches_y || keys.rows() != queries.rows()) throw std::invalid_argument("apply_vision_rope_2d: patch grid mismatch");
    VisionRope2D<HeadDim, Base> rope;
    for (size_t token = 0; token < queries.rows(); ++token) {
        const size_t x = token % patches_x;
        const size_t y = token / patches_x;
        for (size_t head = 0; head < Heads; ++head) {
            rope.apply(MutVecView<HeadDim>{queries.data() + token * Heads * HeadDim + head * HeadDim}, x, y);
            rope.apply(MutVecView<HeadDim>{keys.data() + token * Heads * HeadDim + head * HeadDim}, x, y);
        }
    }
}

template <size_t Patch>
Matrix<Patch * Patch * 3> patchify_rgb(std::span<const uint8_t> pixels, size_t width, size_t height) {
    if (width % Patch != 0 || height % Patch != 0 || pixels.size() != width * height * 3) throw std::invalid_argument("patchify_rgb: invalid image shape");
    const size_t patches_x = width / Patch;
    const size_t patches_y = height / Patch;
    Matrix<Patch * Patch * 3> patches(patches_x * patches_y);
    par_for(patches.rows(), [&](size_t patch_index) {
        const size_t patch_y = patch_index / patches_x;
        const size_t patch_x = patch_index % patches_x;
        MutVecView<Patch * Patch * 3> patch = patches.row_mut(patch_index);
        for (size_t channel = 0; channel < 3; ++channel)
            for (size_t y = 0; y < Patch; ++y)
                for (size_t x = 0; x < Patch; ++x) {
                    const size_t source = ((patch_y * Patch + y) * width + patch_x * Patch + x) * 3 + channel;
                    const size_t target = x + Patch * (y + Patch * channel);
                    patch[target] = 2.f * (Scalar(pixels[source]) / 255.f) - 1.f;
                }
    });
    return patches;
}

template <size_t Pool, size_t Channels>
Matrix<Channels> average_pool_2d(MatrixView<Channels> input, size_t width, size_t height) {
    if (input.rows() != width * height || width % Pool != 0 || height % Pool != 0) throw std::invalid_argument("average_pool_2d: invalid input shape");
    const size_t output_width = width / Pool;
    const size_t output_height = height / Pool;
    Matrix<Channels> output(output_width * output_height);
    par_for(output.rows(), [&](size_t output_index) {
        const size_t output_y = output_index / output_width;
        const size_t output_x = output_index % output_width;
        MutVecView<Channels> pooled = output.row_mut(output_index);
        for (size_t y = 0; y < Pool; ++y)
            for (size_t x = 0; x < Pool; ++x) {
                const VecView<Channels> source = input.row((output_y * Pool + y) * width + output_x * Pool + x);
                const Scalar weight = 1.f / Scalar(Pool * Pool);
                for (size_t channel = 0; channel < Channels; ++channel) pooled[channel] += weight * source[channel];
            }
    });
    return output;
}

// One ViT block. Unlike a decoder layer this one CAN run itself: an encoder
// pass has no cache, no positions carried in from a schedule, and no window —
// the patch grid is the whole input. So the tensors are public data and the
// equation is right below them.
template <size_t D, size_t H, size_t HeadDim, size_t FF>
struct GemmaVisionLayer {
    static_assert(D == H * HeadDim);
    using DLinear = ClippedLinear<D, D>;

    RMSNorm<D> input_norm;
    DLinear WQ;
    PerHeadNorm<H, HeadDim, RMSNorm<HeadDim>> q_norm;
    DLinear WK;
    PerHeadNorm<H, HeadDim, RMSNorm<HeadDim>> k_norm;
    DLinear WV;
    PerHeadNorm<H, HeadDim, RMSNormNoScale<HeadDim>> v_norm;  // no learned scale
    DLinear WO;
    RMSNorm<D> attention_post_norm;
    RMSNorm<D> ffn_norm;
    ClippedLinear<D, FF> W_gate;
    ClippedLinear<D, FF> W_up;
    ClippedLinear<FF, D> W_down;
    RMSNorm<D> ffn_post_norm;

    //  Q = rope2d( q_norm( U WQ ) ),  K = rope2d( k_norm( U WK ) ),  U = norm(x)
    //  V = v_norm( U WV )                                (never rotated)
    //  h = x + post_norm( softmax(Q K^T) V WO )          (dense: every patch
    //  out = h + post_norm( (gelu(h W_gate) (*) h W_up) W_down )   sees every patch)
    Matrix<D> operator()(MatrixView<D> input, size_t patches_x, size_t patches_y) const {
        if (input.rows() != patches_x * patches_y) throw std::invalid_argument("GemmaVisionLayer: patch grid mismatch");
        Matrix<D> U = input_norm(input);
        Matrix<D> Q = q_norm(WQ(U.view()));
        Matrix<D> K = k_norm(WK(U.view()));
        Matrix<D> V = v_norm(WV(U.view()));

        apply_vision_rope_2d<H, HeadDim, 100>(Q, K, patches_x, patches_y);
        Matrix<D> A = scaled_dot_product_attention<H, HeadDim, HeadDim>(Q.view(), K.view(), V.view(), 1.f);

        Matrix<D> attention_branch = attention_post_norm(WO(A.view()));
        Matrix<D> residual = add(input, attention_branch.view());

        Matrix<D> ffn_input = ffn_norm(residual.view());
        Matrix<FF> gate = W_gate(ffn_input.view());
        gelu_in_place(gate.mutable_view());
        Matrix<FF> up = W_up(ffn_input.view());
        Matrix<D> ffn = ffn_post_norm(W_down(hadamard(gate.view(), up.view()).view()));
        return add(residual.view(), ffn.view());
    }
};

class Gemma4E4BVisionEncoder {
    using C = Gemma4E4BVisionConfig;

public:
    using Layer = GemmaVisionLayer<C::D, C::H, C::HEAD_DIM, C::FF>;

    Gemma4E4BVisionEncoder(Weight<C::PATCH * C::PATCH * 3, C::D> patch_embedding, PositionTable2D<C::D, C::POSITION_ROWS> positions, std::vector<Layer> layers, Linear<C::D, C::OUT> projection) : patch_embedding_(std::move(patch_embedding)), positions_(std::move(positions)), layers_(std::move(layers)), projection_(std::move(projection)) {
        if (layers_.size() != C::L) throw std::invalid_argument("Gemma4E4BVisionEncoder: wrong layer count");
    }

    EmbeddingSegment<C::OUT> encode(const RGBImage& image) const {
        const Prepared prepared = prepare(image);
        const size_t px = prepared.width / C::PATCH;
        const size_t py = prepared.height / C::PATCH;
        Matrix<C::PATCH * C::PATCH * 3> patches = patchify_rgb<C::PATCH>(prepared.pixels, prepared.width, prepared.height);
        Matrix<C::D> patch_tokens = patch_embedding_.matmul(patches.view());
        Matrix<C::D> positions = positions_.grid(px, py);
        Matrix<C::D> hidden = add(patch_tokens.view(), positions.view());
        for (const Layer& layer : layers_) hidden = layer(hidden.view(), px, py);

        const size_t out_x = px / C::POOL, out_y = py / C::POOL;
        if (out_x == 0 || out_y == 0) throw std::invalid_argument("Gemma4E4BVisionEncoder: image produces no pooled tokens");
        Matrix<C::D> pooled = average_pool_2d<C::POOL>(hidden.view(), px, py);
        scale_in_place(pooled.mutable_view(), std::sqrt(Scalar(C::D)));
        RMSNormNoScale<C::D> pre_projection(C::EPS);
        Matrix<C::D> normalized = pre_projection(pooled.view());
        Matrix<C::OUT> output = projection_(normalized.view());
        return EmbeddingSegment<C::OUT>(Modality::Image, std::move(output));
    }

private:
    struct Prepared {
        size_t width, height;
        std::vector<uint8_t> pixels;
    };

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
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x) {
                const Scalar sx = Scalar(x) * xr, sy = Scalar(y) * yr;
                const size_t x0 = std::min(size_t(sx), image.width() - 1);
                const size_t y0 = std::min(size_t(sy), image.height() - 1);
                const size_t x1 = std::min(x0 + 1, image.width() - 1);
                const size_t y1 = std::min(y0 + 1, image.height() - 1);
                const Scalar xf = sx - Scalar(x0), yf = sy - Scalar(y0);
                for (size_t c = 0; c < 3; ++c) {
                    const Scalar top = Scalar(source[(y0 * image.width() + x0) * 3 + c]) * (1 - xf) + Scalar(source[(y0 * image.width() + x1) * 3 + c]) * xf;
                    const Scalar bottom = Scalar(source[(y1 * image.width() + x0) * 3 + c]) * (1 - xf) + Scalar(source[(y1 * image.width() + x1) * 3 + c]) * xf;
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
