// tensor_loader.hpp - architecture-neutral GGUF tensor construction.
#pragma once
#include "components.hpp"
#include "gguf.hpp"
#include <string>

namespace tensor_loader {

inline void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("tensor loader: " + message);
}

// Construct a 2-D weight tensor (GGUF dimensions [In, Out]).  F32 and
// quantized tensors remain zero-copy views over the mmap; F16/BF16 tensors are
// materialized as F32 until native half-precision kernels exist.
template <size_t In, size_t Out>
Weight<In, Out> load_weight(const GGUF& gguf, const std::string& name) {
    const TensorInfo& tensor = gguf.require(name);
    expect(tensor.dims.size() == 2, name + " is not 2-D");
    expect(tensor.dims[0] == In && tensor.dims[1] == Out,
           name + " shape [" + std::to_string(tensor.dims[0]) + "," +
           std::to_string(tensor.dims[1]) + "] != expected [" +
           std::to_string(In) + "," + std::to_string(Out) + "]");

    if (tensor.type == GT::F32) {
        MatT<In, Out> view;
        view.p = reinterpret_cast<const Scalar*>(tensor.data);
        return Weight<In, Out>(std::move(view), gguf.keepalive());
    }
    if (tensor.type == GT::F16 || tensor.type == GT::BF16) {
        MatT<In, Out> materialized = MatT<In, Out>::owning();
        dequant_to_f32(tensor.type, tensor.data, materialized.raw(), In * Out);
        return Weight<In, Out>(std::move(materialized));
    }

    expect(In % block_len(tensor.type) == 0,
           name + " In=" + std::to_string(In) + " not divisible by " +
           gt_name(tensor.type) + " block length");
    return Weight<In, Out>(tensor.type, tensor.data, gguf.keepalive());
}

template <size_t N>
Vec<N> load_vector(const GGUF& gguf, const std::string& name) {
    const TensorInfo& tensor = gguf.require(name);
    expect(tensor.nelem() == N,
           name + " length " + std::to_string(tensor.nelem()) +
           " != " + std::to_string(N));
    Vec<N> result;
    dequant_to_f32(tensor.type, tensor.data, result.begin(), N);
    return result;
}

} // namespace tensor_loader
