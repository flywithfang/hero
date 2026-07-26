// gguf.hpp — GGUF v3 reader: mmap the file, parse the header (magic, version,
// tensor + metadata counts), the typed metadata KV store, and the tensor
// registry (name, dims, ggml type, data pointer). No copying of tensor data —
// callers get a pointer into the mmap. (INFERENCE_PLAN.md §M0.)
#pragma once
#include "quant.hpp"  // GT + type_size_bytes
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

// GGUF metadata value types (gguf.h).
enum class GGUFType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12,
};

// A metadata value. Scalars land in `scalar`; arrays keep their element type
// and either a numeric vector (widened to double) or a string vector.
struct GGUFValue {
    GGUFType type = GGUFType::UINT32;
    GGUFType elem_type = GGUFType::UINT32;  // for ARRAY
    // scalar payloads
    double num = 0;  // int/float/bool widened
    std::string str;
    // array payloads (exactly one is non-empty for ARRAY)
    std::vector<double> arr_num;
    std::vector<std::string> arr_str;

    bool is_array() const { return type == GGUFType::ARRAY; }
    double as_num() const { return num; }
    int64_t as_int() const { return int64_t(num); }
};

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims;  // ggml order: dims[0] contiguous
    GT type = GT::F32;
    uint64_t offset = 0;            // relative to data section start
    const uint8_t* data = nullptr;  // absolute pointer into mmap
    uint64_t nelem() const {
        uint64_t n = 1;
        for (uint64_t d : dims) {
            if (d != 0 && n > std::numeric_limits<uint64_t>::max() / d) throw std::runtime_error("GGUF: tensor '" + name + "' element count overflows");
            n *= d;
        }
        return n;
    }
    uint64_t nbytes() const {
        const uint64_t n = nelem();
        const uint64_t block = block_len(type);
        if (n % block != 0) throw std::runtime_error("GGUF: tensor '" + name + "' has a partial quant block");
        const uint64_t blocks = n / block;
        const uint64_t bytes_per_block = type_size_bytes(type, block);
        if (blocks != 0 && bytes_per_block > std::numeric_limits<uint64_t>::max() / blocks) throw std::runtime_error("GGUF: tensor '" + name + "' byte size overflows");
        return blocks * bytes_per_block;
    }
};

class GGUF {
public:
    explicit GGUF(const std::string& path);
    ~GGUF();
    GGUF(const GGUF&) = delete;
    GGUF& operator=(const GGUF&) = delete;

    uint32_t version() const { return version_; }
    uint64_t alignment() const { return alignment_; }

    // metadata access
    const std::map<std::string, GGUFValue>& meta() const { return meta_; }
    bool has(const std::string& k) const { return meta_.count(k) != 0; }
    const GGUFValue& get(const std::string& k) const;
    int64_t get_int(const std::string& k, int64_t def) const;
    double get_float(const std::string& k, double def) const;
    std::string get_str(const std::string& k, const std::string& def = "") const;

    // tensor access
    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo* find(const std::string& name) const;
    const TensorInfo& require(const std::string& name) const;

    // the raw mmap, kept alive so zero-copy Weight views stay valid.
    std::shared_ptr<const void> keepalive() const { return holder_; }

private:
    void* map_ = nullptr;
    size_t size_ = 0;
    int fd_ = -1;
    uint32_t version_ = 0;
    uint64_t alignment_ = 32;
    std::map<std::string, GGUFValue> meta_;
    std::vector<TensorInfo> tensors_;
    std::shared_ptr<const void> holder_;  // custom deleter munmaps
};
