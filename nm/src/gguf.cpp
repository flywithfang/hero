// gguf.cpp — GGUF v3 parsing. Layout (little-endian throughout):
//   "GGUF" magic | u32 version | u64 tensor_count | u64 metadata_kv_count
//   metadata_kv_count x { string key ; u32 value_type ; value }
//   tensor_count    x { string name ; u32 n_dims ; u64 dims[n_dims] ;
//                       u32 ggml_type ; u64 offset }
//   padding to `general.alignment` (default 32)
//   tensor data (each tensor at data_start + offset)
#include "gguf.hpp"
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

// A cursor that bounds-checks every read against the mmap end.
struct Cursor {
    const uint8_t* p;
    const uint8_t* end;
    void need(size_t n) const {
        if (size_t(end - p) < n) throw std::runtime_error("GGUF: truncated file");
    }
    template <class T>
    T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string read_str() {
        uint64_t n = read<uint64_t>();
        need(n);
        std::string s((const char*)p, (const char*)p + n);
        p += n;
        return s;
    }
};

GGUFType u32_to_gguftype(uint32_t v) {
    if (v > uint32_t(GGUFType::FLOAT64)) throw std::runtime_error("GGUF: bad value type");
    return GGUFType(v);
}

double read_num(Cursor& c, GGUFType t) {
    switch (t) {
        case GGUFType::UINT8:
            return c.read<uint8_t>();
        case GGUFType::INT8:
            return c.read<int8_t>();
        case GGUFType::UINT16:
            return c.read<uint16_t>();
        case GGUFType::INT16:
            return c.read<int16_t>();
        case GGUFType::UINT32:
            return c.read<uint32_t>();
        case GGUFType::INT32:
            return c.read<int32_t>();
        case GGUFType::FLOAT32:
            return c.read<float>();
        case GGUFType::BOOL:
            return c.read<uint8_t>() != 0;
        case GGUFType::UINT64:
            return double(c.read<uint64_t>());
        case GGUFType::INT64:
            return double(c.read<int64_t>());
        case GGUFType::FLOAT64:
            return c.read<double>();
        default:
            throw std::runtime_error("GGUF: non-scalar in read_num");
    }
}

GGUFValue read_value(Cursor& c) {
    GGUFValue v;
    v.type = u32_to_gguftype(c.read<uint32_t>());
    if (v.type == GGUFType::STRING) {
        v.str = c.read_str();
        return v;
    }
    if (v.type == GGUFType::ARRAY) {
        v.elem_type = u32_to_gguftype(c.read<uint32_t>());
        uint64_t n = c.read<uint64_t>();
        if (v.elem_type == GGUFType::STRING) {
            v.arr_str.reserve(n);
            for (uint64_t i = 0; i < n; ++i) v.arr_str.push_back(c.read_str());
        } else {
            v.arr_num.reserve(n);
            for (uint64_t i = 0; i < n; ++i) v.arr_num.push_back(read_num(c, v.elem_type));
        }
        return v;
    }
    v.num = read_num(c, v.type);
    return v;
}

GT u32_to_gt(uint32_t v) {
    switch (v) {
        case 0:
            return GT::F32;
        case 1:
            return GT::F16;
        case 2:
            return GT::Q4_0;
        case 8:
            return GT::Q8_0;
        case 12:
            return GT::Q4_K;
        case 13:
            return GT::Q5_K;
        case 14:
            return GT::Q6_K;
        case 30:
            return GT::BF16;
        default:
            throw std::runtime_error("GGUF: unsupported ggml_type " + std::to_string(v) + " (this engine reads F32/F16/BF16/Q4_0/Q8_0/Q4_K/Q5_K/Q6_K)");
    }
}

}  // namespace

GGUF::GGUF(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("GGUF: cannot open " + path);
    struct stat st{};
    if (::fstat(fd_, &st) != 0) {
        ::close(fd_);
        throw std::runtime_error("GGUF: fstat failed");
    }
    size_ = size_t(st.st_size);
    map_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (map_ == MAP_FAILED) {
        ::close(fd_);
        throw std::runtime_error("GGUF: mmap failed");
    }

    // keepalive: munmap + close when the last Weight view drops.
    void* mp = map_;
    size_t sz = size_;
    int fd = fd_;
    holder_ = std::shared_ptr<const void>(map_, [mp, sz, fd](const void*) {
        ::munmap(mp, sz);
        ::close(fd);
    });

    Cursor c{(const uint8_t*)map_, (const uint8_t*)map_ + size_};
    uint32_t magic = c.read<uint32_t>();
    if (magic != 0x46554747u)  // 'G''G''U''F' little-endian
        throw std::runtime_error("GGUF: bad magic (not a GGUF file)");
    version_ = c.read<uint32_t>();
    if (version_ != 3) throw std::runtime_error("GGUF: unsupported version " + std::to_string(version_) + " (expect 3)");
    uint64_t tensor_count = c.read<uint64_t>();
    uint64_t kv_count = c.read<uint64_t>();

    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = c.read_str();
        meta_[key] = read_value(c);
    }
    alignment_ = has("general.alignment") ? uint64_t(get("general.alignment").as_int()) : 32;
    if (alignment_ == 0) alignment_ = 32;
    if ((alignment_ & (alignment_ - 1)) != 0) throw std::runtime_error("GGUF: general.alignment is not a power of two");

    tensors_.reserve(tensor_count);
    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo t;
        t.name = c.read_str();
        uint32_t nd = c.read<uint32_t>();
        t.dims.resize(nd);
        for (uint32_t d = 0; d < nd; ++d) t.dims[d] = c.read<uint64_t>();
        t.type = u32_to_gt(c.read<uint32_t>());
        t.offset = c.read<uint64_t>();
        tensors_.push_back(std::move(t));
    }

    // data section starts at the aligned end of the header (trap T8).
    uint64_t header_end = uint64_t(c.p - (const uint8_t*)map_);
    const uint64_t padding = alignment_ - 1;
    if (header_end > std::numeric_limits<uint64_t>::max() - padding) throw std::runtime_error("GGUF: aligned header offset overflows");
    uint64_t data_start = (header_end + padding) & ~padding;
    if (!tensors_.empty() && data_start > size_) throw std::runtime_error("GGUF: data section starts past end of file");

    for (auto& t : tensors_) {
        if (t.offset % alignment_ != 0) throw std::runtime_error("GGUF: tensor '" + t.name + "' offset is misaligned");
        if (t.offset > uint64_t(size_) - data_start) throw std::runtime_error("GGUF: tensor '" + t.name + "' data out of range");
        const uint64_t abs = data_start + t.offset;
        const uint64_t bytes = t.nbytes();
        if (bytes > uint64_t(size_) - abs) throw std::runtime_error("GGUF: tensor '" + t.name + "' data out of range");
        t.data = (const uint8_t*)map_ + abs;
    }
}

GGUF::~GGUF() { /* mmap/fd freed by holder_'s deleter */ }

const GGUFValue& GGUF::get(const std::string& k) const {
    auto it = meta_.find(k);
    if (it == meta_.end()) throw std::runtime_error("GGUF: missing metadata key '" + k + "'");
    return it->second;
}
int64_t GGUF::get_int(const std::string& k, int64_t def) const {
    auto it = meta_.find(k);
    return it == meta_.end() ? def : it->second.as_int();
}
double GGUF::get_float(const std::string& k, double def) const {
    auto it = meta_.find(k);
    return it == meta_.end() ? def : it->second.as_num();
}
std::string GGUF::get_str(const std::string& k, const std::string& def) const {
    auto it = meta_.find(k);
    return it == meta_.end() ? def : it->second.str;
}
const TensorInfo* GGUF::find(const std::string& name) const {
    for (auto& t : tensors_)
        if (t.name == name) return &t;
    return nullptr;
}
const TensorInfo& GGUF::require(const std::string& name) const {
    const TensorInfo* t = find(name);
    if (!t) throw std::runtime_error("GGUF: required tensor '" + name + "' not found");
    return *t;
}
