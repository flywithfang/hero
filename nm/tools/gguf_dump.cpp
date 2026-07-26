// gguf_dump — M0 deliverable. Prints every metadata key/value and a tensor
// table (name, shape, type, bytes, offset). Mirrors gguf_dump.py / llama.cpp's
// listing so the GATE M0 diff is trivial.
//
//   gguf_dump model.gguf            full dump
//   gguf_dump model.gguf --tensors  tensor table only
//   gguf_dump model.gguf --meta     metadata only
//   gguf_dump model.gguf --values NAME [N]   first N values of one tensor
#include "../src/gguf.hpp"
#include <cstdio>
#include <string>

static const char* gguftype_name(GGUFType t) {
    switch (t) {
        case GGUFType::UINT8: return "u8";   case GGUFType::INT8: return "i8";
        case GGUFType::UINT16: return "u16"; case GGUFType::INT16: return "i16";
        case GGUFType::UINT32: return "u32"; case GGUFType::INT32: return "i32";
        case GGUFType::FLOAT32: return "f32"; case GGUFType::BOOL: return "bool";
        case GGUFType::STRING: return "str"; case GGUFType::ARRAY: return "arr";
        case GGUFType::UINT64: return "u64"; case GGUFType::INT64: return "i64";
        case GGUFType::FLOAT64: return "f64";
    }
    return "?";
}

static std::string fmt_num(GGUFType t, double v) {
    char buf[64];
    switch (t) {
        case GGUFType::FLOAT32: case GGUFType::FLOAT64:
            std::snprintf(buf, sizeof buf, "%g", v); break;
        case GGUFType::BOOL:
            return v ? "true" : "false";
        default:
            std::snprintf(buf, sizeof buf, "%lld", (long long)v); break;
    }
    return buf;
}

static void print_value(const GGUFValue& v) {
    if (v.type == GGUFType::STRING) {
        std::string s = v.str;
        if (s.size() > 80) s = s.substr(0, 77) + "...";
        // single-line
        for (auto& ch : s) if (ch == '\n') ch = ' ';
        std::printf("\"%s\"", s.c_str());
        return;
    }
    if (v.type == GGUFType::ARRAY) {
        size_t n = v.elem_type == GGUFType::STRING ? v.arr_str.size() : v.arr_num.size();
        std::printf("[%s x %zu] ", gguftype_name(v.elem_type), n);
        size_t show = std::min<size_t>(n, 6);
        std::printf("{");
        for (size_t i = 0; i < show; ++i) {
            if (i) std::printf(", ");
            if (v.elem_type == GGUFType::STRING) {
                std::string s = v.arr_str[i];
                if (s.size() > 20) s = s.substr(0, 17) + "...";
                std::printf("\"%s\"", s.c_str());
            } else std::printf("%s", fmt_num(v.elem_type, v.arr_num[i]).c_str());
        }
        if (n > show) std::printf(", ...");
        std::printf("}");
        return;
    }
    std::printf("%s", fmt_num(v.type, v.num).c_str());
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s model.gguf [--tensors|--meta|--values NAME [N]]\n", argv[0]);
        return 2;
    }
    std::string mode = argc > 2 ? argv[2] : "";
    if (mode == "--values") {
        if (argc < 4) { std::fprintf(stderr, "--values needs a tensor name\n"); return 2; }
        try {
            GGUF g(argv[1]);
            const TensorInfo& t = g.require(argv[3]);
            const size_t want = argc > 4 ? std::stoul(argv[4]) : 16;
            const size_t n = std::min(want, size_t(t.nelem()));
            std::vector<Scalar> values(t.nelem());
            dequant_to_f32(t.type, t.data, values.data(), t.nelem());
            std::printf("%s  %s  nelem=%zu\n", argv[3], gt_name(t.type),
                        size_t(t.nelem()));
            for (size_t i = 0; i < n; ++i)
                std::printf("  [%zu] %.9g\n", i, double(values[i]));
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error: %s\n", e.what());
            return 1;
        }
    }
    try {
        GGUF g(argv[1]);
        std::printf("GGUF version %u   alignment %llu   %zu tensors   %zu metadata keys\n",
                    g.version(), (unsigned long long)g.alignment(),
                    g.tensors().size(), g.meta().size());

        if (mode != "--tensors") {
            std::printf("\n== metadata ==\n");
            for (const auto& [k, v] : g.meta()) {   // std::map => sorted keys
                std::printf("  %-40s %-5s ", k.c_str(), gguftype_name(v.type));
                print_value(v);
                std::printf("\n");
            }
        }
        if (mode != "--meta") {
            std::printf("\n== tensors ==\n");
            std::printf("  %-32s %-18s %-6s %14s %14s\n", "name", "shape", "type", "bytes", "offset");
            uint64_t total = 0;
            for (const auto& t : g.tensors()) {
                std::string shape;
                for (size_t i = 0; i < t.dims.size(); ++i)
                    shape += (i ? "x" : "") + std::to_string(t.dims[i]);
                std::printf("  %-32s %-18s %-6s %14llu %14llu\n",
                            t.name.c_str(), shape.c_str(), gt_name(t.type),
                            (unsigned long long)t.nbytes(), (unsigned long long)t.offset);
                total += t.nbytes();
            }
            std::printf("  -- total tensor bytes: %.2f MB\n", double(total) / (1024.0*1024.0));
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
