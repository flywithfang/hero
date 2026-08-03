#include "../src/gemma4_vision.hpp"
#include <cstdio>

static int failures = 0;
static void check(bool condition, const char* message) {
    std::printf("  [%s] %s\n", condition ? "PASS" : "FAIL", message);
    failures += !condition;
}

int main() {
    std::printf("== Gemma 4 vision 2-D RoPE ==\n");
    Vec<64> value;
    value[0] = 1.f;
    value[16] = 2.f;
    value[32] = 3.f;
    value[48] = 4.f;
    VisionRope2D<64, 100> rope;
    rope.apply(MutVecView<64>(value), 0, 0);
    check(value[0] == 1.f && value[16] == 2.f && value[32] == 3.f && value[48] == 4.f, "position zero is identity on both axes");
    rope.apply(MutVecView<64>(value), 1, 0);
    check(std::fabs(value[32] - 3.f) < 1e-6f && std::fabs(value[48] - 4.f) < 1e-6f, "x position rotates only the first half of each head");

    std::printf("== learned 2-D positions ==\n");
    std::vector<Scalar> raw(2 * 3 * 2);
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = Scalar(i);
    PositionTable2D<2, 3> table(raw.data(), std::make_shared<int>(1));
    Vec<2> position = table.at(1, 2);
    check(position[0] == raw[2] + raw[10] && position[1] == raw[3] + raw[11], "x and y lookup tables are summed");

    std::printf("== patchify and average-pool tensor operations ==\n");
    std::vector<uint8_t> pixels{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    Matrix<12> patches = patchify_rgb<2>(pixels, 2, 2);
    check(patches.rows() == 1 && patches.row(0)[0] == -1.f && patches.row(0)[4] == 2.f * (1.f / 255.f) - 1.f && patches.row(0)[8] == 2.f * (2.f / 255.f) - 1.f, "patchify produces channel-major patch rows");

    Matrix<1> grid{{1.f}, {2.f}, {3.f}, {4.f}};
    Matrix<1> pooled = average_pool_2d<2>(grid.view(), 2, 2);
    check(pooled.rows() == 1 && std::fabs(pooled.row(0)[0] - 2.5f) < 1e-6f, "2-D average pooling preserves the grid interpretation");

    std::printf("== dense vision attention equation ==\n");
    Matrix<2> queries{{1.f, 0.f}, {0.f, 1.f}};
    Matrix<2> keys{{1.f, 0.f}, {0.f, 1.f}};
    Matrix<2> values{{10.f, 20.f}, {30.f, 40.f}};
    Matrix<2> attended = scaled_dot_product_attention<1, 2, 2>(queries.view(), keys.view(), values.view(), 1.f);
    const Scalar high = std::exp(1.f) / (std::exp(1.f) + 1.f);
    check(std::fabs(attended.row(0)[0] - (high * 10.f + (1.f - high) * 30.f)) < 1e-5f && std::fabs(attended.row(1)[1] - ((1.f - high) * 20.f + high * 40.f)) < 1e-5f, "full attention is softmax(QK^T)V");

    std::printf("\n%s (%d failures)\n", failures ? "VISION TESTS FAILED" : "ALL VISION TESTS PASSED", failures);
    return failures ? 1 : 0;
}
