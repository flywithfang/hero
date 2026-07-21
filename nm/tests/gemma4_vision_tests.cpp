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
    rope.apply(MutVecView<64>{value.begin()}, 0, 0);
    check(value[0] == 1.f && value[16] == 2.f && value[32] == 3.f && value[48] == 4.f,
          "position zero is identity on both axes");
    rope.apply(MutVecView<64>{value.begin()}, 1, 0);
    check(std::fabs(value[32] - 3.f) < 1e-6f && std::fabs(value[48] - 4.f) < 1e-6f,
          "x position rotates only the first half of each head");

    std::printf("== learned 2-D positions ==\n");
    std::vector<Scalar> raw(2 * 3 * 2);
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = Scalar(i);
    PositionTable2D<2, 3> table(raw.data(), std::make_shared<int>(1));
    Vec<2> position = table.at(1, 2);
    check(position[0] == raw[2] + raw[10] && position[1] == raw[3] + raw[11],
          "x and y lookup tables are summed");

    std::printf("\n%s (%d failures)\n", failures ? "VISION TESTS FAILED" : "ALL VISION TESTS PASSED", failures);
    return failures ? 1 : 0;
}
