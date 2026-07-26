# nm — uniform pure-C++ CPU transformer inference engine.
#
# Configure + build (Ninja):
#   cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ctest --test-dir build            # self-contained test suite
#
# Products: build/chat, build/gguf_dump, the gemma4_* inspection tools, and
# the test binaries.
cmake_minimum_required(VERSION 3.20)
project(nm LANGUAGES CXX)

# Default to Ninja when the user didn't pass a generator on a fresh tree.
if(NOT CMAKE_GENERATOR)
  set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# -O3 + -march=native (matches the Makefile; NEON kernels select on __ARM_NEON).
add_compile_options(-Wall -Wextra)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" NM_HAS_MARCH_NATIVE)
if(NM_HAS_MARCH_NATIVE)
  add_compile_options($<$<CONFIG:Release>:-march=native>)
endif()

find_package(Threads REQUIRED)

# ---- engine library (strata 0-4 + gguf + tokenizer) ----
add_library(nm_core STATIC
  src/core.cpp
  src/gguf.cpp
  src/tokenizer.cpp)
if(APPLE)
  target_sources(nm_core PRIVATE src/image_io_apple.cpp)
  target_link_libraries(nm_core PUBLIC
    "-framework ImageIO" "-framework CoreGraphics" "-framework CoreFoundation")
endif()
target_include_directories(nm_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nm_core PUBLIC Threads::Threads)

# ---- binaries ----
add_executable(chat       tools/chat.cpp)
add_executable(gguf_dump  tools/gguf_dump.cpp)
add_executable(gemma4_check tools/gemma4_check.cpp)
add_executable(qwen35_check tools/qwen35_check.cpp)
add_executable(gemma4_text tools/gemma4_text.cpp)
add_executable(gemma4_vision_check tools/gemma4_vision_check.cpp)
add_executable(gemma4_image tools/gemma4_image.cpp)
foreach(t chat gguf_dump gemma4_check qwen35_check gemma4_text gemma4_vision_check gemma4_image)
  target_link_libraries(${t} PRIVATE nm_core)
endforeach()

# ---- tests (self-contained; no external deps) ----
enable_testing()
add_executable(property_tests tests/property_tests.cpp)
add_executable(quant_tests    tests/quant_tests.cpp)
add_executable(tokenizer_test tests/tokenizer_test.cpp)
add_executable(multimodal_tests tests/multimodal_tests.cpp)
add_executable(gemma4_tests tests/gemma4_tests.cpp)
add_executable(qwen35_tests tests/qwen35_tests.cpp)
add_executable(gemma4_tokenizer_test tests/gemma4_tokenizer_test.cpp)
add_executable(gemma4_vision_tests tests/gemma4_vision_tests.cpp)
add_executable(chat_session_tests tests/chat_session_tests.cpp)
foreach(t property_tests quant_tests tokenizer_test multimodal_tests gemma4_tests qwen35_tests gemma4_tokenizer_test gemma4_vision_tests chat_session_tests)
  target_link_libraries(${t} PRIVATE nm_core)
endforeach()

add_test(NAME property_tests COMMAND property_tests)
add_test(NAME quant_tests    COMMAND quant_tests)
add_test(NAME multimodal_tests COMMAND multimodal_tests)
add_test(NAME gemma4_tests COMMAND gemma4_tests)
add_test(NAME qwen35_tests COMMAND qwen35_tests)
add_test(NAME gemma4_vision_tests COMMAND gemma4_vision_tests)
add_test(NAME chat_session_tests COMMAND chat_session_tests
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/four-colors.ppm")

# Fixture-dependent tests run only when the vocab/model files are present.
# The vocab GGUFs ship with llama.cpp; the full checkpoints do not.
set(NM_VOCAB_GEMMA4 "$ENV{HOME}/.unsloth/llama.cpp/models/ggml-vocab-gemma-4.gguf"
    CACHE FILEPATH "Gemma 4 vocab GGUF for the tokenizer round-trip test")
set(NM_VOCAB_QWEN35 "$ENV{HOME}/.unsloth/llama.cpp/models/ggml-vocab-qwen35.gguf"
    CACHE FILEPATH "Qwen 3.5 vocab GGUF for the tokenizer round-trip test")
set(NM_GEMMA4 "$ENV{NM_GEMMA4}"
    CACHE FILEPATH "Gemma 4 E4B GGUF for tokenizer/model parity tests")
foreach(vocab gemma4 qwen35)
  string(TOUPPER ${vocab} VOCAB_UPPER)
  if(EXISTS "${NM_VOCAB_${VOCAB_UPPER}}")
    add_test(NAME tokenizer_roundtrip_${vocab}
      COMMAND ${CMAKE_COMMAND}
        -DEXE=$<TARGET_FILE:tokenizer_test> -DVOCAB=${NM_VOCAB_${VOCAB_UPPER}}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_tokenizer_roundtrip.cmake)
  endif()
endforeach()
if(EXISTS "${NM_GEMMA4}")
  add_test(NAME gemma4_tokenizer_parity COMMAND gemma4_tokenizer_test "${NM_GEMMA4}")
endif()
