# nm — uniform pure-C++ CPU transformer inference engine.
#
# Configure + build (Ninja):
#   cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ctest --test-dir build            # self-contained test suite
#
# Products: build/chat, build/gguf_dump, build/logit_diff, and the test binaries.
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
add_executable(chat       chat.cpp)
add_executable(gguf_dump  tools/gguf_dump.cpp)
add_executable(logit_diff tools/logit_diff.cpp)
add_executable(gemma4_check tools/gemma4_check.cpp)
add_executable(gemma4_text tools/gemma4_text.cpp)
add_executable(gemma4_vision_check tools/gemma4_vision_check.cpp)
add_executable(gemma4_image tools/gemma4_image.cpp)
foreach(t chat gguf_dump logit_diff gemma4_check gemma4_text gemma4_vision_check gemma4_image)
  target_link_libraries(${t} PRIVATE nm_core)
endforeach()

# ---- tests (self-contained; no external deps) ----
enable_testing()
add_executable(property_tests tests/property_tests.cpp)
add_executable(quant_tests    tests/quant_tests.cpp)
add_executable(e2e_stories    tests/e2e_stories.cpp)
add_executable(tokenizer_test tests/tokenizer_test.cpp)
add_executable(multimodal_tests tests/multimodal_tests.cpp)
add_executable(gemma4_tests tests/gemma4_tests.cpp)
add_executable(gemma4_tokenizer_test tests/gemma4_tokenizer_test.cpp)
add_executable(gemma4_vision_tests tests/gemma4_vision_tests.cpp)
add_executable(chat_session_tests tests/chat_session_tests.cpp)
foreach(t property_tests quant_tests e2e_stories tokenizer_test multimodal_tests gemma4_tests gemma4_tokenizer_test gemma4_vision_tests chat_session_tests)
  target_link_libraries(${t} PRIVATE nm_core)
endforeach()

add_test(NAME property_tests COMMAND property_tests)
add_test(NAME quant_tests    COMMAND quant_tests)
add_test(NAME multimodal_tests COMMAND multimodal_tests)
add_test(NAME gemma4_tests COMMAND gemma4_tests)
add_test(NAME gemma4_vision_tests COMMAND gemma4_vision_tests)
add_test(NAME chat_session_tests COMMAND chat_session_tests
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/four-colors.ppm")

# Fixture-dependent tests run only when the model/vocab files are present.
set(NM_STORIES "$ENV{HOME}/.unsloth/.cache/stories260K.gguf"
    CACHE FILEPATH "tiny Llama-arch GGUF for the e2e test")
set(NM_VOCAB "$ENV{HOME}/.unsloth/llama.cpp/models/ggml-vocab-llama-bpe.gguf"
    CACHE FILEPATH "llama-bpe vocab GGUF for the tokenizer test")
set(NM_GEMMA4 "$ENV{NM_GEMMA4}"
    CACHE FILEPATH "Gemma 4 E4B GGUF for tokenizer/model parity tests")
if(EXISTS "${NM_STORIES}")
  add_test(NAME e2e_stories COMMAND e2e_stories "${NM_STORIES}")
endif()
if(EXISTS "${NM_VOCAB}")
  add_test(NAME tokenizer_roundtrip
    COMMAND ${CMAKE_COMMAND}
      -DEXE=$<TARGET_FILE:tokenizer_test> -DVOCAB=${NM_VOCAB}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_tokenizer_roundtrip.cmake)
endif()
if(EXISTS "${NM_GEMMA4}")
  add_test(NAME gemma4_tokenizer_parity COMMAND gemma4_tokenizer_test "${NM_GEMMA4}")
endif()
