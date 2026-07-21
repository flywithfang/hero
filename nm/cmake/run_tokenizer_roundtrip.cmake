# Runs tokenizer_test in --roundtrip mode over a small corpus piped on stdin,
# and fails the test if any line does not round-trip. Invoked by ctest with
# -DEXE=<tokenizer_test> -DVOCAB=<vocab.gguf>.
set(corpus "${CMAKE_CURRENT_LIST_DIR}/tokenizer_corpus.txt")
execute_process(
  COMMAND "${EXE}" "${VOCAB}" --roundtrip
  INPUT_FILE "${corpus}"
  RESULT_VARIABLE rc
  ERROR_VARIABLE err)
message(STATUS "${err}")
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "tokenizer roundtrip failed (rc=${rc})")
endif()
if(err MATCHES "ROUNDTRIP FAIL")
  message(FATAL_ERROR "tokenizer roundtrip mismatch")
endif()
