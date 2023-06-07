file(COPY ${CMAKE_CURRENT_LIST_DIR}/../../python DESTINATION ${CMAKE_BINARY_DIR}/dist)

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/dist/python/npu_fx_compiler)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../fx_compiler npu_fx_compiler)

add_custom_target(npu_fx_compiler_st ALL
        COMMAND cd ${CMAKE_BINARY_DIR}/dist/python/ && ${PYTHON_BIN_PATH} -m unittest npu_fx_compiler_st
        DEPENDS _npu_fx_compiler
        VERBATIM)
