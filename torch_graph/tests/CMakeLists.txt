cmake_minimum_required(VERSION 3.14)
project(AscendAdapter2)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "-D_FORTIFY_SOURCE=2 -O2 -DNDEBUG -Wno-class-memaccess -Werror -Wno-float-equal -Wextra -Wno-unused-parameter -Wno-ignored-qualifiers -Wno-comment -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,--no-as-needed -Wl,-z,relro,-z,now,-z,noexecstack -s -fno-common -pipe -fno-strict-aliasing -Wdate-time -Wformat=2 -Wno-shadow -Wno-undef -Wunused -Wstrict-prototypes ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-D_FORTIFY_SOURCE=2 -O2 -DNDEBUG -Werror -Wno-class-memaccess -Wno-float-equal -Wextra -Wno-unused-parameter -Wno-ignored-qualifiers -Wno-comment -Wno-deprecated-declarations -Wall -fPIC -fstack-protector-all -Wl,--no-as-needed -Wl,-z,relro,-z,now,-z,noexecstack -s -fno-common -pipe -fno-strict-aliasing -Wdate-time -Wformat=2 -Wno-shadow -Wno-undef -Wunused -Wdelete-non-virtual-dtor -Wnon-virtual-dtor -Wno-overloaded-virtual ${CMAKE_CXX_FLAGS}")
set(CMAKE_SKIP_RPATH TRUE)

IF (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    add_definitions(-Wno-builtin-macro-redefined)
ENDIF ()

if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/tools/COMPILE_FLAGS OR NOT EXISTS
        ${CMAKE_CURRENT_LIST_DIR}/tools/TORCH_INSTALLED_PATH OR NOT EXISTS
        ${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_BIN_PATH)
    message(FATAL_ERROR "No validate configuration found. Did you forget to configure first?")
endif ()

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/TORCH_INSTALLED_PATH" TORCH_INSTALLED_PATH)
file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/PYTHON_BIN_PATH" PYTHON_BIN_PATH)

file(STRINGS "${CMAKE_CURRENT_LIST_DIR}/tools/COMPILE_FLAGS" CUSTOM_COMPILE_FLAGS)
foreach (COMPILE_FLAG ${CUSTOM_COMPILE_FLAGS})
    set(CMAKE_C_FLAGS "${COMPILE_FLAG} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${COMPILE_FLAG} ${CMAKE_CXX_FLAGS}")
endforeach (COMPILE_FLAG)

include_directories(${CMAKE_CURRENT_LIST_DIR}/stub/include)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/nlohmann_json.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/acl/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/pytorch/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/graph_engine/module.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/secure_c.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/aoe/module.cmake)

file(COPY ${CMAKE_CURRENT_LIST_DIR}/python DESTINATION ${CMAKE_BINARY_DIR}/dist)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/dist/python/npu_fx_compiler)

add_subdirectory(st)