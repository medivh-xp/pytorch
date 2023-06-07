cmake_minimum_required(VERSION 3.14)
project(fx_compiler)

include_directories(${CMAKE_CURRENT_LIST_DIR}/core)

# file(GLOB_RECURSE PROTO_SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.pb.cc)
# add_library(proto_obj SHARED ${PROTO_SOURCES})
# target_compile_definitions(proto_obj PRIVATE google=ascend_private)
# target_link_directories(proto_obj PRIVATE /home/medivh/workspace/ascend/sdk/ai_cann_x86/opensdk/opensdk/lib)
# target_link_libraries(proto_obj PRIVATE ascend_protobuf)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.pb.cc)
foreach (CPP_SOURCE ${SOURCES})
    file(RELATIVE_PATH RELATIVE_CPP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR} ${CPP_SOURCE})
    set_property(SOURCE ${CPP_SOURCE} PROPERTY COMPILE_DEFINITIONS __FILE__=\"${RELATIVE_CPP_SOURCE}\")
endforeach (CPP_SOURCE)

add_library(_npu_fx_compiler SHARED ${SOURCES})
target_compile_definitions(_npu_fx_compiler PRIVATE google=ascend_private)
set_target_properties(_npu_fx_compiler PROPERTIES PREFIX "")
target_link_directories(_npu_fx_compiler PRIVATE /home/medivh/workspace/ascend/sdk/ai_cann_x86/opensdk/opensdk/lib)
target_link_libraries(_npu_fx_compiler PRIVATE
        torch_libs
        c_sec
        ge_libs
        acl_libs
        ascend_protobuf)
