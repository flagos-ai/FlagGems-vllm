cmake_minimum_required(VERSION 3.26)
project(flaggems_m3_ascend_bridge LANGUAGES CXX)
find_package(Python REQUIRED COMPONENTS Interpreter)
find_package(Torch REQUIRED)
if(NOT ASCEND_HOME_PATH)
  set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
if(NOT EXISTS "${ASCEND_HOME_PATH}/include/acl/acl.h")
  message(FATAL_ERROR "ASCEND_HOME_PATH must select the active CANN toolkit")
endif()
execute_process(COMMAND ${Python_EXECUTABLE} -c
  "import pathlib,torch_npu; print(pathlib.Path(torch_npu.__file__).resolve().parent)"
  OUTPUT_VARIABLE TORCH_NPU_PATH OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE QUERY_RESULT)
if(NOT QUERY_RESULT EQUAL 0)
  message(FATAL_ERROR "torch_npu is required")
endif()
get_filename_component(NATIVE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
add_library(_flaggems_m3_ascend SHARED torch_binding.cpp
  ${NATIVE_ROOT}/aclnn_torch_adapter/NPUBridge.cpp
  ${NATIVE_ROOT}/aclnn_torch_adapter/NPUStorageImpl.cpp)
target_compile_features(_flaggems_m3_ascend PRIVATE cxx_std_17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_include_directories(_flaggems_m3_ascend PRIVATE ${NATIVE_ROOT}
  ${TORCH_INCLUDE_DIRS} ${TORCH_NPU_PATH}/include ${ASCEND_HOME_PATH}/include)
target_link_directories(_flaggems_m3_ascend PRIVATE
  ${TORCH_NPU_PATH}/lib ${ASCEND_HOME_PATH}/lib64)
target_link_libraries(_flaggems_m3_ascend PRIVATE ${TORCH_LIBRARIES}
  torch_npu ascendcl dl opapi)
set_target_properties(_flaggems_m3_ascend PROPERTIES PREFIX ""
  CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
  BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
install(TARGETS _flaggems_m3_ascend LIBRARY DESTINATION .)
