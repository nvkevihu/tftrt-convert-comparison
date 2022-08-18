cmake_minimum_required(VERSION 3.13)
project(TFTRT_CPP_Examples)

#-------------------------------------------------------------
# Configuration
#-------------------------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(TF_INSTALL_DIR "/usr/local" CACHE PATH "Path to Tensorflow install directory")
set(TRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu" CACHE PATH "Path to third party deps of TF-TRT")
set(CUDA_HOME "/usr/local/cuda" CACHE PATH "Path to CUDA install directory")

#-------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------
find_library(tf_shared_lib2 NAMES libtensorflow_cc.so.2 HINTS ${TF_INSTALL_DIR}/lib/tensorflow)
find_library(tf_framework_shared_lib2 NAMES libtensorflow_framework.so.2 HINTS ${TF_INSTALL_DIR}/lib/python3.8/dist-packages/tensorflow)
find_path(trt_include_path NAME NvInfer.h HINTS ${TRT_INCLUDE_DIR})
get_filename_component(tf_dir ${tf_shared_lib2} DIRECTORY)
get_filename_component(tf_python_dir ${tf_framework_shared_lib2} DIRECTORY)

set(tf_shared_lib ${tf_dir}/libtensorflow_cc.so)
set(tf_framework_shared_lib ${tf_python_dir}/libtensorflow_framework.so)

add_custom_command(OUTPUT ${tf_framework_shared_lib} ${tf_shared_lib}
  COMMAND ln -s ${tf_shared_lib2} ${tf_shared_lib}
  COMMAND ln -s ${tf_framework_shared_lib2} ${tf_framework_shared_lib}
  COMMENT "Generating legacy symbolic link")

set(tftrt_third_party_includes ${tf_python_dir}/include/third_party/tensorrt)

add_custom_command(OUTPUT ${tftrt_third_party_includes}
  COMMAND ln -s ${trt_include_path} ${tftrt_third_party_includes}
  COMMENT "Generating TFTRT third party deps symbolic link")

add_custom_target(tf_symlinks DEPENDS ${tf_framework_shared_lib} ${tf_shared_lib} ${tftrt_third_party_includes})

#-----------------------------------------------------------
# Converter Targets
#-----------------------------------------------------------
add_executable(tftrt_converter converter.cc)

target_link_libraries(tftrt_converter tensorflow_cc)
target_link_libraries(tftrt_converter tensorflow_framework)

target_compile_options(tftrt_converter PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1 -DGOOGLE_CUDA -DGOOGLE_TENSORRT)

target_link_directories(tftrt_converter PRIVATE ${tf_python_dir})
target_link_directories(tftrt_converter PRIVATE ${tf_dir})

target_compile_options(tftrt_converter PRIVATE -O2 -Wl,-rpath=${tf_python_dir})

target_include_directories(tftrt_converter PRIVATE ${tf_python_dir}/include)
target_include_directories(tftrt_converter PRIVATE ${CUDA_HOME}/include)

add_dependencies(tftrt_converter tf_symlinks)

#-----------------------------------------------------------
# E2E Converter Targets
#-----------------------------------------------------------
add_executable(tftrt_e2e_converter e2e_converter.cc)

target_link_libraries(tftrt_e2e_converter tensorflow_cc)
target_link_libraries(tftrt_e2e_converter tensorflow_framework)

target_compile_options(tftrt_e2e_converter PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1 -DGOOGLE_CUDA -DGOOGLE_TENSORRT)

target_link_directories(tftrt_e2e_converter PRIVATE ${tf_python_dir})
target_link_directories(tftrt_e2e_converter PRIVATE ${tf_dir})

target_compile_options(tftrt_e2e_converter PRIVATE -O2 -Wl,-rpath=${tf_python_dir})

target_include_directories(tftrt_e2e_converter PRIVATE ${tf_python_dir}/include)
target_include_directories(tftrt_e2e_converter PRIVATE ${CUDA_HOME}/include)

add_dependencies(tftrt_e2e_converter tf_symlinks)

#-----------------------------------------------------------
# Freeze Example Targets
#-----------------------------------------------------------
add_executable(tf_freezer freeze.cc)

target_link_libraries(tf_freezer tensorflow_cc)
target_link_libraries(tf_freezer tensorflow_framework)

target_compile_options(tf_freezer PRIVATE -D_GLIBCXX_USE_CXX11_ABI=1 -DGOOGLE_CUDA -DGOOGLE_TENSORRT)

target_link_directories(tf_freezer PRIVATE ${tf_python_dir})
target_link_directories(tf_freezer PRIVATE ${tf_dir})

target_compile_options(tf_freezer PRIVATE -O2 -Wl,-rpath=${tf_python_dir})

target_include_directories(tf_freezer PRIVATE ${tf_python_dir}/include)

add_dependencies(tf_freezer tf_symlinks)
