#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/compiler/tf2tensorrt/trt_convert_api.h"

using tensorflow::Flag;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;

#define TFTRT_ENSURE_OK(x)                                                 \
  do {                                                                     \
    Status s = x;                                                          \
    if (!s.ok()) {                                                         \
      std::cerr << __FILE__ << ":" << __LINE__ << " " << s.error_message() \
                << std::endl;                                              \
      return 1;                                                            \
    }                                                                      \
  } while (0)

Status LoadGraph(const string& graph_file_name,
                 tensorflow::GraphDef* graph_def) {
  TF_RETURN_IF_ERROR(
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, graph_def));
  return Status::OK();
}

Status LoadTensorNames(const string& tensors_file_name,
                       std::vector<string>& tensors) {
  std::ifstream tensor_file(tensors_file_name);
  string line;
  if (!tensor_file.is_open() || !getline(tensor_file, line)) {
    return Status(tensorflow::errors::Code::INTERNAL, "Error reading file");
  }
  std::stringstream ss(line);
  std::string tensor;
  while (getline(ss, tensor, ',')) {
    tensors.push_back(tensor);
  }
  return Status::OK();
}

int main(int argc, char* argv[]) {
  // Parse arguments
  string graph_path = "/path/to/graph_def/";
  string inputs_path = "/path/to/inputs/file/";
  string outputs_path = "/path/to/outputs/file/";
  string out_dir = "";
  string engine_out_dir = "";
  std::vector<Flag> flag_list = {
      Flag("graph_path", &graph_path, "frozen GraphDef to convert"),
      Flag("inputs_path", &inputs_path, "path to text file with comma-separated input tensor names"),
      Flag("outputs_path", &outputs_path, "path to text file with comma-separated output tensor names"),
      Flag("out_dir", &out_dir, "the directory to store the converted GraphDef"),
      Flag("engine_out_dir", &engine_out_dir, "the directory to store the serialized TRT Engines"),
  };
  string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << usage;
    return -1;
  }

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return -1;
  }

  // Setup session
  tensorflow::GraphDef graph;
  TFTRT_ENSURE_OK(LoadGraph(graph_path, &graph));
  std::vector<string> input_names;
  std::vector<string> output_names;
  TFTRT_ENSURE_OK(LoadTensorNames(inputs_path, input_names));
  TFTRT_ENSURE_OK(LoadTensorNames(outputs_path, output_names));

  // Convert Args
  tensorflow::tensorrt::TfTrtConversionParams params;
  params.convert_to_static_engine = true;
  params.allow_build_at_runtime = true;
  params.max_workspace_size_bytes = 1 << 30;
  params.max_cached_engines = 1;
  params.minimum_segment_size = 3;
  params.precision_mode = tensorflow::tensorrt::TrtPrecisionMode::FP32;
  params.use_calibration = false;
  params.use_dynamic_shape = false;
  
  // Setup Inputs - change this depending on the model being run
  // TODO: Make this not hardcoded
  tensorflow::TensorShape input_shape = { 64, 128 };
  Tensor input(tensorflow::DataType::DT_INT32, input_shape);
  std::fill_n((float*)input.data(), input.AllocatedBytes() / 4, 1.0f);

  // Run conversion and build
  tensorflow::GraphDef converted_graph_def;
  std::vector<std::vector<Tensor>> inputs = {{ input, input }};
  tensorflow::StatusOr<tensorflow::GraphDef> status_or_gdef;
  status_or_gdef = tensorflow::tensorrt::ConvertAndBuild(
      graph, input_names, output_names, inputs,
      params);
  TFTRT_ENSURE_OK(status_or_gdef.status());
  converted_graph_def = status_or_gdef.ValueOrDie();
  
  // Serialize converted GraphDef
  TFTRT_ENSURE_OK(tensorflow::Env::Default()->RecursivelyCreateDir(out_dir));
  string converted_graph_serialized;
  converted_graph_def.SerializeToString(&converted_graph_serialized);
  std::ofstream ofs(out_dir + "/converted_graph_cpp.pb");
  ofs << converted_graph_serialized;
  ofs.close();

  // Export Engines
  TFTRT_ENSURE_OK(tensorflow::Env::Default()->RecursivelyCreateDir(engine_out_dir));
  for (auto& n : *(converted_graph_def.mutable_node())) {
    if (!n.op().compare("TRTEngineOp")) {
      LOG(INFO) << "Writing " << n.name() << " to " << engine_out_dir;
      std::ofstream ofs(engine_out_dir + "/" + n.name());
      auto* attrs = n.mutable_attr();
      ofs << (*attrs)["serialized_segment"].s();
      ofs.close();
    }
  }

  return 0;
}
