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
#include "tensorflow/cc/saved_model/loader.h"
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

// Returns info for nodes listed in the signature definition.
std::vector<tensorflow::TensorInfo> GetNodeInfo(
    const google::protobuf::Map<string, tensorflow::TensorInfo>& signature) {
  std::vector<tensorflow::TensorInfo> info;
  for (const auto& item : signature) {
    info.push_back(item.second);
  }
  return info;
}

// Load the `SavedModel` located at `model_dir`.
Status LoadModel(const string& model_dir, const string& signature_key,
                 tensorflow::SavedModelBundle* bundle,
                 std::vector<tensorflow::TensorInfo>* input_info,
                 std::vector<tensorflow::TensorInfo>* output_info) {
  tensorflow::RunOptions run_options;
  tensorflow::SessionOptions sess_options;

  tensorflow::OptimizerOptions* optimizer_options =
      sess_options.config.mutable_graph_options()->mutable_optimizer_options();
  optimizer_options->set_opt_level(tensorflow::OptimizerOptions::L0);
  optimizer_options->set_global_jit_level(tensorflow::OptimizerOptions::OFF);

  sess_options.config.mutable_gpu_options()->force_gpu_compatible();
  TF_RETURN_IF_ERROR(tensorflow::LoadSavedModel(sess_options, run_options,
                                                model_dir, {"serve"}, bundle));

  // Get input and output names
  auto signature_map = bundle->GetSignatures();
  const tensorflow::SignatureDef& signature = signature_map[signature_key];
  *input_info = GetNodeInfo(signature.inputs());
  *output_info = GetNodeInfo(signature.outputs());

  return Status::OK();
}

// Create arbitrary inputs matching `input_info`.
Status SetupInputs(std::vector<tensorflow::TensorInfo>& input_info,
                   std::vector<Tensor>* inputs) {
  std::vector<Tensor> inputs_host;
  for (auto& info : input_info) {
    // Set input batch size
    auto* shape = info.mutable_tensor_shape();
    shape->mutable_dim(0)->set_size(64);

    for (size_t i = 1; i < shape->dim_size(); i++) {
      auto* dim = shape->mutable_dim(i);
      if (dim->size() < 0) {
	dim->set_size(128);
      }
    }

    // Allocate memory and fill host tensor
    Tensor input_host(info.dtype(), *shape);
    std::fill_n((uint8_t*)input_host.data(), input_host.AllocatedBytes(), 1);
    inputs_host.push_back(input_host);
  }
  *inputs = inputs_host;
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
  string model_dir = "/path/to/saved/model/";
  string signature_key = "serving_default";
  string out_dir = "";
  string engine_out_dir = "";
  std::vector<Flag> flag_list = {
      Flag("model_dir", &model_dir, "SavedModel to convert"),
      Flag("signature_key", &signature_key, "the serving signature to use"),
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
  tensorflow::SavedModelBundle bundle;
  std::vector<tensorflow::TensorInfo> input_info;
  std::vector<tensorflow::TensorInfo> output_info;
  TFTRT_ENSURE_OK(
      LoadModel(model_dir, signature_key, &bundle, &input_info, &output_info));

  // Create inputs
  std::vector<Tensor> inputs;
  TFTRT_ENSURE_OK(SetupInputs(input_info, &inputs));

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
  
  // Run conversion and build
  tensorflow::GraphDef converted_graph_def;
  tensorflow::StatusOr<tensorflow::GraphDef> status_or_gdef;
  status_or_gdef = tensorflow::tensorrt::ConvertAndBuild(
      &bundle, signature_key, { inputs },
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
