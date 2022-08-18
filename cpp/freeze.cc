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
#include "tensorflow/cc/tools/freeze_saved_model.h"

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

int main(int argc, char* argv[]) {
  // Parse arguments
  string model_dir = "/path/to/saved/model";
  string out_dir = "";
  std::vector<Flag> flag_list = {
      Flag("model_dir", &model_dir, "SavedModel to freeze"),
      Flag("out_dir", &out_dir, "the directory to store the frozen GraphDef"),
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

  // Load model
  tensorflow::RunOptions run_options;
  tensorflow::SessionOptions sess_options;
  tensorflow::SavedModelBundle bundle;
  TFTRT_ENSURE_OK(tensorflow::LoadSavedModel(sess_options, run_options,
                                                model_dir, {"serve"}, &bundle));

  // Freeze model
  std::unordered_set<std::string> inputs;
  std::unordered_set<std::string> outputs;
  tensorflow::GraphDef frozen_graph;
  TFTRT_ENSURE_OK(FreezeSavedModel(bundle, &frozen_graph, &inputs, &outputs));

  // Serialize frozen GraphDef
  TFTRT_ENSURE_OK(tensorflow::Env::Default()->RecursivelyCreateDir(out_dir));
  string frozen_graph_serialized;
  frozen_graph.SerializeToString(&frozen_graph_serialized);
  std::ofstream ofs(out_dir + "/frozen_graph_cpp.pb");
  ofs << frozen_graph_serialized;
  ofs.close();

  return 0;
}
