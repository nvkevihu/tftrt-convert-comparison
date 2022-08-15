"""
Converts a SavedModel and serializes
the resulting converted GraphDef to disk.
"""

import argparse

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a SavedModel."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the SavedModel."
    )
    parser.add_argument(
        "--signature_key",
        type=str,
        default=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
        help="SavedModel signature to use."
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default=tag_constants.SERVING,
        help="SavedModel inference tag to use."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to serialize the converted GraphDef."
    )
    return parser.parse_args()

def main(args):
    # Convert Args
    trt_converter_params = dict(
        allow_build_at_runtime=True,
        input_saved_model_dir=args.model_path,
        input_saved_model_signature_key=args.signature_key,
        input_saved_model_tags=[args.model_tag],
        max_workspace_size_bytes=(1 << 30),
        maximum_cached_engines=1,
        minimum_segment_size=3,
        precision_mode=trt.TrtPrecisionMode.FP32,
        use_calibration=False,
        use_dynamic_shape=False,
    )

    # Convert
    converter = trt.TrtGraphConverterV2(**trt_converter_params)
    converted_func = converter.convert()
    converted_graph_def = converted_func.graph.as_graph_def()

    # Serialize
    tf.io.write_graph(
        converted_graph_def,
        args.output_path,
        'converted_graph_py.pb',
        as_text=False
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
