"""
Freezes a SavedModel and writes the frozen
GraphDef to disk.
"""

import argparse
import os

import tensorflow as tf
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

def parse_args():
    parser = argparse.ArgumentParser(
        description="Freeze a SavedModel."
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
        help="Path to serialize the frozen GraphDef."
    )
    return parser.parse_args()

def main(args):
    # Load SavedModel
    model = tf.saved_model.load(
        export_dir=args.model_path,
        tags=[args.model_tag]
    ).signatures[args.signature_key]
    
    # Freeze
    frozen_model = convert_to_constants.convert_variables_to_constants_v2(model)
    frozen_graph_def = frozen_model.graph.as_graph_def()

    # Serialize
    tf.io.write_graph(
        frozen_graph_def,
        args.output_path,
        'frozen_graph.pb',
        as_text=False
    )

    # Write input / output names
    with open(os.path.join(args.output_path, 'inputs'), 'w') as f:
        inputs = ','.join([tensor.name for tensor in frozen_model.inputs])
        f.write(inputs)
    with open(os.path.join(args.output_path, 'outputs'), 'w') as f:
        outputs = ','.join([tensor.name for tensor in frozen_model.outputs])
        f.write(outputs)

if __name__ == "__main__":
    args = parse_args()
    main(args)
