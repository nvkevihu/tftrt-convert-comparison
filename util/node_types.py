"""
Finds nodes of given types in a SavedModel.
"""

import argparse

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

def parse_args():
    parser = argparse.ArgumentParser(
        description="Find certain nodes in a SavedModel."
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
    return parser.parse_args()

def find(graph, node_types):
    for n in graph.node:
        if n.op in node_types:
            print("Found {} of type {}".format(n.name, n.op)) 

def main(args):
    # Load SavedModel
    saved_model = tf.saved_model.load(
        export_dir=args.model_path,
        tags=[args.model_tag]
    )
    model = saved_model.signatures[args.signature_key]
    model._backref_to_saved_model = saved_model

    graph = model.graph.as_graph_def()

    node_types = ["Placeholder", "Case", "Merge", "PartitionedCall", "StatefulPartitionedCall", "ReadVariableOp", "ResourceGather", "ResourceGatherNd", "If", "StatelessIf", "While", "StatelessWhile", "Enter", "Exit", "Identity", "NextIteration", "Switch", "_SwitchN"]
    find(graph, node_types)

if __name__ == "__main__":
    args = parse_args()
    main(args)
