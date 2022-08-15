"""
Compares the nodes of two GraphDefs.
"""

import argparse

import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two GraphDefs."
    )
    parser.add_argument(
        "--first_graph_path",
        type=str,
        help="Path to the first GraphDef."
    )
    parser.add_argument(
        "--second_graph_path",
        type=str,
        help="Path to the second GraphDef."
    )
    return parser.parse_args()

def load(graph_path):
    with open(graph_path, 'rb') as f:
        graph = tf.compat.v1.GraphDef()
        graph.ParseFromString(f.read())
    return graph

def as_dict(g, include_attrs=True):
    def node_dict(n):
        d = {
            'op': n.op,
            'input': n.input,
            'device': n.device
        }
        if include_attrs:
            d['attr'] = dict(n.attr)
        return d
    return { n.name: node_dict(n) for n in g.node }

def print_graph_nodes(graph, include_attrs=False):
    print(as_dict(graph, include_attrs=include_attrs))

def compare(g1, g2):
    def compare_dict(d1, d2, d1_name="d1", d2_name="d2"):
        """ Returns True if the two dictionaries match. """
        match = True
        d1_keys, d2_keys = set(d1.keys()), set(d2.keys())
        shared = d1_keys.intersection(d2_keys)
        d1_only = d1_keys - shared
        if len(d1_only) > 0:
            print("Only {} contains the following keys: {}".format(d1_name, d1_only))
            match &= False
        d2_only = d2_keys - shared
        if len(d2_only) > 0:
            print("Only {} contains the following keys: {}".format(d2_name, d2_only))
            match &= False
        for k in shared:
            if type(d1[k]) is not type(d2[k]):
                print("Types do not match for key {} between dicts {} and {}".format(k, d1_name, d2_name))
                match &= False
                continue
            if isinstance(d1[k], dict) and not compare_dict(
                    d1[k], d2[k],
                    d1_name="{}[{}]".format(d1_name, k),
                    d2_name="{}[{}]".format(d2_name, k)):
                match &= False
                continue
            if not isinstance(d1[k], dict) and d1[k] != d2[k]:
                print("Values do not match for key {} between dicts {} and {}".format(k, d1_name, d2_name))
                if k != 'serialized_segment':
                    print(d1[k], d2[k])
                match &= False
                continue
        return match
    g1_dict, g2_dict = as_dict(g1), as_dict(g2)
    if compare_dict(g1_dict, g2_dict, d1_name="graph1", d2_name="graph2"):
        print("GraphDefs match")

def main(args):
    first_graph = load(args.first_graph_path)
    second_graph = load(args.second_graph_path)
    compare(first_graph, second_graph)

if __name__ == "__main__":
    args = parse_args()
    main(args)
