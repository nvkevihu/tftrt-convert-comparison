"""
Use TREx to plot engine layers.
"""

import argparse

import trex

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot engine layers."
    )
    parser.add_argument(
        "--plan_path",
        type=str,
        help="Path to the plan-graph JSON file from trtexec."
    )
    return parser.parse_args()

def main(args):
    plan = trex.EnginePlan(args.plan_path)
    graph = trex.to_dot(plan, trex.layer_type_formatter, display_regions=True, expand_layer_details=True)
    trex.render_dot(graph, plan.name, 'svg')

if __name__ == "__main__":
    args = parse_args()
    main(args)
