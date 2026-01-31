import argparse

def parse_runtime_args():
    parser = argparse.ArgumentParser(description="Runtime arguments")

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional experiment name"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Root directory to save runs"
    )

    return parser.parse_known_args()
