"""CLI: print top-k category log-probabilities for a name."""

from __future__ import annotations

import argparse
from pathlib import Path

from .inference import predict_cli
from .paths import REPO_ROOT


def main() -> None:
    default_weights = REPO_ROOT / "weights.pt"
    p = argparse.ArgumentParser(description="Classify a name with a trained checkpoint")
    p.add_argument(
        "-w",
        "--weights",
        type=Path,
        default=default_weights if default_weights.exists() else Path("./weights.pt"),
        help="Checkpoint from name_classification.train",
    )
    p.add_argument(
        "-n",
        "--top-n",
        type=int,
        default=3,
        help="Number of top labels to show",
    )
    p.add_argument("name", type=str, help="Name to classify")
    args = p.parse_args()

    preds = predict_cli(args.name, args.top_n, args.weights)
    for pr in preds:
        print(f"({pr.log_probability:.2f}) {pr.label}")


if __name__ == "__main__":
    main()
