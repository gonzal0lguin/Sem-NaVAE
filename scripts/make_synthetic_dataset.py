"""Write a tiny random dataset in the training format (smoke tests / CI only).

Example:
    python scripts/make_synthetic_dataset.py data/synthetic --num-samples 16
    python scripts/train.py data.root=data/synthetic training.max_epochs=1 device=cpu
"""

from __future__ import annotations

import argparse

from sem_navae.config import load_config
from sem_navae.data.synthetic import make_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", help="output directory")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-c", "--config", default=None, help="config whose shapes to follow")
    args = parser.parse_args()
    index = make_synthetic_dataset(args.root, args.num_samples, load_config(args.config), args.seed)
    print(f"wrote {args.num_samples} samples, index {index}")


if __name__ == "__main__":
    main()
