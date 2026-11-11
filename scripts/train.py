"""Train NaVAE.

Example:
    python scripts/train.py -c configs/train/navae.yaml data.root=/data training.max_epochs=10
"""

from __future__ import annotations

import argparse
import logging

from sem_navae.config import load_config
from sem_navae.data.dataset import make_loaders
from sem_navae.training.trainer import Trainer
from sem_navae.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--config", default=None, help="YAML config file")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("overrides", nargs="*", help="key.sub=value overrides")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    train_loader, val_loader = make_loaders(cfg)
    Trainer(cfg, train_loader, val_loader).train(resume=args.resume)


if __name__ == "__main__":
    main()
