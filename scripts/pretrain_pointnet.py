"""Pre-train the traversability PointNet on a NaVAE dataset (paper Sec. III-A2).

Labels come from each sample's ground-truth semantic map; points higher than
--height-factor x --robot-height above the ground are ignored. Use the resulting
checkpoint as ``model.pointnet_weights`` when training NaVAE.

Example:
    python scripts/pretrain_pointnet.py --robot-height 0.6 --sensor-height 0.7 \\
        -c configs/train/navae.yaml data.root=/data
    python scripts/train.py -c configs/train/navae.yaml data.root=/data \\
        model.pointnet_weights=results/navae/pointnet/pointnet_best.pth
"""

from __future__ import annotations

import argparse
import logging

from sem_navae.config import load_config
from sem_navae.training.pointnet_pretrain import PointNetPretrainConfig, pretrain_pointnet
from sem_navae.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-c", "--config", default=None, help="NaVAE YAML config (data, model)")
    parser.add_argument("--robot-height", type=float, required=True, help="[m]")
    parser.add_argument(
        "--sensor-height", type=float, required=True, help="LiDAR origin above ground [m]"
    )
    parser.add_argument("--height-factor", type=float, default=1.5)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", default=None, help="default <output_dir>/<name>/pointnet")
    parser.add_argument("overrides", nargs="*", help="key.sub=value config overrides")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)
    pcfg = PointNetPretrainConfig(
        robot_height=args.robot_height,
        sensor_height=args.sensor_height,
        height_factor=args.height_factor,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    path = pretrain_pointnet(cfg, pcfg, args.out_dir)
    print(f"best checkpoint: {path}")


if __name__ == "__main__":
    main()
