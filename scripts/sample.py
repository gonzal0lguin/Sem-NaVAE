"""Sample K trajectories for one observation offline (no ROS, no robot).

Reads one sample pickle in the dataset format (docs/dataset_format.md) and writes
``<out>.npz`` (``trajectories`` (K, N_w, 2) in the robot frame [m], ``heatmap``) and
``<out>.png`` (heatmap with the trajectories and the goal direction drawn on it).

Example:
    python scripts/sample.py --ckpt results/navae/best.pt \\
        --sample data/samples/000000.pkl -k 50 -o sample
    # without a trained model (pipeline check only):
    python scripts/sample.py --untrained --sample data/samples/000000.pkl -o sample
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from sem_navae.config import load_config
from sem_navae.data.dataset import observation_inputs
from sem_navae.inference import Generator
from sem_navae.utils.io import read_pkl
from sem_navae.utils.seed import seed_everything


def render(trajs: np.ndarray, heatmap, goal: np.ndarray, spec, scale: int = 4) -> np.ndarray:
    """BGR image: heatmap (grey, bright = traversable), trajectories, robot and goal ray."""
    n = spec.size_px
    if heatmap is None:
        img = np.full((n, n), 128, np.uint8)
    else:
        h = heatmap - heatmap.min()
        img = (255 * h / max(float(h.max()), 1e-6)).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, (n * scale, n * scale), interpolation=cv2.INTER_NEAREST)

    def px(xy: np.ndarray) -> np.ndarray:  # column = x, row = y (as the heatmap)
        return np.round((xy / spec.resolution + spec.origin_px) * scale).astype(np.int32)

    for traj in trajs:
        pts = px(np.vstack([[0.0, 0.0], traj]))
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (40, 40, 220), 1, cv2.LINE_AA)
    centre = tuple(px(np.zeros(2)).tolist())
    direction = goal[:2] / max(np.linalg.norm(goal[:2]), 1e-6)
    end = tuple(px(direction * spec.size_m / 2).tolist())
    cv2.arrowedLine(img, centre, end, (40, 180, 40), 2, cv2.LINE_AA, tipLength=0.05)
    cv2.circle(img, centre, 2 * scale, (220, 120, 0), -1)
    return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="NaVAE checkpoint (its config is stored inside)")
    src.add_argument("--untrained", action="store_true", help="random weights from --config")
    parser.add_argument("-c", "--config", default=None, help="config for --untrained")
    parser.add_argument("--sample", required=True, help="sample pickle (dataset format)")
    parser.add_argument("-k", "--num-samples", type=int, default=50)
    parser.add_argument("-o", "--out", default="sample", help="output path without extension")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    if args.ckpt:
        gen = Generator(checkpoint=args.ckpt, device=args.device)
    else:
        gen = Generator(cfg=load_config(args.config), device=args.device)
    meta = read_pkl(args.sample)
    obs = {
        k: torch.as_tensor(v, dtype=torch.float32)[None]
        for k, v in observation_inputs(meta, gen.cfg).items()
    }
    trajs, heatmap = gen.sample(obs, num_samples=args.num_samples)

    np.savez(args.out + ".npz", trajectories=trajs, heatmap=heatmap if heatmap is not None else [])
    cv2.imwrite(
        args.out + ".png", render(trajs, heatmap, np.asarray(meta["goal"]), gen.cfg.model.heatmap)
    )
    print(f"wrote {args.out}.npz and {args.out}.png ({len(trajs)} trajectories)")


if __name__ == "__main__":
    main()
