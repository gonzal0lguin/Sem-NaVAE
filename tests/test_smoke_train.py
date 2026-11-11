"""One-epoch training run on synthetic data, then sampling from the saved checkpoint."""

import math
import os

import pytest

torch = pytest.importorskip("torch")

from sem_navae.config import Config  # noqa: E402
from sem_navae.data.dataset import make_loaders  # noqa: E402
from sem_navae.data.synthetic import make_synthetic_dataset  # noqa: E402
from sem_navae.inference import Generator  # noqa: E402
from sem_navae.keys import Keys  # noqa: E402
from sem_navae.training.trainer import Trainer  # noqa: E402
from sem_navae.utils.seed import seed_everything  # noqa: E402


def smoke_config(tmp_path) -> Config:
    cfg = Config(name="smoke", device="cpu")
    cfg.data.root = str(tmp_path / "data")
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    cfg.data.val_fraction = 0.25
    cfg.model.num_proposals = 64
    cfg.loss.num_prior_samples = 2
    t = cfg.training
    t.output_dir = str(tmp_path / "out")
    t.max_epochs, t.eval_every, t.save_every = 1, 1, 1
    t.eval_num_samples = 3
    t.logger = "none"
    return cfg


def test_one_epoch_on_synthetic_data(tmp_path):
    cfg = smoke_config(tmp_path)
    seed_everything(cfg.seed)
    make_synthetic_dataset(cfg.data.root, num_samples=8, cfg=cfg)
    train_loader, val_loader = make_loaders(cfg)
    trainer = Trainer(cfg, train_loader, val_loader)
    trainer.train()
    stats = trainer.eval_epoch()
    for key in (Keys.loss, Keys.kl, Keys.collision_loss, "min_ade", "min_fde"):
        assert math.isfinite(stats[key]), key

    out_dir = os.path.join(cfg.training.output_dir, cfg.name)
    assert os.path.exists(os.path.join(out_dir, "split.json"))
    ckpt = os.path.join(out_dir, "last.pt")
    gen = Generator(checkpoint=ckpt)
    batch = next(iter(val_loader))
    obs = {k: batch[k][:1] for k in (Keys.lidar, Keys.past_traj, Keys.goal)}
    trajs, heatmap = gen.sample(obs, num_samples=5)
    assert trajs.shape == (5, cfg.model.num_waypoints, 2)
    assert heatmap.shape == (cfg.model.heatmap.size_px,) * 2

    resumed = Trainer(cfg, train_loader, val_loader)
    resumed.load_checkpoint(ckpt)
    assert resumed.epoch == 1
