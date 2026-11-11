"""Training loop."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from sem_navae.config import Config, save_config
from sem_navae.keys import Keys
from sem_navae.losses import NaVAELoss
from sem_navae.metrics import min_ade_fde
from sem_navae.models import NaVAE
from sem_navae.utils.io import get_device, release_cuda, to_device

log = logging.getLogger(__name__)


class _Logger:
    def __init__(self, kind: str, log_dir: str):
        self.writer = None
        if kind == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=log_dir)

    def scalars(self, prefix: str, values: dict, step: int) -> None:
        if self.writer is not None:
            for k, v in values.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


class Trainer:
    def __init__(self, cfg: Config, train_loader, val_loader):
        self.cfg = cfg
        self.tcfg = cfg.training
        self.device = get_device(cfg.device)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.model = NaVAE(cfg.model).to(self.device)
        self.loss = NaVAELoss(cfg.loss, cfg.data)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=self.tcfg.lr, weight_decay=self.tcfg.weight_decay
        )
        # Exponential decay applied every ``lr_decay_every`` epochs (paper Sec. IV-A).
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, self.tcfg.lr_decay_every, gamma=self.tcfg.lr_decay
        )
        self.epoch = 0
        self.iteration = 0
        self.out_dir = os.path.join(self.tcfg.output_dir, cfg.name)
        os.makedirs(self.out_dir, exist_ok=True)
        save_config(cfg, os.path.join(self.out_dir, "config.yaml"))
        self.logger = _Logger(self.tcfg.logger, os.path.join(self.out_dir, "logs"))

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "config": self.cfg.to_dict(),
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "iteration": self.iteration,
            },
            path,
        )
        log.info("Saved checkpoint to %s", path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt["epoch"]
        self.iteration = ckpt["iteration"]
        log.info("Resumed from %s (epoch %d)", path, self.epoch)

    # ------------------------------------------------------------------ loops
    def train_epoch(self) -> None:
        self.model.train()
        max_iter = self.tcfg.max_iterations_per_epoch
        for i, batch in enumerate(tqdm(self.train_loader, desc=f"epoch {self.epoch}")):
            if max_iter is not None and i >= max_iter:
                break
            self.iteration += 1
            batch = to_device(batch, self.device)
            out = self.model(batch, num_prior_samples=self.cfg.loss.num_prior_samples)
            losses = self.loss(out, self.epoch)
            self.optimizer.zero_grad(set_to_none=True)
            losses[Keys.loss].backward()
            if self.tcfg.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.tcfg.grad_clip_norm)
            self.optimizer.step()
            self.logger.scalars("train", release_cuda(losses), self.iteration)
        self.scheduler.step()

    @torch.no_grad()
    def eval_epoch(self) -> dict:
        self.model.eval()
        sums, n = defaultdict(float), 0
        for batch in tqdm(self.val_loader, desc="eval"):
            batch = to_device(batch, self.device)
            out = self.model(batch, num_prior_samples=self.cfg.loss.num_prior_samples)
            stats = release_cuda(self.loss(out, self.epoch))
            samples = self.model.sample_forward(batch, num_samples=self.tcfg.eval_num_samples)
            positions = samples[Keys.y_hat].cumsum(dim=2) * self.cfg.model.dt
            stats.update(release_cuda(min_ade_fde(positions, batch[Keys.path][:, :, :2])))
            for k, v in stats.items():
                sums[k] += float(np.mean(v))
            n += 1
        means = {k: v / max(n, 1) for k, v in sums.items()}
        self.logger.scalars("eval", means, self.epoch)
        return means

    def train(self, resume: Optional[str] = None) -> None:
        if resume:
            self.load_checkpoint(resume)
        while self.epoch < self.tcfg.max_epochs:
            self.epoch += 1
            self.train_epoch()
            if self.epoch % self.tcfg.eval_every == 0:
                log.info("epoch %d eval: %s", self.epoch, self.eval_epoch())
            if self.epoch % self.tcfg.save_every == 0 or self.epoch == self.tcfg.max_epochs:
                self.save_checkpoint(os.path.join(self.out_dir, f"epoch_{self.epoch}.pt"))
                self.save_checkpoint(os.path.join(self.out_dir, "last.pt"))
        self.logger.close()
