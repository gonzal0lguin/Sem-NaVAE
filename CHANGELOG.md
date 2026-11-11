# Changelog

## [1.0.0] — 2026-11-11

First public release of the Sem-NaVAE code: a cleaned-up and refactored version of the
original research code used for the paper.
Code only: no pre-trained weights or datasets are included.

### Added
- `sem_navae`: a pip-installable, ROS-free package with the NaVAE model, losses, data
  loading, training, offline inference and trajectory selection (`sem_navae.selection`:
  costmap, projection, costs and the asynchronous update policy).
- Typed configuration with the paper's values as defaults (`configs/`). Checkpoints embed
  their config.
- Scripts: `train.py` (with resume), `pretrain_pointnet.py`, `sample.py` (offline
  sampling to `.npz`/`.png`) and `make_synthetic_dataset.py`.
- Local planner plugin interface with `move_base` and `null` built-ins
  (`docs/local_planner.md`).
- Three trajectory update strategies: `cost_to_go` (default), `paper` and `periodic`.
- Documentation: paper ↔ code map, errata, dataset format, third-party notices.
- Tests: unit tests, regression tests against the original implementation, a rostest of
  the switching decisions and a one-epoch training smoke test.

### Changed
- The model, losses, decoder constraints, heatmap and hyper-parameters follow the paper
  (`docs/paper_to_code.md`). The training objective is the same for training and
  validation, and the train/val split is deterministic and saved to disk.
- Selection: costs follow Eqs. 7–9, switching applies the ε hysteresis, and the executing
  trajectory is re-scored as new images arrive. The adopted trajectory is stored in a
  fixed frame.
- The ROS package is now `sem_navae_ros`, with topics under `/sem_navae/*`. Topics,
  frames and parameters are configurable.

### Removed
- The VLM (GPT) selection strategy, the data-collection node, the evaluation scripts and
  the unused model variants.
