# Dataset format

No dataset is distributed with this release. Training expects the format below;
`scripts/make_synthetic_dataset.py` writes a small random dataset in this format (for smoke
tests only, it carries no navigation signal), and `sem_navae/data/synthetic.py` is a
compact reference implementation of a writer.

## Layout

```
<data.root>/
├── index.pkl            # {"root": [maps_dir, samples_dir], "ids": [(sample_file, map_file), ...]}
├── <samples_dir>/       # one pickle per observation
└── <maps_dir>/          # optional; not read by the loader
```

`map_file` must be named `<integer>.<3-letter extension>`; the integer becomes the
validation `sample_id`. The train/val split is made over `sample_file` names and written to
`split.json` (see `data.split_file`). Consecutive observations of one run are highly
correlated; for honest validation numbers, give each run's samples a common prefix and split
by run (edit `split.json`, which is reused once it exists).

## Sample pickle

A `dict` with NumPy arrays. **Every quantity is in the robot frame at the observation time**
(x forward, y left, origin at the robot, metres and m/s):

| Key | Shape | Content |
|-----|-------|---------|
| `previous_path` | `(≥ N_v, 4)` | past states `[x, y, vx, vy]`, oldest first, sampled every 0.5 s; the last `model.past_traj_len` rows are used. The last row is the current state (position ≈ 0). |
| `path` | `(≥ N_w, 4)` | main ground-truth future `[x, y, vx, vy]`, waypoints `model.dt` apart (first `N_w` used) |
| `all_paths` | list of `(N_w, 4)` | every valid ground truth for this observation (Eq. 4); optional, defaults to `[path]` |
| `goal` | `(2,)` | goal position `[x, y]` |
| `lidar_dn` | `(N_l, N_p, 4)` | `[x, y, z, intensity]` of the last `N_l` scans, already voxelised to `N_p` points (zero-padded). Alternatively `lidar`: list of raw `(n, 4)` scans, voxelised by the loader. Scans are not motion-compensated. An optional fifth column holds per-point traversability labels (0/1) for `scripts/pretrain_pointnet.py`; without it, labels are derived from `local_map`. |
| `local_map` | `(H, W)` | ground-truth semantic map, `1` = non-traversable, `0` = traversable, centred on the robot. **Axis convention: `local_map[i, j]` with `i = x / res + H/2`, `j = y / res + W/2`** (rows follow x). `res = data.semantic_map_resolution` (0.1 m/px); the loader crops `2 · data.semantic_map_half_size_px` (600 px = 60 m) around the centre. Used by the collision loss and to label points for PointNet pre-training. |
| `pose` | `(3,)` | robot pose `[x, y, yaw]` in the dataset's world frame (kept for analysis, not a model input) |

## What the loader does

- `previous_path` velocities are divided by `model.max_vel` (2 m/s); positions stay in metres.
- `goal` becomes `[‖goal‖ / data.goal_norm_dist, atan2(y, x) / π]` (`goal_norm_dist` = 300 m).
- LiDAR xyz are divided by `model.lidar_norm_dist` (50 m) and intensity is standardised over
  the stacked scans.

The ROS generator node builds the same inputs online (`ros/sem_navae_ros/nodes/generator_node.py`,
`sem_navae.data.transforms.past_trajectory`).
