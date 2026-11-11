# Paper ↔ code

Where each part of *Sem-NaVAE* (IEEE RA-L) is implemented, and the choices the code makes
where the paper leaves details open. Mistakes in the paper itself are in
[errata.md](errata.md).

## Equations

| Eq. | What | Code |
|-----|------|------|
| 1 | K samples ŷ⁽ᵏ⁾ ∼ p_ψ(y \| x) | `NaVAE.sample_forward` (`sem_navae/models/navae.py`), `Generator.sample` (`sem_navae/inference.py`) |
| 2 | Traversability heatmap I_seg = G_σ ∗ (M⁺ − M⁻ + 1) | `traversability_heatmap`, `gaussian_heatmap`, `metric_to_pixel` (`sem_navae/models/heatmap.py`); spec `HeatmapSpec` (`sem_navae/config.py`) |
| 3 | ELBO: prior and posterior reconstruction + β·KL | `NaVAE.forward` (terms), `NaVAELoss.forward` (`sem_navae/losses.py`), `kl_diag_gaussians` |
| 4 | L_recon: log-mean-exp over the M ground truths for the prior, plain expectation for the posterior | `prior_log_likelihood` and the posterior term in `NaVAE.forward` |
| 5 | L_col = log Σ GK(M_sem(p)) | `NaVAELoss.collision_loss` |
| 6 | Pitch between consecutive waypoints on the elevation map, θ_i > θ_max ⇒ collision | `ros/sem_navae_ros/src/elevation_collision_filter.cpp` |
| 7 | Cost image I_c(u, v) = C[argmax_i S_i(u, v)] | `SemanticCostmap.from_scores` (`sem_navae/selection/costmap.py`), CLIPSeg in `ClipSegCostmapper` |
| 8 | Discounted semantic cost J(y′) with C_u for occluded / unseen waypoints | `sample_waypoint_costs`, `discount`, `semantic_cost` (`sem_navae/selection/costs.py`), projection in `project_to_image` (`projection.py`) |
| 9 | J_goal = α₁ log(1 + d) + α₂ \|θ_end\|/π | `goal_cost` (`costs.py`), `get_path_dist_to_goal` (`sem_navae/utils/geometry.py`) |
| 10 | y* = argmin J(y′) + J_goal(y), switch only if better by ε (Sec. III-B4) | `SelectionCore.on_candidates` (`selector.py`), `UpdatePolicy` (`update_policy.py`) |

## Hyper-parameters (Sec. IV-A)

All defaults are the paper's values: `configs/train/navae.yaml` (generated from
`sem_navae/config.py`) and `configs/selection/default.yaml`.

| Paper | Config key | Value |
|-------|-----------|-------|
| N_l, N_p | `model.lidar_num_scans`, `model.lidar_max_points` | 3, 2560 |
| N_v (5 s at 0.5 s) | `model.past_traj_len` | 10 |
| N_w, Δt | `model.num_waypoints`, `model.dt` | 12, 1 s |
| velocity normalisation | `model.max_vel` | 2 m/s |
| feature size, N_z | `model.feature_dim`, `model.latent_dim` | 256, 512 |
| I_seg: size, δ, σ | `model.heatmap.{size_m, resolution, sigma_px}` | 18 m, 0.2 m/px, 5 |
| acceleration limit | `model.max_accel` | 0.5 m/s² |
| λ | `loss.collision_weight` | 10 |
| optimiser | `training.{lr, lr_decay, lr_decay_every}` | AdamW, 1e-3, exponential every 10 epochs |
| epochs | `training.max_epochs` | 1000 |
| K | `model/n_paths` (ROS) | 200 |
| classes and costs, C_u, T_occ, γ, α₁, α₂, ε | `selection.*` | see `configs/selection/default.yaml` |
| f_gen, f_clip | `model/trajectory_gen_period`, `selection/clip_rate_hz` (ROS) | 0.5 Hz (2 s), 2.5 Hz |

## Choices the paper leaves open

- **Eq. 5 offset.** `log Σ` is −∞ for a collision-free path. The code uses
  `log(1 + Σ_p GK(M_sem)(p))`: finite, zero for free paths, same gradient direction.
  GK is a Gaussian with `loss.collision_sigma_px` = 4 px on the 0.1 m/px semantic map (not
  given in the paper), sampled with bilinear interpolation so the loss is differentiable in
  the waypoint positions.
- **Which trajectories L_col sees.** The means of all `loss.num_prior_samples` prior samples
  of each observation (the paper says "trajectories").
- **Prior samples in Eq. 4.** Eq. 4 has one z_p. The code averages the likelihood over
  `loss.num_prior_samples` (default 6, as in the original code) prior samples inside the log
  (IWAE-style, as the text describes); `1` gives Eq. 4 exactly.
- **Decoder constraints (Sec. III-A3).** Besides the acceleration limit and the heatmap, the
  rejection sampler also limits speed (`model.min_speed`, `model.max_speed`) and the heading
  change per step (`model.max_heading_change_deg` = 45°, the angle between consecutive
  velocities). `model.num_proposals` (20,000) proposals are drawn per step; the first valid
  one is kept, the mean if none is valid. The heatmap test is `I_seg > 0` on the clipped
  `M⁺ − M⁻`, i.e. `> 1` with Eq. 2's +1.
- **Free bits.** `loss.free_bits` floors each observation's KL. Off by default.
- **Frames.** Every model input and output is in the robot frame at the current time; see
  [dataset_format.md](dataset_format.md).
