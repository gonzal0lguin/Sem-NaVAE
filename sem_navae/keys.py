"""String keys of the sample / batch dictionaries passed between dataset, model and loss."""


class Keys:
    # --- inputs (produced by the dataset) ---
    past_traj = "previous_path"  # (B, N_v, 4) past [x, y, vx, vy] in the robot frame
    lidar = "lidar"  # (B, N_l, N_p, 4) normalised [x, y, z, intensity]
    goal = "goal"  # (B, 2) normalised polar goal [rho, theta]
    path = "path"  # (B, N_w, 4) ground-truth future [x, y, vx, vy]
    all_paths = "all_paths"  # list[B] of list[M_b] tensors (N_w, 4): every valid GT
    semantic_map = "semantic_map"  # (B, H, W) ground-truth semantic map (1 = non-traversable)
    pose = "pose"  # (B, 3) robot pose [x, y, yaw] in the dataset world frame
    last_poses = "last_poses"  # (B, 4) last GT waypoint
    sample_id = "sample_id"  # (B, 1) numeric id of the sample (validation only)

    # --- model outputs ---
    y_hat = "y_hat"
    y_hat_q = "y_hat_q"
    p_dist = "p_dist"
    q_dist = "q_dist"
    heatmap = "heatmap"
    prior_mean_paths = "prior_mean_paths"
    post_mean_paths = "post_mean_paths"
    nll = "nll"
    kl = "kl"
    ll_prior = "ll_prior"
    ll_posterior = "ll_posterior"

    # --- loss outputs ---
    loss = "loss"
    kl_loss = "vae_kld_loss"
    loglikelihood = "loglikelihood"
    collision_loss = "collision_loss_mean"
    kl_weight = "kl_weight"
    ade = "ade_best"
    fde = "fde_best"
