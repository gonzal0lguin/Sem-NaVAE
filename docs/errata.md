# Errata

Corrections and clarifications to *Sem-NaVAE: Semantically-Guided Outdoor Mapless
Navigation via Generative Trajectory Priors* (IEEE RA-L), found while preparing this code
release. Item 3 describes the training configuration in the original repository
([gonzal0lguin/NaVAE](https://github.com/gonzal0lguin/NaVAE)), with which the reported
results were obtained.

1. **Eq. 3, KL direction.** The ELBO term is D_KL(q_ϕ(z|x, y) ‖ p_θ(z|x)), not
   D_KL(p_θ ‖ q_ϕ) as printed. The training code computes KL(q‖p), and so does this release.
2. **Eq. 5** is written as `log Σ GK(M_sem(p))`, which is −∞ for a collision-free
   trajectory. This release uses `log(1 + Σ …)`.
3. **Training hyper-parameters (Sec. IV-A).** The training configuration in the original
   repository uses N_z = 256, Adam with learning rate 1e-4, λ = 1 and a decoder
   acceleration limit of 0.9 m/s², instead of N_z = 512, AdamW with 1e-3, λ = 10 and
   0.5 m/s². The original decoder required N_z to equal the context size (256), so
   N_z = 512 could not run. This release uses the values stated in the paper.
