# WuBuGAADHybridGen VAE-GAN (v0.1) - Comprehensive Training Guide

Welcome! This guide provides a structured approach to training the `WuBuGAADHybridGen_v0.1.py` model, a VAE-GAN hybrid designed for video generation. It leverages WuBu Nesting for hierarchical hyperbolic representations and Golden Aspect Adaptive Decomposition (GAAD) for regional processing, with an optional optical flow-based motion encoding branch.

**Current Status (Based on Recent Experiments):**
The model is showing healthy initial learning when configured with:
*   `lambda_kl = 0.0001` (very low KL divergence weight initially)
*   `learning_rate_gen = 3e-4`
*   `learning_rate_disc = 1e-4`
*   Discriminator FiLM conditioning disabled (`--disc_use_gaad_film_condition` not present or set to false).

This guide will build upon these successful starting parameters.

## Table of Contents

1.  [Prerequisites & Setup](#1-prerequisites--setup)
2.  [Understanding Key Hyperparameters](#2-understanding-key-hyperparameters)
3.  [The Phased Training Strategy](#3-the-phased-training-strategy)
    *   [Phase 1: Prioritizing Reconstruction & Basic GAN Stability](#phase-1-prioritizing-reconstruction--basic-gan-stability)
    *   [Phase 2: KL Annealing - Regularizing the Latent Space](#phase-2-kl-annealing---regularizing-the-latent-space)
    *   [Phase 3: Enhancing Discriminator with GAAD-FiLM (Optional)](#phase-3-enhancing-discriminator-with-gaad-film-optional)
    *   [Phase 4: Fine-Tuning & Long Runs](#phase-4-fine-tuning--long-runs)
4.  [Using the `.bat` Script for Phased Training](#4-using-the-bat-script-for-phased-training)
5.  [Monitoring Training with WandB](#5-monitoring-training-with-wandb)
6.  [Troubleshooting Common Issues](#6-troubleshooting-common-issues)
7.  [Advanced Experimentation Ideas](#7-advanced-experimentation-ideas)

## 1. Prerequisites & Setup

*   **Environment**: Ensure you have a Python environment set up (conda or venv) with all dependencies from `requirements.txt` installed, especially PyTorch with CUDA support. Refer to the main `README.md` for setup instructions.
*   **Data**:
    *   For initial tests, the script can generate a dummy video if `VIDEO_DATA_PATH` in your `.bat` file points to `demo_video_data_dir` and the video file is missing.
    *   For actual training, prepare your video dataset(s). Update `VIDEO_DATA_PATH` and optionally `VALIDATION_VIDEO_PATH` in your `.bat` script.
*   **Hardware**: A CUDA-enabled GPU is essential. Training these models is computationally intensive.
*   **WandB (Weights & Biases)**: Highly recommended for experiment tracking. Ensure you have an account and are logged in (`wandb login`). The script will automatically use it if the `--wandb` flag is present.

## 2. Understanding Key Hyperparameters

The `.bat` script (e.g., `run_wubu_GANVAE_HYBRIDv.0.1.bat`) is your primary interface for setting these.

*   **VAE Loss Weights**:
    *   `LAMBDA_RECON`: Weight for the VAE's pixel-wise reconstruction loss (MSE). *Default in successful runs: `10.0`*
    *   `LAMBDA_KL`: Weight for the KL divergence between the encoder's latent distribution and the prior (standard normal). *Crucial for phased training. Start very low: `0.0001`*
    *   `LAMBDA_GAN`: Weight for the generator's adversarial loss component (how well it fools the discriminator). *Default: `1.0`*
*   **Learning Rates**:
    *   `LEARNING_RATE_GEN`: Base learning rate for the generator and encoder. *Successful start: `3e-4`*
    *   `LEARNING_RATE_DISC`: Base learning rate for the discriminator. *Successful start: `1e-4`*
*   **Q-Controller**:
    *   `Q_CONTROLLER_ENABLED=true`: Activates the HAKMEM-inspired Q-learning agent to dynamically adjust LR and momentum scaling factors for both optimizers based on training dynamics. Highly recommended.
*   **Discriminator Configuration**:
    *   `DISC_USE_GAAD_FILM_CONDITION`: Boolean (`true`/`false`). Enables/disables FiLM conditioning in the discriminator based on GAAD regions. *Start with `false`.*
    *   `DISC_APPLY_SPECTRAL_NORM`: Boolean. Applies spectral normalization to discriminator layers for stability. *Keep `true`.*
*   **Architectural (WuBu, GAAD, etc.)**:
    *   Refer to `argparse` in the Python script for the many parameters controlling WuBu levels, dimensions, curvatures, GAAD regions, etc. The defaults in the `.bat` file are a good starting point based on current experiments.
*   **Checkpointing**:
    *   `LOAD_CHECKPOINT`: Path to a previously saved checkpoint (`.pt` file) to resume training or start a new phase. Leave empty or comment out for a fresh run.
    *   `CHECKPOINT_OUTPUT_DIR`: Where new checkpoints will be saved.

## 3. The Phased Training Strategy

This strategy aims to stabilize training and build capabilities incrementally. **Always load the best checkpoint from the previous phase when starting a new one.**

### Phase 1: Prioritizing Reconstruction & Basic GAN Stability

**Goal**:
*   Train the VAE component (Encoder + Generator) to achieve decent reconstruction of target frames.
*   Establish a stable learning dynamic for the discriminator.

**Key `.bat` Settings for Phase 1:**
```batch
SET "LAMBDA_KL=0.0001"                  REM Critically low KL weight
SET "LEARNING_RATE_GEN=3e-4"            REM Generator LR
SET "LEARNING_RATE_DISC=1e-4"           REM Discriminator LR (lower than or equal to G)
SET "DISC_USE_GAAD_FILM_CONDITION=false" REM Keep D simple initially
SET "LOAD_CHECKPOINT="                  REM Start fresh for Phase 1
SET "EPOCHS=10"                         REM Example: Adjust based on dataset size/complexity
```

**Monitoring & Targets for Phase 1:**
*   **Primary Metric**: Training `Rec` loss should consistently decrease. Validation `PSNR` should increase, and `SSIM` should increase. `LPIPS` should decrease.
    *   *Target Example*: Aim for `Rec` loss < 0.15 (ideally < 0.10). PSNR > 15-18dB.
*   **Visuals (WandB)**: `train_recon` and `val_reconstruction_samples` should transition from noise/color-casts to blurry but recognizable images. The strong purple/green hues observed in initial failed runs should diminish significantly.
*   **Discriminator Losses**: `D_tot` should stabilize, ideally in the 0.3 - 0.65 range. `D_real` and `D_fake` should both be reasonably low (e.g., < 0.7, ideally < 0.6).
*   **Generator Adversarial Loss (`Adv`)**: Will likely be high (e.g., > 0.7) as the discriminator learns effectively. This is expected.
*   **Q-Controller**: Ensure it's active and epsilon is decaying. Observe chosen LR scales.

**When to Proceed**: When reconstruction quality (both metric-wise and visually) shows significant improvement and starts to plateau. This might take 5-20+ epochs. The logs ending Epoch 4 (PSNR ~14.04, Rec ~0.15-0.18) show good progress *within* this phase. Continue until PSNR is higher and `Rec` is lower.

---

### Phase 2: KL Annealing - Regularizing the Latent Space

**Goal**: Gradually increase `lambda_kl` to encourage a more structured latent space, improving the potential for generating diverse and coherent novel samples, while trying to maintain reconstruction quality.

**Procedure**: This is typically done in sub-steps.

**Phase 2a: Small KL Increase**
*   **`.bat` Settings Changes from Phase 1 Best Checkpoint**:
    ```batch
    SET "LAMBDA_KL=0.001"  REM Increased 10x
    SET "LOAD_CHECKPOINT=path\to\best_checkpoint_from_Phase1.pt"
    SET "EPOCHS=5-10"      REM Run for several epochs
    REM Keep other LRs, etc., the same initially, let Q-controller adapt.
    ```
*   **Monitoring**:
    *   `KL` (training loss) should now start to decrease more significantly.
    *   `Rec` loss might slightly increase or become more volatile. Monitor it closely; it should not degrade catastrophically.
    *   `G_tot` will likely increase due to the higher `lambda_kl` contribution.
    *   Validation PSNR/SSIM: Aim to maintain or only slightly dip before recovering.
    *   Visuals: Reconstructions should remain coherent.

**Phase 2b: Moderate KL Increase**
*   If Phase 2a is stable and reconstruction quality is acceptable:
*   **`.bat` Settings Changes**:
    ```batch
    SET "LAMBDA_KL=0.01"   REM Increased 10x again
    SET "LOAD_CHECKPOINT=path\to\best_checkpoint_from_Phase2a.pt"
    SET "EPOCHS=5-10"
    ```
*   **Monitoring**: Similar to Phase 2a. The balance between reconstruction and KL regularization becomes more critical.

**Phase 2c (Optional): Target KL Value**
*   If Phase 2b is stable:
*   **`.bat` Settings Changes**:
    ```batch
    SET "LAMBDA_KL=0.1"    REM Your original target `lambda_kl` from args
    SET "LOAD_CHECKPOINT=path\to\best_checkpoint_from_Phase2b.pt"
    SET "EPOCHS=10+"
    ```
*   **Monitoring**: At this stage, the VAE should have a reasonably good balance. If `Rec` loss degrades too much, your target `lambda_kl` might be too high for the model's current capacity, or LRs might need adjustment (which the Q-controller should attempt).

---

### Phase 3: Enhancing Discriminator with GAAD-FiLM (Optional)

**Goal**: If regional details or consistency in generated samples are lacking, enabling FiLM conditioning in the discriminator *might* help by making it more sensitive to regional properties.

**When to Consider**: After KL annealing (Phase 2) has progressed, reconstruction is good (e.g., PSNR > 20-22dB), and the basic GAN is stable.

**`.bat` Script Settings Changes:**
```batch
SET "LAMBDA_KL=..." REM Use the lambda_kl value from your best Phase 2 checkpoint
SET "LEARNING_RATE_GEN=..." REM Likely same as end of Phase 2
SET "LEARNING_RATE_DISC=..." REM Might need a slight temporary increase if D struggles to learn FiLM initially, or let Q-Ctrl handle.
SET "DISC_USE_GAAD_FILM_CONDITION=true" <--- ENABLE THIS
SET "LOAD_CHECKPOINT=path\to\best_checkpoint_from_Phase2.pt"
SET "EPOCHS=10+"
```

**Monitoring & Expectations for Phase 3:**
*   **Initial D Loss Fluctuation**: `D_tot` might increase or become unstable as it learns to incorporate FiLM.
*   **`G_adv`**: Might increase if D becomes a stronger, more regionally-aware critic.
*   **Visuals**: Look for improved alignment of generated content with implicit GAAD regions or fewer regional artifacts. This is a subtle effect and might require careful visual inspection of many samples.

---

### Phase 4: Fine-Tuning & Long Runs

**Goal**: Achieve the best possible overall sample quality, diversity, and training stability.

**`.bat` Script Settings:**
*   Load the best checkpoint from the most successful previous phase.
*   **Learning Rates**: Primarily let the Q-controller manage them. If training becomes imbalanced (e.g., D always wins, G_adv sky-high with no recovery), you might consider a small manual adjustment to the *base* `LEARNING_RATE_GEN` or `LEARNING_RATE_DISC` in the `.bat` file to shift the Q-controller's operating range.
*   **`LAMBDA_GAN`**: If reconstructions are excellent but fakes still lack realism, you *could* experiment with slightly increasing `LAMBDA_GAN` (e.g., to 1.5 or 2.0) to give more weight to the adversarial signal for the generator. This is an advanced tuning step.
*   **`EPOCHS`**: Set for a longer duration for convergence.

**Monitoring & Targets for Phase 4:**
*   Stable `G_tot` and `D_tot` losses (they will still oscillate, but hopefully around a stable mean).
*   High and stable validation metrics (PSNR, SSIM, LPIPS).
*   Visually appealing and diverse generated samples (check `fixed_noise_generated` in WandB).

## 4. Using the `.bat` Script for Phased Training

1.  **Copy & Rename**: Make a copy of your main `.bat` script for each phase (e.g., `run_phase1.bat`, `run_phase2a.bat`).
2.  **Edit Settings**: Modify the relevant `SET "PARAM=VALUE"` lines in each phase's script.
3.  **Set `LOAD_CHECKPOINT`**: For Phase 2 onwards, update `SET "LOAD_CHECKPOINT=..."` to point to the `.pt` file from the *end* of the previous successful phase (usually the `..._best.pt` or the last epoch's checkpoint).
4.  **Run**: Execute the `.bat` script for the current phase.

## 5. Monitoring Training with WandB

If `WANDB_ENABLED=true` in your `.bat` script:
*   A link to the WandB run will be printed in the console.
*   **Key Plots to Watch**:
    *   `train/loss_recon`, `train/loss_kl`, `train/loss_g_adv`, `train/loss_g_total`
    *   `train/loss_d_real`, `train/loss_d_fake`, `train/loss_d_total`
    *   `train/lr_gen`, `train/lr_disc` (to see actual LRs after Q-controller)
    *   `q_ctrl_gen/lrscale`, `q_ctrl_disc/lrscale` (and momentumscale) to see Q-controller actions.
    *   `val/avg_val_psnr`, `val/avg_val_ssim`, `val/avg_val_lpips`, `val/avg_val_recon_mse`
    *   **Image Samples**: `samples/train_recon`, `samples/val_reconstruction_samples`, `samples/fixed_noise_generated`. These are often the most telling indicators of progress or problems.

## 6. Troubleshooting Common Issues

*   **Output is Random Noise / Color Casts (Initial)**:
    *   **Primary Suspect**: `LAMBDA_KL` is too high. Reduce it drastically (e.g., to `0.0001` or `0.00001`) as done in Phase 1.
    *   **Secondary**: Check input to Generator's final `Tanh` activation (see previous discussions on adding hooks if `Rec` improves but colors are still off).
*   **Losses Explode (NaN/Inf)**:
    *   Reduce learning rates (both base LRs in `.bat` and potentially narrow `lr_scale_options` for Q-controller).
    *   Ensure `GLOBAL_MAX_GRAD_NORM` is active and at a reasonable value (e.g., 1.0-5.0).
    *   Check for issues in WuBu layers if using very aggressive curvatures or if inputs to hyperbolic functions are problematic.
    *   Enable `--detect_anomaly` for one run to pinpoint the operation causing NaNs (will be very slow).
*   **Discriminator Loss Stays at ~0.69 (Guessing)**:
    *   Generator is likely producing poor fakes. Focus on improving G's `Rec` loss first (low `LAMBDA_KL`, adequate `LEARNING_RATE_GEN`).
    *   Ensure D's LR isn't too high. The current `1e-4` for D seems to work well once G starts improving.
*   **Discriminator Loss Goes to Zero (and Stays There)**:
    *   D has become too powerful, and G cannot fool it at all (G_adv will be very high).
    *   The Q-controller for D *should* try to reduce D's LR.
    *   If not, manually reduce `LEARNING_RATE_DISC` or increase `LEARNING_RATE_GEN`. Consider reducing `LAMBDA_GAN` temporarily if G needs to focus more on reconstruction.
*   **Generator Loss (Adv) Goes to Zero (and Stays There)**:
    *   G is perfectly fooling D, meaning D has collapsed or isn't learning.
    *   The Q-controller for G *should* reduce G's LR. The D Q-controller should try to increase D's LR.
    *   If not, manually increase `LEARNING_RATE_DISC` or reduce `LEARNING_RATE_GEN`.

## 7. Advanced Experimentation Ideas

Once the basic training is stable and yielding decent results:
*   Experiment with different WuBu stack configurations (levels, dims, curvatures, rotations) for S, M, T.
*   Explore different GAAD parameters (`gaad_num_regions`, decomposition types) and their impact when D-FiLM is active.
*   Tune Q-Controller `reward_weights` for more specific behaviors.
*   Try different optical flow models for the motion branch.
*   If you have multiple GPUs, explore DDP training by setting `NPROC_PER_NODE` in the `.bat` script.

This guide provides a roadmap. The key is methodical experimentation, careful observation of metrics and visuals, and leveraging checkpoints to iterate through different training phases. Good luck!
