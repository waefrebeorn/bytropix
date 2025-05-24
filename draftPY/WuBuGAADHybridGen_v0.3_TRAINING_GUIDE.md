# WuBuGAADHybridGen VAE-GAN (v0.3) - Comprehensive Training Guide

Welcome! This guide details how to train `WuBuGAADHybridGen_v0.3.py`, an advanced VAE-GAN hybrid model for video generation. This version introduces:

*   **Dual Spectral Features**: Option to use both Discrete Fourier Transform (DFT) and Discrete Cosine Transform (DCT) for rich regional appearance representations.
*   **Dual Discriminator Architecture**: Employs a primary (typically pixel-based) and an alternative (feature-based, e.g., using WuBu on spectral features) discriminator.
*   **Heuristic Discriminator Switching**: Dynamically switches between the primary and alternative discriminators based on training state to improve stability and performance.
*   **Advanced Training Heuristics**: Incorporates a suite of adaptive mechanisms (from `WuBuSpecTrans_v0.1.1`) to manage loss weights, learning rates, and intervene in common GAN training pitfalls.
*   **WuBu Nesting**: Continues to leverage WuBu for hierarchical hyperbolic representations in encoder (appearance, motion, temporal) and potentially in the feature-based discriminator.
*   **GAAD Regional Processing**: Utilizes Golden Aspect Adaptive Decomposition for defining regions.
*   **Optical Flow Motion Encoding**: Optional motion branch using optical flow.

**This guide assumes you are working with the `v0.3` version of the script and its corresponding `.bat` configuration file.**

## Table of Contents

1.  [Prerequisites & Setup](#1-prerequisites--setup)
2.  [Core Concepts of v0.3](#2-core-concepts-of-v03)
    *   [Spectral Features (DFT & DCT)](#spectral-features-dft--dct)
    *   [Dual Discriminators & Switching](#dual-discriminators--switching)
    *   [Advanced Training Heuristics](#advanced-training-heuristics)
3.  [Key Hyperparameters & Configuration (`.bat` script)](#3-key-hyperparameters--configuration-bat-script)
4.  [Phased Training Strategy for v0.3](#4-phased-training-strategy-for-v03)
    *   [Phase 0: Initial Sanity Check (Optional, Low Capacity)](#phase-0-initial-sanity-check-optional-low-capacity)
    *   [Phase 1: VAE Reconstruction Focus (DFT/DCT/Pixel)](#phase-1-vae-reconstruction-focus-dftdctpixel)
    *   [Phase 2: Introducing GAN with Primary Discriminator](#phase-2-introducing-gan-with-primary-discriminator)
    *   [Phase 3: Activating KL Regularization & Heuristics](#phase-3-activating-kl-regularization--heuristics)
    *   [Phase 4: Engaging Dual Discriminator Switching & Full Heuristics](#phase-4-engaging-dual-discriminator-switching--full-heuristics)
    *   [Phase 5: Fine-Tuning & Long Convergence](#phase-5-fine-tuning--long-convergence)
5.  [Using the `.bat` Script for Phased Training](#5-using-the-bat-script-for-phased-training)
6.  [Monitoring Training with WandB](#6-monitoring-training-with-wandb)
7.  [Troubleshooting Common v0.3 Issues](#7-troubleshooting-common-v03-issues)
8.  [Advanced Experimentation & Customization](#8-advanced-experimentation--customization)

## 1. Prerequisites & Setup

*   **Environment**:
    *   Python environment (conda recommended) with PyTorch (CUDA enabled).
    *   Install all dependencies: `pip install -r requirements.txt` (ensure this file includes `torch-dct` if not already listed).
    *   **CRITICAL FOR DCT**: `pip install torch-dct`. The script will warn if not found.
*   **Data**:
    *   Prepare your video dataset(s) (e.g., `.mp4`, `.avi`).
    *   Update `VIDEO_DATA_PATH` in your `.bat` script to point to your main training video or a directory of videos.
    *   Optionally, set `VALIDATION_VIDEO_PATH` for a separate validation set. If not provided, a split from the training data will be used.
    *   For quick tests, the script can generate a dummy video if `VIDEO_DATA_PATH` in your `.bat` file points to the default `demo_video_data_dir...` and the video file specified in `main()` (e.g., `dummy_video_hybridgen_v03.mp4`) is missing.
*   **Hardware**:
    *   A CUDA-enabled NVIDIA GPU is essential. **8GB VRAM is the absolute minimum for experimentation with reduced settings; 12GB+ is recommended, 24GB+ for larger configurations.**
    *   Ample system RAM (e.g., 32GB+, 64GB+ if VRAM is limited and swapping occurs) is beneficial.
*   **WandB (Weights & Biases)**:
    *   Highly recommended. Create an account at [wandb.ai](https://wandb.ai).
    *   Log in via CLI: `wandb login`.
    *   Set `WANDB_ENABLED=true` in your `.bat` script.

## 2. Core Concepts of v0.3

### Spectral Features (DFT & DCT)

*   **What**: Instead of (or in addition to) directly processing pixel patches, v0.3 can transform regional image patches into their frequency domain representations using:
    *   **DFT (Discrete Fourier Transform)**: Captures magnitude and phase information of different spatial frequencies. Good for textures and fine details. Implemented via `torch.fft`.
    *   **DCT (Discrete Cosine Transform)**: Captures energy compaction, primarily magnitude-like information of cosine basis functions. Often used in image/video compression (like JPEG). Implemented via `torch-dct`.
*   **Why**:
    *   May provide a more disentangled or efficient representation of patch appearance.
    *   Potentially easier for the VAE to reconstruct spectral coefficients than raw pixels.
    *   Can be combined for a richer feature set.
*   **Configuration**:
    *   `--use_dft_features_appearance`, `--use_dct_features_appearance` in `.bat`/args.
    *   `--spectral_patch_size_h/w`: Size of patches transformed.
    *   Normalization args for DFT/DCT (`--dft_norm_scale_video`, `--dct_norm_type`, etc.).

### Dual Discriminators & Switching

*   **Concept**: Uses two distinct discriminator architectures that can be switched during training.
    *   **Primary Discriminator** (`PRIMARY_DISC_ARCHITECTURE_VARIANT`): Often a pixel-based CNN (like `default_pixel_cnn` which is your `RegionalDiscriminator`).
    *   **Alternative Discriminator** (`ALT_DISC_ARCHITECTURE_VARIANT`): Can be a feature-based discriminator (like `global_wubu_video_feature` which operates on aggregated spectral features) or another pixel-based variant.
*   **Why**:
    *   Different discriminators might focus on different aspects of realism (e.g., pixel D for local artifacts, feature D for global consistency or spectral properties).
    *   Switching can prevent one discriminator from becoming too dominant or stale, potentially improving GAN stability and generator quality.
*   **Mechanism**:
    *   The `HybridTrainer` manages an `active_discriminator_key`.
    *   Heuristics (`_check_and_perform_disc_switch`) evaluate training state and can trigger a switch.
    *   Each discriminator has its own optimizer and Q-controller.
*   **Configuration**:
    *   `--primary_disc_architecture_variant`, `--alt_disc_architecture_variant`.
    *   `--enable_heuristic_disc_switching`, `--initial_disc_type`, and other `disc_switch_*` args.

### Advanced Training Heuristics

*   **Goal**: To automatically adapt training parameters and apply interventions to stabilize training and guide the model out of common failure modes.
*   **Key Heuristics (controlled by `--enable_heuristic_interventions` and specific flags):**
    *   **VAE Feature Matching**: If a feature-based D is active, encourages generator features (from D's intermediate layers) to match real data features from D. `lambda_feat_match_heuristic_video`.
    *   **Generator Easy Win Penalty**: If G fools D too easily (`loss_g_adv` very low) while reconstruction is still poor, penalize G. `lambda_g_easy_win_penalty_heuristic_video`.
    *   **Active Discriminator LR Boost**: If G is winning too easily and recon is stuck, temporarily boost active D's LR. `heuristic_active_d_lr_boost_factor`.
    *   **Forced Q-Exploration for Discriminator**: If D's Q-controller seems stuck (low reward), temporarily boost its epsilon. `heuristic_d_q_explore_boost_epsilon`.
    *   **Dynamic Lambda Overrides**: Factors (`heuristic_override_lambda_recon_factor`, etc.) can temporarily adjust base loss weights.
*   **Monitoring**: Heuristic activations and their effects are logged to console and WandB.

## 3. Key Hyperparameters & Configuration (`.bat` script)

Your `.bat` script is the central place to configure a training run. Refer to the script provided in the prompt for a full list. Below are critical groups:

*   **Paths**: `VIDEO_DATA_PATH`, `CHECKPOINT_OUTPUT_DIR`, `LOAD_CHECKPOINT`.
*   **Core Model**: `IMAGE_H/W`, `NUM_INPUT_FRAMES`, `NUM_PREDICT_FRAMES`, `LATENT_DIM`.
*   **GAAD**: `GAAD_NUM_REGIONS`, `GAAD_DECOMP_TYPE`.
*   **Spectral Transforms**: `USE_DFT/DCT_FEATURES_APPEARANCE`, `SPECTRAL_PATCH_SIZE_H/W`, normalization params.
*   **Encoder/Generator/Discriminator Architectures**: All `ENCODER_*`, `GEN_*`, `DISC_*` flags.
    *   Pay attention to `PRIMARY_DISC_ARCHITECTURE_VARIANT` and `ALT_DISC_ARCHITECTURE_VARIANT`.
*   **WuBu Stacks**: `WUBU_S/M/T/D_GLOBAL_VIDEO_*` flags for levels, dims, curvatures, etc.
*   **Training**: `EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, learning rates, AMP.
*   **Loss Weights**: `LAMBDA_RECON_DFT`, `LAMBDA_RECON_DCT`, `LAMBDA_KL` (base), `LAMBDA_GAN` (base).
*   **Q-Controllers & Heuristics**: All `Q_CONTROLLER_*`, `LAMBDA_KL_UPDATE_INTERVAL`, `ENABLE_HEURISTIC_*`, and specific heuristic threshold/factor flags.
*   **Logging & Validation**: `WANDB_*`, `LOG_INTERVAL`, `SAVE_INTERVAL`, `VAL_*` flags.

**Default WuBu Parameters:**
The script now includes common WuBu parameters like `initial_scales`, `initial_spread_values`, and `boundary_points_per_level` for each WuBu stack prefix (e.g., `wubu_s_initial_scales`). Ensure these are set appropriately for each stack. For video, `wubu_s_boundary_points_per_level` often defaults to `[4]`, while others might be `[0]`.

## 4. Phased Training Strategy for v0.3

Given the added complexity, a phased approach is even more critical. **Always load the best checkpoint from the previous phase.**

### Phase 0: Initial Sanity Check (Optional, Low Capacity)

*   **Goal**: Verify the full pipeline runs without errors using minimal settings.
*   **Settings**:
    *   `EPOCHS=1`
    *   `BATCH_SIZE=1`, `GRAD_ACCUM_STEPS=1`
    *   `NUM_INPUT_FRAMES=1`, `NUM_PREDICT_FRAMES=1`
    *   `USE_DFT_FEATURES_APPEARANCE=true`, `USE_DCT_FEATURES_APPEARANCE=false` (start with one spectral type)
    *   `GAAD_NUM_REGIONS=4`, `SPECTRAL_PATCH_SIZE_H/W=8`
    *   `LATENT_DIM=64`
    *   All WuBu `_NUM_LEVELS=1`, small `_HYPERBOLIC_DIMS` (e.g., `[32]`)
    *   `ENABLE_HEURISTIC_DISC_SWITCHING=false`, `ENABLE_HEURISTIC_INTERVENTIONS=false`
    *   `Q_CONTROLLER_ENABLED=false` (or keep on with high initial epsilon if preferred)
    *   `USE_AMP=true`
*   **Check**: Runs a few iterations without crashing. Losses are computed.

### Phase 1: VAE Reconstruction Focus (DFT/DCT/Pixel)

**Goal**: Get the VAE (Encoder + Generator) to reconstruct the chosen appearance features well.
*   **Settings**:
    *   `LAMBDA_KL=0.00001` (Extremely low, almost off)
    *   `LAMBDA_GAN=0.0` or `0.01` (Very low, or off if D is unstable)
    *   `LAMBDA_RECON_DFT`, `LAMBDA_RECON_DCT` to their target values (e.g., `7.0`). If only one spectral type is used, set the other to `0.0`. If no spectral, ensure `LAMBDA_RECON` (for pixels) is set.
    *   Start with `PRIMARY_DISC_ARCHITECTURE_VARIANT=default_pixel_cnn` and `ENABLE_HEURISTIC_DISC_SWITCHING=false`.
    *   `ENABLE_HEURISTIC_INTERVENTIONS=false`.
    *   `Q_CONTROLLER_ENABLED=true` (for LRs). `LAMBDA_KL_UPDATE_INTERVAL` can be high or 0.
    *   Moderate architectural settings (e.g., `LATENT_DIM=512`, `GAAD_NUM_REGIONS=16-24`, `SPECTRAL_PATCH_SIZE_H/W=16`, WuBu levels 2-3 with moderate dims).
    *   `LOAD_CHECKPOINT=` (fresh) or from Phase 0.
    *   `EPOCHS=10-30` (or until reconstruction loss plateaus).
*   **Monitoring**:
    *   `train/loss_recon_dft` and/or `train/loss_recon_dct` (or `train/loss_recon_pixel`) should steadily decrease. This is the **primary focus**.
    *   Visuals: `train_recon` (assembled pixels) and `val_predicted` should start showing blurry but recognizable content corresponding to the input. HUD elements might appear first.
    *   KL loss will be high but its contribution small. D and G_adv losses might be erratic or D might win easily; this is secondary.

### Phase 2: Introducing GAN with Primary Discriminator

**Goal**: Stabilize the GAN training with the primary discriminator, improving realism.
*   **Settings (load best from Phase 1)**:
    *   `LAMBDA_GAN=1.0` (or a moderate value like `0.1` to `0.5` initially, then increase).
    *   Keep `LAMBDA_KL` very low (`0.0001`).
    *   Primary Discriminator active (`INITIAL_DISC_TYPE` matching primary, `ENABLE_HEURISTIC_DISC_SWITCHING=false`).
    *   `LEARNING_RATE_GEN` and `LEARNING_RATE_DISC` as per converged Phase 1 or slightly adjusted.
    *   `Q_CONTROLLER_ENABLED=true`.
    *   `ENABLE_HEURISTIC_INTERVENTIONS=false` (or enable only very conservative ones like G easy win penalty).
    *   `EPOCHS=20-50+`.
*   **Monitoring**:
    *   `loss_g_adv` and `loss_d_total` should show typical GAN oscillations but hopefully not diverge.
    *   Reconstruction quality (`loss_recon_dft/dct/pixel`, PSNR, SSIM) should be maintained or continue improving slowly.
    *   Visuals: `fixed_noise_generated` samples should start showing some structure beyond noise. Reconstructions should get sharper/more detailed.

### Phase 3: Activating KL Regularization & Core Heuristics

**Goal**: Regularize the latent space and begin using core heuristics for stability.
*   **Settings (load best from Phase 2)**:
    *   Gradually increase `LAMBDA_KL` (e.g., to `0.001`, then `0.01`) OR enable the Q-controller for `lambda_kl` by setting `LAMBDA_KL_UPDATE_INTERVAL` to a reasonable value (e.g., `25-100`).
    *   `ENABLE_HEURISTIC_INTERVENTIONS=true` (enable common ones, perhaps not D-switching yet).
    *   Keep primary D active.
    *   `EPOCHS=20-50+`.
*   **Monitoring**:
    *   `train/loss_kl` should start to decrease or be managed by its Q-controller.
    *   Reconstruction might take a slight hit initially but should recover/stabilize.
    *   Observe if heuristics trigger and their impact on loss curves (e.g., `lambda_gan_eff_contrib`).
    *   `fixed_noise_generated` samples should improve in diversity and coherence.

### Phase 4: Engaging Dual Discriminator Switching & Full Heuristics

**Goal**: Leverage both discriminators and full heuristic suite for robust, high-quality generation.
*   **Settings (load best from Phase 3)**:
    *   `ENABLE_HEURISTIC_DISC_SWITCHING=true`.
    *   Ensure `PRIMARY_DISC_ARCHITECTURE_VARIANT` and `ALT_DISC_ARCHITECTURE_VARIANT` are correctly set.
    *   `INITIAL_DISC_TYPE` can guide the start.
    *   All relevant Q-controllers active.
    *   Full `ENABLE_HEURISTIC_INTERVENTIONS=true`.
    *   `EPOCHS=50-100+`.
*   **Monitoring**:
    *   Discriminator switching events (logged to console and potentially WandB).
    *   How losses behave for G and *both* D's (log active D's losses separately).
    *   Impact of all heuristics. Does `lambda_feat_match_heuristic_video` help when the feature D is active?
    *   Overall sample quality from `fixed_noise_generated` and `val_predicted`.

### Phase 5: Fine-Tuning & Long Convergence

*   **Goal**: Polish results, potentially run for many epochs.
*   **Settings (load best from Phase 4)**:
    *   May involve slight manual tweaks to base LRs if Q-controllers seem stuck in a suboptimal range.
    *   Adjust heuristic thresholds if they are too aggressive or too passive.
    *   Very long `EPOCHS` count.
    *   Lower `SAVE_INTERVAL` if disk space allows, or rely on `SAVE_EPOCH_INTERVAL` and best model saves.

## 5. Using the `.bat` Script for Phased Training

1.  **Create a Base `.bat` Script**: Use the comprehensive one you provided as a template.
2.  **For Each Phase**:
    *   **Copy** the base script (e.g., `run_phase1_wubuv03.bat`, `run_phase2_wubuv03.bat`).
    *   **Modify** the specific `SET "PARAM=VALUE"` lines according to the phase's goals.
    *   **Crucially, update `SET "LOAD_CHECKPOINT=path\to\best_checkpoint_from_PREVIOUS_phase.pt"` for phases 1 onwards.** Use the checkpoint that performed best on your `VAL_PRIMARY_METRIC`.
    *   Adjust `EPOCHS` for the duration of that phase.
    *   Consider giving a unique `WANDB_RUN_NAME` for each phase if you want them as separate WandB runs (e.g., by appending `_phase1`, `_phase2`). If `WANDB_RUN_NAME` is empty and `LOAD_CHECKPOINT` is set, WandB will try to resume the *original* run ID if found in the checkpoint's WandB info.
3.  **Execute**: Run the `.bat` script for the current phase.

## 6. Monitoring Training with WandB

(Largely same as v0.2, but with new losses/metrics)

*   **Losses**:
    *   `train/loss_recon_dft`, `train/loss_recon_dct`, `train/loss_recon_pixel` (only one of these will be non-zero depending on config and if G outputs pixels).
    *   `train/loss_kl`, `train/lambda_kl_eff`, `train/lambda_kl_base_val`.
    *   `train/loss_g_adv`, `train/loss_g_total`.
    *   `train/loss_d_total`, `train/loss_d_real`, `train/loss_d_fake` (these will be for the *active* discriminator).
    *   Heuristic contributions: `train/loss_g_feat_match_eff_contrib`, `train/loss_g_easy_win_penalty_eff_contrib`.
    *   Effective lambdas used in G_total: `train/lambda_recon_eff_contrib`, `train/lambda_kl_eff_contrib`, `train/lambda_gan_eff_contrib`.
*   **Learning Rates & Q-Controller**:
    *   `train/lr_gen`, `train/lr_disc_primary_*`, `train/lr_disc_alternative_*`.
    *   `q_info/*` for epsilon, actions, rewards of each Q-controller.
*   **Heuristics & Discriminator Switching**:
    *   `heuristic/*_active_val` flags.
    *   `heuristic/trigger_count/*` for how often specific heuristic conditions are met.
    *   `disc_switch/*` for D switching stats.
    *   `active_disc_is_primary_val` (1 if primary, 0 if alternative).
*   **Validation Metrics**: `val/avg_val_psnr`, `val/avg_val_ssim`, `val_avg_val_lpips`, etc.
*   **Image/Video Samples**:
    *   `samples_video/train_recon_pixels`, `samples_video/train_context_pixels`, `samples_video/train_ground_truth_pixels`.
    *   `samples_video/val_predicted_frames`, `samples_video/val_context_frames`, `samples_video/val_ground_truth_frames`.
    *   `samples_video/fixed_noise_generated_pixels`.

## 7. Troubleshooting Common v0.3 Issues

*   **OOM Errors**:
    *   **Enable AMP!** (`USE_AMP=true`).
    *   Reduce `BATCH_SIZE` (to 1 if necessary) and increase `GRAD_ACCUM_STEPS`.
    *   Reduce `NUM_INPUT_FRAMES` + `NUM_PREDICT_FRAMES`.
    *   Reduce `SPECTRAL_PATCH_SIZE_H/W`.
    *   Reduce `GAAD_NUM_REGIONS`.
    *   Reduce `ENCODER_SHALLOW_CNN_CHANNELS` (if RoIAlign used for spectral input path).
    *   Reduce `LATENT_DIM` and WuBu `_HYPERBOLIC_DIMS`.
    *   Reduce `DISC_MAX_DISC_CHANNELS`.
*   **NaN/Inf Losses**:
    *   Reduce learning rates (base LRs in `.bat`).
    *   Ensure `GLOBAL_MAX_GRAD_NORM` is active.
    *   Check `dft_norm_scale_video` and `dct_norm_tanh_scale` - if too small, input to `tanh` might be too large, leading to saturation and zero gradients, or if too large, input might be too small leading to precision issues. The `atanh` in reconstruction can also be sensitive if inputs are exactly +/-1. Your `SpectralTransformUtils` has clamping for `atanh` which is good.
    *   Hyperbolic math instability: Unlikely with current robust utils, but possible with extreme curvatures or scales.
    *   Consider disabling AMP for specific problematic WuBu modules if identified.
*   **Discriminator(s) Overpowering Generator (D losses near 0, G_adv very high and not improving G):**
    *   Heuristics (`heuristic_active_d_lr_boost_factor`, `heuristic_d_q_explore_duration`) should try to address this for the active D.
    *   Q-controller for D's LR should reduce it.
    *   Manually reduce D's base LR or increase G's base LR.
    *   Consider reducing `lambda_gan_base` temporarily.
    *   If D switching is on, it might switch to a (hopefully) weaker or different D.
*   **Generator Overpowering Discriminator(s) (G_adv near 0, D losses high):**
    *   Heuristics (`heuristic_penalize_g_easy_win_active`) should apply.
    *   Q-controllers should adjust LRs.
    *   Manually increase D's LR or decrease G's LR.
*   **Stagnant Training (losses plateau, samples don't improve):**
    *   Q-controllers (especially for LKL) and heuristics are designed to combat this.
    *   Look at Q-controller rewards; if consistently low, it might be stuck. Consider `reset_q_controllers_on_load` for a fresh Q-state on next run.
    *   Discriminator switching might help break stagnation.
    *   Try a different set of base LRs or WuBu configurations.

## 8. Advanced Experimentation & Customization

*   **Tune Heuristic Parameters**: The many `heuristic_*` and `disc_switch_*` args offer fine control.
*   **Different Discriminator Architectures**: Implement and register new variants in `VideoDiscriminatorWrapper` (e.g., a multi-scale pixel D, a Transformer-based D).
*   **Spectral Feature Engineering**:
    *   Experiment with DFT vs. DCT vs. both.
    *   Different normalization schemes for spectral coefficients.
    *   Select specific frequency bands instead of all coefficients.
*   **WuBu Architecture**: Deeper/wider stacks, different aggregation methods, more/less complex inter-level transforms.
*   **Data Augmentation**: (Not explicitly in script) Could be added to `VideoFrameDataset`.

This v0.3 model is a powerful research tool. The phased approach, diligent monitoring, and iterative refinement of hyperparameters will be key to unlocking its potential. Good luck with your "balls out" Fortnite generation!