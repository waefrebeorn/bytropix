@echo off
REM Batch file to execute Phase 2 (Adversarial Latent Alignment) ETP Training
REM This is for a CODING-ONLY task. Paths are illustrative.
REM Actual execution requires a fully configured environment and valid paths.

echo Starting ETP Phase 2 Training Runner...
echo Ensure your Python environment with PyTorch and other dependencies is activated.
echo Replace placeholder paths with actual paths to your embedding files and checkpoints before execution.

REM Example: SET EMBEDDING_DIR_COMMON=C:\bytropix_project\draftPY\etp_common
REM Example: SET CHECKPOINT_DIR_PHASE2=C:\bytropix_project\draftPY\etp_phase2_ala\checkpoints_p2

REM Assuming the script is run from the root of the Bytropix project,
REM and draftPY is in the PYTHONPATH or accessible directly.

python draftPY/etp_phase2_ala/run_phase2.py ^
    REM --- Dataset Paths (Update these to your actual embedding files) ---
    --embeddings_file_A "draftPY/etp_common/dummy_corpus_A_embeddings.npz" ^
    --embeddings_file_B "draftPY/etp_common/dummy_corpus_B_embeddings.npz" ^
    REM --- ETP Sphere Model Config ---
    --ds_r1_embedding_dim 768 ^
    --wubu_initial_tangent_dim 256 ^
    --head_mlp_layers 2 ^
    --decoder_mlp_layers 2 ^
    REM --wubu_core_config_json "path/to/your/wubu_core_config.json" ^
    REM --- Discriminator Config ---
    --disc_hidden_dims_json "[256, 128]" ^
    --disc_activation_fn "leaky_relu" ^
    --disc_use_spectral_norm True ^
    REM --- Loss Weights for Phase 2 ---
    --lambda_ala 0.1 ^
    --lambda_rec 1.0 ^
    --lambda_vsp 0.01 ^
    REM --- Training Hyperparameters ---
    --epochs 100 ^
    --batch_size 32 ^
    --grad_accum_steps 1 ^
    --global_max_grad_norm 1.0 ^
    REM --- Optimizer Settings ---
    --lr_sphere_wubu_core 5e-5 ^
    --lr_sphere_mlps 5e-5 ^
    --lr_discriminator 1e-4 ^
    REM --optimizer_kwargs_wubu_core_json "{...}" ^
    REM --optimizer_kwargs_mlps_json "{...}" ^
    REM --optimizer_kwargs_discriminator_json "{...}" ^
    REM --- Q-Controller Config (Disabled for this example, enable and configure if needed) ---
    --q_controller_enabled False ^
    REM --q_config_sphere_wubu_core_json "path/to/q_wubu_core_config.json" ^
    REM --q_config_sphere_mlps_json "path/to/q_mlps_config.json" ^
    REM --q_config_discriminator_json "path/to/q_discriminator_config.json" ^
    REM --- Logging/Checkpointing (Update checkpoint_dir and load_checkpoint as needed) ---
    --checkpoint_dir "draftPY/etp_phase2_ala/checkpoints_phase2_ala_example" ^
    --load_checkpoint "draftPY/etp_phase1_reconstruction/checkpoints_phase1_rec_example/checkpoint_p1_epoch_X_step_Y.pth.tar" ^
    --log_interval 50 ^
    --save_interval 1000 ^
    --val_interval_epochs 1 ^
    --wandb_project "ETP_Phase2_ALA_Example" ^
    --wandb_run_name "phase2_ala_example_run" ^
    REM --- Device and AMP Settings (CPU and no AMP for this example) ---
    --device "cpu" ^
    --use_amp False

echo Phase 2 Training script call prepared.
echo Note: --load_checkpoint path should be updated to a valid Phase 1 checkpoint.
echo Remember this is for a CODING-ONLY task.
pause
