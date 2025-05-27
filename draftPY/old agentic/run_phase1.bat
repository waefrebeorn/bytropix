@echo off
REM Batch file to execute Phase 1 (Reconstruction Focus) ETP Training
REM This is for a CODING-ONLY task. Paths are illustrative.
REM Actual execution requires a fully configured environment and valid paths.

echo Starting ETP Phase 1 Training Runner...
echo Ensure your Python environment with PyTorch and other dependencies is activated.
echo Replace placeholder paths with actual paths to your embedding files before execution.

REM Example: SET EMBEDDING_DIR_COMMON=C:\bytropix_project\draftPY\etp_common
REM Example: SET CHECKPOINT_DIR_PHASE1=C:\bytropix_project\draftPY\etp_phase1_reconstruction\checkpoints_p1

REM Assuming the script is run from the root of the Bytropix project,
REM and draftPY is in the PYTHONPATH or accessible directly.

python draftPY/etp_phase1_reconstruction/run_phase1.py ^
    REM --- Dataset Paths (Update these to your actual embedding files) ---
    --embeddings_file_A "draftPY/etp_common/dummy_corpus_A_embeddings.npz" ^
    --embeddings_file_B "draftPY/etp_common/dummy_corpus_B_embeddings.npz" ^
    REM --- ETP Sphere Model Config ---
    --ds_r1_embedding_dim 768 ^
    --wubu_initial_tangent_dim 256 ^
    --head_mlp_layers 2 ^
    --decoder_mlp_layers 2 ^
    REM --wubu_core_config_json "path/to/your/wubu_core_config.json" ^
    REM --- Loss Weights for Pure Phase 1 (Reconstruction only) ---
    --lambda_rec 1.0 ^
    --lambda_vsp 0.0 ^
    REM --- Training Hyperparameters ---
    --epochs 50 ^
    --batch_size 32 ^
    --grad_accum_steps 1 ^
    --global_max_grad_norm 1.0 ^
    REM --- Optimizer Settings ---
    --lr_sphere_wubu_core 1e-4 ^
    --lr_sphere_mlps 1e-4 ^
    REM --- Q-Controller Config (Disabled for this example, enable and configure if needed) ---
    --q_controller_enabled False ^
    REM --q_config_sphere_wubu_core_json "path/to/q_wubu_core_config.json" ^
    REM --q_config_sphere_mlps_json "path/to/q_mlps_config.json" ^
    REM --- Logging/Checkpointing (Update checkpoint_dir as needed) ---
    --checkpoint_dir "draftPY/etp_phase1_reconstruction/checkpoints_phase1_rec_example" ^
    --log_interval 20 ^
    --save_interval 0 ^
    --val_interval_epochs 1 ^
    --wandb_project "ETP_Phase1_Reconstruction_Example" ^
    --wandb_run_name "phase1_rec_example_run" ^
    REM --- Device and AMP Settings (CPU and no AMP for this example) ---
    --device "cpu" ^
    --use_amp False
    REM --load_checkpoint "path/to/phase1_checkpoint.pth.tar" ^

echo Phase 1 Training script call prepared.
echo Remember this is for a CODING-ONLY task.
pause
