@echo off
echo Activating virtual environment and running WuBuNest_Trainer.py with poem dataset...
echo --- WuBu Nesting Sequence Model Training ---

REM --- Check for Virtual Environment ---
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment activation script not found in .\venv\Scripts\
    echo Please ensure the virtual environment exists and is named 'venv'.
    pause
    exit /b 1
)

REM --- Activate Virtual Environment ---
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment. Check path and script integrity.
    pause
    exit /b 1
)
echo Virtual environment activated.

REM ====================================================================
REM === Set WuBu Nesting Sequence Model Parameters for Poem Dataset  ===
REM ====================================================================
REM === Note: List arguments (dims, curvatures, scales, etc.) MUST match NUM_LEVELS ===

REM --- Data Paths ---
set "DATA_PATH=C:/projects/bytropix/data/poems/poems_train.npy"
set "VAL_DATA_PATH=C:/projects/bytropix/data/poems/poems_val.npy"
set "CHECKPOINT_DIR=C:/projects/bytropix/wubunest_poem_checkpoints"
set "WANDB_PROJECT=bytropix-wubunest-poem"

REM --- Sequence Model Base Config ---
set "LOCAL_HIDDEN_SIZE=256"
REM Reduced decoder memory
set "DECODER_MEM_DIM=512"
set "CONTEXT_WINDOW=256"
set "N_GRAM_SIZES=3 4"
REM Smaller vocab for poems? Keep reasonable for now.
set "N_GRAM_VOCAB_SIZE=30000"
set "DROPOUT=0.2"
REM Use flat decoder for simplicity maybe
set "NO_HIERARCHICAL_DECODER_FLAG=--no_hierarchical_decoder"

REM --- WuBu Nesting Config (Defaults based on WuBuNesting.py, adjust as needed) ---
REM --- !!! Lengths of lists below MUST match NUM_LEVELS !!! ---
set "NUM_LEVELS=3"
REM Reduced hyperbolic dims
set "HYPERBOLIC_DIMS=128 64 32"
set "INITIAL_CURVATURES=1.0 1.0 1.0"
REM Use "" to enable, "--no_learnable_curvature" to disable
set "LEARNABLE_CURVATURE_FLAG="
set "INITIAL_SCALES=1.0 1.0 1.0"
REM Use "" to enable, "--no_learnable_scales" to disable
set "LEARNABLE_SCALES_FLAG="
set "BOUNDARY_POINTS_PER_LEVEL=5 4 3"
REM Length NUM_LEVELS - 1. 'quat' requires dims divisible by 4.
set "ROTATION_TYPES=so_n so_n"
REM Length NUM_LEVELS - 1. 'quat' requires dims divisible by 4.
set "TRANSFORM_TYPES=mlp mlp"
REM Length NUM_LEVELS - 1. For MLP transforms.
set "TRANSFORM_HIDDEN_DIMS=96 48"
REM Use "" to enable, "--no_level_descriptors" to disable
set "LEVEL_DESCRIPTORS_FLAG="
REM Use "" to enable, "--no_level_spread" to disable
set "LEVEL_SPREAD_FLAG="
REM Use "" to enable, "--no_learnable_spread" to disable
set "LEARNABLE_SPREAD_FLAG="
REM Use "" to enable, "--no_tangent_flow" to disable
set "TANGENT_FLOW_FLAG="
set "TANGENT_FLOW_TYPE=mlp"
set "AGGREGATION_METHOD=concat_tangent"
set "RELATIVE_VECTOR_AGGREGATION=mean"

REM --- Training Hyperparameters (Adjusted for Poems) ---
REM Very small batch size
set "BATCH_SIZE=2"
REM Accumulate more due to small batch
set "GRAD_ACCUM_STEPS=4"
REM Smaller LR for potentially complex model
set "LEARNING_RATE=5e-5"
REM More epochs than integrated_hyper, fewer than original
set "EPOCHS=5"
set "WEIGHT_DECAY=0.01"
REM Stricter gradient clipping
set "MAX_GRAD_NORM=0.5"

REM --- Optimizer Q-Learning Controller ---
set "Q_LEARNING_RATE=0.01"
set "Q_DISCOUNT=0.95"
set "Q_EPSILON=0.25"
REM Slightly faster decay
set "Q_EPSILON_DECAY=0.998"
set "Q_MIN_EPSILON=0.02"

REM --- Logging & Saving ---
REM Log more often with small dataset
set "LOG_INTERVAL=2"
REM Save only at epoch end (0 disables step saving)
set "SAVE_INTERVAL=0"
REM Toggle WandB logging
set "WANDB_FLAG=--wandb"
REM To disable WandB, uncomment the next line and comment the line above
REM set "WANDB_FLAG="

REM --- Misc ---
set "SEED=42"
REM Safer default for Windows
set "NUM_WORKERS=0"
REM Disable AMP initially for stability debugging
set "AMP_FLAG=--no_amp"
REM Use --detect_anomaly if needed
set "DETECT_ANOMALY_FLAG="

REM --- Resume Flag (Optional) ---
REM e.g., set "RESUME_FLAG=--resume C:/path/to/checkpoint.pt"
set "RESUME_FLAG="

REM === Ensure the poem dataset exists ===
echo Checking/generating poem dataset...
python poem_dataset_generator.py
if errorlevel 1 (
    echo Failed to generate poem dataset. Exiting.
    pause
    exit /b 1
)

REM ======================================================
REM === Execute the Python Training Script             ===
REM ======================================================
echo Starting WuBu Nesting Sequence Model training on poem dataset...
echo Configuration: LR=%LEARNING_RATE%, GradNorm=%MAX_GRAD_NORM%, Batch=%BATCH_SIZE%, Accum=%GRAD_ACCUM_STEPS%, Epochs=%EPOCHS%
echo WuBu Config: Levels=%NUM_LEVELS%, Dims=%HYPERBOLIC_DIMS%
echo.

REM --- Construct the command ---
python WuBuNest_Trainer.py ^
  --data_path "%DATA_PATH%" ^
  --val_data_path "%VAL_DATA_PATH%" ^
  --checkpoint_dir "%CHECKPOINT_DIR%" ^
  --local_hidden_size %LOCAL_HIDDEN_SIZE% ^
  --decoder_memory_dim %DECODER_MEM_DIM% ^
  --context_window %CONTEXT_WINDOW% ^
  --n_gram_sizes %N_GRAM_SIZES% ^
  --n_gram_vocab_size %N_GRAM_VOCAB_SIZE% ^
  --dropout %DROPOUT% ^
  %NO_HIERARCHICAL_DECODER_FLAG% ^
  --num_levels %NUM_LEVELS% ^
  --hyperbolic_dims %HYPERBOLIC_DIMS% ^
  --initial_curvatures %INITIAL_CURVATURES% ^
  %LEARNABLE_CURVATURE_FLAG% ^
  --initial_scales %INITIAL_SCALES% ^
  %LEARNABLE_SCALES_FLAG% ^
  --boundary_points_per_level %BOUNDARY_POINTS_PER_LEVEL% ^
  --rotation_types %ROTATION_TYPES% ^
  --transform_types %TRANSFORM_TYPES% ^
  --transform_hidden_dims %TRANSFORM_HIDDEN_DIMS% ^
  %LEVEL_DESCRIPTORS_FLAG% ^
  %LEVEL_SPREAD_FLAG% ^
  %LEARNABLE_SPREAD_FLAG% ^
  %TANGENT_FLOW_FLAG% ^
  --tangent_flow_type %TANGENT_FLOW_TYPE% ^
  --aggregation_method %AGGREGATION_METHOD% ^
  --relative_vector_aggregation %RELATIVE_VECTOR_AGGREGATION% ^
  --batch_size %BATCH_SIZE% ^
  --grad_accum_steps %GRAD_ACCUM_STEPS% ^
  --learning_rate %LEARNING_RATE% ^
  --epochs %EPOCHS% ^
  --weight_decay %WEIGHT_DECAY% ^
  --max_grad_norm %MAX_GRAD_NORM% ^
  --q_learning_rate %Q_LEARNING_RATE% ^
  --q_discount %Q_DISCOUNT% ^
  --q_epsilon %Q_EPSILON% ^
  --q_epsilon_decay %Q_EPSILON_DECAY% ^
  --q_min_epsilon %Q_MIN_EPSILON% ^
  --log_interval %LOG_INTERVAL% ^
  --save_interval %SAVE_INTERVAL% ^
  %WANDB_FLAG% ^
  --wandb_project "%WANDB_PROJECT%" ^
  --seed %SEED% ^
  --num_workers %NUM_WORKERS% ^
  %AMP_FLAG% ^
  %DETECT_ANOMALY_FLAG% ^
  %RESUME_FLAG%

REM --- Check exit code ---
if errorlevel 1 (
    echo Script failed with error code %errorlevel%.
) else (
    echo Script finished successfully.
)

echo.
echo Execution completed. Press any key to exit.
pause > nul