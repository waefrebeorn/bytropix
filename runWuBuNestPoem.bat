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
REM === For boolean flags: set VAR_FLAG="--flag-name" to enable, set VAR_FLAG="--no-flag-name" to disable, set VAR_FLAG="" to use default ===
REM === For store_true flags (like --no_amp): set VAR_FLAG="--flag-name" to enable the flag (disable the feature), set VAR_FLAG="" to disable the flag (enable the feature) ===

REM --- Data Paths ---
set "DATA_PATH=C:/projects/bytropix/data/poems/poems_train.npy"
set "VAL_DATA_PATH=C:/projects/bytropix/data/poems/poems_val.npy"


set "CHECKPOINT_DIR=C:/projects/bytropix/wubunest_poem_checkpoints_v03"
set "WANDB_PROJECT=bytropix-wubunest-poem-v03"

REM --- Sequence Model Base Config ---
set "LOCAL_HIDDEN_SIZE=256"
set "DECODER_MEM_DIM=512"
set "CONTEXT_WINDOW=256"
set "N_GRAM_SIZES=3 4"
set "N_GRAM_VOCAB_SIZE=30000"
set "DROPOUT=0.2"
REM Default is True for use_hierarchical_decoder. Set "--no-use-hierarchical-decoder" to disable. Use "" for default.
set "HIERARCHICAL_DECODER_FLAG=--no-use-hierarchical-decoder"

REM --- WuBu Nesting Config (Defaults based on WuBuNesting.py, adjust as needed) ---
REM --- !!! Lengths of lists below MUST match NUM_LEVELS !!! ---
set "NUM_LEVELS=3"
set "HYPERBOLIC_DIMS=128 64 32"
set "INITIAL_CURVATURES=1.0 1.0 1.0"
set "INITIAL_SCALES=1.0 1.0 1.0"
set "BOUNDARY_POINTS_PER_LEVEL=5 4 3"

REM --- Flags Controlling WuBu Features (Use "" for default, or --flag / --no-flag) ---
REM Default is True. Set "--no-learnable-curvature" to disable. Use "" for default.
set "LEARNABLE_CURVATURE_FLAG="
REM Default is True. Set "--no-learnable-scales" to disable. Use "" for default.
set "LEARNABLE_SCALES_FLAG="
REM Default is True. Set "--no-learnable-spread" to disable. Use "" for default.
set "LEARNABLE_SPREAD_FLAG="
REM Default is True. Set "--no-use-level-descriptors" to disable. Use "" for default.
set "LEVEL_DESCRIPTORS_FLAG="
REM Default is True. Set "--no-use-level-spread" to disable. Use "" for default.
set "LEVEL_SPREAD_FLAG="
REM Default is True. Set "--no-use-tangent-flow" to disable. Use "" for default.
set "TANGENT_FLOW_FLAG="

REM --- Other WuBu parameters (non-flags) ---
set "TANGENT_FLOW_TYPE=mlp"
set "AGGREGATION_METHOD=concat_tangent"
set "RELATIVE_VECTOR_AGGREGATION=mean"

REM --- Training Hyperparameters (Adjusted for Poems) ---
set "BATCH_SIZE=16"
set "GRAD_ACCUM_STEPS=2"
set "LEARNING_RATE=1e-4"
set "EPOCHS=5"
set "WEIGHT_DECAY=0.01"
set "MAX_GRAD_NORM=55054544896.0"

REM --- Optimizer Q-Learning Controller ---
REM Default is True for enable_q_controller. Set "--no-enable-q-controller" to disable. Use "" for default.
set "Q_CONTROLLER_FLAG="
set "Q_LEARNING_RATE=0.01"
set "Q_DISCOUNT=0.95"
set "Q_EPSILON=0.25"
set "Q_EPSILON_DECAY=0.998"
set "Q_MIN_EPSILON=0.02"

REM --- Logging & Saving ---
set "LOG_INTERVAL=2"
set "SAVE_INTERVAL=0"
REM Set "--wandb" to enable, "" to disable (default is False in argparse)
set "WANDB_FLAG=--wandb"

REM --- Misc ---
set "SEED=42"
set "NUM_WORKERS=0"
REM Default is False for no_amp (meaning AMP is ON by default). Set "--no_amp" to disable AMP. Use "" for default (AMP enabled).
set "AMP_FLAG="
REM Default is False for detect_anomaly. Set "--detect-anomaly" to enable.
set "DETECT_ANOMALY_FLAG="

REM --- Resume Flag (Optional) ---
REM Example: set "RESUME_FLAG=--resume C:/projects/bytropix/wubunest_poem_checkpoints_v03/checkpoint_epoch_1_final_vloss0.123.pt"
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
REM Using torchrun is recommended for DDP, but this script assumes single process for simplicity now.
REM Add 'torchrun --standalone --nproc_per_node=1' before 'python' if using DDP locally.
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
  %HIERARCHICAL_DECODER_FLAG% ^
  --num_levels %NUM_LEVELS% ^
  --hyperbolic_dims %HYPERBOLIC_DIMS% ^
  --initial_curvatures %INITIAL_CURVATURES% ^
  %LEARNABLE_CURVATURE_FLAG% ^
  --initial_scales %INITIAL_SCALES% ^
  %LEARNABLE_SCALES_FLAG% ^
  --boundary_points_per_level %BOUNDARY_POINTS_PER_LEVEL% ^
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
  %Q_CONTROLLER_FLAG% ^
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