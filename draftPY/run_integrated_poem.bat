@echo off
echo Activating virtual environment and running integrated_hyper_hakmem_model.py with poem dataset...

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
REM === Set Integrated HyperHAKMEM Parameters for Poem Dataset      ===
REM ====================================================================

REM --- Data Paths ---
set "DATA_PATH=C:/projects/bytropix/data/poems/poems_train.npy"
set "VAL_DATA_PATH=C:/projects/bytropix/data/poems/poems_val.npy"
set "CHECKPOINT_DIR=C:/projects/bytropix/poem_checkpoints"
set "WANDB_PROJECT=bytropix-poem-test"

REM --- Model Architecture (Common) ---
set "LOCAL_HIDDEN_SIZE=256"
set "DECODER_MEM_DIM=512"
set "CONTEXT_WINDOW=256"
set "N_GRAM_SIZES=3 4"
set "N_GRAM_VOCAB_SIZE=69420"
set "DROPOUT=0.2"

REM --- Model Architecture (Hyperbolic Attention Specific) ---
set "HYPERBOLIC_EMBEDDING_DIM=384"
set "NUM_HYPERBOLIC_LAYERS=8"
set "NUM_HYPERBOLIC_HEADS=8"
set "CURVATURE=0.8"
set "CLIPPING_RADIUS=0.9"

REM --- Boolean Flags ---
set "NO_HIERARCHICAL_DECODER_FLAG="

REM --- Training Hyperparameters (Optimized for Smaller Dataset) ---
set "BATCH_SIZE=2"
set "GRAD_ACCUM_STEPS=8"         REM Reduced from 4 since dataset is smaller
set "LEARNING_RATE=0.0001"       REM Slightly increased for faster learning
set "EPOCHS=3"                   REM Reduced from 12 due to smaller dataset
set "WEIGHT_DECAY=0.01"
set "MAX_GRAD_NORM=0.5"

REM --- Optimizer Q-Learning Controller ---
set "Q_LEARNING_RATE=0.01"
set "Q_DISCOUNT=0.9"
set "Q_EPSILON=0.3"
set "Q_EPSILON_DECAY=0.995"      REM Faster decay for shorter training
set "Q_MIN_EPSILON=0.05"

REM --- Logging & Saving ---
set "LOG_INTERVAL=1"             REM More frequent logging
set "SAVE_INTERVAL=5000"           REM More frequent saving
REM Toggle WandB logging as needed
set "WANDB_FLAG=--wandb"
REM set "WANDB_FLAG="

REM --- Misc ---
set "SEED=42"
set "NUM_WORKERS=2"
set "AMP_FLAG="
set "DETECT_ANOMALY_FLAG="

REM --- Resume Flag (Optional) ---
set "RESUME_FLAG="

REM === First ensure the poem dataset exists by running the generator ===
echo Checking/generating poem dataset...
python poem_dataset_generator.py

REM ======================================================
REM === Execute the Python Training Script             ===
REM ======================================================
echo Starting Integrated HyperHAKMEM training on poem dataset...
echo Configuration: LR=%LEARNING_RATE%, GradNorm=%MAX_GRAD_NORM%, Batch=%BATCH_SIZE%, Accum=%GRAD_ACCUM_STEPS%, Epochs=%EPOCHS%
echo Model: HypDim=%HYPERBOLIC_EMBEDDING_DIM%, HypLayers=%NUM_HYPERBOLIC_LAYERS%, HypHeads=%NUM_HYPERBOLIC_HEADS%
echo.

REM --- Construct the command ---
python integrated_hyper_hakmem_model.py ^
  --data_path "%DATA_PATH%" ^
  --val_data_path "%VAL_DATA_PATH%" ^
  --checkpoint_dir "%CHECKPOINT_DIR%" ^
  --local_hidden_size %LOCAL_HIDDEN_SIZE% ^
  --hyperbolic_embedding_dim %HYPERBOLIC_EMBEDDING_DIM% ^
  --num_hyperbolic_layers %NUM_HYPERBOLIC_LAYERS% ^
  --num_hyperbolic_heads %NUM_HYPERBOLIC_HEADS% ^
  --decoder_memory_dim %DECODER_MEM_DIM% ^
  --context_window %CONTEXT_WINDOW% ^
  --n_gram_sizes %N_GRAM_SIZES% ^
  --n_gram_vocab_size %N_GRAM_VOCAB_SIZE% ^
  --dropout %DROPOUT% ^
  %NO_HIERARCHICAL_DECODER_FLAG% ^
  --curvature %CURVATURE% ^
  --clipping_radius %CLIPPING_RADIUS% ^
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