@echo off
echo Activating virtual environment and running bsfin_main.py...

if not exist "venv" (
    echo Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate

REM --- Set BSFIN Training Parameters ---
set DATA_PATH=C:/projects/bytropix/data/wikitext_train.npy
set VAL_DATA_PATH=C:/projects/bytropix/data/wikitext_val.npy
set CHECKPOINT_DIR=C:/projects/bytropix/checkpoints_v2
set WANDB_PROJECT=bytropix-v2

REM --- Model Architecture ---
set LOCAL_HIDDEN_SIZE=512
set COMPLEX_DIM=512
set NUM_COMPLEX_LAYERS=12
set NUM_COMPLEX_HEADS=8
set DECODER_MEM_DIM=768
set CONTEXT_WINDOW=256
set N_GRAM_SIZES=3 4
set SFIN_NOISE_SCALE=0.05
REM Add --no_entanglement or --no_rope flags below if needed

REM --- Training Hyperparameters ---
set BATCH_SIZE=20
set GRAD_ACCUM_STEPS=2
set LEARNING_RATE=0.001
set EPOCHS=15
set WEIGHT_DECAY=0.01
set MAX_GRAD_NORM=1.0

REM --- Logging & Saving ---
set LOG_INTERVAL=10
set SAVE_INTERVAL=999999
set WANDB_FLAG=--wandb

REM --- Misc ---
set SEED=42
set NUM_WORKERS=2
set AMP_FLAG=
REM Use set AMP_FLAG=--no_amp to disable Automatic Mixed Precision

REM --- Run the bsfin_main.py script ---
echo Starting BSFIN training...
python bsfin_main.py ^
  --data_path %DATA_PATH% ^
  --val_data_path %VAL_DATA_PATH% ^
  --checkpoint_dir %CHECKPOINT_DIR% ^
  --local_hidden_size %LOCAL_HIDDEN_SIZE% ^
  --complex_dim %COMPLEX_DIM% ^
  --num_complex_layers %NUM_COMPLEX_LAYERS% ^
  --num_complex_heads %NUM_COMPLEX_HEADS% ^
  --decoder_memory_dim %DECODER_MEM_DIM% ^
  --context_window %CONTEXT_WINDOW% ^
  --n_gram_sizes %N_GRAM_SIZES% ^
  --sfin_noise_scale %SFIN_NOISE_SCALE% ^
  --batch_size %BATCH_SIZE% ^
  --grad_accum_steps %GRAD_ACCUM_STEPS% ^
  --learning_rate %LEARNING_RATE% ^
  --epochs %EPOCHS% ^
  --weight_decay %WEIGHT_DECAY% ^
  --max_grad_norm %MAX_GRAD_NORM% ^
  --log_interval %LOG_INTERVAL% ^
  --save_interval %SAVE_INTERVAL% ^
  %WANDB_FLAG% ^
  --wandb_project %WANDB_PROJECT% ^
  --seed %SEED% ^
  --num_workers %NUM_WORKERS% ^
  %AMP_FLAG%

REM Add other flags like --resume C:/path/to/checkpoint.pt, --no_entanglement, --no_rope after %AMP_FLAG% if needed

echo.
echo bsfin_main.py execution completed.
pause
