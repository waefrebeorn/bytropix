@echo off
echo Activating virtual environment and running LIVEBSFIN.py...

if not exist "venv" (
    echo Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate

REM --- Set LIVEBSFIN Parameters ---
set BASE_CHECKPOINT=C:/projects/bytropix/checkpoints_v2/checkpoint_final.pt
set DATA_STREAM_FILE=C:/projects/bytropix/data/live_data_stream.txt
set LIVE_CHECKPOINT_DIR=C:/projects/bytropix/live_checkpoints_v1

REM --- Continual Learning Parameters ---
set MEMORY_CAPACITY=100000
set LIVE_BATCH_SIZE=8
set MAX_STEPS=10000
set SAVE_INTERVAL=500
set LOG_INTERVAL=20
set DECAY_INTERVAL=500
set LIVE_LEARNING_RATE=5e-6
set IMPORTANCE_FACTOR=0.5

REM --- Model Architecture (MUST MATCH THE --base_checkpoint MODEL) ---
set LOCAL_HIDDEN_SIZE=512
set COMPLEX_DIM=512
set NUM_COMPLEX_LAYERS=12
set NUM_COMPLEX_HEADS=8
set DECODER_MEM_DIM=768
set CONTEXT_WINDOW=256
set N_GRAM_SIZES=3 4
set N_GRAM_VOCAB_SIZE=30000
set SFIN_NOISE_SCALE=0.05
REM Add --no_entanglement or --no_rope flags below if the base model used them

REM --- Misc ---
set SEED=42
REM set NUM_WORKERS=2 REM Removed as LIVEBSFIN.py parser doesn't accept it
set AMP_FLAG=
REM Use set AMP_FLAG=--no_amp to disable Automatic Mixed Precision

REM --- Run the LIVEBSFIN.py script ---
echo Starting LIVEBSFIN continual learning...
python LIVEBSFIN.py ^
  --base_checkpoint %BASE_CHECKPOINT% ^
  --data_stream_file %DATA_STREAM_FILE% ^
  --checkpoint_dir %LIVE_CHECKPOINT_DIR% ^
  --context_size %CONTEXT_WINDOW% ^
  --memory_capacity %MEMORY_CAPACITY% ^
  --batch_size %LIVE_BATCH_SIZE% ^
  --max_steps %MAX_STEPS% ^
  --save_interval %SAVE_INTERVAL% ^
  --log_interval %LOG_INTERVAL% ^
  --decay_interval %DECAY_INTERVAL% ^
  --learning_rate %LIVE_LEARNING_RATE% ^
  --importance_factor %IMPORTANCE_FACTOR% ^
  --local_hidden_size %LOCAL_HIDDEN_SIZE% ^
  --complex_dim %COMPLEX_DIM% ^
  --num_complex_layers %NUM_COMPLEX_LAYERS% ^
  --num_complex_heads %NUM_COMPLEX_HEADS% ^
  --decoder_memory_dim %DECODER_MEM_DIM% ^
  --n_gram_sizes %N_GRAM_SIZES% ^
  --n_gram_vocab_size %N_GRAM_VOCAB_SIZE% ^
  --sfin_noise_scale %SFIN_NOISE_SCALE% ^
  --seed %SEED% %AMP_FLAG%
  REM Removed caret ^ from previous line and moved AMP_FLAG here
  REM Add --no_entanglement, --no_rope, or other BSFIN flags on the line above if needed

echo.
echo LIVEBSFIN.py execution completed.
pause
