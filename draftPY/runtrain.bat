@echo off
echo Activating virtual environment and running main.py...
REM Check if the virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Run the main.py script with updated arguments
echo Running main.py...
python main.py ^
  --data_path C:/projects/bytropix/data/wikitext_train.npy ^
  --val_data_path C:/projects/bytropix/data/wikitext_val.npy ^
  --context_size 256 ^
  --local_hidden_size 512 ^
  --global_hidden_size 1024 ^
  --num_local_encoder_layers 2 ^
  --num_global_layers 12 ^
  --num_local_decoder_layers 6 ^
  --window_size 256 ^
  --batch_size 20 ^
  --grad_accum_steps 2 ^
  --learning_rate 0.003 ^
  --epochs 15 ^
  --checkpoint_dir C:/projects/bytropix/checkpoints ^
  --log_interval 10 ^
  --weight_decay 0.01 ^
  --max_grad_norm 1.0 ^
  --dropout 0.1 ^
  --n_gram_sizes 3 4 ^
  --n_gram_vocab_size 30000 ^
  --seed 42 ^
  --num_workers 2 ^
  --wandb ^
  --wandb_project bytropix

REM Inform the user that the script has completed
echo main.py execution completed.
pause