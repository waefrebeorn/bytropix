# === THE FINAL, PERFECTED ONE-SHOT SCRIPT (v2025.2) ===

# --- 1. Update System & Install Prerequisites ---
echo "--- [1/5] Updating and installing prerequisites... ---"
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3.12 python3.12-venv

# --- 2. Create the File Bridge (Symbolic Link) ---
echo "--- [2/5] Creating the File Bridge to your Windows project... ---"
ln -s /mnt/c/Projects/HASHMIND/ ~/HASHMIND

# --- 3. Create the Python Environment ---
echo "--- [3/5] Forging the Python 3.12 environment... ---"
cd ~
python3.12 -m venv wubumind_env

# --- 4. Activate the Environment ---
echo "--- [4/5] Activating the environment for installation... ---"
source ~/wubumind_env/bin/activate

# --- 5. Install GPU-Accelerated Python Packages via PIP ---
echo "--- [5/5] Installing modern, self-contained GPU libraries... ---"
pip install -U pip
pip install -U "jax[cuda12]"
pip install "faiss-gpu-cu12" # <-- This is now a required step!
pip install flax optax tqdm numpy tokenizers
pip install av tensorflow-cpu scikit-learn git+https://github.com/openai/CLIP.git
echo ""
echo "--- SETUP COMPLETE ---"
echo ">>> Please CLOSE this terminal and OPEN A NEW ONE to finalize the setup. <<<"
echo "source ~/wubumind_env/bin/activate && cd ~/HASHMIND && python3 WuBuMindV7.1.py train"