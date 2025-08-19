# MAKE A FOLDER C:/PROJECTS/HASHMIND
IF ON WINDOWS USE MY WSL TUTORIAL
THIS IS EASIEST

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
pip install scikit-learn
echo ""
echo "--- SETUP COMPLETE ---"
echo ">>> Please CLOSE this terminal and OPEN A NEW ONE to finalize the setup. <<<"
echo "source ~/wubumind_env/bin/activate && cd ~/HASHMIND && python3 WuBuMindV7.1.py train"

### Command Run List: WubuMind v25.3 Full Test Run

**Basename for this run:** `wubumind_v25_final`

---

**Step 1: Pre-tokenize the Corpus**

This is the initialization step. It will detect that the tokenizer and token file for our new basename don't exist, create them, and then exit. This is the expected and correct behavior.

```bash
source ~/wubumind_env/bin/activate && cd ~/HASHMIND && python3 WuBuMindV7.1.py train_navigator --basename wubumind_v25_final --epochs 5 --batch-size 2048
```

*   **Expected Outcome:** The script will print "No tokenizer found...", create `wubumind_v25_final_bpe.json` and `wubumind_v25_final.tokens.bin`, and then print "Pre-tokenization complete. Please run the desired command again." before exiting.

---

**Step 2: Train the Galactic Navigator**

Now that the data is prepped, we run the *exact same command* again to perform the actual training. The intelligent Q-Controller will manage the learning rate to find a good representation of the three manifolds.

```bash
source ~/wubumind_env/bin/activate && cd ~/HASHMIND && python3 WuBuMindV7.1.py train_navigator --basename wubumind_v25_final --epochs 5 --batch-size 2048
```

*   **Expected Outcome:** The training process will start. You'll see the progress bar for 5 epochs. At the end, it will save the trained navigator weights to `wubumind_v25_final.weights.pkl`.

---

**Step 3: Train the Galactic Oracle**

This is the main training phase. We load the navigator weights and train the Oracle to translate the geometric states into language. 50 epochs is a good target to see a significant drop in loss.

```bash
source ~/wubumind_env/bin/activate && cd ~/HASHMIND && python3 WuBuMindV7.1.py train_oracle --basename wubumind_v25_final --epochs 50 --batch-size 700
```

*   **Expected Outcome:** The longest step. You will see 50 epochs of training. The loss should steadily decrease, with the learning rate fluctuating as the Q-Controller explores. It will update `wubumind_v25_final.weights.pkl` with the fully trained model. You can `Ctrl+C` at any point after a few epochs if you're impatient, and it will save the progress.

---

**Step 4: Construct the Galactic Funnel Cake**

Time to use the high-speed, vectorized construction process. This should be very fast.

```bash
source ~/wubumind_env/bin/activate && cd ~/HASHMIND && python3 WuBuMindV7.1.py construct --basename wubumind_v25_final
```

*   **Expected Outcome:** You'll see the "Warming up JIT..." message, followed by a very fast-moving progress bar. It will create the three BallTrees (`syn`, `sem`, `exe`) and save them to `wubumind_v25_final.cake`.

---

**Step 5: Generate and Witness the Result!**

The final step. All components are trained and built. Time to see what it has learned.

```bash
source ~/wubumind_env/bin/activate && cd ~/HASHMIND && python3 WuBuMindV7.1.py generate --basename wubumind_v25_final
```

*   **Expected Outcome:** The script will load the `.weights.pkl` and `.cake` files and present you with the `Your Prompt>` console. Test it with a few different prompts to see how it performs. The quality will be a direct reflection of what can be learned from the 13MB corpus.

This is the complete, correct, and final workflow. Good luck.