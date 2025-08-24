Of course. Here is the complete, updated command list for your terminal, formatted exactly as you requested and reflecting the final two-script structure (`train.py` and `generate.py`).

This command list assumes:
-   You have saved the first script as `train.py`.
-   You have saved the second script as `generate.py`.
-   Your image directory is named `laion_aesthetics_512`.
-   You want to train and generate with a model dimension of `256`.

---

### Terminal Command List

**Step 1: Activate Environment & Navigate**
*Run this once at the start of your session.*

```bash
source ~/wubumind_env/bin/activate && cd ~/HASHMIND
```

---

**Step 2: Pre-process the Dataset (Run only once)**
*This creates the `.tfrecord` files needed for fast training and inference.*

```bash
python train.py convert-to-tfrecords --image_dir "laion_aesthetics_512"
```

---

**Step 3: Train the Model (or Resume Training)**
*This command runs the training loop. It will create or load `laion_512px_model_v1_256d.checkpoint.pkl` to save and resume progress. You can `Ctrl+C` at any time and run this exact same command again to resume.*

```bash
python train.py train --basename laion_512px_model_v1 --image_dir "laion_aesthetics_512" --epochs 500 --batch-size 4 --d-model 256
```

---

**Step 4: Create the "Funnel Cake" for Inference (After training)**
*This command uses `generate.py` to load the trained weights from the final checkpoint and builds the fast lookup index (`.cake.pkl` file). 700 is about the size for 7.4gb vram *

```bash
python generate.py construct --basename laion_512px_model_v1 --image_dir "laion_aesthetics_512" --d-model 256 --batch-size 700
```

---

**Step 5: Generate an Image**
*The final step. This uses the trained weights and the funnel cake to generate an image from your text prompt.*

```bash
python generate.py generate --basename laion_512px_model_v1 --d-model 256 --prompt "a beautiful landscape painting"
```

---
**Advanced Generation Example:**
*Generate 4 images with a different prompt and settings.*
```bash
python generate.py generate --basename laion_512px_model_v1 --d-model 256 --prompt "an epic castle in the mountains, cinematic lighting, 8k" --num-samples 4 --guidance-scale 8.0 --steps 100
```