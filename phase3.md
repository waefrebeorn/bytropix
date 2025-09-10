Of course. Here is a brand new, comprehensive command list formatted as a clear, step-by-step guide. It showcases the new advanced training features you've integrated, explaining how and when to use them for the best results.

---

## 🚀 HASHMIND Phase 3: Generative Workflow Commands

This guide outlines the complete workflow for training and using the text-to-image generative model, incorporating the advanced training toolkit for optimal performance.

### **Step 1: Prepare the Paired Dataset**

First, we must process the raw image/text data into a format the model can use: latent vectors from the Phase 1 Autoencoder paired with CLIP text embeddings. This only needs to be done once per dataset.

*You must specify the parameters (`--d-model`, `--latent-grid-size`) of the Phase 1 model you are using.*

```bash
# Example: Using a Phase 1 model with d_model=96 and latent_grid_size=96
python phase3_generative.py prepare-paired-data \
  --data-dir laion_aesthetics_512 \
  --basename my_model \
  --d-model 96 \
  --latent-grid-size 96 \
  --batch-size 64
```
> **What this does:** Creates a file named `paired_data_laion_ae_96d.pkl` inside the `laion_aesthetics_512` directory.

### New Workflow
Your new, memory-safe workflow for training the tokenizer will be:
First, run the new preprocessing script:
code
```bash
python phase3_generative.py prepare-tokenizer-data \
  --data-dir laion_aesthetics_512 \
  --basename my_model \
  --d-model 96 \
  --latent-grid-size 96 \
  --batch-size 64
```

This will create a tokenizer_latents_my_model.pkl file in your data directory.
---

### **Step 2: Train the VQ-VAE Tokenizer**

The tokenizer learns to convert the continuous latent space of the Autoencoder into a discrete set of "visual words" or codes.

#### **Option A: Basic Training**
A standard training run without the advanced toolkit. Good for a baseline or quick tests.

```bash
python phase3_generative.py train-tokenizer \
  --data-dir laion_aesthetics_512 \
  --basename my_model \
  --d-model 96 \
  --latent-grid-size 96 \
  --steps 20000 \
  --batch-size 8 \
  --num-codes 8192
```

#### **Option B: Advanced Training (Recommended)**
Leverage the full toolkit for a more stable and efficient training process.

```bash
# This command enables the Q-Controller for adaptive LR and Sentinel for stability.
# Using bfloat16 is highly recommended on modern GPUs to save VRAM. LOWER BATCH SAVE IS FASTER AND TRAINS BETTER, NO LIKE BATCHING IS WORSE OVERALL???
python phase3_generative.py train-tokenizer \
  --data-dir laion_aesthetics_512 \
  --basename my_model \
  --d-model 96 \
  --latent-grid-size 96 \
  --steps 25000 \
  --batch-size 1 \
  --num-codes 8192 \
  --use-q-controller \
  --use-sentinel \
  --use-bfloat16
```
> **What this does:** Creates `tokenizer_laion_ae_96d_8192c.pkl` and its config file.

---

### **Step 3: Train the Generative Conductor (Transformer)**

This is the core generative model. It learns to predict the sequence of visual tokens (from the tokenizer) based on a text prompt. The advanced training GUI will activate for this step.

#### **Option A: Basic Training**
A standard training run.

```bash
python phase3_generative.py train-conductor \
  --data-dir laion_aesthetics_512 \
  --basename my_model \
  --latent-grid-size 96 \
  --steps 150000 \
  --batch-size 1 \
  --num-codes 8192 \
  --num-layers 12 \
  --d-model-cond 768 \
  --num-heads 12
```

#### **Option B: Advanced Training with Interactive GUI (Recommended)**
This is the most powerful way to train. It provides a full dashboard and allows for real-time control over the Sentinel optimizer.

```bash
# The "Kitchen Sink" - all advanced features enabled for the best training experience.
# The interactive GUI will appear in your terminal. Use ↑/↓ to adjust Sentinel.
python phase3_generative.py train-conductor \
  --data-dir laion_aesthetics_512 \
  --basename my_model \
  --latent-grid-size 96 \
  --steps 150000 \
  --batch-size 1 \
  --num-codes 8192 \
  --num-layers 12 \
  --d-model-cond 768 \
  --num-heads 12 \
  --use-q-controller \
  --use-sentinel \
  --use-bfloat16
```

#### **Option C: Finetuning an Existing Model**
Use this to resume training with a lower learning rate and more aggressive Q-Controller settings, perfect for pushing the loss down on a nearly-converged model.

```bash
python phase3_generative.py train-conductor \
  --data-dir laion_aesthetics_512 \
  --basename my_model \
  --latent-grid-size 96 \
  --steps 50000 \
  --batch-size 32 \
  --num-codes 8192 \
  --num-layers 12 \
  --d-model-cond 768 \
  --num-heads 12 \
  --lr 5e-5 \
  --use-q-controller \
  --use-sentinel \
  --use-bfloat16 \
  --finetune
```
> **What this does:** Creates `conductor_laion_ae_96d_12l.pkl` and its config file.

---

### **Step 4: ✨ Generate and Edit Images ✨**

After training, you can generate and edit images. The script will automatically find all the necessary model files (`conductor_*.pkl`, `tokenizer_*.pkl`, `laion_ae_96d_512.pkl`) based on the `--basename`.

#### **Generate from a Text Prompt**

```bash
# Simple generation
python phase3_generative.py generate \
  --basename my_model \
  --prompt "An epic fantasy landscape with a dragon flying over a castle, digital art"

# Generation with more control over the output
python phase3_generative.py generate \
  --basename my_model \
  --prompt "A photorealistic close-up of a honeybee on a sunflower" \
  --seed 12345 \
  --temp 0.95 \
  --top-k 512
```

#### **Edit an Existing Image with a New Prompt**

```bash
# First, generate a source image (or use any other image)
python phase3_generative.py generate --basename my_model --prompt "a cute corgi dog sitting in a field of flowers" --seed 42

# Now, edit the generated image with a new prompt
python phase3_generative.py edit \
  --basename my_model \
  --source-image "GEN_laion_ae_96d_a_cute_corgi_dog_sitting_in_a_field_o_42.png" \
  --prompt "A robot corgi dog sitting in a field of metallic flowers, sci-fi"
```