# ===================================================================
#  Step 1: Data Preparation (Only needs to be run once)
# ===================================================================
# This will create the multi-resolution TFRecord files in the specified directory.
python chimera_Resnet.py prepare-data --image-dir ./laion_aesthetics_512


# ===================================================================
#  Step 2: Training Commands
# ===================================================================
# --- Option A: Start a NEW training run from scratch (Search Mode) ---
# This uses the Q-Controller's "Search Mode" to find a good learning rate.
# NOTE: Batch size is kept low for the 512x512 cascade's VRAM requirements.
python chimera_Resnet.py train \
    --image-dir ./laion_aesthetics_512 \
    --basename laion_resnet_chimera \
    --d-model 256 \
    --batch-size 8 \
    --epochs 100 \
    --lr 3e-4 \
    --use-q-controller \
    --use-sentinel

# --- Option B: Continue training in Fine-tuning Mode ---
# This is similar to your last command. It uses the Q-Controller's "Goal-Seeking"
# mode, which is good for pushing the loss down once you're in a good spot.
python chimera_Resnet.py train \
    --image-dir ./laion_aesthetics_512 \
    --basename laion_resnet_chimera \
    --d-model 256 \
    --batch-size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --use-q-controller \
    --use-sentinel \
    --finetune


# ===================================================================
#  Step 3: Generation & Inference Commands (using the new model)
# ===================================================================
# --- Generate a single 512x512 image ---
python chimera_Resnet.py generate \
    --prompt "a photorealistic 3D render of a glass apple on a marble pedestal, dramatic lighting" \
    --basename laion_resnet_chimera \
    --d-model 256

# --- Create a smooth animation ---
python chimera_Resnet.py animate \
    --start "a photorealistic red apple" \
    --end "a crystal sculpture of an apple" \
    --steps 120 \
    --basename laion_resnet_chimera \
    --d-model 256

# --- Blend two concepts together ---
python chimera_Resnet.py blend \
    --base "a stoic roman statue" \
    --modifier "made of glowing neon lines" \
    --strength 0.6 \
    --basename laion_resnet_chimera \
    --d-model 256

# --- Iteratively refine an image towards a prompt ---
python chimera_Resnet.py refine \
    --prompt "a surrealist painting of a clock melting on a tree" \
    --steps 5 \
    --guidance-strength 0.25 \
    --basename laion_resnet_chimera \
    --d-model 256
	


### How to Use the New Staged Training Workflow

You now must train the model in four separate steps, in order. Each step loads the weights from the previous one.

**Step 1: Train the Base Model (64px)**
This trains the core components on 64x64 images and text embeddings.

```bash
python chimera_Resnet.py train \
    --image-dir ./laion_aesthetics_512 \
    --basename laion_resnet_chimera \
    --stage base \
    --d-model 256 \
    --batch-size 16 \
    --epochs 100 \
    --lr 3e-4 \
    --use-q-controller \
    --use-sentinel
```
This will create a checkpoint: `laion_resnet_chimera_256d_cascade_base.pkl`

**Step 2: Train the 128px Upscaler**
This freezes the base model and trains only the first upscaler.

```bash
python chimera_Resnet.py train \
    --image-dir ./laion_aesthetics_512 \
    --basename laion_resnet_chimera \
    --stage upscale128 \
    --d-model 256 \
    --batch-size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --use-q-controller 
```
This will create `laion_resnet_chimera_256d_cascade_upscale128.pkl`.

**Step 3: Train the 256px Upscaler**

```bash
python chimera_Resnet.py train \
    --image-dir ./laion_aesthetics_512 \
    --basename laion_resnet_chimera \
    --stage upscale256 \
    --d-model 256 \
    --batch-size 4 \
    --epochs 50 \
    --lr 1e-4 \
    --use-q-controller
```
This will create `laion_resnet_chimera_256d_cascade_upscale256.pkl`.

**Step 4: Train the 512px Upscaler (with Perceptual Loss)**

```bash
python chimera_Resnet.py train \
    --image-dir ./laion_aesthetics_512 \
    --basename laion_resnet_chimera \
    --stage upscale512 \
    --d-model 256 \
    --batch-size 1 \
    --epochs 50 \
    --lr 5e-5 \
    --perceptual-weight 0.1 \
    --use-q-controller
```
This creates the final model `laion_resnet_chimera_256d_cascade_upscale512.pkl`, which the generation commands will automatically use.

This new workflow solves both the `KeyError` and, critically, the memory problem, allowing the entire model to be trained sequentially on consumer-grade hardware.