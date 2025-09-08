Of course. Here is the updated command list that reflects the recent changes to the script, specifically the consolidation of video compression into a single, efficient `.wubu` file.

---

### **Phase 1: Static Image Model**

**1. Prepare Image Data**
*   **Purpose:** Scans a directory of images and creates a `.tfrecord` file for efficient training.
*   **Command:**
    ```bash
    python QAE_Advanced.py prepare-data --image-dir ./my_images/
    ```

**2. Train the Image Autoencoder**
*   **Purpose:** Trains the main model on your prepared image dataset. This creates the base model checkpoint (`.pkl` file).
*   **Command:**
    ```bash
    python QAE_Advanced.py train --image-dir ./my_images/ --basename my_topo_coord_12864 --d-model 128 --latent-grid-size 64 --batch-size 8 --epochs 100
    ```

**3. Compress an Image**
*   **Purpose:** Takes a standard image (e.g., PNG, JPG) and compresses it into the model's latent format (`.npy` file).
*   **Command:**
    ```bash
    python QAE_Advanced.py compress --basename my_topo_coord_12864 --d-model 128 --latent-grid-size 64 --image-path ./input_photo.png --output-path ./compressed_photo.npy
    ```

**4. Decompress an Image**
*   **Purpose:** Takes a compressed `.npy` file and reconstructs it back into a viewable image.
*   **Command:**
    ```bash
    python QAE_Advanced.py decompress --basename my_topo_coord_12864 --d-model 128 --latent-grid-size 64 --compressed-path ./compressed_photo.npy --output-path ./reconstructed_photo.png
    ```

---

### **Phase 2: Video Model**

**5. Prepare Video Data**
*   **Purpose:** Extracts frames and calculates optical flow from a video file, preparing it for video model training.
*   **Command:**
    ```bash
    python QAE_Advanced.py prepare-video-data --video-path input_video.mp4 --data-dir ./prepared_video_data/
    ```

**6. Train the Video Dynamics Model**
*   **Purpose:** Loads the pre-trained Phase 1 model and trains only the small "correction network" to predict how latent spaces change over time based on optical flow.
*   **Command:**
    ```bash
    python QAE_Advanced.py train-video --data-dir ./prepared_video_data/ --basename my_topo_coord_12864 --d-model 128 --latent-grid-size 64 --batch-size 2 --clip-len 8
    ```

**7. Compress a Video**
*   **Purpose:** Compresses a video file into a single, efficient custom file (`.wubu`) containing the I-frame and all P-frame data.
*   **Command:**
    ```bash
    python QAE_Advanced.py video-compress --basename my_topo_coord_12864 --d-model 128 --latent-grid-size 64 --video-path input_video.mp4 --output-path compressed_video.wubu --batch-size 32
    ```

**8. Decompress a Video**
*   **Purpose:** Reconstructs the video from a single compressed `.wubu` file.
*   **Command:**
    ```bash
    python QAE_Advanced.py video-decompress --basename my_topo_coord_12864 --d-model 128 --latent-grid-size 64 --input-path compressed_video.wubu --output-path reconstructed_video.mp4
    ```

---

### **Phase 3: Generative AI**

**9. Build the Latent Database**
*   **Purpose:** Encodes all images from your training dataset into their latent representations and calculates their CLIP embeddings. This is required for text-to-image generation.
*   **Command:**
    ```bash
    python QAE_Advanced.py build-db --basename my_topo_coord_12864 --d-model 128 --latent-grid-size 64 --image-dir ./my_images/ --batch-size 16
    ```

**10. Generate an Image from a Text Prompt**
*   **Purpose:** Finds the image in your database that best matches the text prompt and decodes its latent representation to create a new image.
*   **Command:**
    ```bash
    python QAE_Advanced.py generate --basename my_topo_coord_12864 --d-model 128 --latent-grid-size 64 --image-dir ./my_images/ --prompt "a beautiful sunset over the ocean"
    ```

**11. Animate Between Two Prompts**
*   **Purpose:** Finds the best matching images for a start and end prompt, then interpolates between their latent representations to create a smooth animation (GIF).
*   **Command:**
    ```bash
    python QAE_Advanced.py animate --basename my_topo_coord_12864 --d-model 128 --latent-grid-size 64 --image-dir ./my_images/ --start "a photo of a cat" --end "a photo of a dog" --steps 60
    ```

### How to Use the New Script:

Save the Code: Save the code above as `QAE_Advanced.py`.

**Continue Phase 1 Training:** You can pick up exactly where you left off, but now with the new tools.
```bash
# Activate your environment
source ~/wubumind_env/bin/activate && cd ~/HASHMIND

# Continue training, but now with the Q-Controller and Sentinel
python QAE_Advanced.py train \
    --image-dir ./laion_aesthetics_512/ \
    --basename my_topo_coord_12864 \
    --d-model 128 \
    --latent-grid-size 64 \
    --batch-size 2 \
    --epochs 10000 \
    --use-q-controller \
    --use-sentinel
```
You will see a yellow warning message: `Warning: Optimizer state mismatch... Re-initializing optimizer.`. This is expected and correct! It means the script has successfully loaded your old model weights and is now starting with a fresh, advanced optimizer to continue training.

**Run Phase 2 Video Training:** After you're satisfied with the Phase 1 model, you can proceed to Phase 2. The process is the same as before, but you call the new script.
```bash
# Prepare video data (if you haven't already)
# python QAE_Advanced.py prepare-video-data --video-path input_video.mp4 --data-dir ./prepared_video_data/

# Start Phase 2 training
python QAE_Advanced.py train-video \
    --data-dir ./prepared_video_data/ \
    --basename my_topo_coord_12864 \
    --d-model 128 \
    --latent-grid-size 64 \
    --batch-size 2 \
    --clip-len 8 \
    --use-q-controller \
    --use-sentinel
```
This will now work correctly, loading the Phase 1 model and starting the training for the `CorrectionNetwork` with the advanced training chassis, fixing the errors you originally had.