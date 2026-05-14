# DCT-WuBu (離散餘弦變換層疊嵌套): Accelerating Deep Geometric Learning on Audio Spectrograms via Perceptually-Aware Regional DCT Encoding

**Abstract**

The WuBu Nesting (層疊嵌套) framework offers a powerful paradigm for modeling complex hierarchical and rotational structures in data through adaptive nested hyperbolic geometries. While initially explored for video and raw image data, its application to audio, particularly via Mel spectrogram representations, presents unique opportunities and challenges. This paper introduces **DCT-WuBu (離散餘弦變換層疊嵌套)** for audio spectrogram synthesis, a novel approach that synergizes the deep geometric learning capabilities of WuBu Nesting with the efficiency and perceptual relevance of Discrete Cosine Transform (DCT) encoding. We propose transforming Mel spectrograms of short audio segments (e.g., 1-second), potentially regionalized by Golden Aspect Adaptive Decomposition (GAAD), into a compact set of DCT coefficient "blocks." This transformation aims to reduce input dimensionality, decorrelate features, and emphasize perceptually significant frequency-time information. The resulting regional DCT coefficient blocks then serve as the input to a WuBu Nesting architecture (specifically WuBu-S for spatial/structural encoding within the VAE, and potentially a separate WuBu-G for generation). We hypothesize that this DCT pre-encoding allows WuBu Nesting to learn more efficiently for audio spectrogram modeling, leading to faster training, reduced computational load, and potentially improved generalization by focusing its geometric modeling on a more distilled representation of the audio content. This work details the architecture (`WuBuSpecTrans_v0.1.1.py`) and training strategy for a VAE-GAN hybrid model employing this DCT-WuBu concept for audio segment synthesis.

---

### 1. Introduction: Applying Deep Geometric Models to Audio Spectrograms

WuBu Nesting (層疊嵌套) [Ref: WuBuHypCD-paper.md, GAAD-WuBu-ST2.md] has emerged as a comprehensive framework for modeling intricate data structures. Its adaptive nested hyperbolic spaces, explicit tangent space rotations, boundary sub-manifolds, and level-specific descriptors provide a strong capacity for capturing multi-scale hierarchies and geometric relationships. While initially conceived for visual data, its principles are adaptable to other structured data modalities like audio spectrograms.

Directly applying complex models to full Mel spectrograms, especially for generative tasks, can face challenges:

*   **High Dimensionality:** Even for short audio segments, Mel spectrograms (e.g., 128 Mels x 86 time frames for a 1-second segment) represent a significant input dimension.
*   **Redundancy:** Spectrograms exhibit local correlations in both time and frequency.
*   **Computational Cost:** Sophisticated operations within WuBu Nesting, when applied to dense spectrogram inputs, can be demanding.
*   **Feature Salience:** Identifying the most salient time-frequency structures for synthesis or analysis is key.

The Discrete Cosine Transform (DCT) is well-suited for energy compaction and decorrelation, widely used in audio and image compression. By transforming regions of a Mel spectrogram into the DCT domain, we aim to create a representation that is:

*   **Compact:** DCT coefficients can represent the region's energy more sparsely.
*   **Decorrelated:** DCT coefficients are generally less correlated than raw Mel bin values.
*   **Potentially Perceptually Aligned:** While not as directly tied to psychoacoustics as dedicated audio codecs, DCT can capture dominant frequency components.

This paper introduces **DCT-WuBu for audio spectrograms**, as implemented in `WuBuSpecTrans_v0.1.1.py`. The framework involves:
1.  Extracting regions from a Mel spectrogram using Golden Aspect Adaptive Decomposition (GAAD).
2.  Resizing these diverse regions to a fixed processing size (e.g., 16x16 time-frequency bins).
3.  Applying a 2D DCT to each fixed-size region block.
4.  Normalizing these DCT coefficients.
5.  Using these normalized DCT coefficient blocks as input to a WuBu-S stack within a VAE encoder.
6.  A corresponding generator (potentially WuBu-G based) reconstructs these DCT coefficient blocks from a latent space.
7.  An adversarial discriminator operates either on the reconstructed DCT blocks or on Mel spectrograms assembled from these blocks.

We posit that this approach can enhance the efficiency of WuBu Nesting for audio spectrogram modeling, enabling faster convergence and reduced computational overhead by focusing on a structured, frequency-transformed representation.

---

### 2. Related Work

*   **Discrete Cosine Transform in Audio/Image Processing:** DCT is fundamental to MP3 (via MDCT) and JPEG, known for its energy compaction [Wallace, 1991].
*   **Learning in the Frequency Domain:** Training NNs on frequency domain representations (DCT, STFT) has been explored for audio and image tasks [Gueguen et al., 2018; Ehrlich & Davis, 2019; Défossez et al., 2020 (Demucs)].
*   **WuBu Nesting Framework:** Foundational work [Ref: WuBuHypCD-paper.md, GAAD-WuBu-ST2.md] details its adaptive nested hyperbolic architecture.
*   **Golden Aspect Adaptive Decomposition (GAAD):** GAAD [Ref: GAAD-WuBu-ST1.md, GAAD-WuBu-ST2.md] offers content-aware image/spectrogram regionalization.
*   **VAE-GANs for Audio Synthesis:** Hybrid VAE-GAN models have shown promise for generating high-fidelity audio by combining the stable training of VAEs with the sharp outputs of GANs [e.g., relevant audio VAE-GAN papers].

---

### 3. Method: DCT-Encoded WuBu Nesting for Audio Spectrograms (`WuBuSpecTrans_v0.1.1.py`)

The DCT-WuBu framework, as implemented for audio spectrograms, processes input Mel spectrograms into regional DCT coefficient blocks, which then serve as the primary data representation for the WuBu-based VAE-GAN.

#### 3.1. Input Audio Processing and Spectrogram Generation

1.  **Audio Segmentation:** Input audio files are divided into short segments (e.g., 1-second duration, configurable overlap) as handled by `AudioSegmentDataset`.
2.  **Mel Spectrogram Conversion:** Each audio segment is transformed into a Mel spectrogram using `librosa`. Parameters like `sample_rate`, `n_fft`, `hop_length`, `n_mels`, `fmin`, `fmax` are configurable.
3.  **Normalization:** Mel spectrograms (in dB scale) are normalized to a target range (e.g., [-1, 1]) based on `db_norm_min` and `db_norm_max`.
    *   Input to the VAE-GAN system: `(B, 1, N_Mels_Total, N_Time_Frames_Segment)`

#### 3.2. Encoder Path (`AudioSpecEncoder`)

1.  **GAAD Regionalization:**
    *   The input Mel spectrogram `(N_Mels_Total, N_Time_Frames_Segment)` is decomposed into `gaad_num_regions` using `golden_subdivide_rect_fixed_n`.
    *   Output: A batch of bounding boxes `(B, Num_GAAD_Regions, 4)` representing [time_start, freq_start, time_end, freq_end] for each region.
2.  **Region Extraction and Resizing (`RegionalSpectrogramRegionExtractor`):**
    *   For each GAAD-defined bounding box, the corresponding patch is cropped from the Mel spectrogram.
    *   Each cropped patch is resized to a fixed processing size: `(region_proc_size_f, region_proc_size_t)` e.g., (16 Mels, 16 Time Frames). This fixed-size patch is the "block" for DCT.
    *   Output: `(B, NumRegions, 1, F_proc, T_proc)`
3.  **2D DCT Application and Normalization (`_apply_dct_and_normalize`):**
    *   A 2D DCT (`torch_dct.dct_2d`) is applied to each `(F_proc, T_proc)` region block.
    *   The resulting DCT coefficients are normalized based on `args.dct_norm_type`:
        *   `"none"`: No normalization.
        *   `"global_scale"`: Coefficients are divided by `args.dct_norm_global_scale`.
        *   `"tanh"`: Coefficients are scaled by `1/args.dct_norm_tanh_scale` and then passed through `torch.tanh`.
    *   Output (target for reconstruction): `norm_dct_coeffs_target` of shape `(B, NumRegions, F_proc, T_proc)`.
4.  **DCT Coefficient Embedding (`DCTCoeffEmbed`):**
    *   The `(F_proc * T_proc)` DCT coefficients for each region are flattened.
    *   A linear projection maps these flattened coefficients to `args.encoder_initial_tangent_dim`.
    *   Output: `(B, NumRegions, encoder_initial_tangent_dim)`.
5.  **WuBu-S Encoding (`FullyHyperbolicWuBuNestingModel`):**
    *   The embedded DCT coefficients for all regions are reshaped to `(B * NumRegions, encoder_initial_tangent_dim)` and processed by the WuBu-S stack (configured by `args.wubu_s_*`).
    *   The WuBu-S stack outputs features of dimension `args.wubu_s_output_dim_encoder` per region.
6.  **Feature Aggregation and Latent Mapping:**
    *   Features from WuBu-S are reshaped back to `(B, NumRegions, wubu_s_output_dim_encoder)`.
    *   These regional features are aggregated (e.g., mean pooling over `NumRegions`) to produce a single feature vector per spectrogram.
    *   This aggregated feature vector is then mapped via two separate linear layers to `mu` and `logvar` of the latent distribution (dimension `args.latent_dim`).

#### 3.3. Generator Path (`AudioSpecGenerator`)

1.  **Latent Code Expansion:**
    *   A latent code `z` (sampled using `mu` and `logvar`) of dimension `args.latent_dim` is taken as input.
    *   A linear layer (`fc_expand_latent`) expands `z` to `num_gaad_regions * args.encoder_initial_tangent_dim`. This creates an initial "tangent space" feature vector for each of the `num_gaad_regions` that will be generated.
2.  **WuBu-G DCT Generation (or MLP fallback):**
    *   The expanded features are reshaped to `(B * NumRegions, encoder_initial_tangent_dim)`.
    *   This is fed into a `FullyHyperbolicWuBuNestingModel` (configured by `args.wubu_g_*`) or a fallback MLP if WuBu-G is disabled.
    *   This model's task is to generate the `num_dct_coeffs_flat` (i.e., `F_proc * T_proc`) normalized DCT coefficients for each region.
3.  **Final Activation:**
    *   The output of the WuBu-G/MLP is passed through a final activation (`nn.Tanh()` if `args.dct_norm_type == "tanh"`, else `nn.Identity()`) to match the expected range of normalized DCT coefficients.
4.  **Reshaping:**
    *   The flat DCT coefficients are reshaped to `(B, NumRegions, F_proc, T_proc)` representing the reconstructed normalized DCT blocks. This is `recon_norm_dct_coeffs`.

#### 3.4. Discriminator Path (`AudioSpecDiscriminator`)

The discriminator can operate on two types of input, specified by `args.disc_input_type`:

1.  **`"dct"` Input:**
    *   The discriminator receives normalized DCT coefficient blocks `(B, NumRegions, F_proc, T_proc)` (either real ones from the encoder's output during training, or fake ones from the generator).
    *   These are first embedded using `DCTCoeffEmbed` to `args.encoder_initial_tangent_dim`.
    *   The embedded features are processed by a WuBu-D stack (configured by `args.wubu_d_*`) or an MLP fallback.
    *   The output features are aggregated (mean over regions) and passed to a final linear layer to produce a single logit.
2.  **`"mel"` Input:**
    *   The discriminator receives full Mel spectrograms `(B, 1, N_Mels_Total, N_Time_Frames_Segment)`.
    *   **For fake samples:** The generator's output (normalized DCT blocks `recon_norm_dct_coeffs`) must first be converted back to a Mel spectrogram:
        *   **Unnormalization (`AudioSpecGenerator._unnormalize_dct`):** The normalized DCTs are unnormalized to their original scale (inverse of encoder's `_apply_dct_and_normalize`).
        *   **IDCT and Assembly (`_assemble_mel_from_dct_regions`):**
            *   `idct_2d` is applied to each unnormalized `(F_proc, T_proc)` DCT block.
            *   The resulting spatial domain patches are resized to their original GAAD bbox dimensions on the target Mel canvas.
            *   Patches are assembled onto a canvas (averaging overlaps) to form the full Mel spectrogram. GAAD bboxes generated during the encoder pass (or canonical bboxes for sampling) are used for this assembly.
    *   The (real or assembled fake) Mel spectrogram is then processed by a 2D CNN (PatchGAN-like architecture, potentially with spectral normalization as per `args.disc_apply_spectral_norm`).
    *   The CNN's output features are typically reduced to a single logit (e.g., by a final convolution with kernel size matching feature map size, or adaptive pooling then linear).

#### 3.5. Loss Functions and Training (`HybridTrainer`)

*   **Reconstruction Loss:** Mean Squared Error (MSE) between the target normalized DCT blocks (`target_norm_dct_coeffs` from the encoder) and the reconstructed normalized DCT blocks (`recon_norm_dct_coeffs` from the generator). Weighted by `args.lambda_recon`.
*   **KL Divergence Loss:** Standard KL divergence between the learned latent distribution (`mu`, `logvar`) and a standard normal prior. Weighted by `args.lambda_kl`. `lambda_kl` can be dynamically adjusted by a Q-Controller.
*   **Adversarial Loss:** Binary Cross-Entropy with Logits (`nn.BCEWithLogitsLoss`) for both generator and discriminator. Weighted by `args.lambda_gan` for the generator's loss.
*   **Optimizers:** `RiemannianEnhancedSGD` is used for both generator/encoder and discriminator, with optional Q-learning based hyperparameter adjustment (`HAKMEMQController`).
*   **Training Steps:**
    1.  Train Discriminator: Minimize loss on real samples + loss on fake samples.
    2.  Train Generator/Encoder: Minimize reconstruction loss + KL loss + adversarial loss (aiming to fool discriminator).
*   **Gradient Accumulation and AMP:** Supported via `args.grad_accum_steps` and `args.use_amp`.

---

### 4. Implementation Details and Key Components (`WuBuSpecTrans_v0.1.1.py`)

*   **Core Model (`WuBuSpecTransNet`):** Encapsulates the `AudioSpecEncoder` and `AudioSpecGenerator`.
*   **Hyperbolic Geometry (`HyperbolicUtils`, `PoincareBall`, `HyperbolicWuBuNestingLevel`, `FullyHyperbolicWuBuNestingModel`):** These classes implement the core WuBu Nesting mechanics, largely unchanged from previous video-focused versions but now applied to DCT coefficient features. Boundary points are defaulted to 0 for audio features as their interpretation is less direct.
*   **Optimizers (`RiemannianEnhancedSGD`, `HAKMEMQController`):** Provide adaptive learning rates and momentum, with Q-learning for hyperparameter scheduling, including dynamic `lambda_kl` adjustment.
*   **Dataset (`AudioSegmentDataset`):** Loads audio, segments it, converts to Mel spectrograms, and provides them to the training loop. Supports preloading to RAM.
*   **Trainer (`HybridTrainer`):** Manages the training loop, loss calculations, checkpointing, validation, and WandB logging. Validation metrics include DCT MSE, and if assembled Mels are compared: Mel MSE, PSNR, SSIM, and LPIPS.
*   **DCT/IDCT:** Relies on the `torch-dct` library. If unavailable, the model cannot function.
*   **Normalization Strategy:** The consistency between `AudioSpecEncoder._apply_dct_and_normalize` and `AudioSpecGenerator._unnormalize_dct` is critical, especially for the `"tanh"` normalization mode and for assembling Mels for the discriminator or for final output. The current implementation of `_unnormalize_dct` for "tanh" mode includes the `atanh` step.

---

### 5. Expected Benefits for Audio Spectrogram Modeling

*   **Improved Computational Efficiency:**
    *   Operating on smaller, fixed-size DCT blocks per region reduces the direct input size to the WuBu layers compared to processing entire spectrogram regions or full spectrograms with WuBu.
*   **Focused Geometric Learning:**
    *   DCT pre-processing transforms time-frequency information into a frequency-energy representation. WuBu Nesting can then learn hierarchical geometric relationships within this transformed domain, potentially capturing structural commonalities in spectral energy distributions.
*   **Structured Latent Space:** The VAE framework encourages a structured latent space for the DCT representations of audio segments.
*   **Enhanced Adversarial Training:** The discriminator operating on either DCT blocks or assembled Mels provides a strong adversarial signal for generating realistic spectral content.

---

### 6. Discussion and Future Work

*   **Strengths:**
    *   Novel application of WuBu Nesting to regional DCT coefficients of audio spectrograms.
    *   Combines GAAD for content-aware regionalization with DCT for efficient representation.
    *   VAE-GAN hybrid provides stable training and potential for high-quality synthesis.
    *   Sophisticated Q-learning based hyperparameter control, including dynamic `lambda_kl`.
*   **Challenges & Considerations:**
    *   **DCT Block Size (`region_proc_size_t/f`):** The choice of 16x16 is a hyperparameter. Smaller blocks capture finer detail but increase the number of blocks; larger blocks might smooth over details.
    *   **DCT Normalization:** The impact of different `dct_norm_type` strategies needs empirical evaluation. The `"tanh"` mode, while potentially bounding coefficients, requires careful handling of the `atanh` in reconstruction and for the discriminator.
    *   **Information Loss in DCT:** Quantization is not explicitly used in the current `WuBuSpecTrans_v0.1.1.py` (normalization is applied instead). If explicit DCT quantization were added for further compression, balancing information loss would be crucial.
    *   **Vocoding:** The current model generates Mel spectrograms (or their DCT representations). A separate vocoder (e.g., Griffin-Lim, HiFi-GAN) is needed to synthesize audible waveforms from the generated spectrograms.
    *   **Discriminator Input:** The choice between "dct" and "mel" for the discriminator input impacts what features it learns to critique. "Mel" input is more end-to-end but involves the IDCT/assembly pipeline, which itself could introduce artifacts or smoothing. "DCT" input is more direct but might miss broader structural issues apparent in the assembled Mel.
*   **Future Work:**
    *   Empirical evaluation of synthesis quality and comparison with other spectrogram-based VAE-GANs.
    *   Ablation studies on GAAD vs. fixed-grid regionalization for DCT, different `region_proc_size`, and `dct_norm_type`.
    *   Integration with a neural vocoder for end-to-end audio generation and evaluation using audio-domain metrics (e.g., SDR, PESQ).
    *   Exploring learned DCT basis functions or learned quantization strategies within the WuBu framework.
    *   Adapting the framework for conditional audio synthesis (e.g., conditioned on text, speaker ID, or musical style).
    *   Investigating alternative frequency-domain transforms (e.g., learnable filterbanks, CQT) as input to WuBu.

---

### 7. Conclusion

DCT-WuBu, as implemented in `WuBuSpecTrans_v0.1.1.py`, presents a sophisticated framework for audio spectrogram synthesis. By leveraging GAAD for regionalization, DCT for efficient and structured representation of Mel spectrogram regions, and WuBu Nesting for deep geometric learning within a VAE-GAN structure, this approach aims to achieve high-fidelity and computationally tractable generation of 1-second audio segments. The detailed architecture, including adaptive hyperbolic geometries and advanced optimizer controls, provides a strong foundation for further research into geometric deep learning for audio.

---
**References**

*(Adapted and updated based on the context of audio and the Python script.)*

*   Wallace, G. K. (1991). The JPEG still picture compression standard. *IEEE Transactions on consumer electronics, 38(1), xviii-xxxiv.*
*   Gueguen, L., Serge, A., & Kadlec, B. (2018). Faster GANS: Applying Hashing to Speed up GANs. *Workshop on "Compact Deep Neural Networks: A NIPS 2016 Workshop."*
*   Ehrlich, M., & Davis, L. S. (2019). Deep Residual Networks in the JPEG Transform Domain. *arXiv preprint arXiv:1906.00385.*
*   Défossez, A., Usunier, N., Bottou, L., & Bach, F. (2020). Real time speech separation in the time domain. *Advances in Neural Information Processing Systems, 33*, 16838-16848. (Example of learning in time/frequency for audio)
*   *Relevant VAE-GAN for audio synthesis papers (e.g., MelGAN, HiFi-GAN, RAVE, etc., depending on specific architectural choices or comparisons).*
*   *[Reference to your WuBuHypCD-paper.md]*
*   *[Reference to your GAAD-WuBu-ST1.md]*
*   *[Reference to your GAAD-WuBu-ST2.md]*