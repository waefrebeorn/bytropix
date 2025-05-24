# Bytropix: A Playground for WuBu Nesting & GAAD

[![Status: Research Playground - Experimental](https://img.shields.io/badge/status-research%20playground%20(experimental)-orange)](https://shields.io/)
[![Discord](https://img.shields.io/discord/1303046473985818654?label=Discord&logo=discord&style=for-the-badge)](http://wubu.waefrebeorn.com) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Welcome to **Bytropix**! This repository is an open, experimental playground dedicated to exploring the theoretical frameworks of **WuBu Nesting (層疊嵌套)** and **Golden Aspect Adaptive Decomposition (GAAD)**. If you're interested in pushing the boundaries of how we can model complex data, especially video, with deep geometric and compositional priors, you're in the right place.

Think of this as a collection of **experimental scaffolds and starting beds**. The Python scripts and batch files here are my attempts to implement these advanced mathematical theories. They are functional, but more importantly, they are designed to be pulled apart, improved, and used as inspiration. The goal isn't a polished, final product, but a space to collectively figure out what works, what doesn't, and how to make these powerful ideas better. This is for those who like to get their hands dirty with "theory math" that might be a tough sell to the "tech head goobers" without seeing it in action (or at least, in attempted action!).

## Core Theories Under Exploration

At the heart of Bytropix are two main theoretical constructs:

1.  **WuBu Nesting (層疊嵌套):** A framework for building models with recursively nested hyperbolic spaces (`H^{n_i}_{c_i,s_i}`). The geometry of these spaces (dimensionality `n_i`, curvature `c_i`, scale `s_i`) can adapt during learning. Key features include learnable Boundary Sub-Manifolds, Level Descriptor Vectors, Level Spread Parameters, Intra-Level Tangent Flows, and, crucially, inter-level transitions orchestrated in tangent space with explicit `SO(n_i)` Rotations and Mappings. This aims to capture deep multi-scale hierarchies and rotational dynamics.
    *Visualizing Nested Hyperbolic Levels (Conceptual Example from `wubu_nesting_example.py`):*
    ![Nested Hyperbolic Levels](https://github.com/waefrebeorn/bytropix/raw/master/wubu_results/visualizations/nested_spheres_epoch_10.png)
    *This image illustrates how different levels can have varying curvatures, scales, and dimensionalities, with data points (projected to 3D) residing within their respective adaptive geometric "bubbles."*

2.  **Golden Aspect Adaptive Decomposition (GAAD):** A method inspired by the Golden Ratio (φ) for decomposing visual data (like video frames) into multi-scale, aspect-ratio agnostic regions. It uses techniques like Recursive Golden Subdivision and Phi-Spiral Patching to guide feature extraction, respecting natural compositions.

The primary application currently explored here is **WuBuGAADHybridGen** (a VAE-GAN model) and previously **WuBuNestDiffusion**, models built upon these principles, aiming to generate and understand video with a strong geometric and compositional foundation.

## The Playground: What's Inside?

This repository is a collection of my explorations and implementations:

*   **Python Scripts (`draftPY/`, root):** You'll find various Python files representing different stages and versions of models. These include:
    *   `WuBuGAADHybridGen_v0.3.py`: The **current VAE-GAN model focus**, incorporating WuBu Nesting, GAAD, dual spectral features (DFT+DCT), dual discriminators, and advanced training heuristics. The `WuBuGAADHybridGen_v0.1_TRAINING_GUIDE.md` (despite its v0.1 name) and `WuBuGAADHybridGen_v0.3_TRAINING_GUIDE.md` (more recent) provide extensive phased training strategies.
    *   Older Diffusion Model: `draftPY/WuBuNestDiffusion_v0.10.1_OpticalFlow.py` and earlier versions like `draftPY/WuBuNestDiffusion_v0.05_GAAD_MotionWuBu_Live.py`.
    *   Supporting modules: `RiemannianEnhancedSGD.py` (custom optimizer with Q-Controller support), `HAKMEMQController.py` (Q-learning for hyperparameter tuning), components for hyperbolic math, spectral transforms, and GAAD.
    *   Utility and data generation scripts.
    *   Example implementations and visualizations for core WuBu concepts: `wubu_nesting_example.py`, `wubu_nesting_impl.py`, `wubu_nesting_visualization.py`.
*   **Batch Files (`.bat`):** These are your primary interface for running experiments! They call the Python scripts with a multitude of command-line arguments, allowing you to tweak parameters, switch components, and test different configurations of the theories.
*   **Theoretical Documents (The "Why"):**
    *   Markdown Papers: Deeper dives into specific aspects and the main models. These provide the context for the code:
        *   [`./WuBuHypCD-paper.md`](./WuBuHypCD-paper.md) (Foundational WuBu Nesting theory)
        *   [`./WuBuNestingFindings5.19.25.md`](./WuBuNestingFindings5.19.25.md) (Advanced conceptual insights for WuBu)
        *   [`./GAAD-WuBu-ST1.md`](./GAAD-WuBu-ST1.md) & [`./GAAD-WuBu-ST2.md`](./GAAD-WuBu-ST2.md) (GAAD and Spatio-Temporal WuBu, foundational for video applications)
        *   [`./WuBu Spatio-Temporal Nesting.md`](./WuBu%20Spatio-Temporal%20Nesting.md) (Focus on the temporal aspect of WuBu)
        *   [`./draftPY/WuBuNestDiffusion (v0.05.2).md`](./draftPY/WuBuNestDiffusion%20(v0.05.2).md) & [`./draftPY/DFT-WuBu.md`](./draftPY/DFT-WuBu.md) & [`./draftPY/DCT-WuBu.md`](./draftPY/DCT-WuBu.md) (Specifics for diffusion and spectral transform variants)
        *   [`./draftPY/WuBuGAADHybridGen_v0.1_TRAINING_GUIDE.md`](./draftPY/WuBuGAADHybridGen_v0.1_TRAINING_GUIDE.md) & [`./draftPY/WuBuGAADHybridGen_v0.3_TRAINING_GUIDE.md`](./draftPY/WuBuGAADHybridGen_v0.3_TRAINING_GUIDE.md) (Practical guides for training the VAE-GAN models)

## Navigating the Codebase

```
└── ./
    ├── draftPY/                             # Primary hub for Python experiments & latest models
    │   ├── WuBuGAADHybridGen_v0.3.py        # << CURRENT VAE-GAN MODEL (DFT+DCT, Dual-D)
    │   ├── WuBuGAADHybridGen_v0.2.py        # Previous VAE-GAN (DFT only)
    │   ├── WuBuNestDiffusion_v0.10.1_OpticalFlow.py # Recent Diffusion model
    │   ├── RiemannianEnhancedSGD.py         # Custom optimizer
    │   ├── HAKMEMQController.py             # Q-learning hyperparameter controller
    │   ├── *.bat                            # Batch files to RUN THE EXPERIMENTS
    │   └── ... (many other Python scripts: older models, utilities)
    ├── wubu_results/                        # Directory for storing results, visualizations
    │   └── visualizations/
    │       └── nested_spheres_epoch_10.png  # Example visualization
    ├── requirements.txt                     # Dependencies
    ├── setup.bat, venv.bat                  # Windows environment setup
    ├── WuBuGAADHybridGen_v0.3_TRAINING_GUIDE.md # This document! (Or its latest version)
    └── ... (other utilities, markdown papers)
```

**The general idea:**
1.  Read the papers (start with foundational ones like `WuBuHypCD-paper.md`, then specific model papers like `GAAD-WuBu-ST2.md`, and then the `v0.3` training guide) to understand the theory.
2.  Examine the Python scripts in `draftPY/` (especially `WuBuGAADHybridGen_v0.3.py` and its supporting modules).
3.  Use the `.bat` files as templates to run your own experiments, modifying parameters according to the training guides and your hypotheses.

## Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/waefrebeorn/bytropix.git 
    cd bytropix
    ```

2.  **Set up Environment (Windows):**
    *   A virtual environment is **highly recommended**.
    *   Run `setup.bat`. This attempts to create a `venv` and install dependencies.
    *   Activate: `.\venv.bat` (or `.\venv\Scripts\activate`).

3.  **Install Dependencies:**
    *   If `setup.bat` doesn't cover everything, or for manual setup:
        ```bash
        pip install -r requirements.txt
        ```
    *   **Crucial for v0.3:** Ensure `torch-dct` is installed: `pip install torch-dct`.
    *   Ensure PyTorch is installed with **CUDA support** for GPU acceleration. Verify versions are compatible.

4.  **Data:**
    *   The scripts can create a dummy video (e.g., `dummy_video_hybridgen_v03.mp4`) if `VIDEO_DATA_PATH` points to the default demo directory and the file is missing.
    *   For actual training, prepare your video datasets and update `VIDEO_DATA_PATH` (and optionally `VALIDATION_VIDEO_PATH`) in your `.bat` execution script.

## Running Experiments & Exploring (`WuBuGAADHybridGen_v0.3.py` Focus)

This is where the "playground" comes alive! We'll focus on the current `v0.3` VAE-GAN model.

1.  **The Master `.bat` File:**
    *   You have a comprehensive `.bat` script for `v0.3`. This is your primary tool.
    *   It sets environment variables for all hyperparameters and then constructs the command to run `WuBuGAADHybridGen_v0.3.py`.

2.  **Phased Training (See `WuBuGAADHybridGen_v0.3_TRAINING_GUIDE.md` for details):**
    *   The key to training these complex models is a phased approach.
    *   **Phase 0 (Optional Sanity Check):** Minimal settings to ensure the pipeline runs.
    *   **Phase 1 (VAE Reconstruction):** Focus on getting good reconstruction of spectral features (DFT/DCT) or pixels if spectral features are off. Keep `LAMBDA_KL` and `LAMBDA_GAN` very low. Use primary D, heuristics mostly off.
    *   **Phase 2 (Introduce GAN):** Gradually increase `LAMBDA_GAN`. Primary D still active.
    *   **Phase 3 (KL Annealing & Core Heuristics):** Increase `LAMBDA_KL` (or let Q-controller manage it). Enable core training heuristics.
    *   **Phase 4 (Dual Discriminator & Full Heuristics):** Enable `ENABLE_HEURISTIC_DISC_SWITCHING` and all relevant heuristics.
    *   **Phase 5 (Fine-tuning):** Long runs for convergence, potential manual tweaks.

3.  **Modifying the `.bat` file:**
    *   Open your main run script (`.bat` file for `v0.3`).
    *   To change a hyperparameter for the next phase/experiment:
        1.  Modify its `SET "PARAM_NAME=NEW_VALUE"` line.
        2.  If resuming or starting a new phase, set `SET "LOAD_CHECKPOINT=path\to\your\best_checkpoint_from_previous_phase.pt"`.
    *   Run the `.bat` file.

4.  **Interpreting Outcomes:**
    *   **Console Logs & WandB**: Your primary sources of information. Track spectral reconstruction losses (`RecDFT`, `RecDCT`), pixel recon (if applicable), `KL`, adversarial losses (`Adv`, `D_tot`), validation metrics (`PSNR`, `SSIM`, `LPIPS`), Q-Controller decisions, and heuristic activations.
    *   **WandB Image Samples**:
        *   `train_recon_pixels`, `val_predicted_frames`: How well is the VAE reconstructing visual appearance (after IDFT/IDCT if applicable)?
        *   `fixed_noise_generated_pixels`: Quality and diversity from fixed latent vectors.
    *   **"What I did wrong, and how to make it better"**: This iterative process is central to the playground. If training is unstable or quality is poor, analyze the logs and metrics, consult the training guide's troubleshooting section, adjust parameters in the `.bat` file, and try again, loading from the last good checkpoint.

## Visualizations (Expected)

As training progresses, especially when using WandB:
*   **Loss Curves:** Track all loss components (Recon DFT/DCT/Pixel, KL, G_Adv, D_Total for active D).
*   **Validation Metrics:** PSNR, SSIM, LPIPS curves should show improvement.
*   **Generated Image Samples:** Progress from noise to coherent images/videos.
*   **Q-Controller Stats:** Epsilon decay, chosen LR/Momentum/Lambda_KL scales, rewards.
*   **Heuristic & Discriminator Switching Logs:** To understand adaptive training behavior.

## Key Hyperparameters in `.bat` for `WuBuGAADHybridGen_v0.3.py`

*   Spectral Transform Toggles: `USE_DFT_FEATURES_APPEARANCE`, `USE_DCT_FEATURES_APPEARANCE`.
*   Spectral Patch Sizes: `SPECTRAL_PATCH_SIZE_H`, `SPECTRAL_PATCH_SIZE_W`.
*   Discriminator Variants: `PRIMARY_DISC_ARCHITECTURE_VARIANT`, `ALT_DISC_ARCHITECTURE_VARIANT`.
*   Heuristic Controls: `ENABLE_HEURISTIC_DISC_SWITCHING`, `ENABLE_HEURISTIC_INTERVENTIONS`, and all their sub-parameters.
*   Loss Weights: `LAMBDA_RECON_DFT`, `LAMBDA_RECON_DCT`, `LAMBDA_KL` (base), `LAMBDA_GAN` (base).
*   Learning Rates: `LEARNING_RATE_GEN`, `LEARNING_RATE_DISC`, `LEARNING_RATE_DISC_ALT`.
*   WuBu stack parameters (e.g., `WUBU_S_NUM_LEVELS`, `WUBU_D_GLOBAL_VIDEO_HYPERBOLIC_DIMS`, etc.).
*   `Q_CONTROLLER_ENABLED` and Q-controller JSON override paths.

## Spirit of the Project & Limitations

*   **This is Research in Progress:** The code is experimental. It's a vehicle for exploring difficult theories. Expect rough edges and debugging cycles.
*   **Iterative Improvement**: The core ethos is to learn from experiments. If theory suggests X, and the code implementing X needs tuning or refinement, that's the process.
*   **Computational Demands**: These are complex models; training can be resource-intensive.

## Contributing / Feedback

This is a personal exploration ground, but insights and discussions are welcome!
*   If you run experiments and find interesting behavior (good or bad).
*   If you spot discrepancies between the theoretical documents and the code.
*   If you have ideas for improving the stability or performance of these theoretical models.
*   Best way to provide feedback currently is via the [Discord Server](http://wubu.waefrebeorn.com).

## License

This project is licensed under the MIT License - see the [LICENSE](./.LICENSE) file for details.

## Acknowledgments

The ideas explored here build upon a vast body of work in hyperbolic geometry, geometric deep learning, compositional theories, spectral analysis, and video modeling. Specific theoretical inspirations are cited within the accompanying papers.

## Citation

If you find the theories (WuBu Nesting, GAAD), the codebase, or the papers useful in your research, please consider citing the source documents:

```
@misc{BytropixPlayground2025,
  author       = {W. WaefreBeorn and Collaborators},
  title        = {Bytropix: An Experimental Playground for WuBu Nesting & Golden Aspect Adaptive Decomposition},
  year         = {2025},
  howpublished = {GitHub Repository},
  note         = {URL: https://github.com/waefrebeorn/bytropix (Replace with actual URL)}
}

@techreport{WaefreBeornWuBuNestingFramework,
  author       = {W. WaefreBeorn},
  title        = {WuBu Nesting (層疊嵌套): A Comprehensive Geometric Framework for Adaptive Multi-Scale Hierarchical Representation with Integrated Rotational Dynamics},
  year         = {2024-2025},
  institution  = {Bytropix Project (Self-Published)},
  note         = {Available at Bytropix GitHub repository. See WuBuHypCD-paper.md or WuBu_Nesting.pdf.}
}

@techreport{WaefreBeornGAADWuBuST,
  author       = {W. WaefreBeorn},
  title        = {GAAD-WuBu-ST: A Golden Ratio-Infused, Adaptive, Rotation-Aware, Nested Hyperbolic Framework for Aspect-Ratio Agnostic Video Understanding},
  year         = {2024-2025},
  institution  = {Bytropix Project (Self-Published)},
  note         = {Available at Bytropix GitHub repository. See GAAD-WuBu-ST1.md or GAAD-WuBu-ST2.md.}
}
```
