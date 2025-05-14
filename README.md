# Bytropix: A Playground for WuBu Nesting & GAAD

[![Status: Research Playground - Experimental](https://img.shields.io/badge/status-research%20playground%20(experimental)-orange)](https://shields.io/)
[![Discord](https://img.shields.io/discord/1303046473985818654?label=Discord&logo=discord&style=for-the-badge)](http://wubu.waefrebeorn.com) 

Welcome to **Bytropix**! This repository is an open, experimental playground dedicated to exploring the theoretical frameworks of **WuBu Nesting (層疊嵌套)** and **Golden Aspect Adaptive Decomposition (GAAD)**. If you're interested in pushing the boundaries of how we can model complex data, especially video, with deep geometric and compositional priors, you're in the right place.

Think of this as a collection of **experimental scaffolds and starting beds**. The Python scripts and batch files here are my attempts to implement these advanced mathematical theories. They are functional, but more importantly, they are designed to be pulled apart, improved, and used as inspiration. The goal isn't a polished, final product, but a space to collectively figure out what works, what doesn't, and how to make these powerful ideas better. This is for those who like to get their hands dirty with "theory math" that might be a tough sell to the "tech head goobers" without seeing it in action (or at least, in attempted action!).

## Core Theories Under Exploration

At the heart of Bytropix are two main theoretical constructs:

1.  **WuBu Nesting (層疊嵌套):** A framework for building models with recursively nested hyperbolic spaces (`H^{n_i}_{c_i,s_i}`). The geometry of these spaces (dimensionality `n_i`, curvature `c_i`, scale `s_i`) can adapt during learning. Key features include learnable Boundary Sub-Manifolds, Level Descriptor Vectors, Level Spread Parameters, Intra-Level Tangent Flows, and, crucially, inter-level transitions orchestrated in tangent space with explicit `SO(n_i)` Rotations and Mappings. This aims to capture deep multi-scale hierarchies and rotational dynamics.
2.  **Golden Aspect Adaptive Decomposition (GAAD):** A method inspired by the Golden Ratio (φ) for decomposing visual data (like video frames) into multi-scale, aspect-ratio agnostic regions. It uses techniques like Recursive Golden Subdivision and Phi-Spiral Patching to guide feature extraction, respecting natural compositions.

The primary application currently explored here is **WuBuNestDiffusion** (and more recently, a **VAE-GAN variant**), models built upon these principles, aiming to generate and understand video with a strong geometric and compositional foundation.

## The Playground: What's Inside?

This repository is a collection of my explorations and implementations:

*   **Python Scripts (`draftPY/`, root):** You'll find various Python files representing different stages and versions of models. These include:
    *   `WuBuGAADHybridGen_v0.1.py`: The **current focus VAE-GAN model** incorporating WuBu Nesting and GAAD for appearance and motion, which this `TRAINING_GUIDE.md` primarily addresses.
    *   Older Diffusion Model: `draftPY/WuBuNestDiffusion_v0.05_GAAD_MotionWuBu_Live.py` (and `..._PaperCopy.py`).
    *   Trainers for different variants: `WuBuNest_TrainerV1.py`, `WuBuNestmRnaTrainerV1.py`. These explore different aspects or applications of WuBu Nesting.
    *   Inference scripts: `inference.py`, `sfin_inference.py`, `WuBuNest_Inferencev1.py`.
    *   Supporting modules: `RiemannianEnhancedSGD.py` (custom optimizer with Q-Controller support), `HAKMEMQController.py` (Q-learning for hyperparameter tuning), components for hyperbolic math.
    *   Utility and data generation scripts: `create_demo_data.py`, `poem_dataset_generator.py`.
*   **Batch Files (`.bat`):** These are your primary interface for running experiments! They call the Python scripts with a multitude of command-line arguments, allowing you to tweak parameters, switch components, and test different configurations of the theories.
*   **Theoretical Documents (The "Why"):**
    *   [`WuBu_Nesting.pdf`](./WuBu_Nesting.pdf): The core PDF detailing the foundational WuBu Nesting framework. *(The papers often use more elaborate naming for concepts for "eloquence and appeal to goobers," but the core is WuBu Nesting & GAAD.)*
    *   Markdown Papers: Deeper dives into specific aspects and the main models. These provide the context for the code:
        *   [`./draftPY/WuBuNestDiffusion (v0.05.2).md`](./draftPY/WuBuNestDiffusion%20(v0.05.2).md) (For the diffusion variant)
        *   [`./WuBuHypCD-paper.md`](./WuBuHypCD-paper.md) (Foundational WuBu Nesting)
        *   [`./GAAD-WuBu-ST1.md`](./GAAD-WuBu-ST1.md) & [`./GAAD-WuBu-ST2.md`](./GAAD-WuBu-ST2.md) (GAAD and Spatio-Temporal WuBu)
        *   [`./WuBu Spatio-Temporal Nesting.md`](./WuBu%20Spatio-Temporal%20Nesting.md)

## Navigating the Codebase

```
└── ./
    ├── draftPY/                             # Primary hub for Python experiments & latest models
    │   ├── WuBuGAADHybridGen_v0.1.py        # << CURRENT VAE-GAN MODEL
    │   ├── WuBuNestDiffusion_v0.05_GAAD_MotionWuBu_Live.py  # Previous Diffusion model
    │   ├── RiemannianEnhancedSGD.py         # Custom optimizer
    │   ├── HAKMEMQController.py             # Q-learning hyperparameter controller
    │   ├── *.bat                            # Batch files to RUN THE EXPERIMENTS
    │   └── ... (many other Python scripts: inference, older models, utilities)
    ├── WuBu_Nesting.pdf                     # Main PDF for WuBu Nesting theory
    ├── requirements.txt                     # Dependencies
    ├── setup.bat, venv.bat                  # Windows environment setup
    ├── TRAINING_GUIDE.md                    # This document!
    └── ... (other utilities, data generators like poem_dataset_generator.py, markdown papers)
```

**The general idea:**
1.  Read the papers (especially `WuBu_Nesting.pdf` and relevant markdown docs for the model version you're interested in) to understand the theory.
2.  Look at the Python scripts in `draftPY/` (especially `WuBuGAADHybridGen_v0.1.py` and its supporting modules for the current VAE-GAN experiments).
3.  Use the `.bat` files (e.g., the one you've been running for `WuBuGAADHybridGen_v0.1.py`) as templates to run your own experiments, changing parameters to test different hypotheses.

## Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/waefrebeorn/bytropix.git # Replace with actual repo URL if public
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
    *   Ensure PyTorch is installed with **CUDA support** for GPU acceleration. Verify versions are compatible (see `requirements.txt` and PyTorch website).

4.  **Data:**
    *   The scripts can create a dummy video if `VIDEO_DATA_PATH` points to `demo_video_data_dir` and the video file is missing (see `.bat` file logic). This is useful for initial functionality tests.
    *   For actual training, prepare your video datasets and update `VIDEO_DATA_PATH` (and optionally `VALIDATION_VIDEO_PATH`) in your `.bat` execution script or command line.

## Running Experiments & Exploring (`WuBuGAADHybridGen_v0.1.py` Focus)

This is where the "playground" comes alive! We'll focus on the current VAE-GAN model.

1.  **The Master `.bat` File:**
    *   You have a comprehensive `.bat` script (like the one you've been sharing in the logs). This is your primary tool.
    *   It sets environment variables for all hyperparameters and then constructs the command to run `WuBuGAADHybridGen_v0.1.py`.

2.  **Phased Training (See `TRAINING_GUIDE.md` for details):**
    *   The key to training these complex models is a phased approach. Don't try to turn everything on at once!
    *   **Phase 1: Reconstruction First**
        *   Set `LAMBDA_KL` to a very small value (e.g., `0.0001`).
        *   Set `LEARNING_RATE_GEN` moderately (e.g., `3e-4`) and `LEARNING_RATE_DISC` lower (e.g., `1e-4`).
        *   Keep `DISC_USE_GAAD_FILM_CONDITION=false`.
        *   **Goal:** Achieve good reconstruction loss (`Rec` decreasing, PSNR/SSIM improving). Visuals should go from noise to blurry images.
    *   **Phase 2: KL Annealing**
        *   Load the best checkpoint from Phase 1.
        *   Gradually increase `LAMBDA_KL` in steps (e.g., `0.001` -> `0.01` -> `0.1`).
        *   **Goal**: Regularize the latent space while maintaining reconstruction.
    *   **Phase 3: Enhance Discriminator (Optional)**
        *   Load the best checkpoint from Phase 2.
        *   Set `DISC_USE_GAAD_FILM_CONDITION=true`.
        *   **Goal**: Improve regional details if the unconditional D isn't sufficient.
    *   **Phase 4: Fine-tuning**
        *   Adjust LRs, `LAMBDA_GAN` for final quality.

3.  **Modifying the `.bat` file:**
    *   Open your main run script (`.bat` file).
    *   To change a hyperparameter for the next phase:
        1.  Modify its `SET "PARAM_NAME=NEW_VALUE"` line.
        2.  Crucially, set `SET "LOAD_CHECKPOINT=path\to\your\best_checkpoint_from_previous_phase.pt"`.
    *   Run the `.bat` file.

4.  **Interpreting Outcomes:**
    *   **Console Logs & WandB**: Your primary sources of information. Track losses (`Rec`, `KL`, `Adv`, `D_tot`, `D_real`, `D_fake`), validation metrics (`PSNR`, `SSIM`, `LPIPS`), and Q-Controller decisions (`LR`, `Q_Scl`).
    *   **WandB Image Samples**:
        *   `train_recon`: How well is the VAE reconstructing?
        *   `val_reconstruction_samples`: Similar, on validation data.
        *   `fixed_noise_generated`: Quality and diversity from a fixed latent vector.
    *   **"What I did wrong, and how to make it better"**: If a phase doesn't improve things or destabilizes training, revert to the previous good checkpoint and try smaller changes or a different approach. This is part of the exploration!

## Visualizations (Expected)

As training progresses, especially when using WandB:
*   **Loss Curves:** Track all loss components.
*   **Validation Metrics:** PSNR, SSIM, LPIPS curves should show improvement.
*   **Generated Image Samples:** Progress from noise to coherent images/videos.
*   **Q-Controller Stats:** Epsilon decay, chosen LR/Momentum scales.
*   **(Future/Conceptual for WuBu Specifics)** If you add logging for WuBu level curvatures or scales, these can be plotted to see how the geometry adapts.

## Key Hyperparameters in `.bat` for `WuBuGAADHybridGen_v0.1.py`

*   `LAMBDA_KL`, `LAMBDA_RECON`, `LAMBDA_GAN`
*   `LEARNING_RATE_GEN`, `LEARNING_RATE_DISC`
*   `DISC_USE_GAAD_FILM_CONDITION`
*   WuBu stack parameters (e.g., `WUBU_S_NUM_LEVELS`, `WUBU_S_INITIAL_CURVATURES`, etc.) if you want to experiment with the WuBu architecture itself.
*   `Q_CONTROLLER_ENABLED` (keep `true` to use the adaptive LRs)

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

MIT License (Assumed, please verify if a specific license file is present)

## Acknowledgments

The ideas explored here build upon a vast body of work in hyperbolic geometry, geometric deep learning, compositional theories, and video modeling. Specific theoretical inspirations are cited within the accompanying papers.

## Citation

If you find the theories (WuBu Nesting, GAAD), the codebase, or the papers useful in your research, please consider citing the source documents:

```
@misc{BytropixPlayground2025,
  author       = {W. WaefreBeorn and Collaborators},
  title        = {Bytropix: An Experimental Playground for WuBu Nesting & Golden Aspect Adaptive Decomposition},
  year         = {2025},
  howpublished = {GitHub Repository},
  note         = {URL: https://github.com/waefrebeorn/bytropix} 
}

@techreport{WaefreBeornWuBuNestingFramework,
  author       = {W. WaefreBeorn}, 
  title        = {WuBu Nesting (層疊嵌套): A Comprehensive Geometric Framework for Adaptive Multi-Scale Hierarchical Representation with Integrated Rotational Dynamics},
  year         = {2024-2025}, // Or year on PDF
  institution  = {Bytropix Project (Self-Published)},
  note         = {Available at Bytropix GitHub repository: WuBu_Nesting.pdf}
}
```
