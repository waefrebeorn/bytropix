---

# Bytropix: A Playground for WuBu Nesting & GAAD

[![Status: Research Playground - Experimental](https://img.shields.io/badge/status-research%20playground%20(experimental)-orange)](https://shields.io/)
[![Discord](https://img.shields.io/discord/1303046473985818654?label=Discord&logo=discord&style=for-the-badge)](http://wubu.waefrebeorn.com)

Welcome to **Bytropix**! This repository is an open, experimental playground dedicated to exploring the theoretical frameworks of **WuBu Nesting (層疊嵌套)** and **Golden Aspect Adaptive Decomposition (GAAD)**. If you're interested in pushing the boundaries of how we can model complex data, especially video, with deep geometric and compositional priors, you're in the right place.

Think of this as a collection of **experimental scaffolds and starting beds**. The Python scripts and batch files here are my attempts to implement these advanced mathematical theories. They are functional, but more importantly, they are designed to be pulled apart, improved, and used as inspiration. The goal isn't a polished, final product, but a space to collectively figure out what works, what doesn't, and how to make these powerful ideas better. This is for those who like to get their hands dirty with "theory math" that might be a tough sell to the "tech head goobers" without seeing it in action (or at least, in attempted action!).

## Core Theories Under Exploration

At the heart of Bytropix are two main theoretical constructs:

1.  **WuBu Nesting (層疊嵌套):** A framework for building models with recursively nested hyperbolic spaces (`H^{n_i}_{c_i,s_i}`). The geometry of these spaces (dimensionality, curvature, scale) can adapt during learning. Key features include learnable Boundary Sub-Manifolds, Level Descriptor Vectors, Level Spread Parameters, Intra-Level Tangent Flows, and, crucially, inter-level transitions orchestrated in tangent space with explicit `SO(n_i)` Rotations and Mappings. This aims to capture deep multi-scale hierarchies and rotational dynamics.
2.  **Golden Aspect Adaptive Decomposition (GAAD):** A method inspired by the Golden Ratio (φ) for decomposing visual data (like video frames) into multi-scale, aspect-ratio agnostic regions. It uses techniques like Recursive Golden Subdivision and Phi-Spiral Patching to guide feature extraction, respecting natural compositions.

The main application explored here is **WuBuNestDiffusion**, a video diffusion model built upon these principles, aiming to generate and understand video with a strong geometric and compositional foundation.

## The Playground: What's Inside?

This repository is a collection of my explorations and implementations:

*   **Diverse Python Scripts (`draftPY/`, root):** You'll find various Python files that are more than just one final model. These include:
    *   The main `WuBuNestDiffusion_v0.05_GAAD_MotionWuBu_Live.py`: My latest attempt at a comprehensive video diffusion model using WuBu Nesting and GAAD for appearance and motion.
    *   Trainers for different variants: `WuBuNest_TrainerV1.py`, `WuBuNestmRnaTrainerV1.py` (and their root-level counterparts). These explore different aspects or applications of WuBu Nesting.
    *   Inference scripts: `inference.py`, `sfin_inference.py`, `WuBuNest_Inferencev1.py`.
    *   Supporting modules: `EnhancedSGD.py` (custom optimizer components), `HAKMEMQController.py` (Q-learning for hyperparameter tuning), components for hyperbolic math (`HypBSFIN.py`, `HypCD.py`).
    *   Utility and data generation scripts: `create_demo_data.py`, `poem_dataset_generator.py`, `convert_video_simple.py`.
    *   Older experimental files: `oldwubunest.py`, `integrated_hyper_hakmem_model.py`.
*   **Batch Files (`.bat`):** These are your primary interface for running experiments! They call the Python scripts with a multitude of command-line arguments, allowing you to tweak parameters, switch components, and test different configurations of the theories without digging into the Python code immediately.
*   **Theoretical Documents (The "Why"):**
    *   [`WuBu_Nesting.pdf`](./WuBu_Nesting.pdf): The core PDF detailing the foundational WuBu Nesting framework. *The papers often use more elaborate naming for concepts for "elequence and appeal to goobers," but the core is WuBu Nesting & GAAD.*
    *   Markdown Papers: Deeper dives into specific aspects and the main `WuBuNestDiffusion` model. These provide the context for the code:
        *   [`./draftPY/WuBuNestDiffusion (v0.05.2).md`](./draftPY/WuBuNestDiffusion%20(v0.05.2).md)
        *   [`./WuBuHypCD-paper.md`](./WuBuHypCD-paper.md) (Foundational WuBu Nesting)
        *   [`./GAAD-WuBu-ST1.md`](./GAAD-WuBu-ST1.md) & [`./GAAD-WuBu-ST2.md`](./GAAD-WuBu-ST2.md) (GAAD and Spatio-Temporal WuBu)
        *   [`./WuBu Spatio-Temporal Nesting.md`](./WuBu%20Spatio-Temporal%20Nesting.md)

## Navigating the Codebase

```
└── ./
    ├── draftPY/                             # Primary hub for Python experiments & latest models
    │   ├── WuBuNestDiffusion_v0.05_GAAD_MotionWuBu_Live.py  # Main live model
    │   ├── WuBuNestDiffusion_v0.05_GAAD_MotionWuBu_PaperCopy.py # Variant for paper
    │   ├── WuBuNest_TrainerV1.py            # Generic WuBu Nesting trainer
    │   ├── WuBuNestmRnaTrainerV1.py         # Specific "mRNA" variant trainer
    │   ├── EnhancedSGD.py                   # Custom optimizer components
    │   ├── HAKMEMQController.py             # Q-learning hyperparameter controller
    │   ├── *.bat                            # Batch files to RUN THE EXPERIMENTS
    │   └── ... (many other Python scripts: inference, older models, utilities)
    ├── WuBuNest_Trainer.py                  # Root-level trainer (possibly earlier version)
    ├── WuBuNestmRnaTrainer.py               # Root-level "mRNA" trainer
    ├── WuBuNest_Inference.py                # Root-level inference
    ├── wubu_nesting_impl.py                 # Core WuBu Nesting logic (example/ref)
    ├── GAAD-WuBu-ST1.md, GAAD-WuBu-ST2.md   # Papers on GAAD + WuBu-ST
    ├── WuBu Spatio-Temporal Nesting.md      # Paper on WuBu-ST
    ├── WuBuHypCD-paper.md                   # Paper on foundational WuBu Nesting
    ├── WuBu_Nesting.pdf                     # Main PDF for WuBu Nesting theory
    ├── requirements.txt                     # Dependencies
    ├── setup.bat, venv.bat                  # Windows environment setup
    └── ... (other utilities, data generators like poem_dataset_generator.py)
```

**The general idea:**
1.  Read the papers (especially `WuBu_Nesting.pdf` and `draftPY/WuBuNestDiffusion (v0.05.2).md`) to understand the theory.
2.  Look at the Python scripts in `draftPY/` (especially `WuBuNestDiffusion_v0.05_GAAD_MotionWuBu_Live.py` and supporting modules) to see how the theory is attempted in code.
3.  Use the `.bat` files as templates to run your own experiments, changing parameters to test different hypotheses.

## Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone <https://github.com/waefrebeorn/bytropix.git> 
    cd bytropix 
    ```

2.  **Set up Environment (Windows):**
    *   It's highly recommended to use a virtual environment.
    *   Run `setup.bat`. This should attempt to create a `venv` virtual environment and install dependencies.
    *   Activate the virtual environment: `.\venv.bat` (or `.\venv\Scripts\activate`).

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Ensure PyTorch is installed with CUDA support for GPU acceleration.

4.  **Data:**
    *   `draftPY/create_demo_data.py` can generate simple text DEMO data for initial tests of BYTE encoders.
    *   For more serious experiments, prepare your video datasets and update paths in the `.bat` files or scripts.
    *   Scripts like `poem_dataset_generator.py` hint at explorations beyond standard video.

## Running Experiments & Exploring the Theories

This is where the "playground" comes alive!

1.  **Study the `.bat` files:** These are your keys. Open them (e.g., `draftPY/run_integrated.bat` or `runWuBuNestmRnaTrainer.bat`) in a text editor. You'll see they launch Python scripts with many command-line arguments. These arguments control the theoretical knobs (WuBu levels, GAAD types, φ-influences, etc.).
2.  **Modify & Run:**
    *   **Test a hypothesis:** Want to see if fewer boundary points in WuBu levels simplify training without losing too much? Find the relevant argument (e.g., `--boundary_points_per_level`), change its values, save, and run.
    *   Run from the command line: `draftPY\your_chosen_script.bat`.
3.  **Core Scripts to Target via `.bat` files:**
    *   `draftPY/WuBuNestDiffusion_v0.05_GAAD_MotionWuBu_Live.py`: For the main video diffusion experiments.
    *   `draftPY/WuBuNest_TrainerV1.py` or `draftPY/WuBuNestmRnaTrainerV1.py` (and their root counterparts): For other WuBu Nesting applications.
4.  **Interpreting Outcomes:**
    *   Check console output for losses, metrics (LPIPS, SSIM might be logged).
    *   If WandB is used (check batch files for `--wandb`), use its dashboard.
    *   Look at generated videos/outputs. Do they make sense? How do they change with different theoretical settings?
    *   **This is where you help improve things!** If something doesn't work as expected by the theory, that's a finding. If a change makes it better, that's progress!

## Visualizations

The project includes visualization capabilities, especially helpful for understanding the geometric aspects:

-   **Hyperbolic Boundary Points (Conceptual):** For models using `BoundaryManifoldHyperbolic`, visualizations (like the `nested_spheres_epoch_X.png` example) can project these learned hyperbolic points into 3D Euclidean space (e.g., via PCA). This helps to intuitively grasp how the model is structuring its learned landmarks within the nested hyperbolic levels.
    ![Example Nested Spheres Visualization](wubu_results/visualizations/nested_spheres_epoch_20.png)
-   **Training Metrics:** Plots of training/validation loss, learning rates, gradient norms, Q-controller statistics, and potentially learned geometric parameters (curvatures, scales of WuBu levels) are essential for tracking progress and diagnosing issues.
    ![Example Training Metrics](wubu_results/training_metrics.png)

*Note: Image paths are illustrative from a common output structure (`wubu_results/visualizations/`). Actual paths depend on your `--checkpoint_dir`.*

## Key Hyperparameters to Play With

The theories of WuBu Nesting and GAAD offer a vast parameter space. The best way to see them is:
1.  Look at the `argparse` sections in the Python scripts (e.g., `draftPY/WuBuNestDiffusion_v0.05_GAAD_MotionWuBu_Live.py`).
2.  Examine the `.bat` files for examples of how these are set.

Some key areas for experimentation:
*   **GAAD:** `gaad_num_regions`, `gaad_decomposition_type` (hybrid, subdivide, spiral), φ-influences on GAAD.
*   **WuBu Stacks (S, M, T):** `_num_levels`, `_hyperbolic_dims`, initial/learnable `_curvatures` & `_scales`, `_boundary_points_per_level`, use of explicit rotations (if implemented in a version), tangent flows, φ-influences.
*   **Diffusion:** `timesteps`, `beta_schedule`, time embedding types (e.g., φ-scaled).

## Features of this Playground & The Theories

*   **Exploring WuBu Nesting:**
    *   Adaptive Hyperbolic Geometries (learnable `c_i, s_i`).
    *   Tangent Space Transitions (with/without explicit rotations).
    *   Hyperbolic Boundary Manifolds, Level Descriptors, Spreads, Tangent Flows.
*   **Exploring GAAD:**
    *   φ-inspired visual decomposition (Recursive Golden Subdiv, Phi-Spiral Patching).
    *   Aspect-ratio agnostic processing.
*   **Application to Video Diffusion (`WuBuNestDiffusion`):** A concrete, complex use-case.
*   **Diverse Scaffolding:** Includes various trainers (`mRNA`, `poem`) and utilities showing different angles of attack.
*   **Tools for Stability:** `EnhancedSGD`, `HAKMEMQController`.

## Spirit of the Project & Limitations

*   **This is Research in Progress:** The code is experimental. It's a vehicle for exploring difficult theories. Expect rough edges.
*   **"What I did wrong, and how to make it better":** This is the core ethos. If the theory suggests X, and the code implementing X doesn't quite nail it, that's an opportunity for improvement and learning.
*   **Computational Demands:** These models can be heavy. Deep geometric models are complex.
*   **Optimization is Key:** Finding the right way to train these architectures is a major part of the research.

## Contributing

Given the experimental nature, "contributions" can be:
*   Identifying discrepancies between theory (papers) and implementation (code).
*   Suggesting improvements to make the implementations more robust or more faithful to the theory.
*   Sharing results of experiments that shed light on what works or doesn't.
*   Fork, experiment, and share back if you have breakthroughs!

## License

MIT License

## Acknowledgments

The ideas explored here build upon a vast body of work in hyperbolic geometry, geometric deep learning, compositional theories, and video modeling. Specific theoretical inspirations are cited within the accompanying papers.

## Citation

If you find the theories (WuBu Nesting, GAAD), the codebase, or the papers useful in your research, please consider citing the source documents:

```
@misc{BytropixWuBuNestingPlayground2025,
  author       = {W. WaefreBeorn, et al.},
  title        = {Bytropix: A Playground for WuBu Nesting & GAAD},
  year         = {2025},
  howpublished = {GitHub Repository},
  note         = {URL: https://github.com/waefrebeorn/bytropix} % Replace if public
}

@techreport{WaefreBeornWuBuNestingPaper,
  author       = {W. WaefreBeorn}, % Or appropriate authorship for the PDF
  title        = {WuBu Nesting: A Comprehensive Geometric Framework for Adaptive Multi-Scale Hierarchical Representation with Integrated Rotational Dynamics},
  year         = {2025}, % Or year on PDF
  institution  = {Bytropix Project},
  note         = {Referenced from WuBu_Nesting.pdf in the Bytropix repository}
}

@techreport{WaefreBeornWuBuNestDiffusionPaper,
  author       = {W. WaefreBeorn, et al.},
  title        = {WuBuNestDiffusion (v0.05.2): Motion-Aware Spatio-Temporal Modeling with φ-Infused Golden Aspect Adaptive Decomposition and Adaptive Hyperbolic Nesting for Video Diffusion},
  year         = {2025},
  institution  = {Bytropix Project},
  note         = {Referenced from ./draftPY/WuBuNestDiffusion (v0.05.2).md in the Bytropix repository}
}
```
(And similarly for other specific markdown papers if you draw heavily from them.)

---
