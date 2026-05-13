# Vault: Phase 3 — Text-to-Image Generative Pipeline
#
## Architecture (3-Stage)
1. **AE Latent** (Phase 1/2): Images → `path_params` grid (δ, χ, radius) via Poincaré sphere polarization physics
2. **VQ-VAE Tokenizer**: path_params → discrete codes (3072 codes, 256-dim). Conv(3×3 stride=2)×2 → VQ → ConvTranspose×2. Output: 4×24×6 = 576 tokens/image
3. **Conductor Transformer**: Hierarchical 3-tier world model (Kid/Student/Polisher). FlashAttention, CLIP text conditioning, 3D rotary positions. Masked token prediction (mask=0.5), iterative non-autoregressive decoding

## CORPUS.py (14.6MB, 66K lines)
18 `LORE_*` dictionaries, each with:
- TASK_TYPE: "Foundational Text" or "Instruction Following"
- SOURCE: identifier string
- NARRATIVE_TEXT or INSTRUCTION/RESPONSE: the content
- Used for fine-tuning the text-generation head (not the image pipeline)
- Includes: Codex Assistentia (10 protocols), Dialogus (5 convos), Systema Interna (5 axioms), ALL_CORPUS (7 corpora), Decreta Primordialia (5 safety rules), Shakespeare CODICIL, WuBu Manifesto

## Training Tools
- Q-Controller: RL-based LR scheduler (10-state × 5-action Q-table)
- PID Lambda Controller: Second-order loss balancing
- Sentinel: Gradient sign-history damping
- DSlider: Entropic sampling (Dirichlet fit of logit distribution)
- Rich TUI: Live metrics, preview images, keyboard controls

## Key Scripts
 script (~3873 lines, ~225KB)
    25|- `backupcorpus.py` — Source template for CORPUS.py (~1756 lines)
    26|- `FORXLADEVS.py` — Standalone WubuMind text generation (~252 lines, minimal deps)
    27|

---

*Part of the WuBuText AI project. See [Project Overview](../../README.md) and [Presentation Layer](../presentation/README.md) for navigation.*
