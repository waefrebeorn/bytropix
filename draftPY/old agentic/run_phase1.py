import argparse
import torch 
import os 
import json
import logging
from pathlib import Path 
from typing import Dict, Optional, List, Any 

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Imports from etp_common and local trainer_phase1
try:
    from draftPY.etp_common.etp_datasets import DeepSeekR1EmbeddingDataset
    from draftPY.etp_common.etp_wubu_architectures import ETP_WuBuText_DS_R1_Sphere, DEFAULT_WUBU_TEXT_CONFIG
    # Optimizer utils are used by the trainer, not directly here, but good to ensure they are conceptually available
    from draftPY.etp_common.optimizer_utils import DEFAULT_CONFIG_QLEARN_HYBRID as _dqch_utils_common
    from .trainer_phase1 import ETPTrainerPhase1 # Relative import for local trainer
except ImportError as e:
    logger.error(f"Phase1 Runner: Critical error importing modules: {e}. Using stubs for some classes.")
    # Define stubs if imports fail for CODING-ONLY robustness
    class DeepSeekR1EmbeddingDataset: # type: ignore
        def __init__(self, *args: Any, **kwargs: Any): logger.info("[STUB] DeepSeekR1EmbeddingDataset (Phase1 Runner)")
        def __len__(self) -> int: return 0

    class ETP_WuBuText_DS_R1_Sphere(torch.nn.Module): # type: ignore
        def __init__(self, *args: Any, **kwargs: Any): 
            super().__init__()
            _wubu_initial_tangent_dim_safe = kwargs.get('wubu_initial_tangent_dim',1) if kwargs.get('wubu_initial_tangent_dim',1) > 0 else 1
            self.wubu_core = torch.nn.Sequential(torch.nn.Linear(_wubu_initial_tangent_dim_safe, _wubu_initial_tangent_dim_safe))
            self.wubu_core.output_tangent_dim = _wubu_initial_tangent_dim_safe # type: ignore[attr-defined]
            logger.info("[STUB] ETP_WuBuText_DS_R1_Sphere (Phase1 Runner)")
        def to(self, device: Any, *args: Any, **kwargs: Any) -> 'ETP_WuBuText_DS_R1_Sphere': return self

    DEFAULT_WUBU_TEXT_CONFIG: Dict[str, Any] = {} # type: ignore
    _dqch_utils_common: Dict[str, Any] = {} # type: ignore

    class ETPTrainerPhase1: # type: ignore
        def __init__(self, *args: Any, **kwargs: Any): logger.info("[STUB] ETPTrainerPhase1 (Phase1 Runner)")
        def train(self, resume_from_checkpoint: Optional[str] = None) -> None: logger.info("[STUB] ETPTrainerPhase1.train() called.")

# Stub for torch.utils.data.DataLoader if not imported above (though it's standard)
try: from torch.utils.data import DataLoader
except ImportError: 
    class DataLoader: # type: ignore
        def __init__(self, *args: Any, **kwargs: Any): logger.info("[STUB] DataLoader (Phase1 Runner)")


def parse_arguments_phase1() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ETP WuBuText DS-R1 - Phase 1 (Reconstruction)")

    # Dataset Paths
    parser.add_argument("--embeddings_file_A", required=True, type=str, help="Path to target embeddings for Corpus A (reconstruction target)")
    parser.add_argument("--embeddings_file_B", required=True, type=str, help="Path to (potentially unused) embeddings for Corpus B (dataset class might need it)")

    # ETP Sphere Model Config
    parser.add_argument("--ds_r1_embedding_dim", type=int, default=768, help="DeepSeek R1 embedding dimension")
    parser.add_argument("--wubu_initial_tangent_dim", type=int, default=256, help="WuBu core input tangent dimension")
    parser.add_argument("--head_mlp_layers", type=int, default=2, help="MLP layers in transfusion head")
    parser.add_argument("--decoder_mlp_layers", type=int, default=2, help="MLP layers in decoder")
    parser.add_argument("--wubu_core_config_json", type=str, default=None, help="Path to JSON for WuBu core config")

    # Loss Weights for Phase 1
    parser.add_argument("--lambda_rec", type=float, default=1.0, help="Weight for Reconstruction loss (primary for Phase 1)")
    parser.add_argument("--lambda_vsp", type=float, default=0.0, help="Weight for VSP loss (typically 0.0 or small for pure Phase 1)")
    # lambda_ala is intentionally omitted for Phase 1 trainer

    # Training Hyperparameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training (cuda or cpu)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs for Phase 1")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--use_amp", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use Automatic Mixed Precision (AMP)")
    parser.add_argument("--global_max_grad_norm", type=float, default=1.0, help="Global max gradient norm for clipping")

    # Optimizer Settings for ETP Sphere Model
    parser.add_argument("--lr_sphere_wubu_core", type=float, default=1e-4, help="LR for WuBu core (RiemannianEnhancedSGD)")
    parser.add_argument("--lr_sphere_mlps", type=float, default=1e-4, help="LR for ETP Sphere MLPs (Head/Decoder)")
    parser.add_argument("--optimizer_kwargs_wubu_core_json", type=str, default='{}', help="JSON string for WuBu core optimizer kwargs")
    parser.add_argument("--optimizer_kwargs_mlps_json", type=str, default='{}', help="JSON string for MLPs optimizer kwargs")
    
    # Q-Controller Config for ETP Sphere Model
    parser.add_argument("--q_controller_enabled", type=lambda x: (str(x).lower() == 'true'), default=True, help="Enable Q-Controllers for ETP Sphere optimizers")
    parser.add_argument("--q_config_sphere_wubu_core_json", type=str, default=None, help="Path to JSON for WuBu core Q-Controller config")
    parser.add_argument("--q_config_sphere_mlps_json", type=str, default=None, help="Path to JSON for MLPs Q-Controller config")

    # Logging/Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="etp_phase1_reconstruction/checkpoints_phase1_rec", help="Directory for saving Phase 1 checkpoints")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to a Phase 1 checkpoint to load and resume training")
    parser.add_argument("--log_interval", type=int, default=20, help="Log training status every N global steps")
    parser.add_argument("--save_interval", type=int, default=500, help="Save checkpoint every N global steps (0 for end of epoch only)")
    parser.add_argument("--val_interval_epochs", type=int, default=1, help="Run validation every N epochs")
    parser.add_argument("--wandb_project", type=str, default="ETP_Phase1_Reconstruction", help="WandB project name for Phase 1")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name for Phase 1 (auto-generated if None)")

    args = parser.parse_args()

    # Parse JSON string arguments for optimizer kwargs
    for json_arg_name_suffix in ['optimizer_kwargs_wubu_core', 'optimizer_kwargs_mlps']:
        json_arg_name_full = f"{json_arg_name_suffix}_json"
        json_str_val = getattr(args, json_arg_name_full)
        try:
            parsed_val = json.loads(json_str_val)
            setattr(args, json_arg_name_suffix, parsed_val) 
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON for --{json_arg_name_full.replace('_','-')}: '{json_str_val}'. Error: {e}")
            setattr(args, json_arg_name_suffix, {}) # Default to empty dict on error
    return args

def main_phase1() -> None:
    args = parse_arguments_phase1()
    
    logger.info("Starting ETP WuBuText DS-R1 - Phase 1 Runner (CODING-ONLY Mode)")
    logger.info(f"Phase 1 Parsed Arguments (sample): epochs={args.epochs}, lambda_rec={args.lambda_rec}, lambda_vsp={args.lambda_vsp}")

    run_device = torch.device(args.device) # type: ignore
    logger.info(f"Conceptual device for Phase 1: {run_device}")

    # Load WuBu Core Config
    effective_wubu_core_config = DEFAULT_WUBU_TEXT_CONFIG.copy() # type: ignore[attr-defined]
    if args.wubu_core_config_json:
        try:
            with open(args.wubu_core_config_json, 'r') as f_wubu:
                custom_wubu_config = json.load(f_wubu)
                effective_wubu_core_config.update(custom_wubu_config)
            logger.info(f"Loaded custom WuBu core config for Phase 1 from: {args.wubu_core_config_json}")
        except FileNotFoundError: logger.warning(f"WuBu core config JSON not found: {args.wubu_core_config_json}. Using default.")
        except json.JSONDecodeError as e_wubu_json: logger.warning(f"Error parsing WuBu JSON {args.wubu_core_config_json}: {e_wubu_json}. Using default.")
    
    # Load Q-Controller Configs
    q_controller_configs: Dict[str, Optional[Dict[str, Any]]] = {}
    for qc_name_key in ["sphere_wubu_core", "sphere_mlps"]:
        json_path_val = getattr(args, f"q_config_{qc_name_key}_json", None)
        if json_path_val:
            try:
                with open(json_path_val, 'r') as f_qc: q_controller_configs[qc_name_key] = json.load(f_qc)
                logger.info(f"Loaded Q-Controller config for {qc_name_key} (Phase 1) from: {json_path_val}")
            except FileNotFoundError: logger.warning(f"Q-Ctrl config for {qc_name_key} not found: {json_path_val}. Defaults used."); q_controller_configs[qc_name_key] = None
            except json.JSONDecodeError as e_qc_json: logger.warning(f"Error parsing Q-Ctrl JSON for {qc_name_key} ({json_path_val}): {e_qc_json}. Defaults used."); q_controller_configs[qc_name_key] = None
        else: q_controller_configs[qc_name_key] = None 
    
    # Instantiate Dataset and DataLoader (using stubs from etp_common)
    logger.info("Instantiating Dataset and DataLoader for Phase 1 (using stubs)...")
    dataset_instance = DeepSeekR1EmbeddingDataset(args.embeddings_file_A, args.embeddings_file_B)
    safe_batch_size = max(1, args.batch_size)
    train_loader_instance = DataLoader(dataset_instance, batch_size=safe_batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader_instance = None # Or configure a validation split if available
    logger.info("Dataset and DataLoader stubs for Phase 1 instantiated.")

    # Instantiate ETP Sphere Model (from etp_common)
    etp_sphere_model_instance = ETP_WuBuText_DS_R1_Sphere(
        ds_r1_embedding_dim=args.ds_r1_embedding_dim,
        wubu_initial_tangent_dim=args.wubu_initial_tangent_dim,
        wubu_core_config=effective_wubu_core_config,
        head_mlp_layers=args.head_mlp_layers,
        decoder_mlp_layers=args.decoder_mlp_layers
    ) 
    logger.info("ETP_WuBuText_DS_R1_Sphere stub for Phase 1 instantiated.")
    
    # Instantiate ETPTrainerPhase1 (from local trainer_phase1.py)
    trainer_instance = ETPTrainerPhase1(
        etp_sphere_model=etp_sphere_model_instance,
        train_loader=train_loader_instance, # type: ignore
        val_loader=val_loader_instance, # type: ignore
        lr_sphere_wubu_core=args.lr_sphere_wubu_core,
        lr_sphere_mlps=args.lr_sphere_mlps,
        optimizer_kwargs_wubu_core=args.optimizer_kwargs_wubu_core,
        optimizer_kwargs_mlps=args.optimizer_kwargs_mlps,
        lambda_rec=args.lambda_rec,
        lambda_vsp=args.lambda_vsp, # Pass VSP, trainer decides if to use it based on value
        device=run_device, # type: ignore
        epochs=args.epochs,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=args.use_amp,
        global_max_grad_norm=args.global_max_grad_norm,
        q_controller_enabled=args.q_controller_enabled,
        q_config_sphere_wubu_core=q_controller_configs["sphere_wubu_core"],
        q_config_sphere_mlps=q_controller_configs["sphere_mlps"],
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        val_interval_epochs=args.val_interval_epochs,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )
    logger.info("ETPTrainerPhase1 stub instantiated.")

    logger.info("Calling ETPTrainerPhase1.train() (conceptual call for CODING-ONLY)...")
    trainer_instance.train(resume_from_checkpoint=args.load_checkpoint)
    logger.info("Conceptual Phase 1 training process finished (CODING-ONLY).")

if __name__ == '__main__':
    main_phase1()
    logger.info("run_phase1.py (CODING-ONLY) main_phase1() execution completed.")
