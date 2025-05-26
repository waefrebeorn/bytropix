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

# Imports from etp_common and local trainer_phase2
try:
    from draftPY.etp_common.etp_datasets import DeepSeekR1EmbeddingDataset
    from draftPY.etp_common.etp_wubu_architectures import ETP_WuBuText_DS_R1_Sphere, DEFAULT_WUBU_TEXT_CONFIG
    from draftPY.etp_common.etp_discriminators import LatentDiscriminatorMLP
    # Optimizer utils are used by the trainer, not directly here
    from draftPY.etp_common.optimizer_utils import DEFAULT_CONFIG_QLEARN_HYBRID as _dqch_utils_common
    from .trainer_phase2 import ETPTrainerPhase2 # Relative import for local trainer
except ImportError as e:
    logger.error(f"Phase2 Runner: Critical error importing modules: {e}. Using stubs for some classes.")
    # Define stubs if imports fail for CODING-ONLY robustness
    class DeepSeekR1EmbeddingDataset: # type: ignore
        def __init__(self, *args: Any, **kwargs: Any): logger.info("[STUB] DeepSeekR1EmbeddingDataset (Phase2 Runner)")
        def __len__(self) -> int: return 0

    class ETP_WuBuText_DS_R1_Sphere(torch.nn.Module): # type: ignore
        def __init__(self, *args: Any, **kwargs: Any): 
            super().__init__()
            _wubu_initial_tangent_dim_safe = kwargs.get('wubu_initial_tangent_dim',1) if kwargs.get('wubu_initial_tangent_dim',1) > 0 else 1
            self.wubu_core = torch.nn.Sequential(torch.nn.Linear(_wubu_initial_tangent_dim_safe, _wubu_initial_tangent_dim_safe))
            self.wubu_core.output_tangent_dim = _wubu_initial_tangent_dim_safe # type: ignore[attr-defined]
            logger.info("[STUB] ETP_WuBuText_DS_R1_Sphere (Phase2 Runner)")
        def to(self, device: Any, *args: Any, **kwargs: Any) -> 'ETP_WuBuText_DS_R1_Sphere': return self
        def get_latent(self, x: torch.Tensor) -> torch.Tensor: return torch.randn_like(x) # type: ignore[no-untyped-call]
        def forward(self, x: torch.Tensor) -> torch.Tensor: return torch.randn_like(x) # type: ignore[no-untyped-call]


    class LatentDiscriminatorMLP(torch.nn.Module): # type: ignore
        def __init__(self, *args: Any, **kwargs: Any): 
            super().__init__()
            _input_dim_safe = kwargs.get('input_dim', 1) if kwargs.get('input_dim',1) > 0 else 1
            self.fc = torch.nn.Linear(_input_dim_safe,1)
            logger.info("[STUB] LatentDiscriminatorMLP (Phase2 Runner)")
        def to(self, device: Any, *args: Any, **kwargs: Any) -> 'LatentDiscriminatorMLP': return self
        def forward(self, x: torch.Tensor) -> torch.Tensor: return torch.randn(x.size(0),1,device=x.device) # type: ignore[no-untyped-call]


    DEFAULT_WUBU_TEXT_CONFIG: Dict[str, Any] = {} # type: ignore
    _dqch_utils_common: Dict[str, Any] = {} # type: ignore

    class ETPTrainerPhase2: # type: ignore
        def __init__(self, *args: Any, **kwargs: Any): logger.info("[STUB] ETPTrainerPhase2 (Phase2 Runner)")
        def train(self, resume_from_checkpoint: Optional[str] = None) -> None: logger.info("[STUB] ETPTrainerPhase2.train() called.")

# Stub for torch.utils.data.DataLoader if not imported above
try: from torch.utils.data import DataLoader
except ImportError: 
    class DataLoader: # type: ignore
        def __init__(self, *args: Any, **kwargs: Any): logger.info("[STUB] DataLoader (Phase2 Runner)")
        def __iter__(self): yield {'source_A': torch.randn(2,10), 'source_B': torch.randn(2,10)}; return self # type: ignore
        def __len__(self): return 1


def parse_arguments_phase2() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ETP WuBuText DS-R1 - Phase 2 (Adversarial Latent Alignment)")

    # Dataset Paths
    parser.add_argument("--embeddings_file_A", required=True, type=str, help="Path to embeddings for Corpus A")
    parser.add_argument("--embeddings_file_B", required=True, type=str, help="Path to embeddings for Corpus B")

    # ETP Sphere Model Config (same as Phase 1 runner)
    parser.add_argument("--ds_r1_embedding_dim", type=int, default=768)
    parser.add_argument("--wubu_initial_tangent_dim", type=int, default=256)
    parser.add_argument("--head_mlp_layers", type=int, default=2)
    parser.add_argument("--decoder_mlp_layers", type=int, default=2)
    parser.add_argument("--wubu_core_config_json", type=str, default=None, help="Path to JSON for WuBu core config")

    # Discriminator Config (from original full runner)
    parser.add_argument("--disc_hidden_dims_json", type=str, default='[256, 128]', help="JSON list of discriminator hidden dims")
    parser.add_argument("--disc_activation_fn", type=str, default="leaky_relu")
    parser.add_argument("--disc_use_spectral_norm", type=lambda x: (str(x).lower() == 'true'), default=True)

    # Loss Weights for Phase 2
    parser.add_argument("--lambda_ala", type=float, default=0.1, help="Weight for ALA loss")
    parser.add_argument("--lambda_rec", type=float, default=1.0, help="Weight for Reconstruction loss")
    parser.add_argument("--lambda_vsp", type=float, default=0.01, help="Weight for VSP loss")

    # Training Hyperparameters (same as original full runner)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--use_amp", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--global_max_grad_norm", type=float, default=1.0)

    # Optimizer Settings (for all three optimizers)
    parser.add_argument("--lr_sphere_wubu_core", type=float, default=1e-4)
    parser.add_argument("--lr_sphere_mlps", type=float, default=1e-4)
    parser.add_argument("--lr_discriminator", type=float, default=2e-4) # Added for Phase 2
    parser.add_argument("--optimizer_kwargs_wubu_core_json", type=str, default='{}')
    parser.add_argument("--optimizer_kwargs_mlps_json", type=str, default='{}')
    parser.add_argument("--optimizer_kwargs_discriminator_json", type=str, default='{}') # Added for Phase 2
    
    # Q-Controller Config (for all three optimizers)
    parser.add_argument("--q_controller_enabled", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--q_config_sphere_wubu_core_json", type=str, default=None)
    parser.add_argument("--q_config_sphere_mlps_json", type=str, default=None)
    parser.add_argument("--q_config_discriminator_json", type=str, default=None) # Added for Phase 2

    # Logging/Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="etp_phase2_ala/checkpoints_phase2_ala", help="Checkpoint directory for Phase 2")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to a checkpoint to load (can be from Phase 1 or Phase 2)")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--val_interval_epochs", type=int, default=1)
    parser.add_argument("--wandb_project", type=str, default="ETP_Phase2_ALA")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()

    # Parse JSON string arguments
    for json_arg_name_suffix in ['disc_hidden_dims', 'optimizer_kwargs_wubu_core', 
                                 'optimizer_kwargs_mlps', 'optimizer_kwargs_discriminator']:
        json_arg_name_full = f"{json_arg_name_suffix}_json"
        json_str_val = getattr(args, json_arg_name_full)
        try:
            parsed_val = json.loads(json_str_val)
            setattr(args, json_arg_name_suffix, parsed_val) 
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON for --{json_arg_name_full.replace('_','-')}: '{json_str_val}'. Error: {e}")
            setattr(args, json_arg_name_suffix, {} if 'kwargs' in json_arg_name_suffix else []) 
    return args


def main_phase2() -> None:
    args = parse_arguments_phase2()
    
    logger.info("Starting ETP WuBuText DS-R1 - Phase 2 Runner (CODING-ONLY Mode)")
    logger.info(f"Phase 2 Parsed Arguments (sample): epochs={args.epochs}, lambda_ala={args.lambda_ala}")

    run_device = torch.device(args.device) # type: ignore
    logger.info(f"Conceptual device for Phase 2: {run_device}")

    # Load WuBu Core Config
    effective_wubu_core_config = DEFAULT_WUBU_TEXT_CONFIG.copy() # type: ignore[attr-defined]
    if args.wubu_core_config_json:
        try:
            with open(args.wubu_core_config_json, 'r') as f_wubu:
                custom_wubu_config = json.load(f_wubu)
                effective_wubu_core_config.update(custom_wubu_config)
            logger.info(f"Loaded custom WuBu core config for Phase 2 from: {args.wubu_core_config_json}")
        except FileNotFoundError: logger.warning(f"WuBu core config JSON not found: {args.wubu_core_config_json}. Using default.")
        except json.JSONDecodeError as e_wubu_json: logger.warning(f"Error parsing WuBu JSON {args.wubu_core_config_json}: {e_wubu_json}. Using default.")
    
    # Load Q-Controller Configs
    q_controller_configs: Dict[str, Optional[Dict[str, Any]]] = {}
    for qc_name_key in ["sphere_wubu_core", "sphere_mlps", "discriminator"]: # Include discriminator Q-config
        json_path_val = getattr(args, f"q_config_{qc_name_key}_json", None)
        if json_path_val:
            try:
                with open(json_path_val, 'r') as f_qc: q_controller_configs[qc_name_key] = json.load(f_qc)
                logger.info(f"Loaded Q-Controller config for {qc_name_key} (Phase 2) from: {json_path_val}")
            except FileNotFoundError: logger.warning(f"Q-Ctrl config for {qc_name_key} not found: {json_path_val}. Defaults used."); q_controller_configs[qc_name_key] = None
            except json.JSONDecodeError as e_qc_json: logger.warning(f"Error parsing Q-Ctrl JSON for {qc_name_key} ({json_path_val}): {e_qc_json}. Defaults used."); q_controller_configs[qc_name_key] = None
        else: q_controller_configs[qc_name_key] = None 
    
    # Instantiate Dataset and DataLoader (from etp_common)
    logger.info("Instantiating Dataset and DataLoader for Phase 2 (using stubs)...")
    dataset_instance = DeepSeekR1EmbeddingDataset(args.embeddings_file_A, args.embeddings_file_B)
    safe_batch_size = max(1, args.batch_size)
    train_loader_instance = DataLoader(dataset_instance, batch_size=safe_batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)
    val_loader_instance = None 
    logger.info("Dataset and DataLoader stubs for Phase 2 instantiated.")

    # Instantiate ETP Sphere Model (from etp_common)
    etp_sphere_model_instance = ETP_WuBuText_DS_R1_Sphere(
        ds_r1_embedding_dim=args.ds_r1_embedding_dim,
        wubu_initial_tangent_dim=args.wubu_initial_tangent_dim,
        wubu_core_config=effective_wubu_core_config,
        head_mlp_layers=args.head_mlp_layers,
        decoder_mlp_layers=args.decoder_mlp_layers
    ) 
    logger.info("ETP_WuBuText_DS_R1_Sphere stub for Phase 2 instantiated.")
    
    # Determine Discriminator input_dim from ETP Sphere model stub
    disc_input_dim = args.wubu_initial_tangent_dim # Fallback
    if hasattr(etp_sphere_model_instance.wubu_core, 'output_tangent_dim'):
        retrieved_dim = etp_sphere_model_instance.wubu_core.output_tangent_dim # type: ignore[attr-defined]
        if isinstance(retrieved_dim, int) and retrieved_dim > 0: disc_input_dim = retrieved_dim
        else: logger.warning(f"Sphere stub's wubu_core.output_tangent_dim invalid ({retrieved_dim}). Defaulting D input.")
    else: logger.warning("Sphere stub's wubu_core lacks 'output_tangent_dim'. Defaulting D input.")
    logger.info(f"Discriminator input dimension for Phase 2 resolved to: {disc_input_dim}")

    # Instantiate Discriminator Model (from etp_common)
    discriminator_model_instance = LatentDiscriminatorMLP(
        input_dim=disc_input_dim,
        hidden_dims=args.disc_hidden_dims, 
        activation_fn=args.disc_activation_fn,
        use_spectral_norm=args.disc_use_spectral_norm
    ) 
    logger.info("LatentDiscriminatorMLP stub for Phase 2 instantiated.")
    
    # Instantiate ETPTrainerPhase2 (from local trainer_phase2.py)
    trainer_instance = ETPTrainerPhase2(
        etp_sphere_model=etp_sphere_model_instance,
        discriminator_model=discriminator_model_instance, # Pass discriminator
        train_loader=train_loader_instance, # type: ignore
        val_loader=val_loader_instance, # type: ignore
        lr_sphere_wubu_core=args.lr_sphere_wubu_core,
        lr_sphere_mlps=args.lr_sphere_mlps,
        lr_discriminator=args.lr_discriminator, # Pass D LR
        optimizer_kwargs_wubu_core=args.optimizer_kwargs_wubu_core,
        optimizer_kwargs_mlps=args.optimizer_kwargs_mlps,
        optimizer_kwargs_discriminator=args.optimizer_kwargs_discriminator, # Pass D optimizer kwargs
        lambda_ala=args.lambda_ala, # Pass all three lambdas
        lambda_rec=args.lambda_rec,
        lambda_vsp=args.lambda_vsp,
        device=run_device, # type: ignore
        epochs=args.epochs,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=args.use_amp,
        global_max_grad_norm=args.global_max_grad_norm,
        q_controller_enabled=args.q_controller_enabled,
        q_config_sphere_wubu_core=q_controller_configs["sphere_wubu_core"],
        q_config_sphere_mlps=q_controller_configs["sphere_mlps"],
        q_config_discriminator=q_controller_configs["discriminator"], # Pass D Q-config
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        val_interval_epochs=args.val_interval_epochs,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )
    logger.info("ETPTrainerPhase2 stub instantiated.")

    logger.info("Calling ETPTrainerPhase2.train() (conceptual call for CODING-ONLY)...")
    trainer_instance.train(resume_from_checkpoint=args.load_checkpoint)
    logger.info("Conceptual Phase 2 training process finished (CODING-ONLY).")

if __name__ == '__main__':
    main_phase2()
    logger.info("run_phase2.py (CODING-ONLY) main_phase2() execution completed.")
