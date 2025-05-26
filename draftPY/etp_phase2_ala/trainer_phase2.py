import logging
import time
from typing import Dict, Optional, List, Tuple, Any, Deque
from pathlib import Path
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from torch.cuda.amp import GradScaler

# Imports from etp_common
try:
    from draftPY.etp_common.etp_wubu_architectures import ETP_WuBuText_DS_R1_Sphere
    from draftPY.etp_common.etp_discriminators import LatentDiscriminatorMLP
    # DeepSeekR1EmbeddingDataset used by runner, not directly by trainer, but good for type hints if needed
    # from draftPY.etp_common.etp_datasets import DeepSeekR1EmbeddingDataset 
    from draftPY.etp_common.etp_losses import (
        calculate_reconstruction_loss, 
        calculate_vector_space_preservation_loss,
        calculate_adversarial_latent_alignment_loss_discriminator,
        calculate_adversarial_latent_alignment_loss_generator
    )
    from draftPY.etp_common.optimizer_utils import (
        RiemannianEnhancedSGD, 
        HAKMEMQController, 
        GradientStats, # Though not directly used in this trainer, it's part of optimizer_utils
        DEFAULT_CONFIG_QLEARN_HYBRID as _dqch_utils # Import with an alias
    )
except ImportError as e:
    # Fallback to stubs if full implementations are not found (for CODING-ONLY robustness)
    logger_stub = logging.getLogger(__name__) 
    logger_stub.error(f"Phase2 Trainer: Critical error importing from etp_common: {e}. Using stubs.")
    
    class ETP_WuBuText_DS_R1_Sphere(nn.Module): # type: ignore
        def __init__(self, *args: Any, **kwargs: Any): 
            super().__init__()
            _wubu_initial_tangent_dim_safe = kwargs.get('wubu_initial_tangent_dim',1) if kwargs.get('wubu_initial_tangent_dim',1) > 0 else 1
            _ds_r1_embedding_dim_safe = kwargs.get('ds_r1_embedding_dim',1) if kwargs.get('ds_r1_embedding_dim',1) > 0 else 1
            self.wubu_core = nn.Sequential(nn.Linear(_wubu_initial_tangent_dim_safe, _wubu_initial_tangent_dim_safe))
            self.wubu_core.output_tangent_dim = _wubu_initial_tangent_dim_safe # type: ignore[attr-defined]
            self.transfusion_head = nn.Sequential(nn.Linear(_ds_r1_embedding_dim_safe, _wubu_initial_tangent_dim_safe))
            self.decoder = nn.Sequential(nn.Linear(self.wubu_core.output_tangent_dim, _ds_r1_embedding_dim_safe)) # type: ignore[attr-defined]
        def get_latent(self, x: torch.Tensor) -> torch.Tensor: return torch.randn_like(x) # type: ignore[no-untyped-call]
        def forward(self, x: torch.Tensor) -> torch.Tensor: return torch.randn_like(x) # type: ignore[no-untyped-call]
        def to(self, device: Any, *args: Any, **kwargs: Any) -> 'ETP_WuBuText_DS_R1_Sphere': return self

    class LatentDiscriminatorMLP(nn.Module): # type: ignore
        def __init__(self, *args: Any, **kwargs: Any): 
            super().__init__()
            _input_dim_safe = kwargs.get('input_dim', 1) if kwargs.get('input_dim',1) > 0 else 1
            self.fc = nn.Linear(_input_dim_safe,1)
        def forward(self, x: torch.Tensor) -> torch.Tensor: return torch.randn(x.size(0),1,device=x.device) # type: ignore[no-untyped-call]
        def to(self, device: Any, *args: Any, **kwargs: Any) -> 'LatentDiscriminatorMLP': return self
    
    def calculate_reconstruction_loss(*args: Any, **kwargs: Any) -> torch.Tensor: return torch.tensor(0.0, requires_grad=True) # type: ignore[no-untyped-def]
    def calculate_vector_space_preservation_loss(*args: Any, **kwargs: Any) -> torch.Tensor: return torch.tensor(0.0, requires_grad=True) # type: ignore[no-untyped-def]
    def calculate_adversarial_latent_alignment_loss_discriminator(*args: Any, **kwargs: Any) -> torch.Tensor: return torch.tensor(0.0, requires_grad=True) # type: ignore[no-untyped-def]
    def calculate_adversarial_latent_alignment_loss_generator(*args: Any, **kwargs: Any) -> torch.Tensor: return torch.tensor(0.0, requires_grad=True) # type: ignore[no-untyped-def]
    
    _dqch_utils = {} # type: ignore[no-redef]
    class RiemannianEnhancedSGD(optim.Optimizer): # type: ignore
        def __init__(self, params: Any, lr: float, q_learning_config: Optional[Dict[str,Any]]=None, **kwargs: Any): # type: ignore[no-untyped-def]
            param_list = list(params) if not isinstance(params, list) else params
            if not param_list: param_list = [nn.Parameter(torch.randn(1))] # type: ignore[no-untyped-call]
            super().__init__(param_list, {"lr": lr})
            self.q_controller = HAKMEMQController(lr, q_learning_config if q_learning_config else {}, logger_suffix=kwargs.get("q_logger_suffix","")) if q_learning_config else None # type: ignore[operator]
        def get_q_controller(self) -> Optional['HAKMEMQController']: return self.q_controller # type: ignore[name-defined]
        def step(self, closure: Any =None) -> Optional[torch.Tensor]: return None # type: ignore[override]

    class HAKMEMQController: # type: ignore
        def __init__(self, initial_lr: float, config: Dict[str,Any], logger_suffix: str = ""): self.lr = initial_lr; self.logger_suffix=logger_suffix # type: ignore[no-untyped-def]
        def choose_action(self, current_metric_val: Optional[float]=None, current_loss_val: Optional[float]=None) -> float: return self.lr
        def log_reward(self, reward: float, metric_val: Optional[float]=None, loss_val: Optional[float]=None): pass # type: ignore[no-untyped-def]
        def get_current_lr(self) -> float: return self.lr
        def state_dict(self) -> Dict[str, Any]: return {"lr": self.lr, "logger_suffix": self.logger_suffix} 
        def load_state_dict(self, state_dict: Dict[str, Any]) -> None: self.lr = state_dict.get("lr", self.lr) # type: ignore[no-untyped-def]

# Optional WandB import
try:
    import wandb
except ImportError:
    wandb = None # type: ignore
    logging.info("wandb not found, WandB logging will be disabled for ETPTrainerPhase2.")

# Configure module-level logger
logger = logging.getLogger(__name__)

class ETPTrainerPhase2:
    def __init__(self,
                 etp_sphere_model: ETP_WuBuText_DS_R1_Sphere,
                 discriminator_model: LatentDiscriminatorMLP, # Added discriminator model
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 lr_sphere_wubu_core: float,
                 lr_sphere_mlps: float,
                 lr_discriminator: float, # Added discriminator LR
                 optimizer_kwargs_wubu_core: Optional[Dict[str, Any]] = None,
                 optimizer_kwargs_mlps: Optional[Dict[str, Any]] = None,
                 optimizer_kwargs_discriminator: Optional[Dict[str, Any]] = None, # Added D optimizer kwargs
                 lambda_ala: float = 0.1, # Default ALA weight for Phase 2
                 lambda_rec: float = 1.0,
                 lambda_vsp: float = 0.01, # Small VSP weight for Phase 2
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 epochs: int = 10,
                 grad_accum_steps: int = 1,
                 use_amp: bool = True,
                 global_max_grad_norm: float = 1.0,
                 q_controller_enabled: bool = True,
                 q_config_sphere_wubu_core: Optional[Dict[str, Any]] = None,
                 q_config_sphere_mlps: Optional[Dict[str, Any]] = None,
                 q_config_discriminator: Optional[Dict[str, Any]] = None, # Added D Q-controller config
                 checkpoint_dir: str = "checkpoints_etp_phase2",
                 log_interval: int = 50,
                 save_interval: int = 500,
                 val_interval_epochs: int = 1,
                 wandb_project: Optional[str] = None,
                 wandb_run_name: Optional[str] = None,
                 best_val_metric_name: str = "val_combined_loss", 
                 best_val_metric_higher_is_better: bool = False):

        self.etp_sphere_model = etp_sphere_model.to(device)
        self.discriminator_model = discriminator_model.to(device) # Initialize discriminator
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.lr_sphere_wubu_core = lr_sphere_wubu_core
        self.lr_sphere_mlps = lr_sphere_mlps
        self.lr_discriminator = lr_discriminator # Store D LR
        
        self.optimizer_kwargs_wubu_core = optimizer_kwargs_wubu_core if optimizer_kwargs_wubu_core is not None else {}
        self.optimizer_kwargs_mlps = optimizer_kwargs_mlps if optimizer_kwargs_mlps is not None else {}
        self.optimizer_kwargs_discriminator = optimizer_kwargs_discriminator if optimizer_kwargs_discriminator is not None else {} # Store D kwargs

        self.lambda_ala = lambda_ala
        self.lambda_rec = lambda_rec
        self.lambda_vsp = lambda_vsp
        
        self.device = device
        self.epochs = epochs
        self.grad_accum_steps = grad_accum_steps if grad_accum_steps > 0 else 1
        self.use_amp = use_amp if self.device.type == 'cuda' else False
        self.global_max_grad_norm = global_max_grad_norm if global_max_grad_norm > 0 else -1.0
        
        self.q_controller_enabled = q_controller_enabled
        _default_q_config_local = _dqch_utils.copy() if _dqch_utils else {}

        self.q_config_sphere_wubu_core = q_config_sphere_wubu_core if q_config_sphere_wubu_core is not None else _default_q_config_local
        self.q_config_sphere_mlps = q_config_sphere_mlps if q_config_sphere_mlps is not None else _default_q_config_local
        self.q_config_discriminator = q_config_discriminator if q_config_discriminator is not None else _default_q_config_local # Store D Q-config
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval_epochs = val_interval_epochs
        
        self.current_epoch = 0
        self.global_step = 0
        
        self.best_val_metric = float('-inf') if best_val_metric_higher_is_better else float('inf')
        self.best_val_metric_name = best_val_metric_name
        self.best_val_metric_higher_is_better = best_val_metric_higher_is_better

        self.wandb_run = None
        if wandb_project and wandb is not None:
            try:
                self.wandb_run = wandb.init(project=wandb_project, name=wandb_run_name, config=self._get_config_dict()) # type: ignore
                if self.wandb_run: 
                    wandb.watch(self.etp_sphere_model, log="all", log_freq=max(1,log_interval*5)) # type: ignore
                    wandb.watch(self.discriminator_model, log="all", log_freq=max(1,log_interval*5)) # type: ignore
            except Exception as e_wandb_ph2:
                logger.error(f"WandB initialization failed for Phase 2 Trainer: {e_wandb_ph2}. Disabling WandB.")
                self.wandb_run = None

        self._setup_optimizers_and_q_controllers()

        self.scaler_sphere = GradScaler(enabled=self.use_amp)
        self.scaler_discriminator = GradScaler(enabled=self.use_amp) # Scaler for discriminator

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ETPTrainerPhase2 initialized. Device: {self.device}, AMP: {self.use_amp}")

    def _get_config_dict(self) -> Dict[str, Any]:
        # Returns a dictionary of relevant trainer configurations for logging.
        return {
            "phase": 2,
            "lr_sphere_wubu_core": self.lr_sphere_wubu_core, "lr_sphere_mlps": self.lr_sphere_mlps,
            "lr_discriminator": self.lr_discriminator,
            "lambda_ala": self.lambda_ala, "lambda_rec": self.lambda_rec, "lambda_vsp": self.lambda_vsp, 
            "epochs": self.epochs, "grad_accum_steps": self.grad_accum_steps, "use_amp": self.use_amp,
            "global_max_grad_norm": self.global_max_grad_norm, "q_controller_enabled": self.q_controller_enabled,
            "best_val_metric_name": self.best_val_metric_name,
            "best_val_metric_higher_is_better": self.best_val_metric_higher_is_better,
        }

    def _setup_optimizers_and_q_controllers(self) -> None:
        # ETP Sphere Parameters
        wubu_core_params_ids = set(id(p) for p in self.etp_sphere_model.wubu_core.parameters())
        wubu_core_params_list = [p for p in self.etp_sphere_model.wubu_core.parameters() if p.requires_grad]
        mlp_params_list = [p for n, p in self.etp_sphere_model.named_parameters() if p.requires_grad and id(p) not in wubu_core_params_ids]
        
        self.q_controllers: Dict[str, Optional[HAKMEMQController]] = {
            "sphere_wubu_core": None, "sphere_mlps": None, "discriminator": None
        }

        # Sphere WuBu Core Optimizer
        self.optimizer_sphere_wubu_core = RiemannianEnhancedSGD(
            wubu_core_params_list if wubu_core_params_list else [nn.Parameter(torch.zeros(1))], 
            lr=self.lr_sphere_wubu_core, 
            q_learning_config=self.q_config_sphere_wubu_core if self.q_controller_enabled else None,
            optimizer_type="generator_wubu_core_phase2", q_logger_suffix="SphereWuBuCoreP2", 
            **self.optimizer_kwargs_wubu_core
        )
        if self.q_controller_enabled and hasattr(self.optimizer_sphere_wubu_core, 'get_q_controller'):
            self.q_controllers["sphere_wubu_core"] = self.optimizer_sphere_wubu_core.get_q_controller()

        # Sphere MLPs Optimizer (Head + Decoder)
        self.optimizer_sphere_mlps = optim.AdamW(
            mlp_params_list if mlp_params_list else [nn.Parameter(torch.zeros(1))], 
            lr=self.lr_sphere_mlps, **self.optimizer_kwargs_mlps
        )
        if self.q_controller_enabled:
            self.q_controllers["sphere_mlps"] = HAKMEMQController(
                initial_lr=self.lr_sphere_mlps, config=self.q_config_sphere_mlps, logger_suffix="SphereMLPsP2"
            )

        # Discriminator Optimizer
        disc_params_list = list(self.discriminator_model.parameters())
        self.optimizer_discriminator = optim.AdamW(
            disc_params_list if disc_params_list else [nn.Parameter(torch.zeros(1))], 
            lr=self.lr_discriminator, **self.optimizer_kwargs_discriminator
        )
        if self.q_controller_enabled:
            self.q_controllers["discriminator"] = HAKMEMQController(
                initial_lr=self.lr_discriminator, config=self.q_config_discriminator, logger_suffix="DiscriminatorP2"
            )
        logger.info("Phase 2 Optimizers and Q-Controllers (if specified) set up.")

    def _get_q_controller_for_optimizer(self, optimizer_name: str) -> Optional[HAKMEMQController]:
        return self.q_controllers.get(optimizer_name)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        embeddings_A = batch['source_A'].to(self.device, non_blocking=True)
        embeddings_B = batch['source_B'].to(self.device, non_blocking=True)
        
        raw_losses_dict: Dict[str, float] = {} 
        loss_d_tensor: Optional[torch.Tensor] = None
        loss_g_total_tensor: Optional[torch.Tensor] = None

        # === Train Discriminator ===
        self.etp_sphere_model.eval() 
        self.discriminator_model.train()
        for param in self.etp_sphere_model.parameters(): param.requires_grad = False 
        for param in self.discriminator_model.parameters(): param.requires_grad = True
        # self.optimizer_discriminator.zero_grad(set_to_none=True) # Zeroed in train_epoch

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            latents_A_d = self.etp_sphere_model.get_latent(embeddings_A).detach()
            latents_B_d = self.etp_sphere_model.get_latent(embeddings_B).detach()
            d_output_A = self.discriminator_model(latents_A_d)
            d_output_B = self.discriminator_model(latents_B_d)
            loss_d_tensor = calculate_adversarial_latent_alignment_loss_discriminator(d_output_A, d_output_B)
        
        if loss_d_tensor is not None: raw_losses_dict['loss_d_phase2'] = loss_d_tensor.item()
        
        # === Train ETP Sphere (Generator) ===
        self.etp_sphere_model.train() 
        self.discriminator_model.eval() 
        for param in self.etp_sphere_model.parameters(): param.requires_grad = True
        for param in self.discriminator_model.parameters(): param.requires_grad = False
        # self.optimizer_sphere_wubu_core.zero_grad(set_to_none=True) # Zeroed in train_epoch
        # self.optimizer_sphere_mlps.zero_grad(set_to_none=True)     # Zeroed in train_epoch

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            latents_A_for_g = self.etp_sphere_model.get_latent(embeddings_A)
            latents_B_for_g = self.etp_sphere_model.get_latent(embeddings_B)
            d_output_A_for_g = self.discriminator_model(latents_A_for_g)
            d_output_B_for_g = self.discriminator_model(latents_B_for_g)
            
            loss_g_ala_tensor = calculate_adversarial_latent_alignment_loss_generator(d_output_A_for_g, d_output_B_for_g)
            reconstructed_A = self.etp_sphere_model(embeddings_A)
            loss_rec_tensor = calculate_reconstruction_loss(reconstructed_A, embeddings_A)
            loss_vsp_tensor = calculate_vector_space_preservation_loss(embeddings_A, latents_A_for_g) # VSP on Corpus A latents
            
            loss_g_total_tensor = (self.lambda_ala * loss_g_ala_tensor) + \
                                  (self.lambda_rec * loss_rec_tensor) + \
                                  (self.lambda_vsp * loss_vsp_tensor)
        
        if loss_g_ala_tensor is not None: raw_losses_dict['loss_g_ala_phase2'] = loss_g_ala_tensor.item()
        if loss_rec_tensor is not None: raw_losses_dict['loss_rec_phase2'] = loss_rec_tensor.item()
        if loss_vsp_tensor is not None: raw_losses_dict['loss_vsp_phase2'] = loss_vsp_tensor.item()
        if loss_g_total_tensor is not None: raw_losses_dict['loss_g_total_phase2'] = loss_g_total_tensor.item()
        
        for param in self.etp_sphere_model.parameters(): param.requires_grad = True
        for param in self.discriminator_model.parameters(): param.requires_grad = True
        
        return raw_losses_dict, loss_d_tensor, loss_g_total_tensor

    def train_epoch(self) -> Dict[str,float]:
        epoch_losses_sum = defaultdict(float); num_batches_this_epoch = 0; batch_times: List[float] = []
        
        self.optimizer_discriminator.zero_grad(set_to_none=True)
        self.optimizer_sphere_wubu_core.zero_grad(set_to_none=True)
        self.optimizer_sphere_mlps.zero_grad(set_to_none=True)

        for batch_idx, batch_data in enumerate(self.train_loader): # type: ignore
            start_time = time.time()
            step_raw_losses, loss_d_tensor, loss_g_total_tensor = self._train_step(batch_data) # type: ignore
            
            if loss_d_tensor is not None: pass # Conceptual: self.scaler_discriminator.scale(loss_d_tensor / self.grad_accum_steps).backward()
            if loss_g_total_tensor is not None: pass # Conceptual: self.scaler_sphere.scale(loss_g_total_tensor / self.grad_accum_steps).backward()

            for k, v in step_raw_losses.items(): epoch_losses_sum[k] += v 
            num_batches_this_epoch +=1; batch_times.append(time.time() - start_time)

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.q_controller_enabled:
                    for opt_name, current_opt in [("sphere_mlps",self.optimizer_sphere_mlps), ("discriminator",self.optimizer_discriminator)]:
                        qc = self._get_q_controller_for_optimizer(opt_name)
                        if qc:
                            loss_key = 'loss_g_total_phase2' if "sphere" in opt_name else 'loss_d_phase2'
                            new_lr = qc.choose_action(current_loss_val=step_raw_losses.get(loss_key))
                            if new_lr != current_opt.param_groups[0]['lr']:
                                logger.info(f"QController ({qc.logger_suffix}): Updating LR for {opt_name} to {new_lr:.2e}")
                                for pg in current_opt.param_groups: pg['lr'] = new_lr
                
                # Conceptual Optimizer Steps & Grad Clipping (omitted for CODING-ONLY)
                
                self.optimizer_discriminator.zero_grad(set_to_none=True)
                self.optimizer_sphere_wubu_core.zero_grad(set_to_none=True)
                self.optimizer_sphere_mlps.zero_grad(set_to_none=True)
                self.global_step += 1

                if self.log_interval > 0 and self.global_step % self.log_interval == 0:
                    avg_bt = sum(batch_times)/len(batch_times) if batch_times else 0; batch_times=[]
                    avg_losses = {f"train_p2/{k}": v_sum/num_batches_this_epoch for k,v_sum in epoch_losses_sum.items()}
                    log_metrics = {**avg_losses}
                    log_metrics.update({
                        "train_p2/lr_wubu_core": self.optimizer_sphere_wubu_core.param_groups[0]['lr'],
                        "train_p2/lr_mlps": self.optimizer_sphere_mlps.param_groups[0]['lr'],
                        "train_p2/lr_disc": self.optimizer_discriminator.param_groups[0]['lr'],
                        "train_p2/avg_batch_time_ms": avg_bt*1000,
                        "progress/global_step": self.global_step, "progress/epoch": self.current_epoch+1
                    })
                    logger.info(f"Epoch {self.current_epoch+1} | Step {self.global_step} | " + " | ".join([f"{k.split('/')[-1]}:{v:.4f}" for k,v in avg_losses.items() if v is not None]))
                    if self.wandb_run: self.wandb_run.log(log_metrics, step=self.global_step)
                
                if self.save_interval > 0 and self.global_step % self.save_interval == 0:
                    self._save_checkpoint(is_best=False, reason="interval_step_p2")
        
        self.current_epoch += 1
        return {k: v / num_batches_this_epoch if num_batches_this_epoch > 0 else 0.0 for k,v in epoch_losses_sum.items()}

    def validate_epoch(self) -> Dict[str, float]:
        if not self.val_loader: return {"val_p2_no_loader": 0.0}
        
        self.etp_sphere_model.eval(); self.discriminator_model.eval()
        val_losses = defaultdict(float); num_val_batches = 0
        val_losses.update({'val_p2_mmd_latent':0.0, 'val_p2_semantic_coherence':0.0, 'val_p2_latent_viz':0.0, 'val_p2_wubu_geom':0.0})

        with torch.no_grad():
            for batch_data in self.val_loader: # type: ignore
                emb_A = batch_data['source_A'].to(self.device,non_blocking=True); emb_B = batch_data['source_B'].to(self.device,non_blocking=True) # type: ignore
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    recon_A = self.etp_sphere_model(emb_A); lat_A = self.etp_sphere_model.get_latent(emb_A)
                    val_losses['val_p2_loss_rec'] += calculate_reconstruction_loss(recon_A, emb_A).item()
                    val_losses['val_p2_loss_vsp'] += calculate_vector_space_preservation_loss(emb_A, lat_A).item()
                    
                    lat_B = self.etp_sphere_model.get_latent(emb_B)
                    d_out_A = self.discriminator_model(lat_A); d_out_B = self.discriminator_model(lat_B)
                    preds_A = (torch.sigmoid(d_out_A)>0.5).float(); preds_B = (torch.sigmoid(d_out_B)<0.5).float()
                    val_losses['val_p2_disc_accuracy'] += (preds_A.sum()+preds_B.sum())/(len(preds_A)+len(preds_B)+1e-8).item()
                num_val_batches+=1
        
        avg_val = {k:v/num_val_batches if num_val_batches>0 else 0.0 for k,v in val_losses.items()}
        avg_val['val_p2_combined_loss'] = avg_val.get('val_p2_loss_rec',0.0) + avg_val.get('val_p2_loss_vsp',0.0) - avg_val.get('val_p2_disc_accuracy',0.0) # Example combined

        if self.best_val_metric_name not in avg_val:
            avg_val[self.best_val_metric_name] = avg_val.get('val_p2_loss_rec', float('inf') if not self.best_val_metric_higher_is_better else float('-inf'))
        
        logger.info(f"Validation P2 Epoch {self.current_epoch}: " + " | ".join([f"{k}:{v:.4f}" for k,v in avg_val.items()]))
        if self.wandb_run:
            wandb_log = {f"val_p2/{k.replace('val_p2_','')}":v for k,v in avg_val.items()}
            wandb_log["progress/epoch"] = self.current_epoch
            self.wandb_run.log(wandb_log, step=self.global_step)
        return avg_val

    def _save_checkpoint(self, is_best: bool = False, reason: str = "") -> None:
        name = ["ckpt_p2", reason] if reason else ["ckpt_p2"]
        if is_best: name.append("best")
        name.extend([f"ep{self.current_epoch}",f"gs{self.global_step}"]); fn="_".join(filter(None,name))+".pth.tar"; fp=self.checkpoint_dir/fn
        
        state = {
            'phase':2, 'epoch':self.current_epoch, 'global_step':self.global_step,
            'model_state':self.etp_sphere_model.state_dict(), 'disc_state':self.discriminator_model.state_dict(),
            'opt_wubu_state':self.optimizer_sphere_wubu_core.state_dict(), 'opt_mlp_state':self.optimizer_sphere_mlps.state_dict(),
            'opt_disc_state':self.optimizer_discriminator.state_dict(),
            'scaler_sphere_state':self.scaler_sphere.state_dict(), 'scaler_disc_state':self.scaler_discriminator.state_dict(),
            'best_val_metric':self.best_val_metric, 'best_val_metric_name':self.best_val_metric_name,
            'best_val_metric_higher_is_better':self.best_val_metric_higher_is_better,
        }
        for qc_n, qc_i in self.q_controllers.items():
            if qc_i: state[f'q_ctrl_{qc_n}_state'] = qc_i.state_dict()
        logger.info(f"Ckpt structure for Phase 2 to {fp} (CODING-ONLY: Not saved).")

    def load_checkpoint(self, path: str, load_optimizers: bool=True, load_q_controllers: bool=True) -> None:
        fp=Path(path); logger.info(f"Conceptual load P2 ckpt: {fp} (CODING-ONLY: No file read).")
        # ckpt: Dict[str,Any] = torch.load(fp, map_location=self.device) # Omitted for CODING-ONLY
        ckpt: Dict[str,Any] = {'epoch':0, 'global_step':0, 'best_val_metric':float('inf')} # Dummy
        self.current_epoch=ckpt.get('epoch',0); self.global_step=ckpt.get('global_step',0)
        self.best_val_metric=ckpt.get('best_val_metric', self.best_val_metric)
        # Conceptual loads: self.etp_sphere_model.load_state_dict(ckpt.get('model_state',{})) etc.
        if load_q_controllers:
            for qc_n, qc_i in self.q_controllers.items():
                if qc_i and f'q_ctrl_{qc_n}_state' in ckpt: pass # Conceptual: qc_i.load_state_dict(...)
        logger.info(f"P2 Ckpt conceptually loaded. Resume ep {self.current_epoch+1}, gs {self.global_step}.")

    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        if resume_from_checkpoint: self.load_checkpoint(resume_from_checkpoint)
        init_epochs=self.current_epoch; logger.info(f"Start P2 training. Target ep: {self.epochs}. Current completed: {init_epochs}.")
        for _ in range(init_epochs, self.epochs):
            logger.info(f"Commencing P2 Epoch {self.current_epoch+1}/{self.epochs}")
            epoch_losses = self.train_epoch()
            if self.q_controller_enabled:
                for opt_n, _ in self.q_controllers.items():
                    qc = self.q_controllers.get(opt_n)
                    if qc: qc.log_reward(-epoch_losses.get('loss_g_total_phase2' if "sphere" in opt_n else 'loss_d_phase2', float('inf')))
            if self.val_loader and (self.current_epoch % self.val_interval_epochs == 0 or self.current_epoch == self.epochs):
                val_metrics = self.validate_epoch()
                curr_val_met = val_metrics.get(self.best_val_metric_name, float('-inf') if self.best_val_metric_higher_is_better else float('inf'))
                is_better = (curr_val_met > self.best_val_metric) if self.best_val_metric_higher_is_better else (curr_val_met < self.best_val_metric)
                if is_better: self.best_val_metric=curr_val_met; logger.info(f"New best P2 val metric ({self.best_val_metric_name}): {self.best_val_metric:.4f}."); self._save_checkpoint(is_best=True, reason=f"best_val_p2_{self.best_val_metric_name.replace('val_p2_','')}")
            if self.save_interval == 0: self._save_checkpoint(is_best=False, reason="end_of_epoch_p2")
        logger.info(f"P2 Training completed after {self.current_epoch} epochs.")
        if self.wandb_run and hasattr(self.wandb_run, 'finish'): self.wandb_run.finish()

if __name__ == '__main__':
    logger.info("ETPTrainerPhase2 class definition complete (CODING-ONLY CHECK).")
    pass
