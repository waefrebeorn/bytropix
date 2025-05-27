import logging
import time
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
from torch.cuda.amp import GradScaler

# Imports from etp_common
try:
    from draftPY.etp_common.etp_wubu_architectures import ETP_WuBuText_DS_R1_Sphere
    # DeepSeekR1EmbeddingDataset is used by the runner, not directly by trainer, but good for type hints if needed
    # from draftPY.etp_common.etp_datasets import DeepSeekR1EmbeddingDataset 
    from draftPY.etp_common.etp_losses import calculate_reconstruction_loss, calculate_vector_space_preservation_loss
    from draftPY.etp_common.optimizer_utils import RiemannianEnhancedSGD, HAKMEMQController, DEFAULT_CONFIG_QLEARN_HYBRID as _dqch_utils
except ImportError as e:
    # Fallback to stubs if full implementations are not found (for CODING-ONLY robustness)
    logger_stub = logging.getLogger(__name__) 
    logger_stub.error(f"Phase1 Trainer: Critical error importing from etp_common: {e}. Using stubs.")
    
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

    def calculate_reconstruction_loss(*args: Any, **kwargs: Any) -> torch.Tensor: return torch.tensor(0.0, requires_grad=True) # type: ignore[no-untyped-def]
    def calculate_vector_space_preservation_loss(*args: Any, **kwargs: Any) -> torch.Tensor: return torch.tensor(0.0, requires_grad=True) # type: ignore[no-untyped-def]
    
    _dqch_utils = {} # type: ignore[no-redef]
    class RiemannianEnhancedSGD(optim.Optimizer): # type: ignore
        def __init__(self, params: Any, lr: float, q_learning_config: Optional[Dict]=None, **kwargs: Any): # type: ignore[no-untyped-def]
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
    logging.info("wandb not found, WandB logging will be disabled for ETPTrainerPhase1.")

# Configure module-level logger
logger = logging.getLogger(__name__)

class ETPTrainerPhase1:
    def __init__(self,
                 etp_sphere_model: ETP_WuBuText_DS_R1_Sphere,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader],
                 lr_sphere_wubu_core: float,
                 lr_sphere_mlps: float,
                 optimizer_kwargs_wubu_core: Optional[Dict[str, Any]] = None,
                 optimizer_kwargs_mlps: Optional[Dict[str, Any]] = None,
                 lambda_rec: float = 1.0, # Primary focus for Phase 1
                 lambda_vsp: float = 0.0, # Typically zero or very small for pure Phase 1
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 epochs: int = 10,
                 grad_accum_steps: int = 1,
                 use_amp: bool = True,
                 global_max_grad_norm: float = 1.0,
                 q_controller_enabled: bool = True,
                 q_config_sphere_wubu_core: Optional[Dict[str, Any]] = None,
                 q_config_sphere_mlps: Optional[Dict[str, Any]] = None,
                 checkpoint_dir: str = "checkpoints_etp_phase1",
                 log_interval: int = 50,
                 save_interval: int = 500,
                 val_interval_epochs: int = 1,
                 wandb_project: Optional[str] = None,
                 wandb_run_name: Optional[str] = None,
                 best_val_metric_name: str = "val_loss_rec", # Default to reconstruction loss
                 best_val_metric_higher_is_better: bool = False):

        self.etp_sphere_model = etp_sphere_model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.lr_sphere_wubu_core = lr_sphere_wubu_core
        self.lr_sphere_mlps = lr_sphere_mlps
        
        self.optimizer_kwargs_wubu_core = optimizer_kwargs_wubu_core if optimizer_kwargs_wubu_core is not None else {}
        self.optimizer_kwargs_mlps = optimizer_kwargs_mlps if optimizer_kwargs_mlps is not None else {}

        self.lambda_rec = lambda_rec
        self.lambda_vsp = lambda_vsp # Store it, runner script controls its value (e.g. 0.0 for pure REC)
        
        self.device = device
        self.epochs = epochs
        self.grad_accum_steps = grad_accum_steps if grad_accum_steps > 0 else 1
        self.use_amp = use_amp if self.device.type == 'cuda' else False
        self.global_max_grad_norm = global_max_grad_norm if global_max_grad_norm > 0 else -1.0
        
        self.q_controller_enabled = q_controller_enabled
        _default_q_config_local = _dqch_utils.copy() if _dqch_utils else {}

        self.q_config_sphere_wubu_core = q_config_sphere_wubu_core if q_config_sphere_wubu_core is not None else _default_q_config_local
        self.q_config_sphere_mlps = q_config_sphere_mlps if q_config_sphere_mlps is not None else _default_q_config_local
        
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
            except Exception as e_wandb_ph1:
                logger.error(f"WandB initialization failed for Phase 1 Trainer: {e_wandb_ph1}. Disabling WandB.")
                self.wandb_run = None

        self._setup_optimizers_and_q_controllers()

        self.scaler_sphere = GradScaler(enabled=self.use_amp) # Only one scaler needed

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ETPTrainerPhase1 initialized. Device: {self.device}, AMP: {self.use_amp}")

    def _get_config_dict(self) -> Dict[str, Any]:
        return {
            "phase": 1,
            "lr_sphere_wubu_core": self.lr_sphere_wubu_core, "lr_sphere_mlps": self.lr_sphere_mlps,
            "lambda_rec": self.lambda_rec, "lambda_vsp": self.lambda_vsp, "epochs": self.epochs,
            "grad_accum_steps": self.grad_accum_steps, "use_amp": self.use_amp,
            "global_max_grad_norm": self.global_max_grad_norm, "q_controller_enabled": self.q_controller_enabled,
            "best_val_metric_name": self.best_val_metric_name,
            "best_val_metric_higher_is_better": self.best_val_metric_higher_is_better,
        }

    def _setup_optimizers_and_q_controllers(self) -> None:
        wubu_core_params_ids = set(id(p) for p in self.etp_sphere_model.wubu_core.parameters())
        wubu_core_params_list = [p for p in self.etp_sphere_model.wubu_core.parameters() if p.requires_grad]
        mlp_params_list = [p for n, p in self.etp_sphere_model.named_parameters() if p.requires_grad and id(p) not in wubu_core_params_ids]
        
        self.q_controllers: Dict[str, Optional[HAKMEMQController]] = {"sphere_wubu_core": None, "sphere_mlps": None}

        self.optimizer_sphere_wubu_core = RiemannianEnhancedSGD(
            wubu_core_params_list if wubu_core_params_list else [nn.Parameter(torch.zeros(1))], 
            lr=self.lr_sphere_wubu_core, 
            q_learning_config=self.q_config_sphere_wubu_core if self.q_controller_enabled else None,
            optimizer_type="etp_sphere_wubu_core_phase1", q_logger_suffix="SphereWuBuCoreP1", 
            **self.optimizer_kwargs_wubu_core
        )
        if self.q_controller_enabled and hasattr(self.optimizer_sphere_wubu_core, 'get_q_controller'):
            self.q_controllers["sphere_wubu_core"] = self.optimizer_sphere_wubu_core.get_q_controller()

        self.optimizer_sphere_mlps = optim.AdamW(mlp_params_list if mlp_params_list else [nn.Parameter(torch.zeros(1))], lr=self.lr_sphere_mlps, **self.optimizer_kwargs_mlps)
        if self.q_controller_enabled:
            self.q_controllers["sphere_mlps"] = HAKMEMQController(initial_lr=self.lr_sphere_mlps, config=self.q_config_sphere_mlps, logger_suffix="SphereMLPsP1")
        
        logger.info("Phase 1 Optimizers and Q-Controllers (if specified) set up.")


    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Optional[torch.Tensor]]:
        # For Phase 1, only embeddings_A are strictly needed for reconstruction.
        # If VSP is used, then embeddings_A are used for pairwise similarity.
        embeddings_A = batch['source_A'].to(self.device, non_blocking=True)
        # embeddings_B = batch['source_B'].to(self.device, non_blocking=True) # Not used in Phase 1 _train_step

        raw_losses_dict: Dict[str, float] = {}
        loss_total_tensor: Optional[torch.Tensor] = None
        
        self.etp_sphere_model.train()
        for param in self.etp_sphere_model.parameters(): param.requires_grad = True # Ensure all relevant params are trainable

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            reconstructed_A = self.etp_sphere_model(embeddings_A)
            loss_rec_tensor = calculate_reconstruction_loss(reconstructed_A, embeddings_A)
            raw_losses_dict['loss_rec'] = loss_rec_tensor.item()
            
            loss_total_tensor = self.lambda_rec * loss_rec_tensor

            if self.lambda_vsp > 0: # Only calculate VSP if its weight is non-zero
                latents_A = self.etp_sphere_model.get_latent(embeddings_A)
                loss_vsp_tensor = calculate_vector_space_preservation_loss(embeddings_A, latents_A)
                raw_losses_dict['loss_vsp'] = loss_vsp_tensor.item()
                loss_total_tensor = loss_total_tensor + (self.lambda_vsp * loss_vsp_tensor)
        
        if loss_total_tensor is not None: raw_losses_dict['loss_total_phase1'] = loss_total_tensor.item()
        
        return raw_losses_dict, loss_total_tensor

    def train_epoch(self) -> Dict[str,float]:
        epoch_losses_sum = defaultdict(float); num_batches_this_epoch = 0; batch_times: List[float] = []
        
        self.optimizer_sphere_wubu_core.zero_grad(set_to_none=True)
        self.optimizer_sphere_mlps.zero_grad(set_to_none=True)

        for batch_idx, batch_data in enumerate(self.train_loader): # type: ignore
            start_time = time.time()
            step_raw_losses, loss_total_tensor = self._train_step(batch_data) # type: ignore
            
            if loss_total_tensor is not None: # Conceptual backward
                # self.scaler_sphere.scale(loss_total_tensor / self.grad_accum_steps).backward()
                pass

            for k, v in step_raw_losses.items(): epoch_losses_sum[k] += v 
            num_batches_this_epoch +=1; batch_times.append(time.time() - start_time)

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.q_controller_enabled:
                    qc_mlps = self.q_controllers.get("sphere_mlps")
                    if qc_mlps:
                        new_lr_mlps = qc_mlps.choose_action(current_loss_val=step_raw_losses.get('loss_total_phase1'))
                        if new_lr_mlps != self.optimizer_sphere_mlps.param_groups[0]['lr']:
                            logger.info(f"QController (SphereMLPsP1): Updating LR to {new_lr_mlps:.2e}")
                            for pg in self.optimizer_sphere_mlps.param_groups: pg['lr'] = new_lr_mlps
                    # RESGD Q-controller update is typically internal or based on its own step logic
                
                # Conceptual Optimizer Steps & Grad Clipping
                # if self.global_max_grad_norm > 0:
                #    self.scaler_sphere.unscale_(self.optimizer_sphere_wubu_core)
                #    self.scaler_sphere.unscale_(self.optimizer_sphere_mlps)
                #    torch.nn.utils.clip_grad_norm_(self.etp_sphere_model.parameters(), self.global_max_grad_norm)
                # self.scaler_sphere.step(self.optimizer_sphere_wubu_core)
                # self.scaler_sphere.step(self.optimizer_sphere_mlps)
                # self.scaler_sphere.update()
                
                self.optimizer_sphere_wubu_core.zero_grad(set_to_none=True)
                self.optimizer_sphere_mlps.zero_grad(set_to_none=True)
                self.global_step += 1

                if self.log_interval > 0 and self.global_step % self.log_interval == 0:
                    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0; batch_times = []
                    current_avg_losses = {f"train_p1/{k}": v_sum / num_batches_this_epoch for k, v_sum in epoch_losses_sum.items()}
                    log_metrics = {**current_avg_losses}
                    log_metrics["train_p1/lr_wubu_core"] = self.optimizer_sphere_wubu_core.param_groups[0]['lr']
                    log_metrics["train_p1/lr_mlps"] = self.optimizer_sphere_mlps.param_groups[0]['lr']
                    log_metrics["train_p1/avg_batch_time_ms"] = avg_batch_time * 1000
                    log_metrics["progress/global_step"] = self.global_step
                    log_metrics["progress/epoch"] = self.current_epoch + 1
                    logger.info(f"Epoch {self.current_epoch + 1} | Step {self.global_step} | " + " | ".join([f"{k.split('/')[-1]}: {v:.4f}" for k,v in current_avg_losses.items() if v is not None]))
                    if self.wandb_run: self.wandb_run.log(log_metrics, step=self.global_step)
                
                if self.save_interval > 0 and self.global_step % self.save_interval == 0:
                    self._save_checkpoint(is_best=False, reason="interval_step_p1")
        
        self.current_epoch += 1
        avg_epoch_losses = {k: v / num_batches_this_epoch if num_batches_this_epoch > 0 else 0.0 for k,v in epoch_losses_sum.items()}
        return avg_epoch_losses

    def validate_epoch(self) -> Dict[str, float]:
        if not self.val_loader: return {"val_p1_no_loader": 0.0}
        
        self.etp_sphere_model.eval()
        val_losses_sum = defaultdict(float); num_val_batches = 0
        
        # Initialize placeholders for metrics that might not be calculated every time
        val_losses_sum.update({'val_p1_loss_vsp': 0.0})


        with torch.no_grad():
            for batch_data in self.val_loader: # type: ignore
                embeddings_A = batch_data['source_A'].to(self.device, non_blocking=True) # type: ignore
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    reconstructed_A = self.etp_sphere_model(embeddings_A)
                    val_losses_sum['val_p1_loss_rec'] += calculate_reconstruction_loss(reconstructed_A, embeddings_A).item()
                    
                    if self.lambda_vsp > 0: # Only calculate if relevant for this phase's config
                        latents_A = self.etp_sphere_model.get_latent(embeddings_A)
                        val_losses_sum['val_p1_loss_vsp'] += calculate_vector_space_preservation_loss(embeddings_A, latents_A).item()
                num_val_batches +=1
        
        avg_val_losses = {k: v / num_val_batches if num_val_batches > 0 else 0.0 for k,v in val_losses_sum.items()}
        # For Phase 1, combined loss is primarily reconstruction, VSP if active.
        avg_val_losses['val_p1_combined_loss'] = avg_val_losses.get('val_p1_loss_rec', 0.0) + avg_val_losses.get('val_p1_loss_vsp', 0.0)

        if self.best_val_metric_name not in avg_val_losses:
             avg_val_losses[self.best_val_metric_name] = avg_val_losses.get('val_p1_loss_rec', float('inf') if not self.best_val_metric_higher_is_better else float('-inf'))

        logger.info(f"Validation P1 Epoch {self.current_epoch}: " + " | ".join([f"{k}: {v:.4f}" for k,v in avg_val_losses.items()]))
        if self.wandb_run:
            wandb_log = {f"val_p1/{k.replace('val_p1_','')}": v for k,v in avg_val_losses.items()}
            wandb_log["progress/epoch"] = self.current_epoch
            self.wandb_run.log(wandb_log, step=self.global_step)
        return avg_val_losses

    def _save_checkpoint(self, is_best: bool = False, reason: str = "") -> None:
        name_parts = ["checkpoint_p1", reason] if reason else ["checkpoint_p1"]
        if is_best: name_parts.append("best")
        name_parts.extend([f"epoch{self.current_epoch}", f"step{self.global_step}"])
        filename = "_".join(filter(None, name_parts)) + ".pth.tar"
        filepath = self.checkpoint_dir / filename
        
        checkpoint_data: Dict[str, Any] = {
            'phase': 1, 'epoch': self.current_epoch, 'global_step': self.global_step,
            'etp_sphere_model_state_dict': self.etp_sphere_model.state_dict(),
            'optimizer_sphere_wubu_core_state_dict': self.optimizer_sphere_wubu_core.state_dict(),
            'optimizer_sphere_mlps_state_dict': self.optimizer_sphere_mlps.state_dict(),
            'scaler_sphere_state_dict': self.scaler_sphere.state_dict(),
            'best_val_metric': self.best_val_metric,
            'best_val_metric_name': self.best_val_metric_name,
            'best_val_metric_higher_is_better': self.best_val_metric_higher_is_better,
        }
        for qc_name, qc_instance in self.q_controllers.items():
            if qc_instance: checkpoint_data[f'q_controller_{qc_name}_state_dict'] = qc_instance.state_dict()
        
        logger.info(f"Checkpoint structure for Phase 1 prepared for {filepath} (CODING-ONLY: Not actually saved).")
        # torch.save(checkpoint_data, filepath) 
        # if self.wandb_run and is_best and hasattr(wandb, 'save') and wandb.save is not None:
        #    wandb.save(str(filepath), base_path=str(self.checkpoint_dir))

    def load_checkpoint(self, path: str, load_optimizers: bool = True, load_q_controllers: bool = True) -> None:
        filepath = Path(path)
        logger.info(f"Attempting to load Phase 1 checkpoint from {filepath} (CODING-ONLY: No actual file read).")
        checkpoint: Dict[str, Any] = { # Dummy structure for CODING-ONLY
            'epoch': 0, 'global_step': 0, 'best_val_metric': float('inf'),
            'best_val_metric_name': self.best_val_metric_name,
            'best_val_metric_higher_is_better': self.best_val_metric_higher_is_better,
        } 
        
        # self.etp_sphere_model.load_state_dict(checkpoint.get('etp_sphere_model_state_dict', {}))
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_metric = checkpoint.get('best_val_metric', self.best_val_metric)
        # ... (load other metrics)

        if load_optimizers:
            # self.optimizer_sphere_wubu_core.load_state_dict(checkpoint.get('optimizer_sphere_wubu_core_state_dict',{}))
            # self.optimizer_sphere_mlps.load_state_dict(checkpoint.get('optimizer_sphere_mlps_state_dict',{}))
            # self.scaler_sphere.load_state_dict(checkpoint.get('scaler_sphere_state_dict',{}))
            pass
        if load_q_controllers:
            for qc_name, qc_instance in self.q_controllers.items():
                if qc_instance and f'q_controller_{qc_name}_state_dict' in checkpoint:
                    # qc_instance.load_state_dict(checkpoint[f'q_controller_{qc_name}_state_dict'])
                    pass
        logger.info(f"Phase 1 Checkpoint conceptually loaded. Resuming from epoch {self.current_epoch + 1}.")

    def train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        if resume_from_checkpoint: self.load_checkpoint(resume_from_checkpoint)
        
        initial_completed_epochs = self.current_epoch
        logger.info(f"Starting Phase 1 training. Target epochs: {self.epochs}. Current completed: {initial_completed_epochs}.")
        
        for epoch_iter in range(initial_completed_epochs, self.epochs):
            logger.info(f"Commencing Phase 1 Epoch {self.current_epoch + 1}/{self.epochs}")
            epoch_train_losses = self.train_epoch() 

            if self.q_controller_enabled:
                for opt_name, _ in self.q_controllers.items(): # Iterate through configured Q-controllers
                    qc = self.q_controllers.get(opt_name)
                    if qc: qc.log_reward(-epoch_train_losses.get('loss_total_phase1', float('inf')))
            
            if self.val_loader and (self.current_epoch % self.val_interval_epochs == 0 or self.current_epoch == self.epochs):
                val_metrics = self.validate_epoch()
                current_val_metric = val_metrics.get(self.best_val_metric_name, float('-inf') if self.best_val_metric_higher_is_better else float('inf'))
                is_better = (current_val_metric > self.best_val_metric) if self.best_val_metric_higher_is_better else (current_val_metric < self.best_val_metric)
                if is_better:
                    self.best_val_metric = current_val_metric
                    logger.info(f"New best Phase 1 val metric ({self.best_val_metric_name}): {self.best_val_metric:.4f}.")
                    self._save_checkpoint(is_best=True, reason=f"best_val_p1_{self.best_val_metric_name.replace('val_p1_','')}")
            
            if self.save_interval == 0: self._save_checkpoint(is_best=False, reason="end_of_epoch_p1")

        logger.info(f"Phase 1 Training completed after {self.current_epoch} epochs.")
        if self.wandb_run and hasattr(self.wandb_run, 'finish'): self.wandb_run.finish()

if __name__ == '__main__':
    logger.info("ETPTrainerPhase1 class definition complete (CODING-ONLY CHECK).")
    pass
