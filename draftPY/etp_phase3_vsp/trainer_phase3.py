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
    from draftPY.etp_common.etp_losses import (
        calculate_reconstruction_loss, 
        calculate_vector_space_preservation_loss,
        calculate_adversarial_latent_alignment_loss_discriminator,
        calculate_adversarial_latent_alignment_loss_generator
    )
    from draftPY.etp_common.optimizer_utils import (
        RiemannianEnhancedSGD, 
        HAKMEMQController, 
        GradientStats, 
        DEFAULT_CONFIG_QLEARN_HYBRID as _dqch_utils 
    )
except ImportError as e:
    logger_stub = logging.getLogger(__name__) 
    logger_stub.error(f"Phase3 Trainer: Critical error importing from etp_common: {e}. Using stubs.")
    # Define stubs (copied from Phase2 Trainer for brevity, should be identical)
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
            param_list = list(params) if not isinstance(params, list) else params;
            if not param_list: param_list = [nn.Parameter(torch.randn(1))] # type: ignore[no-untyped-call]
            super().__init__(param_list, {"lr": lr}); self.q_controller = HAKMEMQController(lr, q_learning_config if q_learning_config else {}, logger_suffix=kwargs.get("q_logger_suffix","")) if q_learning_config else None # type: ignore[operator]
        def get_q_controller(self) -> Optional['HAKMEMQController']: return self.q_controller # type: ignore[name-defined]
        def step(self, closure: Any =None) -> Optional[torch.Tensor]: return None # type: ignore[override]
    class HAKMEMQController: # type: ignore
        def __init__(self, initial_lr: float, config: Dict[str,Any], logger_suffix: str = ""): self.lr = initial_lr; self.logger_suffix=logger_suffix # type: ignore[no-untyped-def]
        def choose_action(self, current_metric_val: Optional[float]=None, current_loss_val: Optional[float]=None) -> float: return self.lr
        def log_reward(self, reward: float, metric_val: Optional[float]=None, loss_val: Optional[float]=None): pass # type: ignore[no-untyped-def]
        def get_current_lr(self) -> float: return self.lr
        def state_dict(self) -> Dict[str, Any]: return {"lr": self.lr, "logger_suffix": self.logger_suffix} 
        def load_state_dict(self, state_dict: Dict[str, Any]) -> None: self.lr = state_dict.get("lr", self.lr) # type: ignore[no-untyped-def]

try: import wandb
except ImportError: wandb = None; logging.info("wandb not found for ETPTrainerPhase3.") # type: ignore

logger = logging.getLogger(__name__)

class ETPTrainerPhase3: # Renamed class
    def __init__(self,
                 etp_sphere_model: ETP_WuBuText_DS_R1_Sphere,
                 discriminator_model: LatentDiscriminatorMLP, 
                 train_loader: DataLoader, val_loader: Optional[DataLoader],
                 lr_sphere_wubu_core: float, lr_sphere_mlps: float, lr_discriminator: float,
                 optimizer_kwargs_wubu_core: Optional[Dict[str, Any]] = None,
                 optimizer_kwargs_mlps: Optional[Dict[str, Any]] = None,
                 optimizer_kwargs_discriminator: Optional[Dict[str, Any]] = None,
                 lambda_ala: float = 0.01, # Default ALA weight for Phase 3 (can be small or zero)
                 lambda_rec: float = 1.0,  # REC can still be active
                 lambda_vsp: float = 0.1,  # VSP is a key focus in Phase 3
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 epochs: int = 10, grad_accum_steps: int = 1, use_amp: bool = True,
                 global_max_grad_norm: float = 1.0, q_controller_enabled: bool = True,
                 q_config_sphere_wubu_core: Optional[Dict[str, Any]] = None,
                 q_config_sphere_mlps: Optional[Dict[str, Any]] = None,
                 q_config_discriminator: Optional[Dict[str, Any]] = None,
                 checkpoint_dir: str = "checkpoints_etp_phase3", # Updated default
                 log_interval: int = 50, save_interval: int = 500, val_interval_epochs: int = 1,
                 wandb_project: Optional[str] = "ETP_Phase3_VSP", # Updated default
                 wandb_run_name: Optional[str] = None,
                 best_val_metric_name: str = "val_loss_vsp", # Focus on VSP
                 best_val_metric_higher_is_better: bool = False): # Loss, so lower is better

        self.etp_sphere_model = etp_sphere_model.to(device)
        self.discriminator_model = discriminator_model.to(device) 
        self.train_loader = train_loader; self.val_loader = val_loader
        self.lr_sphere_wubu_core = lr_sphere_wubu_core; self.lr_sphere_mlps = lr_sphere_mlps
        self.lr_discriminator = lr_discriminator
        self.optimizer_kwargs_wubu_core = optimizer_kwargs_wubu_core or {}; self.optimizer_kwargs_mlps = optimizer_kwargs_mlps or {}
        self.optimizer_kwargs_discriminator = optimizer_kwargs_discriminator or {}
        self.lambda_ala = lambda_ala; self.lambda_rec = lambda_rec; self.lambda_vsp = lambda_vsp
        self.device = device; self.epochs = epochs
        self.grad_accum_steps = grad_accum_steps if grad_accum_steps > 0 else 1
        self.use_amp = use_amp if self.device.type == 'cuda' else False
        self.global_max_grad_norm = global_max_grad_norm if global_max_grad_norm > 0 else -1.0
        
        self.q_controller_enabled = q_controller_enabled
        _default_q_config_local = _dqch_utils.copy() if _dqch_utils else {}
        self.q_config_sphere_wubu_core = q_config_sphere_wubu_core if q_config_sphere_wubu_core is not None else _default_q_config_local
        self.q_config_sphere_mlps = q_config_sphere_mlps if q_config_sphere_mlps is not None else _default_q_config_local
        self.q_config_discriminator = q_config_discriminator if q_config_discriminator is not None else _default_q_config_local
        
        self.checkpoint_dir = Path(checkpoint_dir); self.log_interval = log_interval
        self.save_interval = save_interval; self.val_interval_epochs = val_interval_epochs
        self.current_epoch = 0; self.global_step = 0
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
            except Exception as e_wandb_ph3:
                logger.error(f"WandB initialization failed for Phase 3 Trainer: {e_wandb_ph3}. Disabling WandB.")
                self.wandb_run = None

        self._setup_optimizers_and_q_controllers()
        self.scaler_sphere = GradScaler(enabled=self.use_amp)
        self.scaler_discriminator = GradScaler(enabled=self.use_amp)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ETPTrainerPhase3 initialized. Device: {self.device}, AMP: {self.use_amp}. VSP focus.")

    def _get_config_dict(self) -> Dict[str, Any]:
        return {"phase": 3, "lr_sphere_wubu_core": self.lr_sphere_wubu_core, "lr_sphere_mlps": self.lr_sphere_mlps,
                "lr_discriminator": self.lr_discriminator, "lambda_ala": self.lambda_ala, 
                "lambda_rec": self.lambda_rec, "lambda_vsp": self.lambda_vsp, "epochs": self.epochs,
                "grad_accum_steps": self.grad_accum_steps, "use_amp": self.use_amp,
                "global_max_grad_norm": self.global_max_grad_norm, "q_controller_enabled": self.q_controller_enabled,
                "best_val_metric_name": self.best_val_metric_name,
                "best_val_metric_higher_is_better": self.best_val_metric_higher_is_better}

    def _setup_optimizers_and_q_controllers(self) -> None:
        wubu_core_ids = set(id(p) for p in self.etp_sphere_model.wubu_core.parameters())
        wubu_params = [p for p in self.etp_sphere_model.wubu_core.parameters() if p.requires_grad]
        mlp_params = [p for n,p in self.etp_sphere_model.named_parameters() if p.requires_grad and id(p) not in wubu_core_ids]
        self.q_controllers: Dict[str,Optional[HAKMEMQController]] = {"sphere_wubu_core":None,"sphere_mlps":None,"discriminator":None}

        self.optimizer_sphere_wubu_core = RiemannianEnhancedSGD(
            wubu_params if wubu_params else [nn.Parameter(torch.zeros(1))], lr=self.lr_sphere_wubu_core, 
            q_learning_config=self.q_config_sphere_wubu_core if self.q_controller_enabled else None,
            optimizer_type="generator_wubu_core_phase3", q_logger_suffix="SphereWuBuCoreP3", **self.optimizer_kwargs_wubu_core)
        if self.q_controller_enabled and hasattr(self.optimizer_sphere_wubu_core, 'get_q_controller'):
            self.q_controllers["sphere_wubu_core"] = self.optimizer_sphere_wubu_core.get_q_controller()

        self.optimizer_sphere_mlps = optim.AdamW(mlp_params if mlp_params else [nn.Parameter(torch.zeros(1))], lr=self.lr_sphere_mlps, **self.optimizer_kwargs_mlps)
        if self.q_controller_enabled:
            self.q_controllers["sphere_mlps"] = HAKMEMQController(self.lr_sphere_mlps, self.q_config_sphere_mlps, "SphereMLPsP3")
        
        disc_params = list(self.discriminator_model.parameters())
        self.optimizer_discriminator = optim.AdamW(disc_params if disc_params else [nn.Parameter(torch.zeros(1))], lr=self.lr_discriminator, **self.optimizer_kwargs_discriminator)
        if self.q_controller_enabled:
            self.q_controllers["discriminator"] = HAKMEMQController(self.lr_discriminator, self.q_config_discriminator, "DiscriminatorP3")
        logger.info("Phase 3 Optimizers and Q-Controllers (if specified) set up.")

    # _train_step logic remains the same as Phase 2, as it calculates all losses.
    # The lambda values in __init__ will determine which losses are active.
    # For pure VSP, lambda_ala and lambda_rec might be set to 0 by the runner.
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        embeddings_A = batch['source_A'].to(self.device, non_blocking=True)
        embeddings_B = batch['source_B'].to(self.device, non_blocking=True) 
        raw_losses: Dict[str,float] = {}; loss_d: Optional[torch.Tensor]=None; loss_g: Optional[torch.Tensor]=None

        # Discriminator training (only if ALA is active, controlled by lambda_ala)
        if self.lambda_ala > 0:
            self.etp_sphere_model.eval(); self.discriminator_model.train()
            for p in self.etp_sphere_model.parameters(): p.requires_grad=False
            for p in self.discriminator_model.parameters(): p.requires_grad=True
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                lat_A_d = self.etp_sphere_model.get_latent(embeddings_A).detach()
                lat_B_d = self.etp_sphere_model.get_latent(embeddings_B).detach()
                d_A = self.discriminator_model(lat_A_d); d_B = self.discriminator_model(lat_B_d)
                loss_d = calculate_adversarial_latent_alignment_loss_discriminator(d_A, d_B)
            if loss_d is not None: raw_losses['loss_d_phase3'] = loss_d.item()
        
        # ETP Sphere Model training
        self.etp_sphere_model.train()
        if self.lambda_ala > 0: self.discriminator_model.eval() # D is eval if G uses it
        for p in self.etp_sphere_model.parameters(): p.requires_grad=True
        if self.lambda_ala > 0: # Ensure D params are frozen if D was used
            for p in self.discriminator_model.parameters(): p.requires_grad=False

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            loss_g_components = []
            # Latents are needed for VSP and potentially for ALA
            latents_A_g = self.etp_sphere_model.get_latent(embeddings_A) 
            
            if self.lambda_ala > 0:
                latents_B_g = self.etp_sphere_model.get_latent(embeddings_B)
                d_A_g = self.discriminator_model(latents_A_g); d_B_g = self.discriminator_model(latents_B_g)
                loss_ala_g = calculate_adversarial_latent_alignment_loss_generator(d_A_g, d_B_g)
                raw_losses['loss_g_ala_phase3'] = loss_ala_g.item()
                loss_g_components.append(self.lambda_ala * loss_ala_g)
            
            if self.lambda_rec > 0:
                # For Phase 3, reconstruction might be on embeddings_A or embeddings_B,
                # or not at all if lambda_rec is 0. Here, we assume on A.
                recon_A = self.etp_sphere_model(embeddings_A) 
                loss_rec = calculate_reconstruction_loss(recon_A, embeddings_A)
                raw_losses['loss_rec_phase3'] = loss_rec.item()
                loss_g_components.append(self.lambda_rec * loss_rec)

            if self.lambda_vsp > 0: 
                # VSP can be calculated on latents from Corpus A, Corpus B, or both.
                # Here, we do it on A, and then on B, and sum them or average them.
                loss_vsp_A = calculate_vector_space_preservation_loss(embeddings_A, latents_A_g)
                raw_losses['loss_vsp_A_phase3'] = loss_vsp_A.item()
                loss_g_components.append(self.lambda_vsp * loss_vsp_A) # VSP for A

                if embeddings_B is not None: # If B is available and used for VSP
                    latents_B_for_vsp = self.etp_sphere_model.get_latent(embeddings_B)
                    loss_vsp_B = calculate_vector_space_preservation_loss(embeddings_B, latents_B_for_vsp)
                    raw_losses['loss_vsp_B_phase3'] = loss_vsp_B.item()
                    loss_g_components.append(self.lambda_vsp * loss_vsp_B) # Add VSP for B (e.g. with same lambda)
            
            if loss_g_components: loss_g = sum(loss_g_components) # type: ignore
            else: loss_g = torch.tensor(0.0, device=self.device, requires_grad=True) # No active ETP model losses

        if loss_g is not None: raw_losses['loss_g_total_phase3'] = loss_g.item()
        
        # Restore requires_grad states
        for p in self.etp_sphere_model.parameters(): p.requires_grad=True
        if self.lambda_ala > 0 : 
            for p in self.discriminator_model.parameters(): p.requires_grad=True
        return raw_losses, loss_d, loss_g

    def train_epoch(self) -> Dict[str,float]:
        epoch_losses_sum=defaultdict(float); num_batches=0; batch_times:List[float]=[]
        self.optimizer_discriminator.zero_grad(set_to_none=True)
        self.optimizer_sphere_wubu_core.zero_grad(set_to_none=True)
        self.optimizer_sphere_mlps.zero_grad(set_to_none=True)

        for batch_idx, batch_data in enumerate(self.train_loader): # type: ignore
            start_time=time.time()
            step_raw_losses, loss_d_tensor, loss_g_tensor = self._train_step(batch_data) # type: ignore
            
            if loss_d_tensor is not None and self.lambda_ala > 0: pass # Conceptual D backward
            if loss_g_tensor is not None and (self.lambda_rec > 0 or self.lambda_vsp > 0 or self.lambda_ala > 0) : pass # Conceptual G backward

            for k,v in step_raw_losses.items(): epoch_losses_sum[k]+=v
            num_batches+=1; batch_times.append(time.time()-start_time)

            if (batch_idx+1)%self.grad_accum_steps == 0:
                if self.q_controller_enabled: 
                    for opt_n,opt_inst in [("sphere_mlps",self.optimizer_sphere_mlps), ("discriminator",self.optimizer_discriminator)]:
                        qc=self._get_q_controller_for_optimizer(opt_n)
                        if qc:
                            loss_k='loss_g_total_phase3' if "sphere" in opt_n else 'loss_d_phase3'
                            current_loss_for_qc = step_raw_losses.get(loss_k, None) 
                            new_lr=qc.choose_action(current_loss_val=current_loss_for_qc)
                            if abs(new_lr - opt_inst.param_groups[0]['lr']) > 1e-9:
                                logger.info(f"QCtrl ({qc.logger_suffix}): Update LR for {opt_n} to {new_lr:.2e}")
                                for pg in opt_inst.param_groups: pg['lr']=new_lr
                
                self.optimizer_discriminator.zero_grad(set_to_none=True)
                self.optimizer_sphere_wubu_core.zero_grad(set_to_none=True)
                self.optimizer_sphere_mlps.zero_grad(set_to_none=True)
                self.global_step+=1
                if self.log_interval>0 and self.global_step%self.log_interval==0:
                    avg_bt=sum(batch_times)/len(batch_times) if batch_times else 0; batch_times=[]
                    avg_losses={f"train_p3/{k}":v_s/num_batches for k,v_s in epoch_losses_sum.items()}
                    logs={**avg_losses, "train_p3/lr_wubu":self.optimizer_sphere_wubu_core.param_groups[0]['lr'],
                          "train_p3/lr_mlp":self.optimizer_sphere_mlps.param_groups[0]['lr'],
                          "train_p3/lr_disc":self.optimizer_discriminator.param_groups[0]['lr'],
                          "train_p3/batch_ms":avg_bt*1000, "progress/gs":self.global_step,"progress/ep":self.current_epoch+1}
                    logger.info(f"Ep {self.current_epoch+1}|GS {self.global_step}|"+", ".join([f"{k.split('/')[-1]}:{v:.3f}" for k,v in avg_losses.items() if v is not None]))
                    if self.wandb_run: self.wandb_run.log(logs, step=self.global_step)
                if self.save_interval>0 and self.global_step%self.save_interval==0: self._save_checkpoint(False,"interval_p3")
        self.current_epoch+=1
        return {k:v/num_batches if num_batches>0 else 0.0 for k,v in epoch_losses_sum.items()}

    def validate_epoch(self) -> Dict[str, float]: # Adapted for Phase 3
        if not self.val_loader: return {"val_p3_no_loader": 0.0}
        self.etp_sphere_model.eval(); self.discriminator_model.eval()
        val_loss_agg = defaultdict(float); num_val_b = 0
        # Initialize all potential metrics to ensure they are in the dict for best_val_metric_name check
        val_loss_agg.update({'val_p3_loss_rec':0.0, 'val_p3_loss_vsp':0.0, 'val_p3_loss_vsp_B':0.0, 
                             'val_p3_disc_acc':0.0, 'val_p3_mmd':0.0, 'val_p3_sem_coh':0.0, 
                             'val_p3_lat_viz':0.0, 'val_p3_wubu_geom':0.0})

        with torch.no_grad():
            for batch_d_val in self.val_loader: # type: ignore
                emb_A=batch_d_val['source_A'].to(self.device,non_blocking=True); emb_B=batch_d_val['source_B'].to(self.device,non_blocking=True) # type: ignore
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    recon_A=self.etp_sphere_model(emb_A); lat_A=self.etp_sphere_model.get_latent(emb_A)
                    if self.lambda_rec>0: val_loss_agg['val_p3_loss_rec']+=calculate_reconstruction_loss(recon_A,emb_A).item()
                    if self.lambda_vsp>0: 
                        val_loss_agg['val_p3_loss_vsp']+=calculate_vector_space_preservation_loss(emb_A,lat_A).item()
                        # VSP for Corpus B latents as well
                        lat_B_vsp = self.etp_sphere_model.get_latent(emb_B)
                        val_loss_agg['val_p3_loss_vsp_B'] += calculate_vector_space_preservation_loss(emb_B, lat_B_vsp).item()

                    if self.lambda_ala>0:
                        lat_B_ala=self.etp_sphere_model.get_latent(emb_B) # Use lat_A from VSP calculation
                        d_A=self.discriminator_model(lat_A); d_B=self.discriminator_model(lat_B_ala)
                        preds_A=(torch.sigmoid(d_A)>0.5).float(); preds_B=(torch.sigmoid(d_B)<0.5).float()
                        val_loss_agg['val_p3_disc_acc']+=(preds_A.sum()+preds_B.sum())/(len(preds_A)+len(preds_B)+1e-8).item()
                num_val_b+=1
        avg_val_l={k:v/num_val_b if num_val_b>0 else 0.0 for k,v in val_loss_agg.items()}
        # Example combined metric for Phase 3, prioritizing VSP
        avg_val_l['val_p3_combined_loss'] = avg_val_l.get('val_p3_loss_vsp',0.0) + \
                                        avg_val_l.get('val_p3_loss_vsp_B',0.0) + \
                                        0.1 * avg_val_l.get('val_p3_loss_rec',0.0) - \
                                        0.1 * avg_val_l.get('val_p3_disc_acc',0.0) 

        if self.best_val_metric_name not in avg_val_l: # Ensure primary metric is present
            avg_val_l[self.best_val_metric_name] = avg_val_l.get('val_p3_loss_vsp', float('inf') if not self.best_val_metric_higher_is_better else float('-inf'))
        
        logger.info(f"Validation P3 Ep {self.current_epoch}: "+", ".join([f"{k}:{v:.3f}" for k,v in avg_val_l.items()]))
        if self.wandb_run:
            wandb_l={f"val_p3/{k.replace('val_p3_','')}":v for k,v in avg_val_l.items()}; wandb_l["progress/epoch"]=self.current_epoch
            self.wandb_run.log(wandb_l,step=self.global_step)
        return avg_val_l

    def _save_checkpoint(self, is_best: bool=False, reason: str="") -> None:
        name_parts=["ckpt_p3",reason] if reason else ["ckpt_p3"]; 
        if is_best:name_parts.append("best")
        name_parts.extend([f"ep{self.current_epoch}",f"gs{self.global_step}"]); fn="_".join(filter(None,name_parts))+".pth.tar"; fp=self.checkpoint_dir/fn
        state={'phase':3,'epoch':self.current_epoch,'global_step':self.global_step,
               'model_state':self.etp_sphere_model.state_dict(),'disc_state':self.discriminator_model.state_dict(),
               'opt_wubu_state':self.optimizer_sphere_wubu_core.state_dict(),'opt_mlp_state':self.optimizer_sphere_mlps.state_dict(),
               'opt_disc_state':self.optimizer_discriminator.state_dict(),
               'scaler_sphere_state':self.scaler_sphere.state_dict(),'scaler_disc_state':self.scaler_discriminator.state_dict(),
               'best_val_metric':self.best_val_metric,'best_val_metric_name':self.best_val_metric_name,
               'best_val_metric_higher_is_better':self.best_val_metric_higher_is_better}
        for qc_n,qc_i in self.q_controllers.items():
            if qc_i:state[f'q_ctrl_{qc_n}_state']=qc_i.state_dict()
        logger.info(f"Ckpt structure P3 to {fp} (CODING-ONLY: Not saved).")

    def load_checkpoint(self,path:str,load_optimizers:bool=True,load_q_controllers:bool=True)->None:
        fp=Path(path);logger.info(f"Conceptual P3 ckpt load: {fp} (CODING-ONLY: No file read).")
        ckpt:Dict[str,Any]={'epoch':0,'global_step':0,'best_val_metric':float('inf')}
        self.current_epoch=ckpt.get('epoch',0);self.global_step=ckpt.get('global_step',0)
        self.best_val_metric=ckpt.get('best_val_metric',self.best_val_metric)
        if load_q_controllers:
            for qc_n,qc_i in self.q_controllers.items():
                if qc_i and f'q_ctrl_{qc_n}_state' in ckpt:pass
        logger.info(f"P3 Ckpt conceptually loaded. Resume ep {self.current_epoch+1}, gs {self.global_step}.")

    def train(self,resume_from_checkpoint:Optional[str]=None)->None:
        if resume_from_checkpoint:self.load_checkpoint(resume_from_checkpoint)
        init_ep=self.current_epoch;logger.info(f"Start P3 training. Target ep:{self.epochs}. Done:{init_ep}.")
        for _ in range(init_ep,self.epochs):
            logger.info(f"Commencing P3 Epoch {self.current_epoch+1}/{self.epochs}")
            ep_losses=self.train_epoch()
            if self.q_controller_enabled:
                for opt_n,_ in self.q_controllers.items():
                    qc=self.q_controllers.get(opt_n)
                    if qc:qc.log_reward(-ep_losses.get('loss_g_total_phase3' if "sphere" in opt_n else 'loss_d_phase3',float('inf')))
            if self.val_loader and (self.current_epoch%self.val_interval_epochs==0 or self.current_epoch==self.epochs):
                val_mets=self.validate_epoch()
                curr_val_met=val_mets.get(self.best_val_metric_name, float('-inf') if self.best_val_metric_higher_is_better else float('inf'))
                is_better=(curr_val_met>self.best_val_metric) if self.best_val_metric_higher_is_better else (curr_val_met<self.best_val_metric)
                if is_better:self.best_val_metric=curr_val_met;logger.info(f"New best P3 val metric ({self.best_val_metric_name}):{self.best_val_metric:.4f}.");self._save_checkpoint(True,f"best_val_p3_{self.best_val_metric_name.replace('val_p3_','')}")
            if self.save_interval==0:self._save_checkpoint(False,"end_of_epoch_p3")
        logger.info(f"P3 Training completed after {self.current_epoch} epochs.")
        if self.wandb_run and hasattr(self.wandb_run,'finish'):self.wandb_run.finish()

if __name__ == '__main__':
    logger.info("ETPTrainerPhase3 class definition complete (CODING-ONLY CHECK).")
    pass
