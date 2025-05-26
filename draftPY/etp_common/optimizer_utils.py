import logging
import math
import random
from typing import Dict, Optional, List, Any, Tuple, Deque 
from collections import deque 

import torch
import torch.optim as optim 

logger = logging.getLogger(__name__)

# --- Default Q-Learning Config ---
DEFAULT_CONFIG_QLEARN_HYBRID: Dict[str, Any] = {
    "q_table_size": 10, 
    "num_lr_actions": 5,
    "lr_change_factors": [0.5, 0.9, 1.0, 1.1, 1.5], 
    "learning_rate_q": 0.1,
    "discount_factor_q": 0.9,
    "exploration_rate_q": 0.1, 
    "lr_min": 1e-7, 
    "lr_max": 1e-1,
    "metric_history_len": 5,
    "loss_min": 0.0, "loss_max": 10.0,
    "grad_stats_window": 20,
}


# --- Gradient Statistics Tracker ---
class GradientStats:
    """Tracks basic statistics of gradients."""
    def __init__(self, window_size: int = 100, device: Optional[torch.device] = None):
        self.window_size = window_size
        self.device = device if device else torch.device("cpu")
        self.grad_norms: Deque[float] = deque(maxlen=window_size)
        self.grad_means: Deque[float] = deque(maxlen=window_size)
        self.grad_stds: Deque[float] = deque(maxlen=window_size)
        self.update_magnitudes: Deque[float] = deque(maxlen=window_size)
        self.ewma_norm = 0.0; self.ewma_mean = 0.0; self.ewma_std = 0.0
        self.alpha = 0.1 

    @torch.no_grad()
    def update(self, params_with_grad: List[torch.nn.Parameter], current_lr: Optional[float] = None) -> None:
        if not params_with_grad: return
        valid_grads = [p.grad.data.view(-1) for p in params_with_grad if p.grad is not None]
        if not valid_grads: return
        all_grads_flat = torch.cat(valid_grads)
        if all_grads_flat.numel() == 0: return
        all_grads_flat = all_grads_flat.to(self.device)

        current_grad_norm = torch.norm(all_grads_flat, p=2).item()
        self.grad_norms.append(current_grad_norm)
        self.ewma_norm = self.alpha * current_grad_norm + (1 - self.alpha) * self.ewma_norm
        current_grad_mean = all_grads_flat.mean().item(); current_grad_std = all_grads_flat.std().item()
        self.grad_means.append(current_grad_mean); self.grad_stds.append(current_grad_std)
        self.ewma_mean = self.alpha * current_grad_mean + (1 - self.alpha) * self.ewma_mean
        self.ewma_std = self.alpha * current_grad_std + (1 - self.alpha) * self.ewma_std
        if current_lr is not None: self.update_magnitudes.append(current_lr * current_grad_norm)

    def get_stats(self) -> Dict[str, float]:
        return {
            "grad_norm_current": self.grad_norms[-1] if self.grad_norms else 0.0,
            "grad_norm_ewma": self.ewma_norm, "grad_mean_current": self.grad_means[-1] if self.grad_means else 0.0,
            "grad_mean_ewma": self.ewma_mean, "grad_std_current": self.grad_stds[-1] if self.grad_stds else 0.0,
            "grad_std_ewma": self.ewma_std,
            "update_magnitude_current": self.update_magnitudes[-1] if self.update_magnitudes else 0.0,
        }
    def state_dict(self) -> Dict[str, Any]:
        return {
            "grad_norms": list(self.grad_norms), "grad_means": list(self.grad_means),
            "grad_stds": list(self.grad_stds), "update_magnitudes": list(self.update_magnitudes),
            "ewma_norm": self.ewma_norm, "ewma_mean": self.ewma_mean, "ewma_std": self.ewma_std,
            "window_size": self.window_size, "alpha": self.alpha
        }
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.window_size = state_dict.get("window_size", self.window_size)
        self.grad_norms=deque(state_dict.get("grad_norms",[]),maxlen=self.window_size)
        self.grad_means=deque(state_dict.get("grad_means",[]),maxlen=self.window_size)
        self.grad_stds=deque(state_dict.get("grad_stds",[]),maxlen=self.window_size)
        self.update_magnitudes=deque(state_dict.get("update_magnitudes",[]),maxlen=self.window_size)
        self.ewma_norm=state_dict.get("ewma_norm",0.0); self.ewma_mean=state_dict.get("ewma_mean",0.0); self.ewma_std=state_dict.get("ewma_std",0.0)
        self.alpha=state_dict.get("alpha",self.alpha)

# --- HAKMEM Q-Controller ---
class HAKMEMQController:
    def __init__(self, initial_lr: float, config: Optional[Dict[str, Any]] = None, logger_suffix: str = ""):
        self.config = config if config is not None else DEFAULT_CONFIG_QLEARN_HYBRID.copy()
        self.initial_lr = initial_lr; self.current_lr = initial_lr; self.logger_suffix = logger_suffix
        self.q_table_size = int(self.config.get("q_table_size",10)); self.q_table_size = max(1, self.q_table_size)
        self.num_actions = int(self.config.get("num_lr_actions",5))
        self.lr_change_factors = self.config.get("lr_change_factors",[0.5,0.9,1.0,1.1,1.5])
        if self.num_actions != len(self.lr_change_factors): self.num_actions = len(self.lr_change_factors)
        self.q_table = torch.zeros((self.q_table_size, self.num_actions))
        self.learning_rate_q=float(self.config.get("learning_rate_q",0.1)); self.discount_factor_q=float(self.config.get("discount_factor_q",0.9))
        self.exploration_rate_q=float(self.config.get("exploration_rate_q",0.1))
        self.lr_min=float(self.config.get("lr_min",1e-7)); self.lr_max=float(self.config.get("lr_max",1e-1))
        self.loss_history:Deque[float]=deque(maxlen=int(self.config.get("metric_history_len",5)))
        self.loss_min=float(self.config.get("loss_min",0.0)); self.loss_max=float(self.config.get("loss_max",10.0))
        if self.loss_min >= self.loss_max: self.loss_max = self.loss_min + 10.0
        self.grad_stats=GradientStats(window_size=int(self.config.get("grad_stats_window",20)))
        self.last_action_idx:Optional[int]=None; self.last_state_idx:Optional[int]=None
        logger.info(f"HAKMEMQController ({self.logger_suffix}) initialized. LR: {self.initial_lr:.2e}, Q-Table: {self.q_table.shape}")

    def _discretize_value(self,value:float,min_val:float,max_val:float,num_bins:int)->int:
        if num_bins<=0: return 0
        if value<=min_val: return 0
        if value>=max_val: return num_bins-1
        bin_size=(max_val-min_val)/num_bins
        if bin_size<=0: return num_bins//2
        return min(int((value-min_val)/bin_size),num_bins-1)

    def _get_current_state_idx(self,current_loss_val:Optional[float])->int:
        if current_loss_val is not None: return self._discretize_value(current_loss_val,self.loss_min,self.loss_max,self.q_table_size)
        return self.q_table_size//2

    def choose_action(self,current_loss_val:Optional[float]=None,params_with_grad:Optional[List[torch.nn.Parameter]]=None)->float:
        if params_with_grad: self.grad_stats.update(params_with_grad,self.current_lr)
        self.last_state_idx=self._get_current_state_idx(current_loss_val)
        if random.random()<self.exploration_rate_q: self.last_action_idx=random.randint(0,self.num_actions-1)
        else:
            with torch.no_grad():self.last_action_idx=torch.argmax(self.q_table[self.last_state_idx]).item()
        self.current_lr=max(self.lr_min,min(self.current_lr*self.lr_change_factors[self.last_action_idx],self.lr_max))
        return self.current_lr

    def log_reward(self,reward:float,current_loss_val:Optional[float]=None)->None:
        if self.last_state_idx is None or self.last_action_idx is None: return
        current_q=self.q_table[self.last_state_idx,self.last_action_idx]
        next_state_idx=self._get_current_state_idx(current_loss_val)
        with torch.no_grad():max_next_q=torch.max(self.q_table[next_state_idx]).item()
        new_q=current_q+self.learning_rate_q*(reward+self.discount_factor_q*max_next_q-current_q)
        self.q_table[self.last_state_idx,self.last_action_idx]=new_q
        if current_loss_val is not None: self.loss_history.append(current_loss_val)

    def get_current_lr(self)->float: return self.current_lr
    def state_dict(self)->Dict[str,Any]:
        return {"current_lr":self.current_lr,"q_table":self.q_table.tolist(),"loss_history":list(self.loss_history),
                "last_action_idx":self.last_action_idx,"last_state_idx":self.last_state_idx,"initial_lr":self.initial_lr,
                "config":self.config,"grad_stats_state_dict":self.grad_stats.state_dict()}
    def load_state_dict(self,state_dict:Dict[str,Any])->None:
        self.current_lr=state_dict.get("current_lr",self.initial_lr)
        if "q_table" in state_dict:
            loaded_q=torch.tensor(state_dict["q_table"])
            if loaded_q.shape==self.q_table.shape: self.q_table=loaded_q
            else: logger.warning(f"HAKMEM ({self.logger_suffix}): Q-table shape mismatch. Not loading.")
        self.loss_history=deque(state_dict.get("loss_history",[]),maxlen=self.config.get("metric_history_len",5))
        self.last_action_idx=state_dict.get("last_action_idx"); self.last_state_idx=state_dict.get("last_state_idx")
        if "grad_stats_state_dict" in state_dict: self.grad_stats.load_state_dict(state_dict["grad_stats_state_dict"])

# --- Riemannian Enhanced SGD ---
class RiemannianEnhancedSGD(optim.Optimizer):
    def __init__(self, params: Any, lr: float = 1e-3, 
                 q_learning_config: Optional[Dict[str,Any]] = None,
                 q_logger_suffix: str = "", **kwargs: Any):
        param_list = list(params) if not isinstance(params, list) else params
        if not param_list: param_list = [torch.nn.Parameter(torch.zeros(1))]
        defaults = dict(lr=lr, **kwargs)
        super().__init__(param_list, defaults)
        self.q_controller: Optional[HAKMEMQController] = None
        if q_learning_config:
            self.q_controller = HAKMEMQController(initial_lr=lr, config=q_learning_config, logger_suffix=f"RESGD_{q_logger_suffix}")

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]: # type: ignore[override]
        loss: Optional[torch.Tensor] = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()

        params_with_grad_for_q = [p for group in self.param_groups for p in group['params'] if p.grad is not None]
        if self.q_controller:
            current_loss_val = loss.item() if loss is not None else None
            new_lr = self.q_controller.choose_action(current_loss_val=current_loss_val, params_with_grad=params_with_grad_for_q)
            if abs(new_lr - self.param_groups[0]['lr']) > 1e-9 : # Check for actual change
                 logger.info(f"RESGD ({self.q_controller.logger_suffix}) LR updated by Q-Ctrl to: {new_lr:.2e}")
                 for group in self.param_groups: group['lr'] = new_lr
        
        for group in self.param_groups:
            lr_group = group['lr']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse: raise RuntimeError('RiemannianSGD does not support sparse gradients for now.')

                if hasattr(p, 'manifold') and p.manifold is not None:
                    manifold = p.manifold
                    p_on_manifold = p.data # Assume param is already on manifold or proj is part of manifold methods
                    if hasattr(manifold, 'proj') and callable(manifold.proj):
                        p_on_manifold = manifold.proj(p.data) # Ensure point is on manifold

                    riemannian_grad: torch.Tensor
                    if hasattr(manifold, 'egrad2rgrad') and callable(manifold.egrad2rgrad):
                        # Use manifold's method to convert Euclidean grad to Riemannian grad
                        # Pass curvature 'c' if available in the group, specific to this manifold context
                        curvature_val = group.get('c', getattr(manifold,'c',1.0)) # Get c from group or manifold default
                        riemannian_grad = manifold.egrad2rgrad(p_on_manifold, grad, curvature_val)
                    else:
                        logger.warning(f"Parameter (shape: {p.shape}) has a manifold but no 'egrad2rgrad' method. Using Euclidean gradient as Riemannian.")
                        riemannian_grad = grad
                    
                    update_vec = -lr_group * riemannian_grad

                    new_p_val: torch.Tensor
                    if hasattr(manifold, 'expmap') and callable(manifold.expmap):
                        # Use manifold's expmap for retraction
                        curvature_val = group.get('c', getattr(manifold,'c',1.0))
                        new_p_val = manifold.expmap(p_on_manifold, update_vec, curvature_val)
                    else:
                        logger.warning(f"Parameter (shape: {p.shape}) manifold has no 'expmap'. Using Euclidean update for retraction.")
                        new_p_val = p_on_manifold + update_vec
                    
                    if hasattr(manifold, 'proj') and callable(manifold.proj):
                        p.data.copy_(manifold.proj(new_p_val))
                    else:
                        p.data.copy_(new_p_val)
                else:
                    # Standard Euclidean SGD update
                    p.data.add_(grad, alpha=-lr_group)
        return loss

    def get_q_controller(self) -> Optional[HAKMEMQController]: return self.q_controller
    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        if self.q_controller: state_dict['q_controller'] = self.q_controller.state_dict()
        return state_dict
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        if self.q_controller and 'q_controller' in state_dict:
            self.q_controller.load_state_dict(state_dict['q_controller'])
