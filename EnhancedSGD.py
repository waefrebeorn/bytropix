# EnhancedSGD.py

import torch
import math
from torch.optim import Optimizer
from collections import deque
import logging
from typing import List, Optional, Dict, Any, Iterable
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class QLearningController:
    """Optimized Q-Learning Controller with enhanced state and action spaces."""
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount: float = 0.95,  # Increased from 0.9 for longer-term rewards
        epsilon: float = 0.3,    # Reduced from 0.5 for more exploitation
        epsilon_decay: float = 0.999,  # Slower decay
        initial_mix_prob: float = 0.7,
        lr_scale_bounds: tuple = (0.1, 2.0),  # Wider bounds
        momentum_scale_bounds: tuple = (0.1, 0.999)  # More realistic bounds
    ):
        self.q_table = {}
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.mix_prob = initial_mix_prob
        self.prev_loss = None
        self.prev_state = None
        self.prev_action = None
        self.lr_scale_bounds = lr_scale_bounds
        self.momentum_scale_bounds = momentum_scale_bounds
        
        # Expanded action space with finer granularity
        self.action_ranges = {
            'lr_scale': np.array([0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]),
            'momentum_scale': np.array([0.95, 0.97, 0.99, 1.0, 1.01, 1.02, 1.03]),
            'entropy_scale': np.array([-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02])
        }
        
        # Initialize statistics tracking
        self.reward_history = deque(maxlen=1000)
        self.action_history = {k: deque(maxlen=1000) for k in self.action_ranges.keys()}

    def get_state(self, loss: float, grad_var: float, entropy: float, epoch: int) -> tuple:
        """Enhanced state discretization with better binning strategy."""
        # More granular bins for loss changes
        loss_bins = np.array([0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, float('inf')])
        
        # Gradient variance bins on log scale
        grad_var_bins = np.array([0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, float('inf')])
        
        # Entropy bins based on typical values
        entropy_bins = np.array([0, 0.5, 1.0, 2.0, 3.0, 4.0, float('inf')])
        
        # Add loss trend (increasing/decreasing)
        loss_trend = 0
        if self.prev_loss is not None:
            loss_trend = 1 if loss > self.prev_loss else -1
            
        # Safe handling of inputs
        if isinstance(grad_var, torch.Tensor):
            grad_var = grad_var.item() if grad_var.numel() == 1 else grad_var.mean().item()
            
        values = np.array([loss, grad_var, entropy])
        indices = np.array([np.searchsorted(b, v, side='right') - 1 for b, v in zip(
            [loss_bins, grad_var_bins, entropy_bins], 
            values
        )])
        
        # Include more state information
        epoch_bin = min(epoch // 5, 9)  # More fine-grained epoch bins
        return tuple(indices.tolist() + [epoch_bin, loss_trend])

    def choose_action(self, state: tuple) -> Dict[str, float]:
        """Enhanced action selection with adaptive exploration."""
        # Adaptive epsilon based on recent performance
        if len(self.reward_history) > 0:
            recent_rewards = np.mean(list(self.reward_history)[-10:])
            self.epsilon = max(0.05, min(0.3, self.epsilon * (1.0 - recent_rewards * 0.1)))
            
        if np.random.random() < self.epsilon:
            # Smart exploration: bias towards actions that worked well recently
            action = {}
            for param, space in self.action_ranges.items():
                if len(self.action_history[param]) > 0:
                    recent_actions = np.mean(list(self.action_history[param])[-5:])
                    # Bias exploration around recent successful actions
                    indices = np.argsort(np.abs(space - recent_actions))[:3]
                    action[param] = float(np.random.choice(space[indices]))
                else:
                    action[param] = float(np.random.choice(space))
        else:
            if state not in self.q_table:
                self.q_table[state] = {f"{param}_mult": 0.0 for param in self.action_ranges}
                
            action = {}
            q_values = self.q_table[state]
            
            for param in self.action_ranges:
                key = f"{param}_mult"
                best_action = max(self.action_ranges[param], 
                                key=lambda x: q_values.get(key, 0.0))
                action[param] = float(best_action)
                
        # Record actions for history
        for param, value in action.items():
            self.action_history[param].append(value)
            
        return action

    def update(self, state: tuple, action: Dict[str, float], reward: float, next_state: tuple):
        """Enhanced Q-value updates with eligibility traces and prioritized updates."""
        # Record reward
        self.reward_history.append(reward)
        
        # Initialize states if needed
        if state not in self.q_table:
            self.q_table[state] = {f"{param}_mult": 0.0 for param in self.action_ranges}
        if next_state not in self.q_table:
            self.q_table[next_state] = {f"{param}_mult": 0.0 for param in self.action_ranges}

        # Compute TD error for prioritized updates
        max_future_q = max(self.q_table[next_state].values(), default=0.0)
        
        # Update Q-values with eligibility traces
        trace_decay = 0.9
        for param, value in action.items():
            key = f"{param}_mult"
            old_q = self.q_table[state].get(key, 0.0)
            
            # TD error
            td_error = float(reward) + self.gamma * max_future_q - old_q
            
            # Adaptive learning rate based on TD error magnitude
            effective_lr = self.alpha * (1.0 + abs(td_error) * 0.1)
            
            # Update with eligibility trace
            self.q_table[state][key] = old_q + effective_lr * td_error
            
            # Propagate updates to previous states with decay
            if self.prev_state is not None:
                prev_q = self.q_table[self.prev_state].get(key, 0.0)
                self.q_table[self.prev_state][key] = prev_q + \
                    effective_lr * td_error * trace_decay

class EnhancedSGD(Optimizer):
    """Enhanced SGD Implementation with improved stability and adaptive hyperparameters."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 3e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        smoothing_factor: float = 0.03,
        entropy_threshold: float = 0.2,
        max_grad_norm: float = 0.5,
        noise_scale: float = 1e-5,
        lr_scale_bounds: tuple = (0.8, 1.2),
        momentum_scale_bounds: tuple = (0.8, 1.2),
        **kwargs
    ):
        # Ensure params is properly formatted as parameter groups
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            param_groups = params
        else:
            param_groups = [{'params': params}]
        
        # Add default values to all parameter groups
        for group in param_groups:
            group.setdefault('lr', lr)
            group.setdefault('momentum', momentum)
            group.setdefault('weight_decay', weight_decay)
        
        super().__init__(param_groups, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))
        
        # Initialize optimization state
        self._init_optimization_state(
            smoothing_factor=smoothing_factor,
            entropy_threshold=entropy_threshold,
            max_grad_norm=max_grad_norm,
            noise_scale=noise_scale,
            lr_scale_bounds=lr_scale_bounds,
            momentum_scale_bounds=momentum_scale_bounds
        )
        
        # Pre-allocate buffers with proper device handling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stats = {
            'grad_norms': deque(maxlen=100),
            'learning_rates': deque(maxlen=100),
            'momentum_values': deque(maxlen=100),
            'entropy_values': deque(maxlen=100),
            'update_norms': deque(maxlen=100)
        }

    def _init_optimization_state(self, **kwargs):
        """Initialize optimization state with safe handling."""
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        self.q_controller = QLearningController(
            learning_rate=0.1,
            discount=0.95,
            epsilon=0.3,
            epsilon_decay=0.999,
            initial_mix_prob=0.7,
            lr_scale_bounds=self.lr_scale_bounds,
            momentum_scale_bounds=self.momentum_scale_bounds
        )
        
        self._step_count = 0
        self.prev_state = None
        self.prev_action = None
        self.prev_loss = None

    def _compute_entropy(self, tensor: torch.Tensor) -> float:
        """Efficient entropy computation using torch operations with safe calculations."""
        if tensor.numel() <= 1:
            return 0.0  # Return default entropy for single-element tensors
            
        values = tensor.flatten()
        # Safe histogram calculation
        hist = torch.histc(values, bins=min(100, values.numel()))
        # Add small epsilon to prevent log(0)
        eps = 1e-7
        hist = hist / (hist.sum() + eps) + eps
        return float(-torch.sum(hist * torch.log(hist)).item())

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[torch.Tensor]:
        """Enhanced step function with better gradient handling and adaptation."""
        loss = None if closure is None else closure()
        
        # Track if we've seen any gradients this step
        saw_grads = False
        
        # Default to no gradient anomalies
        is_anomalous = False
        
        # Compute global gradient statistics once per step
        grad_norms = []
        grad_vars = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                grad_norms.append(torch.norm(grad).item())
                if grad.numel() > 1:
                    grad_vars.append(torch.var(grad.float()).item())
                    
        # Get learning rate adjustment from Q-learning if we have gradients and loss
        if grad_norms and loss is not None:
            mean_grad_norm = np.mean(grad_norms)
            mean_grad_var = np.mean(grad_vars) if grad_vars else 0.0
            
            # Update statistics tracking
            self.stats['grad_norms'].append(mean_grad_norm)
            
            # Detect gradient anomalies
            is_anomalous = mean_grad_norm > self.max_grad_norm * 10 or \
                        np.isnan(mean_grad_norm) or \
                        np.isnan(mean_grad_var)
            
            current_loss = loss.item()
            entropy = self._compute_entropy(torch.tensor(grad_vars)) if grad_vars else 0.0
            
            # Get Q-learning state and action ONCE per step
            q_state = self.q_controller.get_state(
                loss=current_loss,
                grad_var=mean_grad_var,
                entropy=entropy,
                epoch=self._step_count
            )
            
            q_action = self.q_controller.choose_action(q_state)
            
            # Apply bounded scaling with adaptive bounds
            lr_bounds = (
                max(1e-6, self.lr_scale_bounds[0]),
                min(10.0, self.lr_scale_bounds[1])  # Removed expanding bounds
            )
            
            # Scale factors with bounds
            lr_scale = float(np.clip(
                q_action['lr_scale'],
                lr_bounds[0],
                lr_bounds[1]
            ))
            
            # Compute reward based on loss improvement and gradient health
            if self.q_controller.prev_loss is not None:
                loss_improvement = self.q_controller.prev_loss - current_loss
                grad_health = 1.0 / (1.0 + mean_grad_var) if mean_grad_var > 0 else 1.0
                reward = loss_improvement * grad_health
                
                # Additional reward for stable updates
                if not is_anomalous and abs(loss_improvement) < self.q_controller.prev_loss * 0.1:
                    reward += 0.1
                    
                self.q_controller.update(
                    state=self.q_controller.prev_state,
                    action=self.q_controller.prev_action,
                    reward=reward,
                    next_state=q_state
                )
                
            self.q_controller.prev_state = q_state
            self.q_controller.prev_action = q_action
            self.q_controller.prev_loss = current_loss
            
            # Apply the same learning rate scale to ALL parameter groups
            for group in self.param_groups:
                old_lr = group['lr']
                # More aggressive learning rate adaptation
                group['lr'] = float(np.clip(
                    old_lr * lr_scale,  # Direct scaling without smoothing
                    1e-8,
                    10.0
                ))
        
        # Apply updates with the adjusted learning rates
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.to(p.device, non_blocking=True)
                if is_anomalous:
                    grad = torch.clamp(grad, -1.0, 1.0)
                    
                state = self.state[p]
                saw_grads = True
                
                # Initialize momentum buffer if needed
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                # Apply update with current learning rate
                self._apply_update(
                    p=p,
                    grad=grad,
                    momentum=group['momentum'],
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    state=state
                )
        
        if saw_grads:
            self._step_count += 1
            self._adjust_hyperparameters()
            
        return loss
        
    def _apply_update(
        self, 
        p: torch.Tensor, 
        grad: torch.Tensor, 
        momentum: float,
        lr: float, 
        weight_decay: float,
        state: dict
    ):
        """Enhanced parameter update with adaptive momentum and noise injection."""
        buf = state['momentum_buffer']
        
        # Apply weight decay with adaptive scaling
        if weight_decay != 0:
            # Scale weight decay based on parameter magnitude
            param_norm = torch.norm(p)
            adaptive_wd = weight_decay * (1.0 + torch.log1p(param_norm))
            grad = grad.add(p, alpha=adaptive_wd)
        
        # Update momentum buffer
        buf.mul_(momentum).add_(grad)
        
        # Inject noise for escape velocity
        noise = torch.randn_like(buf) * self.noise_scale * (1.0 / (1.0 + self._step_count * 0.001))
        buf.add_(noise)
        
        # Compute the parameter update
        update = -lr * buf
        
        # Track update for oscillation detection
        if 'update_history' not in state:
            state['update_history'] = deque(maxlen=5)
        state['update_history'].append(update.clone())
        
        # Apply gradient clipping on parameter update
        max_update_norm = 1.0  # Prevent too large updates
        update_norm = torch.norm(update)
        if update_norm > max_update_norm:
            update.mul_(max_update_norm / (update_norm + 1e-6))
            self.stats['update_norms'].append(float(update_norm.item()))
        
        # Update parameters with smoothing for stability
        smooth_factor = min(1.0, self._step_count / 1000)  # Gradual increase in update magnitude
        p.data.add_(update * smooth_factor)
        
    def _adjust_hyperparameters(self) -> None:
        """Adaptively adjust optimizer hyperparameters based on training statistics."""
        if not hasattr(self, 'stats') or len(self.stats.get('grad_norms', [])) < 50:
            return

        # Calculate recent statistics
        recent_grad_norms = list(self.stats['grad_norms'])[-50:]
        recent_update_norms = list(self.stats.get('update_norms', []))[-50:]
        
        if not recent_update_norms:
            return

        grad_norm_mean = np.mean(recent_grad_norms)
        grad_norm_std = np.std(recent_grad_norms)
        update_norm_mean = np.mean(recent_update_norms)
        
        # Detect training instability
        is_unstable = grad_norm_std / (grad_norm_mean + 1e-8) > 0.5
        is_diverging = grad_norm_mean > 10.0 * self.max_grad_norm
        updates_too_small = update_norm_mean < 1e-7

        for group in self.param_groups:
            current_lr = group['lr']
            current_momentum = group['momentum']
            
            if is_unstable:
                # Reduce learning rate and increase momentum for stability
                group['lr'] = max(1e-8, current_lr * 0.9)
                group['momentum'] = min(0.99, current_momentum * 1.02)
                logging.info(f"Adjusted lr to {group['lr']:.6f} and momentum to {group['momentum']:.4f} due to instability.")
                
            elif is_diverging:
                # Significantly reduce learning rate
                group['lr'] = max(1e-8, current_lr * 0.7)
                group['momentum'] = max(0.1, current_momentum * 0.9)
                logging.info(f"Adjusted lr to {group['lr']:.6f} and momentum to {group['momentum']:.4f} due to divergence.")
                
            elif updates_too_small:
                # Carefully increase learning rate
                group['lr'] = min(10.0, current_lr * 1.05)
                logging.info(f"Increased lr to {group['lr']:.6f} due to small updates.")
                
            # Adjust weight decay based on gradient magnitude
            if grad_norm_mean > 1.0:
                group['weight_decay'] = min(0.1, group.get('weight_decay', 0.01) * 1.02)
            else:
                group['weight_decay'] = max(0.001, group.get('weight_decay', 0.01) * 0.98)
                
        # Log adjustments
        if hasattr(self, 'stats'):
            self.stats['learning_rates'].append(self.param_groups[0]['lr'])
            self.stats['momentum_values'].append(self.param_groups[0]['momentum'])
   
    def get_statistics(self) -> Dict[str, float]:
        """Compute statistics using pre-allocated tensors with safe calculations."""
        stats = {}
        for key, values in self.stats.items():
            if values:
                try:
                    tensor_values = torch.tensor(list(values), dtype=torch.float32)
                    stats[f'avg_{key}'] = float(torch.mean(tensor_values).item())
                    stats[f'std_{key}'] = float(torch.std(tensor_values).item())
                except (RuntimeError, ValueError) as e:
                    logging.warning(f"Error computing statistics for {key}: {e}")
                    stats[f'avg_{key}'] = 0.0
                    stats[f'std_{key}'] = 0.0
        return stats

    def state_dict(self) -> Dict[str, Any]:
        """Returns the optimizer's state dict with safe serialization."""
        state_dict = super().state_dict()
        try:
            state_dict['statistics'] = self.get_statistics()
            state_dict['q_table'] = self.q_controller.q_table
            state_dict['epsilon'] = float(self.q_controller.epsilon)
        except Exception as e:
            logging.error(f"Error creating state dict: {e}")
            state_dict['statistics'] = {}
            state_dict['q_table'] = {}
            state_dict['epsilon'] = 0.3
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads optimizer state with safe type handling."""
        try:
            statistics = state_dict.pop('statistics', None)
            q_table = state_dict.pop('q_table', None)
            epsilon = state_dict.pop('epsilon', None)
            
            super().load_state_dict(state_dict)
            
            if statistics is not None:
                for key in self.stats:
                    avg_key = f'avg_{key}'
                    if avg_key in statistics:
                        # Fill the deque with the average value for simplicity
                        self.stats[key].extend([float(statistics[avg_key])] * self.stats[key].maxlen)
                logging.info("Loaded optimizer statistics")
                
            if q_table is not None:
                self.q_controller.q_table = q_table
                logging.info("Loaded Q-Learning controller Q-table")
                
            if epsilon is not None:
                self.q_controller.epsilon = float(epsilon)
                logging.info("Loaded Q-Learning controller epsilon")
                
        except Exception as e:
            logging.error(f"Error loading state dict: {e}")
            # Initialize fresh statistics and Q-table if loading fails
            self.stats = {
                'grad_norms': deque(maxlen=100),
                'learning_rates': deque(maxlen=100),
                'momentum_values': deque(maxlen=100),
                'entropy_values': deque(maxlen=100),
                'update_norms': deque(maxlen=100)
            }
            self.q_controller.q_table = {}
            self.q_controller.epsilon = 0.3
            logging.warning("Initialized fresh optimization state due to loading error")

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Safely zeros gradients with optional memory efficiency."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()
                        
    def add_param_group(self, param_group: dict) -> None:
        """
        Add a param group to the optimizer's param groups safely.
        
        Args:
            param_group (dict): The param group to add
        """
        try:
            # Validate param group
            required_keys = {'params', 'lr', 'momentum', 'weight_decay'}
            if not all(key in param_group for key in required_keys):
                raise ValueError(f"Param group must contain all of: {required_keys}")
                
            # Ensure params is iterable
            if isinstance(param_group['params'], torch.nn.Parameter):
                param_group['params'] = [param_group['params']]
                
            # Validate numerical parameters
            for key in ['lr', 'momentum', 'weight_decay']:
                if not isinstance(param_group[key], (int, float)):
                    raise TypeError(f"{key} must be a number")
                param_group[key] = float(param_group[key])
            
            super().add_param_group(param_group)
            logging.info(f"Added parameter group with {len(param_group['params'])} parameters")
            
        except Exception as e:
            logging.error(f"Error adding parameter group: {e}")
            raise
