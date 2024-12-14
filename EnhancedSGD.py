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
    """Optimized Q-Learning Controller with vectorized operations."""
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount: float = 0.9,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.998,
        initial_mix_prob: float = 0.7,
        lr_scale_bounds: tuple = (0.8, 1.2),
        momentum_scale_bounds: tuple = (0.8, 1.2)
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
        
        # Pre-compute action space as numpy array for faster sampling
        self.action_ranges = {
            'lr_scale': np.array([0.95, 1.0, 1.05]),
            'momentum_scale': np.array([0.95, 1.0, 1.05]),
            'entropy_scale': np.array([-0.01, 0.0, 0.01])
        }

    def get_state(self, loss: float, grad_var: float, entropy: float, epoch: int) -> tuple:
        """Optimized state discretization using numpy."""
        # Pre-defined bins
        bins = np.array([
            [0, 0.5, 1.0, 1.5, 2.0, float('inf')],  # loss bins
            [0, 0.1, 0.2, 0.3, 0.4, float('inf')],  # grad_var bins
            [0, 0.5, 1.0, 1.5, 2.0, float('inf')]   # entropy bins
        ])
        
        values = np.array([loss, grad_var, entropy])
        indices = np.searchsorted(bins, values.reshape(-1,1), side='right') - 1
        epoch_bin = min(epoch // 10, 5)
        
        return tuple(np.append(indices.flatten(), epoch_bin))

    def choose_action(self, state: tuple) -> Dict[str, float]:
        """Vectorized action selection with numpy."""
        if np.random.random() < self.epsilon:
            return {
                param: float(np.random.choice(space))
                for param, space in self.action_ranges.items()
            }
            
        if state not in self.q_table:
            self.q_table[state] = {f"{param}_mult": 0.0 for param in self.action_ranges}
            
        action = {}
        q_values = self.q_table[state]
        
        for param in self.action_ranges:
            mult_q = q_values.get(f"{param}_mult", 0.0)
            # Select the action with the highest Q-value
            best_idx = np.argmax([mult_q] + [0.0])  # Placeholder for future actions
            action[param] = self.action_ranges[param][best_idx]

        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        return action

    def update(self, state: tuple, action: Dict[str, float], reward: float, next_state: tuple):
        """Vectorized Q-value updates."""
        if state not in self.q_table:
            self.q_table[state] = {f"{param}_mult": 0.0 for param in self.action_ranges}
            
        if next_state not in self.q_table:
            self.q_table[next_state] = {f"{param}_mult": 0.0 for param in self.action_ranges}

        # Vectorized max calculation
        max_future_q = max(self.q_table[next_state].values())
        
        # Vectorized updates
        for param, value in action.items():
            key = f"{param}_mult"
            old_q = self.q_table[state].get(key, 0.0)
            self.q_table[state][key] = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
            logging.debug(f"Updated Q-value for state {state}, action {key}: {self.q_table[state][key]}")

class EnhancedSGD(Optimizer):
    """Optimized Enhanced SGD Implementation"""
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 3e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        **kwargs
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        # Initialize all parameters at once
        self._init_optimization_state(
            smoothing_factor=kwargs.get('smoothing_factor', 0.03),
            entropy_threshold=kwargs.get('entropy_threshold', 0.2),
            max_grad_norm=kwargs.get('max_grad_norm', 0.5),
            noise_scale=kwargs.get('noise_scale', 1e-5),
            lr_scale_bounds=kwargs.get('lr_scale_bounds', (0.8, 1.2)),
            momentum_scale_bounds=kwargs.get('momentum_scale_bounds', (0.8, 1.2))
        )

        # Pre-allocate buffers (ensure device compatibility)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.stats = {
            'grad_norms': deque(maxlen=100),
            'learning_rates': deque(maxlen=100),
            'momentum_values': deque(maxlen=100),
            'entropy_values': deque(maxlen=100)
        }
        
        self.stats_tensors = {
            'grad_norm_buffer': torch.zeros(100, device=device),
            'lr_buffer': torch.zeros(100, device=device),
            'momentum_buffer': torch.zeros(100, device=device), 
            'entropy_buffer': torch.zeros(100, device=device)
        }

    def _init_optimization_state(self, **kwargs):
        """Initialize optimization state with vectorized operations."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        self.q_controller = QLearningController(
            learning_rate=0.1,
            discount=0.9,
            epsilon=0.5,
            epsilon_decay=0.998,
            initial_mix_prob=0.7,
            lr_scale_bounds=self.lr_scale_bounds,
            momentum_scale_bounds=self.momentum_scale_bounds
        )
        
        self._step_count = 0
        self._has_unscaled = False

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[torch.Tensor]:
        """Optimized step function with proper gradient scaling."""
        self._step_count += 1
        loss = None if closure is None else closure()

        # Process all parameter groups at once
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Calculate gradient statistics
                grad_norm = torch.norm(grad).item()
                grad_entropy = self._compute_entropy(grad)
                
                # Update statistics vectors
                self.stats['grad_norms'].append(grad_norm)
                self.stats['entropy_values'].append(grad_entropy)

                # Update Q-Learning controller if we have loss
                if loss is not None:
                    current_loss = loss.item()
                    q_state = self.q_controller.get_state(
                        loss=current_loss,
                        grad_var=torch.var(grad).item(),
                        entropy=grad_entropy,
                        epoch=self._step_count
                    )

                    q_action = self.q_controller.choose_action(q_state)
                    
                    # Apply bounds with vectorized operations
                    lr_scale = np.clip(
                        q_action['lr_scale'],
                        self.lr_scale_bounds[0],
                        self.lr_scale_bounds[1]
                    )
                    
                    momentum_scale = np.clip(
                        q_action['momentum_scale'], 
                        self.momentum_scale_bounds[0],
                        self.momentum_scale_bounds[1]
                    )

                    # Update parameters with vectorized operations
                    original_lr = group['lr']
                    original_momentum = group['momentum']
                    lr = np.clip(original_lr * lr_scale, 1e-6, 10.0)
                    momentum = np.clip(original_momentum * momentum_scale, 0.0, 0.999)
                    
                    group['lr'] = lr
                    group['momentum'] = momentum

                    # Log the adjustments
                    logging.info(f"Step {self._step_count}: Adjusted lr from {original_lr:.6f} to {group['lr']:.6f}")
                    logging.info(f"Step {self._step_count}: Adjusted momentum from {original_momentum:.6f} to {group['momentum']:.6f}")

                    # Apply gradient updates efficiently
                    self._apply_gradient_updates(
                        p=p,
                        grad=grad,
                        momentum=momentum,
                        lr=lr,
                        weight_decay=group['weight_decay'],
                        state=state
                    )

                    # Compute reward and update Q-table
                    if self.q_controller.prev_loss is not None:
                        reward = 1.0 if current_loss < self.q_controller.prev_loss else -1.0
                        self.q_controller.update(
                            state=self.q_controller.prev_state,
                            action=self.q_controller.prev_action,
                            reward=reward,
                            next_state=q_state
                        )

                    self.q_controller.prev_state = q_state
                    self.q_controller.prev_action = q_action
                    self.q_controller.prev_loss = current_loss

        return loss

    def _compute_entropy(self, tensor: torch.Tensor) -> float:
        """Efficient entropy computation using torch operations."""
        values = tensor.flatten()
        hist = torch.histc(values, bins=100)
        hist = hist / (hist.sum() + 1e-7) + 1e-7
        return float(-torch.sum(hist * torch.log(hist)))

    def _apply_gradient_updates(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        momentum: float,
        lr: float,
        weight_decay: float,
        state: dict
    ) -> None:
        """Optimized gradient update implementation."""
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(p)
        
        momentum_buffer = state['momentum_buffer']

        # Apply operations in-place when possible
        if weight_decay != 0:
            grad.add_(p, alpha=weight_decay)

        if self.gradient_centering:
            grad.sub_(grad.mean())

        if self.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(grad, self.max_grad_norm)

        # Update momentum buffer and parameters
        momentum_buffer.mul_(momentum).add_(grad)
        p.data.add_(momentum_buffer, alpha=-lr)

    def get_statistics(self) -> Dict[str, float]:
        """Compute statistics using pre-allocated tensors."""
        stats = {}
        for key, values in self.stats.items():
            if values:
                tensor_values = torch.tensor(list(values))
                stats[f'avg_{key}'] = float(torch.mean(tensor_values))
                stats[f'std_{key}'] = float(torch.std(tensor_values))
        return stats

    def state_dict(self) -> Dict[str, Any]:
        """Returns the optimizer's state dict with additional statistics and Q-table."""
        state_dict = super().state_dict()
        state_dict['statistics'] = self.get_statistics()
        state_dict['q_table'] = self.q_controller.q_table
        state_dict['epsilon'] = self.q_controller.epsilon
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads optimizer state with statistics and Q-table handling."""
        statistics = state_dict.pop('statistics', None)
        q_table = state_dict.pop('q_table', None)
        epsilon = state_dict.pop('epsilon', None)
        super().load_state_dict(state_dict)
        if statistics is not None:
            # Optionally, load statistics if needed
            self.stats['grad_norms'] = deque(statistics.get('avg_grad_norms', []), maxlen=100)
            self.stats['learning_rates'] = deque(statistics.get('avg_learning_rates', []), maxlen=100)
            self.stats['momentum_values'] = deque(statistics.get('avg_momentum_values', []), maxlen=100)
            self.stats['entropy_values'] = deque(statistics.get('avg_entropy_values', []), maxlen=100)
            logging.info("Loaded optimizer statistics")
        if q_table is not None:
            self.q_controller.q_table = q_table
            logging.info("Loaded Q-Learning controller Q-table")
        if epsilon is not None:
            self.q_controller.epsilon = epsilon
            logging.info("Loaded Q-Learning controller epsilon")
