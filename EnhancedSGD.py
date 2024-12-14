import torch
import math
import numpy as np
from torch.optim import Optimizer
from collections import deque
import logging
from typing import List, Optional, Union, Dict, Any, Iterable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class QLearningController:
    """
    Q-Learning Controller with adaptive mixed action space and entropy-based patching.
    """
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount: float = 0.9,
        epsilon: float = 0.5,
        epsilon_decay: float = 0.998,
        initial_mix_prob: float = 0.7
    ):
        self.q_table: Dict[tuple, Dict[str, float]] = {}
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.mix_prob = initial_mix_prob
        self.prev_loss = None

        # Initialize learning parameters
        self.params = {
            'lr_scale': 1.0,
            'momentum_scale': 1.0,
            'entropy_scale': 1.0
        }

    def get_state(self, loss: float, grad_var: float, entropy: float, epoch: int) -> tuple:
        """Creates a discrete state representation from continuous values."""
        return (
            round(float(loss), 3),
            round(float(grad_var), 3),
            round(float(entropy), 3),
            int(epoch)
        )

    def choose_action(self, state: tuple) -> Dict[str, float]:
        """Selects action using epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            action = self._random_action()
        else:
            action = self._best_action(state)
        
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        return action

    def _random_action(self) -> Dict[str, float]:
        """Generates random action with mixed multiplicative/additive adjustments."""
        action = {}
        for param in self.params:
            if np.random.random() < self.mix_prob:
                # Multiplicative adjustment
                action[param] = np.random.choice([0.95, 1.0, 1.05])
            else:
                # Additive adjustment
                action[param] = np.random.choice([-0.01, 0.0, 0.01])
        return action

    def _best_action(self, state: tuple) -> Dict[str, float]:
        """Selects best action based on Q-values."""
        if state not in self.q_table:
            self.q_table[state] = {
                f"{param}_{type_}": 0.0 
                for param in self.params 
                for type_ in ['mult', 'add']
            }

        action = {}
        for param in self.params:
            mult_val = self.q_table[state].get(f"{param}_mult", 0.0)
            add_val = self.q_table[state].get(f"{param}_add", 0.0)
            
            if mult_val >= add_val:
                action[param] = 1.0 + mult_val
            else:
                action[param] = add_val
        return action

    def update(self, state: tuple, action: Dict[str, float], reward: float, next_state: tuple):
        """Updates Q-values using the Q-learning update rule."""
        if state not in self.q_table:
            self.q_table[state] = {
                f"{param}_{type_}": 0.0 
                for param in self.params 
                for type_ in ['mult', 'add']
            }

        max_future_q = max(self.q_table.get(next_state, {'default': 0.0}).values())

        for param, value in action.items():
            action_type = 'mult' if abs(value - 1.0) < 0.1 else 'add'
            key = f"{param}_{action_type}"
            old_q = self.q_table[state][key]
            new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
            self.q_table[state][key] = new_q


class EnhancedSGD(Optimizer):
    """
    Enhanced SGD optimizer with dynamic patching, entropy-based compute allocation,
    and adaptive parameter adjustments via Q-Learning.
    """
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        momentum: float = 0.9,
        smoothing_factor: float = 0.1,
        entropy_threshold: float = 0.3,
        patch_size: int = 6,
        weight_decay: float = 0.01,
        apply_noise: bool = True,
        adaptive_momentum: bool = True,
        gradient_centering: bool = True,
        gradient_clipping: bool = True,
        noise_scale: float = 1e-4,
        max_grad_norm: float = 1.0
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

        # Core hyperparameters
        self.smoothing_factor = smoothing_factor
        self.entropy_threshold = entropy_threshold
        self.target_patch_size = patch_size
        self.max_grad_norm = max_grad_norm
        self.noise_scale = noise_scale

        # Feature flags
        self.apply_noise = apply_noise
        self.adaptive_momentum = adaptive_momentum
        self.gradient_centering = gradient_centering
        self.gradient_clipping = gradient_clipping

        # Initialize state tracking
        self.state['step'] = 0
        self.state['grad_variance'] = 0.0
        self.state['entropy_history'] = deque(maxlen=100)
        self.state['patch_boundaries'] = []
        self.state['loss_history'] = deque(maxlen=100)

        # Initialize Q-Learning controller
        self.q_controller = QLearningController()

        # Statistics tracking
        self._init_statistics()

    def _init_statistics(self):
        """Initializes tracking statistics."""
        self.stats = {
            'grad_norms': deque(maxlen=100),
            'learning_rates': deque(maxlen=100),
            'momentum_values': deque(maxlen=100),
            'patch_sizes': deque(maxlen=100),
            'entropy_values': deque(maxlen=100)
        }

    def _compute_entropy(self, tensor: torch.Tensor) -> float:
        """Computes entropy of a tensor's distribution."""
        values = tensor.detach().cpu().numpy().flatten()
        hist, _ = np.histogram(values, bins='auto', density=True)
        hist = hist + 1e-7  # Avoid log(0)
        return float(-np.sum(hist * np.log(hist)))

    def _detect_patch_boundaries(self, grad_entropy: float) -> bool:
        """Determines if a new patch should start based on entropy."""
        return grad_entropy > self.entropy_threshold

    def _apply_gradient_updates(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        group: dict,
        state: dict
    ) -> None:
        """Applies gradient updates to parameters with all optimizations."""
        # Weight decay
        if group['weight_decay'] != 0:
            grad = grad.add(p, alpha=group['weight_decay'])

        # Gradient centering
        if self.gradient_centering:
            grad = grad - grad.mean()

        # Gradient clipping
        if self.gradient_clipping:
            grad_norm = torch.norm(grad)
            if grad_norm > self.max_grad_norm:
                grad = grad * (self.max_grad_norm / grad_norm)

        # Gradient noise
        if self.apply_noise:
            noise = torch.randn_like(grad) * self.noise_scale * math.sqrt(group['lr'])
            grad = grad + noise

        # Momentum update
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(p)

        momentum = group['momentum']
        if self.adaptive_momentum:
            grad_var = torch.var(grad).item()
            momentum = min(momentum * (1 + grad_var), 0.999)

        buf = state['momentum_buffer']
        buf.mul_(momentum).add_(grad)

        # Final update
        p.data.add_(buf, alpha=-group['lr'])

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[torch.Tensor]:
        """Performs a single optimization step with dynamic patching."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.state['step'] += 1

        # Process parameter groups
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Compute gradient statistics
                grad_norm = torch.norm(grad).item()
                grad_entropy = self._compute_entropy(grad)
                
                # Track statistics
                self.stats['grad_norms'].append(grad_norm)
                self.stats['entropy_values'].append(grad_entropy)

                # Detect patch boundary
                is_boundary = self._detect_patch_boundaries(grad_entropy)
                if is_boundary:
                    self.state['patch_boundaries'].append(self.state['step'])

                # Q-Learning state and action
                if loss is not None:
                    q_state = self.q_controller.get_state(
                        loss.item(),
                        grad_norm,
                        grad_entropy,
                        self.state['step']
                    )
                    q_action = self.q_controller.choose_action(q_state)

                    # Apply Q-Learning adjustments
                    lr_scale = q_action['lr_scale']
                    group['lr'] = group['lr'] * lr_scale
                    
                    if self.adaptive_momentum:
                        group['momentum'] = group['momentum'] * q_action['momentum_scale']

                # Apply updates
                self._apply_gradient_updates(p, grad, group, state)

                # Track hyperparameters
                self.stats['learning_rates'].append(group['lr'])
                self.stats['momentum_values'].append(group['momentum'])

        # Update patch size statistics
        if len(self.state['patch_boundaries']) >= 2:
            avg_patch_size = (self.state['patch_boundaries'][-1] - 
                            self.state['patch_boundaries'][-2])
            self.stats['patch_sizes'].append(avg_patch_size)

        return loss

    def get_statistics(self) -> Dict[str, float]:
        """Returns current optimizer statistics."""
        stats = {}
        for key, values in self.stats.items():
            if values:
                stats[f'avg_{key}'] = float(np.mean(values))
                stats[f'std_{key}'] = float(np.std(values))
        return stats

    def state_dict(self) -> Dict[str, Any]:
        """Returns the optimizer's state dict with additional statistics."""
        state_dict = super().state_dict()
        state_dict['statistics'] = self.get_statistics()
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads optimizer state with statistics handling."""
        stats = state_dict.pop('statistics', None)
        super().load_state_dict(state_dict)
        if stats is not None:
            self._init_statistics()
            logging.info("Loaded optimizer state with statistics")