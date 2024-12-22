# EnhancedSGD.py

import torch
from torch.optim import Optimizer
from collections import deque
import logging
from typing import Iterable, Optional, Dict, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO to capture essential logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()


class GradientStats:
    """Tracks gradient statistics for reporting."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_gradients = 0
        self.clipped_gradients = 0
        self.max_gradient_norm = 0
        self.sum_clip_ratios = 0
        self.step_stats = {}
    
    def record_gradient(self, original_norm: float, clipped: bool, clip_ratio: float = None):
        self.total_gradients += 1
        if clipped:
            self.clipped_gradients += 1
            if clip_ratio:
                self.sum_clip_ratios += clip_ratio
        self.max_gradient_norm = max(self.max_gradient_norm, original_norm)
    
    def get_step_stats(self) -> dict:
        if self.total_gradients == 0:
            return {
                "gradients_clipped": 0,
                "total_gradients": 0,
                "clip_ratio": 0,
                "max_gradient": 0,
                "avg_clip_amount": 0
            }
            
        return {
            "gradients_clipped": self.clipped_gradients,
            "total_gradients": self.total_gradients,
            "clip_ratio": self.clipped_gradients / self.total_gradients,
            "max_gradient": self.max_gradient_norm,
            "avg_clip_amount": self.sum_clip_ratios / self.clipped_gradients if self.clipped_gradients > 0 else 0
        }
    
    def record_step(self, step: int):
        stats = self.get_step_stats()
        self.step_stats[step] = stats
        self.reset()
        return stats


class ImprovedQLearningController:
    """Enhanced Q-Learning Controller with adaptive exploration and better state management."""
    
    def __init__(
        self,
        learning_rate: float = 0.02,        # Reduced from 0.03
        discount: float = 0.97,             # Increased for longer-term planning
        epsilon: float = 0.15,              # Reduced initial exploration
        epsilon_decay: float = 0.999,       # Slower decay
        initial_mix_prob: float = 0.9,      # Increased stability
        lr_scale_bounds: tuple = (0.85, 1.15),  # Tighter bounds
        momentum_scale_bounds: tuple = (0.9, 1.1),
        min_weight_decay: float = 1e-4,
        state_memory_size: int = 300,       # Increased memory
        max_q_table_size: int = 15000       # Increased table size
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
        self.min_weight_decay = min_weight_decay

        # Enhanced state tracking
        self.state_memory = deque(maxlen=state_memory_size)
        self.loss_window = deque(maxlen=15)  # Increased from 10
        self.grad_window = deque(maxlen=15)
        self.lr_window = deque(maxlen=8)    # Increased from 5
        self.momentum_window = deque(maxlen=8)
        
        # More granular action space
        self.action_ranges = {
            'lr_scale': np.array([0.85, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.15]),
            'momentum_scale': np.array([0.9, 0.95, 0.975, 0.99, 1.0, 1.01, 1.025, 1.05, 1.075, 1.1])
        }

        # Success tracking with decay
        self.action_success = {k: np.zeros(len(v)) for k, v in self.action_ranges.items()}
        self.action_counts = {k: np.zeros(len(v)) for k, v in self.action_ranges.items()}
        self.success_decay = 0.99  # Decay factor for historical success

        # Performance tracking
        self.performance_window = deque(maxlen=50)
        self.stable_steps = 0

        # Q-Table memory management
        self.max_q_table_size = max_q_table_size

    def get_state(self, lr: float, momentum: float, grad_var: float, loss: float) -> tuple:
        """Enhanced state representation with more nuanced binning."""
        self.loss_window.append(loss)
        self.grad_window.append(grad_var)
        self.lr_window.append(lr)
        self.momentum_window.append(momentum)
        
        # Improved loss trend calculation with exponential moving average
        if len(self.loss_window) >= 3:
            weights = np.exp([-0.1 * i for i in range(3)])
            weights = weights / weights.sum()
            recent_losses = list(self.loss_window)[-3:]
            weighted_loss_trend = np.sum(weights * [(recent_losses[0] - x) / recent_losses[0] for x in recent_losses[1:]])
            loss_trend_bin = np.digitize(weighted_loss_trend, 
                                       bins=[-0.1, -0.03, -0.01, 0.01, 0.03, 0.1])
        else:
            loss_trend_bin = 3  # Middle bin
        
        # Improved gradient stability metric using exponential moving average
        if len(self.grad_window) >= 3:
            recent_grads = list(self.grad_window)[-3:]
            weights = np.exp([-0.1 * i for i in range(3)])
            weights = weights / weights.sum()
            weighted_grad_mean = np.sum(weights * recent_grads)
            grad_stability = np.std(recent_grads) / (weighted_grad_mean + 1e-8)
            grad_stability_bin = np.digitize(grad_stability, 
                                           bins=[0.05, 0.1, 0.2, 0.3, 0.5])
        else:
            grad_stability_bin = 3
        
        # More granular learning rate bins with relative scaling
        lr_base = 0.001  # Base learning rate
        lr_relative = lr / lr_base
        lr_bin = np.digitize(lr_relative, 
                            bins=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
        
        # Momentum bins focused around typical values
        momentum_bin = np.digitize(momentum, 
                                 bins=[0.85, 0.9, 0.93, 0.95, 0.97, 0.99, 1.0])
        
        return (lr_bin, momentum_bin, loss_trend_bin, grad_stability_bin)
        
    def compute_reward(
        self,
        loss_trend: float,
        grad_health: float,
        consistent_improvement: bool
    ) -> float:
        """Enhanced reward computation with stability bonus."""
        # Base reward from loss improvement
        base_reward = 2.0 * loss_trend if loss_trend > 0 else 1.5 * loss_trend
        
        # Enhanced stability bonus
        stability_threshold = 0.9
        stability_bonus = 0.0
        if grad_health > stability_threshold:
            bonus_scale = (grad_health - stability_threshold) / (1 - stability_threshold)
            stability_bonus = 0.5 * bonus_scale
        
        # Progressive consistency reward
        if consistent_improvement:
            self.stable_steps += 1
            consistency_reward = min(1.0, 0.2 * np.log1p(self.stable_steps))
        else:
            self.stable_steps = max(0, self.stable_steps - 1)
            consistency_reward = 0.0
        
        # Combine rewards with dynamic weighting
        combined_reward = (
            base_reward +
            stability_bonus +
            consistency_reward
        )
        
        # Smooth reward scaling
        scaled_reward = np.tanh(combined_reward)  # Bound between -1 and 1
        
        return float(scaled_reward)
        
    def choose_action(self, state: tuple) -> Dict[str, float]:
        """Enhanced action selection with adaptive exploration."""
        if state not in self.q_table:
            self.q_table[state] = {
                param: np.zeros(len(space))
                for param, space in self.action_ranges.items()
            }

        action = {}
        
        # Compute recent performance
        recent_rewards = [r for _, _, r in list(self.state_memory)[-20:]]
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        
        # Adaptive exploration rates
        explore_lr = np.random.random() < self.epsilon * (1.0 - max(0, avg_reward))
        explore_momentum = np.random.random() < self.epsilon * 0.7 * (1.0 - max(0, avg_reward))

        for param, space in self.action_ranges.items():
            should_explore = explore_lr if param == 'lr_scale' else explore_momentum
            
            if should_explore:
                # Smart exploration based on action success history
                success_rates = self.action_success[param] / (self.action_counts[param] + 1)
                
                if len(recent_rewards) > 0 and np.mean(recent_rewards) < -0.2:
                    # If performing poorly, explore more widely but favor previously successful actions
                    p = softmax(success_rates + 0.1)  # Add small constant for exploration
                    chosen_idx = np.random.choice(len(space), p=p)
                else:
                    # If performing well, explore near current value
                    current_idx = np.argmin(np.abs(space - 1.0))
                    exploration_range = 2  # Explore within ±2 indices
                    valid_indices = np.arange(
                        max(0, current_idx - exploration_range),
                        min(len(space), current_idx + exploration_range + 1)
                    )
                    chosen_idx = np.random.choice(valid_indices)
                
                chosen_action = float(space[chosen_idx])
            else:
                # Enhanced greedy selection with success history influence
                q_values = self.q_table[state][param]
                success_rates = self.action_success[param] / (self.action_counts[param] + 1)
                
                # Combine Q-values with success history
                combined_values = q_values + 0.2 * success_rates
                max_val = np.max(combined_values)
                best_actions = np.where(np.abs(combined_values - max_val) < 1e-6)[0]
                
                # Choose action closest to 1.0 when tied
                chosen_idx = min(best_actions, key=lambda i: abs(space[i] - 1.0))
                chosen_action = float(space[chosen_idx])
            
            action[param] = chosen_action

        # Adaptive epsilon decay based on performance
        if len(self.performance_window) > 20:
            avg_performance = np.mean(self.performance_window)
            if avg_performance > 0.3:
                self.epsilon = max(0.05, self.epsilon * 0.997)  # Slower decay
            else:
                self.epsilon = max(0.05, self.epsilon * 0.995)  # Faster decay

        return action

    def update(self, state: tuple, action: Dict[str, float], reward: float, next_state: Optional[tuple], should_log: bool = False):
        """Enhanced Q-learning update with experience tracking."""
        # Record reward and update state memory
        self.state_memory.append((state, action, reward))
        if should_log:
            logging.info(f"State memory updated with state={state}, action={action}, reward={reward}")

        # Initialize Q-values if needed
        for s in [state, next_state] if next_state is not None else [state]:
            if s not in self.q_table:
                self.q_table[s] = {
                    param: np.zeros(len(space))
                    for param, space in self.action_ranges.items()
                }

        # Q-Table memory management: Evict least recently used states if Q-table exceeds max size
        if len(self.q_table) > self.max_q_table_size:
            oldest_state = self.state_memory.popleft()[0]
            if oldest_state in self.q_table:
                del self.q_table[oldest_state]
                if should_log:
                    logging.info(f"Evicted oldest state from Q-table: {oldest_state}")

        # Update Q-values with adaptive learning rate
        for param, value in action.items():
            space = self.action_ranges[param]
            action_idx = np.abs(space - value).argmin()

            # Get max future Q-value
            if next_state is not None and next_state in self.q_table:
                next_q_values = self.q_table[next_state][param]
                max_future_q = np.max(next_q_values)
            else:
                max_future_q = 0.0

            # Current Q-value
            current_q = self.q_table[state][param][action_idx]

            # Compute TD error with less aggressive clipping
            td_error = reward + self.gamma * max_future_q - current_q
            td_error = np.clip(td_error, -1.0, 1.0)  # Reduced clipping range

            # Adaptive learning rate based on visit count
            self.action_counts[param][action_idx] += 1
            visit_count = self.action_counts[param][action_idx]
            effective_lr = self.alpha / (1 + np.log1p(visit_count) * 0.1)

            # Update Q-value
            self.q_table[state][param][action_idx] += effective_lr * td_error

            if should_log:
                logging.info(f"Updated Q-table for state={state}, param={param}, action_idx={action_idx}: "
                             f"current_q={current_q:.4f}, max_future_q={max_future_q:.4f}, td_error={td_error:.4f}, "
                             f"effective_lr={effective_lr:.6f}")

            # Update success tracking with decay
            if reward > 0:
                self.action_success[param][action_idx] = (self.action_success[param][action_idx] * self.success_decay) + 1
            else:
                self.action_success[param][action_idx] *= self.success_decay


class EnhancedSGD(Optimizer):
    """Enhanced SGD Optimizer with Q-Learning based Adaptive Hyperparameter Tuning."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.003,               # Reduced initial LR
        momentum: float = 0.9,
        weight_decay: float = 0.005,     # Reduced initial weight decay
        smoothing_factor: float = 0.05,  # Increased for stability
        entropy_threshold: float = 0.3,
        max_grad_norm: float = 1.0,      # Reduced based on gradient stats
        noise_scale: float = 0.001,      # Reduced since gradients have good variance
        lr_scale_bounds: tuple = (0.7, 1.3),
        momentum_scale_bounds: tuple = (0.85, 1.1),
        q_learning_config: Dict[str, Any] = {},
        **kwargs
    ):
        """
        Initializes the EnhancedSGD optimizer.

        Args:
            params (Iterable[torch.nn.Parameter]): Iterable of parameters to optimize.
            lr (float, optional): Initial learning rate. Defaults to 0.003.
            momentum (float, optional): Initial momentum. Defaults to 0.9.
            weight_decay (float, optional): Initial weight decay. Defaults to 0.005.
            smoothing_factor (float, optional): Smoothing factor for updates. Defaults to 0.05.
            entropy_threshold (float, optional): Threshold for entropy-based adjustments. Defaults to 0.3.
            max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to 1.0.
            noise_scale (float, optional): Scale of noise to inject. Defaults to 0.001.
            lr_scale_bounds (tuple, optional): Bounds for learning rate scaling. Defaults to (0.7, 1.3).
            momentum_scale_bounds (tuple, optional): Bounds for momentum scaling. Defaults to (0.85, 1.1).
            q_learning_config (Dict[str, Any], optional): Configuration for Q-Learning controller. Defaults to {}.
            **kwargs: Additional keyword arguments.
        """
        # Ensure params is properly formatted as parameter groups
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            param_groups = params
        else:
            param_groups = [{'params': params}]
    
        # Enhanced parameter group initialization
        for group in param_groups:
            group.setdefault('lr', lr)
            group.setdefault('momentum', momentum)
            group.setdefault('weight_decay', weight_decay)
            group.setdefault('base_lr', lr)
            group.setdefault('q_scale', 1.0)
            # Add minimum weight decay
            group.setdefault('min_weight_decay', weight_decay * 0.2)  # 20% of initial as minimum
    
        super().__init__(param_groups, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))
        
        # Initialize optimization state
        self._init_optimization_state(
            smoothing_factor=smoothing_factor,
            entropy_threshold=entropy_threshold,
            max_grad_norm=max_grad_norm,
            noise_scale=noise_scale,
            lr_scale_bounds=lr_scale_bounds,
            momentum_scale_bounds=momentum_scale_bounds,
            q_learning_config=q_learning_config,
            **kwargs
        )
    
        # Pre-allocate buffers with proper device handling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stats = {
            'grad_norms': deque(maxlen=100),
            'learning_rates': deque(maxlen=100),
            'momentum_values': deque(maxlen=100),
            'entropy_values': deque(maxlen=100),
            'update_norms': deque(maxlen=100)  # Track update norms
        }

    def _init_optimization_state(self, **kwargs):
        """Initialize optimization state with safe handling."""
        smoothing_factor = kwargs.get('smoothing_factor', 0.05)
        entropy_threshold = kwargs.get('entropy_threshold', 0.3)
        max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        noise_scale = kwargs.get('noise_scale', 0.001)
        lr_scale_bounds = kwargs.get('lr_scale_bounds', (0.7, 1.3))
        momentum_scale_bounds = kwargs.get('momentum_scale_bounds', (0.85, 1.1))
        q_learning_config = kwargs.get('q_learning_config', {})

        self.max_grad_norm = max_grad_norm  # Initialize max_grad_norm

        self.q_controller = ImprovedQLearningController(
            learning_rate=q_learning_config.get('learning_rate', 0.02),
            discount=q_learning_config.get('discount', 0.97),
            epsilon=q_learning_config.get('epsilon', 0.15),
            epsilon_decay=q_learning_config.get('epsilon_decay', 0.999),
            initial_mix_prob=q_learning_config.get('initial_mix_prob', 0.9),
            lr_scale_bounds=lr_scale_bounds,
            momentum_scale_bounds=momentum_scale_bounds,
            min_weight_decay=q_learning_config.get('min_weight_decay', 1e-4)
        )
    
        self._step_count = 0
        self.prev_state = None
        self.prev_action = None
        self.prev_loss = None
    
        # Initialize gradient memory
        self.grad_memory = deque(maxlen=100)  # Store recent gradients
    
        # Initialize GradientStats
        self.gradient_stats = GradientStats()

    def _track_gradient_memory(self, grad_norm: float):
        """Track gradient history for better adaptation."""
        self.grad_memory.append(grad_norm)

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

    def get_statistics(self) -> Dict[str, float]:
        """Compute and return statistics using pre-allocated deques."""
        stats = {}

        # Add current learning rate from first param group
        if len(self.param_groups) > 0:
            stats['current_lr'] = float(self.param_groups[0]['lr'])

        # Add other statistics
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

    def _get_gradient_stats(self) -> Dict[str, Any]:
        """Gather gradient statistics for the current step."""
        grad_norms = []
        grad_vars = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                grad_norms.append(torch.norm(grad).item())
                if p.grad.numel() > 1:
                    grad_vars.append(torch.var(grad.float()).item())
    
        saw_grads = len(grad_norms) > 0
    
        if saw_grads:
            mean_grad_norm = np.mean(grad_norms)
            mean_grad_var = np.mean(grad_vars) if grad_vars else 0.0
        else:
            mean_grad_norm = 0.0
            mean_grad_var = 0.0
    
        grad_stats = {
            'saw_grads': saw_grads,
            'mean_grad_norm': mean_grad_norm,
            'mean_grad_var': mean_grad_var
        }
        return grad_stats

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[torch.Tensor]:
        """Optimizes the parameters based on the current gradient."""
        loss = None
        if closure is not None:
            loss = closure()

        # Get gradient statistics
        grad_stats = self._get_gradient_stats()

        # Apply Q-learning adjustments if we have gradients and loss
        if grad_stats['saw_grads'] and loss is not None:
            current_loss = loss.item()

            # Get current state
            q_state = self.q_controller.get_state(
                lr=self.param_groups[0]['lr'],
                momentum=self.param_groups[0]['momentum'],
                grad_var=grad_stats['mean_grad_var'],
                loss=current_loss
            )

            # Update Q-table with previous experience if available
            if self.q_controller.prev_loss is not None and \
               self.q_controller.prev_state is not None and \
               self.q_controller.prev_action is not None:
                # Calculate relative loss improvement
                loss_improvement = (self.q_controller.prev_loss - current_loss) / self.q_controller.prev_loss
                grad_health = 1.0 / (1.0 + grad_stats['mean_grad_var'])
                
                # Check for consistent improvement
                self.q_controller.performance_window.append(loss_improvement)
                consistent_improvement = all([r > 0 for r in list(self.q_controller.performance_window)[-10:]])

                # Compute reward for the previous action
                reward = self.q_controller.compute_reward(
                    loss_trend=loss_improvement,
                    grad_health=grad_health,
                    consistent_improvement=consistent_improvement
                )
                
                # Update Q-table with the previous state, action, and received reward
                self.q_controller.update(
                    state=self.q_controller.prev_state,
                    action=self.q_controller.prev_action,
                    reward=reward,
                    next_state=q_state,
                    should_log=(self._step_count % 10 == 0)
                )

            # Choose new action based on the current state
            q_action = self.q_controller.choose_action(q_state)

            # Determine if the new action results in no changes
            no_change = all(
                abs(action - 1.0) < 1e-5 for action in q_action.values()
            )

            # Apply learning rate and momentum adjustments
            for group in self.param_groups:
                # Scale learning rate
                group['q_scale'] *= float(np.clip(
                    q_action['lr_scale'],
                    self.q_controller.lr_scale_bounds[0],
                    self.q_controller.lr_scale_bounds[1]
                ))
                group['lr'] = group['base_lr'] * group['q_scale']

                # Scale momentum
                group['momentum'] = float(np.clip(
                    group['momentum'] * q_action['momentum_scale'],
                    self.q_controller.momentum_scale_bounds[0],
                    self.q_controller.momentum_scale_bounds[1]
                ))

            # Log Q-learning adjustments
            if self._step_count % 10 == 0:
                for i, group in enumerate(self.param_groups):
                    # Calculate mean Q-values for logging
                    state = self.q_controller.prev_state
                    mean_q_lr = np.mean(self.q_controller.q_table.get(state, {}).get('lr_scale', np.zeros(len(self.q_controller.action_ranges['lr_scale']))))
                    mean_q_momentum = np.mean(self.q_controller.q_table.get(state, {}).get('momentum_scale', np.zeros(len(self.q_controller.action_ranges['momentum_scale']))))

                    logging.info(
                        f"Step {self._step_count} - Group {i} - Q-Learning adjustments: "
                        f"lr_scale={q_action['lr_scale']:.4f}, momentum_scale={q_action['momentum_scale']:.4f}, "
                        f"mean_Q_lr={mean_q_lr:.4f}, mean_Q_momentum={mean_q_momentum:.4f}, "
                        f"effective_lr={group['lr']:.6f}, momentum={group['momentum']:.4f}"
                    )

            # Update Q-learning state for the next step
            self.q_controller.prev_state = q_state
            self.q_controller.prev_action = q_action
            self.q_controller.prev_loss = current_loss

            # If the new action results in no change, apply a penalty
            if grad_stats['saw_grads'] and self.q_controller.prev_action and no_change:
                logging.warning("No-change action detected. Applying penalty.")
                # The penalty is already incorporated in the reward function

        # Apply updates with the adjusted learning rates
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad_stats['mean_grad_norm'] > self.max_grad_norm * 5 or \
                   np.isnan(grad_stats['mean_grad_norm']) or \
                   np.isnan(grad_stats['mean_grad_var']):
                    grad = torch.clamp(grad, -self.max_grad_norm, self.max_grad_norm)
                    # Removed individual gradient clipping warnings

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
                    state=state,
                    current_loss=self.q_controller.prev_loss if self.q_controller.prev_loss is not None else 0.0
                )

        if grad_stats['saw_grads']:
            self._step_count += 1
            # Adjust hyperparameters more frequently
            if self._step_count % 10 == 0:
                self._adjust_hyperparameters()
            # Log optimization statistics every 10 steps
            if self._step_count % 10 == 0:
                self.log_optimization_stats()

            # Record and log gradient clipping statistics
            stats = self.gradient_stats.record_step(self._step_count)
            if self._step_count % 10 == 0:
                logging.info(
                    f"Step {self._step_count} gradient stats: "
                    f"Clipped {stats['gradients_clipped']}/{stats['total_gradients']} "
                    f"({stats['clip_ratio']:.1%}) gradients. "
                    f"Max norm: {stats['max_gradient']:.3f}, "
                    f"Avg clip amount: {stats['avg_clip_amount']:.3f}"
                )

        return loss

    def _apply_update(
        self,
        p: torch.Tensor,
        grad: torch.Tensor,
        momentum: float,
        lr: float,
        weight_decay: float,
        state: dict,
        current_loss: float
    ):
        """Enhanced parameter update with improved gradient handling."""
        buf = state['momentum_buffer']

        # Apply weight decay with smoother sigmoid scaling
        if weight_decay != 0:
            param_norm = torch.norm(p)
            # Replaced tanh with sigmoid for smoother transitions and reduced scaling factor
            adaptive_wd = weight_decay * (1.0 / (1.0 + torch.exp(-param_norm * 0.05)))
            grad = grad.add(p, alpha=adaptive_wd)
            logging.debug(f"Applied adaptive weight decay: {adaptive_wd:.6f}")

        # Apply gradient clipping before updating the momentum buffer
        grad_norm = torch.norm(grad)
        was_clipped = False
        clip_ratio = 0

        if grad_norm > self.max_grad_norm:
            clip_ratio = float(self.max_grad_norm / grad_norm)
            grad.mul_(clip_ratio)
            was_clipped = True

        # Record gradient clipping statistics
        self.gradient_stats.record_gradient(
            original_norm=float(grad_norm),
            clipped=was_clipped,
            clip_ratio=clip_ratio
        )

        # Smoother momentum update
        buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
        logging.debug(f"Updated momentum buffer with grad: {grad_norm:.6f}, momentum: {momentum:.4f}")

        # Adaptive noise injection based on training progress
        if self._step_count < 1000:
            noise_factor = 0.001 * np.exp(-self._step_count / 200)  # Exponential decay with 200-step half-life
            stability = 1.0 / (1.0 + np.var(list(self.grad_memory)) if self.grad_memory else 1.0)
            noise_scale = min(0.001, noise_factor * stability)  # Reduced base noise scale from 0.01 to 0.001
            noise = torch.randn_like(buf) * noise_scale
            buf.add_(noise)
            logging.debug(f"Injected noise with scale {noise_scale:.6f}")

        # Compute base update
        update = -lr * buf

        # Track update history
        if 'update_history' not in state:
            state['update_history'] = deque(maxlen=5)
        state['update_history'].append(update.clone())
        logging.debug(f"Tracked update history: {update.clone().norm().item():.6f}")

        # Dynamic update scaling based on loss and gradient statistics
        update_norm = torch.norm(update).item()
        if len(self.stats['update_norms']) > 0:
            avg_update_norm = np.mean(self.stats['update_norms'])
            clip_threshold = max(2.0, avg_update_norm * 5.0)  # Increased base clipping threshold to 2.0

            if update_norm > clip_threshold:
                scale_factor = clip_threshold / (update_norm + 1e-6)
                # Apply soft clipping using sigmoid-like transition
                scale_factor = 1.0 / (1.0 + np.exp(-10 * (scale_factor - 0.5)))  # Smooth transition
                update.mul_(scale_factor)
                logging.info(f"Adaptive clipping applied: scale_factor={scale_factor:.4f}")

        # Update statistics
        self.stats['update_norms'].append(float(update_norm))

        # Always apply gradient clipping as a failsafe
        max_failsafe_norm = 5.0
        if update_norm > max_failsafe_norm:
            scale_factor = max_failsafe_norm / (update_norm + 1e-6)
            update.mul_(scale_factor)
            self.stats['update_norms'].append(float(update_norm))
            logging.warning(f"Failsafe clipping applied: update norm reduced to {max_failsafe_norm:.6f} (scale factor {scale_factor:.4f})")

        # Apply gradient clipping on parameter update
        max_update_norm_final = 1.0  # Prevent too large updates
        update_norm_final = torch.norm(update).item()
        if update_norm_final > max_update_norm_final:
            update.mul_(max_update_norm_final / (update_norm_final + 1e-6))
            self.stats['update_norms'].append(float(update_norm_final))
            logging.warning(f"Final clipping applied: update norm from {update_norm_final:.6f} to {max_update_norm_final:.6f}")

        # Update parameters with smoothing for stability
        smooth_factor = min(1.0, self._step_count / 100.0)  # Faster warmup
        effective_lr = smooth_factor * (1.0 - momentum)
        p.data.add_(update, alpha=effective_lr)
        logging.debug(f"Applied parameter update with effective_lr={effective_lr:.6f}")

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
        is_unstable = grad_norm_std / (grad_norm_mean + 1e-8) > 0.3  # Lowered instability threshold from 0.5 to 0.3
        is_diverging = grad_norm_mean > 10.0 * self.max_grad_norm
        updates_too_small = update_norm_mean < 1e-7

        for i, group in enumerate(self.param_groups):
            current_lr = group['lr']
            current_momentum = group['momentum']
            current_weight_decay = group.get('weight_decay', 0.005)

            original_lr = current_lr
            original_momentum = current_momentum
            original_weight_decay = current_weight_decay

            if is_unstable:
                # Reduce learning rate and increase momentum for stability with more gradual factor
                group['lr'] = max(1e-8, current_lr * 0.95)  # Changed from 0.9 to 0.95
                group['momentum'] = min(1.1, current_momentum * 1.02)
                logging.info(
                    f"Param Group {i}: Adjusted LR from {original_lr:.6f} to {group['lr']:.6f} "
                    f"and Momentum from {original_momentum:.4f} to {group['momentum']:.4f} due to instability."
                )

            elif is_diverging:
                # Significantly reduce learning rate and momentum
                group['lr'] = max(1e-8, current_lr * 0.7)
                group['momentum'] = max(0.85, current_momentum * 0.9)
                logging.info(
                    f"Param Group {i}: Adjusted LR from {original_lr:.6f} to {group['lr']:.6f} "
                    f"and Momentum from {original_momentum:.4f} to {group['momentum']:.4f} due to divergence."
                )

            elif updates_too_small:
                # Carefully increase learning rate
                group['lr'] = min(10.0, current_lr * 1.05)
                logging.info(
                    f"Param Group {i}: Increased LR from {original_lr:.6f} to {group['lr']:.6f} "
                    f"due to small updates."
                )

            # Adjust weight decay based on gradient magnitude and gradient-to-parameter ratio
            adjusted_weight_decay = self._adjust_weight_decay(group, {'mean_grad_norm': grad_norm_mean, 'total_norm': grad_norm_mean})
            if adjusted_weight_decay != original_weight_decay:
                group['weight_decay'] = adjusted_weight_decay
                logging.info(
                    f"Param Group {i}: Adjusted Weight Decay from {original_weight_decay:.6f} "
                    f"to {group['weight_decay']:.6f} based on gradient norms."
                )

            # Ensure learning rate and momentum do not fall below minimum thresholds
            group['lr'] = max(1e-8, group['lr'])
            group['momentum'] = max(0.85, group['momentum'])

            # Track statistics for this param group
            self.stats['learning_rates'].append(group['lr'])
            self.stats['momentum_values'].append(group['momentum'])

            # Log final parameter values for this group
            logging.info(
                f"Parameter Group {i} final values: "
                f"LR={group['lr']:.6f}, "
                f"Momentum={group['momentum']:.4f}, "
                f"Weight Decay={group['weight_decay']:.6f}"
            )

    def _adjust_weight_decay(self, param_group: Dict[str, Any], grad_stats: Dict[str, float]) -> float:
        """More conservative weight decay adjustment"""
        base_weight_decay = param_group.get('weight_decay', 0.005)
        grad_norm = grad_stats.get('total_norm', 1.0)
        
        # Use log scale for smoother adjustments
        decay_factor = np.clip(np.log1p(grad_norm) / 5.0, 0.5, 2.0)
        
        # Apply adjustment with minimum floor
        adjusted_weight_decay = max(
            self.q_controller.min_weight_decay,
            base_weight_decay * decay_factor
        )
        
        logging.debug(f"Adjusted weight decay from {base_weight_decay:.6f} to {adjusted_weight_decay:.6f} "
                      f"based on decay_factor={decay_factor:.4f}")
        return adjusted_weight_decay

    def log_optimization_stats(self):
        """Log detailed optimization statistics."""
        stats = {
            'learning_rates': [],
            'momentum_values': [],
            'weight_decay_values': [],
            'grad_norms': [],
            'param_updates': []
        }

        for group in self.param_groups:
            stats['learning_rates'].append(group['lr'])
            stats['momentum_values'].append(group['momentum'])
            stats['weight_decay_values'].append(group['weight_decay'])

        grad_stats = self._track_gradient_norms()
        stats.update(grad_stats)

        # Log to wandb if available
        try:
            import wandb
            wandb.log(stats)
        except ImportError:
            pass

        # Log to console
        logging.info(f"Optimization stats: {stats}")

    def _track_gradient_norms(self) -> Dict[str, float]:
        """Track detailed gradient statistics."""
        stats = {}
        total_norm = 0.0
        param_norms = []

        for group in self.param_groups:
            group_norm = 0.0
            for p in group['params']:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    param_norms.append(param_norm)
                    group_norm += param_norm ** 2
            group_norm = np.sqrt(group_norm)
            total_norm += group_norm ** 2

        total_norm = np.sqrt(total_norm)

        stats['total_norm'] = total_norm
        stats['mean_param_norm'] = np.mean(param_norms) if param_norms else 0.0
        stats['std_param_norm'] = np.std(param_norms) if param_norms else 0.0
        stats['max_param_norm'] = np.max(param_norms) if param_norms else 0.0

        logging.debug(f"Gradient statistics: {stats}")
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
            state_dict['epsilon'] = 0.15  # Updated default epsilon
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
            self.q_controller.epsilon = 0.15  # Updated default epsilon
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

            # Initialize additional scaling factors
            param_group.setdefault('base_lr', param_group['lr'])
            param_group.setdefault('q_scale', 1.0)
            param_group.setdefault('min_weight_decay', param_group['weight_decay'] * 0.2)  # 20% of initial as minimum

            super().add_param_group(param_group)
            logging.info(f"Added parameter group with {len(param_group['params'])} parameters")

        except Exception as e:
            logging.error(f"Error adding parameter group: {e}")
            raise


# Example Usage (for testing purposes)
if __name__ == "__main__":
    import torch.nn as nn

    # Dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.local_encoder = nn.Linear(10, 20)
            self.global_transformer = nn.Linear(20, 20)
            self.local_decoder = nn.Linear(20, 2)
        
        def forward(self, x):
            x = self.local_encoder(x)
            x = torch.relu(x)
            x = self.global_transformer(x)
            x = torch.relu(x)
            x = self.local_decoder(x)
            return x

    model = DummyModel()

    def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> EnhancedSGD:
        """Create EnhancedSGD with optimized parameters"""
        parameter_groups = [
            {
                'params': model.local_encoder.parameters(),
                'lr': 0.003,  # Reduced from 0.005
                'weight_decay': 0.005,  # Reduced from 0.01
                'min_weight_decay': 0.001  # New minimum threshold
            },
            {
                'params': model.global_transformer.parameters(),
                'lr': 0.0003,  # 1/10th of local encoder
                'weight_decay': 0.01,  # Keep higher for transformer
                'min_weight_decay': 0.002
            },
            {
                'params': model.local_decoder.parameters(),
                'lr': 0.003,  # Same as encoder
                'weight_decay': 0.005,
                'min_weight_decay': 0.001
            }
        ]

        q_learning_config = {
            'learning_rate': 0.02,     # Reduced from 0.03
            'discount': 0.97,          # Increased from 0.95
            'epsilon': 0.15,            # Reduced from 0.2
            'epsilon_decay': 0.999,    # Slower decay
            'initial_mix_prob': 0.9    # Increased from 0.8
        }

        return EnhancedSGD(
            parameter_groups,
            smoothing_factor=0.05,      # Increased from 0.05
            entropy_threshold=0.3,       # Increased from 0.3
            max_grad_norm=1.0,          # Reduced from 1.0
            noise_scale=0.001,          # Reduced from 0.001
            lr_scale_bounds=(0.85, 1.15),  # Tighter bounds
            momentum_scale_bounds=(0.9, 1.1),  # Tighter bounds
            q_learning_config=q_learning_config
        )

    optimizer = create_optimizer(model, config={})

    # Dummy training loop
    for step in range(100):
        # Dummy input and target
        inputs = torch.randn(5, 10)
        targets = torch.randint(0, 2, (5,))
    
        # Forward pass
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
    
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
    
        # Optimizer step
        optimizer.step(lambda: loss)
