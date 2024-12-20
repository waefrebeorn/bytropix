# EnhancedSGD.py

import torch
from torch.optim import Optimizer
from collections import deque
import logging
from typing import Iterable, Optional, Dict, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class QLearningController:
    """Enhanced Q-Learning Controller for Adaptive Hyperparameter Tuning."""

    def __init__(
        self,
        learning_rate: float = 0.05,        # Reduced from 0.1
        discount: float = 0.98,             # Increased from 0.95 for longer-term planning
        epsilon: float = 0.3,               # Reduced from 0.5 for more exploitation
        epsilon_decay: float = 0.995,       # Slower decay
        initial_mix_prob: float = 0.7,
        lr_scale_bounds: tuple = (0.5, 1.5),
        momentum_scale_bounds: tuple = (0.8, 1.2),
        state_memory_size: int = 200,       # Added parameter for state memory
        max_q_table_size: int = 10000        # Maximum Q-table size
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

        # Enhanced state tracking
        self.state_memory = deque(maxlen=state_memory_size)
        self.loss_window = deque(maxlen=10)
        self.grad_window = deque(maxlen=10)
        self.lr_window = deque(maxlen=5)         # Track recent lr changes
        self.momentum_window = deque(maxlen=5)   # Track recent momentum changes

        # Refined action space with more granular options
        self.action_ranges = {
            'lr_scale': np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15]),
            'momentum_scale': np.array([0.95, 0.975, 0.99, 1.0, 1.01, 1.025, 1.05])
        }

        # Initialize success tracking
        self.action_success = {k: np.zeros(len(v)) for k, v in self.action_ranges.items()}
        self.action_counts = {k: np.zeros(len(v)) for k, v in self.action_ranges.items()}

        # Consistent improvement tracking
        self.consistent_improvement = deque(maxlen=10)

        # Q-Table memory management
        self.max_q_table_size = max_q_table_size

    def get_state(self, lr: float, momentum: float, grad_var: float, loss: float) -> tuple:
        """Enhanced state representation with current hyperparameters and gradient metrics."""
        # Track loss history
        self.loss_window.append(loss)
        self.grad_window.append(grad_var)

        # Track learning rate and momentum changes
        self.lr_window.append(lr)
        self.momentum_window.append(momentum)

        # Compute loss trend
        if len(self.loss_window) >= 3:
            recent_losses = list(self.loss_window)[-3:]
            loss_trend = np.polyfit(range(3), recent_losses, 1)[0]
            loss_trend_bin = np.digitize(loss_trend, bins=[-0.1, -0.01, -0.001, 0.001, 0.01, 0.1])
        else:
            loss_trend_bin = 3  # Middle bin as default

        # Compute gradient stability
        if len(self.grad_window) >= 3:
            grad_std = np.std(list(self.grad_window)[-3:])
            grad_stability_bin = np.digitize(grad_std, bins=[1e-6, 1e-4, 1e-2, 1e-1])
        else:
            grad_stability_bin = 2  # Middle bin as default

        # Current learning rate bin
        lr_bin = np.digitize(lr, bins=[0.5, 0.7, 0.9, 1.0, 1.1, 1.3])

        # Current momentum bin
        momentum_bin = np.digitize(momentum, bins=[0.8, 0.9, 1.0, 1.1, 1.2])

        state = (lr_bin, momentum_bin, loss_trend_bin, grad_stability_bin)
        logging.debug(f"Generated state: {state}")
        return state

    def compute_reward(
        self,
        loss_trend: float,
        grad_health: float,
        consistent_improvement: bool,
        no_change: bool = False
    ) -> float:
        """Enhanced reward computation emphasizing stability and consistent improvement."""
        # Linear scaling for loss improvement within bounds
        loss_improvement = np.clip(loss_trend, -0.5, 0.5)  # Define bounds
        base_reward = loss_improvement  # Linear scaling

        # Gradient health factor with stronger influence
        grad_factor = np.clip(grad_health, 0, 1)

        # Enhanced stability reward
        stability_factor = np.clip(grad_health * 1.5, 0, 1)

        # Consistent improvement bonus
        consistency_bonus = 0.1 if consistent_improvement else 0.0

        # Penalize extreme changes
        smoothness_penalty = -0.1 * (1.0 - stability_factor)

        # Combine all factors
        reward = (
            base_reward * 0.4 +
            grad_factor * 0.3 +
            stability_factor * 0.2 +
            consistency_bonus +
            smoothness_penalty
        )

        # Apply reduced penalty for no changes
        if no_change:
            penalty = -0.2  # Reduced from -0.5
            reward += penalty
            logging.debug("Applied reduced no-change penalty to the reward.")

        logging.debug(f"Computed reward: {reward} (base_reward={base_reward}, grad_factor={grad_factor}, "
                      f"stability_factor={stability_factor}, consistency_bonus={consistency_bonus}, "
                      f"smoothness_penalty={smoothness_penalty})")
        return float(np.clip(reward, -1.0, 1.0))

    def update(self, state: tuple, action: Dict[str, float], reward: float, next_state: Optional[tuple]):
        """Enhanced Q-learning update with experience tracking."""
        # Record reward and update state memory
        self.state_memory.append((state, action, reward))
        logging.debug(f"State memory updated with state={state}, action={action}, reward={reward}")

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
                logging.debug(f"Evicted oldest state from Q-table: {oldest_state}")

        # Update Q-values with adaptive learning rate
        for param, value in action.items():
            space = self.action_ranges[param]
            action_idx = np.abs(space - value).argmin()

            # Get max future Q-value
            if next_state is not None:
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

            logging.debug(f"Updated Q-table for state={state}, param={param}, action_idx={action_idx}: "
                          f"current_q={current_q}, max_future_q={max_future_q}, td_error={td_error}, "
                          f"effective_lr={effective_lr}")

            # Update success tracking
            if reward > 0:
                self.action_success[param][action_idx] += 1

    def choose_action(self, state: tuple) -> Dict[str, float]:
        """Enhanced action selection with success-based exploration and adaptive epsilon decay."""
        if state not in self.q_table:
            self.q_table[state] = {
                param: np.zeros(len(space))
                for param, space in self.action_ranges.items()
            }

        # Compute success rates for each action
        success_rates = {}
        for param, space in self.action_ranges.items():
            counts = self.action_counts[param]
            successes = self.action_success[param]
            rates = np.where(counts > 0, successes / counts, 0.5)
            success_rates[param] = rates

        action = {}
        if np.random.random() < self.epsilon:
            # Smart exploration based on success rates
            for param, space in self.action_ranges.items():
                rates = success_rates[param]
                # Softmax temperature decreases with more experience
                temp = 1.0 / (1.0 + len(self.state_memory) * 0.01)
                probs = np.exp(rates / temp)
                probs /= probs.sum()
                chosen_action = float(np.random.choice(space, p=probs))
                action[param] = chosen_action
                logging.debug(f"Exploratory action chosen for {param}: {chosen_action}")
        else:
            # Greedy selection
            for param, space in self.action_ranges.items():
                q_values = self.q_table[state][param]
                best_idx = np.argmax(q_values)
                chosen_action = float(space[best_idx])
                action[param] = chosen_action
                logging.debug(f"Greedy action chosen for {param}: {chosen_action}")

        # Adaptive epsilon decay based on recent rewards
        recent_rewards = [record[2] for record in list(self.state_memory)[-50:]]
        if recent_rewards:
            avg_recent_reward = np.mean(recent_rewards)
            # Decay faster if performing well
            if avg_recent_reward > 0.5:
                decay_factor = 0.997
            else:
                decay_factor = 0.995
            old_epsilon = self.epsilon
            self.epsilon = max(0.05, self.epsilon * decay_factor)
            logging.debug(f"Epsilon decayed from {old_epsilon:.4f} to {self.epsilon:.4f}")

        logging.info(f"Action chosen: {action}")
        return action


class EnhancedSGD(Optimizer):
    """Enhanced SGD Optimizer with Q-Learning based Adaptive Hyperparameter Tuning."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.005,               # Updated default learning rate
        momentum: float = 0.9,
        weight_decay: float = 0.01,
        smoothing_factor: float = 0.03,
        entropy_threshold: float = 0.2,
        max_grad_norm: float = 0.5,
        noise_scale: float = 1e-5,
        lr_scale_bounds: tuple = (0.5, 1.5),
        momentum_scale_bounds: tuple = (0.8, 1.2),
        warmup_steps: int = 50,          # Shortened warmup steps
        **kwargs
    ):
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
            group.setdefault('base_lr', lr)           # Base learning rate for reference
            group.setdefault('initial_lr', lr)        # Initial learning rate
            group.setdefault('warmup_factor', 1.0)    # Warmup scaling factor
            group.setdefault('q_scale', 1.0)          # Q-learning scaling factor

        super().__init__(param_groups, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

        # Initialize optimization state
        self._init_optimization_state(
            smoothing_factor=smoothing_factor,
            entropy_threshold=entropy_threshold,
            max_grad_norm=max_grad_norm,
            noise_scale=noise_scale,
            lr_scale_bounds=lr_scale_bounds,
            momentum_scale_bounds=momentum_scale_bounds,
            warmup_steps=warmup_steps,
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
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.q_controller = QLearningController(
            learning_rate=kwargs.get('learning_rate', 0.05),
            discount=kwargs.get('discount', 0.98),
            epsilon=kwargs.get('epsilon', 0.3),
            epsilon_decay=kwargs.get('epsilon_decay', 0.995),
            initial_mix_prob=kwargs.get('initial_mix_prob', 0.7),
            lr_scale_bounds=self.lr_scale_bounds,
            momentum_scale_bounds=self.momentum_scale_bounds
        )

        self._step_count = 0
        self.prev_state = None
        self.prev_action = None
        self.prev_loss = None

        # Initialize gradient memory
        self.grad_memory = deque(maxlen=100)  # Store recent gradients

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

    def _apply_warmup(self):
        """Warmup logic applied per step with gradient clipping."""
        if self._step_count < self.warmup_steps:
            warmup_factor = float(self._step_count) / float(max(1, self.warmup_steps))
            for group in self.param_groups:
                group['warmup_factor'] = warmup_factor
                # Apply warmup to the effective learning rate
                group['lr'] = group['base_lr'] * group['warmup_factor'] * group['q_scale']
            logging.info(f"Applying warmup: step={self._step_count}, warmup_factor={warmup_factor:.4f}")
        else:
            # After warmup, ensure warmup_factor is 1.0
            for group in self.param_groups:
                if group['warmup_factor'] != 1.0:
                    group['warmup_factor'] = 1.0
                    group['lr'] = group['base_lr'] * group['q_scale']
            if self._step_count == self.warmup_steps:
                logging.info("Warmup completed, transitioning to Q-learning control")

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[torch.Tensor]:
        """Enhanced step function with Q-Learning based adaptive hyperparameter control."""
        # First, compute the loss
        loss = None
        if closure is not None:
            loss = closure()

        # Collect gradient statistics
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

        # Track learning rates and momentum for all parameter groups
        for group in self.param_groups:
            self.stats['learning_rates'].append(group['lr'])
            self.stats['momentum_values'].append(group['momentum'])

        # Track if gradients were seen this step
        saw_grads = len(grad_norms) > 0

        # Detect gradient anomalies
        is_anomalous = False
        if saw_grads:
            mean_grad_norm = np.mean(grad_norms)
            mean_grad_var = np.mean(grad_vars) if grad_vars else 0.0
            self.stats['grad_norms'].append(mean_grad_norm)
            self._track_gradient_memory(mean_grad_norm)

            is_anomalous = mean_grad_norm > self.max_grad_norm * 10 or \
                           np.isnan(mean_grad_norm) or \
                           np.isnan(mean_grad_var)

        # Compute entropy of gradients
        entropy = self._compute_entropy(torch.tensor(grad_vars)) if grad_vars else 0.0
        self.stats['entropy_values'].append(entropy)

        # Apply warmup with gradient clipping
        self._apply_warmup()

        # Apply Q-learning adjustments if we have gradients and loss
        if saw_grads and loss is not None:
            current_loss = loss.item()

            # If there is a previous state and action, compute reward and update Q-table
            if self.q_controller.prev_loss is not None and self.q_controller.prev_state is not None and self.q_controller.prev_action is not None:
                loss_improvement = self.q_controller.prev_loss - current_loss
                grad_health = 1.0 / (1.0 + mean_grad_var)
                stability = float(abs(loss_improvement) < self.q_controller.prev_loss * 0.1)

                # Check if actions resulted in no changes
                no_change = all(
                    action == 1.0 for action in self.q_controller.prev_action.values()
                )

                # Check for consistent improvement
                self.q_controller.consistent_improvement.append(1 if loss_improvement < 0 else 0)
                consistent_improvement = all(self.q_controller.consistent_improvement)

                # Compute reward for the previous action
                reward = self.q_controller.compute_reward(
                    loss_trend=stability,
                    grad_health=grad_health,
                    consistent_improvement=consistent_improvement,
                    no_change=no_change
                )

                # Get the current state for next state reference
                q_state = self.q_controller.get_state(
                    lr=self.param_groups[0]['lr'],
                    momentum=self.param_groups[0]['momentum'],
                    grad_var=mean_grad_var,
                    loss=current_loss
                )

                # Update Q-table with the previous state, action, and received reward
                self.q_controller.update(
                    state=self.q_controller.prev_state,
                    action=self.q_controller.prev_action,
                    reward=reward,
                    next_state=q_state
                )
            else:
                # If no previous state/action, initialize current state
                q_state = self.q_controller.get_state(
                    lr=self.param_groups[0]['lr'],
                    momentum=self.param_groups[0]['momentum'],
                    grad_var=mean_grad_var,
                    loss=current_loss
                )

            # Choose new action based on the current state
            q_action = self.q_controller.choose_action(q_state)

            # Determine if the new action results in no changes
            no_change = all(
                action == 1.0 for action in q_action.values()
            )

            # Apply learning rate adjustment
            lr_scale = float(np.clip(
                q_action['lr_scale'],
                self.q_controller.lr_scale_bounds[0],
                self.q_controller.lr_scale_bounds[1]
            ))

            # Apply momentum adjustment
            momentum_scale = float(np.clip(
                q_action['momentum_scale'],
                self.q_controller.momentum_scale_bounds[0],
                self.q_controller.momentum_scale_bounds[1]
            ))

            for group in self.param_groups:
                # Update q_scale cumulatively instead of overwriting lr
                group['q_scale'] *= lr_scale
                group['lr'] = group['base_lr'] * group['warmup_factor'] * group['q_scale']

                # Update momentum
                group['momentum'] = float(np.clip(
                    group['momentum'] * momentum_scale,
                    0.8,  # Minimum momentum
                    1.1   # Maximum momentum (adjusted for wider range)
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
                        f"lr_scale={lr_scale:.4f}, momentum_scale={momentum_scale:.4f}, "
                        f"mean_Q_lr={mean_q_lr:.4f}, mean_Q_momentum={mean_q_momentum:.4f}, "
                        f"effective_lr={group['lr']:.6f}, momentum={group['momentum']:.4f}"
                    )

            # Update Q-learning state for the next step
            self.q_controller.prev_state = q_state
            self.q_controller.prev_action = q_action
            self.q_controller.prev_loss = current_loss

            # If the new action results in no change, apply a penalty
            if saw_grads and self.q_controller.prev_action and no_change:
                logging.debug("No-change action detected. Applying penalty.")
                # The penalty is already incorporated in the reward function

        # Apply updates with the adjusted learning rates
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if is_anomalous:
                    grad = torch.clamp(grad, -1.0, 1.0)
                    logging.debug("Applied gradient clipping to prevent blowout.")

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
                    current_loss=loss.item() if loss is not None else 0.0
                )

        if saw_grads:
            self._step_count += 1
            # Adjust hyperparameters more frequently
            if self._step_count % 10 == 0:
                self._adjust_hyperparameters()
            # Log optimization statistics
            self.log_optimization_stats()

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

        # Apply weight decay with gentler scaling
        if weight_decay != 0:
            param_norm = torch.norm(p)
            # Reduced weight decay scaling using tanh for smoother transitions
            adaptive_wd = weight_decay * (0.5 + torch.tanh(param_norm * 0.1))
            grad = grad.add(p, alpha=adaptive_wd)
            logging.debug(f"Applied adaptive weight decay: {adaptive_wd:.6f}")

        # Apply gradient clipping before updating the momentum buffer
        grad_norm = torch.norm(grad)
        max_grad_norm = 10.0  # Increased from previous value
        if grad_norm > max_grad_norm:
            grad.mul_(max_grad_norm / (grad_norm + 1e-6))
            logging.debug(f"Clipped gradient norm from {grad_norm:.6f} to {max_grad_norm:.6f}")

        # Smoother momentum update
        buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
        logging.debug(f"Updated momentum buffer with grad: {grad_norm:.6f}, momentum: {momentum:.4f}")

        # Adaptive noise injection based on training progress
        noise_factor = 0.1 * np.exp(-self._step_count / 1000)  # Decay noise over time
        stability = 1.0 / (1.0 + np.var(list(self.grad_memory)) if self.grad_memory else 1.0)
        noise_scale = min(0.1, noise_factor * stability)  # Cap maximum noise
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
            clip_threshold = max(1.0, avg_update_norm * 5.0)  # More permissive clipping threshold

            if update_norm > clip_threshold:
                scale_factor = clip_threshold / (update_norm + 1e-6)
                # Apply soft clipping using tanh
                scale_factor = 0.5 * (1.0 + np.tanh(2.0 * (scale_factor - 0.5)))
                update.mul_(scale_factor)
                logging.debug(f"Soft clipped update norm from {update_norm:.6f} to {update_norm * scale_factor:.6f}")

        # Update statistics
        self.stats['update_norms'].append(float(update_norm))

        # Always apply gradient clipping as a failsafe
        max_failsafe_norm = 5.0
        if update_norm > max_failsafe_norm:
            scale_factor = max_failsafe_norm / (update_norm + 1e-6)
            update.mul_(scale_factor)
            logging.debug(f"Failsafe clipping applied: update norm reduced to {max_failsafe_norm:.6f} (scale factor {scale_factor:.4f})")

        # Apply gradient clipping on parameter update
        max_update_norm_final = 1.0  # Prevent too large updates
        update_norm_final = torch.norm(update).item()
        if update_norm_final > max_update_norm_final:
            update.mul_(max_update_norm_final / (update_norm_final + 1e-6))
            self.stats['update_norms'].append(float(update_norm_final))
            logging.debug(f"Final clipping applied: update norm from {update_norm_final:.6f} to {max_update_norm_final:.6f}")

        # Update parameters with smoothing for stability
        smooth_factor = min(1.0, self._step_count / 100.0)  # Faster warmup
        effective_lr = smooth_factor * (1.0 - momentum)
        p.data.add_(update, alpha=effective_lr)
        logging.debug(f"Applied parameter update with effective_lr={effective_lr:.6f}")

    def _compute_noise_scale(self, param_norm: float, update_norm: float, stability: float) -> float:
        """Compute adaptive noise scale based on parameter and update magnitudes and stability."""
        relative_update = update_norm / (param_norm + 1e-8)
        if relative_update < 1e-4:  # Very small updates
            return 2.0 * stability
        elif relative_update > 1e-2:  # Large updates
            return 0.5 * stability
        return 1.0 * stability

    def _detect_gradient_cliff(self, grad_norm: float) -> bool:
        """Detect sudden large changes in gradient norm."""
        if len(self.grad_memory) > 10:
            recent_norms = list(self.grad_memory)[-10:]
            mean_norm = np.mean(recent_norms)
            if grad_norm > 3.0 * mean_norm:
                logging.debug(f"Gradient norm {grad_norm:.4f} exceeds 3x mean {mean_norm:.4f}")
                return True
        return False

    def _scale_update_by_loss(self, update: torch.Tensor, loss: float) -> torch.Tensor:
        """Scale parameter updates based on current loss magnitude."""
        if loss > 5.0:  # High loss - larger updates
            scaled_update = update * 1.2
            logging.debug(f"Scaled update by 1.2 due to high loss: {loss:.4f}")
            return scaled_update
        elif loss < 1.0:  # Low loss - smaller updates
            scaled_update = update * 0.8
            logging.debug(f"Scaled update by 0.8 due to low loss: {loss:.4f}")
            return scaled_update
        return update

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

        for i, group in enumerate(self.param_groups):
            current_lr = group['lr']
            current_momentum = group['momentum']
            current_weight_decay = group.get('weight_decay', 0.01)

            original_lr = current_lr
            original_momentum = current_momentum
            original_weight_decay = current_weight_decay

            if is_unstable:
                # Reduce learning rate and increase momentum for stability
                group['lr'] = max(1e-8, current_lr * 0.9)
                group['momentum'] = min(1.1, current_momentum * 1.02)
                logging.info(
                    f"Param Group {i}: Adjusted LR from {original_lr:.6f} to {group['lr']:.6f} "
                    f"and Momentum from {original_momentum:.4f} to {group['momentum']:.4f} due to instability."
                )

            elif is_diverging:
                # Significantly reduce learning rate and momentum
                group['lr'] = max(1e-8, current_lr * 0.7)
                group['momentum'] = max(0.8, current_momentum * 0.9)
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
            group['momentum'] = max(0.8, group['momentum'])

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
        """Sophisticated weight decay adjustment based on gradient statistics."""
        base_weight_decay = param_group.get('weight_decay', 0.01)
        grad_norm = grad_stats.get('total_norm', 1.0)
        param_norm = torch.norm(torch.stack([p.norm() for p in param_group['params']])).item()

        # Adjust based on gradient-to-parameter ratio
        grad_param_ratio = grad_norm / (param_norm + 1e-8)

        if grad_param_ratio > 1.0:  # Gradients larger than parameters
            decay_factor = min(2.0, grad_param_ratio)
        elif grad_param_ratio < 0.1:  # Gradients much smaller than parameters
            decay_factor = max(0.5, grad_param_ratio * 5)
        else:
            decay_factor = 1.0

        adjusted_weight_decay = base_weight_decay * decay_factor
        logging.debug(f"Adjusted weight decay from {base_weight_decay:.6f} to {adjusted_weight_decay:.6f} "
                      f"based on grad_param_ratio={grad_param_ratio:.4f}")
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
            state_dict['epsilon'] = 0.3  # Default epsilon
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
            self.q_controller.epsilon = 0.3  # Default epsilon
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
            param_group.setdefault('warmup_factor', 1.0)
            param_group.setdefault('q_scale', 1.0)

            super().add_param_group(param_group)
            logging.info(f"Added parameter group with {len(param_group['params'])} parameters")

        except Exception as e:
            logging.error(f"Error adding parameter group: {e}")
            raise


# Example Usage (for testing purposes)
if __name__ == "__main__":
    # Dummy model
    model = torch.nn.Linear(10, 2)

    # Initialize EnhancedSGD
    optimizer = EnhancedSGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=0.01,
        warmup_steps=50  # Short warmup
    )

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
