# EnhancedSGD_innovative_stable.py

import torch
from torch.optim import Optimizer
from collections import deque
import logging
from typing import Iterable, Optional, Dict, Any, List, Tuple, Union
import numpy as np
import random
import math
import os # Added for QController seed
import itertools # Added for Q-Table pruning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s', # Consistent format
    force=True # Override any root logger configs set by libraries
)

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each set of scores in x (needed for adaptive exploration)."""
    if x.size == 0: return np.array([])
    # Ensure input is float for exponentiation
    x_float = x.astype(np.float64)
    e_x = np.exp(x_float - np.max(x_float)) # Subtract max for numerical stability
    return e_x / (e_x.sum() + 1e-9) # Add epsilon for division stability


class GradientStats:
    """Tracks gradient statistics for reporting (Unchanged from stabilized version)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_gradients = 0
        self.clipped_gradients = 0
        self.max_gradient_norm = 0.0 # Use float
        self.sum_clip_ratios = 0.0 # Use float
        self.step_stats = {}

    def record_gradient(self, original_norm: float, clipped: bool, clip_ratio: Optional[float] = None):
        self.total_gradients += 1
        self.max_gradient_norm = max(self.max_gradient_norm, original_norm)
        if clipped:
            self.clipped_gradients += 1
            self.sum_clip_ratios += (clip_ratio if clip_ratio is not None else 0.0)

    def get_step_stats(self) -> dict:
        if self.total_gradients == 0:
            return {
                "gradients_clipped": 0,
                "total_gradients": 0,
                "clip_ratio_avg": 0.0,
                "max_gradient": 0.0,
                "clip_percentage": 0.0
            }

        clip_percentage = (self.clipped_gradients / self.total_gradients) * 100 if self.total_gradients > 0 else 0.0
        avg_clip_ratio = self.sum_clip_ratios / self.clipped_gradients if self.clipped_gradients > 0 else 0.0

        return {
            "gradients_clipped": self.clipped_gradients,
            "total_gradients": self.total_gradients,
            "clip_ratio_avg": avg_clip_ratio,
            "max_gradient": self.max_gradient_norm,
            "clip_percentage": clip_percentage
        }

    def record_step(self, step: int):
        stats = self.get_step_stats()
        self.step_stats[step] = stats
        # Don't reset here, reset happens in EnhancedSGD step after logging
        return stats


class ImprovedQLearningController:
    """ Merged Q-Learning Controller: Retains stable state/update but adds back features."""

    def __init__(
        self,
        learning_rate: float = 0.02,
        discount: float = 0.97,
        epsilon: float = 0.15,
        epsilon_decay: float = 0.9995, # Use stable decay
        min_epsilon: float = 0.02,    # Use stable min epsilon
        lr_scale_bounds: tuple = (0.85, 1.15), # Use stable bounds
        momentum_scale_bounds: tuple = (0.9, 1.1), # Use stable bounds
        min_weight_decay: float = 1e-4, # Keep min_weight_decay concept
        state_memory_size: int = 300,
        max_q_table_size: int = 15000,
        success_decay: float = 0.99 # Decay for action success tracking
    ):
        self.q_table = {}
        self.alpha = learning_rate # Q-learning rate
        self.gamma = discount      # Q-learning discount factor
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay # Use this for decay logic now
        self.lr_scale_bounds = lr_scale_bounds
        self.momentum_scale_bounds = momentum_scale_bounds
        self.min_weight_decay = min_weight_decay

        # Reintroduce state memory for reward/exploration logic
        self.state_memory = deque(maxlen=state_memory_size)

        # State representation windows (from stable version)
        self.loss_window = deque(maxlen=15)
        self.grad_var_window = deque(maxlen=15)
        self.lr_window = deque(maxlen=8)
        self.momentum_window = deque(maxlen=8)

        # More granular action space (from old 'Improved' version)
        self.action_ranges = {
            'lr_scale': np.array([0.85, 0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1, 1.15]),
            'momentum_scale': np.array([0.9, 0.95, 0.975, 0.99, 1.0, 1.01, 1.025, 1.05, 1.075, 1.1])
        }

        # Reintroduce success tracking (from old 'Improved' version)
        self.action_success = {k: np.zeros(len(v)) for k, v in self.action_ranges.items()}
        self.action_counts = {k: np.zeros(len(v)) for k, v in self.action_ranges.items()}
        self.success_decay = success_decay

        # Performance tracking (from stable version)
        self.performance_window = deque(maxlen=50)
        self.stable_steps = 0

        # Q-Table memory management (access count based, from stable version)
        self.max_q_table_size = max_q_table_size
        self.q_table_access_count = {}

        # Keep track of previous state/action/loss (from stable version)
        self.prev_loss = None
        self.prev_state = None
        self.prev_action = None

    def get_state(self, lr: float, momentum: float, grad_var: float, loss: float) -> tuple:
        """State representation using stable version's logic."""
        # --- Same as stable QController ---
        self.loss_window.append(loss)
        self.grad_var_window.append(grad_var)
        self.lr_window.append(lr)
        self.momentum_window.append(momentum)

        # Loss trend binning (simplified)
        loss_trend_bin = 2 # Default: stable
        if len(self.loss_window) >= 5:
            y = np.array(list(self.loss_window)[-5:])
            x = np.arange(len(y))
            try:
                if np.all(y == y[0]): slope = 0.0
                else: slope = np.polyfit(x, y, 1)[0]
                if np.isfinite(slope):
                     avg_loss = np.mean(y) + 1e-6
                     normalized_slope = slope / avg_loss
                     loss_trend_bin = np.digitize(normalized_slope, bins=[-0.05, -0.005, 0.005, 0.05])
            except (np.linalg.LinAlgError, ValueError): loss_trend_bin = 2

        # Gradient variance binning
        grad_var_bin = 2 # Default: medium variance
        if self.grad_var_window:
            median_grad_var = np.median(list(self.grad_var_window))
            if np.isfinite(median_grad_var):
                 grad_var_bin = np.digitize(median_grad_var, bins=[1e-5, 1e-3, 0.1, 1.0])

        # Learning rate binning
        lr_bin = np.digitize(lr, bins=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
        # Momentum binning
        momentum_bin = np.digitize(momentum, bins=[0.85, 0.9, 0.95, 0.99])
        # --- End stable state logic ---

        state = (loss_trend_bin, grad_var_bin, lr_bin, momentum_bin)
        # Track access count (stable version's logic)
        self.q_table_access_count[state] = self.q_table_access_count.get(state, 0) + 1
        return state

    def compute_reward(
        self,
        loss_improvement: float, # Relative improvement: (prev_loss - current_loss) / (abs(prev_loss) + 1e-9)
        grad_health: float, # Inverse relationship with variance: 1 / (1 + grad_var)
        consistent_improvement: bool # Flag indicating recent positive trend
    ) -> float:
        """ Enhanced reward computation with stability bonus (closer to old 'Improved')."""
        # Base reward from relative loss improvement (stable version's scale)
        base_reward = 2.0 * loss_improvement if loss_improvement > 0 else 1.0 * loss_improvement

        # Stability bonus based on gradient health (more like old 'Improved' logic)
        stability_threshold = 0.8 # Health threshold
        stability_bonus = 0.0
        if grad_health > stability_threshold:
            # Bonus scales from 0 to 0.4 as health goes from threshold to 1.0
            bonus_scale = (grad_health - stability_threshold) / (1.0 - stability_threshold + 1e-9)
            stability_bonus = 0.4 * bonus_scale

        # Consistency reward/penalty (from stable version)
        consistency_reward = 0.0
        if consistent_improvement:
            self.stable_steps += 1
            consistency_reward = min(0.3, 0.05 * math.log1p(self.stable_steps)) # Capped log bonus
        else:
            self.stable_steps = 0 # Reset streak

        # Combine rewards
        combined_reward = base_reward + stability_bonus + consistency_reward

        # Smooth reward scaling (tanh)
        scaled_reward = np.tanh(combined_reward) # Bound between -1 and 1

        # Track performance (stable version)
        self.performance_window.append(scaled_reward)

        return float(scaled_reward)

    def choose_action(self, state: tuple) -> Dict[str, float]:
        """ Enhanced action selection with adaptive exploration (merged logic)."""
        if state not in self.q_table:
            self.q_table[state] = {
                param: np.zeros(len(space))
                for param, space in self.action_ranges.items()
            }
            self._manage_q_table_size() # Prune if needed

        action = {}

        # --- Adaptive Epsilon Decay (based on performance window) ---
        if len(self.performance_window) >= 20: # Need enough history
            avg_performance = np.mean(list(self.performance_window)[-20:]) # Use last 20 rewards
            # Decay faster if performance is poor, slower if good
            decay_factor = self.epsilon_decay
            if avg_performance < -0.1: decay_factor = 0.998 # Faster decay
            elif avg_performance > 0.2: decay_factor = 0.9998 # Slower decay
            self.epsilon = max(self.min_epsilon, self.epsilon * decay_factor)
        else:
            # Standard decay if not enough history
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        # --- End Adaptive Epsilon ---

        # --- Adaptive Exploration Probability ---
        # Calculate probability of exploring based on current epsilon
        explore_prob = self.epsilon
        # Optional: Slightly reduce exploration if recent performance is very good
        # if len(self.performance_window) >= 10 and np.mean(list(self.performance_window)[-10:]) > 0.3:
        #     explore_prob *= 0.8
        # --- End Adaptive Exploration Prob ---


        for param, space in self.action_ranges.items():
            if random.random() < explore_prob:
                # --- Smart Exploration (using success history) ---
                success_rates = self.action_success[param] / (self.action_counts[param] + 1e-6) # Avoid div by zero

                # Option 1: Softmax sampling based on success
                # Add small temp to encourage exploring less successful actions occasionally
                exploration_temp = 0.1
                probabilities = softmax(success_rates / exploration_temp)

                # Option 2: Bias towards successful actions but allow others
                # probabilities = success_rates + 0.05 # Add small base probability
                # probabilities = probabilities / (probabilities.sum() + 1e-9)

                if probabilities.sum() > 1e-9 and np.all(np.isfinite(probabilities)): # Check validity
                    chosen_idx = np.random.choice(len(space), p=probabilities)
                else:
                    # Fallback to uniform random if probabilities are invalid
                    logging.warning(f"Invalid probabilities for exploration in {param}, using uniform.")
                    chosen_idx = random.randrange(len(space))
                # --- End Smart Exploration ---
            else:
                # --- Exploitation (Greedy based on Q-values, tie-break towards 1.0) ---
                q_values = self.q_table[state][param]
                if len(q_values) > 0 and np.any(np.isfinite(q_values)):
                     max_q = np.nanmax(q_values)
                     # Find all indices with Q-value close to maximum
                     best_indices = np.where(np.abs(q_values - max_q) < 1e-6)[0]
                     if len(best_indices) > 0:
                          # Tie-breaking: choose action closest to 1.0 among the best
                          chosen_idx = min(best_indices, key=lambda i: abs(space[i] - 1.0))
                     else: # Fallback if all are NaN or some other issue
                          chosen_idx = random.randrange(len(space))
                else: # Fallback if q_values is empty or all non-finite
                     chosen_idx = random.randrange(len(space))
                # --- End Exploitation ---

            action[param] = float(space[chosen_idx])

        return action

    def update(self, state: tuple, action: Dict[str, float], reward: float, next_state: Optional[tuple], should_log: bool = False):
        """ Q-learning update using stable version's core logic, adds success tracking."""
        # Record experience in state memory (for adaptive exploration/reward)
        self.state_memory.append((state, action, reward))

        # Initialize Q-values if needed (stable version's logic)
        if next_state is not None and next_state not in self.q_table:
            self.q_table[next_state] = {
                param: np.zeros(len(space))
                for param, space in self.action_ranges.items()
            }
            self._manage_q_table_size() # Prune if needed

        # Update Q-values (stable version's core update logic)
        for param, value in action.items():
            space = self.action_ranges[param]
            try:
                action_idx = np.abs(space - value).argmin()
                if not np.isclose(space[action_idx], value): raise ValueError("Action value not found precisely.")
            except ValueError:
                logging.warning(f"Q-update: Action value {value} not found for {param}. Skipping update.")
                continue

            # Get max future Q-value
            max_future_q = 0.0
            if next_state is not None and next_state in self.q_table:
                next_q_values = self.q_table[next_state][param]
                if len(next_q_values) > 0 and np.any(np.isfinite(next_q_values)):
                     max_future_q = np.nanmax(next_q_values)

            current_q = self.q_table[state][param][action_idx]
            td_target = reward + self.gamma * max_future_q
            td_error = td_target - current_q

            # --- Adaptive Learning Rate (based on visit count, from old 'Improved') ---
            self.action_counts[param][action_idx] += 1
            visit_count = self.action_counts[param][action_idx]
            # Decay LR for specific state-action pairs that have been visited often
            effective_alpha = self.alpha / (1 + math.log1p(visit_count * 0.1)) # Log decay based on visits
            effective_alpha = max(effective_alpha, 0.001) # Minimum alpha
            # --- End Adaptive Learning Rate ---

            # Update Q-value using effective alpha
            self.q_table[state][param][action_idx] += effective_alpha * td_error

            # --- Update Success Tracking ---
            if reward > 0.1: # Use a threshold to count success
                # Decay old successes and add new one
                self.action_success[param][action_idx] = (self.action_success[param][action_idx] * self.success_decay) + 1
            else:
                # Only decay if not successful
                self.action_success[param][action_idx] *= self.success_decay
            # --- End Success Tracking ---

            if should_log:
                logging.info(f"Q-Update state={state}, param={param}, action_idx={action_idx}: "
                             f"Q={current_q:.3f} -> {self.q_table[state][param][action_idx]:.3f} | "
                             f"R={reward:.3f} | FutQ={max_future_q:.3f} | TD_err={td_error:.3f} | EffAlpha={effective_alpha:.4f}")


    def _manage_q_table_size(self):
        """ Prunes Q-table using access counts (stable version's logic)."""
        # --- Same as stable QController ---
        if len(self.q_table) > self.max_q_table_size:
            try:
                if self.q_table_access_count:
                     sorted_states = sorted(self.q_table_access_count.items(), key=lambda item: item[1])
                     num_to_remove = len(self.q_table) - int(self.max_q_table_size * 0.9)
                     num_removed = 0
                     for state, count in sorted_states:
                         if num_removed >= num_to_remove: break
                         if state in self.q_table: del self.q_table[state]
                         if state in self.q_table_access_count: del self.q_table_access_count[state]
                         num_removed += 1
                     logging.info(f"Pruned {num_removed} states from Q-table (new size: {len(self.q_table)}).")
            except (ValueError, KeyError) as e:
                logging.warning(f"Could not prune Q-table: {e}")
        # --- End stable pruning logic ---


class EnhancedSGD(Optimizer):
    """ Merged EnhancedSGD: Integrates adaptive adjustments with stability fixes."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.003,
        momentum: float = 0.9,
        weight_decay: float = 0.005,
        max_grad_norm: float = 1.0,
        # Reintroduce parameters for adaptive features if needed
        noise_scale_initial: float = 0.001, # Initial noise scale
        noise_decay_steps: int = 500, # Steps over which noise decays
        update_clip_factor: float = 5.0, # Multiplier for avg update norm clipping threshold
        adjust_interval: int = 50, # How often to run _adjust_hyperparameters
        lr_scale_bounds: tuple = (0.85, 1.15), # Q-Controller bounds
        momentum_scale_bounds: tuple = (0.9, 1.1), # Q-Controller bounds
        q_learning_config: Dict[str, Any] = {},
        **kwargs
    ):
        # --- Parameter group setup (from stable version) ---
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            param_groups = params
        else:
            if not isinstance(params, Iterable): params = list(params)
            param_groups = [{'params': params}]

        for group in param_groups:
            group.setdefault('lr', lr)
            group.setdefault('momentum', momentum)
            group.setdefault('weight_decay', weight_decay)
            group.setdefault('base_lr', lr)
            group.setdefault('q_scale', 1.0)
            # Add min_weight_decay concept from QController
            group.setdefault('min_weight_decay', q_learning_config.get('min_weight_decay', 1e-4))
        # --- End stable param group setup ---

        super().__init__(param_groups, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

        # --- Initialize optimization state ---
        self._init_optimization_state(
            max_grad_norm=max_grad_norm,
            noise_scale_initial=noise_scale_initial,
            noise_decay_steps=noise_decay_steps,
            update_clip_factor=update_clip_factor,
            adjust_interval=adjust_interval,
            lr_scale_bounds=lr_scale_bounds,
            momentum_scale_bounds=momentum_scale_bounds,
            q_learning_config=q_learning_config,
            **kwargs
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _init_optimization_state(self, **kwargs):
        """Initialize optimization state, including adaptive feature parameters."""
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        self.noise_scale_initial = kwargs.get('noise_scale_initial', 0.001)
        self.noise_decay_steps = kwargs.get('noise_decay_steps', 500)
        self.update_clip_factor = kwargs.get('update_clip_factor', 5.0)
        self.adjust_interval = kwargs.get('adjust_interval', 50) # Store adjust interval
        lr_scale_bounds = kwargs.get('lr_scale_bounds', (0.85, 1.15))
        momentum_scale_bounds = kwargs.get('momentum_scale_bounds', (0.9, 1.1))
        q_learning_config = kwargs.get('q_learning_config', {})

        # Initialize QController (now ImprovedQLearningController)
        self.q_controller = ImprovedQLearningController(
            learning_rate=q_learning_config.get('learning_rate', 0.02),
            discount=q_learning_config.get('discount', 0.97),
            epsilon=q_learning_config.get('epsilon', 0.15),
            epsilon_decay=q_learning_config.get('epsilon_decay', 0.9995),
            min_epsilon=q_learning_config.get('min_epsilon', 0.02),
            lr_scale_bounds=lr_scale_bounds,
            momentum_scale_bounds=momentum_scale_bounds,
            min_weight_decay=q_learning_config.get('min_weight_decay', 1e-4),
            max_q_table_size=q_learning_config.get('max_q_table_size', 15000),
            success_decay=q_learning_config.get('success_decay', 0.99) # Pass success decay
        )

        self._step_count = 0
        # GradientStats (from stable version)
        self.gradient_stats = GradientStats()
        # Reintroduce self.stats deque for adaptive adjustments
        self.stats = {
            'grad_norms': deque(maxlen=100), # Norms of gradients before clipping
            'update_norms': deque(maxlen=100) # Norms of final updates applied to parameters
        }

    # --- Robust Gradient Stats Calculation (from stable version) ---
    def _get_gradient_stats(self) -> Dict[str, Any]:
        """Gather gradient statistics, checking for finite values."""
        # --- Same as stable EnhancedSGD ---
        grad_norms = []
        grad_vars = []
        num_finite_params = 0
        num_non_finite_params = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.detach()
                if torch.isfinite(grad).all():
                    grad_float = grad.float()
                    norm = torch.norm(grad_float).item()
                    grad_norms.append(norm)
                    if grad_float.numel() > 1: grad_vars.append(torch.var(grad_float).item())
                    num_finite_params += 1
                else: num_non_finite_params += 1

        saw_grads = num_finite_params + num_non_finite_params > 0
        saw_finite_grads = num_finite_params > 0
        if saw_finite_grads:
            mean_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
            mean_grad_var = np.mean(grad_vars) if grad_vars else 0.0
        else: mean_grad_norm, mean_grad_var = 0.0, 0.0

        is_norm_finite = np.isfinite(mean_grad_norm)
        is_var_finite = np.isfinite(mean_grad_var)
        return {
            'saw_grads': saw_grads, 'saw_finite_grads': saw_finite_grads,
            'mean_grad_norm': mean_grad_norm if is_norm_finite else 0.0,
            'mean_grad_var': mean_grad_var if is_var_finite else float('inf'),
            'num_non_finite_params': num_non_finite_params,
            'is_norm_finite': is_norm_finite, 'is_var_finite': is_var_finite,
            # Add raw norms list for _adjust_hyperparameters
            '_raw_grad_norms': grad_norms
        }
        # --- End stable gradient stats ---

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[torch.Tensor]:
        """ Performs optimization step, integrating Q-learning and adaptive adjustments. """
        loss = None
        if closure is not None:
            try:
                with torch.enable_grad(): loss = closure()
            except Exception as e:
                logging.error(f"Error in closure execution: {e}", exc_info=True); return None

        # Get gradient statistics
        grad_stats = self._get_gradient_stats()

        # --- Q-Learning Update (adapted from stable version) ---
        if grad_stats['saw_finite_grads'] and loss is not None and torch.isfinite(loss):
            current_loss = loss.item()
            safe_grad_var = grad_stats['mean_grad_var'] if grad_stats['is_var_finite'] else 0.0

            q_state = self.q_controller.get_state(
                lr=self.param_groups[0]['lr'], momentum=self.param_groups[0]['momentum'],
                grad_var=safe_grad_var, loss=current_loss
            )

            if self.q_controller.prev_loss is not None and \
               self.q_controller.prev_state is not None and \
               self.q_controller.prev_action is not None:
                if np.isfinite(self.q_controller.prev_loss) and abs(self.q_controller.prev_loss) > 1e-9:
                    loss_improvement = (self.q_controller.prev_loss - current_loss) / abs(self.q_controller.prev_loss + 1e-9)
                    grad_health = 1.0 / (1.0 + max(0, safe_grad_var))
                    consistent_improvement = all([r > -0.01 for r in list(self.q_controller.performance_window)[-10:]])
                    reward = self.q_controller.compute_reward(loss_improvement, grad_health, consistent_improvement)
                    if np.isfinite(reward):
                        self.q_controller.update(self.q_controller.prev_state, self.q_controller.prev_action, reward, q_state, should_log=(self._step_count % 50 == 0))
                    else: logging.warning(f"Step {self._step_count}: Skipping Q-update due to non-finite reward.")
                else: logging.warning(f"Step {self._step_count}: Skipping Q-update due to non-finite or zero previous loss.")

            q_action = self.q_controller.choose_action(q_state)

            for group in self.param_groups:
                group['q_scale'] *= float(np.clip(q_action['lr_scale'], self.q_controller.lr_scale_bounds[0], self.q_controller.lr_scale_bounds[1]))
                min_lr, max_lr = 1e-7, 0.01
                group['lr'] = float(np.clip(group['base_lr'] * group['q_scale'], min_lr, max_lr))
                group['momentum'] = float(np.clip(group['momentum'] * q_action['momentum_scale'], self.q_controller.momentum_scale_bounds[0], self.q_controller.momentum_scale_bounds[1]))

            self.q_controller.prev_state = q_state
            self.q_controller.prev_action = q_action
            self.q_controller.prev_loss = current_loss
        # --- End Q-Learning Update ---

        # --- Parameter Update Loop ---
        num_params_updated = 0
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            min_weight_decay = group['min_weight_decay'] # Get min WD for this group

            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                updated = self._apply_update(
                    p=p, grad_in=p.grad, momentum=momentum, lr=lr,
                    weight_decay=weight_decay, min_weight_decay=min_weight_decay, # Pass min WD
                    state=state
                )
                if updated: num_params_updated += 1
        # --- End Parameter Update Loop ---

        if num_params_updated > 0:
             self._step_count += 1

             # --- Call _adjust_hyperparameters periodically ---
             if self._step_count % self.adjust_interval == 0:
                 self._adjust_hyperparameters(grad_stats) # Pass current grad_stats
             # --- End Adjust Hyperparameters ---

             # Log gradient stats (using GradientStats)
             grad_clip_stats = self.gradient_stats.get_step_stats() # Get stats for the step
             self.gradient_stats.reset() # Reset for next step

             if self._step_count % 50 == 0: # Log less frequently
                 lr_scale = self.q_controller.prev_action.get('lr_scale', 1.0) if self.q_controller.prev_action else 1.0
                 mom_scale = self.q_controller.prev_action.get('momentum_scale', 1.0) if self.q_controller.prev_action else 1.0
                 current_lr = self.param_groups[0]['lr']
                 current_mom = self.param_groups[0]['momentum']
                 logging.info(
                     f"Step {self._step_count} | "
                     f"LR: {current_lr:.2e} (QScale:{lr_scale:.2f}) | Mom: {current_mom:.3f} (QScale:{mom_scale:.2f}) | "
                     f"Grad Clip: {grad_clip_stats['gradients_clipped']}/{grad_clip_stats['total_gradients']} ({grad_clip_stats['clip_percentage']:.1%}) | "
                     f"Max Grad: {grad_clip_stats['max_gradient']:.3f}"
                 )

        elif grad_stats['saw_grads']:
             logging.warning(f"Step {self._step_count}: Gradients present but no parameters updated.")

        return loss

    # --- Safer Parameter Update (incorporating noise/clipping from old version) ---
    def _apply_update(
        self,
        p: torch.Tensor, grad_in: torch.Tensor, momentum: float, lr: float,
        weight_decay: float, min_weight_decay: float, state: dict # Added min_weight_decay
    ) -> bool:
        # Check for non-finite gradient (stable version)
        if not torch.isfinite(grad_in).all():
            grad = torch.zeros_like(grad_in)
        else:
            grad = grad_in.clone()

        # Initialize state (stable version)
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        if 'update_history' not in state: # For dynamic update scaling
             state['update_history'] = deque(maxlen=5)

        buf = state['momentum_buffer']

        # --- Adaptive Weight Decay (closer to old logic, using min_wd) ---
        if weight_decay != 0 and torch.isfinite(grad).all():
            param_norm = torch.norm(p).item()
            # Sigmoid scaling, but ensure it doesn't go below min_weight_decay
            adaptive_wd_factor = 1.0 / (1.0 + math.exp(-param_norm * 0.1)) # Gentler scaling factor than before
            effective_wd = base_weight_decay = weight_decay # Start with base WD for group
            effective_wd = max(min_weight_decay, base_weight_decay * adaptive_wd_factor)
            grad = grad.add(p, alpha=effective_wd)
        # --- End Adaptive WD ---

        # Gradient Clipping (stable version's GradientStats recording)
        grad_norm_before_clip = torch.norm(grad).item()
        was_clipped = False
        clip_ratio = 1.0
        if grad_norm_before_clip > self.max_grad_norm:
            clip_ratio = self.max_grad_norm / (grad_norm_before_clip + 1e-6)
            grad.mul_(clip_ratio)
            was_clipped = True
        self.gradient_stats.record_gradient(grad_norm_before_clip, was_clipped, clip_ratio if was_clipped else 1.0)
        # Track raw norms for _adjust_hyperparameters
        if torch.isfinite(grad_norm_before_clip): self.stats['grad_norms'].append(grad_norm_before_clip)


        # Momentum buffer update (standard momentum, from stable version)
        buf.mul_(momentum).add_(grad)

        # --- Adaptive Noise Injection (reintroduced from old version) ---
        # Decay noise over initial steps
        current_noise_scale = 0.0
        if self._step_count < self.noise_decay_steps:
             decay_progress = self._step_count / self.noise_decay_steps
             # Exponential decay for noise scale
             current_noise_scale = self.noise_scale_initial * math.exp(-5.0 * decay_progress)
             # Optional: Modulate by gradient variance (if grads are stable, less noise)
             # grad_var = np.var(list(self.stats['grad_norms'])) if len(self.stats['grad_norms']) > 1 else 0.0
             # stability_factor = 1.0 / (1.0 + math.sqrt(grad_var))
             # current_noise_scale *= stability_factor

             if current_noise_scale > 1e-7: # Only add noise if scale is meaningful
                 noise = torch.randn_like(buf) * current_noise_scale
                 buf.add_(noise)
        # --- End Noise Injection ---

        # Calculate update step (Negated LR * buffer)
        update = buf.mul(-lr) # Calculate update: -lr * momentum_buffer

        # --- Dynamic Update Clipping (reintroduced from old version) ---
        update_norm = torch.norm(update).item()
        if np.isfinite(update_norm):
             self.stats['update_norms'].append(update_norm) # Track norm of calculated update

             # Clip based on average historical update norm
             if len(self.stats['update_norms']) > 10: # Need some history
                 avg_update_norm = np.mean(list(self.stats['update_norms']))
                 clip_threshold = avg_update_norm * self.update_clip_factor # e.g., 5x average update

                 if update_norm > clip_threshold:
                      scale_factor = clip_threshold / (update_norm + 1e-6)
                      update.mul_(scale_factor)
                      # Log clipping event less frequently
                      # if self._step_count % 100 == 0:
                      #      logging.info(f"Dynamic update clipping applied: Factor={scale_factor:.3f}, Threshold={clip_threshold:.3e}")

             # Failsafe clipping (hard limit)
             max_failsafe_norm = 10.0 # Generous failsafe limit
             if update_norm > max_failsafe_norm:
                 scale_factor = max_failsafe_norm / (update_norm + 1e-6)
                 update.mul_(scale_factor)
                 logging.warning(f"Failsafe update clipping triggered! Factor={scale_factor:.3f}")
        else:
            logging.warning(f"Non-finite update norm ({update_norm}) detected before applying update. Skipping update for safety.")
            return False # Don't apply non-finite update
        # --- End Dynamic Update Clipping ---


        # Apply final update (stable version's method)
        p.add_(update) # p = p + update (update already contains -lr)

        return True


    # --- Reintroduce _adjust_hyperparameters ---
    def _adjust_hyperparameters(self, current_grad_stats: Dict[str, Any]) -> None:
        """ Adaptively adjust optimizer hyperparameters based on recent training statistics. """
        if len(self.stats.get('grad_norms', [])) < 50 or len(self.stats.get('update_norms', [])) < 50:
            logging.debug("Skipping hyperparameter adjustment: Insufficient statistics.")
            return # Need enough history

        # Use grad norms collected over the last interval
        recent_grad_norms = list(self.stats['grad_norms'])
        recent_update_norms = list(self.stats['update_norms'])

        grad_norm_mean = np.mean(recent_grad_norms)
        grad_norm_std = np.std(recent_grad_norms)
        update_norm_mean = np.mean(recent_update_norms)

        # Check stability conditions
        grad_coeff_var = grad_norm_std / (grad_norm_mean + 1e-8) if grad_norm_mean > 1e-8 else 0
        is_unstable = grad_coeff_var > 0.4 # If coefficient of variation is high
        is_diverging = grad_norm_mean > self.max_grad_norm * 3.0 # If avg grad norm is much larger than clip value
        updates_too_small = update_norm_mean < 1e-6 # If parameters aren't changing much

        logging.info(f"Adjusting Hyperparameters (Step {self._step_count}): Grad Mean={grad_norm_mean:.3f}, "
                     f"Grad Std={grad_norm_std:.3f}, Grad CoeffVar={grad_coeff_var:.3f}, Update Mean={update_norm_mean:.3e}")

        for i, group in enumerate(self.param_groups):
            original_lr = group['lr']
            original_momentum = group['momentum']
            original_wd = group['weight_decay']
            lr_changed, mom_changed, wd_changed = False, False, False

            if is_unstable or is_diverging:
                # Reduce learning rate and increase momentum (more cautious adjustments)
                factor = 0.85 if is_diverging else 0.95 # More drastic cut if diverging
                new_lr = max(1e-7, original_lr * factor)
                if new_lr < original_lr: group['lr'] = new_lr; lr_changed = True

                new_mom = min(0.99, original_momentum / 0.98) # Increase momentum slowly
                if new_mom > original_momentum: group['momentum'] = new_mom; mom_changed = True

                # Optionally increase weight decay slightly if unstable
                if is_unstable:
                     new_wd = original_wd * 1.05
                     group['weight_decay'] = new_wd; wd_changed = True

                reason = "divergence" if is_diverging else "instability"
                if lr_changed or mom_changed or wd_changed:
                     logging.warning(f"Group {i}: Adjusting for {reason}. "
                                     f"{'LR:'+f'{original_lr:.2e} -> {group["lr"]:.2e}' if lr_changed else ''} "
                                     f"{'Mom:'+f'{original_momentum:.3f} -> {group["momentum"]:.3f}' if mom_changed else ''} "
                                     f"{'WD:'+f'{original_wd:.2e} -> {group["weight_decay"]:.2e}' if wd_changed else ''}")

            elif updates_too_small:
                # Increase learning rate cautiously, decrease momentum slightly
                new_lr = min(0.01, original_lr * 1.05) # Careful increase, capped
                if new_lr > original_lr: group['lr'] = new_lr; lr_changed = True

                new_mom = max(0.85, original_momentum * 0.99) # Slight decrease
                if new_mom < original_momentum: group['momentum'] = new_mom; mom_changed = True

                # Decrease weight decay slightly
                new_wd = original_wd * 0.95
                group['weight_decay'] = max(group['min_weight_decay'], new_wd); wd_changed = True


                if lr_changed or mom_changed or wd_changed:
                     logging.info(f"Group {i}: Adjusting for small updates. "
                                  f"{'LR:'+f'{original_lr:.2e} -> {group["lr"]:.2e}' if lr_changed else ''} "
                                  f"{'Mom:'+f'{original_momentum:.3f} -> {group["momentum"]:.3f}' if mom_changed else ''} "
                                  f"{'WD:'+f'{original_wd:.2e} -> {group["weight_decay"]:.2e}' if wd_changed else ''}")

            # --- Adaptive Weight Decay Adjustment (based on grad norm) ---
            # This is simpler than the complex logic in the old version's _adjust_weight_decay
            # If gradients are large, increase WD slightly; if small, decrease slightly
            wd_adjust_factor = 1.0 + 0.1 * math.tanh(math.log1p(grad_norm_mean) - math.log1p(self.max_grad_norm)) # Compare log norm to max_grad_norm
            wd_adjust_factor = np.clip(wd_adjust_factor, 0.8, 1.2) # Limit adjustment factor
            new_wd = group['weight_decay'] * wd_adjust_factor
            new_wd = max(group['min_weight_decay'], new_wd) # Ensure min WD
            if abs(new_wd - group['weight_decay']) > 1e-7:
                 if not wd_changed: # Avoid logging twice if changed above
                     logging.info(f"Group {i}: Adjusting WD based on grad norm: {group['weight_decay']:.2e} -> {new_wd:.2e} (Factor: {wd_adjust_factor:.2f})")
                 group['weight_decay'] = new_wd
            # --- End Adaptive WD Adjustment ---

            # Reset Q-scale if LR was adjusted by this method, so Q-controller starts fresh
            if lr_changed:
                group['q_scale'] = 1.0
                group['lr'] = max(1e-8, group['lr']) # Ensure LR bounds after adjustment

    # --- End _adjust_hyperparameters ---


    def state_dict(self) -> Dict[str, Any]:
        """ Returns the optimizer's state dict, including adaptive state. """
        state_dict = super().state_dict()
        try:
            # Q-Controller state (from stable version)
            state_dict['q_table'] = self.q_controller.q_table
            state_dict['q_controller_epsilon'] = float(self.q_controller.epsilon)
            state_dict['q_controller_prev_loss'] = self.q_controller.prev_loss
            state_dict['q_controller_prev_state'] = self.q_controller.prev_state
            state_dict['q_controller_prev_action'] = self.q_controller.prev_action
            state_dict['q_controller_access_count'] = self.q_controller.q_table_access_count
            # Add Q-controller success/count tracking state
            state_dict['q_controller_action_success'] = self.q_controller.action_success
            state_dict['q_controller_action_counts'] = self.q_controller.action_counts

            state_dict['_step_count'] = self._step_count
            # Save the stats deques (convert to lists)
            state_dict['stats_grad_norms'] = list(self.stats['grad_norms'])
            state_dict['stats_update_norms'] = list(self.stats['update_norms'])
        except Exception as e:
            logging.error(f"Error creating EnhancedSGD state dict: {e}", exc_info=True)
            # Provide defaults for Q-Controller state
            state_dict['q_table'] = {}
            state_dict['q_controller_epsilon'] = 0.15
            state_dict['q_controller_action_success'] = {k: np.zeros(len(v)) for k, v in self.q_controller.action_ranges.items()}
            state_dict['q_controller_action_counts'] = {k: np.zeros(len(v)) for k, v in self.q_controller.action_ranges.items()}
            # Provide defaults for stats
            state_dict['stats_grad_norms'] = []
            state_dict['stats_update_norms'] = []
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """ Loads optimizer state, including adaptive state. """
        # Pop custom state elements before calling super
        q_table = state_dict.pop('q_table', None)
        epsilon = state_dict.pop('q_controller_epsilon', None)
        prev_loss = state_dict.pop('q_controller_prev_loss', None)
        prev_state = state_dict.pop('q_controller_prev_state', None)
        prev_action = state_dict.pop('q_controller_prev_action', None)
        access_count = state_dict.pop('q_controller_access_count', None)
        action_success = state_dict.pop('q_controller_action_success', None)
        action_counts = state_dict.pop('q_controller_action_counts', None)
        step_count = state_dict.pop('_step_count', None)
        stats_grad_norms = state_dict.pop('stats_grad_norms', None)
        stats_update_norms = state_dict.pop('stats_update_norms', None)

        try:
            super().load_state_dict(state_dict)

            # Load Q-controller state
            if q_table is not None: self.q_controller.q_table = q_table
            if epsilon is not None: self.q_controller.epsilon = float(epsilon)
            if prev_loss is not None: self.q_controller.prev_loss = float(prev_loss) if prev_loss is not None else None
            if prev_state is not None: self.q_controller.prev_state = prev_state
            if prev_action is not None: self.q_controller.prev_action = prev_action
            if access_count is not None: self.q_controller.q_table_access_count = access_count
            if action_success is not None: self.q_controller.action_success = action_success
            if action_counts is not None: self.q_controller.action_counts = action_counts
            if step_count is not None: self._step_count = int(step_count)

            # Restore stats deques
            if stats_grad_norms is not None: self.stats['grad_norms'] = deque(stats_grad_norms, maxlen=self.stats['grad_norms'].maxlen)
            if stats_update_norms is not None: self.stats['update_norms'] = deque(stats_update_norms, maxlen=self.stats['update_norms'].maxlen)

            # Restore parameter group state (q_scale, base_lr)
            for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
                 if 'q_scale' in saved_group: group['q_scale'] = saved_group['q_scale']
                 elif 'base_lr' in group and group['base_lr'] > 1e-9: group['q_scale'] = group['lr'] / group['base_lr']
                 else: group['q_scale'] = 1.0
                 # Restore min_weight_decay if saved (might not be in older checkpoints)
                 group.setdefault('min_weight_decay', group.get('weight_decay', 1e-4) * 0.1) # Add default if missing


            logging.info(f"Loaded checkpoint state. Resuming step {self._step_count}.")

        except Exception as e:
            logging.error(f"Error loading EnhancedSGD state dict: {e}", exc_info=True)
            # Re-initialize adaptive state if loading fails
            self._init_optimization_state( # Call re-init with current settings
                 max_grad_norm=self.max_grad_norm,
                 noise_scale_initial=self.noise_scale_initial,
                 noise_decay_steps=self.noise_decay_steps,
                 update_clip_factor=self.update_clip_factor,
                 adjust_interval=self.adjust_interval,
                 lr_scale_bounds=self.q_controller.lr_scale_bounds,
                 momentum_scale_bounds=self.q_controller.momentum_scale_bounds,
                 q_learning_config={ # Rebuild q_config dict
                     'learning_rate': self.q_controller.alpha,
                     'discount': self.q_controller.gamma,
                     'epsilon': 0.15, # Reset epsilon
                     'epsilon_decay': self.q_controller.epsilon_decay,
                     'min_epsilon': self.q_controller.min_epsilon,
                     'min_weight_decay': self.q_controller.min_weight_decay,
                     'max_q_table_size': self.q_controller.max_q_table_size,
                     'success_decay': self.q_controller.success_decay
                 }
            )
            self._step_count = 0
            for group in self.param_groups: # Reset group state
                 group['q_scale'] = 1.0
                 group['lr'] = group['base_lr']
            logging.warning("Re-initialized adaptive state due to loading error.")


# Example Usage (Reflecting main.py and Parameter Groups)
if __name__ == "__main__":
    import torch.nn as nn

    logging.info("Starting EnhancedSGD (Innovative + Stable) example usage...")

    # Dummy model with distinct layers for group testing
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 10) # Smaller layer
            self.transformer = nn.Linear(10, 10) # Larger layer
            self.output = nn.Linear(10, 5) # Output layer

        def forward(self, x_idx, x_cont):
            emb = self.embedding(x_idx) # Shape [B, Seq, Emb]
            # Combine embedding with continuous features (simple addition for example)
            combined = emb.mean(dim=1) + x_cont # [B, Emb]
            trans = torch.relu(self.transformer(combined))
            out = self.output(trans)
            return out

    model = DummyModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logging.info(f"Using device: {device}")

    # --- Configuration with Parameter Groups ---
    # Example: Lower LR for embedding, higher WD for transformer
    param_groups = [
        {
            'params': model.embedding.parameters(),
            'lr': 0.001, 'base_lr': 0.001, # Store base_lr explicitly
            'weight_decay': 0.01, 'min_weight_decay': 0.001
        },
        {
            'params': model.transformer.parameters(),
            'lr': 0.0005, 'base_lr': 0.0005,
            'weight_decay': 0.05, 'min_weight_decay': 0.005
        },
        {
            'params': model.output.parameters(),
            'lr': 0.001, 'base_lr': 0.001,
            'weight_decay': 0.01, 'min_weight_decay': 0.001
        }
    ]

    # Q-learning config
    q_learning_config = {
        "learning_rate": 0.02, "discount": 0.97, "epsilon": 0.15,
        "epsilon_decay": 0.9995, "min_epsilon": 0.02,
        "max_q_table_size": 20000, "min_weight_decay": 1e-4, # Base min WD
        "success_decay": 0.99
    }

    optimizer = EnhancedSGD(
        param_groups, # Pass parameter groups
        momentum=0.9,
        max_grad_norm=1.0,
        noise_scale_initial=0.0005, # Slightly lower initial noise
        noise_decay_steps=1000, # Longer noise decay
        update_clip_factor=4.0, # Slightly lower update clip factor
        adjust_interval=50, # Adjust hyperparameters every 50 steps
        lr_scale_bounds=(0.85, 1.15),
        momentum_scale_bounds=(0.9, 1.1),
        q_learning_config=q_learning_config
    )
    logging.info("Optimizer created with parameter groups.")

    # Dummy training loop
    num_steps = 200
    batch_size = 8
    seq_len = 10
    accumulation_steps = 1 # No accumulation in this simple example

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logging.info(f"AMP {'enabled' if use_amp else 'disabled'}.")

    global_step = 0
    for step in range(num_steps * accumulation_steps):
        is_optimizing_step = (step + 1) % accumulation_steps == 0

        # Dummy input and target
        input_ids = torch.randint(0, 100, (batch_size, seq_len), device=device)
        input_continuous = torch.randn(batch_size, 10, device=device)
        targets = torch.randint(0, 5, (batch_size,), device=device)

        # Closure for optimizer step
        def closure():
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids, input_continuous)
                loss_closure = torch.nn.functional.cross_entropy(outputs, targets)
            return loss_closure

        # Forward pass for backward
        with torch.cuda.amp.autocast(enabled=use_amp):
             outputs = model(input_ids, input_continuous)
             loss = torch.nn.functional.cross_entropy(outputs, targets)
             loss_scaled = loss / accumulation_steps

        # Backward pass
        optimizer.zero_grad(set_to_none=True) # Zero grad before backward
        scaler.scale(loss_scaled).backward()

        step_loss = loss.item()

        if is_optimizing_step:
            global_step += 1
            scaler.unscale_(optimizer) # Unscale before step

            # Optimizer step with closure
            scaler.step(optimizer, closure=closure)
            scaler.update()

            if global_step % 10 == 0:
                logging.info(f"Global Step: {global_step}, Loss: {step_loss:.4f}")
                # Log LR/Mom for each group to see adaptation
                for i, group in enumerate(optimizer.param_groups):
                    logging.info(f"  Group {i}: LR={group['lr']:.2e}, Mom={group['momentum']:.3f}, WD={group['weight_decay']:.2e}")


    logging.info("Finished EnhancedSGD (Innovative + Stable) example usage.")