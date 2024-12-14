# EnhancedSGD.py

import torch
import math
import random
import logging
import torch.optim as optim
from scipy.stats import norm
from collections import deque
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QLearningController:
    """
    Q-Learning Controller with adaptive mixed action space for multiplicative and additive adjustments.
    """
    def __init__(self, param_init, learning_rate=0.1, discount=0.9, epsilon=0.5, epsilon_decay=0.998, initial_mix_prob=0.7):
        """
        Initializes the Q-Learning controller.

        Args:
            param_init (dict): Initial parameter scales (e.g., {'lr_scale': 1.0, 'momentum_scale': 1.0, 'grad_scale': 1.0}).
            learning_rate (float): Learning rate for Q-learning updates.
            discount (float): Discount factor for future rewards.
            epsilon (float): Initial exploration rate for epsilon-greedy strategy.
            epsilon_decay (float): Factor to decay epsilon after each update for more exploitation over time.
            initial_mix_prob (float): Initial probability to choose multiplicative action over additive, decaying over time.
        """
        self.params = {k: float(v) for k, v in param_init.items()}
        self.q_table = {}
        self.alpha = float(learning_rate)
        self.gamma = float(discount)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.mix_prob = float(initial_mix_prob)  # Start with lower multiplicative probability
        self.initial_mix_prob = float(initial_mix_prob)
        self.prev_loss = None  # Track previous loss for adaptive decay

    def update_mix_prob(self, current_loss, epoch, decay_factor=0.98, loss_threshold=0.01):
        """
        Adapts `mix_prob` to favor additive adjustments as training converges.

        Args:
            current_loss (float): The current loss.
            epoch (int): The current training epoch.
            decay_factor (float): Factor for exponential decay over epochs.
            loss_threshold (float): Threshold for considering convergence based on loss change.
        """
        if self.prev_loss is None:
            self.prev_loss = current_loss
            return

        # Calculate loss reduction rate
        loss_change = abs(self.prev_loss - current_loss)

        # High loss change -> favor multiplicative, low loss change -> favor additive
        if loss_change < loss_threshold:
            self.mix_prob = max(0.1, self.mix_prob * decay_factor)  # Decay towards additive
            logging.debug(f"Loss change {loss_change:.4f} < threshold {loss_threshold}. Decaying mix_prob to {self.mix_prob:.4f}")
        else:
            self.mix_prob = min(self.initial_mix_prob, self.mix_prob * (1 + 0.05 * decay_factor))  # Slightly bias back to multiplicative
            logging.debug(f"Loss change {loss_change:.4f} >= threshold {loss_threshold}. Increasing mix_prob to {self.mix_prob:.4f}")

        self.prev_loss = current_loss  # Update previous loss

    def get_state(self, loss, gradient_var, entropy, epoch, layer_depth):
        """
        Defines the state based on rounded loss, gradient variance, entropy, epoch, and layer depth.

        Args:
            loss (torch.Tensor): Current loss value.
            gradient_var (float): Current gradient variance.
            entropy (float): Current gradient entropy.
            epoch (int): Current epoch number.
            layer_depth (int): Depth of the layer in the model.

        Returns:
            tuple: Rounded loss, gradient variance, entropy, epoch, and layer depth as the state.
        """
        return (
            round(float(loss.item()), 2),
            round(float(gradient_var), 2),
            round(float(entropy), 2),
            int(epoch),
            int(layer_depth)
        )

    def choose_action(self, state):
        """
        Chooses an action based on the epsilon-greedy strategy, with decaying epsilon.

        Args:
            state (tuple): Current state.

        Returns:
            dict: Chosen action as parameter adjustments.
        """
        if random.uniform(0, 1) < self.epsilon:
            action = self.random_action()
            logging.debug("Chosen action via exploration (random).")
        else:
            action = self.best_action(state)
            logging.debug("Chosen action via exploitation (best action).")
        # Decay epsilon for less exploration over time
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
        logging.debug(f"Epsilon decayed to {self.epsilon:.4f}")
        return action

    def random_action(self):
        """
        Generates a random action using a mixed approach of multiplicative and additive adjustments.

        Returns:
            dict: Random adjustments for each parameter.
        """
        action_type = "multiplicative" if random.uniform(0, 1) < self.mix_prob else "additive"
        if action_type == "multiplicative":
            action = {param: random.choice([0.95, 1.0, 1.05]) for param in self.params}
        else:
            action = {param: random.choice([-0.005, 0.0, 0.005]) for param in self.params}
        logging.debug(f"Random action generated: {action_type} - {action}")
        return action

    def best_action(self, state):
        """
        Chooses the best action based on the current Q-table.

        Args:
            state (tuple): Current state.

        Returns:
            dict: Best action for the given state.
        """
        if state not in self.q_table:
            # Initialize both multiplicative and additive actions for the state
            self.q_table[state] = {f"{param}_mult": 0.0 for param in self.params}
            self.q_table[state].update({f"{param}_add": 0.0 for param in self.params})

        # Select the best action type for each parameter based on Q-values
        action = {}
        for param in self.params:
            mult_key = f"{param}_mult"
            add_key = f"{param}_add"
            mult_q = self.q_table[state].get(mult_key, 0.0)
            add_q = self.q_table[state].get(add_key, 0.0)
            if mult_q >= add_q:
                action[param] = 1.0 + mult_q  # Assuming Q-values represent multiplicative factors around 1.0
            else:
                action[param] = add_q  # Assuming Q-values represent additive factors
        logging.debug(f"Best action selected: {action}")
        return action

    def update_q_value(self, state, action, reward, next_state):
        """
        Updates the Q-table with the received reward and the maximum future Q-value.

        Args:
            state (tuple): Previous state.
            action (dict): Action taken.
            reward (float): Reward received.
            next_state (tuple): New state after taking the action.
        """
        reward = float(reward)
        if state not in self.q_table:
            self.q_table[state] = {f"{param}_mult": 0.0 for param in self.params}
            self.q_table[state].update({f"{param}_add": 0.0 for param in self.params})
        max_future_q = max(self.q_table.get(next_state, {}).values(), default=0.0)
        for param, value in action.items():
            action_type = "mult" if value in [0.95, 1.0, 1.05] else "add"
            key = f"{param}_{action_type}"
            old_q = self.q_table[state][key]
            self.q_table[state][key] += self.alpha * (reward + self.gamma * max_future_q - old_q)
            logging.debug(f"Updated Q-value for {key}: {old_q:.4f} -> {self.q_table[state][key]:.4f}")

    def adjust_params(self, action, blending_factor=0.05):
        """
        Softly adjusts parameters with blending for smooth updates.

        Args:
            action (dict): Parameter adjustments.
            blending_factor (float): Proportion of adjustment applied to previous parameter values.

        Returns:
            dict: Updated parameters.
        """
        for param in action:
            if action[param] in [0.95, 1.0, 1.05]:
                # Multiplicative adjustment
                self.params[param] *= action[param]
            else:
                # Additive adjustment
                self.params[param] += action[param]
        logging.debug(f"Parameters after adjustment: {self.params}")
        return self.params


class EnhancedSGD(optim.Optimizer):
    """
    Enhanced Stochastic Gradient Descent optimizer with adaptive learning rate and momentum adjustments via Q-Learning,
    incorporating entropy-based adjustments, loss spike correction, and gradient retention.
    """
    def __init__(self, params, model=None, lr=0.01, momentum=0.9,
                 smoothing_factor=0.1, decay=0.99, usage_case="LLM", max_steps=100000,
                 lookahead_k=5, lookahead_alpha=0.5, apply_noise=True,
                 adaptive_momentum=True, layer_type_scaling=None, grad_cam_scaling=False,
                 gradient_centering=True, noise_scale=1e-4, gradient_clipping=True,
                 entropy_weight=0.1, gradient_buffer_size=100, loss_correction_factor=0.5):
        """
        Initializes the EnhancedSGD optimizer.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            model (nn.Module, optional): The model being optimized for parameter depth mapping.
            lr (float): Initial learning rate.
            momentum (float): Initial momentum.
            smoothing_factor (float): Smoothing factor for gradient variance calculation.
            decay (float): Decay factor for moving averages.
            usage_case (str): The usage scenario to set defaults for.
            max_steps (int): Maximum number of optimization steps.
            lookahead_k (int): Number of steps to look ahead for weight updates.
            lookahead_alpha (float): Step size for lookahead updates.
            apply_noise (bool): Whether to apply noise to gradients.
            adaptive_momentum (bool): Whether to adapt momentum based on Q-Learning.
            layer_type_scaling (dict, optional): Scaling factors per layer type.
            grad_cam_scaling (bool): Whether to apply Grad-CAM scaling.
            gradient_centering (bool): Whether to center gradients by subtracting the mean.
            noise_scale (float): Scale of the noise to apply to gradients.
            gradient_clipping (bool): Whether to apply gradient clipping.
            entropy_weight (float): Weight for entropy in reward calculation.
            gradient_buffer_size (int): Maximum size of the gradient retention buffer.
            loss_correction_factor (float): Factor to reduce learning rate upon loss spike detection.
        """
        defaults = dict(
            lr=float(lr),
            momentum=float(momentum),
            smoothing_factor=float(smoothing_factor),
            decay=float(decay),
            weight_decay=0.01
        )
        super(EnhancedSGD, self).__init__(params, defaults)

        # Optimizer state and hyperparameter initialization
        self.gradient_clipping = gradient_clipping
        self.state['step_count'] = 0
        self.state['grad_history'] = []
        self.state['lr_history'] = []
        self.state['entropy_history'] = []
        self.state['loss_history'] = deque(maxlen=100)  # Track recent losses for moving average
        self.grad_var = 0.0
        self.prev_grad_var = 0.0
        self.current_momentum = float(momentum)
        self.smoothing_factor = float(smoothing_factor)
        self.gradient_centering = bool(gradient_centering)
        self.noise_scale = float(noise_scale)
        self.loss_correction_factor = float(loss_correction_factor)
        self.stabilizing_factor = 0.9  # Factor for dynamic variance stabilization
        self.phased_threshold_multiplier = 1.5  # Multiplier for phase-based corrections

        # Q-Learning controller for adaptive parameter adjustment
        self.q_controller = QLearningController(
            {'lr_scale': 1.0, 'momentum_scale': 1.0, 'grad_scale': 1.0},
            epsilon_decay=0.995,
            initial_mix_prob=0.8
        )

        # Lookahead settings
        self.lookahead_k = int(lookahead_k)
        self.lookahead_alpha = float(lookahead_alpha)
        self.lookahead_step = 0
        self.slow_weights = [p.clone().detach() for p in self.param_groups[0]['params']]
        for sw in self.slow_weights:
            sw.requires_grad = False

        # Adaptive, noise, and scaling settings
        self.apply_noise = bool(apply_noise)
        self.adaptive_momentum = bool(adaptive_momentum)
        self.layer_type_scaling = layer_type_scaling
        self.grad_cam_scaling = bool(grad_cam_scaling)

        # Internal tracking variables
        self.entropy_weight = float(entropy_weight)
        self.gradient_buffer_size = int(gradient_buffer_size)
        self.gradient_buffer = deque(maxlen=self.gradient_buffer_size)  # Buffer to retain partial gradients

    def step(self, closure=None, current_epoch=1):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
            current_epoch (int): The current epoch number.

        Returns:
            torch.Tensor or None: The loss if closure is provided, else None.
        """
        loss = closure() if closure is not None else None
        self.state['step_count'] += 1
        base_lr = float(self.param_groups[0]['lr'])
        effective_lr = base_lr

        # Gradient collection for variance and entropy calculation
        grad_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad_list.append(grad.view(-1))

        # Process gradients
        if grad_list:
            all_grads = torch.cat(grad_list)
            current_grad_var = float(all_grads.var().item())
            self.grad_var = self.smoothing_factor * current_grad_var + (1 - self.smoothing_factor) * self.grad_var

            # Adaptive variance stabilization
            if abs(self.grad_var - self.prev_grad_var) > self.stabilizing_factor:
                self.grad_var = self.prev_grad_var * self.stabilizing_factor + (1 - self.stabilizing_factor) * self.grad_var
                logging.debug(f"Gradient variance stabilized to {self.grad_var:.6f}")

            entropy = self.calculate_entropy(all_grads.cpu().numpy())
            self.state['grad_history'].append(self.grad_var)
            self.state['entropy_history'].append(entropy)

        # Loss handling with adaptive threshold
        if loss is not None:
            # Track losses and detect spikes
            self.state['loss_history'].append(loss.item())
            avg_loss = sum(self.state['loss_history']) / len(self.state['loss_history'])
            loss_threshold = avg_loss * self.phased_threshold_multiplier
            if loss.item() > loss_threshold:
                logging.warning(f"Significant loss spike detected at epoch {current_epoch}. Applying corrective measures.")
                effective_lr *= self.loss_correction_factor
                self.param_groups[0]['lr'] = effective_lr
                self.current_momentum *= 0.9  # Temporary damp momentum
                logging.info(f"Adjusting LR to {effective_lr:.6f} and momentum to {self.current_momentum:.4f} to stabilize.")

            # Adaptive Q-Learning adjustments
            layer_depth = 1  # Simplification; can be mapped if layer depths are tracked
            current_entropy = self.state['entropy_history'][-1] if self.state['entropy_history'] else 0.0
            current_state = self.q_controller.get_state(loss, self.grad_var, current_entropy, current_epoch, layer_depth)
            action = self.q_controller.choose_action(current_state)
            adjusted_params = self.q_controller.adjust_params(action, blending_factor=0.05)

            # Dynamic learning rate adjustment with phased decay
            adjustment_decay = max(0.5, 1 - (current_epoch / 20))
            lr_adjustment = min(max(adjusted_params.get('lr_scale', 1.0), 0.5), 1.5 * adjustment_decay)
            effective_lr = base_lr * lr_adjustment
            self.param_groups[0]['lr'] = effective_lr

            # Adaptive momentum handling with phased decay
            if self.adaptive_momentum:
                self.current_momentum = min(max(self.current_momentum * adjusted_params.get('momentum_scale', 1.0), 0.85), 0.99 * adjustment_decay)

            self.state['lr_history'].append(effective_lr)
            logging.info(f"Epoch {current_epoch}, Step {self.state['step_count']}: Effective LR = {effective_lr:.6f}, Momentum = {self.current_momentum:.4f}")

        # Apply updates with adjusted parameters
        grad_means = []
        for group in self.param_groups:
            group['lr'] = effective_lr
            group['momentum'] = self.current_momentum  # Update momentum in parameter group
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if self.gradient_centering:
                    grad = grad - grad.mean()

                if self.gradient_clipping:
                    grad = self.adaptive_clipping(grad)

                if self.apply_noise:
                    grad = grad + torch.randn_like(grad) * self.noise_scale

                # Retain partial gradients based on entropy
                if isinstance(self.q_controller, QLearningController):
                    current_entropy = self.state['entropy_history'][-1] if 'entropy_history' in self.state and self.state['entropy_history'] else 0.0
                    if current_entropy > 0.5:  # Threshold can be adjusted
                        self.gradient_buffer.append(grad.clone().detach())
                        logging.debug(f"Gradient retained in buffer. Buffer size: {len(self.gradient_buffer)}")

                state = self.state.setdefault(p, {})
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(self.current_momentum).add_(grad)
                p.data.add_(momentum_buffer, alpha=-group['lr'])

                grad_means.append(float(grad.mean()))  # Collect for summary

        # Summarized batch stats
        if grad_means:
            avg_grad_mean = sum(grad_means) / len(grad_means)
            logging.debug(f"Batch Summary - Avg Grad Mean: {avg_grad_mean:.6e}, LR: {effective_lr:.4e}, Momentum: {self.current_momentum:.4f}")

        # Post-update Q-Learning reward and stabilization
        if loss is not None:
            try:
                next_state = self.q_controller.get_state(loss, self.grad_var, current_entropy, current_epoch, layer_depth)
                grad_var_change = self.grad_var - self.prev_grad_var
                entropy_change = current_entropy - (self.state['entropy_history'][-2] if len(self.state['entropy_history']) > 1 else current_entropy)
                reward = - (abs(grad_var_change) + self.entropy_weight * abs(entropy_change))
                self.q_controller.update_q_value(current_state, action, reward, next_state)
                self.prev_grad_var = self.grad_var
            except Exception as e:
                logging.error(f"Error in Q-Learning update: {e}")
                raise

        return loss if isinstance(loss, torch.Tensor) else None

    def adaptive_clipping(self, grad):
        """
        Applies adaptive gradient clipping based on gradient variance.

        Args:
            grad (torch.Tensor): The gradient tensor.

        Returns:
            torch.Tensor: Clipped gradient tensor.
        """
        if self.grad_var > 0:
            clip_value = min(max(math.sqrt(self.grad_var) * 2, 0.5), 10.0)
            return torch.clamp(grad, -clip_value, clip_value)
        return grad

    def calculate_entropy(self, gradients):
        """
        Calculates the entropy of the gradient distribution.

        Args:
            gradients (numpy.ndarray): Flattened gradients.

        Returns:
            float: Entropy value.
        """
        histogram, _ = np.histogram(gradients, bins=100, density=True)
        histogram += 1e-12  # Prevent log(0)
        entropy = -np.sum(histogram * np.log(histogram))
        return entropy

    def bayesian_initialize_params(self):
        """
        Initializes parameters using a Bayesian approach with normal distributions.
        """
        for param, init_val in self.q_controller.params.items():
            self.q_controller.params[param] = float(norm.rvs(loc=init_val, scale=0.05))
        logging.info(f"Bayesian initialization of parameters: {self.q_controller.params}")

    def load_state_dict(self, state_dict):
        """
        Loads the state dictionary and ensures any new keys introduced in EnhancedSGD are handled.

        Args:
            state_dict (dict): State dictionary.
        """
        super().load_state_dict(state_dict)
        # Ensure any new keys introduced in EnhancedSGD are handled
        for group in self.param_groups:
            for p in group['params']:
                self.state.setdefault(p, {})
        logging.info("State dictionary loaded successfully.")
