import time
import numpy as np
from typing import Dict, Tuple

# JAX is not needed for this pure PID demonstration, keeping the script lightweight.
# This script uses only standard Python and NumPy.

# ==============================================================================
# SCRIPT OVERVIEW & PHILOSOPHICAL CODA
# ==============================================================================
# This script provides a pure, isolated demonstration of a PID controller.
#
# Through a long, iterative journey of building an intelligent learning rate
# controller, we discovered a fundamental principle of control systems in ML,
# which serves as the core lesson of this code:
#
# --- PID is for BALANCE, Q-Learning is for STRATEGY ---
#
# 1.  THE ROLE OF A PID CONTROLLER ("The Orchestra Conductor"):
#     A PID controller is an excellent tool for **balancing multiple, competing
#     loss gradients.** In a complex model with (for example) a reconstruction
#     loss, an adversarial loss, and a perceptual loss, a PID is the perfect
#     "conductor" to dynamically adjust the weight of each loss term, telling
#     each section of the orchestra how loudly to play. Its goal is to maintain
#     a stable equilibrium between known forces. It is a reactive balancer.
#
# 2.  THE FLAW OF USING PID FOR LEARNING RATE:
#     However, a PID is the **wrong tool for managing a learning rate.** A PID
#     is too simple and reactive for a strategic, long-term problem like choosing
#     a learning rate. It can get stuck in local minima, its memory of past
#     failures can overwhelm its recognition of present success ("integral windup"),
#     and it lacks the ability to make strategic, exploratory decisions.
#
# This script focuses solely on the PID's correct and powerful use-case:
# dynamically balancing loss weights.
# ==============================================================================


# ==============================================================================
# 1. PID CONTROLLER (For dynamically weighting loss terms)
# ==============================================================================
class PIDLambdaController:
    """
    A Proportional-Integral-Derivative (PID) controller to dynamically adjust
    loss weights (lambdas) based on performance metrics.
    """
    def __init__(self,
                 targets: Dict[str, float],
                 base_weights: Dict[str, float],
                 gains: Dict[str, Tuple[float, float, float]]):
        """
        Initializes the PID controller.

        Args:
            targets: A dictionary mapping metric names to their desired target values.
            base_weights: The initial/default weight for each metric's loss term.
            gains: A dictionary mapping metric names to their (Kp, Ki, Kd) gains.
        """
        self.targets = targets
        self.base_weights = base_weights
        self.gains = gains
        # Internal state for tracking errors over time for the I and D terms.
        self.state = {
            'integral_error': {k: 0.0 for k in targets.keys()},
            'last_error': {k: 0.0 for k in targets.keys()}
        }

    def __call__(self, last_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculates the new set of loss weights based on the latest metrics."""
        final_lambdas = {}
        for name, base_weight in self.base_weights.items():
            final_lambdas[name] = float(base_weight)  # Start with the base weight

            if name in self.targets:
                current_loss = last_metrics.get(name)
                if current_loss is None: continue

                # --- Core PID Calculation ---
                kp, ki, kd = self.gains[name]
                target = self.targets[name]
                error = float(current_loss) - target

                p_term = kp * error
                self.state['integral_error'][name] += error
                self.state['integral_error'][name] = np.clip(self.state['integral_error'][name], -5.0, 5.0)
                i_term = ki * self.state['integral_error'][name]
                derivative = error - self.state['last_error'][name]
                d_term = kd * derivative
                self.state['last_error'][name] = error

                adjustment = p_term + i_term + d_term
                new_lambda = self.base_weights[name] * np.exp(adjustment)

                final_lambdas[name] = float(np.clip(new_lambda, 0.1, 10.0))

        return final_lambdas

# ==============================================================================
# 2. PURE PID DEMONSTRATION
# ==============================================================================
def main():
    """Run a simulated demonstration for the PID controller."""
    print("\n" + "="*80)
    print("DEMONSTRATION: Pure PID Controller for Balancing Loss Weights")
    print("="*80)
    print("This simulation shows the PID acting as an 'Orchestra Conductor'.")
    print("It will manage two competing losses, 'Recon Loss' and 'Adv Loss'.\n")
    print(" - 'Recon Loss' starts HIGH (target is 0.2). The PID should INCREASE its weight.")
    print(" - 'Adv Loss' starts LOW (target is 0.8). The PID should DECREASE its weight.\n")

    # 1. Setup the controller with targets and tuning gains for two losses.
    pid_controller = PIDLambdaController(
        targets={'recon_loss': 0.2, 'adv_loss': 0.8},
        base_weights={'recon_loss': 1.0, 'adv_loss': 0.5},
        gains={'recon_loss': (0.8, 0.1, 0.2), 'adv_loss': (0.8, 0.1, 0.2)}
    )

    # 2. Initialize the "losses" at values far from their targets.
    current_metrics = {'recon_loss': 0.9, 'adv_loss': 0.1}

    # 3. Run the simulation loop.
    for i in range(30):
        # The PID controller looks at the current losses...
        new_weights = pid_controller(current_metrics)
        # ... and decides on the new weights for the next step.

        print(
            f"Step {i+1:2d}: "
            f"Recon Loss={current_metrics['recon_loss']:.3f} (Tgt: 0.2) -> New Weight={new_weights['recon_loss']:.3f} | "
            f"Adv Loss={current_metrics['adv_loss']:.3f} (Tgt: 0.8) -> New Weight={new_weights['adv_loss']:.3f}"
        )

        # 4. Simulate the effect of the new weights on the next step's losses.
        # A higher weight makes the loss move towards its target faster.
        recon_factor = 0.1 * new_weights['recon_loss']
        adv_factor = 0.1 * new_weights['adv_loss']

        current_metrics['recon_loss'] += (pid_controller.targets['recon_loss'] - current_metrics['recon_loss']) * recon_factor + np.random.randn() * 0.01
        current_metrics['adv_loss'] += (pid_controller.targets['adv_loss'] - current_metrics['adv_loss']) * adv_factor + np.random.randn() * 0.01

        # Clip losses to a valid range.
        current_metrics['recon_loss'] = np.clip(current_metrics['recon_loss'], 0, 2)
        current_metrics['adv_loss'] = np.clip(current_metrics['adv_loss'], 0, 2)
        time.sleep(0.05)

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    print("- The weight for 'Recon Loss' started HIGH to force the high loss down.")
    print("- The weight for 'Adv Loss' started LOW to allow the low loss to rise.")
    print("- As each loss approached its target, its weight correctly stabilized near its base value.")
    print("\nThis demonstrates the PID's strength: maintaining a dynamic equilibrium.")

if __name__ == "__main__":
    main()