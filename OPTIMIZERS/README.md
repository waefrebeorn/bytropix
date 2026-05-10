# Optimizers: Learning How to Learn

The WuBu project doesn't just build geometric architectures — it builds geometric *optimizers* that learn to tune the learning process itself.

## Contents

**`qcontroller.py`** — The HAKMEMQController: a Q-learning agent that dynamically tunes learning rate and momentum based on real-time diagnostic data from training (loss trends, gradient norms, etc.). Acts as "adaptive strain engineering" for the geometry.

**`qlearnerexample.py`** — Standalone Q-learner example with discrete action space (multiply/divide LR by factors).

**`pidexample.py`** — PID controller example for training dynamics (classic proportional-integral-derivative control).

**`toroidexample.py`** — Toroidal loss landscape example — training on a torus topology.

## Why This Matters

Standard optimizers (Adam, SGD) use fixed update rules. The WuBu philosophy says: if the architecture is geometric and adaptive, the optimizer should be too. The Q-controller learns a policy for adjusting hyperparameters based on the state of training — it's a meta-learning system that operates in real-time.

This is particularly important for hyperbolic models, where the geometry itself can destabilize training if curvatures, scales, and rotations aren't carefully tuned. The Q-controller acts as a stabilizing feedback loop.
