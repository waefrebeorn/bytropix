#!/usr/bin/env python3
"""Run our model with 1, 2, 5, 10, 20, 30, 40 SSM layers and compare outputs."""
import subprocess
import numpy as np

# Build and run model with varying numbers of layers
# Actually, our model always has 40 layers. We can't easily skip layers.
# But we CAN save intermediate hidden states.

# Let me instead build a tool that saves hidden state after each layer
# and compare layer by layer to check for divergence.

print("Need to modify model to save per-layer hidden states. Using C tool instead.")
