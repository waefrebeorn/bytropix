import logging
import math
from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn.init import _calculate_fan_in_and_fan_out

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Weight Initialization (copied from etp_wubu_architectures.py for self-containment) ---
def init_weights_general(m: nn.Module, init_type: str = 'xavier_uniform', nonlinearity: str = 'relu', gain_factor: float = 1.0):
    """
    Initializes weights of a given module.
    :param m: PyTorch module.
    :param init_type: 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal'
    :param nonlinearity: 'relu', 'leaky_relu', 'tanh', 'sigmoid', 'linear'
    :param gain_factor: Multiplicative factor for the gain (e.g., for specific activations like SiLU)
    """
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        gain = nn.init.calculate_gain(nonlinearity, param=0.2 if nonlinearity == 'leaky_relu' else None) * gain_factor # param for leaky_relu
        
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=gain)
        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=gain)
        elif init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity, a=0.2 if nonlinearity == 'leaky_relu' else 0)
        elif init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity, a=0.2 if nonlinearity == 'leaky_relu' else 0)
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight, gain=gain)
        else:
            logging.warning(f"Unsupported init_type: {init_type}. Using default initialization for {m.__class__.__name__}.")

        if m.bias is not None:
            if init_type in ['kaiming_uniform', 'kaiming_normal']:
                fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                if fan_in != 0:
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
                else:
                    nn.init.zeros_(m.bias)
            else:
                nn.init.zeros_(m.bias)

    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.02)


class LatentDiscriminatorMLP(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: Optional[List[int]] = None, 
                 activation_fn: str = "leaky_relu", 
                 use_spectral_norm: bool = False):
        """
        MLP Discriminator for latent vectors.

        Args:
            input_dim: Dimensionality of the input latent vector.
            hidden_dims: List of integers specifying the sizes of hidden layers.
                         If None, defaults to [input_dim, input_dim // 2].
            activation_fn: String specifying activation: "leaky_relu", "relu", "gelu", "tanh", "sigmoid".
            use_spectral_norm: Boolean to control spectral normalization on linear layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.use_spectral_norm = use_spectral_norm

        if hidden_dims is None:
            # Default hidden layers if none provided
            h_dim1 = input_dim
            h_dim2 = max(1, input_dim // 2) # Ensure hidden_dim is at least 1
            # Ensure a reasonable progression if input_dim is very small (e.g. 1 or 2)
            if input_dim <= 2:
                h_dim1 = max(4, input_dim * 2) # Boost small dims
                h_dim2 = max(2, input_dim)
            self.hidden_dims = [h_dim1, h_dim2]
        else:
            self.hidden_dims = hidden_dims
        
        if not self.hidden_dims : # Ensure there's at least one layer before output if list is empty
            self.hidden_dims = [max(1, input_dim //2)]


        layers = []
        current_dim = input_dim

        for h_dim in self.hidden_dims:
            if h_dim <= 0:
                logging.warning(f"Hidden dimension {h_dim} is invalid, skipping layer.")
                continue
            linear_layer = nn.Linear(current_dim, h_dim)
            if self.use_spectral_norm:
                layers.append(spectral_norm(linear_layer))
            else:
                layers.append(linear_layer)
            
            # Add activation function
            if activation_fn == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif activation_fn == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation_fn == "gelu":
                layers.append(nn.GELU())
            elif activation_fn == "tanh":
                layers.append(nn.Tanh())
            elif activation_fn == "sigmoid": # Sigmoid not typical for hidden layers in D but possible
                layers.append(nn.Sigmoid())
            else:
                logging.warning(f"Unsupported activation: {activation_fn}. Using LeakyReLU(0.2).")
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            current_dim = h_dim

        # Final output layer (logit)
        output_layer = nn.Linear(current_dim, 1)
        if self.use_spectral_norm:
            layers.append(spectral_norm(output_layer))
        else:
            layers.append(output_layer)
        # No activation after the final layer (outputs raw logits)

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        # Kaiming for LeakyReLU/ReLU, Xavier for Tanh/Sigmoid/GELU (approx)
        init_nonlinearity = activation_fn if activation_fn in ["leaky_relu", "relu"] else "linear" # For Gelu/Tanh use linear gain approx.
        init_method = 'kaiming_normal' if init_nonlinearity in ["leaky_relu", "relu"] else 'xavier_uniform'
        
        self.apply(lambda m: init_weights_general(m, init_type=init_method, nonlinearity=init_nonlinearity))
        logging.info(f"LatentDiscriminatorMLP initialized with input_dim={input_dim}, hidden_dims={self.hidden_dims}, "
                     f"activation={activation_fn}, spectral_norm={use_spectral_norm}. Model: {self.mlp}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.
        Args:
            x: Input tensor (latent vector), shape (batch_size, input_dim).
        Returns:
            Logits, shape (batch_size, 1).
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Input tensor last dimension ({x.shape[-1]}) "
                             f"does not match discriminator input_dim ({self.input_dim})")
        return self.mlp(x)


if __name__ == '__main__':
    logging.info("Starting example usage of LatentDiscriminatorMLP.")
    
    batch_size = 32
    latent_dim_example = 64

    # Test case 1: Default hidden_dims
    logging.info("\n--- Test Case 1: Default hidden_dims ---")
    try:
        discriminator1 = LatentDiscriminatorMLP(
            input_dim=latent_dim_example,
            activation_fn="leaky_relu",
            use_spectral_norm=False
        )
        dummy_latent_vector1 = torch.randn(batch_size, latent_dim_example)
        logits1 = discriminator1(dummy_latent_vector1)
        logging.info(f"Discriminator 1 (default hidden): {discriminator1}")
        logging.info(f"Input shape: {dummy_latent_vector1.shape}, Output (logits) shape: {logits1.shape}")
        assert logits1.shape == (batch_size, 1)
    except Exception as e:
        logging.error(f"Error in Test Case 1: {e}", exc_info=True)

    # Test case 2: Custom hidden_dims and spectral norm
    logging.info("\n--- Test Case 2: Custom hidden_dims and Spectral Norm ---")
    custom_hidden = [128, 64, 32]
    try:
        discriminator2 = LatentDiscriminatorMLP(
            input_dim=latent_dim_example,
            hidden_dims=custom_hidden,
            activation_fn="relu",
            use_spectral_norm=True
        )
        dummy_latent_vector2 = torch.randn(batch_size, latent_dim_example)
        logits2 = discriminator2(dummy_latent_vector2)
        logging.info(f"Discriminator 2 (custom hidden, SN): {discriminator2}")
        logging.info(f"Input shape: {dummy_latent_vector2.shape}, Output (logits) shape: {logits2.shape}")
        assert logits2.shape == (batch_size, 1)
        # Check if spectral norm is applied (by checking for weight_orig attribute)
        sn_applied = any('weight_orig' in param_name for param_name, _ in discriminator2.named_parameters())
        assert sn_applied, "Spectral norm was requested but not detected in parameters."
        logging.info("Spectral norm appears to be applied.")
    except Exception as e:
        logging.error(f"Error in Test Case 2: {e}", exc_info=True)

    # Test case 3: Small input dimension
    logging.info("\n--- Test Case 3: Small input_dim (e.g., 2) with default hidden ---")
    small_latent_dim = 2
    try:
        discriminator3 = LatentDiscriminatorMLP(
            input_dim=small_latent_dim, # Very small input dim
            activation_fn="gelu"
        )
        dummy_latent_vector3 = torch.randn(batch_size, small_latent_dim)
        logits3 = discriminator3(dummy_latent_vector3)
        logging.info(f"Discriminator 3 (small input_dim): {discriminator3}")
        logging.info(f"Input shape: {dummy_latent_vector3.shape}, Output (logits) shape: {logits3.shape}")
        assert logits3.shape == (batch_size, 1)
    except Exception as e:
        logging.error(f"Error in Test Case 3: {e}", exc_info=True)

    # Test case 4: Empty hidden_dims list (should default)
    logging.info("\n--- Test Case 4: Empty hidden_dims list ---")
    try:
        discriminator4 = LatentDiscriminatorMLP(
            input_dim=latent_dim_example,
            hidden_dims=[], # Empty list
            activation_fn="tanh"
        )
        dummy_latent_vector4 = torch.randn(batch_size, latent_dim_example)
        logits4 = discriminator4(dummy_latent_vector4)
        logging.info(f"Discriminator 4 (empty hidden_dims): {discriminator4}")
        logging.info(f"Input shape: {dummy_latent_vector4.shape}, Output (logits) shape: {logits4.shape}")
        assert logits4.shape == (batch_size, 1)
        # Check that hidden_dims were actually populated by default logic
        assert len(discriminator4.hidden_dims) > 0 and discriminator4.hidden_dims[0] > 0
    except Exception as e:
        logging.error(f"Error in Test Case 4: {e}", exc_info=True)

    logging.info("\nExample usage of LatentDiscriminatorMLP finished.")
