import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt
from geoopt.manifolds.poincare import PoincareBall
import math
import copy # For deep copying config if needed

# --- Utility Functions / Modules (QuaternionLinear from previous code assumed) ---
# Include QuaternionLinear class here if needed for transform_2_to_3
def check_quat_dim(dim):
    if dim % 4 != 0:
        raise ValueError(f"Quaternion dimension must be divisible by 4, but got {dim}")

class QuaternionLinear(nn.Module):
    # (Keep the QuaternionLinear implementation from the previous response)
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        check_quat_dim(in_features)
        check_quat_dim(out_features)
        self.in_features_quat = in_features // 4
        self.out_features_quat = out_features // 4
        self.r_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat))
        self.i_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat))
        self.j_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat))
        self.k_weight = nn.Parameter(torch.Tensor(self.out_features_quat, self.in_features_quat))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features_quat)
        self.r_weight.data.uniform_(-stdv, stdv)
        self.i_weight.data.uniform_(-stdv, stdv)
        self.j_weight.data.uniform_(-stdv, stdv)
        self.k_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        assert x.shape[-1] == self.in_features_quat * 4
        batch_dims = x.shape[:-1]
        x_reshaped = x.view(*batch_dims, self.in_features_quat, 4)
        r_x, i_x, j_x, k_x = x_reshaped[..., 0], x_reshaped[..., 1], x_reshaped[..., 2], x_reshaped[..., 3]
        out_r = F.linear(r_x, self.r_weight) - F.linear(i_x, self.i_weight) - F.linear(j_x, self.j_weight) - F.linear(k_x, self.k_weight)
        out_i = F.linear(r_x, self.i_weight) + F.linear(i_x, self.r_weight) + F.linear(j_x, self.k_weight) - F.linear(k_x, self.j_weight)
        out_j = F.linear(r_x, self.j_weight) - F.linear(i_x, self.k_weight) + F.linear(j_x, self.r_weight) + F.linear(k_x, self.i_weight)
        out_k = F.linear(r_x, self.k_weight) + F.linear(i_x, self.j_weight) - F.linear(j_x, self.i_weight) + F.linear(k_x, self.r_weight)
        output = torch.stack([out_r, out_i, out_j, out_k], dim=-1)
        output = output.view(*batch_dims, self.out_features_quat * 4)
        if self.bias is not None:
            output = output + self.bias
        return output


# --- Revised Configuration for 3 Layers ---
CONFIG = {
    "input_dim": 784,
    "initial_embedding_dim": 128,
    "num_levels": 3, # Fixed to 3
    "hyperbolic_dims": [64, 48, 32], # Dim for Outer, Middle, Inner (Middle needs to be divisible by 4 if quat used later)
    "curvatures": [1.0, 1.0, 1.0],   # Curvature for each level
    "learnable_scales": True,
    # Specialized Initial Scales (Middle starts smaller)
    "initial_scales": [1.0, 0.5, 1.0], # Outer, Middle (*0.5 applied below), Inner
    # Specialized Scale Constraints (min values)
    "scale_min_values": [1e-4, 1e-5, 1e-6], # Outer, Middle, Inner
    # Fixed transforms for the 3-layer setup
    "transform_types": ['mlp', 'quat'], # T(1->2) = mlp, T(2->3) = quat
    "transform_hidden_dims": [56, 40],  # Hidden dim for MLP T(1->2), Quat T(2->3) needs QuatLinear
                                       # Ensure T(2->3) dims are divisible by 4 if using Quat
    "output_dim": 10,
    "aggregation_method": "concat_tangent",
    "dropout": 0.1,
    # Middle Layer Specific Config
    "middle_layer_vector_dim": 48, # Dimension for vector space ops (can differ from hyperbolic_dim[1])
    "middle_layer_vector_hidden_dim": 96, # Hidden dim for vector space MLP
}

# --- WuBu Nesting Components (InterLevelTransform, WuBuNestingLevel as before) ---

class InterLevelTransform(nn.Module):
    """Module for transformations between tangent spaces of nested levels."""
    # (Keep the InterLevelTransform implementation from the previous response,
    # ensure it handles 'mlp' and 'quat' based on CONFIG)
    def __init__(self, in_dim, out_dim, transform_type, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.transform_type = transform_type

        if transform_type == 'mlp':
            if hidden_dim is None:
                hidden_dim = (in_dim + out_dim) // 2
            self.transform = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim)
            )
        elif transform_type == 'quat':
            check_quat_dim(in_dim)
            check_quat_dim(out_dim)
            # Note: Quat needs appropriate hidden dim handling if complex sequential used
            self.transform = nn.Sequential(
                QuaternionLinear(in_dim, out_dim),
                nn.Dropout(dropout)
            )
        elif transform_type == 'identity':
             self.transform = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        else:
            raise ValueError(f"Unsupported transform_type: {transform_type}")

    def forward(self, x_tangent):
        return self.transform(x_tangent)


class WuBuNestingLevel(nn.Module):
    """Represents a single level in the WuBu Nesting hierarchy."""
    # (Keep the WuBuNestingLevel implementation from the previous response,
    # but ensure scale initialization uses the specific min value)
    def __init__(self, level_idx, dim, curvature, initial_scale, learnable_scale=True, scale_min_value=1e-6, dropout=0.1):
        super().__init__()
        self.level_idx = level_idx
        self.dim = dim
        self.curvature = torch.tensor([max(curvature, 1e-6)], dtype=torch.float)
        self.manifold = PoincareBall(c=self.curvature)
        self.scale_min_value = scale_min_value

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor([max(initial_scale, self.scale_min_value)], dtype=torch.float))
        else:
            self.register_buffer('scale', torch.tensor([max(initial_scale, self.scale_min_value)], dtype=torch.float))

        self.intra_ball_processor = nn.Identity() # Placeholder

    def forward(self, x_tangent):
        self.manifold.c = self.curvature.to(x_tangent.device)
        # Apply minimum value constraint during forward pass too
        current_scale = torch.clamp(self.scale, min=self.scale_min_value)

        v_scaled = x_tangent * current_scale
        x_hyp = self.manifold.expmap0(v_scaled)
        x_hyp = self.manifold.projx(x_hyp) # Stability

        x_hyp_processed = self.intra_ball_processor(x_hyp)
        x_hyp_processed = self.manifold.projx(x_hyp_processed) # Stability

        v_out_scaled = self.manifold.logmap0(x_hyp_processed)
        v_out_tangent = v_out_scaled / current_scale # Apply inverse scaling

        return x_hyp_processed, v_out_tangent

# --- New/Modified Components for 3-Layer Architecture ---

class MiddleLayerTransform(InterLevelTransform):
    """Special transform for Layer 1 -> 2, containing the transition phase."""
    def __init__(self, in_dim, out_dim, transform_type, hidden_dim=None, dropout=0.1):
        super().__init__(in_dim, out_dim, transform_type, hidden_dim, dropout)
        # Controls the evolution of the middle layer's vector space processing
        self.transition_phase = nn.Parameter(torch.tensor(0.0)) # Initialized to 0

    def forward(self, x_tangent):
        # Apply the base transformation (e.g., MLP)
        transformed = super().forward(x_tangent)
        # The transition_phase is used *outside* this module by the main model
        # to control the MiddleLevelTangentProcessor
        return transformed

class VectorSpaceProcessor(nn.Module):
    """Applies MLP-based operations in a Euclidean vector space."""
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2 # Default hidden dim
        self.projection_in = nn.Linear(dim, hidden_dim)
        self.vector_ops = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout), # Add dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.projection_out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout) # Add dropout after projection out

    def forward(self, x, activation_weight):
        """
        Args:
            x: Input tensor in the vector space. Shape: (batch_size, ..., dim)
            activation_weight: Scalar tensor (0 to 1) controlling influence.
        """
        # Optimization: Skip computation if weight is near zero
        if activation_weight.item() < 1e-4:
            return x

        # Project, process, project back
        proj = F.relu(self.projection_in(x)) # Add activation after input projection
        processed = self.vector_ops(proj)
        result = self.dropout(self.projection_out(processed)) # Apply dropout

        # Weighted combination: (1-w)*original + w*processed
        output = x * (1 - activation_weight) + result * activation_weight
        return output


class MiddleLevelTangentProcessor(nn.Module):
    """Handles tangent space processing for the middle layer, including vector space detour."""
    def __init__(self, tangent_dim, vector_dim, vector_hidden_dim, dropout=0.1):
        super().__init__()
        self.tangent_dim = tangent_dim
        self.vector_dim = vector_dim

        # Linear layer to map from tangent dim to vector space dim (if different)
        self.tangent_mapper = nn.Linear(tangent_dim, vector_dim) if tangent_dim != vector_dim else nn.Identity()

        # The actual vector space processing module
        self.vector_processor = VectorSpaceProcessor(vector_dim, vector_hidden_dim, dropout)

        # Linear layer to map back from vector space dim to tangent dim (if different)
        self.tangent_projector = nn.Linear(vector_dim, tangent_dim) if tangent_dim != vector_dim else nn.Identity()

    def forward(self, v_tangent, activation_weight):
        """
        Args:
            v_tangent: Input tensor in the middle layer's tangent space.
            activation_weight: Scalar tensor (0 to 1) controlling vector space influence.
        """
        # Map to vector space dimension
        v_vector = F.relu(self.tangent_mapper(v_tangent)) # Add activation

        # Process in vector space (conditionally based on weight)
        v_processed_vector = self.vector_processor(v_vector, activation_weight)

        # Map back to tangent space dimension
        v_processed_tangent = self.tangent_projector(v_processed_vector)

        # Residual connection: Add processed part back to original tangent vector?
        # Option 1: Weighted sum (as in VectorSpaceProcessor) -> Let VSP handle this
        # Option 2: Direct replacement -> Implemented below (VSP output is the new vector)
        # Option 3: Additive residual -> return v_tangent + v_processed_tangent * activation_weight
        # Let's go with Option 2 for now, where VSP handles the weighting internally

        return v_processed_tangent # Return the (potentially modified) tangent vector


# --- Main WuBu Nesting Model (Revised for 3 Layers) ---

class WuBuNestingModel(nn.Module):
    """The WuBu Nesting model adapted for the 3-layer evolving architecture."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config["num_levels"] == 3, "This implementation requires exactly 3 levels."
        assert len(config["transform_types"]) == 2, "Requires 2 transform types for 3 levels."
        assert len(config["hyperbolic_dims"]) == 3
        assert len(config["initial_scales"]) == 3
        assert len(config["scale_min_values"]) == 3

        # 1. Initial Euclidean Encoding
        self.initial_encoder = nn.Sequential(
            nn.Linear(config["input_dim"], config["initial_embedding_dim"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"])
        )
        self.to_first_tangent = nn.Linear(
            config["initial_embedding_dim"],
            config["hyperbolic_dims"][0] # Dimension of outer level
        )

        # 2. Three Nested Hyperbolic Levels (Specific Initialization)
        self.levels = nn.ModuleList()
        self.manifolds = []
        for i in range(3):
            # Apply scale multiplier for middle layer if specified implicitly
            initial_scale = config["initial_scales"][i]
            # (No longer need manual *0.5 here, handled in CONFIG)

            level = WuBuNestingLevel(
                level_idx=i,
                dim=config["hyperbolic_dims"][i],
                curvature=config["curvatures"][i],
                initial_scale=initial_scale,
                learnable_scale=config["learnable_scales"],
                scale_min_value=config["scale_min_values"][i], # Use specific min value
                dropout=config["dropout"]
            )
            self.levels.append(level)
            self.manifolds.append(level.manifold)

        # 3. Specialized Inter-Level Transformations
        # Transform: Outer -> Middle (Layer 1 -> 2) - Includes transition_phase
        self.transform_1_to_2 = MiddleLayerTransform(
            in_dim=config["hyperbolic_dims"][0],
            out_dim=config["hyperbolic_dims"][1],
            transform_type=config["transform_types"][0], # Should be 'mlp' based on guide
            hidden_dim=config["transform_hidden_dims"][0],
            dropout=config["dropout"]
        )

        # Middle Layer Tangent Space Processor (Vector Space Detour)
        self.middle_tangent_processor = MiddleLevelTangentProcessor(
            tangent_dim=config["hyperbolic_dims"][1],
            vector_dim=config["middle_layer_vector_dim"],
            vector_hidden_dim=config["middle_layer_vector_hidden_dim"],
            dropout=config["dropout"]
        )

        # Transform: Middle -> Inner (Layer 2 -> 3)
        self.transform_2_to_3 = InterLevelTransform(
            in_dim=config["hyperbolic_dims"][1],
            out_dim=config["hyperbolic_dims"][2],
            transform_type=config["transform_types"][1], # Should be 'quat' based on guide
            # hidden_dim=config["transform_hidden_dims"][1], # QuatLinear doesn't use hidden_dim this way
            dropout=config["dropout"]
        )
        # Check quaternion compatibility
        if config["transform_types"][1] == 'quat':
             check_quat_dim(config["hyperbolic_dims"][1])
             check_quat_dim(config["hyperbolic_dims"][2])

        # 4. Scale-Aware Aggregation & Final Output
        self.aggregation_method = config["aggregation_method"]
        # Aggregate tangent representations from *after* processing in each level
        total_tangent_dim = sum(config["hyperbolic_dims"])

        if self.aggregation_method == "concat_tangent":
            self.final_processor = nn.Sequential(
                nn.Linear(total_tangent_dim, total_tangent_dim // 2),
                nn.ReLU(),
                nn.Dropout(config["dropout"]),
                nn.Linear(total_tangent_dim // 2, config["output_dim"])
            )
        else:
             raise NotImplementedError(f"Aggregation method '{self.aggregation_method}' not implemented.")


    def forward(self, x):
        """Forward pass through the specialized 3-layer structure."""
        # 1. Initial Encoding
        encoded = self.initial_encoder(x)
        outer_tangent_in = self.to_first_tangent(encoded)

        # Store tangent vectors *after* level processing for aggregation
        tangent_outputs = []

        # --- Layer 1 (Outer) ---
        outer_hyp, outer_tangent_out = self.levels[0](outer_tangent_in)
        tangent_outputs.append(outer_tangent_out)

        # --- Transition 1 -> 2 ---
        middle_tangent_pre_process = self.transform_1_to_2(outer_tangent_out)

        # --- Layer 2 (Middle) - With Evolving Tangent Processing ---
        # Get activation weight based on the transition phase parameter
        activation_weight = torch.sigmoid(self.transform_1_to_2.transition_phase)
        # Apply the specialized tangent processor (includes vector space detour)
        middle_tangent_in = self.middle_tangent_processor(middle_tangent_pre_process, activation_weight)
        # Pass through the hyperbolic level
        middle_hyp, middle_tangent_out = self.levels[1](middle_tangent_in)
        tangent_outputs.append(middle_tangent_out)

        # --- Transition 2 -> 3 ---
        inner_tangent_in = self.transform_2_to_3(middle_tangent_out)

        # --- Layer 3 (Inner) ---
        inner_hyp, inner_tangent_out = self.levels[2](inner_tangent_in)
        tangent_outputs.append(inner_tangent_out)

        # 7. Scale-Aware Aggregation
        if self.aggregation_method == "concat_tangent":
            aggregated_repr = torch.cat(tangent_outputs, dim=-1)
        else:
             # Handle other aggregation methods if implemented
             pass

        # 8. Final Processing
        output = self.final_processor(aggregated_repr)

        return output

    def evolve_middle_layer(self, epoch, max_epochs, schedule='linear'):
        """Updates the transition_phase parameter based on training progress."""
        if schedule == 'linear':
             # Simple linear schedule from approx -2.5 to +2.5 -> sigmoid(0.07) to sigmoid(0.93)
             # Adjust range/slope as needed
             transition_value = 5.0 * (epoch / max(max_epochs - 1, 1) - 0.5)
        elif schedule == 'constant':
            # For debugging: keep it fixed
            transition_value = self.transform_1_to_2.transition_phase.item() # Keep current value
        else:
            raise ValueError(f"Unknown evolution schedule: {schedule}")

        # Update the parameter in the MiddleLayerTransform module
        self.transform_1_to_2.transition_phase.data.fill_(transition_value)
        print(f"Epoch {epoch}: Middle layer transition_phase set to {transition_value:.4f} "
              f"(activation weight ~ {torch.sigmoid(self.transform_1_to_2.transition_phase).item():.4f})")


    def get_middle_layer_metrics(self):
        """Returns a dictionary of metrics specific to the middle layer's state."""
        middle_layer = self.levels[1]
        transition_phase_param = self.transform_1_to_2.transition_phase
        metrics = {
            "transition_phase": transition_phase_param.item(),
            "activation_weight": torch.sigmoid(transition_phase_param).item(),
            "scale": middle_layer.scale.item(),
            "curvature": middle_layer.curvature.item(),
            # Potentially add gradient norms or other diagnostics here
        }
        return metrics

# --- Example Usage ---
if __name__ == "__main__":
    # Verify config consistency (especially quat dims)
    if CONFIG["transform_types"][1] == 'quat':
        try:
            check_quat_dim(CONFIG["hyperbolic_dims"][1])
            check_quat_dim(CONFIG["hyperbolic_dims"][2])
            print("Quaternion dimensions seem compatible.")
        except ValueError as e:
            print(f"Configuration Error: {e}")
            exit()

    # Create the model
    model = WuBuNestingModel(CONFIG)
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create dummy input data
    batch_size = 4
    dummy_input = torch.randn(batch_size, CONFIG["input_dim"])

    # --- Simulate Training Evolution ---
    max_epochs_sim = 10
    print("\n--- Simulating Middle Layer Evolution ---")
    for epoch in range(max_epochs_sim):
        # Update the middle layer's evolution parameter
        model.evolve_middle_layer(epoch, max_epochs_sim)

        # Perform a forward pass (optional, just to show it runs)
        if epoch % 2 == 0: # Do forward pass less often
            try:
                output = model(dummy_input)
                # print(f"Epoch {epoch}: Forward pass successful. Output shape: {output.shape}")
            except Exception as e:
                print(f"Epoch {epoch}: Error during forward pass: {e}")
                break # Stop simulation on error

        # Get and print metrics
        metrics = model.get_middle_layer_metrics()
        print(f"Epoch {epoch}: Middle Layer Metrics: {metrics}")
        print("-" * 20)

    print("\n--- Final Model State Metrics ---")
    print(model.get_middle_layer_metrics())

    # Example: Final forward pass and loss calculation
    print("\n--- Final Forward Pass Example ---")
    try:
        final_output = model(dummy_input)
        print("Final forward pass successful!")
        print("Output shape:", final_output.shape)

        dummy_labels = torch.randint(0, CONFIG["output_dim"], (batch_size,))
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(final_output, dummy_labels)
        print("Loss calculation successful:", loss.item())

        # loss.backward() # Would work in a full training loop
        # print("Backward pass check successful.")

    except Exception as e:
         print(f"\nAn unexpected error occurred during final pass: {e}")