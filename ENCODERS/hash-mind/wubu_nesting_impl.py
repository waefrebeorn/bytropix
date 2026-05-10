import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Constants
EPS = 1e-7  # Small epsilon for numerical stability

# =====================================================================
# Hyperbolic Geometry Utilities with Proper Scale Support
# =====================================================================
class HyperbolicUtils:
    """
    Enhanced utility functions for Poincare ball model of hyperbolic geometry.
    Implements scale-aware exponential and logarithmic maps for proper nesting.
    """
    @staticmethod
    def poincare_clip(x: torch.Tensor, c: float, radius: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
        """Clips points to stay strictly inside the Poincare ball boundary."""
        if c <= 0: return x  # Not hyperbolic if curvature is non-positive
        sqrt_c = math.sqrt(max(c, eps))  # Ensure c is positive for sqrt
        max_norm = (radius / sqrt_c) * (1.0 - eps)  # Max Euclidean norm allowed

        # Use float32 for norm calculation for stability, then cast back
        original_dtype = x.dtype
        x_norm_sq = torch.sum(x.float().pow(2), dim=-1, keepdim=True)
        norm = torch.sqrt(torch.clamp(x_norm_sq, min=0) + eps)

        cond = norm > max_norm
        # Ensure scale_factor is calculated and applied using the same dtype as x
        scale_factor = torch.where(cond, max_norm / (norm + eps), torch.ones_like(norm)).to(original_dtype)
        clipped_x = x * scale_factor

        # Final sanity check
        if not torch.isfinite(clipped_x).all():
            print("NaN/Inf detected *after* poincare_clip. Replacing.")
            clipped_x = torch.nan_to_num(clipped_x, nan=0.0)  # Replace with 0
        return clipped_x

    @staticmethod
    def scale_aware_exponential_map(v: torch.Tensor, c: float, scale: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
        """
        Maps a tangent vector v at the origin to the Poincare ball with scale awareness.
        Implements the scale-aware version:
        exp_0^c,s(v) = tanh(s * sqrt(c) * ||v|| / 2) * v / (sqrt(c) * ||v||)
        """
        if c <= 0: return v  # No mapping needed for Euclidean space
        original_dtype = v.dtype
        
        # Compute norm in float32 for stability
        v_norm_sq = torch.sum(v.float().pow(2), dim=-1, keepdim=True)
        v_norm = torch.sqrt(torch.clamp(v_norm_sq, min=0) + eps)
        sqrt_c = math.sqrt(max(c, eps))
        
        # Apply scale to the hyperbolic radius calculation
        scaled_radius = scale * sqrt_c * v_norm
        
        # Use tanh for the scale-aware map
        tanh_term = torch.tanh(scaled_radius).to(original_dtype)
        
        # Ensure lambda calculation uses consistent dtype
        lambda_v = torch.where(
            v_norm > eps, 
            tanh_term / (sqrt_c * v_norm + eps).to(original_dtype), 
            torch.ones_like(v_norm).to(original_dtype)
        )

        mapped_v = lambda_v * v
        # Clip result to ensure it stays in the ball
        return HyperbolicUtils.poincare_clip(mapped_v, c)

    @staticmethod
    def scale_aware_logarithmic_map(y: torch.Tensor, c: float, scale: float = 1.0, eps: float = 1e-7) -> torch.Tensor:
        """
        Maps a point y in the Poincare ball back to the tangent space at the origin with scale awareness.
        Implements the scale-aware version:
        log_0^c,s(y) = (1/s) * atanh(sqrt(c) * ||y||) * y / (sqrt(c) * ||y||)
        """
        if c <= 0: return y  # No mapping needed for Euclidean space
        original_dtype = y.dtype
        
        # Clip input first to ensure it's inside the ball
        y_clipped = HyperbolicUtils.poincare_clip(y, c)
        
        # Compute norm in float32 for stability
        y_norm_sq = torch.sum(y_clipped.float().pow(2), dim=-1, keepdim=True)
        y_norm = torch.sqrt(torch.clamp(y_norm_sq, min=0) + eps)
        sqrt_c = math.sqrt(max(c, eps))
        
        # Clamp input to atanh carefully
        arctanh_input = torch.clamp(sqrt_c * y_norm, min=-1.0 + eps, max=1.0 - eps)
        atanh_term = torch.atanh(arctanh_input).to(original_dtype)
        
        # Apply inverse scale to the hyperbolic radius calculation
        # Division by scale implements the inverse mapping
        scaled_atanh = atanh_term / scale
        
        # Ensure lambda calculation uses consistent dtype
        lambda_y = torch.where(
            y_norm > eps, 
            scaled_atanh / (sqrt_c * y_norm + eps).to(original_dtype), 
            torch.ones_like(y_norm).to(original_dtype)
        )

        mapped_y = lambda_y * y_clipped
        
        # Handle numerical instabilities
        if not torch.isfinite(mapped_y).all():
            print("NaN/Inf detected in logarithmic_map output. Replacing.")
            mapped_y = torch.nan_to_num(mapped_y, nan=0.0)
            
        return mapped_y

# =====================================================================
# Quaternion Operations for Tangent Space Rotations 
# =====================================================================
def hamilton_product(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Hamilton product of two quaternions (or batches of quaternions).
    Input shapes can be:
    - Individual quaternions: [4] × [4] -> [4]
    - Batched quaternions: [B, 4] × [B, 4] -> [B, 4]
    - Batched with broadcasting: [B, 1, 4] × [1, N, 4] -> [B, N, 4]
    """
    # Ensure inputs are broadcastable and extract components
    q1_shape = list(q1.shape)
    q2_shape = list(q2.shape)
    
    # Ensure both have the same number of dimensions
    while len(q1_shape) < len(q2_shape):
        q1_shape.insert(0, 1)
    while len(q2_shape) < len(q1_shape):
        q2_shape.insert(0, 1)
    
    q1 = q1.view(q1_shape)
    q2 = q2.view(q2_shape)

    # Extract quaternion components
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    # Hamilton product formula
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Stack components back into a tensor
    return torch.stack([w, x, y, z], dim=-1)

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion (negate vector part)."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

def quat_rotate_via_pvq(v: torch.Tensor, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Rotates vector v (represented as a quaternion) using p * v * q.
    Handles batching correctly.
    """
    # Ensure dimensions are 4 for quaternion operations
    if v.shape[-1] != 4 or p.shape[-1] != 4 or q.shape[-1] != 4:
        raise ValueError(f"Inputs must be 4D for quat_rotate_via_pvq, shapes: v={v.shape}, p={p.shape}, q={q.shape}")

    # Expand p and q for broadcasting if they have fewer dimensions
    if p.dim() < v.dim():
        p = p.expand_as(v)
    if q.dim() < v.dim():
        q = q.expand_as(v)

    # Perform rotation: p * v * q
    pv = hamilton_product(p, v)
    pvq = hamilton_product(pv, q)
    return pvq

# =====================================================================
# SO(n) Rotation Implementation 
# =====================================================================
class SO_n_Rotation(nn.Module):
    """
    Implements SO(n) rotation matrix using exponential map from skew-symmetric matrices.
    This provides a differentiable parameterization of rotation matrices.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Initialize skew-symmetric parameters close to zero (near-identity rotation)
        self.skew_params = nn.Parameter(torch.randn(dim, dim) * 0.01)
        
    def _get_rotation_matrix(self) -> torch.Tensor:
        """Constructs a rotation matrix from skew-symmetric parameters."""
        # Create skew-symmetric matrix: A = P - P^T
        skew_matrix = self.skew_params - self.skew_params.T
        # Compute rotation matrix using matrix exponential: R = exp(A)
        R = torch.matrix_exp(skew_matrix)
        return R
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotation to inputs.
        Input shape: [..., dim]
        Output shape: [..., dim]
        """
        R = self._get_rotation_matrix()  # [dim, dim]
        
        # Handle multi-dimensional inputs
        original_shape = x.shape
        # Reshape to [-1, dim] for batch matrix multiplication
        x_flat = x.reshape(-1, self.dim)
        # Apply rotation: x_rotated = x @ R
        x_rotated = torch.matmul(x_flat, R)
        # Restore original dimensions
        return x_rotated.reshape(original_shape)

# =====================================================================
# Tangent Space Rotation with Proper Broadcasting
# =====================================================================
class TangentSpaceRotation(nn.Module):
    """
    Applies rotation to vectors in tangent space, properly handling
    broadcasting for main, boundary, and descriptor vectors.
    """
    def __init__(self, dim: int, rotation_type: str = 'so_n'):
        super().__init__()
        self.dim = dim
        self.rotation_type = rotation_type

        if rotation_type == 'so_n':
            # Use SO(n) rotation for arbitrary dimensions
            self.rotation = SO_n_Rotation(dim)
            print(f"Using SO({dim}) rotation")
        elif rotation_type == 'quat':
            if dim != 4:
                raise ValueError("Quaternion rotation requires dim=4")
            # Parameterize with two quaternions p, q for p*v*q rotation
            init_p = torch.tensor([1.0, 0.0, 0.0, 0.0]) + torch.randn(4) * 0.01
            init_q = torch.tensor([1.0, 0.0, 0.0, 0.0]) + torch.randn(4) * 0.01
            self.quat_p = nn.Parameter(init_p)
            self.quat_q = nn.Parameter(init_q)
            print("Using Quaternion rotation")
        elif rotation_type == 'identity':
            # No learnable parameters for identity rotation
            print(f"Using Identity (no rotation)")
        else:
            raise ValueError(f"Unsupported rotation type: {rotation_type}")

    def forward(self, v_main: torch.Tensor, v_boundaries_tangent: torch.Tensor = None, 
                v_descriptor: torch.Tensor = None) -> tuple:
        """
        Applies rotation to main, boundary, and descriptor vectors.

        Args:
            v_main: Tensor of shape [batch_size, seq_len, dim] or [..., dim]
            v_boundaries_tangent: Tensor of shape [num_points, dim] or None
            v_descriptor: Tensor of shape [dim] or [1, 1, dim]

        Returns:
            Tuple of rotated tensors (v_main_rotated, v_boundaries_rotated, v_descriptor_rotated)
        """
        # Validate inputs
        if v_main.shape[-1] != self.dim:
            raise ValueError(f"Main vector dimension mismatch: {v_main.shape[-1]} != {self.dim}")
        
        # Identity rotation - just pass through
        if self.rotation_type == 'identity':
            return v_main, v_boundaries_tangent, v_descriptor
        
        # Get device for operations
        device = v_main.device
        
        # Prepare descriptor (ensure it's [1, 1, dim] for broadcasting)
        if v_descriptor is not None:
            if v_descriptor.dim() == 1:
                v_descriptor = v_descriptor.view(1, 1, self.dim).to(device)
            elif v_descriptor.shape[-1] != self.dim:
                raise ValueError(f"Descriptor dimension mismatch: {v_descriptor.shape[-1]} != {self.dim}")
        
        # Apply rotation based on type
        if self.rotation_type == 'so_n':
            # Apply SO(n) rotation to all vectors
            v_main_rotated = self.rotation(v_main)
            
            v_boundaries_rotated = None
            if v_boundaries_tangent is not None:
                v_boundaries_rotated = self.rotation(v_boundaries_tangent)
                
            v_descriptor_rotated = None
            if v_descriptor is not None:
                v_descriptor_rotated = self.rotation(v_descriptor)
                
        elif self.rotation_type == 'quat':
            # Normalize quaternions to unit length
            p_norm = torch.norm(self.quat_p, p=2, dim=-1, keepdim=True).clamp(min=EPS)
            q_norm = torch.norm(self.quat_q, p=2, dim=-1, keepdim=True).clamp(min=EPS)
            unit_p = self.quat_p / p_norm
            unit_q = self.quat_q / q_norm
            
            # For main input [batch, seq, dim=4]
            # Reshape p and q for broadcasting: [1, 1, 4]
            p_broadcast = unit_p.view(1, 1, 4)
            q_broadcast = unit_q.view(1, 1, 4)
            
            # Rotate main vectors
            v_main_rotated = quat_rotate_via_pvq(v_main, p_broadcast, q_broadcast)
            
            # Rotate boundary vectors if provided
            v_boundaries_rotated = None
            if v_boundaries_tangent is not None:
                # Handle broadcasting for boundary points [num_points, dim=4]
                if v_boundaries_tangent.dim() == 2:
                    # Reshape p and q to [1, 4] for broadcasting with [num_points, 4]
                    p_b = unit_p.view(1, 4)
                    q_b = unit_q.view(1, 4)
                    v_boundaries_rotated = quat_rotate_via_pvq(v_boundaries_tangent, p_b, q_b)
                else:
                    # Handle other shapes if needed
                    p_expanded = unit_p.expand_as(v_boundaries_tangent[..., :1].expand(-1, -1, -1, 4))
                    q_expanded = unit_q.expand_as(v_boundaries_tangent[..., :1].expand(-1, -1, -1, 4))
                    v_boundaries_rotated = quat_rotate_via_pvq(v_boundaries_tangent, p_expanded, q_expanded)
            
            # Rotate descriptor if provided
            v_descriptor_rotated = None
            if v_descriptor is not None:
                v_descriptor_rotated = quat_rotate_via_pvq(v_descriptor, p_broadcast, q_broadcast)
        
        # Check for NaN/Inf
        outputs = [v_main_rotated, v_boundaries_rotated, v_descriptor_rotated]
        cleaned_outputs = []
        
        for output in outputs:
            if output is not None and not torch.isfinite(output).all():
                print(f"NaN/Inf detected in rotation output. Replacing with zeros.")
                cleaned_outputs.append(torch.nan_to_num(output, nan=0.0))
            else:
                cleaned_outputs.append(output)
                
        return tuple(cleaned_outputs)

# =====================================================================
# Inter-Level Transform with Proper Broadcasting
# =====================================================================
class InterLevelTransform(nn.Module):
    """
    Handles transformation between tangent spaces of different hyperbolic levels.
    Properly broadcasts operations between main, boundary and descriptor vectors.
    """
    def __init__(self, in_dim: int, out_dim: int, transform_type: str, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        self.transform_type = transform_type
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Determine hidden dimension for MLP
        if transform_type == 'mlp':
            h_dim = hidden_dim or max(16, (in_dim + out_dim) // 2)
            self.transform = nn.Sequential(
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),  # Stable normalization
                nn.GELU(),  # Smooth activation
                nn.Dropout(dropout),
                nn.Linear(h_dim, out_dim)
            )
            print(f"InterLevelTransform: MLP {in_dim}->{h_dim}->{out_dim}")
        elif transform_type == 'linear':
            self.transform = nn.Linear(in_dim, out_dim)
            print(f"InterLevelTransform: Linear {in_dim}->{out_dim}")
        else:
            raise ValueError(f"Unsupported transform_type: {transform_type}")
        
        # Initialize weights with Xavier uniform
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, v_main: torch.Tensor, v_boundaries: torch.Tensor = None, 
                v_descriptor: torch.Tensor = None) -> tuple:
        """
        Applies transformation to main, boundary, and descriptor vectors.
        
        Args:
            v_main: Tensor of shape [batch_size, seq_len, in_dim]
            v_boundaries: Tensor of shape [num_points, in_dim] or None
            v_descriptor: Tensor of shape [1, 1, in_dim] or None
            
        Returns:
            Tuple of transformed tensors (v_main_transformed, v_boundaries_transformed, v_descriptor_transformed)
        """
        # Apply transformation to main vectors
        v_main_transformed = self.transform(v_main)
        
        # Transform boundary vectors if provided
        v_boundaries_transformed = None
        if v_boundaries is not None:
            v_boundaries_transformed = self.transform(v_boundaries)
            
        # Transform descriptor if provided
        v_descriptor_transformed = None
        if v_descriptor is not None:
            v_descriptor_transformed = self.transform(v_descriptor)
            
        # Check for NaN/Inf
        outputs = [v_main_transformed, v_boundaries_transformed, v_descriptor_transformed]
        cleaned_outputs = []
        
        for output in outputs:
            if output is not None and not torch.isfinite(output).all():
                print(f"NaN/Inf detected in transform output. Replacing with zeros.")
                cleaned_outputs.append(torch.nan_to_num(output, nan=0.0))
            else:
                cleaned_outputs.append(output)
                
        return tuple(cleaned_outputs)

# =====================================================================
# Boundary Manifold with Learnable Points
# =====================================================================
class BoundaryManifold(nn.Module):
    """
    Represents the learnable boundary points for a WuBu level.
    These points define sub-manifolds in the tangent space.
    """
    def __init__(self, level_idx: int, num_points: int, point_dim: int, init_scale: float = 0.01):
        super().__init__()
        self.level_idx = level_idx
        self.num_points = num_points
        self.point_dim = point_dim
        
        # Initialize tangent points as learnable parameters
        if num_points > 0 and point_dim > 0:
            # Initialize with small random values
            tangent_points = torch.randn(num_points, point_dim) * init_scale
            self.tangent_points = nn.Parameter(tangent_points)
            print(f"BoundaryManifold L{level_idx}: {num_points} points in {point_dim}D")
        else:
            # Register None if no points
            self.register_parameter('tangent_points', None)
            print(f"BoundaryManifold L{level_idx}: No boundary points")
            
    def get_tangent_vectors_at_origin(self) -> torch.Tensor:
        """Returns the current boundary points (tangent vectors at origin)."""
        if self.tangent_points is None:
            return None
            
        # Stability check
        if not torch.isfinite(self.tangent_points).all():
            print(f"NaN/Inf in BoundaryManifold L{self.level_idx}. Reinitializing.")
            self.tangent_points.data.normal_(0, 0.01)
            
        return self.tangent_points

# =====================================================================
# WuBu Nesting Level Implementation
# =====================================================================
class WuBuNestingLevel(nn.Module):
    """
    Implements a single level of the WuBu Nesting architecture.
    Handles processing within a level including tangent space operations.
    """
    def __init__(self, level_idx: int, dim: int, 
                 learnable_curvature: bool = True, 
                 learnable_scale: bool = True,
                 learnable_spread: bool = True,
                 initial_curvature: float = 1.0,
                 initial_scale: float = 1.0,
                 initial_spread: float = 1.0,
                 use_level_descriptors: bool = True,
                 use_level_spread: bool = True,
                 use_flow: bool = True,
                 descriptor_init_scale: float = 0.01,
                 min_curvature: float = 1e-5,
                 min_scale: float = 1e-5,
                 min_spread: float = 1e-5,
                 dropout: float = 0.1):
        super().__init__()
        self.level_idx = level_idx
        self.dim = dim
        self.use_ld = use_level_descriptors
        self.use_spread = use_level_spread
        self.use_flow = use_flow
        self.dropout = dropout
        
        # Minimum values for constrained parameters
        self.min_curvature = min_curvature
        self.min_scale = min_scale
        self.min_spread = min_spread
        
        # --- Curvature (c_i) ---
        # Parameterize as log(c - min_c) for positivity
        init_c = max(initial_curvature, min_curvature + 1e-4)
        log_init_c = torch.tensor(math.log(init_c - min_curvature))
        if learnable_curvature:
            self.log_curvature = nn.Parameter(log_init_c)
        else:
            self.register_buffer('log_curvature', log_init_c)
            
        # --- Scale (s_i) ---
        # Parameterize as log(s - min_s) for positivity
        init_s = max(initial_scale, min_scale + 1e-4)
        log_init_s = torch.tensor(math.log(init_s - min_scale))
        if learnable_scale:
            self.log_scale = nn.Parameter(log_init_s)
        else:
            self.register_buffer('log_scale', log_init_s)
            
        # --- Level Descriptor (ld_i) ---
        if use_level_descriptors:
            self.level_descriptor = nn.Parameter(torch.randn(dim) * descriptor_init_scale)
        else:
            self.register_buffer('level_descriptor', torch.zeros(dim))
            
        # --- Spread (sigma_i) ---
        if self.use_spread:
            init_spread = max(initial_spread, min_spread + 1e-4)
            log_init_spread = torch.tensor(math.log(init_spread - min_spread))
            if learnable_spread:
                self.log_spread = nn.Parameter(log_init_spread)
            else:
                self.register_buffer('log_spread', log_init_spread)
        else:
            # Fixed minimum value if spread is not used
            self.register_buffer('log_spread', torch.tensor(math.log(1e-4)))
        
        # --- Tangent Space MLP ---
        # For processing combined inputs (tangent vector, relative vectors, descriptor, spread)
        input_dim = dim  # Base dimension from main vector
        input_dim += dim  # Add dim for relative vectors
        if use_level_descriptors:
            input_dim += dim  # Add dim for level descriptor
        if self.use_spread:
            input_dim += 1  # Add 1 for scalar spread
            
        # MLP to process combined tangent space inputs
        hidden_dim = max(16, input_dim // 2)
        self.tangent_combiner = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )
        
        # --- Tangent Flow (optional) ---
        if use_flow:
            self.tangent_flow = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim)
            )
            self.flow_scale = 1.0
        else:
            self.tangent_flow = None
            self.flow_scale = 0.0
            
        # Initialize weights
        self.apply(self._init_weights)
        
        print(f"WuBuLevel {level_idx} (Dim {dim}): c={initial_curvature:.3f}, s={initial_scale:.3f}")
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def get_curvature(self) -> torch.Tensor:
        """Returns the constrained positive curvature parameter."""
        return torch.exp(self.log_curvature) + self.min_curvature
        
    def get_scale(self) -> torch.Tensor:
        """Returns the constrained positive scale parameter."""
        return torch.exp(self.log_scale) + self.min_scale
        
    def get_spread(self) -> torch.Tensor:
        """Returns the constrained positive spread parameter."""
        if not self.use_spread:
            return torch.tensor(self.min_spread, device=self.log_spread.device)
        return torch.exp(self.log_spread) + self.min_spread
    
    def forward(self, v_tangent_in: torch.Tensor, 
                relative_vectors_in: torch.Tensor = None,
                ld_tangent_in: torch.Tensor = None, 
                sigma_in: torch.Tensor = None) -> tuple:
        """
        Process tangent vectors through the WuBu level.
        
        Args:
            v_tangent_in: Tangent vectors [batch, seq, dim]
            relative_vectors_in: Relative vectors from previous level [batch, seq, dim]
            ld_tangent_in: Level descriptor from previous level [batch, seq, dim]
            sigma_in: Spread from previous level [batch, seq, 1] or [1]
            
        Returns:
            Tuple of (processed vectors in ball, tangent vectors, level descriptor, spread)
        """
        batch_size, seq_len, d_in = v_tangent_in.shape
        device = v_tangent_in.device
        
        # Validate input dimension
        if d_in != self.dim:
            raise ValueError(f"Input dimension mismatch: {d_in} != {self.dim}")
            
        # Get current parameters
        curvature = self.get_curvature().to(device)
        scale = self.get_scale().to(device)
        spread = self.get_spread().to(device)
        
        # --- Prepare inputs for tangent combiner ---
        inputs_to_combine = [v_tangent_in]
        
        # Add relative vectors if provided or zeros
        if relative_vectors_in is not None:
            if relative_vectors_in.shape == (batch_size, seq_len, self.dim):
                inputs_to_combine.append(relative_vectors_in)
            else:
                print(f"L{self.level_idx}: Unexpected relative_vectors shape. Using zeros.")
                inputs_to_combine.append(torch.zeros_like(v_tangent_in))
        else:
            # Use zeros if no relative vectors provided
            inputs_to_combine.append(torch.zeros_like(v_tangent_in))
        
        # Add level descriptor input if enabled
        if self.use_ld:
            if ld_tangent_in is None:
                inputs_to_combine.append(torch.zeros_like(v_tangent_in))
            elif ld_tangent_in.shape == (batch_size, seq_len, self.dim):
                inputs_to_combine.append(ld_tangent_in)
            else:
                # Try to reshape to match dimensions
                try:
                    if ld_tangent_in.dim() == 1 and ld_tangent_in.shape[0] == self.dim:
                        # Expand single vector to batch size
                        expanded_ld = ld_tangent_in.view(1, 1, self.dim).expand(batch_size, seq_len, self.dim)
                        inputs_to_combine.append(expanded_ld)
                    else:
                        print(f"L{self.level_idx}: Cannot reshape level descriptor")
                        inputs_to_combine.append(torch.zeros_like(v_tangent_in))
                except:
                    print(f"L{self.level_idx}: Error expanding level descriptor")
                    inputs_to_combine.append(torch.zeros_like(v_tangent_in))
        
        # Add spread input if enabled
        if self.use_spread:
            if sigma_in is None:
                sigma_in_tensor = torch.zeros(batch_size, seq_len, 1, device=device)
            elif sigma_in.numel() == 1:  # Scalar spread
                sigma_in_tensor = sigma_in.expand(batch_size, seq_len, 1)
            elif sigma_in.shape == (batch_size, seq_len):  # Per-sequence spread
                sigma_in_tensor = sigma_in.unsqueeze(-1)
            elif sigma_in.shape == (batch_size, seq_len, 1):  # Already correct shape
                sigma_in_tensor = sigma_in
            else:
                print(f"L{self.level_idx}: Unexpected sigma shape. Using zeros.")
                sigma_in_tensor = torch.zeros(batch_size, seq_len, 1, device=device)
            
            inputs_to_combine.append(sigma_in_tensor)
        
        # Concatenate all inputs for the combiner MLP
        combined_inputs = torch.cat(inputs_to_combine, dim=-1)
        
        # Ensure inputs are finite
        if not torch.isfinite(combined_inputs).all():
            combined_inputs = torch.nan_to_num(combined_inputs, nan=0.0)
        
        # --- Process combined inputs ---
        # Pass through MLP combiner
        v_combined = self.tangent_combiner(combined_inputs)
        
        # Apply optional flow
        if self.use_flow and self.tangent_flow is not None:
            flow_displacement = self.tangent_flow(v_combined)
            # Ensure displacement is finite
            if not torch.isfinite(flow_displacement).all():
                flow_displacement = torch.nan_to_num(flow_displacement, nan=0.0)
            # Add scaled displacement
            v_combined = v_combined + flow_displacement * self.flow_scale
        
        # --- Map to Hyperbolic Space ---
        # Use scale-aware exponential map
        x_hyperbolic = HyperbolicUtils.scale_aware_exponential_map(
            v_combined, curvature.item(), scale.item())
        
        # --- Map back to Tangent Space ---
        # Use scale-aware logarithmic map
        v_tangent_out = HyperbolicUtils.scale_aware_logarithmic_map(
            x_hyperbolic, curvature.item(), scale.item())
        
        # Return processed states
        return x_hyperbolic, v_tangent_out, self.level_descriptor, spread
# =====================================================================
# Complete WuBu Nesting Model
# =====================================================================
class WuBuNestingModel(nn.Module):
    """
    Full implementation of the WuBu Nesting architecture with proper nested hyperbolic
    spaces and tangent space transitions, including boundary manifolds, rotation, and
    relative vector computation.
    """
    def __init__(self, input_dim, output_dim, config=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Set default configuration or use provided config
        if config is None:
            config = {
                "num_levels": 3,
                "hyperbolic_dims": [128, 64, 32],
                "boundary_points_per_level": [5, 4, 3],
                "rotation_types": ["so_n", "so_n"],
                "transform_types": ["linear", "linear"],
                "learnable_curvature": True,
                "learnable_scales": True,
                "learnable_spread": True,
                "use_level_descriptors": True,
                "use_level_spread": True,
                "use_tangent_flow": True,
                "initial_curvatures": [1.0, 2.0, 4.0],
                "initial_scales": [1.0, 0.8, 0.5],
                "initial_spread_values": [1.0, 0.7, 0.4],
                "dropout": 0.1,
                "level_descriptor_init_scale": 0.01,
                "relative_vector_aggregation": "mean",
            }
        
        # Extract configuration parameters
        self.num_levels = config.get("num_levels", 3)
        self.hyperbolic_dims = config.get("hyperbolic_dims", [128, 64, 32])
        self.boundary_points = config.get("boundary_points_per_level", [5, 4, 3])
        self.rotation_types = config.get("rotation_types", ["so_n"] * (self.num_levels - 1))
        self.transform_types = config.get("transform_types", ["linear"] * (self.num_levels - 1))
        self.use_level_descriptors = config.get("use_level_descriptors", True)
        self.use_level_spread = config.get("use_level_spread", True)
        self.use_tangent_flow = config.get("use_tangent_flow", True)
        self.dropout = config.get("dropout", 0.1)
        self.descriptor_init_scale = config.get("level_descriptor_init_scale", 0.01)
        self.learnable_curvature = config.get("learnable_curvature", True)
        self.learnable_scales = config.get("learnable_scales", True)
        self.learnable_spread = config.get("learnable_spread", True)
        self.initial_curvatures = config.get("initial_curvatures", [1.0] * self.num_levels)
        self.initial_scales = config.get("initial_scales", [1.0] * self.num_levels)
        self.initial_spread_values = config.get("initial_spread_values", [1.0] * self.num_levels)
        self.relative_vector_aggregation = config.get("relative_vector_aggregation", "mean")
        
        # Validate configurations
        if len(self.hyperbolic_dims) != self.num_levels:
            raise ValueError(f"Hyperbolic dimensions list length {len(self.hyperbolic_dims)} "
                             f"must match num_levels {self.num_levels}")
        if len(self.boundary_points) != self.num_levels:
            raise ValueError(f"Boundary points list length {len(self.boundary_points)} "
                             f"must match num_levels {self.num_levels}")
        if len(self.initial_curvatures) != self.num_levels:
            raise ValueError(f"Initial curvatures list length {len(self.initial_curvatures)} "
                             f"must match num_levels {self.num_levels}")
        if len(self.initial_scales) != self.num_levels:
            raise ValueError(f"Initial scales list length {len(self.initial_scales)} "
                             f"must match num_levels {self.num_levels}")
        if len(self.initial_spread_values) != self.num_levels:
            raise ValueError(f"Initial spread values list length {len(self.initial_spread_values)} "
                             f"must match num_levels {self.num_levels}")
        if len(self.rotation_types) != self.num_levels - 1:
            raise ValueError(f"Rotation types list length {len(self.rotation_types)} "
                             f"must be num_levels - 1 = {self.num_levels - 1}")
        if len(self.transform_types) != self.num_levels - 1:
            raise ValueError(f"Transform types list length {len(self.transform_types)} "
                             f"must be num_levels - 1 = {self.num_levels - 1}")
        
        # --- Input and Output Projections ---
        # Input projection: input_dim -> first hyperbolic dim
        self.input_to_tangent = nn.Linear(input_dim, self.hyperbolic_dims[0])
        
        # Output projection: sum of all hyperbolic dims -> output_dim
        combined_dim = sum(self.hyperbolic_dims)
        self.tangent_to_output = nn.Linear(combined_dim, output_dim)
        
        # --- Create WuBu Levels ---
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = WuBuNestingLevel(
                level_idx=i,
                dim=self.hyperbolic_dims[i],
                learnable_curvature=self.learnable_curvature,
                learnable_scale=self.learnable_scales,
                learnable_spread=self.learnable_spread,
                initial_curvature=self.initial_curvatures[i],
                initial_scale=self.initial_scales[i],
                initial_spread=self.initial_spread_values[i],
                use_level_descriptors=self.use_level_descriptors,
                use_level_spread=self.use_level_spread,
                use_flow=self.use_tangent_flow,
                descriptor_init_scale=self.descriptor_init_scale,
                dropout=self.dropout
            )
            self.levels.append(level)
        
        # --- Create Boundary Manifolds ---
        self.boundaries = nn.ModuleList()
        for i in range(self.num_levels):
            boundary = BoundaryManifold(
                level_idx=i,
                num_points=self.boundary_points[i],
                point_dim=self.hyperbolic_dims[i],
                init_scale=0.1  # Small initialization for stability
            )
            self.boundaries.append(boundary)
        
        # --- Create Rotation Modules (between levels) ---
        self.rotations = nn.ModuleList()
        for i in range(self.num_levels - 1):
            rotation = TangentSpaceRotation(
                dim=self.hyperbolic_dims[i],
                rotation_type=self.rotation_types[i]
            )
            self.rotations.append(rotation)
        
        # --- Create Transform Modules (between levels) ---
        self.transforms = nn.ModuleList()
        for i in range(self.num_levels - 1):
            transform = InterLevelTransform(
                in_dim=self.hyperbolic_dims[i],
                out_dim=self.hyperbolic_dims[i+1],
                transform_type=self.transform_types[i],
                dropout=self.dropout
            )
            self.transforms.append(transform)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Print model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"WuBuNestingModel initialized with {self.num_levels} levels")
        print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        """
        Forward pass through the WuBu Nesting model.
        
        Args:
            x: Input tensor [batch_size, ..., input_dim]
            
        Returns:
            Output tensor [batch_size, ..., output_dim]
        """
        # Get shapes and device
        original_shape = x.shape
        batch_dims = original_shape[:-1]  # All dims except the last one
        device = x.device
        
        # Reshape input to [batch_size, num_elements, input_dim] for processing
        # This flattens any spatial dimensions into a sequence dimension
        if len(original_shape) > 2:
            flat_batch_size = np.prod(batch_dims)
            x_flat = x.reshape(flat_batch_size, self.input_dim)
        else:
            # If already [batch_size, input_dim], add sequence dim of 1
            x_flat = x.unsqueeze(1)
            flat_batch_size = original_shape[0]
        
        # Project input to first tangent space
        current_tangent = self.input_to_tangent(x_flat)
        current_tangent = current_tangent.view(flat_batch_size, 1, self.hyperbolic_dims[0])
        
        # Storage for level outputs (for final aggregation)
        level_tangent_outputs = []
        
        # Initial values for inter-level passing
        aggregated_relative_vectors = None
        current_ld_tangent = None
        current_sigma = None
        
        # Process through levels
        for i in range(self.num_levels):
            level_module = self.levels[i]
            boundary_module = self.boundaries[i]
            
            # Process through this level
            _, v_tangent_out, ld_param, sigma_param = level_module(
                v_tangent_in=current_tangent,
                relative_vectors_in=aggregated_relative_vectors,
                ld_tangent_in=current_ld_tangent,
                sigma_in=current_sigma
            )
            
            # Store the tangent output for final aggregation
            level_tangent_outputs.append(v_tangent_out)
            
            # --- Inter-Level Transition (if not the last level) ---
            if i < self.num_levels - 1:
                rotation_module = self.rotations[i]
                transform_module = self.transforms[i]
                
                # Get boundary points for this level
                boundary_points = boundary_module.get_tangent_vectors_at_origin()
                
                # Ensure level descriptor is on correct device/dtype
                ld_param_ready = ld_param.to(device).view(1, 1, -1)
                
                # 1. Apply rotation to main output, boundaries, and level descriptor
                v_main_rotated, v_boundaries_rotated, ld_rotated = rotation_module(
                    v_main=v_tangent_out,
                    v_boundaries_tangent=boundary_points,
                    v_descriptor=ld_param_ready
                )
                
                # 2. Apply transformation to map to next level's tangent space
                v_next_main, v_boundaries_transformed, ld_next = transform_module(
                    v_main=v_main_rotated,
                    v_boundaries=v_boundaries_rotated,
                    v_descriptor=ld_rotated
                )
                
                # 3. Calculate relative vectors between main and boundary points
                relative_vectors = None
                if v_boundaries_transformed is not None and v_boundaries_transformed.numel() > 0:
                    # Properly handle broadcasting for subtraction
                    # main: [batch, seq, 1, dim]
                    main_expanded = v_next_main.unsqueeze(2)
                    # boundaries: [1, 1, num_points, dim]
                    boundaries_expanded = v_boundaries_transformed.unsqueeze(0).unsqueeze(0)
                    
                    # Calculate vectors: main - boundary
                    # Shape: [batch, seq, num_points, dim]
                    relative_vectors_calc = main_expanded - boundaries_expanded
                    
                    # Aggregate relative vectors based on configuration
                    if self.relative_vector_aggregation == "mean":
                        aggregated_relative_vectors = torch.mean(relative_vectors_calc, dim=2)
                    elif self.relative_vector_aggregation == "sum":
                        aggregated_relative_vectors = torch.sum(relative_vectors_calc, dim=2)
                    else:  # "none" or invalid value
                        aggregated_relative_vectors = None
                
                # 4. Prepare inputs for next level
                current_tangent = v_next_main
                current_ld_tangent = ld_next.expand(flat_batch_size, 1, -1)
                current_sigma = sigma_param  # Scalar or tensor to be expanded as needed
        
        # Aggregate outputs from all levels (concatenation)
        aggregated_tangent = torch.cat(level_tangent_outputs, dim=-1)
        
        # Project to output space
        output = self.tangent_to_output(aggregated_tangent)
        
        # Reshape output back to match input batch dimensions
        if len(original_shape) > 2:
            output = output.view(*batch_dims, self.output_dim)
        else:
            output = output.squeeze(1)  # Remove sequence dimension
            
        return output
# =====================================================================
# Usage Example
# =====================================================================
def test_wubu_nesting():
    """
    Demonstrates how to create and use a WuBu Nesting model.
    """
    # Model configuration
    config = {
        "num_levels": 3,
        "hyperbolic_dims": [128, 64, 32],  # Decreasing dimensions for deeper levels
        "boundary_points_per_level": [5, 5, 5],
        "rotation_types": ["quat", "so_n"],  # Quaternion for 1->2, SO(n) for 2->3
        "transform_types": ["mlp", "mlp"],
        "learnable_curvature": True,
        "learnable_scales": True,
        "learnable_spread": True,
        "use_level_descriptors": True,
        "use_level_spread": True,
        "use_tangent_flow": True,
        "initial_curvatures": [1.0, 2.0, 4.0],  # Increasing curvature for deeper levels
        "initial_scales": [1.0, 0.8, 0.5],      # Decreasing scale for deeper levels
        "initial_spread_values": [1.0, 0.7, 0.4],
        "dropout": 0.1,
        "level_descriptor_init_scale": 0.01,
        "relative_vector_aggregation": "mean",
    }
    
    # Create model instance
    input_dim = 256
    output_dim = 10
    model = WuBuNestingModel(input_dim, output_dim, config)
    
    # Test forward pass with batch input
    batch_size = 16
    x = torch.randn(batch_size, input_dim)
    
    # Model forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with spatial dimensions (e.g., images)
    height, width = 8, 8
    x_spatial = torch.randn(batch_size, height, width, input_dim)
    output_spatial = model(x_spatial)
    print(f"Spatial input shape: {x_spatial.shape}")
    print(f"Spatial output shape: {output_spatial.shape}")
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print level parameters
    for i, level in enumerate(model.levels):
        c = level.get_curvature().item()
        s = level.get_scale().item()
        sigma = level.get_spread().item()
        print(f"Level {i}: Curvature={c:.4f}, Scale={s:.4f}, Spread={sigma:.4f}")
    
    return model

if __name__ == "__main__":
    # Run the test
    model = test_wubu_nesting()
    print("WuBu Nesting test completed successfully!")