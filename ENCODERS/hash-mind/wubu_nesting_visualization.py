import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import os

# Import the WuBu Nesting implementation
from wubu_nesting_impl import (
    HyperbolicUtils, 
    BoundaryManifold, 
    WuBuNestingLevel,
    WuBuNestingModel
)

# =====================================================================
# Visualization Utilities
# =====================================================================

def visualize_poincare_disk(points, boundary_points=None, title="Poincaré Disk", 
                            curvature=1.0, scale=1.0, ax=None, show=True, 
                            point_colors=None, boundary_colors=None,
                            point_labels=None, boundary_labels=None,
                            save_path=None):
    """
    Visualize points in the Poincaré disk model of hyperbolic space.
    
    Args:
        points: Tensor of shape [N, 2] representing points in the Poincaré disk
        boundary_points: Optional tensor of shape [M, 2] for boundary manifold points
        title: Title for the plot
        curvature: Curvature parameter c
        scale: Scale parameter s
        ax: Optional matplotlib Axes to plot on
        show: Whether to show the plot
        point_colors: Optional colors for the points
        boundary_colors: Optional colors for the boundary points
        point_labels: Optional labels for the points
        boundary_labels: Optional labels for the boundary points
        save_path: Optional path to save the figure
    """
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw the unit circle (boundary of the Poincaré disk)
    circle = Circle((0, 0), 1, fill=False, color='black', linestyle='-', alpha=0.5)
    ax.add_patch(circle)
    
    # Draw hyperbolic circles at different radii (geodesic circles)
    hyperbolic_radii = [0.25, 0.5, 0.75, 0.9]
    for r in hyperbolic_radii:
        # Convert hyperbolic radius to Euclidean radius
        sqrt_c = np.sqrt(curvature)
        euclidean_r = np.tanh(r * sqrt_c / scale) / sqrt_c
        circle = Circle((0, 0), euclidean_r, fill=False, color='gray', linestyle='--', alpha=0.3)
        ax.add_patch(circle)
    
    # Default colors if not provided
    if point_colors is None:
        point_colors = 'blue'
    if boundary_colors is None:
        boundary_colors = 'red'
    
    # Plot the points
    if points is not None and points.shape[0] > 0:
        points_np = points.detach().cpu().numpy()
        ax.scatter(points_np[:, 0], points_np[:, 1], c=point_colors, marker='o', s=50, alpha=0.8, label='Points')
        
        # Add labels if provided
        if point_labels is not None:
            for i, (x, y) in enumerate(points_np):
                if i < len(point_labels):
                    ax.annotate(point_labels[i], (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Plot the boundary points
    if boundary_points is not None and boundary_points.shape[0] > 0:
        boundary_np = boundary_points.detach().cpu().numpy()
        ax.scatter(boundary_np[:, 0], boundary_np[:, 1], c=boundary_colors, marker='x', s=80, alpha=0.8, label='Boundary')
        
        # Add labels if provided
        if boundary_labels is not None:
            for i, (x, y) in enumerate(boundary_np):
                if i < len(boundary_labels):
                    ax.annotate(boundary_labels[i], (x, y), xytext=(5, -5), textcoords='offset points')
    
    # Set up the axes
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_title(f"{title} (c={curvature:.2f}, s={scale:.2f})")
    ax.legend()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    
    return ax

def visualize_poincare_nested(model, input_sample=None, level_idx=0, show=True, save_dir=None):
    """
    Visualize the nested structure of a WuBu model by showing how points and boundary
    manifolds are represented across the nested levels.
    
    Args:
        model: A WuBuNestingModel instance
        input_sample: Optional input tensor to visualize its flow through the model
        level_idx: Index of the level to visualize (must be 2D for visualization)
        show: Whether to show plots
        save_dir: Optional directory to save figures
    """
    # Check if the model is a WuBuNestingModel
    if not isinstance(model, WuBuNestingModel):
        raise ValueError("Model must be a WuBuNestingModel instance")
    
    # Create save directory if provided
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Process input sample through the model to get intermediate representations
    level_outputs = []
    if input_sample is not None:
        # Forward pass with hooks to capture intermediate outputs
        hooks = []
        for i, level in enumerate(model.levels):
            def get_hook(idx):
                def hook(module, input, output):
                    # Handle different output structures safely
                    input_tensor = input[0] if isinstance(input, tuple) and len(input) > 0 else None
                    
                    # For output, extract the tangent vector (second element) if available
                    output_tensor = None
                    if isinstance(output, tuple) and len(output) > 1:
                        output_tensor = output[1]  # Expected to be the tangent vector
                    
                    # Store the input and output tensors
                    level_outputs.append((idx, input_tensor, output_tensor))
                return hook
            
            handle = level.register_forward_hook(get_hook(i))
            hooks.append(handle)
        
        # Run the forward pass
        with torch.no_grad():
            model(input_sample)
        
        # Remove hooks
        for handle in hooks:
            handle.remove()
    
    # Create figure grid for all levels
    fig = plt.figure(figsize=(15, 5 * model.num_levels))
    
    # Plot for each level
    for i in range(model.num_levels):
        # Get level and boundary manifold
        level = model.levels[i]
        boundary = model.boundaries[i]
        
        # Only visualize if the dimension is 2
        if model.hyperbolic_dims[i] != 2:
            ax = fig.add_subplot(model.num_levels, 1, i+1)
            ax.text(0.5, 0.5, f"Level {i} has dimension {model.hyperbolic_dims[i]}, cannot visualize", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            continue
        
        # Get level parameters
        curvature = level.get_curvature().item()
        scale = level.get_scale().item()
        
        # Get boundary points
        boundary_points = boundary.get_tangent_vectors_at_origin()
        
        # Map the boundary points to the Poincaré disk
        if boundary_points is not None:
            manifold_points = HyperbolicUtils.scale_aware_exponential_map(
                boundary_points, curvature, scale)
        else:
            manifold_points = None
        
        # Get the level outputs if we ran an input sample
        level_point_inputs = None
        level_point_outputs = None
        level_point_labels = None
        level_point_inputs_mapped = None
        level_point_outputs_mapped = None
        
        if level_outputs:
            # Find the outputs for this level
            for idx, inputs, outputs in level_outputs:
                if idx == i:
                    # Inputs are already in tangent space
                    level_point_inputs = inputs
                    
                    # Map the output tangent vectors to the Poincaré disk
                    if inputs is not None and inputs.shape[-1] == 2:
                        # Just take a few points for visualization
                        sample_inputs = inputs[0, :5] if inputs.shape[1] > 5 else inputs[0]
                        level_point_inputs = sample_inputs
                        
                        # Map to Poincaré disk
                        level_point_inputs_mapped = HyperbolicUtils.scale_aware_exponential_map(
                            level_point_inputs, curvature, scale)
                    
                    # Map output tangent vectors to hyperbolic space
                    if outputs is not None and outputs.shape[-1] == 2:
                        # Just take a few points for visualization
                        sample_outputs = outputs[0, :5] if outputs.shape[1] > 5 else outputs[0]
                        level_point_outputs = sample_outputs
                        
                        # Map to Poincaré disk
                        level_point_outputs_mapped = HyperbolicUtils.scale_aware_exponential_map(
                            level_point_outputs, curvature, scale)
                        
                        # Create labels
                        level_point_labels = [f"P{j}" for j in range(len(sample_outputs))]
        
        # Create subplot
        ax = fig.add_subplot(model.num_levels, 1, i+1)
        
        # Plot the level
        title = f"Level {i} - Poincaré Disk"
        
        # Create points to visualize
        if level_point_outputs is not None and level_point_outputs_mapped is not None:
            visualize_poincare_disk(
                level_point_outputs_mapped, manifold_points, 
                title=title, curvature=curvature, scale=scale, 
                ax=ax, show=False,
                point_colors='blue', boundary_colors='red',
                point_labels=level_point_labels
            )
        else:
            # Just plot the boundary manifold
            visualize_poincare_disk(
                None, manifold_points,
                title=title, curvature=curvature, scale=scale,
                ax=ax, show=False,
                boundary_colors='red'
            )
        
        # Save individual figure if requested
        if save_dir:
            save_path = os.path.join(save_dir, f"level_{i}_poincare.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    # Adjust layout and show
    plt.tight_layout()
    
    # Save the combined figure
    if save_dir:
        save_path = os.path.join(save_dir, "all_levels_poincare.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig

def visualize_nested_spheres(model, input_sample=None, show=True, save_path=None):
    """
    Visualize the nested structure of a WuBu model by projecting higher dimensions
    to 3D using PCA and showing the nested boundary spheres.
    
    Args:
        model: A WuBuNestingModel instance
        input_sample: Optional input tensor to visualize its flow through the model
        show: Whether to show plots
        save_path: Optional path to save the figure
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("sklearn is required for dimensionality reduction. Please install with 'pip install scikit-learn'")
        return None
    
    # Check if the model is a WuBuNestingModel
    if not isinstance(model, WuBuNestingModel):
        raise ValueError("Model must be a WuBuNestingModel instance")
    
    # Create figure with 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set colors for different levels
    level_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Process each level
    for i in range(model.num_levels):
        # Get level and boundary manifold
        level = model.levels[i]
        boundary = model.boundaries[i]
        
        # Get level parameters
        curvature = level.get_curvature().item()
        scale = level.get_scale().item()
        dim = model.hyperbolic_dims[i]
        
        # Get boundary points
        boundary_points = boundary.get_tangent_vectors_at_origin()
        
        if boundary_points is not None:
            # Map boundary points to the Poincaré ball
            manifold_points = HyperbolicUtils.scale_aware_exponential_map(
                boundary_points, curvature, scale)
            
            # Scale for nested visualization (decreasing size for inner levels)
            # Use different scales based on level number
            radius_scale = 0.8 - (i * 0.2)  # Decreasing size for inner levels
            # Ensure min scale
            radius_scale = max(radius_scale, 0.1)
            
            # Project to 3D if dimension > 3
            if dim > 3:
                # Convert to numpy
                points_np = manifold_points.detach().cpu().numpy()
                
                # Apply PCA to reduce to 3D
                pca = PCA(n_components=3)
                points_3d = pca.fit_transform(points_np)
                
                # Normalize to unit sphere
                norms = np.sqrt(np.sum(points_3d**2, axis=1, keepdims=True))
                max_norm = np.max(norms)
                if max_norm > 0:
                    points_3d = points_3d / max_norm
                
                # Apply radius scale
                points_3d = points_3d * radius_scale
                
                # Plot the projected points
                ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                           s=50, alpha=0.8, label=f'Level {i}',
                           color=level_colors[i % len(level_colors)])
                
                # Plot a transparent sphere to represent the boundary
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = radius_scale * np.cos(u) * np.sin(v)
                y = radius_scale * np.sin(u) * np.sin(v)
                z = radius_scale * np.cos(v)
                ax.plot_surface(x, y, z, color=level_colors[i % len(level_colors)], 
                                alpha=0.1, rstride=1, cstride=1)
            
            elif dim == 3:
                # Just use the 3D points directly
                points_np = manifold_points.detach().cpu().numpy()
                
                # Normalize to unit sphere
                norms = np.sqrt(np.sum(points_np**2, axis=1, keepdims=True))
                max_norm = np.max(norms)
                if max_norm > 0:
                    points_np = points_np / max_norm
                
                # Apply radius scale
                points_3d = points_np * radius_scale
                
                # Plot the points
                ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                           s=50, alpha=0.8, label=f'Level {i}',
                           color=level_colors[i % len(level_colors)])
                
                # Plot transparent sphere
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = radius_scale * np.cos(u) * np.sin(v)
                y = radius_scale * np.sin(u) * np.sin(v)
                z = radius_scale * np.cos(v)
                ax.plot_surface(x, y, z, color=level_colors[i % len(level_colors)], 
                                alpha=0.1, rstride=1, cstride=1)
            
            elif dim == 2:
                # For 2D, embed in 3D space with z=0
                points_np = manifold_points.detach().cpu().numpy()
                
                # Normalize to unit circle
                norms = np.sqrt(np.sum(points_np**2, axis=1, keepdims=True))
                max_norm = np.max(norms)
                if max_norm > 0:
                    points_np = points_np / max_norm
                
                # Apply radius scale
                points_2d = points_np * radius_scale
                
                # Create 3D points with z=0
                points_3d = np.column_stack((points_2d, np.zeros(len(points_2d))))
                
                # Plot the points
                ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                           s=50, alpha=0.8, label=f'Level {i}',
                           color=level_colors[i % len(level_colors)])
                
                # Plot a circle to represent the 2D boundary
                theta = np.linspace(0, 2 * np.pi, 100)
                x = radius_scale * np.cos(theta)
                y = radius_scale * np.sin(theta)
                z = np.zeros_like(theta)
                ax.plot(x, y, z, color=level_colors[i % len(level_colors)], linewidth=2)
            
            elif dim == 1:
                # For 1D, embed as a line along x-axis
                points_np = manifold_points.detach().cpu().numpy()
                
                # Normalize to [-1, 1]
                if len(points_np) > 0:
                    max_abs = np.max(np.abs(points_np))
                    if max_abs > 0:
                        points_np = points_np / max_abs
                
                # Apply radius scale
                points_1d = points_np * radius_scale
                
                # Create 3D points with y=0, z=0
                points_3d = np.column_stack((points_1d, np.zeros(len(points_1d)), np.zeros(len(points_1d))))
                
                # Plot the points
                ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                           s=50, alpha=0.8, label=f'Level {i}',
                           color=level_colors[i % len(level_colors)])
                
                # Plot a line to represent the 1D boundary
                x = np.array([-radius_scale, radius_scale])
                y = np.zeros_like(x)
                z = np.zeros_like(x)
                ax.plot(x, y, z, color=level_colors[i % len(level_colors)], linewidth=2)
            
            # Annotate with level parameters
            offset = 0.1
            if dim <= 2:
                # For lower dimensions, place text at the end of the axis
                ax.text(radius_scale + offset, 0, 0, 
                      f"Level {i}\nDim: {dim}\nc: {curvature:.2f}\ns: {scale:.2f}",
                      ha='left', va='center', fontsize=10,
                      color=level_colors[i % len(level_colors)])
            else:
                # For higher dimensions, place text above the sphere
                ax.text(0, 0, radius_scale + offset, 
                      f"Level {i}\nDim: {dim}\nc: {curvature:.2f}\ns: {scale:.2f}",
                      ha='center', va='center', fontsize=10,
                      color=level_colors[i % len(level_colors)])
    
    # Set plot properties
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Nested Hyperbolic Levels (Projected to 3D)')
    
    # Equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Set axis limits
    limit = 1.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend (place outside plot)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    
    return fig

def visualize_tangent_space_transition(model, level_from=0, level_to=1, num_points=10, show=True, save_path=None):
    """
    Visualizes the tangent space transition between two levels, showing how
    points and boundary manifolds are rotated and transformed.
    
    Args:
        model: A WuBuNestingModel instance
        level_from: Index of the source level
        level_to: Index of the target level
        num_points: Number of sample points to generate
        show: Whether to show the plot
        save_path: Optional path to save the figure
    """
    # Check if the model is a WuBuNestingModel
    if not isinstance(model, WuBuNestingModel):
        raise ValueError("Model must be a WuBuNestingModel instance")
    
    # Check if the levels are valid
    if level_from < 0 or level_from >= model.num_levels - 1:
        raise ValueError(f"Source level must be between 0 and {model.num_levels-2}")
    if level_to != level_from + 1:
        raise ValueError("Target level must be the next level after source level")
    
    # Get dimensions for both levels
    dim_from = model.hyperbolic_dims[level_from]
    dim_to = model.hyperbolic_dims[level_to]
    
    # Check if both dimensions are 2 (for 2D visualization)
    if dim_from != 2 or dim_to != 2:
        raise ValueError("Both levels must have dimension 2 for visualization")
    
    # Get levels, rotation, and transform
    level_from_module = model.levels[level_from]
    level_to_module = model.levels[level_to]
    rotation_module = model.rotations[level_from]
    transform_module = model.transforms[level_from]
    
    # Get boundary points for source level
    boundary_module_from = model.boundaries[level_from]
    boundary_points = boundary_module_from.get_tangent_vectors_at_origin()
    
    # Get parameters
    curvature_from = level_from_module.get_curvature().item()
    scale_from = level_from_module.get_scale().item()
    curvature_to = level_to_module.get_curvature().item()
    scale_to = level_to_module.get_scale().item()
    
    # Create sample points in the source level's tangent space
    rng = torch.Generator().manual_seed(42)  # For reproducibility
    points = torch.randn(num_points, dim_from, generator=rng) * 0.5
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Map points to source Poincaré disk for visualization
    points_mapped = HyperbolicUtils.scale_aware_exponential_map(
        points, curvature_from, scale_from)
    
    # Map boundary points to source Poincaré disk
    if boundary_points is not None:
        boundary_mapped = HyperbolicUtils.scale_aware_exponential_map(
            boundary_points, curvature_from, scale_from)
    else:
        boundary_mapped = None
    
    # Plot source level
    visualize_poincare_disk(
        points_mapped, boundary_mapped,
        title=f"Level {level_from} (Source)",
        curvature=curvature_from, scale=scale_from,
        ax=ax1, show=False,
        point_colors='blue', boundary_colors='red',
        point_labels=[f"P{i}" for i in range(num_points)]
    )
    
    # Apply the rotation and transform
    with torch.no_grad():
        # Get level descriptor
        ld_from = level_from_module.level_descriptor.view(1, 1, -1)
        
        # Apply rotation
        points_rotated, boundary_rotated, ld_rotated = rotation_module(
            points.unsqueeze(0), boundary_points, ld_from)
        
        # Extract rotated points (remove batch dimension)
        points_rotated = points_rotated.squeeze(0)
        
        # Apply transform
        points_transformed, boundary_transformed, ld_transformed = transform_module(
            points_rotated.unsqueeze(0), boundary_rotated, ld_rotated)
        
        # Extract transformed points (remove batch dimension)
        points_transformed = points_transformed.squeeze(0)
    
    # Map points to target Poincaré disk for visualization
    points_target_mapped = HyperbolicUtils.scale_aware_exponential_map(
        points_transformed, curvature_to, scale_to)
    
    # Map boundary points to target Poincaré disk
    if boundary_transformed is not None:
        boundary_target_mapped = HyperbolicUtils.scale_aware_exponential_map(
            boundary_transformed, curvature_to, scale_to)
    else:
        boundary_target_mapped = None
    
    # Plot target level
    visualize_poincare_disk(
        points_target_mapped, boundary_target_mapped,
        title=f"Level {level_to} (Target)",
        curvature=curvature_to, scale=scale_to,
        ax=ax2, show=False,
        point_colors='green', boundary_colors='orange',
        point_labels=[f"P{i}" for i in range(num_points)]
    )
    
    # Add a title for the whole figure
    plt.suptitle(f"Tangent Space Transition: Level {level_from} → Level {level_to}", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    
    return fig

# =====================================================================
# Example Usage
# =====================================================================

def test_visualization():
    """
    Test visualization functions with a simple WuBu Nesting model.
    """
    # Create a model with a mix of dimensions for testing visualization
    config = {
        "num_levels": 4,
        "hyperbolic_dims": [8, 6, 3, 2],  # Mix of dimensions
        "boundary_points_per_level": [8, 6, 5, 4],
        "rotation_types": ["so_n", "so_n", "so_n"],
        "transform_types": ["linear", "linear", "linear"],
        "learnable_curvature": True,
        "learnable_scales": True,
        "learnable_spread": True,
        "use_level_descriptors": True,
        "use_level_spread": True,
        "use_tangent_flow": True,
        "initial_curvatures": [1.0, 2.0, 4.0, 8.0],  # Increasing curvature for deeper levels
        "initial_scales": [1.0, 0.8, 0.5, 0.3],      # Decreasing scale for deeper levels
        "initial_spread_values": [1.0, 0.7, 0.4, 0.2],
        "dropout": 0.1,
        "level_descriptor_init_scale": 0.01,
        "relative_vector_aggregation": "mean",
    }
    
    # Create model instance
    input_dim = 8
    output_dim = 10
    model = WuBuNestingModel(input_dim, output_dim, config)
    
    # Create a simple input
    x = torch.randn(1, input_dim)
    
    # Test visualizations
    print("Visualizing Poincaré disk for 2D levels...")
    visualize_poincare_nested(model, x, show=True)
    
    print("Visualizing nested spheres (all dimensions)...")
    visualize_nested_spheres(model, x, show=True)
    
    # Check if we can visualize transition between last two levels (3D->2D)
    try:
        print("Visualizing transition between last two levels...")
        visualize_tangent_space_transition(model, level_from=2, level_to=3, show=True)
    except ValueError as e:
        print(f"Could not visualize transition: {e}")
    
    print("Visualization test completed!")

if __name__ == "__main__":
    test_visualization()