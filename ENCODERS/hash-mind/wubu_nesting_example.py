import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Import the WuBu Nesting implementation
from wubu_nesting_impl import (
    HyperbolicUtils, 
    BoundaryManifold, 
    TangentSpaceRotation,
    InterLevelTransform,
    WuBuNestingLevel,
    WuBuNestingModel
)

# Import visualization tools
from wubu_nesting_visualization import (
    visualize_poincare_disk,
    visualize_poincare_nested,
    visualize_tangent_space_transition
)

# =====================================================================
# Example Dataset - Hierarchical Synthetic Data
# =====================================================================

def generate_hierarchical_data(num_samples=1000, noise=0.05, seed=42):
    """
    Generates synthetic hierarchical data with nested structures.
    The data consists of points from a hierarchical tree structure embedded in 2D.
    
    Args:
        num_samples: Number of samples to generate
        noise: Amount of noise to add
        seed: Random seed for reproducibility
        
    Returns:
        x: Input data (points in 2D)
        y: Target labels (cluster/tree node assignments)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Define a simple hierarchical tree structure
    # Level 1: 3 main clusters
    # Level 2: Each main cluster has 2-3 subclusters
    # Level 3: Each subcluster has 2 sub-subclusters
    
    # Main cluster centers
    main_centers = np.array([
        [-0.6, 0.0],   # Left
        [0.0, 0.6],    # Top
        [0.6, 0.0]     # Right
    ])
    
    # Subclusters for each main cluster
    subclusters = [
        # Left main cluster subclusters
        np.array([
            [-0.7, 0.3],   # Top-left
            [-0.7, -0.3],  # Bottom-left
        ]),
        # Top main cluster subclusters
        np.array([
            [-0.2, 0.7],   # Left-top
            [0.2, 0.7],    # Right-top
        ]),
        # Right main cluster subclusters
        np.array([
            [0.7, 0.3],    # Top-right
            [0.7, -0.3],   # Bottom-right
        ])
    ]
    
    # Sub-subclusters for each subcluster (Level 3)
    sub_subclusters = [
        # Left main cluster, subcluster 0 sub-subclusters
        np.array([
            [-0.8, 0.2],   # Near
            [-0.6, 0.4],   # Far
        ]),
        # Left main cluster, subcluster 1 sub-subclusters
        np.array([
            [-0.8, -0.2],  # Near
            [-0.6, -0.4],  # Far
        ]),
        # Top main cluster, subcluster 0 sub-subclusters
        np.array([
            [-0.3, 0.6],   # Near
            [-0.1, 0.8],   # Far
        ]),
        # Top main cluster, subcluster 1 sub-subclusters
        np.array([
            [0.3, 0.6],    # Near
            [0.1, 0.8],    # Far
        ]),
        # Right main cluster, subcluster 0 sub-subclusters
        np.array([
            [0.8, 0.2],    # Near
            [0.6, 0.4],    # Far
        ]),
        # Right main cluster, subcluster 1 sub-subclusters
        np.array([
            [0.8, -0.2],   # Near
            [0.6, -0.4],   # Far
        ]),
    ]
    
    # Generate points with hierarchical structure
    x = []
    y_level1 = []  # Main cluster labels
    y_level2 = []  # Subcluster labels
    y_level3 = []  # Sub-subcluster labels
    
    # Count samples per level 3 cluster
    samples_per_l3_cluster = num_samples // 12  # 3 main clusters * 2 subclusters * 2 sub-subclusters = 12
    
    # Generate points for each sub-subcluster
    subcluster_idx = 0
    for main_idx, main_center in enumerate(main_centers):
        for sub_idx in range(2):  # 2 subclusters per main cluster
            for subsub_idx in range(2):  # 2 sub-subclusters per subcluster
                # Get the center for this sub-subcluster
                center = sub_subclusters[subcluster_idx][subsub_idx]
                
                # Generate points around the center
                for _ in range(samples_per_l3_cluster):
                    # Add some noise to create a cluster
                    point = center + np.random.normal(0, noise, size=2)
                    
                    # Ensure points are within a reasonable radius
                    norm = np.linalg.norm(point)
                    if norm > 0.9:  # Keep points away from the boundary
                        point = point * (0.9 / norm)
                    
                    x.append(point)
                    y_level1.append(main_idx)
                    y_level2.append(main_idx * 2 + sub_idx)
                    y_level3.append(subcluster_idx * 2 + subsub_idx)
                
            # Increment the subcluster index
            subcluster_idx += 1
    
    # Convert to numpy arrays
    x = np.array(x, dtype=np.float32)
    y_level1 = np.array(y_level1, dtype=np.int64)
    y_level2 = np.array(y_level2, dtype=np.int64)
    y_level3 = np.array(y_level3, dtype=np.int64)
    
    # Combine all labels
    y = np.vstack([y_level1, y_level2, y_level3]).T
    
    # Convert to PyTorch tensors
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    # Create a dictionary with all the generated data
    data = {
        'x': x_tensor,
        'y': y_tensor,
        'main_centers': main_centers,
        'subclusters': subclusters,
        'sub_subclusters': sub_subclusters
    }
    
    return data

def visualize_hierarchical_data(data, show=True, save_path=None):
    """
    Visualizes the hierarchical synthetic data.
    
    Args:
        data: Dictionary containing the generated data
        show: Whether to show the plot
        save_path: Optional path to save the figure
    """
    x = data['x'].numpy()
    y = data['y'].numpy()
    
    # Create figure with 3 subplots (one for each hierarchy level)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colors for each level
    colors_l1 = ['#1f77b4', '#ff7f0e', '#2ca02c']
    colors_l2 = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a']
    
    # Plot Level 1 (Main Clusters)
    for i in range(3):  # 3 main clusters
        mask = y[:, 0] == i
        axs[0].scatter(x[mask, 0], x[mask, 1], s=15, alpha=0.7, c=colors_l1[i], label=f'L1-{i}')
    
    axs[0].set_title('Level 1: Main Clusters')
    axs[0].legend()
    axs[0].set_xlim(-1, 1)
    axs[0].set_ylim(-1, 1)
    axs[0].set_aspect('equal')
    axs[0].grid(alpha=0.3)
    
    # Plot Level 2 (Subclusters)
    for i in range(6):  # 6 subclusters
        mask = y[:, 1] == i
        axs[1].scatter(x[mask, 0], x[mask, 1], s=15, alpha=0.7, c=colors_l2[i], label=f'L2-{i}')
    
    axs[1].set_title('Level 2: Subclusters')
    axs[1].legend()
    axs[1].set_xlim(-1, 1)
    axs[1].set_ylim(-1, 1)
    axs[1].set_aspect('equal')
    axs[1].grid(alpha=0.3)
    
    # Plot Level 3 (Sub-subclusters)
    for i in range(12):  # 12 sub-subclusters
        mask = y[:, 2] == i
        axs[2].scatter(x[mask, 0], x[mask, 1], s=15, alpha=0.7, label=f'L3-{i}')
    
    axs[2].set_title('Level 3: Sub-subclusters')
    axs[2].legend(loc='upper right', bbox_to_anchor=(1.45, 1))
    axs[2].set_xlim(-1, 1)
    axs[2].set_ylim(-1, 1)
    axs[2].set_aspect('equal')
    axs[2].grid(alpha=0.3)
    
    # Add units circles representing the boundary of the Poincaré ball
    for ax in axs:
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='-', alpha=0.5)
        ax.add_patch(circle)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    
    return fig

# =====================================================================
# Model Definition for Hierarchical Classification
# =====================================================================

class WuBuHierarchicalClassifier(nn.Module):
    """
    A classification model using the WuBu Nesting architecture for hierarchical data.
    This model outputs predictions for each level of the hierarchy.
    """
    def __init__(self, input_dim, num_levels=3, num_classes_per_level=None, config=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_levels = num_levels
        
        if num_classes_per_level is None:
            # Default classes per level in our example: 3, 6, 12
            self.num_classes_per_level = [3, 6, 12]
        else:
            self.num_classes_per_level = num_classes_per_level
        
        # Create the WuBu Nesting model
        wubu_output_dim = sum(self.num_classes_per_level)  # Combined output dimension
        self.wubu_model = WuBuNestingModel(input_dim, wubu_output_dim, config)
        
        # Create classifier heads for each hierarchy level
        self.classifier_heads = nn.ModuleList()
        
        # Define slice indices for separating the outputs for each level
        self.slice_indices = []
        start_idx = 0
        for num_classes in self.num_classes_per_level:
            self.slice_indices.append((start_idx, start_idx + num_classes))
            start_idx += num_classes
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            List of output logits for each hierarchy level
        """
        # Get combined output from WuBu model
        combined_output = self.wubu_model(x)
        
        # Split output for each hierarchy level
        level_outputs = []
        for start_idx, end_idx in self.slice_indices:
            level_outputs.append(combined_output[:, start_idx:end_idx])
        
        return level_outputs
    
    def predict(self, x):
        """
        Make predictions for all hierarchy levels.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            List of predicted class indices for each hierarchy level
        """
        level_outputs = self.forward(x)
        predictions = [torch.argmax(output, dim=1) for output in level_outputs]
        return predictions

# =====================================================================
# Training and Evaluation Functions
# =====================================================================

def train_model(model, data, num_epochs=50, batch_size=32, learning_rate=1e-3, device='cpu',
                save_dir=None, save_interval=10):
    """
    Train the WuBu hierarchical classifier.
    
    Args:
        model: The WuBuHierarchicalClassifier model
        data: Dictionary containing the training data (x, y)
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for the optimizer
        device: Device to train on ('cpu' or 'cuda')
        save_dir: Directory to save model checkpoints and visualizations
    """
    # Move model to device
    model = model.to(device)
    
    # Get training data
    x = data['x'].to(device)
    y = data['y'].to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create loss function (CrossEntropyLoss for classification)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_samples = len(x)
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    # Lists to store training metrics
    epoch_losses = []
    level_accuracies = [[] for _ in range(model.num_levels)]
    
    # Create save directory if needed
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
    
    print(f"Training model for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Shuffle data
        indices = torch.randperm(num_samples)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        
        # Process batches
        for batch_idx in range(num_batches):
            # Get batch data
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_x = x_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]
            
            # Forward pass
            outputs = model(batch_x)
            
            # Calculate loss for each level and sum them
            batch_loss = 0.0
            for level_idx, level_output in enumerate(outputs):
                level_y = batch_y[:, level_idx]
                level_loss = criterion(level_output, level_y)
                batch_loss += level_loss
            
            # Backward pass and optimization
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += batch_loss.item()
        
        # Calculate average loss for the epoch
        epoch_loss /= num_batches
        epoch_losses.append(epoch_loss)
        
        # Evaluate model after each epoch
        model.eval()
        with torch.no_grad():
            # Get predictions for all samples
            level_outputs = model(x)
            level_preds = [torch.argmax(output, dim=1) for output in level_outputs]
            
            # Calculate accuracy for each level
            for level_idx in range(model.num_levels):
                level_correct = torch.sum(level_preds[level_idx] == y[:, level_idx]).item()
                level_accuracy = level_correct / num_samples
                level_accuracies[level_idx].append(level_accuracy)
        
        # Print progress
        time_elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Time: {time_elapsed:.2f}s")
        for level_idx in range(model.num_levels):
            print(f"  Level {level_idx+1} Accuracy: {level_accuracies[level_idx][-1]:.4f}")
        
        # Save model checkpoint and visualizations at specified intervals
        if save_dir and (epoch + 1) % save_interval == 0:
            # Save model checkpoint
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'accuracies': [acc[-1] for acc in level_accuracies]
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
            
            # Save visualizations
            try:
                # Create visualization directory if it doesn't exist
                vis_dir = os.path.join(save_dir, 'visualizations')
                if not os.path.exists(vis_dir):
                    os.makedirs(vis_dir)
                
                # Import visualization functions
                from wubu_nesting_visualization import (
                    visualize_poincare_disk,
                    visualize_poincare_nested,
                    visualize_tangent_space_transition,
                    visualize_nested_spheres
                )
                
                # Save 3D nested spheres visualization (works for any dimensions)
                nested_vis_path = os.path.join(vis_dir, f"nested_spheres_epoch_{epoch+1}.png")
                visualize_nested_spheres(model.wubu_model, x[:1], show=False, save_path=nested_vis_path)
                print(f"Nested spheres visualization saved to {nested_vis_path}")
                
                # Try to save 2D Poincaré disk visualizations if any level has dim=2
                has_2d_level = any(dim == 2 for dim in model.wubu_model.hyperbolic_dims)
                if has_2d_level:
                    vis_dir_epoch = os.path.join(vis_dir, f"poincare_nested_epoch_{epoch+1}")
                    if not os.path.exists(vis_dir_epoch):
                        os.makedirs(vis_dir_epoch)
                    visualize_poincare_nested(model.wubu_model, x[:1], show=False, save_dir=vis_dir_epoch)
                    print(f"Poincaré disk visualizations saved to {vis_dir_epoch}")
                
                # Try to visualize tangent space transition if adjacent levels have dim=2
                for i in range(model.wubu_model.num_levels - 1):
                    if (model.wubu_model.hyperbolic_dims[i] == 2 and 
                        model.wubu_model.hyperbolic_dims[i+1] == 2):
                        trans_path = os.path.join(vis_dir, f"transition_{i}_{i+1}_epoch_{epoch+1}.png")
                        visualize_tangent_space_transition(model.wubu_model, i, i+1, show=False, save_path=trans_path)
                        print(f"Transition visualization saved to {trans_path}")
                
            except Exception as e:
                print(f"Error saving visualizations: {e}")
                import traceback
                traceback.print_exc()
    
    # Calculate and return final metrics
    final_metrics = {
        'epoch_losses': epoch_losses,
        'level_accuracies': level_accuracies,
        'training_time': time.time() - start_time
    }
    
    return model, final_metrics
def evaluate_model(model, data, device='cpu'):
    """
    Evaluate the trained model on the data.
    
    Args:
        model: Trained WuBuHierarchicalClassifier model
        data: Dictionary containing the evaluation data (x, y)
        device: Device to evaluate on ('cpu' or 'cuda')
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Move model to device
    model = model.to(device)
    
    # Get data
    x = data['x'].to(device)
    y = data['y'].to(device)
    num_samples = len(x)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        level_outputs = model(x)
        level_preds = [torch.argmax(output, dim=1) for output in level_outputs]
    
    # Calculate metrics for each level
    level_metrics = []
    for level_idx in range(model.num_levels):
        level_y = y[:, level_idx]
        level_pred = level_preds[level_idx]
        
        # Calculate accuracy
        correct = torch.sum(level_pred == level_y).item()
        accuracy = correct / num_samples
        
        # Calculate per-class accuracy
        num_classes = model.num_classes_per_level[level_idx]
        class_correct = torch.zeros(num_classes, device=device)
        class_total = torch.zeros(num_classes, device=device)
        
        for i in range(num_samples):
            class_idx = level_y[i].item()
            class_total[class_idx] += 1
            if level_pred[i] == level_y[i]:
                class_correct[class_idx] += 1
        
        class_accuracies = []
        for i in range(num_classes):
            if class_total[i] > 0:
                class_accuracies.append(class_correct[i].item() / class_total[i].item())
            else:
                class_accuracies.append(0.0)
        
        # Store metrics for this level
        level_metrics.append({
            'accuracy': accuracy,
            'class_accuracies': class_accuracies
        })
    
    # Return all metrics
    metrics = {
        'level_metrics': level_metrics,
        'predictions': [p.cpu() for p in level_preds]
    }
    
    return metrics

def visualize_results(model, data, metrics, show=True, save_path=None):
    """
    Visualize the model's predictions on the data.
    
    Args:
        model: Trained WuBuHierarchicalClassifier model
        data: Dictionary containing the data (x, y)
        metrics: Dictionary of evaluation metrics
        show: Whether to show the plot
        save_path: Optional path to save the figure
    """
    x_np = data['x'].cpu().numpy()
    y_np = data['y'].cpu().numpy()
    predictions = metrics['predictions']
    
    # Create figure with one row of plots (one for each level)
    fig, axs = plt.subplots(1, model.num_levels, figsize=(18, 6))
    
    # Set of colors for visualization
    colors = plt.cm.tab20.colors
    
    # Plot each level
    for level_idx in range(model.num_levels):
        ax = axs[level_idx]
        
        # Get predictions for this level
        level_pred = predictions[level_idx].numpy()
        
        # Plot points colored by predicted class
        num_classes = model.num_classes_per_level[level_idx]
        for class_idx in range(num_classes):
            mask = level_pred == class_idx
            ax.scatter(x_np[mask, 0], x_np[mask, 1], s=15, alpha=0.7, 
                       color=colors[class_idx % len(colors)], 
                       label=f'Class {class_idx}')
        
        # Add unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='-', alpha=0.5)
        ax.add_patch(circle)
        
        # Add level accuracy
        accuracy = metrics['level_metrics'][level_idx]['accuracy']
        ax.set_title(f'Level {level_idx+1} Predictions\nAccuracy: {accuracy:.4f}')
        
        # Set plot properties
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    
    return fig

def plot_training_metrics(metrics, show=True, save_path=None):
    """
    Plot training metrics (loss and accuracy over epochs).
    
    Args:
        metrics: Dictionary of training metrics
        show: Whether to show the plot
        save_path: Optional path to save the figure
    """
    # Get metrics
    epoch_losses = metrics['epoch_losses']
    level_accuracies = metrics['level_accuracies']
    num_levels = len(level_accuracies)
    epochs = np.arange(1, len(epoch_losses) + 1)
    
    # Create figure with two subplots (loss and accuracy)
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axs[0].plot(epochs, epoch_losses, 'b-', linewidth=2)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training Loss')
    axs[0].grid(alpha=0.3)
    
    # Plot accuracy for each level
    for level_idx in range(num_levels):
        axs[1].plot(epochs, level_accuracies[level_idx], linewidth=2, 
                   label=f'Level {level_idx+1}')
    
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Training Accuracy by Level')
    axs[1].grid(alpha=0.3)
    axs[1].legend()
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    
    return fig

# =====================================================================
# Main Training and Evaluation Example
# =====================================================================

def run_example(save_dir='results', num_epochs=30, batch_size=64, device='cpu'):
    """
    Run the complete example: generate data, train model, and evaluate results.
    
    Args:
        save_dir: Directory to save results
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to run on ('cpu' or 'cuda')
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Set device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available. Using CPU instead.")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Generate synthetic hierarchical data
    print("Generating hierarchical data...")
    data = generate_hierarchical_data(num_samples=1200, noise=0.05)
    
    # Visualize the data
    print("Visualizing data...")
    data_fig = visualize_hierarchical_data(data, show=False)
    data_fig.savefig(os.path.join(save_dir, 'hierarchical_data.png'), dpi=200, bbox_inches='tight')
    
    # Split data into train and test sets (80/20 split)
    num_samples = len(data['x'])
    num_train = int(0.8 * num_samples)
    
    # Shuffle indices
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]
    
    # Create train and test sets
    train_data = {
        'x': data['x'][train_indices],
        'y': data['y'][train_indices],
    }
    
    test_data = {
        'x': data['x'][test_indices],
        'y': data['y'][test_indices],
    }
    
    print(f"Data split: {len(train_data['x'])} train, {len(test_data['x'])} test")
    
    # Define the WuBu configuration specific to hierarchical data
    wubu_config = {
        "num_levels": 3,  # Match the hierarchy in the data
        "hyperbolic_dims": [8, 6, 4],  # Decreasing dimensions
        "boundary_points_per_level": [3, 6, 12],  # Match number of clusters per level
        "rotation_types": ["so_n", "so_n"],
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
        "dropout": 0.2,
        "level_descriptor_init_scale": 0.01,
        "relative_vector_aggregation": "mean",
    }
    
    # Create hierarchical classifier
    input_dim = 2  # 2D points
    num_classes_per_level = [3, 6, 12]  # Classes per level (main, sub, sub-sub)
    
    print("Creating model...")
    model = WuBuHierarchicalClassifier(
        input_dim=input_dim,
        num_levels=3,
        num_classes_per_level=num_classes_per_level,
        config=wubu_config
    )
    
    # Train the model
    print("Training model...")
    trained_model, training_metrics = train_model(
        model=model,
        data=train_data,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=1e-3,
        device=device,
        save_dir=save_dir,
        save_interval=5
    )
    
    # Plot training metrics
    print("Plotting training metrics...")
    metrics_fig = plot_training_metrics(
        metrics=training_metrics,
        show=False,
        save_path=os.path.join(save_dir, 'training_metrics.png')
    )
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_metrics = evaluate_model(
        model=trained_model,
        data=test_data,
        device=device
    )
    
    # Print test metrics
    print("\nTest Results:")
    for level_idx, level_metric in enumerate(test_metrics['level_metrics']):
        print(f"Level {level_idx+1} Accuracy: {level_metric['accuracy']:.4f}")
    
    # Visualize results
    print("Visualizing results...")
    results_fig = visualize_results(
        model=trained_model,
        data=test_data,
        metrics=test_metrics,
        show=False,
        save_path=os.path.join(save_dir, 'test_predictions.png')
    )
    
    # Try to visualize the final model structure if dimensions allow
    try:
        print("Visualizing model's nested structure...")
        if 2 in trained_model.wubu_model.hyperbolic_dims:
            # Find first level with dim=2 for visualization
            vis_level = trained_model.wubu_model.hyperbolic_dims.index(2)
            visualize_poincare_nested(
                model=trained_model.wubu_model,
                level_idx=vis_level,
                show=False,
                save_dir=os.path.join(save_dir, 'final_model_vis')
            )
    except Exception as e:
        print(f"Could not visualize model structure: {e}")
    
    print(f"Example completed! Results saved to {save_dir}")
    return trained_model, test_metrics

if __name__ == "__main__":
    # Run the complete example
    run_example(save_dir='wubu_results', num_epochs=30, batch_size=64, device='cpu')

