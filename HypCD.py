#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperbolic Category Discovery (HypCD) Implementation

This implementation is based on the paper "Hyperbolic Category Discovery" which proposes
a framework for learning hierarchy-aware representations for generalized category discovery
by using hyperbolic geometry instead of Euclidean or spherical space.

Key components:
1. Feature mapping from Euclidean to hyperbolic space (Poincaré ball)
2. Hyperbolic representation learning with both distance and angle-based losses
3. Hyperbolic classifier for parametric approaches
4. Integration with existing GCD methods
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from typing import Tuple, List, Dict, Optional, Union
import logging
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CURVATURE = 0.05  # Default curvature for fine-grained datasets
DEFAULT_CLIPPING = 2.3    # Default clipping value for fine-grained datasets
EPS = 1e-6               # Small epsilon for numerical stability

#######################
# Hyperbolic Operations
#######################

class HyperbolicOps:
    """Utility class for hyperbolic operations in the Poincaré ball model."""
    
    @staticmethod
    def expmap0(v: torch.Tensor, c: float) -> torch.Tensor:
        """
        Exponential map from the tangent space at the origin to the Poincaré ball.
        
        Args:
            v: Tangent vector(s) at the origin
            c: Curvature value
            
        Returns:
            Points in the Poincaré ball
        """
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        sqrt_c = np.sqrt(c)
        # Handle small norms for numerical stability
        factor = torch.tanh(sqrt_c * v_norm) / (sqrt_c * torch.clamp(v_norm, min=EPS))
        return factor * v
    
    @staticmethod
    def logmap0(x: torch.Tensor, c: float) -> torch.Tensor:
        """
        Logarithmic map from the Poincaré ball to the tangent space at the origin.
        
        Args:
            x: Points in the Poincaré ball
            c: Curvature value
            
        Returns:
            Tangent vectors at the origin
        """
        x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        sqrt_c = np.sqrt(c)
        # Handle small norms for numerical stability
        factor = torch.atanh(torch.clamp(sqrt_c * x_norm, max=1-EPS)) / (sqrt_c * torch.clamp(x_norm, min=EPS))
        return factor * x
    
    @staticmethod
    def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
        """
        Möbius addition in the Poincaré ball.
        
        Args:
            x, y: Points in the Poincaré ball
            c: Curvature value
            
        Returns:
            Result of the Möbius addition x ⊕_c y
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c**2 * x2 * y2
        
        return num / torch.clamp(denom, min=EPS)
    
    @staticmethod
    def distance(x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
        """
        Hyperbolic distance in the Poincaré ball.
        
        Args:
            x, y: Points in the Poincaré ball
            c: Curvature value
            
        Returns:
            Hyperbolic distance between x and y
        """
        # Compute Möbius addition -x ⊕_c y
        neg_x = -x
        mobius_sum = HyperbolicOps.mobius_add(neg_x, y, c)
        
        # Compute the norm of the Möbius sum
        mobius_norm = torch.norm(mobius_sum, p=2, dim=-1)
        
        # Compute hyperbolic distance
        return 2 / np.sqrt(c) * torch.atanh(torch.clamp(np.sqrt(c) * mobius_norm, max=1-EPS))
    
    @staticmethod
    def hyperbolic_softmax(x: torch.Tensor, a: torch.Tensor, p: torch.Tensor, c: float) -> torch.Tensor:
        """
        Hyperbolic softmax as defined in the hyperbolic neural networks literature.
        
        Args:
            x: Vectors in the tangent space
            a: Point in the Poincaré ball (usually the origin)
            p: Prototype points in the Poincaré ball
            c: Curvature value
            
        Returns:
            Hyperbolic softmax probabilities
        """
        # Map x to the Poincaré ball
        x_mapped = HyperbolicOps.expmap0(x, c)
        
        # Compute distance to each prototype
        distances = torch.stack([HyperbolicOps.distance(x_mapped, p_i.unsqueeze(0), c) 
                               for p_i in p], dim=1)
        
        # Apply softmax to negative distances (smaller distance = higher probability)
        return F.softmax(-distances, dim=1)
    
    @staticmethod
    def project(x: torch.Tensor, c: float) -> torch.Tensor:
        """
        Project points to the Poincaré ball to ensure they stay within the manifold.
        
        Args:
            x: Points to project
            c: Curvature value
            
        Returns:
            Projected points
        """
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        max_norm = (1 - EPS) / np.sqrt(c)
        cond = norm > max_norm
        
        projected = x.clone()
        projected[cond] = x[cond] * (max_norm / norm[cond])
        
        return projected
    
    @staticmethod
    def safe_proj(x: torch.Tensor) -> torch.Tensor:
        """
        Safe projection for the hyperbolic linear layer to ensure numerical stability.
        
        Args:
            x: Points in the Poincaré ball
            
        Returns:
            Safely projected points
        """
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        threshold = torch.tensor(1.0 - 1e-3)
        mask = norm > threshold
        
        if mask.any():
            x[mask] = threshold * x[mask] / norm[mask]
        
        return x

#######################
# Core Model Components
#######################

class FeatureClipper(nn.Module):
    """
    Clips features to avoid gradient instabilities near the boundary of the Poincaré ball.
    """
    def __init__(self, clip_value: float = 2.3):
        """
        Args:
            clip_value: Maximum norm allowed for the feature vector
        """
        super().__init__()
        self.clip_value = clip_value
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature vectors in Euclidean space
            
        Returns:
            Clipped feature vectors
        """
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        mask = norm > self.clip_value
        
        if mask.any():
            x_clipped = x.clone()
            x_clipped[mask] = self.clip_value * x[mask] / norm[mask]
            return x_clipped
        return x


class EuclideanToHyperbolic(nn.Module):
    """
    Maps Euclidean features to the Poincaré ball using clipping and exponential mapping.
    """
    def __init__(self, curvature: float = 0.05, clip_value: float = 2.3):
        """
        Args:
            curvature: Curvature parameter of the hyperbolic space (c)
            clip_value: Maximum norm allowed for the feature vector before mapping
        """
        super().__init__()
        self.curvature = curvature
        self.clipper = FeatureClipper(clip_value)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature vectors in Euclidean space
            
        Returns:
            Mapped feature vectors in the Poincaré ball
        """
        # First clip features to avoid issues near boundary
        x_clipped = self.clipper(x)
        
        # Map to the Poincaré ball
        x_hyp = HyperbolicOps.expmap0(x_clipped, self.curvature)
        
        # Ensure points stay in the manifold
        x_hyp = HyperbolicOps.project(x_hyp, self.curvature)
        
        return x_hyp


class HyperbolicLinear(nn.Module):
    """
    Hyperbolic linear layer as described in the paper and in previous literature.
    """
    def __init__(self, in_features: int, out_features: int, c: float = 0.05, bias: bool = True):
        """
        Args:
            in_features: Dimension of input features
            out_features: Dimension of output features
            c: Curvature parameter
            bias: Whether to use bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.use_bias = bias
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
            
    def reset_parameters(self):
        """Initialize weights and biases."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Points in the Poincaré ball
            
        Returns:
            Transformed points in the Poincaré ball
        """
        # Convert to tangent space at origin
        x_tan = HyperbolicOps.logmap0(x, self.c)
        
        # Apply standard linear transformation in tangent space
        output = F.linear(x_tan, self.weight)
        
        if self.bias is not None:
            output = output + self.bias
            
        # Map back to the Poincaré ball
        output = HyperbolicOps.expmap0(output, self.c)
        
        # Safe projection
        output = HyperbolicOps.safe_proj(output)
        
        return output


class HyperbolicFFN(nn.Module):
    """
    Hyperbolic Feed-Forward Network using hyperbolic linear layers.
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 c: float = 0.05,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            c: Curvature parameter
            dropout: Dropout probability
        """
        super().__init__()
        self.c = c
        self.layer1 = HyperbolicLinear(input_dim, hidden_dim, c)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = HyperbolicLinear(hidden_dim, output_dim, c)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Points in the Poincaré ball
            
        Returns:
            Transformed points in the Poincaré ball
        """
        # Convert to tangent space for activation
        x = self.layer1(x)
        x_tan = HyperbolicOps.logmap0(x, self.c)
        
        # Apply ReLU and dropout in tangent space
        x_tan = F.relu(x_tan)
        x_tan = self.dropout(x_tan)
        
        # Map back to Poincaré ball
        x = HyperbolicOps.expmap0(x_tan, self.c)
        
        # Apply second layer
        x = self.layer2(x)
        
        return x


class HyperbolicContrastiveLoss(nn.Module):
    """
    Implements the hybrid contrastive loss combining angle-based and distance-based 
    components for hyperbolic space as described in the paper.
    """
    def __init__(self, 
                 temperature: float = 0.1, 
                 c: float = 0.05,
                 alpha_d: float = 0.5,
                 max_alpha_d: float = 1.0,
                 alpha_epochs: int = 200):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
            c: Curvature parameter
            alpha_d: Initial weight for the distance-based loss component
            max_alpha_d: Maximum weight for the distance-based loss component
            alpha_epochs: Number of epochs to linearly increase alpha_d to max_alpha_d
        """
        super().__init__()
        self.temperature = temperature
        self.c = c
        self.alpha_d = alpha_d
        self.max_alpha_d = max_alpha_d
        self.alpha_epochs = alpha_epochs
        self.current_epoch = 0
        
    def forward(self, 
                z_hyp: torch.Tensor, 
                z_hyp_prime: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            z_hyp: First view embeddings in hyperbolic space
            z_hyp_prime: Second view embeddings in hyperbolic space
            labels: Optional class labels for supervised contrastive loss
            
        Returns:
            Contrastive loss value
        """
        batch_size = z_hyp.size(0)
        device = z_hyp.device
        
        # Calculate distance-based similarity
        distances = torch.zeros((batch_size, batch_size), device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                distances[i, j] = HyperbolicOps.distance(z_hyp[i].unsqueeze(0), 
                                                         z_hyp_prime[j].unsqueeze(0), 
                                                         self.c).squeeze()
        
        # Negative distance as similarity
        distance_sim = -distances / self.temperature
        
        # Calculate angle-based (cosine) similarity
        z_hyp_norm = F.normalize(z_hyp, p=2, dim=1)
        z_hyp_prime_norm = F.normalize(z_hyp_prime, p=2, dim=1)
        angle_sim = torch.mm(z_hyp_norm, z_hyp_prime_norm.t()) / self.temperature
        
        # Compute current alpha_d based on linear schedule
        current_alpha_d = min(self.alpha_d + (self.max_alpha_d - self.alpha_d) * 
                           self.current_epoch / self.alpha_epochs, self.max_alpha_d)
        
        # Combine similarities
        combined_sim = current_alpha_d * distance_sim + (1 - current_alpha_d) * angle_sim
        
        # Mask for positive pairs
        if labels is not None:
            # Supervised contrastive loss
            # Create a mask for samples from the same class
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
            
            # Remove the diagonal (self-similarity)
            mask = mask.fill_diagonal_(0)
            
            # For each anchor, compute denominator considering all negatives
            exp_combined_sim = torch.exp(combined_sim)
            pos_mask = mask.bool()
            neg_mask = ~pos_mask
            
            loss = 0.0
            for i in range(batch_size):
                pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
                if len(pos_indices) > 0:  # If there are positive pairs
                    pos_sim = combined_sim[i, pos_indices]
                    all_sim = exp_combined_sim[i]
                    all_sim[i] = 0  # Exclude self
                    denom = all_sim.sum()
                    
                    # Compute loss for all positive pairs
                    pos_loss = -pos_sim + torch.log(denom)
                    loss += pos_loss.mean()
            
            if len(pos_mask.nonzero()) > 0:
                loss /= len(pos_mask.nonzero())
            else:
                # Fallback to self-supervised if no positives
                diag_mask = torch.eye(batch_size, device=device).bool()
                pos_sim = combined_sim[diag_mask]
                exp_sim = torch.exp(combined_sim)
                exp_sim.fill_diagonal_(0)
                denom = exp_sim.sum(dim=1)
                loss = (-pos_sim + torch.log(denom)).mean()
                
        else:
            # Self-supervised contrastive loss
            # The positive pairs are on the diagonal
            diag_mask = torch.eye(batch_size, device=device).bool()
            pos_sim = combined_sim[diag_mask]
            
            # For each anchor, compute denominator considering all negatives
            exp_sim = torch.exp(combined_sim)
            exp_sim.fill_diagonal_(0)
            denom = exp_sim.sum(dim=1)
            
            # Compute loss
            loss = (-pos_sim + torch.log(denom)).mean()
            
        return loss
    
    def update_epoch(self, epoch: int):
        """Update current epoch for alpha_d scheduling."""
        self.current_epoch = epoch


#######################
# Complete HypCD Model
#######################

class HypCD(nn.Module):
    """
    Main Hyperbolic Category Discovery model.
    """
    def __init__(self, 
                 backbone: nn.Module,
                 feature_dim: int,
                 proj_dim: int,
                 num_classes: int,
                 curvature: float = 0.05,
                 clip_value: float = 2.3,
                 temperature: float = 0.1,
                 alpha_d: float = 0.0,
                 max_alpha_d: float = 1.0,
                 alpha_epochs: int = 200,
                 method: str = 'parametric'):
        """
        Args:
            backbone: Pretrained feature extractor (e.g., DINO ViT)
            feature_dim: Dimension of features from the backbone
            proj_dim: Dimension for the projection head
            num_classes: Total number of classes (seen + unseen)
            curvature: Curvature parameter of hyperbolic space
            clip_value: Maximum feature norm before mapping
            temperature: Temperature for contrastive loss
            alpha_d: Initial weight for distance-based loss
            max_alpha_d: Maximum weight for distance-based loss
            alpha_epochs: Number of epochs to increase alpha_d
            method: 'parametric' or 'non-parametric' approach
        """
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.curvature = curvature
        self.method = method
        
        # Projection head (Euclidean)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # Hyperbolic mapping
        self.euclidean_to_hyperbolic = EuclideanToHyperbolic(curvature, clip_value)
        
        # Loss function
        self.contrastive_loss = HyperbolicContrastiveLoss(
            temperature=temperature,
            c=curvature,
            alpha_d=alpha_d,
            max_alpha_d=max_alpha_d,
            alpha_epochs=alpha_epochs
        )
        
        # For parametric methods, add a hyperbolic classifier
        if method == 'parametric':
            self.classifier = HyperbolicFFN(
                input_dim=proj_dim,
                hidden_dim=proj_dim*2,
                output_dim=num_classes,
                c=curvature
            )
            
            # Initialize class prototypes
            self.register_buffer('prototypes', torch.zeros(num_classes, proj_dim))
            self.initialized = False
        
    def initialize_prototypes(self, features: torch.Tensor, labels: torch.Tensor):
        """Initialize class prototypes with average class features."""
        if self.initialized:
            return
        
        for c in range(self.num_classes):
            # For unseen classes, initialize with random features
            if c not in labels:
                continue
                
            # Get features for this class
            mask = (labels == c)
            if not mask.any():
                continue
                
            class_features = features[mask]
            
            # Average in Euclidean space
            avg_features = class_features.mean(dim=0)
            
            # Normalize
            avg_features = F.normalize(avg_features, p=2, dim=0)
            
            # Map to hyperbolic space
            hyp_avg = self.euclidean_to_hyperbolic(avg_features.unsqueeze(0)).squeeze(0)
            
            # Store
            self.prototypes[c] = hyp_avg
            
        self.initialized = True
    
    def update_prototypes(self, features: torch.Tensor, probs: torch.Tensor, momentum: float = 0.9):
        """Update class prototypes with weighted average of features."""
        # Get predicted class (max probability)
        _, pred_classes = probs.max(dim=1)
        
        # Update each prototype
        for c in range(self.num_classes):
            # Get features assigned to this class
            mask = (pred_classes == c)
            if not mask.any():
                continue
                
            class_features = features[mask]
            class_probs = probs[mask, c].unsqueeze(1)
            
            # Weighted average in Euclidean space (tangent space)
            weighted_avg = (class_features * class_probs).sum(dim=0) / class_probs.sum()
            
            # Normalize
            weighted_avg = F.normalize(weighted_avg, p=2, dim=0)
            
            # Map to hyperbolic space
            hyp_avg = self.euclidean_to_hyperbolic(weighted_avg.unsqueeze(0)).squeeze(0)
            
            # Update with momentum
            self.prototypes[c] = momentum * self.prototypes[c] + (1 - momentum) * hyp_avg
            
            # Project back to manifold
            norm = torch.norm(self.prototypes[c])
            max_norm = (1 - EPS) / np.sqrt(self.curvature)
            if norm > max_norm:
                self.prototypes[c] = self.prototypes[c] * max_norm / norm

    def get_hyperbolic_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract hyperbolic features from input images."""
        # Extract features
        with torch.no_grad():
            features = self.backbone(x)
        
        # Apply projector
        projected = self.projector(features)
        
        # Normalize
        projected_norm = F.normalize(projected, p=2, dim=1)
        
        # Map to hyperbolic space
        hyperbolic_features = self.euclidean_to_hyperbolic(projected_norm)
        
        return hyperbolic_features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification."""
        # Get hyperbolic features
        hyp_features = self.get_hyperbolic_features(x)
        
        if self.method == 'parametric':
            # Use hyperbolic classifier
            logits = self.classifier(hyp_features)
            
            # Convert to tangent space for softmax
            logits_tan = HyperbolicOps.logmap0(logits, self.curvature)
            
            return logits_tan
        else:
            # Non-parametric approach: compute distances to prototypes
            distances = torch.zeros(x.size(0), self.num_classes, device=x.device)
            
            for i in range(x.size(0)):
                for c in range(self.num_classes):
                    distances[i, c] = HyperbolicOps.distance(
                        hyp_features[i].unsqueeze(0), 
                        self.prototypes[c].unsqueeze(0), 
                        self.curvature
                    ).squeeze()
            
            # Return negative distances as logits
            return -distances
    
    def compute_contrastive_loss(self, 
                                z1: torch.Tensor, 
                                z2: torch.Tensor, 
                                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute contrastive loss in hyperbolic space."""
        # Map to hyperbolic space
        z1_hyp = self.euclidean_to_hyperbolic(z1)
        z2_hyp = self.euclidean_to_hyperbolic(z2)
        
        # Compute loss
        loss = self.contrastive_loss(z1_hyp, z2_hyp, labels)
        
        return loss
    
    def update_epoch(self, epoch: int):
        """Update epoch for loss scheduling."""
        self.contrastive_loss.update_epoch(epoch)


#######################
# Training Functions
#######################

def train_hypcd_with_simgcd(
    model: HypCD,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    num_classes: int,
    lambda_b: float = 0.35,
    entropy_weight: float = 1.0,
    device: torch.device = torch.device('cuda'),
    labeled_mask: Optional[torch.Tensor] = None,
):
    """
    Train HypCD with SimGCD-style objective.
    
    Args:
        model: HypCD model
        dataloader: DataLoader for training
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        num_classes: Total number of classes
        lambda_b: Balance factor between supervised and unsupervised loss
        entropy_weight: Weight for entropy loss
        device: Device to train on
        labeled_mask: Boolean mask indicating labeled samples
    """
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.update_epoch(epoch)
        
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            imgs, labels, indices = batch
            imgs1, imgs2 = imgs[:, 0].to(device), imgs[:, 1].to(device)
            labels = labels.to(device)
            
            # Create mask for labeled samples
            if labeled_mask is None:
                # Assume first half are labeled (simplified)
                batch_size = labels.size(0)
                labeled_indices = torch.arange(0, batch_size // 2, device=device)
                labeled_batch_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                labeled_batch_mask[labeled_indices] = True
            else:
                # Use provided mask for the current batch
                batch_labeled_mask = labeled_mask[indices].to(device)
            
            # Forward passes
            optimizer.zero_grad()
            
            # Get features and project to hyperbolic space
            with torch.no_grad():
                features1 = model.backbone(imgs1)
                features2 = model.backbone(imgs2)
            
            z1 = F.normalize(model.projector(features1), p=2, dim=1)
            z2 = F.normalize(model.projector(features2), p=2, dim=1)
            
            # Compute contrastive loss
            labeled_z1 = z1[batch_labeled_mask] if batch_labeled_mask.any() else None
            labeled_z2 = z2[batch_labeled_mask] if batch_labeled_mask.any() else None
            labeled_y = labels[batch_labeled_mask] if batch_labeled_mask.any() else None
            
            # Unsupervised contrastive loss
            loss_u_rep = model.compute_contrastive_loss(z1, z2)
            
            # Supervised contrastive loss (if labeled data exists)
            loss_s_rep = torch.tensor(0.0, device=device)
            if labeled_z1 is not None and labeled_z1.size(0) > 0:
                loss_s_rep = model.compute_contrastive_loss(labeled_z1, labeled_z2, labeled_y)
            
            # Combine representation losses
            loss_rep = (1 - lambda_b) * loss_u_rep + lambda_b * loss_s_rep
            
            # Classification loss
            logits1 = model(imgs1)
            logits2 = model(imgs2)
            probs1 = F.softmax(logits1, dim=1)
            probs2 = F.softmax(logits2, dim=1)
            
            # Mean probability
            mean_probs = (probs1 + probs2) / 2
            
            # Supervised classification loss
            loss_s_cls = torch.tensor(0.0, device=device)
            if batch_labeled_mask.any():
                labeled_logits1 = logits1[batch_labeled_mask]
                labeled_y = labels[batch_labeled_mask]
                loss_s_cls = F.cross_entropy(labeled_logits1, labeled_y)
            
            # Unsupervised classification loss with mean entropy
            mean_entropy = -torch.mean(torch.sum(mean_probs * torch.log(mean_probs + EPS), dim=1))
            loss_u_cls = 0.5 * (F.cross_entropy(logits1, torch.argmax(probs2, dim=1)) + 
                               F.cross_entropy(logits2, torch.argmax(probs1, dim=1)))
            loss_u_cls = loss_u_cls - entropy_weight * mean_entropy
            
            # Combine classification losses
            loss_cls = (1 - lambda_b) * loss_u_cls + lambda_b * loss_s_cls
            
            # Total loss
            loss = loss_rep + loss_cls
            loss.backward()
            optimizer.step()
            
            # Update prototypes
            if model.method == 'parametric' and epoch > 0:
                with torch.no_grad():
                    model.update_prototypes(z1, probs1)
            
            epoch_loss += loss.item()
            
            # Log progress
            if batch_idx % 50 == 0:
                logger.info(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f'Epoch: {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}')


def train_hypcd_with_gcd(
    model: HypCD,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    lambda_b: float = 0.35,
    device: torch.device = torch.device('cuda'),
    labeled_mask: Optional[torch.Tensor] = None,
):
    """
    Train HypCD with the non-parametric GCD approach.
    
    Args:
        model: HypCD model
        dataloader: DataLoader for training
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        lambda_b: Balance factor between supervised and unsupervised loss
        device: Device to train on
        labeled_mask: Boolean mask indicating labeled samples
    """
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.update_epoch(epoch)
        
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            imgs, labels, indices = batch
            imgs1, imgs2 = imgs[:, 0].to(device), imgs[:, 1].to(device)
            labels = labels.to(device)
            
            # Create mask for labeled samples
            if labeled_mask is None:
                # Assume first half are labeled (simplified)
                batch_size = labels.size(0)
                labeled_indices = torch.arange(0, batch_size // 2, device=device)
                labeled_batch_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                labeled_batch_mask[labeled_indices] = True
            else:
                # Use provided mask for the current batch
                batch_labeled_mask = labeled_mask[indices].to(device)
            
            # Forward passes
            optimizer.zero_grad()
            
            # Get features and project
            with torch.no_grad():
                features1 = model.backbone(imgs1)
                features2 = model.backbone(imgs2)
            
            z1 = F.normalize(model.projector(features1), p=2, dim=1)
            z2 = F.normalize(model.projector(features2), p=2, dim=1)
            
            # Unsupervised contrastive loss
            loss_u_rep = model.compute_contrastive_loss(z1, z2)
            
            # Supervised contrastive loss (if labeled data exists)
            loss_s_rep = torch.tensor(0.0, device=device)
            if batch_labeled_mask.any():
                labeled_z1 = z1[batch_labeled_mask]
                labeled_z2 = z2[batch_labeled_mask]
                labeled_y = labels[batch_labeled_mask]
                loss_s_rep = model.compute_contrastive_loss(labeled_z1, labeled_z2, labeled_y)
            
            # Combine losses
            loss = (1 - lambda_b) * loss_u_rep + lambda_b * loss_s_rep
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Log progress
            if batch_idx % 50 == 0:
                logger.info(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f'Epoch: {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}')


def train_hypcd_with_selex(
    model: HypCD,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    num_classes: int,
    lambda_b: float = 0.35,
    alpha: float = 1.0,  # Label smoothing parameter
    device: torch.device = torch.device('cuda'),
    labeled_mask: Optional[torch.Tensor] = None,
    num_hierarchical_levels: int = 3,
    pseudo_labels: Optional[Dict[int, List[torch.Tensor]]] = None,
):
    """
    Train HypCD with the SelEx approach.
    
    Args:
        model: HypCD model
        dataloader: DataLoader for training
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        num_classes: Total number of classes
        lambda_b: Balance factor between supervised and unsupervised loss
        alpha: Label smoothing parameter
        device: Device to train on
        labeled_mask: Boolean mask indicating labeled samples
        num_hierarchical_levels: Number of hierarchical levels in SelEx
        pseudo_labels: Dictionary with pseudo-labels for each hierarchical level
    """
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.update_epoch(epoch)
        
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            imgs, labels, indices = batch
            imgs1, imgs2 = imgs[:, 0].to(device), imgs[:, 1].to(device)
            labels = labels.to(device)
            
            # Create mask for labeled samples
            if labeled_mask is None:
                # Assume first half are labeled (simplified)
                batch_size = labels.size(0)
                labeled_indices = torch.arange(0, batch_size // 2, device=device)
                labeled_batch_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                labeled_batch_mask[labeled_indices] = True
            else:
                # Use provided mask for the current batch
                batch_labeled_mask = labeled_mask[indices].to(device)
            
            # Forward passes
            optimizer.zero_grad()
            
            # Get features and project
            with torch.no_grad():
                features1 = model.backbone(imgs1)
                features2 = model.backbone(imgs2)
            
            z1 = F.normalize(model.projector(features1), p=2, dim=1)
            z2 = F.normalize(model.projector(features2), p=2, dim=1)
            
            # Get hyperbolic features for both views
            z1_hyp = model.euclidean_to_hyperbolic(z1)
            z2_hyp = model.euclidean_to_hyperbolic(z2)
            
            # Get batch pseudo-labels for all levels
            batch_pseudo_labels = {}
            if pseudo_labels is not None:
                for level in range(num_hierarchical_levels):
                    batch_pseudo_labels[level] = pseudo_labels[level][indices].to(device)
            
            # Compute hyperbolic distance between all pairs
            batch_size = z1.size(0)
            distance_matrix = torch.zeros((batch_size, batch_size), device=device)
            
            for i in range(batch_size):
                for j in range(batch_size):
                    distance_matrix[i, j] = HyperbolicOps.distance(
                        z1_hyp[i].unsqueeze(0),
                        z2_hyp[j].unsqueeze(0),
                        model.curvature
                    ).squeeze()
            
            # Compute target matrix for unsupervised self-expertise
            target_matrix = torch.eye(batch_size, device=device)
            
            if batch_pseudo_labels:
                # For each hierarchical level, adjust target matrix
                hierarchical_target = torch.zeros_like(target_matrix)
                
                for level in range(num_hierarchical_levels):
                    level_targets = batch_pseudo_labels[level]
                    for i in range(batch_size):
                        for j in range(batch_size):
                            if level_targets[i] == level_targets[j]:
                                # Same pseudo-label at this level
                                hierarchical_target[i, j] += 1.0 / num_hierarchical_levels
                
                # Apply label smoothing with alpha
                target_matrix = alpha * target_matrix + (1 - alpha) * hierarchical_target
            
            # Unsupervised self-expertise loss using hyperbolic distance
            probs = F.softmax(-distance_matrix / model.contrastive_loss.temperature, dim=1)
            loss_use = F.binary_cross_entropy_with_logits(torch.log(probs + EPS), target_matrix)
            
            # Supervised self-expertise loss for labeled samples
            loss_sse = torch.tensor(0.0, device=device)
            
            if batch_labeled_mask.any():
                labeled_count = batch_labeled_mask.sum().item()
                if labeled_count > 1:  # Need at least 2 samples for contrastive
                    # Compute hierarchical supervised loss across all levels
                    loss_sse_levels = []
                    
                    # Split the feature vector into equal parts for each level
                    feature_size = z1.size(1)
                    level_size = feature_size // num_hierarchical_levels
                    
                    for level in range(num_hierarchical_levels):
                        start_idx = level * level_size
                        end_idx = (level + 1) * level_size if level < num_hierarchical_levels - 1 else feature_size
                        
                        # Get features for this level
                        level_z1 = z1[:, start_idx:end_idx]
                        level_z2 = z2[:, start_idx:end_idx]
                        
                        # Normalize
                        level_z1 = F.normalize(level_z1, p=2, dim=1)
                        level_z2 = F.normalize(level_z2, p=2, dim=1)
                        
                        # Map to hyperbolic space
                        level_z1_hyp = model.euclidean_to_hyperbolic(level_z1)
                        level_z2_hyp = model.euclidean_to_hyperbolic(level_z2)
                        
                        # Compute supervised contrastive loss for this level
                        level_labeled_z1 = level_z1_hyp[batch_labeled_mask]
                        level_labeled_z2 = level_z2_hyp[batch_labeled_mask]
                        labeled_y = labels[batch_labeled_mask]
                        
                        # First compute angle-based loss
                        level_loss_angle = model.compute_contrastive_loss(
                            level_z1[batch_labeled_mask], 
                            level_z2[batch_labeled_mask], 
                            labeled_y
                        )
                        
                        # Then compute distance-based loss for this level
                        level_dist_matrix = torch.zeros((labeled_count, labeled_count), device=device)
                        labeled_indices = torch.where(batch_labeled_mask)[0]
                        
                        for i, idx_i in enumerate(labeled_indices):
                            for j, idx_j in enumerate(labeled_indices):
                                level_dist_matrix[i, j] = HyperbolicOps.distance(
                                    level_z1_hyp[idx_i].unsqueeze(0),
                                    level_z2_hyp[idx_j].unsqueeze(0),
                                    model.curvature
                                ).squeeze()
                        
                        level_loss_dist = 0.0
                        for i in range(labeled_count):
                            label_i = labeled_y[i]
                            pos_indices = (labeled_y == label_i).nonzero(as_tuple=True)[0]
                            pos_indices = pos_indices[pos_indices != i]  # Exclude self
                            
                            if len(pos_indices) > 0:
                                neg_indices = (labeled_y != label_i).nonzero(as_tuple=True)[0]
                                
                                if len(neg_indices) > 0:
                                    pos_distances = level_dist_matrix[i, pos_indices]
                                    neg_distances = level_dist_matrix[i, neg_indices]
                                    
                                    # Compute contrastive loss
                                    pos_term = torch.logsumexp(-pos_distances / model.contrastive_loss.temperature, dim=0)
                                    neg_term = torch.logsumexp(-neg_distances / model.contrastive_loss.temperature, dim=0)
                                    level_loss_dist += neg_term - pos_term
                        
                        if labeled_count > 0:
                            level_loss_dist /= labeled_count
                        
                        # Combine angle and distance losses for this level
                        current_alpha_d = min(model.contrastive_loss.alpha_d + 
                                           (model.contrastive_loss.max_alpha_d - model.contrastive_loss.alpha_d) * 
                                           epoch / model.contrastive_loss.alpha_epochs, 
                                           model.contrastive_loss.max_alpha_d)
                        
                        level_loss = current_alpha_d * level_loss_dist + (1 - current_alpha_d) * level_loss_angle
                        level_loss_scaled = level_loss / (2**level)  # Higher levels have less weight
                        loss_sse_levels.append(level_loss_scaled)
                    
                    # Sum losses from all levels
                    if loss_sse_levels:
                        loss_sse = sum(loss_sse_levels) / num_hierarchical_levels
            
            # Combine losses
            loss = (1 - lambda_b) * loss_use + lambda_b * loss_sse
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Log progress
            if batch_idx % 50 == 0:
                logger.info(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        logger.info(f'Epoch: {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}')


#######################
# Evaluation Functions
#######################

@torch.no_grad()
def extract_features(
    model: HypCD,
    dataloader: torch.utils.data.DataLoader,
    use_hyperbolic: bool = True,
    device: torch.device = torch.device('cuda')
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract features from the model for all samples in the dataloader.
    
    Args:
        model: HypCD model
        dataloader: DataLoader for evaluation
        use_hyperbolic: Whether to return hyperbolic or Euclidean features
        device: Device to evaluate on
        
    Returns:
        tuple of (features, labels, indices)
    """
    model.eval()
    all_features = []
    all_labels = []
    all_indices = []
    
    for batch in dataloader:
        # Unpack batch
        imgs, labels, indices = batch
        imgs = imgs.to(device)
        
        # Get backbone features
        features = model.backbone(imgs)
        
        # Apply projector and normalize
        projected = model.projector(features)
        normalized = F.normalize(projected, p=2, dim=1)
        
        if use_hyperbolic:
            # Map to hyperbolic space
            hyperbolic_features = model.euclidean_to_hyperbolic(normalized)
            all_features.append(hyperbolic_features.cpu())
        else:
            all_features.append(normalized.cpu())
        
        all_labels.append(labels)
        all_indices.append(indices)
    
    # Concatenate
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    
    return all_features, all_labels, all_indices


def assign_labels_parametric(
    model: HypCD,
    test_dataloader: torch.utils.data.DataLoader,
    device: torch.device = torch.device('cuda')
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assign labels to unlabeled samples using the parametric hyperbolic classifier.
    
    Args:
        model: HypCD model
        test_dataloader: DataLoader for unlabeled samples
        device: Device to evaluate on
        
    Returns:
        tuple of (predicted_labels, ground_truth_labels, indices)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Unpack batch
            imgs, labels, indices = batch
            imgs = imgs.to(device)
            
            # Get logits
            logits = model(imgs)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_indices.append(indices)
    
    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    
    return all_preds, all_labels, all_indices


def assign_labels_semi_supervised_kmeans(
    features: torch.Tensor,
    labels: torch.Tensor,
    labeled_mask: torch.Tensor,
    num_classes: int,
    device: torch.device = torch.device('cuda'),
    distance_type: str = 'euclidean',
    curvature: float = 0.05,
    max_iter: int = 100,
) -> torch.Tensor:
    """
    Assign labels to unlabeled samples using semi-supervised k-means clustering.
    
    Args:
        features: Feature vectors (can be in Euclidean or hyperbolic space)
        labels: Ground-truth labels (only valid for labeled samples)
        labeled_mask: Boolean mask indicating labeled samples
        num_classes: Total number of classes
        device: Device to evaluate on
        distance_type: 'euclidean' or 'hyperbolic'
        curvature: Curvature parameter for hyperbolic distance
        max_iter: Maximum iterations for k-means
        
    Returns:
        Tensor of predicted labels for all samples
    """
    features = features.to(device)
    labels = labels.to(device)
    labeled_mask = labeled_mask.to(device)
    
    # Initialize centroids using class means of labeled samples
    centroids = torch.zeros(num_classes, features.size(1), device=device)
    
    for c in range(num_classes):
        # Get labeled samples of this class
        class_mask = (labels == c) & labeled_mask
        
        if class_mask.any():
            # Compute mean in Euclidean space
            if distance_type == 'hyperbolic':
                # For hyperbolic features, first map to tangent space
                tangent_features = HyperbolicOps.logmap0(features[class_mask], curvature)
                mean_tangent = tangent_features.mean(dim=0)
                # Map back to hyperbolic space
                centroids[c] = HyperbolicOps.expmap0(mean_tangent.unsqueeze(0), curvature).squeeze(0)
            else:
                centroids[c] = features[class_mask].mean(dim=0)
    
    # Classes with no labeled samples get random initialization
    empty_classes = torch.where(torch.norm(centroids, dim=1) == 0)[0]
    if len(empty_classes) > 0:
        for c in empty_classes:
            # Random initialization in the feature space
            if distance_type == 'hyperbolic':
                # Initialize in tangent space and map to hyperbolic space
                random_tangent = torch.randn(features.size(1), device=device)
                random_tangent = 0.1 * F.normalize(random_tangent, p=2, dim=0)  # Small norm
                centroids[c] = HyperbolicOps.expmap0(random_tangent.unsqueeze(0), curvature).squeeze(0)
            else:
                random_feat = torch.randn(features.size(1), device=device)
                centroids[c] = F.normalize(random_feat, p=2, dim=0)
    
    # Ensure centroids are valid for hyperbolic space
    if distance_type == 'hyperbolic':
        centroids = HyperbolicOps.project(centroids, curvature)
    
    # Initialize assignments
    assignments = torch.zeros(features.size(0), dtype=torch.long, device=device)
    
    # For labeled samples, use ground truth labels
    assignments[labeled_mask] = labels[labeled_mask]
    
    # Iterate until convergence or max iterations
    for iteration in range(max_iter):
        # Compute distances to centroids
        if distance_type == 'hyperbolic':
            # Hyperbolic distance
            distances = torch.zeros(features.size(0), num_classes, device=device)
            for i in range(features.size(0)):
                for c in range(num_classes):
                    distances[i, c] = HyperbolicOps.distance(
                        features[i].unsqueeze(0),
                        centroids[c].unsqueeze(0),
                        curvature
                    ).squeeze()
        else:
            # Euclidean distance
            distances = torch.cdist(features, centroids)
        
        # Update assignments for unlabeled samples
        new_assignments = torch.argmin(distances, dim=1)
        unlabeled_mask = ~labeled_mask
        assignments[unlabeled_mask] = new_assignments[unlabeled_mask]
        
        # Update centroids based on new assignments
        new_centroids = torch.zeros_like(centroids)
        for c in range(num_classes):
            class_mask = assignments == c
            
            if class_mask.any():
                if distance_type == 'hyperbolic':
                    # For hyperbolic features, first map to tangent space
                    tangent_features = HyperbolicOps.logmap0(features[class_mask], curvature)
                    mean_tangent = tangent_features.mean(dim=0)
                    # Map back to hyperbolic space
                    new_centroids[c] = HyperbolicOps.expmap0(mean_tangent.unsqueeze(0), curvature).squeeze(0)
                else:
                    new_centroids[c] = features[class_mask].mean(dim=0)
                    
                    # Normalize Euclidean centroids
                    if distance_type == 'euclidean':
                        new_centroids[c] = F.normalize(new_centroids[c], p=2, dim=0)
        
        # Ensure centroids are valid for hyperbolic space
        if distance_type == 'hyperbolic':
            new_centroids = HyperbolicOps.project(new_centroids, curvature)
        
        # Check convergence
        if torch.all(assignments == new_assignments):
            break
        
        centroids = new_centroids
    
    return assignments


def evaluate_clustering(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    class_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate clustering performance using clustering accuracy.
    
    Args:
        predictions: Predicted cluster assignments
        ground_truth: Ground truth labels
        class_names: Optional list of class names for detailed results
        
    Returns:
        Dictionary of metrics
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    
    # Convert to numpy
    preds = predictions.cpu().numpy()
    gt = ground_truth.cpu().numpy()
    
    # Create confusion matrix
    num_samples = len(gt)
    num_classes = max(len(np.unique(gt)), len(np.unique(preds)))
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Fill confusion matrix
    for i in range(num_samples):
        confusion[gt[i], preds[i]] += 1
    
    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(-confusion)
    
    # Calculate accuracy
    acc = confusion[row_ind, col_ind].sum() / num_samples
    
    # Calculate per-class accuracy
    per_class_acc = {}
    if class_names is not None:
        for i, class_idx in enumerate(row_ind):
            if class_idx < len(class_names):
                class_name = class_names[class_idx]
                class_total = np.sum(gt == class_idx)
                class_correct = confusion[class_idx, col_ind[i]]
                per_class_acc[class_name] = class_correct / class_total if class_total > 0 else 0.0
    
    # Calculate accuracy for "Old" and "New" classes separately
    # Assuming the first half of classes are "Old" and the rest are "New"
    old_indices = row_ind < num_classes // 2
    old_classes = row_ind[old_indices]
    old_total = np.sum(np.isin(gt, old_classes))
    old_correct = np.sum(confusion[old_classes, col_ind[old_indices]])
    old_acc = old_correct / old_total if old_total > 0 else 0.0
    
    new_indices = ~old_indices
    new_classes = row_ind[new_indices]
    new_total = np.sum(np.isin(gt, new_classes))
    new_correct = np.sum(confusion[new_classes, col_ind[new_indices]])
    new_acc = new_correct / new_total if new_total > 0 else 0.0
    
    return {
        'acc': acc,
        'old_acc': old_acc,
        'new_acc': new_acc,
        'per_class_acc': per_class_acc
    }


def visualize_features(
    features: torch.Tensor,
    labels: torch.Tensor,
    method: str = 'tsne',
    title: str = 'Feature Visualization',
    save_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
):
    """
    Visualize feature embeddings using t-SNE or UMAP.
    
    Args:
        features: Feature vectors
        labels: Class labels
        method: 'tsne' or 'umap'
        title: Plot title
        save_path: Path to save the visualization
        class_names: Optional list of class names
    """
    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Reduce dimensionality
    if method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=0)
        reduced_features = reducer.fit_transform(features_np)
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(random_state=0)
            reduced_features = reducer.fit_transform(features_np)
        except ImportError:
            logger.warning("UMAP not installed. Falling back to t-SNE.")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=0)
            reduced_features = reducer.fit_transform(features_np)
    else:
        raise ValueError(f"Unknown visualization method: {method}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels_np)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels_np == label
        plt.scatter(
            reduced_features[mask, 0],
            reduced_features[mask, 1],
            c=[colors[i]],
            label=class_names[label] if class_names and label < len(class_names) else f"Class {label}",
            alpha=0.7
        )
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    plt.show()


#######################
# Main Implementation
#######################

def main():
    """Main function for training and evaluating HypCD models."""
    import argparse
    import torch.backends.cudnn as cudnn
    import time
    from pathlib import Path
    from PIL import Image
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    
    parser = argparse.ArgumentParser(description='Hyperbolic Category Discovery')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet100', 'cub', 'scars', 'aircraft'])
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--split_file', type=str, default=None, help='Path to split file')
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='dino_vitb16', choices=['dino_vitb16', 'dinov2_vitb14'])
    parser.add_argument('--proj_dim', type=int, default=256, help='Dimension of projection head')
    parser.add_argument('--curvature', type=float, default=None, help='Curvature parameter (0.05 for fine-grained, 0.01 for generic)')
    parser.add_argument('--clip_value', type=float, default=None, help='Clipping value (2.3 for fine-grained, 1.0 for generic)')
    parser.add_argument('--method', type=str, default='simgcd', choices=['simgcd', 'gcd', 'selex'])
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive loss')
    parser.add_argument('--alpha_d', type=float, default=0.0, help='Initial weight for distance-based loss')
    parser.add_argument('--max_alpha_d', type=float, default=None, help='Maximum weight for distance-based loss')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lambda_b', type=float, default=0.35, help='Balance factor for loss components')
    parser.add_argument('--entropy_weight', type=float, default=1.0, help='Weight for entropy loss in SimGCD')
    parser.add_argument('--alpha', type=float, default=None, help='Label smoothing parameter for SelEx')
    
    # Other parameters
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--eval_only', action='store_true', help='Evaluation only')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Create save directory
    save_dir = Path(args.save_dir) / f"{args.dataset}_{args.method}_{args.backbone}_c{args.curvature}"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {save_dir}")
    
    # Setup dataset-specific parameters
    if args.curvature is None:
        # Set default curvature based on dataset type
        if args.dataset in ['cub', 'scars', 'aircraft']:
            args.curvature = 0.05  # Fine-grained
        else:
            args.curvature = 0.01  # Generic
    
    if args.clip_value is None:
        # Set default clipping value based on dataset type
        if args.dataset in ['cub', 'scars', 'aircraft']:
            args.clip_value = 2.3  # Fine-grained
        else:
            args.clip_value = 1.0  # Generic
    
    if args.max_alpha_d is None:
        # Set default max_alpha_d based on dataset type
        if args.dataset in ['cub', 'scars', 'aircraft']:
            args.max_alpha_d = 1.0  # Fine-grained
        else:
            args.max_alpha_d = 0.5  # Generic
            
    if args.alpha is None:
        # Set default alpha for SelEx based on dataset
        if args.dataset == 'aircraft':
            args.alpha = 0.5
        elif args.dataset in ['cub', 'scars']:
            args.alpha = 1.0
        else:
            args.alpha = 0.1
    
    # Define dataset and dataloader
    # This would typically be implemented with a proper dataset class
    # For simplicity, we'll use a placeholder dataset here
    
    class PlaceholderDataset(Dataset):
        def __init__(self, data_path, split='train', transform=None):
            self.data_path = Path(data_path)
            self.split = split
            self.transform = transform
            
            # Placeholder: In a real implementation, load the actual dataset here
            # self.data = ...
            # self.targets = ...
            
            # For demonstration, create random data
            self.data = torch.randn(1000, 3, 224, 224)
            self.targets = torch.randint(0, 10, (1000,))
            
            # Assume first 50% of classes are labeled
            num_classes = len(torch.unique(self.targets))
            self.labeled_classes = torch.arange(num_classes // 2)
            
            # Create labeled/unlabeled mask
            self.labeled_mask = torch.zeros_like(self.targets, dtype=torch.bool)
            for c in self.labeled_classes:
                # 50% of samples from labeled classes are labeled
                class_indices = (self.targets == c).nonzero().flatten()
                labeled_indices = class_indices[:len(class_indices) // 2]
                self.labeled_mask[labeled_indices] = True
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            img = self.data[idx]
            target = self.targets[idx]
            
            if self.transform:
                # In practice, apply transforms to PIL image
                # img = self.transform(img)
                pass
            
            return img, target, idx
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets (placeholder for demonstration)
    train_dataset = PlaceholderDataset(args.data_path, split='train', transform=train_transform)
    val_dataset = PlaceholderDataset(args.data_path, split='val', transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Get number of classes
    num_classes = len(torch.unique(train_dataset.targets))
    
    # Load backbone model
    if args.backbone == 'dino_vitb16':
        import timm
        backbone = timm.create_model('vit_base_patch16_224.dino', pretrained=True)
        feature_dim = backbone.num_features
    elif args.backbone == 'dinov2_vitb14':
        import timm
        backbone = timm.create_model('vit_base_patch14_dinov2', pretrained=True)
        feature_dim = backbone.num_features
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")
    
    # Create HypCD model
    model = HypCD(
        backbone=backbone,
        feature_dim=feature_dim,
        proj_dim=args.proj_dim,
        num_classes=num_classes,
        curvature=args.curvature,
        clip_value=args.clip_value,
        temperature=args.temperature,
        alpha_d=args.alpha_d,
        max_alpha_d=args.max_alpha_d,
        alpha_epochs=args.num_epochs,
        method=args.method
    ).to(device)
    
    # Create optimizer
    if args.method == 'simgcd':
        # For SimGCD, only optimize projection head and classifier
        optimizer = SGD(
            [
                {'params': model.projector.parameters(), 'lr': args.lr},
                {'params': model.classifier.parameters() if args.method == 'parametric' else [], 'lr': args.lr}
            ],
            momentum=0.9
        )
    else:
        # For GCD and SelEx, only optimize projection head
        optimizer = SGD(model.projector.parameters(), lr=args.lr, momentum=0.9)
    
    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.001)
    
    # Training
    if not args.eval_only:
        logger.info(f"Starting training with {args.method} method")
        
        # Choose training function based on method
        if args.method == 'simgcd':
            train_hypcd_with_simgcd(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=args.num_epochs,
                num_classes=num_classes,
                lambda_b=args.lambda_b,
                entropy_weight=args.entropy_weight,
                device=device,
                labeled_mask=train_dataset.labeled_mask
            )
        elif args.method == 'gcd':
            train_hypcd_with_gcd(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=args.num_epochs,
                lambda_b=args.lambda_b,
                device=device,
                labeled_mask=train_dataset.labeled_mask
            )
        elif args.method == 'selex':
            # For SelEx, we need to generate pseudo-labels first
            # This would typically be done using hierarchical semi-supervised k-means
            # For simplicity, we'll use a placeholder here
            
            # Placeholder for pseudo-labels
            pseudo_labels = {
                level: torch.randint(0, num_classes // (2**level), (len(train_dataset),))
                for level in range(3)  # 3 hierarchical levels
            }
            
            train_hypcd_with_selex(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=args.num_epochs,
                num_classes=num_classes,
                lambda_b=args.lambda_b,
                alpha=args.alpha,
                device=device,
                labeled_mask=train_dataset.labeled_mask,
                num_hierarchical_levels=3,
                pseudo_labels=pseudo_labels
            )
        
        # Save model
        torch.save(model.state_dict(), save_dir / 'model.pt')
        logger.info(f"Model saved to {save_dir / 'model.pt'}")
    
    # Evaluation
    logger.info("Starting evaluation")
    
    # Extract features
    features, labels, indices = extract_features(model, val_loader, use_hyperbolic=True, device=device)
    
    # Assign labels
    if args.method == 'simgcd':
        # For SimGCD, use parametric classifier
        predicted_labels, ground_truth, _ = assign_labels_parametric(model, val_loader, device)
    else:
        # For GCD and SelEx, use semi-supervised k-means
        predicted_labels = assign_labels_semi_supervised_kmeans(
            features=features,
            labels=labels,
            labeled_mask=val_dataset.labeled_mask,
            num_classes=num_classes,
            device=device,
            distance_type='hyperbolic',
            curvature=args.curvature
        )
        ground_truth = labels
    
    # Evaluate
    metrics = evaluate_clustering(predicted_labels, ground_truth)
    
    logger.info(f"Clustering accuracy:")
    logger.info(f"  All classes: {metrics['acc']:.4f}")
    logger.info(f"  Old classes: {metrics['old_acc']:.4f}")
    logger.info(f"  New classes: {metrics['new_acc']:.4f}")
    
    # Visualize features
    visualize_features(
        features=features,
        labels=labels,
        method='tsne',
        title=f"{args.dataset} - {args.method} - c={args.curvature}",
        save_path=save_dir / 'tsne.png'
    )


if __name__ == "__main__":
    main()