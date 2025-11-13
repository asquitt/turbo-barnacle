"""
Enhanced GNN Architectures for Materials Discovery

This module implements state-of-the-art GNN architectures with:
- Graph Attention Networks (GAT) for adaptive neighbor weighting
- Radial Basis Function (RBF) edge features for distance encoding
- Attention-based global pooling
- Layer normalization and improved residuals
- Multiple architecture options (CGCNN-inspired, GAT, Transformer-like)

Research-backed improvements:
- RBF expansion for continuous distance features (SchNet-inspired)
- Attention mechanisms for learnable aggregation
- Proper normalization for training stability
- Ensemble-ready architecture

Author: Materials Discovery Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    SAGEConv, GATv2Conv, global_mean_pool, global_max_pool,
    global_add_pool, GlobalAttention, Set2Set
)
from torch_geometric.data import Data
from typing import Optional, Tuple, Dict, List
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)


class RBFExpansion(nn.Module):
    """
    Radial Basis Function expansion for distance features.

    Converts scalar distances into high-dimensional features using
    Gaussian RBF kernels. This helps the network learn smooth distance-dependent
    interactions, similar to SchNet and DimeNet.

    Formula: RBF_k(d) = exp(-gamma * (d - mu_k)^2)

    Args:
        num_rbf: Number of RBF kernels
        cutoff: Maximum distance for interactions
        learnable: Whether RBF parameters are learnable
    """

    def __init__(
        self,
        num_rbf: int = 20,
        cutoff: float = 8.0,
        learnable: bool = False,
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        # Initialize RBF centers (mu) uniformly spaced
        mu = torch.linspace(0, cutoff, num_rbf)

        # Initialize widths (gamma) for smooth overlap
        gamma = torch.ones(num_rbf) * (cutoff / num_rbf) ** 2

        if learnable:
            self.mu = nn.Parameter(mu)
            self.gamma = nn.Parameter(gamma)
        else:
            self.register_buffer('mu', mu)
            self.register_buffer('gamma', gamma)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Expand distances using RBF kernels.

        Args:
            distances: [num_edges] or [num_edges, 1]

        Returns:
            rbf_features: [num_edges, num_rbf]
        """
        if distances.dim() == 1:
            distances = distances.unsqueeze(-1)  # [num_edges, 1]

        # Compute RBF features: exp(-gamma * (d - mu)^2)
        # distances: [num_edges, 1], mu: [num_rbf] -> [num_edges, num_rbf]
        diff = distances - self.mu.view(1, -1)
        rbf = torch.exp(-self.gamma.view(1, -1) * diff ** 2)

        # Apply cutoff envelope
        cutoff_envelope = 0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1)
        cutoff_envelope = cutoff_envelope * (distances < self.cutoff).float()

        return rbf * cutoff_envelope


class AttentionPooling(nn.Module):
    """
    Attention-based global pooling.

    Instead of simple mean/max pooling, learns to weight important atoms
    more heavily. Useful when some atoms contribute more to properties.

    Args:
        hidden_dim: Dimension of node features
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool node features with learned attention.

        Args:
            x: Node features [num_nodes, hidden_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            graph_features: [batch_size, hidden_dim]
        """
        # Compute attention scores
        scores = self.attention_net(x)  # [num_nodes, 1]

        # Apply softmax per graph
        from torch_geometric.utils import softmax
        attention_weights = softmax(scores, batch, dim=0)  # [num_nodes, 1]

        # Weighted sum
        weighted_x = x * attention_weights
        graph_features = global_add_pool(weighted_x, batch)

        return graph_features


class GATMaterialsGNN(nn.Module):
    """
    Graph Attention Network for materials property prediction.

    Uses multi-head attention to learn which neighboring atoms are most
    important for predicting properties. More flexible than fixed aggregation.

    Key improvements over GraphSAGE:
    - Learnable attention weights for neighbors
    - Multi-head attention for diverse patterns
    - Better handling of varying coordination numbers

    Args:
        node_feature_dim: Input node feature dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of GAT layers
        num_heads: Number of attention heads per layer
        dropout: Dropout rate
        use_rbf: Use RBF expansion for edge features
        num_rbf: Number of RBF kernels
    """

    def __init__(
        self,
        node_feature_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_rbf: bool = True,
        num_rbf: int = 20,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_rbf = use_rbf

        # Node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # RBF expansion for edge features
        if use_rbf:
            self.rbf_expansion = RBFExpansion(num_rbf=num_rbf)
            self.edge_embedding = nn.Sequential(
                nn.Linear(num_rbf, hidden_dim),
                nn.ReLU(),
            )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            # GATv2Conv: improved version with more expressive attention
            conv = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=hidden_dim if use_rbf else None,
                concat=True,  # Concatenate heads
            )
            self.gat_layers.append(conv)
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        # Global pooling with attention
        self.pool = AttentionPooling(hidden_dim)

        # Task heads (same as before)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.bandgap_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure non-negative
        )

        self.stability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        logger.info(
            f"Initialized GATMaterialsGNN: {num_layers} layers, "
            f"{num_heads} heads, {hidden_dim} hidden dim"
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            edge_attr: Edge features [num_edges, edge_feature_dim]

        Returns:
            predictions: [batch_size, 3]
        """
        # Embed nodes
        h = self.node_embedding(x)

        # Embed edges if using RBF
        edge_features = None
        if self.use_rbf and edge_attr is not None:
            # Assume edge_attr is distances
            rbf_features = self.rbf_expansion(edge_attr)
            edge_features = self.edge_embedding(rbf_features)

        # GAT layers with residuals and layer norm
        for i, (conv, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            h_new = conv(h, edge_index, edge_attr=edge_features)
            h_new = norm(h_new)
            h_new = F.relu(h_new)

            # Residual connection
            if i > 0:
                h = h + h_new
            else:
                h = h_new

        # Global pooling
        graph_embedding = self.pool(h, batch)

        # Task predictions
        formation_energy = self.energy_head(graph_embedding)
        band_gap = self.bandgap_head(graph_embedding)
        stability = self.stability_head(graph_embedding)

        output = torch.cat([formation_energy, band_gap, stability], dim=1)
        return output


class EnsembleMaterialsGNN(nn.Module):
    """
    Ensemble of GNN models for better uncertainty quantification.

    Multiple models vote on predictions, providing:
    - Better accuracy through ensemble averaging
    - More reliable uncertainty estimates
    - Robustness to individual model failures

    Args:
        model_class: GNN model class to ensemble
        num_models: Number of models in ensemble
        model_kwargs: Arguments for each model
    """

    def __init__(
        self,
        model_class: type,
        num_models: int = 5,
        **model_kwargs
    ):
        super().__init__()

        self.num_models = num_models
        self.models = nn.ModuleList([
            model_class(**model_kwargs)
            for _ in range(num_models)
        ])

        logger.info(f"Initialized ensemble with {num_models} models")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through ensemble (averaging).

        For uncertainty, use predict_with_uncertainty instead.
        """
        predictions = []
        for model in self.models:
            pred = model(x, edge_index, batch, edge_attr)
            predictions.append(pred)

        # Average predictions
        predictions = torch.stack(predictions)  # [num_models, batch_size, 3]
        mean_pred = predictions.mean(dim=0)

        return mean_pred

    def predict_with_uncertainty(
        self,
        data: Data,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty using ensemble disagreement.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            mean: Mean predictions [batch_size, 3]
            std: Standard deviations [batch_size, 3]
        """
        self.eval()

        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model(
                    data.x,
                    data.edge_index,
                    data.batch if hasattr(data, 'batch') else torch.zeros(data.num_nodes, dtype=torch.long),
                    data.edge_attr if hasattr(data, 'edge_attr') else None,
                )
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)  # [num_models, batch_size, 3]

        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)

        return mean, std


def create_model(
    architecture: str = "gat",
    **kwargs
) -> nn.Module:
    """
    Factory function for creating GNN models.

    Args:
        architecture: Model architecture ("gat", "sage", "ensemble")
        **kwargs: Model-specific arguments

    Returns:
        Initialized model

    Example:
        >>> model = create_model("gat", hidden_dim=256, num_heads=8)
        >>> model = create_model("ensemble", model_class=GATMaterialsGNN, num_models=5)
    """
    if architecture == "gat":
        return GATMaterialsGNN(**kwargs)
    elif architecture == "sage":
        from .gnn_surrogate import GraphSAGESurrogate
        return GraphSAGESurrogate(**kwargs)
    elif architecture == "ensemble":
        model_class = kwargs.pop('model_class', GATMaterialsGNN)
        num_models = kwargs.pop('num_models', 5)
        return EnsembleMaterialsGNN(model_class, num_models, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


if __name__ == "__main__":
    # Test the enhanced models
    logging.basicConfig(level=logging.INFO)

    print("Testing Enhanced GNN Architectures\n")

    # Create sample data
    x = torch.randn(10, 7)  # 10 atoms, 7 features
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 2, 0],
    ])
    edge_attr = torch.rand(edge_index.size(1), 1) * 5  # Distances
    batch = torch.zeros(10, dtype=torch.long)

    # Test GAT model
    print("1. Testing GAT Model")
    gat_model = GATMaterialsGNN(
        node_feature_dim=7,
        hidden_dim=128,
        num_layers=3,
        num_heads=4,
        use_rbf=True,
    )

    gat_model.eval()
    with torch.no_grad():
        output = gat_model(x, edge_index, batch, edge_attr)

    print(f"   Output shape: {output.shape}")
    print(f"   Formation energy: {output[0, 0].item():.4f}")
    print(f"   Band gap: {output[0, 1].item():.4f}")
    print(f"   Stability: {output[0, 2].item():.4f}")

    # Test ensemble
    print("\n2. Testing Ensemble Model")
    ensemble = EnsembleMaterialsGNN(
        model_class=GATMaterialsGNN,
        num_models=3,
        node_feature_dim=7,
        hidden_dim=64,  # Smaller for faster test
        num_layers=2,
    )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    mean, std = ensemble.predict_with_uncertainty(data)

    print(f"   Mean predictions: {mean[0]}")
    print(f"   Std predictions: {std[0]}")
    print(f"   Uncertainty (formation energy): ±{std[0, 0]:.4f} eV/atom")

    print("\n✅ All tests passed!")
