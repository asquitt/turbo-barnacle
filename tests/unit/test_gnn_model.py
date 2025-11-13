"""
Unit Tests for GNN Surrogate Model

Tests cover:
- Model initialization
- Forward pass shapes
- Prediction ranges
- Uncertainty quantification
- Loss computation

Run with: pytest tests/unit/test_gnn_model.py -v
"""

import pytest
import torch
from torch_geometric.data import Data

import sys
sys.path.append('src')

from models.gnn_surrogate import GraphSAGESurrogate, MultiTaskLoss, count_parameters


@pytest.fixture
def sample_graph():
    """Create a sample crystal graph for testing."""
    # Simple graph: 3 nodes (atoms), 6 edges (bonds)
    x = torch.randn(3, 7)  # 3 nodes, 7 features each
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 0],  # Source nodes
        [1, 0, 2, 1, 0, 2],  # Target nodes
    ], dtype=torch.long)
    edge_attr = torch.randn(6, 1)  # 6 edges, 1 feature each
    y = torch.tensor([[
        -2.5,  # Formation energy
        1.2,   # Band gap
        1.0,   # Stable
    ]])

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


@pytest.fixture
def model():
    """Create a GNN model for testing."""
    return GraphSAGESurrogate(
        node_feature_dim=7,
        hidden_dim=64,  # Smaller for faster tests
        num_layers=2,
        dropout=0.1,
        pooling="mean",
    )


class TestModelInitialization:
    """Test model initialization and configuration."""

    def test_model_creation(self, model):
        """Test that model is created successfully."""
        assert model is not None
        assert isinstance(model, GraphSAGESurrogate)

    def test_parameter_count(self, model):
        """Test that model has reasonable number of parameters."""
        num_params = count_parameters(model)
        assert num_params > 0
        assert num_params < 10_000_000  # Less than 10M params (sanity check)

    def test_different_pooling_strategies(self):
        """Test different pooling strategies."""
        for pooling in ["mean", "max", "add"]:
            model = GraphSAGESurrogate(
                node_feature_dim=7,
                hidden_dim=32,
                num_layers=2,
                pooling=pooling,
            )
            assert model.pooling == pooling


class TestForwardPass:
    """Test model forward pass."""

    def test_output_shape(self, model, sample_graph):
        """Test that output has correct shape."""
        model.eval()
        batch = torch.zeros(sample_graph.num_nodes, dtype=torch.long)

        with torch.no_grad():
            output = model(
                sample_graph.x,
                sample_graph.edge_index,
                batch,
                sample_graph.edge_attr,
            )

        # Should output [batch_size=1, num_tasks=3]
        assert output.shape == (1, 3)

    def test_output_ranges(self, model, sample_graph):
        """Test that outputs are in reasonable ranges."""
        model.eval()
        batch = torch.zeros(sample_graph.num_nodes, dtype=torch.long)

        with torch.no_grad():
            output = model(
                sample_graph.x,
                sample_graph.edge_index,
                batch,
                sample_graph.edge_attr,
            )

        # Formation energy: should be reasonable (-10 to +5 eV/atom)
        assert -15 < output[0, 0].item() < 10

        # Band gap: should be non-negative
        assert output[0, 1].item() >= 0
        assert output[0, 1].item() < 20  # Reasonable upper bound

        # Stability: should be probability [0, 1]
        assert 0 <= output[0, 2].item() <= 1

    def test_batch_processing(self, model):
        """Test that model can handle batches of graphs."""
        # Create batch of 3 graphs
        graphs = []
        for _ in range(3):
            x = torch.randn(4, 7)  # 4 nodes each
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
            graphs.append(Data(x=x, edge_index=edge_index))

        from torch_geometric.data import Batch
        batch_data = Batch.from_data_list(graphs)

        model.eval()
        with torch.no_grad():
            output = model(
                batch_data.x,
                batch_data.edge_index,
                batch_data.batch,
            )

        # Should output [batch_size=3, num_tasks=3]
        assert output.shape == (3, 3)

    def test_gradient_flow(self, model, sample_graph):
        """Test that gradients flow through the model."""
        model.train()
        batch = torch.zeros(sample_graph.num_nodes, dtype=torch.long)

        output = model(
            sample_graph.x,
            sample_graph.edge_index,
            batch,
            sample_graph.edge_attr,
        )

        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestUncertaintyQuantification:
    """Test uncertainty quantification with MC-Dropout."""

    def test_mc_dropout(self, model, sample_graph):
        """Test that MC-Dropout returns mean and std."""
        mean, std = model.predict_with_uncertainty(
            sample_graph,
            num_samples=5,
        )

        # Should return arrays with shape [1, 3]
        assert mean.shape == (1, 3)
        assert std.shape == (1, 3)

        # Std should be non-negative
        assert (std >= 0).all()

    def test_uncertainty_consistency(self, model, sample_graph):
        """Test that uncertainty is consistent across calls."""
        mean1, std1 = model.predict_with_uncertainty(sample_graph, num_samples=10)
        mean2, std2 = model.predict_with_uncertainty(sample_graph, num_samples=10)

        # Means should be similar (within 20%)
        assert torch.allclose(
            torch.tensor(mean1),
            torch.tensor(mean2),
            rtol=0.2,
        )


class TestLossFunction:
    """Test multi-task loss function."""

    def test_loss_computation(self, sample_graph):
        """Test that loss is computed correctly."""
        loss_fn = MultiTaskLoss(
            energy_weight=1.0,
            bandgap_weight=0.5,
            stability_weight=2.0,
        )

        predictions = torch.tensor([[
            -2.0,  # Formation energy
            1.5,   # Band gap
            0.8,   # Stability
        ]])

        targets = sample_graph.y

        loss, loss_dict = loss_fn(predictions, targets)

        # Loss should be a scalar
        assert loss.ndim == 0
        assert loss.item() > 0

        # Loss dict should have all components
        assert 'energy_loss' in loss_dict
        assert 'bandgap_loss' in loss_dict
        assert 'stability_loss' in loss_dict
        assert 'total_loss' in loss_dict

    def test_perfect_prediction(self):
        """Test that perfect predictions give near-zero loss."""
        loss_fn = MultiTaskLoss()

        predictions = torch.tensor([[
            -2.5,  # Formation energy
            1.2,   # Band gap
            1.0,   # Stability
        ]])

        targets = predictions.clone()

        loss, _ = loss_fn(predictions, targets)

        # Should be very close to zero
        assert loss.item() < 0.01

    def test_loss_weights(self):
        """Test that loss weights affect total loss."""
        predictions = torch.tensor([[0.0, 0.0, 0.0]])
        targets = torch.tensor([[1.0, 1.0, 1.0]])

        # High energy weight
        loss_fn1 = MultiTaskLoss(energy_weight=10.0, bandgap_weight=0.1, stability_weight=0.1)
        loss1, _ = loss_fn1(predictions, targets)

        # High stability weight
        loss_fn2 = MultiTaskLoss(energy_weight=0.1, bandgap_weight=0.1, stability_weight=10.0)
        loss2, _ = loss_fn2(predictions, targets)

        # Losses should be different
        assert abs(loss1.item() - loss2.item()) > 0.1


@pytest.mark.slow
class TestModelPerformance:
    """Performance benchmarks (marked as slow)."""

    def test_inference_speed(self, model, benchmark):
        """Benchmark inference speed."""
        graph = Data(
            x=torch.randn(10, 7),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.randn(20, 1),
        )
        batch = torch.zeros(10, dtype=torch.long)

        model.eval()

        def inference():
            with torch.no_grad():
                return model(graph.x, graph.edge_index, batch, graph.edge_attr)

        result = benchmark(inference)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_acceleration(self, model):
        """Test that GPU provides speedup."""
        import time

        graph = Data(
            x=torch.randn(100, 7),
            edge_index=torch.randint(0, 100, (2, 400)),
        )
        batch = torch.zeros(100, dtype=torch.long)

        # CPU time
        model_cpu = model.cpu()
        model_cpu.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model_cpu(graph.x, graph.edge_index, batch)
        cpu_time = time.time() - start

        # GPU time
        model_gpu = model.cuda()
        graph_gpu = Data(
            x=graph.x.cuda(),
            edge_index=graph.edge_index.cuda(),
        )
        batch_gpu = batch.cuda()

        model_gpu.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = model_gpu(graph_gpu.x, graph_gpu.edge_index, batch_gpu)
        gpu_time = time.time() - start

        # GPU should be faster (or at least not much slower)
        assert gpu_time < cpu_time * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
