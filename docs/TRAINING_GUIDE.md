# Training Guide: GNN Surrogate Model

This guide walks you through training the Graph Neural Network surrogate model that predicts material properties.

## Prerequisites

- Python 3.9+
- CUDA-compatible GPU (optional, but recommended for faster training)
- ~10GB disk space for data
- Materials Project API key (optional, can use mock data)

## Quick Start

```bash
# 1. Set up environment
python scripts/setup.py --mode local

# 2. Download and prepare data (or use mock mode)
python scripts/prepare_data.py --num-materials 10000

# 3. Train the model
python scripts/train_surrogate.py --config configs/model_config.yaml

# 4. Evaluate the model
python scripts/evaluate_model.py --model-path models/checkpoints/best_model.pt
```

## Detailed Steps

### 1. Data Preparation

#### Option A: Use Real Materials Project Data

```bash
# Set API key
export MP_API_KEY="your_key_here"

# Download materials data
python -c "
from src.data.materials_api import MaterialsAPIClient

client = MaterialsAPIClient()
materials = client.fetch_materials(
    num_elements=(2, 4),  # Binary to quaternary
    is_stable=True,
    max_results=10000,
)

# Export structures
client.export_structures(materials, 'data/raw/structures')
"
```

#### Option B: Use Mock Data (No API Key)

```bash
# Generate mock dataset
python scripts/generate_mock_data.py --num-samples 10000 --output data/mock
```

### 2. Preprocess Data

Convert crystal structures to graphs:

```python
from src.data.preprocessor import CrystalGraphConverter
from src.data.materials_api import MaterialsAPIClient

# Load data
client = MaterialsAPIClient(use_cache=True)
materials = client.fetch_materials(max_results=1000)

# Convert to graphs
converter = CrystalGraphConverter(bond_strategy="radius_cutoff")
graphs = converter.batch_convert(
    structures=[m.structure for m in materials],
    formation_energies=[m.formation_energy_per_atom for m in materials],
    band_gaps=[m.band_gap for m in materials],
    is_stable_list=[m.is_stable for m in materials],
)

# Save processed graphs
torch.save(graphs, 'data/processed/training_graphs.pt')
```

### 3. Configure Training

Edit `configs/model_config.yaml`:

```yaml
model:
  architecture: graphsage
  hidden_dim: 128  # Increase for better accuracy (more memory)
  num_layers: 3    # More layers = more capacity
  dropout: 0.1

training:
  learning_rate: 0.001
  num_epochs: 200
  batch_size: 32  # Adjust based on GPU memory
  early_stopping:
    patience: 20

hardware:
  device: auto  # Will use CUDA if available
  mixed_precision: true  # 2× faster on modern GPUs
```

### 4. Train the Model

#### Basic Training

```bash
python scripts/train_surrogate.py \
    --config configs/model_config.yaml \
    --data data/processed/training_graphs.pt \
    --output models/checkpoints
```

#### With MLflow Tracking

```bash
# Start MLflow server
mlflow ui --port 5000 &

# Train with tracking
python scripts/train_surrogate.py \
    --config configs/model_config.yaml \
    --data data/processed/training_graphs.pt \
    --use-mlflow
```

#### With Weights & Biases

```bash
# Login to W&B
wandb login

# Train with W&B tracking
python scripts/train_surrogate.py \
    --config configs/model_config.yaml \
    --data data/processed/training_graphs.pt \
    --use-wandb \
    --wandb-project materials-discovery
```

### 5. Monitor Training

#### Real-time Metrics

Watch the training logs:

```bash
tail -f logs/training.log
```

#### MLflow UI

```bash
# Access at http://localhost:5000
mlflow ui --port 5000
```

You'll see:
- Loss curves (training & validation)
- Learning rate schedule
- Model parameters
- System metrics (GPU usage, memory)

#### Weights & Biases

```bash
# Access at https://wandb.ai/<your-username>/<project>
wandb sync
```

### 6. Evaluate Model

```python
from src.models.gnn_surrogate import GraphSAGESurrogate
from src.models.trainer import GNNTrainer
import torch

# Load model
model = GraphSAGESurrogate(...)
checkpoint = torch.load('models/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Load test data
test_loader = ...

# Evaluate
model.eval()
test_metrics = trainer.validate()

print(f"Test MAE (formation energy): {test_metrics['val_energy_mae']:.4f} eV/atom")
print(f"Test MAE (band gap): {test_metrics['val_bandgap_mae']:.4f} eV")
print(f"Test accuracy (stability): {test_metrics['val_stability_acc']:.4f}")
```

## Training Tips

### Improving Accuracy

1. **More Data**: 10K+ materials → better generalization
2. **Larger Model**: Increase `hidden_dim` to 256 or 512
3. **More Layers**: Try 4-5 graph convolution layers
4. **Better Features**: Add more atomic properties
5. **Data Augmentation**: Random rotations, perturbations

### Reducing Training Time

1. **Mixed Precision**: Enable `mixed_precision: true`
2. **Larger Batches**: Increase `batch_size` (if GPU allows)
3. **Fewer Epochs**: Use early stopping aggressively
4. **Gradient Accumulation**: Simulate larger batches
5. **Transfer Learning**: Start from pre-trained model

### Memory Issues

If you run out of GPU memory:

```yaml
training:
  batch_size: 16  # Reduce batch size
  accumulation_steps: 2  # Accumulate gradients

model:
  hidden_dim: 64  # Smaller model
```

Or use gradient checkpointing:

```python
model = GraphSAGESurrogate(..., use_checkpointing=True)
```

### Debugging Training

#### Loss not decreasing?

- Check learning rate (try 1e-4 to 1e-3)
- Verify data preprocessing
- Check for NaN values
- Reduce model complexity
- Increase batch size

#### Overfitting?

- Increase dropout (0.2-0.3)
- Add more data
- Reduce model size
- Enable data augmentation
- Use early stopping

#### Unstable training?

- Reduce learning rate
- Enable gradient clipping
- Check for outliers in data
- Use batch normalization

## Hyperparameter Tuning

### Manual Tuning

Test different configurations:

```bash
# Configuration 1: Small fast model
python scripts/train_surrogate.py --config configs/model_small.yaml

# Configuration 2: Large accurate model
python scripts/train_surrogate.py --config configs/model_large.yaml

# Configuration 3: Custom
python scripts/train_surrogate.py \
    --hidden-dim 256 \
    --num-layers 4 \
    --lr 0.0005
```

### Automated Tuning with Ray Tune

```python
from ray import tune
from src.models.trainer import GNNTrainer

def train_fn(config):
    model = GraphSAGESurrogate(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )

    trainer = GNNTrainer(model, ...)
    history = trainer.train(num_epochs=50)

    return {
        "val_loss": history['val'][-1]['val_loss'],
        "val_energy_mae": history['val'][-1]['val_energy_mae'],
    }

# Define search space
config = {
    "hidden_dim": tune.choice([64, 128, 256]),
    "num_layers": tune.choice([2, 3, 4]),
    "dropout": tune.uniform(0.1, 0.3),
    "learning_rate": tune.loguniform(1e-4, 1e-2),
}

# Run hyperparameter search
analysis = tune.run(
    train_fn,
    config=config,
    num_samples=20,
    metric="val_loss",
    mode="min",
)

print(f"Best config: {analysis.best_config}")
```

## Model Checkpoints

Training automatically saves:

- `best_model.pt`: Best model by validation loss
- `checkpoint_epoch_N.pt`: Periodic snapshots
- `final_model.pt`: Final epoch model

Each checkpoint contains:

```python
checkpoint = {
    'epoch': epoch_number,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_validation_loss,
}
```

## Transfer Learning

Start from a pre-trained model:

```python
# Load pre-trained model
pretrained = torch.load('models/pretrained/mp_pretrained.pt')
model.load_state_dict(pretrained['model_state_dict'])

# Fine-tune on your data
trainer = GNNTrainer(
    model,
    train_loader,
    val_loader,
    learning_rate=1e-4,  # Lower LR for fine-tuning
)

history = trainer.train(num_epochs=50)
```

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
batch_size = 16

# Or enable gradient accumulation
accumulation_steps = 2

# Or use CPU
device = "cpu"
```

### Slow Training

```bash
# Use mixed precision (requires CUDA)
use_amp: true

# Use more workers
num_workers: 8

# Profile to find bottlenecks
python -m torch.utils.bottleneck scripts/train_surrogate.py
```

### Poor Performance on Test Set

1. Check data distribution (train vs test)
2. Visualize predictions vs ground truth
3. Analyze failure cases
4. Consider ensemble of models
5. Add more diverse training data

## Next Steps

- [Evaluate model performance](evaluation.md)
- [Deploy model to production](deployment.md)
- [Integrate with discovery loop](integration.md)
- [Contribute improvements](CONTRIBUTING.md)

## Resources

- [PyTorch Geometric Tutorial](https://pytorch-geometric.readthedocs.io/)
- [GraphSAGE Paper](https://arxiv.org/abs/1706.02216)
- [Materials Project API](https://materialsproject.org/api)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
