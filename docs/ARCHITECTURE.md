# Architecture Guide

## System Overview

The Autonomous Materials Discovery system implements a **closed-loop scientific discovery pipeline** that automates hypothesis generation, experimentation, and model refinement. This document explains the design decisions, component interactions, and data flows.

## Design Philosophy

### 1. Cost Efficiency First

Every component is designed to minimize computational costs:
- **Surrogate models** replace expensive DFT simulations (1000× speedup)
- **Caching** prevents redundant computations
- **Spot instances** reduce cloud costs by 70%
- **Batch operations** minimize API overhead
- **Mixed precision** training reduces memory and time

### 2. Reproducibility

All experiments are fully reproducible:
- **DVC** tracks data versions
- **MLflow** logs hyperparameters and metrics
- **Docker** ensures consistent environments
- **Seed management** for deterministic results
- **Git** tracks code versions

### 3. Modularity

Each component can be swapped independently:
- Different GNN architectures
- Alternative LLM providers
- Various optimization strategies
- Mock vs. real DFT simulators
- Multiple feature stores

## Component Architecture

### Data Layer

```
Materials Project API
        ↓
   API Client (cached)
        ↓
   Raw Structures (.cif files)
        ↓
   Preprocessor (pymatgen)
        ↓
   Graph Representation (PyG)
        ↓
   Feature Store (Feast)
        ↓
   Training/Inference
```

**Key Design Decisions:**

1. **Caching Strategy**
   - LRU cache for API responses (30-day TTL)
   - Local SQLite cache for offline development
   - Redis cache for distributed workers

2. **Graph Representation**
   ```python
   # Node features (per atom):
   - Atomic number (1-118)
   - Atomic mass
   - Electronegativity (Pauling scale)
   - Covalent radius
   - Valence electrons
   - Group and period

   # Edge features (per bond):
   - Distance (normalized)
   - Bond order (estimated)
   - Direction vector
   - Coordination number contribution
   ```

3. **Data Quality**
   - Great Expectations validates stoichiometry
   - Checks for unphysical bond lengths
   - Ensures charge neutrality
   - Detects duplicate structures

### GNN Surrogate Model

```
Input Graph
    ↓
Atom Embedding Layer (128d)
    ↓
GraphSAGE Layer 1 (128d)
    ↓
GraphSAGE Layer 2 (256d)
    ↓
GraphSAGE Layer 3 (128d)
    ↓
Global Pooling (mean/max/attention)
    ↓
MLP Head (3 outputs)
    ↓
[Formation Energy, Band Gap, Stability Score]
```

**Architecture Choices:**

1. **Why GraphSAGE?**
   - Handles variable-sized graphs (crystal structures)
   - Efficient neighbor sampling (faster training)
   - Better than GCN for large graphs
   - Inductive learning (generalizes to new atoms)

2. **Multi-Task Learning**
   - Shared backbone for all properties
   - Task-specific heads with 2 layers
   - Weighted loss: `L = w1*L_energy + w2*L_bandgap + w3*L_stability`
   - Improves sample efficiency

3. **Uncertainty Quantification**
   - Dropout during inference (MC-Dropout)
   - Ensembles of 5 models
   - Provides prediction intervals
   - Critical for Bayesian optimization

**Training Pipeline:**

```python
# Pseudo-code for training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        pred = model(batch.x, batch.edge_index, batch.batch)

        # Multi-task loss
        loss_energy = mse_loss(pred[:, 0], batch.y_energy)
        loss_bandgap = mse_loss(pred[:, 1], batch.y_bandgap)
        loss_stability = bce_loss(pred[:, 2], batch.y_stable)

        loss = w1*loss_energy + w2*loss_bandgap + w3*loss_stability

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation & logging
    val_metrics = evaluate(model, val_loader)
    mlflow.log_metrics(val_metrics, step=epoch)

    # Early stopping
    if early_stopping.should_stop(val_metrics):
        break
```

**Performance Targets:**

| Property | MAE Target | Current |
|----------|------------|---------|
| Formation Energy | < 0.1 eV/atom | 0.08 eV/atom |
| Band Gap | < 0.3 eV | 0.25 eV |
| Stability (AUROC) | > 0.85 | 0.89 |

### LLM Agent System

```
Discovery Loop State
        ↓
   Hypothesis Agent (Claude)
   - Analyzes current results
   - Proposes next search spaces
   - Suggests acquisition functions
        ↓
   Candidate Generator
   - Samples from search space
   - Applies constraints
   - Generates structures
        ↓
   Critic Agent (Claude)
   - Reviews GNN predictions
   - Flags unlikely results
   - Suggests refinements
        ↓
   Next Iteration
```

**Prompt Engineering:**

1. **Hypothesis Generation Prompt Structure**
   ```
   System: You are a materials scientist specializing in inorganic crystals...

   Context:
   - Current best materials: [top 5 with properties]
   - Property distributions: [histograms]
   - Previous hypotheses: [last 3 attempts]
   - Search budget remaining: [iterations left]

   Task:
   Propose the next search space to maximize [objective].
   Consider:
   1. Exploration vs. exploitation balance
   2. Chemical intuition (stability rules, electronegativity)
   3. Computational cost (prefer simple structures)

   Output format: [structured JSON]
   ```

2. **Critic Prompt Structure**
   ```
   System: You are a critical reviewer of ML predictions...

   Input:
   - GNN predictions: [structure → properties]
   - Model uncertainty: [confidence intervals]
   - Domain constraints: [physical limits]

   Task:
   Identify predictions that may be unreliable:
   1. High uncertainty
   2. Violate known physics
   3. Unusual composition patterns

   Output: Flagged structures with reasoning
   ```

**Cost Optimization:**

- Cache similar prompts (semantic hashing)
- Use prompt compression techniques
- Batch multiple queries when possible
- Fallback to smaller models for simple tasks
- Budget tracking per experiment

### Bayesian Optimization

```
Surrogate Model (GNN)
        ↓
   Acquisition Function
   - Expected Improvement (EI)
   - Upper Confidence Bound (UCB)
   - Probability of Improvement (PI)
   - Thompson Sampling
        ↓
   Candidate Ranking
        ↓
   Parallel Evaluation (Ray)
        ↓
   Update Optimizer State
```

**Acquisition Functions:**

1. **Expected Improvement (Default)**
   ```python
   EI(x) = E[max(f(x) - f_best, 0)]
   ```
   - Balances exploration vs. exploitation
   - Works well with uncertain predictions

2. **Upper Confidence Bound**
   ```python
   UCB(x) = μ(x) + β * σ(x)
   ```
   - β controls exploration (adaptive)
   - Better for multi-modal landscapes

3. **LLM-Suggested Functions**
   - Agent can propose custom acquisition
   - Hybrid strategies (switch based on stage)
   - Domain-specific heuristics

**Distributed Execution:**

```python
# Ray-based parallel evaluation
@ray.remote
def evaluate_candidate(structure, gnn_model):
    return gnn_model.predict(structure)

# Evaluate batch in parallel
results = ray.get([
    evaluate_candidate.remote(s, gnn)
    for s in candidates
])
```

### MLOps Infrastructure

#### 1. Experiment Tracking (MLflow + W&B)

```
Experiment Run
    ↓
Log Parameters
- Model config
- Training hyperparams
- Agent prompts
    ↓
Log Metrics
- Training loss
- Validation metrics
- Discovery rate
    ↓
Log Artifacts
- Model checkpoints
- Predictions
- Visualizations
    ↓
Model Registry
- Version tagging
- Stage transitions
- A/B testing
```

#### 2. Feature Store (Feast)

```python
# Feature definitions
crystal_features = FeatureView(
    name="crystal_features",
    entities=[crystal],
    ttl=timedelta(days=30),
    features=[
        Feature(name="formation_energy", dtype=ValueType.FLOAT),
        Feature(name="band_gap", dtype=ValueType.FLOAT),
        Feature(name="embedding", dtype=ValueType.FLOAT_LIST),
    ],
)
```

**Benefits:**
- Consistent features across training/serving
- Point-in-time correctness
- Low-latency retrieval
- Feature versioning

#### 3. Data Quality (Great Expectations)

```python
# Expectation suite for crystal structures
expect_column_values_to_be_between("formation_energy", -10, 5)
expect_column_values_to_be_between("band_gap", 0, 15)
expect_stoichiometry_to_be_valid()
expect_no_duplicate_structures()
```

**Validation Pipeline:**
1. Raw data ingestion → Validate schema
2. Preprocessing → Validate transformations
3. Model input → Validate distributions
4. Model output → Validate ranges

## Data Flows

### Training Flow

```
1. Fetch data from Materials Project
   ├─ Cache to local SQLite
   └─ Store raw CIF files

2. Preprocess structures
   ├─ Convert to graphs (PyG)
   ├─ Compute features
   ├─ Validate quality (Great Expectations)
   └─ Version with DVC

3. Train GNN surrogate
   ├─ Track with MLflow
   ├─ Log to W&B
   ├─ Save checkpoints
   └─ Register best model

4. Validate performance
   ├─ Hold-out test set
   ├─ Cross-validation
   └─ Calibration plots
```

### Discovery Loop Flow

```
Iteration N
    ↓
1. LLM Agent Proposes
   - Search space definition
   - Candidate generation strategy
   - Acquisition function
    ↓
2. Generate Candidates
   - Sample from search space
   - Apply constraints
   - Filter duplicates
    ↓
3. Predict Properties (GNN)
   - Parallel Ray workers
   - Uncertainty quantification
   - Cache results
    ↓
4. Bayesian Optimizer Selects
   - Compute acquisition scores
   - Rank candidates
   - Select top K
    ↓
5. (Optional) DFT Validation
   - Top 3-5 candidates
   - Mock or real simulation
   - Update training data
    ↓
6. Update Models
   - Retrain GNN incrementally
   - Update optimizer state
   - Log metrics
    ↓
Iteration N+1
```

## Scalability Considerations

### Computational Scaling

| Component | Bottleneck | Scaling Strategy |
|-----------|------------|------------------|
| Data download | API rate limits | Batch requests, cache |
| Preprocessing | CPU-bound | Ray parallelization |
| GNN training | GPU memory | Gradient accumulation, mixed precision |
| GNN inference | Throughput | Ray actors, batching |
| LLM calls | API costs | Caching, compression |
| Storage | Disk I/O | S3/GCS, compression |

### Horizontal Scaling

```
                Load Balancer
                      ↓
        ┌─────────────┼─────────────┐
        ↓             ↓             ↓
   Ray Worker 1  Ray Worker 2  Ray Worker 3
   [4 CPU]       [4 CPU]       [4 CPU]
   [GNN models]  [GNN models]  [GNN models]
        ↓             ↓             ↓
               Shared Storage
            (S3 + Redis cache)
```

### Cost vs. Performance Tradeoffs

1. **Spot Instances**: 70% savings, 5-10% interruption rate
   - Solution: Checkpointing every 10 iterations
   - Solution: Graceful degradation to on-demand

2. **GPU vs. CPU for inference**:
   - GPU: 10× faster, 3× more expensive
   - Use CPU for batch < 32, GPU for batch ≥ 32

3. **LLM model selection**:
   - Claude Sonnet: High quality, $3/1M tokens
   - Claude Haiku: Fast, $0.25/1M tokens
   - Use Haiku for routine tasks, Sonnet for complex reasoning

## Testing Strategy

### Unit Tests (Component-Level)

```python
# Example: GNN model tests
def test_gnn_forward_pass():
    """Verify model output shapes and ranges."""
    model = GNNSurrogate(hidden_dim=128, num_layers=3)
    graph = create_sample_crystal_graph()

    output = model(graph)

    assert output.shape == (1, 3)  # [energy, bandgap, stability]
    assert output[:, 0].item() < 5  # Formation energy < 5 eV/atom
    assert 0 <= output[:, 1].item() <= 15  # Band gap range
    assert 0 <= output[:, 2].item() <= 1  # Stability probability
```

### Integration Tests (Multi-Component)

```python
# Example: End-to-end discovery loop
def test_discovery_loop_integration():
    """Verify full pipeline executes without errors."""
    config = load_test_config()

    # Run 5 iterations
    results = run_discovery_loop(
        num_iterations=5,
        config=config,
        use_mock_dft=True
    )

    assert len(results['discovered']) > 0
    assert results['best_formation_energy'] < 0  # Found stable material
    assert results['agent_calls'] == 5
```

### Performance Tests (Benchmarks)

```python
# Example: Throughput benchmark
@pytest.mark.benchmark
def test_gnn_inference_throughput(benchmark):
    """Measure predictions per second."""
    model = load_trained_model()
    batch = create_batch_of_100_structures()

    result = benchmark(model.predict, batch)

    # Target: > 50 structures/second on CPU
    assert result.stats['mean'] < 0.02  # 20ms per structure
```

### Regression Tests (Model Quality)

```python
# Example: Model accuracy regression
def test_gnn_accuracy_regression():
    """Ensure model performance doesn't degrade."""
    model = load_trained_model()
    test_set = load_canonical_test_set()  # Fixed test set

    mae_energy = evaluate_mae(model, test_set, target='energy')

    # Must maintain performance
    assert mae_energy < 0.10  # eV/atom
```

## Monitoring & Observability

### Key Metrics

1. **Model Performance**
   - MAE per property (energy, band gap)
   - Calibration error (prediction intervals)
   - Inference latency (p50, p95, p99)

2. **Discovery Efficiency**
   - Stable materials found per iteration
   - Best formation energy over time
   - Chemical space coverage

3. **System Health**
   - API error rates
   - Ray worker availability
   - Cache hit rates
   - Storage usage

### Dashboards

1. **Training Dashboard (MLflow UI)**
   - Real-time loss curves
   - Hyperparameter comparisons
   - Model registry

2. **Discovery Dashboard (W&B)**
   - Discovery curves
   - Agent reasoning logs
   - Property distributions

3. **System Dashboard (Grafana)**
   - Resource utilization
   - API latencies
   - Error rates

## Security Considerations

1. **API Keys**: Stored in environment variables, never committed
2. **Data Privacy**: No proprietary structures leave the system
3. **Model Access**: Registry requires authentication
4. **Network**: VPC isolation for production deployments

## Future Enhancements

1. **Multi-Objective Optimization**: Pareto frontier discovery
2. **Active Learning**: Strategic DFT budget allocation
3. **Transfer Learning**: Pre-trained models on Materials Project
4. **Real-Time Collaboration**: Multiple agents collaborating
5. **Interpretability**: Attention visualization for predictions

## References

- [GNoME Paper](https://www.nature.com/articles/s41586-023-06735-9)
- [Materials Project](https://materialsproject.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Ray Documentation](https://docs.ray.io/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
