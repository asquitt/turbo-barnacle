# Core Concepts: PyTorch & Neural Network Fundamentals

> **Everything you need to understand before building Graph Neural Networks**

## Table of Contents
1. [What is PyTorch?](#what-is-pytorch)
2. [Tensors: The Foundation](#tensors-the-foundation)
3. [Automatic Differentiation](#automatic-differentiation)
4. [Neural Networks](#neural-networks)
5. [Training Loops](#training-loops)
6. [GPU Acceleration](#gpu-acceleration)

---

## What is PyTorch?

PyTorch is a Python library for building neural networks. Think of it as NumPy with three superpowers:

1. **Automatic differentiation** - Computes gradients for you
2. **GPU acceleration** - Runs on GPUs 10-100× faster
3. **Dynamic computation graphs** - Flexible, Pythonic workflow

### Why PyTorch for Materials Discovery?

- **Research-friendly:** Easy to implement new architectures (we'll implement 3+ GNN variants)
- **Graph support:** PyTorch Geometric builds on PyTorch for graph data
- **Deployment ready:** Can export to production
- **Community:** Huge ecosystem, lots of help available

---

## Tensors: The Foundation

### What is a Tensor?

A tensor is a multi-dimensional array. That's it!

```python
# Scalar (0-D tensor)
x = torch.tensor(5.0)

# Vector (1-D tensor)
v = torch.tensor([1.0, 2.0, 3.0])

# Matrix (2-D tensor)
m = torch.tensor([[1, 2], [3, 4]])

# 3-D tensor (like a cube of numbers)
cube = torch.randn(3, 4, 5)  # 3 × 4 × 5
```

### Why Tensors?

Neural networks are just matrix operations. Tensors let us represent:
- **Batch of data:** `[batch_size, features]`
- **Images:** `[batch, channels, height, width]`
- **Graphs:** `[num_nodes, node_features]`
- **Sequences:** `[sequence_length, batch, features]`

### Key Operations

#### Creation
```python
# From Python list
x = torch.tensor([1, 2, 3])

# Random values (most common in ML)
x = torch.randn(100, 10)  # Mean=0, Std=1

# Zeros/ones
x = torch.zeros(5, 5)
x = torch.ones(3, 3)

# Range
x = torch.arange(0, 10)  # [0, 1, 2, ..., 9]
```

#### Reshaping
```python
x = torch.randn(12)
y = x.view(3, 4)        # Reshape to 3×4
z = x.view(-1, 2)       # Reshape to ?×2 (auto-compute first dim)

# Must preserve total elements: 12 = 3*4 = 6*2
```

#### Slicing
```python
x = torch.randn(10, 5)
first_row = x[0]        # Shape: [5]
first_col = x[:, 0]     # Shape: [10]
top_left = x[:3, :3]    # Shape: [3, 3]
```

#### Matrix Multiplication
```python
# The most important operation in ML!

a = torch.randn(32, 10)  # Batch of 32, each with 10 features
b = torch.randn(10, 5)   # Weight matrix

c = a @ b                # Result: [32, 5]
# Or: c = torch.matmul(a, b)
# Or: c = a.mm(b)

# Rule: (m, n) @ (n, p) = (m, p)
#       Inner dimensions must match!
```

### Broadcasting

PyTorch automatically expands tensors for operations:

```python
x = torch.randn(32, 10)   # [32, 10]
mean = torch.randn(10)     # [10]

# This works! mean is broadcast to [32, 10]
centered = x - mean

# Broadcasting rules:
# 1. Align shapes from the right
# 2. Dimensions of size 1 can be expanded
# 3. Missing dimensions are assumed to be 1
```

**Why this matters:** You'll use broadcasting constantly for normalization, attention, etc.

---

## Automatic Differentiation

### The Problem

To train neural networks, we need gradients. Computing gradients by hand is:
- Error-prone
- Time-consuming
- Impossible for complex networks (1000s of parameters)

### The Solution: Autograd

PyTorch tracks operations and computes gradients automatically!

```python
# Track this tensor's operations
x = torch.tensor(3.0, requires_grad=True)

# Perform operations
y = x ** 2 + 2 * x + 1

# Compute gradient: dy/dx
y.backward()

# Get the gradient
print(x.grad)  # dy/dx = 2x + 2 = 2(3) + 2 = 8
```

### How It Works: Computational Graphs

PyTorch builds a graph of operations:

```
x = 3.0 (requires_grad=True)
  ↓
x² = 9.0
  ↓  +  ← 2x = 6.0
12.0
  ↓  +  ← 1
y = 13.0

During .backward():
dy/dy = 1                    (start)
dy/d(12.0) = 1              (+ doesn't change gradient)
dy/d(x²) = 1                (+ doesn't change gradient)
dy/dx (from x²) = 2x * 1 = 6
dy/d(2x) = 1
dy/dx (from 2x) = 2 * 1 = 2
Total: dy/dx = 6 + 2 = 8    (sum all paths)
```

### Key Points

**1. Only leaf tensors get gradients:**
```python
x = torch.tensor(3.0, requires_grad=True)  # Leaf
y = x * 2                                  # Not leaf (computed)
y.backward()
print(x.grad)  # ✓ Available
print(y.grad)  # ✗ None (not a leaf)
```

**2. Gradients accumulate:**
```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # 6

y = x ** 2
y.backward()
print(x.grad)  # 12 (6 + 6, accumulated!)

# Always zero gradients in training loops!
x.grad.zero_()
```

**3. Gradient flow can be stopped:**
```python
x = torch.tensor(3.0, requires_grad=True)

# Don't compute gradients here (saves memory/time)
with torch.no_grad():
    y = x ** 2

# Or detach from graph
y = (x ** 2).detach()
```

**Why this matters:** In Week 3, you'll train GNNs with thousands of parameters. Autograd makes it possible.

---

## Neural Networks

### What is a Neural Network?

A function approximator composed of:
1. **Layers** - Transform inputs
2. **Activations** - Add nonlinearity
3. **Parameters** - Learned weights

### The Simplest Network: Single Layer

```python
import torch.nn as nn

# Input: [batch, 10], Output: [batch, 5]
layer = nn.Linear(10, 5)

# What it does:
# output = input @ weights.T + bias
# [batch, 10] @ [10, 5] + [5] = [batch, 5]
```

### Multi-Layer Network

```python
class SimpleNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch, input_dim]
        x = self.layer1(x)        # [batch, hidden_dim]
        x = self.relu(x)          # [batch, hidden_dim] (nonlinearity!)
        x = self.layer2(x)        # [batch, output_dim]
        return x

# Create network
model = SimpleNetwork(input_dim=10, hidden_dim=20, output_dim=2)

# Use it
x = torch.randn(32, 10)  # Batch of 32
y = model(x)             # [32, 2]
```

### Why Nonlinearity? (Activation Functions)

Without activation functions:
```python
# Two linear layers
layer1: y = W1 @ x + b1
layer2: z = W2 @ y + b2

# Combine:
z = W2 @ (W1 @ x + b1) + b2
  = (W2 @ W1) @ x + (W2 @ b1 + b2)
  = W_combined @ x + b_combined

# Just one linear layer! Multiple layers are useless!
```

With activation functions (ReLU, sigmoid, etc.), networks can learn complex, nonlinear patterns.

### Common Activations

```python
# ReLU: max(0, x) - Most popular
relu = nn.ReLU()
# Kills negative values, keeps positive
# Fast, works well, simple

# Sigmoid: 1 / (1 + e^(-x))
sigmoid = nn.Sigmoid()
# Squashes to [0, 1]
# Used for probabilities

# Tanh: (e^x - e^(-x)) / (e^x + e^(-x))
tanh = nn.Tanh()
# Squashes to [-1, 1]
# Centered around 0
```

**In this project:** We'll use ReLU for GNNs, LeakyReLU for attention mechanisms.

---

## Training Loops

### The Standard Pattern

Every training loop follows this structure:

```python
# Pseudocode
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch in train_dataloader:
        # 1. Forward pass
        predictions = model(batch.x)
        loss = loss_function(predictions, batch.y)

        # 2. Backward pass
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update weights

    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_dataloader:
            predictions = model(batch.x)
            val_loss = loss_function(predictions, batch.y)
```

### Key Components Explained

#### 1. Loss Function
Measures how wrong the model is.

```python
# For regression (predicting numbers)
criterion = nn.MSELoss()  # Mean Squared Error

# For classification (predicting categories)
criterion = nn.CrossEntropyLoss()

# For multi-task (we'll use this!)
class MultiTaskLoss(nn.Module):
    def forward(self, pred_energy, true_energy, pred_gap, true_gap):
        energy_loss = F.mse_loss(pred_energy, true_energy)
        gap_loss = F.mse_loss(pred_gap, true_gap)
        return energy_loss + gap_loss
```

#### 2. Optimizer
Updates weights to minimize loss.

```python
# Adam: Adaptive learning rates, momentum
# Most popular, great default choice
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# SGD: Simple, requires careful tuning
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# AdamW: Adam + weight decay (we'll use this)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

#### 3. Learning Rate
Most important hyperparameter!

```python
# Too high: Training unstable, diverges
# Too low: Training very slow, gets stuck

# Good defaults:
# - Adam: 0.001
# - SGD: 0.01

# Learning rate scheduling (we'll add in Week 3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)
```

#### 4. Batch Processing

```python
from torch.utils.data import DataLoader, TensorDataset

# Create dataset
X = torch.randn(1000, 10)  # 1000 samples
y = torch.randn(1000, 1)
dataset = TensorDataset(X, y)

# Create batches
dataloader = DataLoader(
    dataset,
    batch_size=32,      # Process 32 samples at a time
    shuffle=True,       # Randomize order (important!)
    num_workers=4       # Parallel data loading
)

# Use in training
for batch_X, batch_y in dataloader:
    # batch_X: [32, 10]
    # batch_y: [32, 1]
    predictions = model(batch_X)
    loss = criterion(predictions, batch_y)
    # ... backward pass ...
```

**Why batching?**
- **Efficiency:** GPU processes batches in parallel
- **Generalization:** Noisy gradients help avoid overfitting
- **Memory:** Can't fit 1M samples in GPU memory at once

### Early Stopping

Prevent overfitting by stopping when validation loss stops improving:

```python
best_val_loss = float('inf')
patience = 20
patience_counter = 0

for epoch in range(1000):  # Large number
    train_loss = train_epoch()
    val_loss = validate()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## GPU Acceleration

### Why GPU?

CPUs: 4-16 cores, designed for sequential tasks
GPUs: 1000s of cores, designed for parallel operations

Matrix multiplication = lots of parallel operations = perfect for GPU!

**Speedup for ML:** 5-100× faster training

### Using GPU in PyTorch

```python
# Check if GPU available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# Move model to device (do this once)
model = model.to(device)

# Move data to device (do this for each batch)
for batch_x, batch_y in dataloader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    predictions = model(batch_x)  # Runs on GPU!
    loss = criterion(predictions, batch_y)
    # ...
```

### GPU Best Practices

**1. Move model once, data every batch:**
```python
model = model.to(device)  # Once at start

for batch in dataloader:
    batch = batch.to(device)  # Every batch
    # ...
```

**2. Use appropriate batch size:**
```python
# Too small: GPU underutilized
# Too large: Out of memory

# Good starting points:
# - Images: 32-128
# - Graphs: 16-64
# - Text: 16-32
```

**3. Handle out-of-memory:**
```python
try:
    predictions = model(batch)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("OOM! Reduce batch size or model size")
        torch.cuda.empty_cache()  # Clear cache
    raise e
```

**4. Monitor GPU usage:**
```bash
# In terminal, run:
watch -n 0.5 nvidia-smi

# Look for:
# - GPU utilization: Should be high (>80%) during training
# - Memory usage: Should be steady, not growing
```

---

## Putting It All Together

Here's a complete example combining everything:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Create simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# 2. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 3. Create toy dataset
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. Training loop
for epoch in range(100):
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        # Move to GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
```

---

## Next Steps

Now that you understand the concepts:

1. **Try it yourself:** Work through `exercises/exercise_1.md`
2. **Fill in templates:** Complete `starter_code/neural_network_template.py`
3. **Test understanding:** Run `test_exercises.py`
4. **Read math:** Check `notes/math_explained.md` for derivations

**Remember:** Understanding > memorization. Focus on WHY, not just WHAT.

---

## Quick Reference

### Tensor Shapes
```python
x.shape          # Get shape
x.view(...)      # Reshape
x.unsqueeze(0)   # Add dimension
x.squeeze(0)     # Remove dimension
```

### Common Operations
```python
x @ y            # Matrix multiply
x * y            # Element-wise multiply
x + y            # Element-wise add
x.sum()          # Sum all elements
x.mean(dim=0)    # Mean along dimension
```

### Training Essentials
```python
optimizer.zero_grad()  # Clear gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update weights
```

### GPU
```python
x.to(device)           # Move tensor
model.to(device)       # Move model
x.cpu()                # Move to CPU
torch.cuda.empty_cache()  # Clear GPU memory
```

---

**Ready to code?** → Start `exercises/exercise_1.md`
