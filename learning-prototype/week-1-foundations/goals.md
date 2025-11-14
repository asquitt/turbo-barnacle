# Week 1: Detailed Learning Goals

## 🎯 Primary Objectives

### 1. Environment Setup (Mastery Level: 100%)

**What you'll learn:**
- Install Python 3.9+ with virtual environments
- Set up PyTorch with CUDA support (if GPU available)
- Configure Jupyter for interactive development
- Install visualization libraries (matplotlib, seaborn)

**Success criteria:**
```bash
python scripts/verify_setup.py
# Should output: ✅ All checks passed!
```

**Why it matters:**
A properly configured environment prevents 90% of future debugging headaches.

---

### 2. Tensor Operations (Mastery Level: 90%)

**What you'll learn:**
- Create tensors from Python lists, NumPy arrays, random values
- Reshape tensors while preserving data
- Slice and index tensors efficiently
- Perform matrix multiplication (`@`, `matmul`, `bmm`)
- Understand broadcasting rules

**Success criteria:**
- Implement 10 tensor operations without looking at docs
- Explain the difference between `view()` and `reshape()`
- Debug shape mismatch errors quickly

**Why it matters:**
Tensor manipulation is 50% of ML coding. Fluency here saves hours later.

---

### 3. Automatic Differentiation (Mastery Level: 85%)

**What you'll learn:**
- How PyTorch builds computational graphs
- Using `requires_grad=True` to track operations
- Computing gradients with `.backward()`
- Understanding gradient accumulation
- Zeroing gradients between batches

**Success criteria:**
- Compute gradients for arbitrary functions
- Explain why we call `optimizer.zero_grad()`
- Implement chain rule manually and verify with autograd

**Why it matters:**
Autograd is the "magic" that makes deep learning practical. Understanding it makes debugging 10× easier.

---

### 4. Neural Network Basics (Mastery Level: 80%)

**What you'll learn:**
- Linear layers (fully connected)
- Activation functions (ReLU, sigmoid, tanh)
- Forward pass implementation
- Backward pass mechanics
- Loss functions (MSE, CrossEntropy)

**Success criteria:**
- Build a 2-layer network from scratch (no `nn.Module`)
- Explain what happens at each layer during forward pass
- Trace gradient flow during backward pass

**Why it matters:**
Graph Neural Networks are just specialized neural networks. Understanding the basics is essential.

---

### 5. Training Loops (Mastery Level: 85%)

**What you'll learn:**
- Standard training loop structure
- Batch iteration with DataLoader
- Optimizer.step() mechanics
- Validation vs training mode
- Early stopping implementation

**Success criteria:**
- Write a complete training loop from memory
- Add learning rate scheduling
- Implement validation with proper `torch.no_grad()`

**Why it matters:**
You'll write 20+ training loops in this program. Master the pattern once, apply everywhere.

---

### 6. GPU Acceleration (Mastery Level: 70%)

**What you'll learn:**
- Moving tensors to GPU (`.to(device)`)
- Moving models to GPU
- Understanding when GPU is beneficial
- Debugging GPU out-of-memory errors

**Success criteria:**
- Train model on GPU 5-10× faster than CPU
- Know how to check GPU usage (`nvidia-smi`)
- Reduce batch size when OOM occurs

**Why it matters:**
GNN training without GPU is painfully slow. GPU usage is non-optional for Week 3+.

---

## 📊 Knowledge Checkpoints

Test your understanding with these questions:

### Checkpoint 1: Tensors
```python
# What is the output shape?
a = torch.randn(32, 10)
b = torch.randn(10, 5)
c = a @ b
# Answer: ?
```

### Checkpoint 2: Autograd
```python
# What happens here?
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1
y.backward()
print(x.grad)
# Answer: ?
```

### Checkpoint 3: Training
```python
# What's wrong with this code?
for epoch in range(100):
    for batch in dataloader:
        output = model(batch)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
# Answer: ?
```

**Answers in:** `solutions/checkpoint_answers.md`

---

## 🎓 Conceptual Understanding Goals

Beyond coding, you should understand:

### The Big Picture
- **Neural networks** approximate complex functions
- **Layers** learn hierarchical representations
- **Gradients** tell us how to improve weights
- **Backpropagation** efficiently computes gradients

### Key Insights
- More layers = more complex patterns (but harder to train)
- Activation functions = nonlinearity (without them, all layers collapse to one)
- Batch size = tradeoff between speed and gradient noise
- Learning rate = most important hyperparameter

### Common Misconceptions
- ❌ "More epochs = better model" → Can overfit!
- ❌ "Bigger network = better" → Can overfit, slower, more memory
- ❌ "GPU always faster" → Overhead for small models
- ❌ "Validation loss should always decrease" → Noisy, can fluctuate

---

## 🛠️ Practical Skills Goals

### By End of Week 1, You Can:

**Debugging Skills:**
- [ ] Fix shape mismatch errors in < 2 minutes
- [ ] Identify gradient flow issues
- [ ] Resolve GPU out-of-memory errors
- [ ] Debug training loops that don't converge

**Development Workflow:**
- [ ] Use Jupyter for experimentation
- [ ] Write reproducible Python scripts
- [ ] Visualize tensors and gradients
- [ ] Profile code to find bottlenecks

**Best Practices:**
- [ ] Set random seeds for reproducibility
- [ ] Separate training/validation data
- [ ] Use proper variable names
- [ ] Comment complex operations

---

## 📈 Progression Milestones

Track your progress:

### Day 1-2: Setup & Basics
- [ ] Environment verified
- [ ] Created first tensor
- [ ] Performed matrix multiplication
- [ ] Understood broadcasting

### Day 3-4: Gradients
- [ ] Computed first gradient
- [ ] Built computational graph
- [ ] Understood chain rule
- [ ] Implemented custom backward pass

### Day 5-6: Networks
- [ ] Built linear layer
- [ ] Added activation function
- [ ] Implemented forward pass
- [ ] Trained first model

### Day 7: Integration
- [ ] Complete training loop
- [ ] GPU acceleration working
- [ ] Validation implemented
- [ ] Results visualized

---

## 🎯 Mastery Indicators

You've mastered Week 1 when you can:

### Code Fluency
- Write a training loop without references in 15 minutes
- Debug common errors without Googling
- Implement new layer types from descriptions

### Conceptual Clarity
- Explain forward/backward pass to a friend
- Draw computational graph for any operation
- Predict tensor shapes without running code

### Problem Solving
- Diagnose why a model isn't learning
- Choose appropriate loss functions
- Tune learning rate effectively

---

## 🚀 Stretch Goals (Optional)

If you finish early or want extra challenge:

### Advanced Topics
- [ ] Implement custom autograd functions
- [ ] Write a learning rate finder
- [ ] Add gradient clipping
- [ ] Implement weight initialization schemes

### Optimization
- [ ] Profile training loop with PyTorch Profiler
- [ ] Optimize data loading with num_workers
- [ ] Use mixed precision training (AMP)
- [ ] Implement gradient accumulation

### Exploration
- [ ] Try different optimizers (SGD, Adam, AdamW)
- [ ] Experiment with learning rate schedules
- [ ] Visualize decision boundaries
- [ ] Compare CPU vs GPU performance

---

## 📚 Required Reading

Must read before Week 2:

1. **PyTorch Documentation:**
   - Tensor Tutorial
   - Autograd Mechanics
   - Neural Networks

2. **Concepts:**
   - `notes/concepts.md` (this folder)
   - `notes/math_explained.md` (this folder)

3. **Optional but Recommended:**
   - Neural Networks and Deep Learning (Chapter 1-2) - Michael Nielsen
   - Deep Learning Book (Chapter 6) - Goodfellow et al.

---

## ✅ Week 1 Completion Checklist

Before moving to Week 2, ensure:

- [ ] All 4 exercises completed
- [ ] All tests passing (`test_exercises.py`)
- [ ] Trained model achieves >90% accuracy
- [ ] Can explain forward/backward pass
- [ ] Comfortable with PyTorch documentation
- [ ] GPU working (if available)
- [ ] No unresolved questions about fundamentals

**Stuck on something?** Check `notes/debugging_tips.md`

**Ready for Week 2?** → You'll build the materials data pipeline!

---

## 💡 Tips from Past Learners

> "Don't rush. I spent 2 days just playing with tensors. Totally worth it." - Sarah, ML Engineer

> "The checkpoint questions are gold. They caught gaps in my understanding." - James, PhD Student

> "GPU was confusing at first. Just remember: model to device, data to device, done." - Wei, Data Scientist

> "I wish I'd spent more time on autograd. It comes up CONSTANTLY in Weeks 3-4." - Maria, Research Scientist

---

**Time to start learning!** → Begin with `notes/concepts.md`
