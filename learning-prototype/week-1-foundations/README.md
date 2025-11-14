# Week 1: Python & ML Foundations ⚡

> **Master PyTorch fundamentals and set up your development environment**

## 🎯 Weekly Overview

This week you'll build a solid foundation in PyTorch and neural network basics. By the end, you'll understand tensors, automatic differentiation, and training loops - the building blocks for everything that follows.

**Why this matters:** Graph Neural Networks, LLM agents, and Bayesian optimization all build on these fundamentals. Master this week, and the rest becomes much easier.

## 📋 Learning Objectives

By the end of Week 1, you will:

- ✅ Set up a complete Python ML development environment
- ✅ Understand PyTorch tensors and operations
- ✅ Implement automatic differentiation (autograd)
- ✅ Build and train a simple neural network from scratch
- ✅ Understand forward/backward passes and gradient descent
- ✅ Use GPU acceleration effectively

## 🗓️ Daily Breakdown

### Day 1-2: Environment Setup & PyTorch Basics (4 hours)
- Install Python, PyTorch, and dependencies
- Learn tensor creation and manipulation
- Practice basic operations (reshape, slice, matmul)
- **Deliverable:** Passing `verify_setup.py` script

### Day 3-4: Automatic Differentiation (3 hours)
- Understand computational graphs
- Practice with `torch.autograd`
- Implement gradient computation manually
- **Deliverable:** Exercise 1 & 2 completed

### Day 5-6: Neural Networks (4 hours)
- Build a 2-layer network from scratch
- Implement forward and backward passes
- Understand loss functions and optimizers
- **Deliverable:** Exercise 3 & 4 completed

### Day 7: Training Loop & GPU (2 hours)
- Implement complete training loop
- Add validation and metrics
- Use GPU acceleration
- **Deliverable:** Trained model with >90% accuracy

## 📚 Study Materials

### Reading Order:
1. `notes/concepts.md` - Core concepts explained simply
2. `notes/math_explained.md` - Mathematical foundations
3. `notes/debugging_tips.md` - Common issues and solutions

### Video Resources:
- PyTorch Official Tutorials: "Introduction to PyTorch"
- 3Blue1Brown: "But what is a neural network?"
- Andrej Karpathy: "Building makemore Part 1"

## 🛠️ Exercises

Work through in order:

1. **Exercise 1: Tensor Operations** (45 min)
   - Create, reshape, and manipulate tensors
   - Practice matrix multiplication
   - Understand broadcasting

2. **Exercise 2: Automatic Differentiation** (45 min)
   - Compute gradients with autograd
   - Build computational graphs
   - Implement custom gradient functions

3. **Exercise 3: Simple Neural Network** (90 min)
   - Implement forward pass
   - Implement backward pass
   - Test on toy dataset

4. **Exercise 4: Training Loop** (90 min)
   - Complete training loop
   - Add early stopping
   - Visualize loss curves

## ✅ Success Criteria

You're ready for Week 2 when:

- ✅ All exercise tests pass
- ✅ You can explain what happens during forward/backward pass
- ✅ You can train a network to >90% accuracy on MNIST-style data
- ✅ You understand when/why to use GPU

## 🚀 Getting Started

```bash
# Navigate to week 1
cd week-1-foundations

# Read core concepts
cat notes/concepts.md

# Verify your setup
python scripts/verify_setup.py

# Start Exercise 1
cat exercises/exercise_1.md
python exercises/test_exercises.py --exercise 1
```

## 📊 Time Estimate

- **Minimum:** 10 hours (if you know Python basics)
- **Recommended:** 12 hours (includes reading and experimentation)
- **If new to ML:** 15 hours (take your time!)

## 🔗 Connections to Future Weeks

What you learn this week:

- **Tensors** → Used to represent graphs in Week 3
- **Autograd** → Powers GNN training in Week 3-4
- **Training loops** → Template for all model training
- **GPU usage** → Essential for large-scale experiments

## 💡 Tips for Success

1. **Type every line of code** - Don't copy/paste, you'll learn better
2. **Break when stuck** - Come back with fresh eyes
3. **Visualize everything** - Print shapes, plot tensors
4. **Test frequently** - Run tests after each TODO
5. **Ask "why"** - Understand the purpose of each line

## 📁 Files in This Folder

```
week-1-foundations/
├── README.md (you are here)
├── goals.md (detailed objectives)
├── starter_code/
│   ├── neural_network_template.py    # Fill-in-the-blanks NN
│   └── training_loop_template.py     # Training loop skeleton
├── exercises/
│   ├── exercise_1_tensors.md
│   ├── exercise_2_autograd.md
│   ├── exercise_3_neural_net.md
│   ├── exercise_4_training.md
│   └── test_exercises.py             # Automated tests
├── solutions/
│   ├── neural_network_solution.py
│   ├── training_loop_solution.py
│   └── explanations.md               # Why it works
├── notes/
│   ├── concepts.md                   # Core ML concepts
│   ├── math_explained.md             # Math derivations
│   └── debugging_tips.md             # Common issues
└── scripts/
    ├── verify_setup.py               # Check installation
    ├── generate_toy_data.py          # Create practice dataset
    └── visualize_training.py         # Plot results
```

## 🎯 Next Week Preview

Week 2 you'll build the data layer:
- Materials Project API integration
- Crystal structure representation
- Caching strategies
- Data validation

But first, master the foundations! 🚀

---

**Ready?** Start with `notes/concepts.md` →
