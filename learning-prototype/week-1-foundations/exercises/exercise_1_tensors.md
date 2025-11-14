# Exercise 1: Tensor Operations

**Estimated Time:** 45 minutes
**Difficulty:** Beginner
**Prerequisites:** Basic Python, understanding of arrays/matrices

## 🎯 Learning Objectives

By completing this exercise, you will:
- Create tensors using various methods
- Reshape and manipulate tensor dimensions
- Perform matrix multiplication
- Understand and apply broadcasting
- Debug shape mismatch errors

---

## Part 1: Tensor Creation (10 minutes)

### Task 1.1: Create tensors from different sources

Create the following tensors:

```python
import torch

# TODO 1: Create a 1D tensor from a Python list
# List: [1, 2, 3, 4, 5]
tensor_from_list = ??? # YOUR CODE HERE

# TODO 2: Create a 2D tensor (matrix) filled with zeros
# Shape: 3 rows × 4 columns
tensor_zeros = ??? # YOUR CODE HERE

# TODO 3: Create a 2D tensor with random values from normal distribution
# Shape: 5 rows × 10 columns
# Hint: Use torch.randn()
tensor_random = ??? # YOUR CODE HERE

# TODO 4: Create a tensor with values from 0 to 9
# Hint: Use torch.arange()
tensor_range = ??? # YOUR CODE HERE
```

**Expected Shapes:**
- tensor_from_list: `[5]`
- tensor_zeros: `[3, 4]`
- tensor_random: `[5, 10]`
- tensor_range: `[10]`

**Test Your Code:**
```python
print(f"tensor_from_list shape: {tensor_from_list.shape}")
print(f"tensor_zeros shape: {tensor_zeros.shape}")
print(f"tensor_random shape: {tensor_random.shape}")
print(f"tensor_range: {tensor_range}")
```

---

## Part 2: Reshaping (10 minutes)

### Task 2.1: Reshape tensors

```python
# Start with a 1D tensor
x = torch.arange(12)  # [0, 1, 2, ..., 11]

# TODO 5: Reshape to 3 rows × 4 columns
# Hint: Use .view() or .reshape()
x_reshaped_3x4 = ??? # YOUR CODE HERE

# TODO 6: Reshape to 2 rows × 6 columns
x_reshaped_2x6 = ??? # YOUR CODE HERE

# TODO 7: Reshape to 4 rows × 3 columns
x_reshaped_4x3 = ??? # YOUR CODE HERE

# TODO 8: Reshape to 3D tensor (2 × 2 × 3)
x_reshaped_3d = ??? # YOUR CODE HERE
```

**Test:**
```python
assert x_reshaped_3x4.shape == (3, 4)
assert x_reshaped_2x6.shape == (2, 6)
assert x_reshaped_4x3.shape == (4, 3)
assert x_reshaped_3d.shape == (2, 2, 3)
print("✓ All reshape tests passed!")
```

### Task 2.2: Add and remove dimensions

```python
x = torch.randn(5, 10)

# TODO 9: Add a dimension at position 0
# Result shape should be: [1, 5, 10]
# Hint: Use .unsqueeze(dim)
x_expanded = ??? # YOUR CODE HERE

# TODO 10: Remove the dimension you just added
# Result shape should be back to: [5, 10]
# Hint: Use .squeeze(dim)
x_squeezed = ??? # YOUR CODE HERE
```

**Test:**
```python
assert x_expanded.shape == (1, 5, 10)
assert x_squeezed.shape == (5, 10)
print("✓ Dimension manipulation tests passed!")
```

---

## Part 3: Matrix Multiplication (15 minutes)

### Task 3.1: Basic matrix multiplication

```python
# TODO 11: Multiply two matrices
# A: 32 × 10 (batch of 32 samples, each with 10 features)
# B: 10 × 5  (weight matrix)
# Result should be: 32 × 5

A = torch.randn(32, 10)
B = torch.randn(10, 5)

# Perform matrix multiplication
# Hint: Use @ operator or torch.matmul()
C = ??? # YOUR CODE HERE

# What's the shape?
print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")
print(f"C shape: {C.shape}")
```

**Expected:** `C.shape = [32, 5]`

### Task 3.2: Common pitfalls

```python
# This will ERROR! Can you fix it?
X = torch.randn(32, 10)
W = torch.randn(20, 5)  # Wrong dimension!

# TODO 12: What should the first dimension of W be?
# Redefine W with the correct shape
W = ??? # YOUR CODE HERE

result = X @ W  # Should work now!
```

**Question:** Why did the original multiplication fail?
**Answer:** Write your explanation here:
```
[YOUR ANSWER]
```

---

## Part 4: Broadcasting (10 minutes)

Broadcasting allows operations on tensors of different shapes.

### Task 4.1: Understand broadcasting

```python
# Batch normalization example
X = torch.randn(32, 10)  # 32 samples, 10 features each

# TODO 13: Compute mean of each feature (across samples)
# Result should have shape [10]
# Hint: Use .mean(dim=0)
mean = ??? # YOUR CODE HERE

# TODO 14: Compute std of each feature
# Hint: Use .std(dim=0)
std = ??? # YOUR CODE HERE

# TODO 15: Normalize X by subtracting mean and dividing by std
# This will broadcast! mean and std are [10], X is [32, 10]
X_normalized = ??? # YOUR CODE HERE

# Verify normalization worked
print(f"Original X mean: {X.mean(dim=0)[:3]}")  # Should be ~0 after normalization
print(f"Normalized X mean: {X_normalized.mean(dim=0)[:3]}")  # Should be ~0
```

### Task 4.2: Broadcasting practice

```python
# TODO 16: Add a vector to each row of a matrix
matrix = torch.ones(5, 10)
vector = torch.arange(10).float()

result = ??? # YOUR CODE HERE (matrix + vector)

# Check: First row should be [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"First row: {result[0]}")
```

---

## Part 5: Practical Application (10 minutes)

### Task 5.1: Implement a linear layer manually

A linear layer computes: `output = input @ weights.T + bias`

```python
# Simulate a mini-batch
batch_size = 32
input_dim = 10
output_dim = 5

# TODO 17: Create input tensor (batch of samples)
X = ??? # torch.randn(???, ???)

# TODO 18: Create weight matrix
# Hint: Weights shape should be [output_dim, input_dim]
W = ??? # torch.randn(???, ???)

# TODO 19: Create bias vector
# Hint: Bias shape should be [output_dim]
b = ??? # torch.randn(???)

# TODO 20: Compute output
# Formula: output = X @ W.T + b
# Hint: Use .T to transpose W
output = ??? # YOUR CODE HERE

print(f"Input shape: {X.shape}")
print(f"Weight shape: {W.shape}")
print(f"Bias shape: {b.shape}")
print(f"Output shape: {output.shape}")
```

**Expected output shape:** `[32, 5]`

**Test:**
```python
assert output.shape == (32, 5), f"Expected [32, 5], got {output.shape}"
print("✓ Linear layer implementation correct!")
```

---

## 🎯 Challenge Problems (Optional)

### Challenge 1: Batch Matrix Multiplication

```python
# You have a batch of matrices
# Each sample is a 3×3 matrix, and you have 10 samples

batch_matrices = torch.randn(10, 3, 3)

# TODO: Multiply each 3×3 matrix by itself
# Result should be 10 matrices of shape 3×3
# Hint: Use torch.bmm() for batch matrix multiplication

result = ??? # YOUR CODE HERE
assert result.shape == (10, 3, 3)
```

### Challenge 2: Implement Batch Normalization

```python
# TODO: Implement batch normalization from scratch
# 1. Compute mean and variance across batch dimension
# 2. Normalize: (X - mean) / sqrt(variance + epsilon)
# 3. Scale and shift: gamma * X_normalized + beta

def batch_norm(X, gamma, beta, epsilon=1e-5):
    """
    X: [batch_size, features]
    gamma: [features]  (learnable scale)
    beta: [features]   (learnable shift)
    """
    # YOUR CODE HERE
    pass

# Test
X = torch.randn(32, 10)
gamma = torch.ones(10)
beta = torch.zeros(10)
X_normalized = batch_norm(X, gamma, beta)

# Verify
assert X_normalized.shape == X.shape
assert abs(X_normalized.mean().item()) < 0.1  # Should be ~0
```

---

## ✅ Completion Checklist

Before moving on, make sure you can:

- [ ] Create tensors using at least 3 different methods
- [ ] Reshape tensors while preserving total elements
- [ ] Perform matrix multiplication and predict output shapes
- [ ] Understand why shape mismatches occur
- [ ] Apply broadcasting for operations on different shapes
- [ ] Implement a simple linear layer manually

---

## 🔍 Self-Check Questions

1. **What's the difference between `.view()` and `.reshape()`?**
   ```
   [YOUR ANSWER]
   ```

2. **Why does matrix multiplication require inner dimensions to match?**
   ```
   [YOUR ANSWER]
   ```

3. **When does broadcasting work, and when does it fail?**
   ```
   [YOUR ANSWER]
   ```

4. **How many parameters does a linear layer have?**
   ```
   [YOUR ANSWER]
   ```

---

## 📚 Additional Resources

- PyTorch Documentation: [Tensor Operations](https://pytorch.org/docs/stable/tensors.html)
- 3Blue1Brown: [Linear Transformations](https://www.youtube.com/watch?v=kYB8IZa5AuE)
- NumPy Broadcasting Tutorial (same concepts): [Link](https://numpy.org/doc/stable/user/basics.broadcasting.html)

---

## 🎓 Next Steps

Once you've completed this exercise:

1. Run the tests in `test_exercises.py`
2. Compare your solutions with `solutions/exercise_1_solution.py`
3. Move on to **Exercise 2: Automatic Differentiation**

**Questions?** Check `notes/debugging_tips.md` or review `notes/concepts.md`

---

**Time to practice!** Start coding and test frequently! 🚀
