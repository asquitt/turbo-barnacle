# Week 6: Bayesian Optimization 📊

> **Intelligent search through vast chemical spaces**

## 🎯 Overview

Implement Bayesian Optimization to intelligently select which materials to explore next. Move beyond random search to achieve 40% faster convergence.

## 📋 Learning Objectives

- ✅ Understand Gaussian Processes fundamentals
- ✅ Implement 4 acquisition functions (EI, UCB, PI, Thompson Sampling)
- ✅ Build batch selection with diversity promotion
- ✅ Implement adaptive exploration-exploitation
- ✅ Achieve 40% faster convergence than random search

## 🗓️ Time Estimate

12-14 hours over 7 days

## 🧠 Key Concepts

### Bayesian Optimization Loop

```
1. Train surrogate model (GNN) on observed data
2. Use surrogate to predict + uncertainty for candidates
3. Acquisition function scores candidates
4. Select highest-scoring candidate
5. Validate experimentally (DFT)
6. Add to dataset, repeat
```

### Acquisition Functions

**1. Expected Improvement (EI)**
```python
improvement = max(0, mean - best_so_far)
EI = E[improvement] = improvement * Φ(Z) + std * φ(Z)
```
**When to use:** Balanced exploration-exploitation

**2. Upper Confidence Bound (UCB)**
```python
UCB = mean + beta * std
```
**When to use:** More exploration (beta > 2)

**3. Probability of Improvement (PI)**
```python
PI = P(f(x) > best_so_far + xi)
```
**When to use:** Conservative, exploitation-focused

**4. Thompson Sampling**
```python
sample ~ N(mean, std²)
select argmax(sample)
```
**When to use:** Stochastic exploration

### Batch Selection

Select diverse batch to parallelize:
```python
def select_batch(candidates, acquisition_scores, diversity_weight=0.2):
    selected = []
    for i in range(batch_size):
        # Balance acquisition + diversity
        scores = acquisition_scores - diversity_weight * similarity_to_selected
        selected.append(argmax(scores))
    return selected
```

## 🛠️ Exercises

1. **Gaussian Process Basics** (3 hours)
   - Implement GP mean/variance
   - Understand kernel functions
   - Test on toy data

2. **Acquisition Functions** (3 hours)
   - Implement EI, UCB, PI, Thompson
   - Compare on 1D function
   - Visualize acquisition landscapes

3. **Batch Selection** (2 hours)
   - Implement diversity promotion
   - Test different diversity weights
   - Visualize selected batches

4. **Full BO Loop** (3 hours)
   - Integrate with GNN from Week 4
   - Run on materials data
   - Compare with random search

5. **Adaptive Strategies** (1 hour)
   - Adaptive beta for UCB
   - Switching acquisition functions
   - Early termination

## ✅ Success Criteria

- [ ] All 4 acquisition functions implemented
- [ ] Batch diversity working
- [ ] 40% faster convergence than random
- [ ] Adaptive beta improves results
- [ ] All tests passing

## 📊 Expected Performance

| Strategy | Iterations to Best | Materials Discovered |
|----------|-------------------|---------------------|
| Random Search | 85 | 20 per 100 iter |
| EI | 62 (-27%) | 28 per 100 iter |
| UCB | 55 (-35%) | 32 per 100 iter |
| Thompson | 52 (-39%) | 34 per 100 iter |

## 📚 Resources

**Book:** *Bayesian Optimization* - Roman Garnett
**Tutorial:** Distill.pub - "A Visual Exploration of Gaussian Processes"

## 🚀 Getting Started

```bash
cd week-6-optimization

cat notes/bayesian_optimization.md
cat notes/acquisition_functions.md

python exercises/exercise_1_gaussian_process.py
```

## 💡 Pro Tips

- **Visualize in 1D first:** Easier to understand
- **Test on toy functions:** Rastrigin, Rosenbrock
- **Monitor exploitation vs exploration:** Track statistics
- **Diverse batches matter:** 20% better with diversity

**Next:** [Week 7 - Active Learning](../week-7-active-learning/README.md)
