# Week 7: Active Learning 🎯

> **Reduce expensive validations by 54% through intelligent sampling**

## 🎯 Overview

Implement active learning strategies to minimize costly DFT validations. Achieve 54% reduction in experiments needed while maintaining discovery quality.

## 📋 Learning Objectives

- ✅ Implement uncertainty sampling
- ✅ Build query-by-committee
- ✅ Add diversity sampling
- ✅ Create hybrid active learning strategy
- ✅ Manage DFT budget effectively
- ✅ Achieve 54% reduction in validations

## 🗓️ Time Estimate

10-12 hours over 7 days

## 🧠 Key Strategies

### 1. Uncertainty Sampling
Query samples where model is most uncertain:
```python
uncertainties = model.predict_uncertainty(candidates)
selected = top_k(uncertainties)  # Highest uncertainty
```

**Variants:**
- **Least confident:** min(max(probabilities))
- **Margin sampling:** difference between top 2 classes
- **Entropy:** H(p) = -Σ p log(p)

### 2. Query-by-Committee
Use ensemble disagreement:
```python
predictions = [model_i(x) for model_i in committee]
disagreement = std(predictions)  # High = uncertain
selected = top_k(disagreement)
```

### 3. Diversity Sampling
Maximize coverage of chemical space:
```python
# K-means clustering
clusters = kmeans(candidates, k=num_samples)
selected = [closest_to_centroid(c) for c in clusters]

# Or max-distance
selected = [candidates[0]]  # Seed
for i in range(num_samples - 1):
    distances = [min_distance_to_selected(c) for c in candidates]
    selected.append(argmax(distances))
```

### 4. Hybrid Strategy
Combine all three:
```python
score = (
    0.5 * uncertainty_score +
    0.3 * diversity_score +
    0.2 * committee_score
)
selected = top_k(score)
```

## 🛠️ Exercises

1. **Uncertainty Sampling** (2 hours)
   - Implement 3 uncertainty metrics
   - Compare on toy dataset
   - Visualize selected samples

2. **Query-by-Committee** (2 hours)
   - Use ensemble from Week 4
   - Compute disagreement
   - Test committee size (3 vs 5 vs 7)

3. **Diversity Sampling** (2 hours)
   - K-means selection
   - Max-distance selection
   - Compare coverage

4. **Hybrid Strategy** (2 hours)
   - Combine strategies
   - Tune weights
   - Ablation study

5. **DFT Budget Management** (2 hours)
   - Set validation budget
   - Adaptive sampling rate
   - Early stopping

## ✅ Success Criteria

- [ ] All 4 strategies implemented
- [ ] Hybrid outperforms individual strategies
- [ ] 54% reduction in DFT validations
- [ ] Discovery quality maintained
- [ ] Budget management working
- [ ] All tests passing

## 📊 Expected Performance

| Strategy | DFT Validations Needed | Discovery Rate |
|----------|----------------------|----------------|
| Random | 100% (baseline) | 34 per 100 |
| Uncertainty Only | 65% (-35%) | 32 per 100 |
| Diversity Only | 70% (-30%) | 30 per 100 |
| Hybrid | 46% (-54%) | 34 per 100 ✓ |

**Key insight:** Hybrid maintains discovery rate while cutting validations in half!

## 💰 Cost Impact

DFT validation: ~$20 per material
With 54% reduction:
- Before: 100 validations = $2,000
- After: 46 validations = $920
- **Savings: $1,080 (54%)**

## 📚 Papers

1. **Active Learning Literature Survey** (Settles, 2009)
2. **Deep Batch Active Learning** (Ash et al., NeurIPS 2020)

## 🚀 Getting Started

```bash
cd week-7-active-learning

cat notes/active_learning.md
cat notes/sampling_strategies.md

python exercises/exercise_1_uncertainty.py
```

## 💡 Pro Tips

- **Start with uncertainty:** Usually most effective
- **Add diversity:** Prevents sampling duplicates
- **Visualize selections:** Plot in chemical space
- **Track coverage:** Monitor chemical space coverage
- **Adaptive weights:** Adjust based on performance

**Next:** [Week 8 - System Integration](../week-8-integration/README.md)
