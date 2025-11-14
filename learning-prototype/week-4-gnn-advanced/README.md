# Week 4: Advanced GNN Architectures 🚀

> **Implement cutting-edge enhancements for 30-100% better performance**

## 🎯 Overview

Take your basic GNN from Week 3 and supercharge it with research-backed enhancements: Graph Attention Networks, Radial Basis Functions, and ensemble uncertainty quantification.

## 📋 Learning Objectives

- ✅ Implement Graph Attention Networks (GAT, GATv2)
- ✅ Add Radial Basis Function (RBF) edge features
- ✅ Build attention pooling mechanisms
- ✅ Create ensemble models for uncertainty
- ✅ Achieve 30% accuracy improvement over Week 3 baseline

## 🗓️ Time Estimate

12-15 hours over 7 days

## 🧠 Key Enhancements

### 1. Graph Attention Networks (GAT)
**Paper:** Veličković et al., ICLR 2018

Instead of equal neighbor weighting (GraphSAGE), learn attention weights:
```python
# GraphSAGE: Mean of neighbors
h_v = mean([h_u for u in neighbors(v)])

# GAT: Weighted mean with learned attention
attention = softmax([score(h_v, h_u) for u in neighbors(v)])
h_v = sum([attention[i] * h_u for i, u in enumerate(neighbors(v))])
```

**Why it helps:** Materials have different bond types - attention learns which bonds matter most!

### 2. Radial Basis Functions (RBF)
**Paper:** SchNet (Schütt et al., NeurIPS 2017)

Encode distances with Gaussian RBF kernels:
```python
# Instead of: edge_attr = [distance]
# Use: edge_attr = RBF(distance)  # [num_rbf_features]

def rbf(distance, centers, gamma):
    return exp(-gamma * (distance - centers)^2)
```

**Why it helps:** Continuous distance encoding captures bond physics better!

### 3. Attention Pooling
Learn which atoms are most important for global properties:
```python
# Instead of: graph_feat = mean(node_features)
# Use weighted attention pooling
attention_weights = softmax(learned_score(node_features))
graph_feat = sum(attention_weights * node_features)
```

### 4. Ensemble Uncertainty
Train 5 models, use disagreement as uncertainty:
```python
predictions = [model_i(x) for model_i in ensemble]
mean = np.mean(predictions)
uncertainty = np.std(predictions)  # High std = uncertain!
```

**Why it helps:** Active learning (Week 7) needs uncertainty to select experiments!

## 🛠️ Exercises

1. **Multi-Head Attention** (3 hours)
   - Implement attention mechanism
   - Add multi-head attention
   - Integrate into GNN

2. **RBF Edge Features** (2 hours)
   - Gaussian RBF expansion
   - Learnable vs fixed centers
   - Integrate with GAT

3. **Attention Pooling** (2 hours)
   - Learn node importance
   - Global pooling with attention
   - Compare with mean pooling

4. **Ensemble Training** (3 hours)
   - Train 5 diverse models
   - Combine predictions
   - Calibrate uncertainty

5. **Benchmark & Ablation** (2 hours)
   - Compare all variants
   - Ablation study
   - Document improvements

## ✅ Success Criteria

- [ ] GAT implemented with multi-head attention
- [ ] RBF edge features working
- [ ] Ensemble uncertainty calibrated
- [ ] Improvements over Week 3:
  - Formation energy MAE: 30% better
  - Band gap MAE: 20% better
  - Uncertainty calibration: AUROC > 0.80
- [ ] All tests passing

## 📊 Expected Performance

| Metric | Week 3 Baseline | Week 4 Target |
|--------|----------------|---------------|
| Formation Energy MAE | 0.115 eV | < 0.081 eV (-30%) |
| Band Gap MAE | 0.312 eV | < 0.249 eV (-20%) |
| Stability AUROC | 0.846 | > 0.892 (+5%) |

## 📚 Papers to Read

**Required:**
1. **Graph Attention Networks** (Veličković et al., ICLR 2018)
2. **SchNet** (Schütt et al., NeurIPS 2017) - Section on RBF

**Optional:**
3. **GATv2** (Brody et al., ICLR 2022) - Improved attention
4. **Deep Ensembles** (Lakshminarayanan et al., NeurIPS 2017)

## 🚀 Getting Started

```bash
cd week-4-gnn-advanced

# Review Week 3 GNN first!
cat notes/week3_review.md

# Read about attention
cat notes/attention_mechanisms.md

# Start Exercise 1
cat exercises/exercise_1_gat.md
```

## 💡 Pro Tips

- **Test attention weights:** Visualize which atoms get high attention
- **Compare architectures:** Run ablation study (GraphSAGE vs GAT vs Ensemble)
- **Monitor overfitting:** Ensemble helps but can overfit
- **GPU memory:** Attention uses more memory than GraphSAGE

**Next:** [Week 5 - LLM Agents](../week-5-llm-agents/README.md)
