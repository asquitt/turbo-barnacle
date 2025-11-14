# Week 3: Graph Neural Networks Basics 🕸️

> **The most important week - Build your first GNN for materials!**

## 🎯 Weekly Overview

This is where things get exciting! You'll learn how to represent crystal structures as graphs and build Graph Neural Networks that can predict material properties. This is the core technology behind our materials discovery system.

**Why this matters:** GNNs replace expensive physics simulations (DFT) that cost $20-100 per material with ML predictions that cost $0.0001. This 100,000× speedup makes discovery possible.

## 📋 Learning Objectives

By the end of Week 3, you will:

- ✅ Represent crystal structures as graphs
- ✅ Understand message passing neural networks
- ✅ Implement GraphSAGE architecture
- ✅ Build multi-task GNN (predict energy + band gap + stability)
- ✅ Train GNN on Materials Project data
- ✅ Achieve MAE < 0.15 eV on formation energy

## 🗓️ Daily Breakdown

### Day 1-2: Graph Fundamentals (5 hours)
- Understand graph representation of crystals
- Learn about node features (atomic properties)
- Learn about edge features (bonds, distances)
- Build crystal-to-graph converter
- **Deliverable:** Exercise 1 completed

### Day 3-4: Message Passing & GraphSAGE (6 hours)
- Understand message passing framework
- Implement GraphSAGE aggregation
- Build GNN layers from scratch
- **Deliverable:** Exercise 2 & 3 completed

### Day 5-6: Multi-Task Learning (5 hours)
- Combine multiple prediction tasks
- Implement multi-task loss
- Train on real materials data
- **Deliverable:** Trained model, Exercise 4 completed

### Day 7: Evaluation & Analysis (2 hours)
- Evaluate model performance
- Analyze predictions vs ground truth
- Understand failure cases
- **Deliverable:** Performance analysis complete

## 🧠 Core Concepts

### What is a Graph?

A graph is:
- **Nodes** (vertices): Atoms in a crystal
- **Edges**: Bonds or spatial relationships between atoms
- **Features**: Properties of nodes/edges

Example for a water molecule (H₂O):
```
Nodes:
  - Node 0: Oxygen (atomic_num=8, radius=0.73Å)
  - Node 1: Hydrogen (atomic_num=1, radius=0.37Å)
  - Node 2: Hydrogen (atomic_num=1, radius=0.37Å)

Edges:
  - Edge 0-1: O-H bond (distance=0.96Å)
  - Edge 0-2: O-H bond (distance=0.96Å)
```

### Message Passing

The core GNN operation in 3 steps:

1. **Message Creation:** Each neighbor creates a message
2. **Aggregation:** Combine messages from all neighbors
3. **Update:** Update node features with aggregated messages

```python
for each layer:
    for each node v:
        # 1. Collect messages from neighbors
        messages = [transform(features[u]) for u in neighbors(v)]

        # 2. Aggregate (e.g., mean)
        aggregated = mean(messages)

        # 3. Update node features
        features[v] = combine(features[v], aggregated)
```

### GraphSAGE Architecture

GraphSAGE (Graph SAmple and aggreGatE):
- Samples a fixed number of neighbors
- Aggregates using mean pooling
- Concatenates self features with aggregated neighbors
- Applies a learned transformation

**Why GraphSAGE?** Efficient, scalable, works well for materials.

## 🛠️ Exercises

### Exercise 1: Crystal to Graph (2 hours)
**File:** `exercises/exercise_1_crystal_to_graph.md`

Convert pymatgen Structure to PyTorch Geometric Data:
- Extract atomic features (number, mass, radius, etc.)
- Find nearest neighbors (bonds)
- Compute edge features (distances)
- Create PyG Data object

### Exercise 2: Message Passing Layer (2 hours)
**File:** `exercises/exercise_2_message_passing.md`

Implement a single message passing layer:
- Neighbor aggregation
- Self-loop handling
- Learned transformations

### Exercise 3: GraphSAGE Model (3 hours)
**File:** `exercises/exercise_3_graphsage.md`

Build complete GraphSAGE model:
- Stack multiple layers
- Add residual connections
- Implement global pooling
- Multi-task prediction heads

### Exercise 4: Training GNN (3 hours)
**File:** `exercises/exercise_4_train_gnn.md`

Train on materials data:
- Load Materials Project data
- Implement multi-task loss
- Train with early stopping
- Evaluate and visualize results

## 📊 Starter Code

### `starter_code/graph_converter_template.py`
Template for converting crystals to graphs with TODOs

### `starter_code/gnn_layer_template.py`
GraphSAGE layer implementation skeleton

### `starter_code/gnn_model_template.py`
Complete GNN model with multi-task heads

### `starter_code/train_gnn_template.py`
Training loop for GNN

## ✅ Success Criteria

You're ready for Week 4 when:

- ✅ Can convert any crystal to a graph
- ✅ Understand message passing intuitively
- ✅ Implemented GraphSAGE from scratch
- ✅ Trained model achieves:
  - Formation energy MAE < 0.15 eV/atom
  - Band gap MAE < 0.40 eV
  - Stability AUROC > 0.80
- ✅ Can explain how GNN differs from regular NN
- ✅ All tests passing

## 🎓 Key Papers to Read

**Required:**
1. **GraphSAGE** (Hamilton et al., NeurIPS 2017)
   - Read Sections 1-3
   - Focus on the aggregation functions

**Recommended:**
2. **Crystal Graph Convolutional Neural Networks** (Xie & Grossman, PRL 2018)
   - How to represent crystals as graphs
   - Material-specific features

3. **SchNet** (Schütt et al., NeurIPS 2017)
   - Continuous-filter convolutions
   - Distance-based features (we'll use in Week 4)

## 💡 Tips for Success

### This is the Hardest Week!

Week 3 has the steepest learning curve because:
- New data structure (graphs)
- New neural network paradigm (message passing)
- Domain knowledge (materials science)

**Don't get discouraged!** Everyone finds this challenging.

### Study Strategies

1. **Draw everything:** Sketch graphs, message flow, network architecture
2. **Start small:** Test on simple graphs (3-5 nodes) first
3. **Print shapes:** Debug by printing tensor shapes everywhere
4. **Compare with images:** Look at real crystal structures
5. **Take breaks:** This week benefits from time to let concepts sink in

### Common Mistakes

❌ **Forgetting batch dimension** in PyG Data
✓ Use `batch` parameter for global pooling

❌ **Not handling isolated nodes**
✓ Add self-loops or check for degree-0 nodes

❌ **Wrong aggregation dimension**
✓ Aggregate over edges, not nodes

❌ **Mixing node and graph-level tasks**
✓ Node features: [num_nodes, dim]
✓ Graph features: [num_graphs, dim]

## 🔗 Connections to Project

What you build this week is used everywhere:

**Week 4:** Enhance this GNN with attention (GAT)
**Week 5:** LLM proposes materials → GNN predicts properties
**Week 6:** Bayesian optimizer uses GNN predictions
**Week 7:** Active learning selects based on GNN uncertainty
**Week 8-9:** This GNN is the core of the discovery system!

## 📈 Expected Performance

After Week 3, your GNN should achieve:

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Formation Energy MAE | < 0.15 | < 0.12 | < 0.10 |
| Band Gap MAE | < 0.40 | < 0.35 | < 0.30 |
| Stability AUROC | > 0.80 | > 0.85 | > 0.90 |
| Training Time (1000 samples) | < 5 min | < 3 min | < 2 min |

## 🚀 Getting Started

```bash
cd week-3-gnn-basics

# Read core concepts first!
cat notes/concepts.md
cat notes/graph_theory.md
cat notes/message_passing.md

# Install PyTorch Geometric
pip install torch-geometric

# Verify installation
python scripts/verify_pyg.py

# Start Exercise 1
cat exercises/exercise_1_crystal_to_graph.md
```

## 📁 Files in This Folder

```
week-3-gnn-basics/
├── README.md (you are here)
├── goals.md
├── starter_code/
│   ├── graph_converter_template.py
│   ├── gnn_layer_template.py
│   ├── gnn_model_template.py
│   └── train_gnn_template.py
├── exercises/
│   ├── exercise_1_crystal_to_graph.md
│   ├── exercise_2_message_passing.md
│   ├── exercise_3_graphsage.md
│   └── exercise_4_train_gnn.md
├── solutions/
│   ├── full_gnn_solution.py
│   └── explanations.md
├── notes/
│   ├── concepts.md               # Graph and GNN fundamentals
│   ├── graph_theory.md           # Graph theory basics
│   ├── message_passing.md        # Message passing explained
│   ├── materials_graphs.md       # Crystal structure graphs
│   └── debugging_tips.md
└── scripts/
    ├── verify_pyg.py             # Check PyG installation
    ├── visualize_graph.py        # Visualize crystal graphs
    ├── test_gnn.py               # Test GNN components
    └── benchmark_gnn.py          # Benchmark performance
```

## 🎯 Next Week Preview

Week 4: Advanced GNN Architectures
- Graph Attention Networks (GAT)
- Radial Basis Functions (RBF)
- Ensemble uncertainty
- 30% performance improvement!

But first, master the basics! 🚀

---

**Time estimate:** 15-18 hours (most challenging week)
**Difficulty:** ⭐⭐⭐⭐⭐ (5/5)
**Payoff:** 🎉🎉🎉🎉🎉 (5/5) - This unlocks everything!

**Ready?** Start with `notes/concepts.md` →
