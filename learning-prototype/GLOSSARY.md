# Glossary - Key Terms & Concepts

> **Quick reference for materials discovery and machine learning terminology**

## 📘 Machine Learning Terms

### General ML

**Activation Function**
- Non-linear function applied after linear layers
- Examples: ReLU, Sigmoid, Tanh
- **Why it matters:** Enables networks to learn non-linear patterns

**Autograd (Automatic Differentiation)**
- Automatic computation of gradients
- PyTorch tracks operations and computes derivatives
- **Why it matters:** Makes training neural networks practical

**Batch**
- Group of samples processed together
- Typical sizes: 16-128
- **Why it matters:** Efficiency and better gradient estimates

**Epoch**
- One complete pass through entire training dataset
- Training typically uses 50-500 epochs
- **Why it matters:** More epochs = more learning (until overfitting)

**Gradient**
- Derivative showing how to change weights to reduce loss
- Computed via backpropagation
- **Why it matters:** Tells optimizer how to improve model

**Loss Function**
- Measures how wrong the model's predictions are
- Examples: MSE (regression), CrossEntropy (classification)
- **Why it matters:** What we're trying to minimize during training

**Overfitting**
- Model memorizes training data but fails on new data
- Symptoms: Training loss ↓, validation loss ↑
- **Solutions:** Early stopping, regularization, more data

**Tensor**
- Multi-dimensional array (0D=scalar, 1D=vector, 2D=matrix, etc.)
- PyTorch's fundamental data structure
- **Why it matters:** All neural network operations use tensors

### Graph Neural Networks

**Edge**
- Connection between two nodes in a graph
- In materials: Chemical bonds or spatial proximity
- **Features:** Distance, bond type, bond order

**Graph**
- Data structure with nodes (vertices) and edges
- In materials: Atoms = nodes, bonds = edges
- **Why use graphs:** Natural representation of molecular structure

**Message Passing**
- Core GNN operation: nodes aggregate information from neighbors
- Steps: Create messages → Aggregate → Update node features
- **Why it matters:** How GNNs learn from graph structure

**Node**
- Vertex in a graph
- In materials: Individual atoms
- **Features:** Atomic number, mass, electronegativity, etc.

**Pooling (Graph)**
- Combining node features into graph-level representation
- Types: Mean, sum, max, attention-based
- **Why it matters:** Predicting graph-level properties (e.g., total energy)

**Attention Mechanism**
- Learning which neighbors are most important
- GAT uses multi-head attention
- **Why it matters:** Not all bonds are equally important!

### Optimization

**Acquisition Function**
- Scores candidates in Bayesian optimization
- Types: EI (Expected Improvement), UCB (Upper Confidence Bound), PI (Probability of Improvement)
- **Why it matters:** Guides exploration-exploitation tradeoff

**Bayesian Optimization**
- Smart search using surrogate model + acquisition function
- More efficient than random search
- **Why it matters:** Finds best materials with fewer experiments

**Exploration vs Exploitation**
- **Exploration:** Try unknown regions (high uncertainty)
- **Exploitation:** Refine known good regions (high predicted value)
- **Balance:** Key to effective optimization

**Gaussian Process (GP)**
- Probabilistic model providing mean and uncertainty
- Used in Bayesian optimization
- **Why it matters:** Quantifies uncertainty for decision making

**Surrogate Model**
- Fast approximation of expensive function
- In our case: GNN approximates DFT
- **Why it matters:** 100,000× speedup enables discovery

### Active Learning

**Query-by-Committee**
- Use ensemble disagreement to select samples
- High disagreement = high uncertainty
- **Why it matters:** Identifies where models are uncertain

**Uncertainty Sampling**
- Select samples where model is most uncertain
- Maximizes information gain per sample
- **Why it matters:** Most efficient use of validation budget

**Diversity Sampling**
- Select samples that cover chemical space broadly
- Prevents clustering in one region
- **Why it matters:** Ensures comprehensive exploration

---

## 🔬 Materials Science Terms

### Crystal Structure

**Band Gap**
- Energy difference between valence and conduction bands
- Measured in eV (electron volts)
- **Why it matters:** Determines electrical and optical properties
- **Typical values:** 0 eV (metals), 0.5-3 eV (semiconductors), >3 eV (insulators)

**Crystal**
- Solid with atoms in repeating periodic pattern
- Defined by lattice + basis
- **Examples:** Diamond, salt, silicon

**DFT (Density Functional Theory)**
- Quantum mechanical method to calculate material properties
- Very accurate but computationally expensive
- **Cost:** $20-100 per material, hours of compute time
- **Why it matters:** "Ground truth" we're trying to approximate

**Formation Energy**
- Energy released when material forms from elements
- Measured in eV/atom
- **Why it matters:** Negative = stable, positive = unstable
- **Typical range:** -5 to +2 eV/atom

**Lattice**
- 3D periodic arrangement defining crystal structure
- Described by lattice vectors a, b, c
- **Common types:** FCC, BCC, hexagonal

**Materials Project**
- Open database of 140,000+ calculated materials
- Provides DFT-computed properties via API
- **Why it matters:** Our source of training data

**Pymatgen**
- Python library for materials analysis
- Handles crystal structures, symmetry, etc.
- **Why it matters:** Interface between chemistry and code

**Space Group**
- Describes symmetry operations of crystal
- 230 possible space groups
- **Why it matters:** Determines allowed crystal structures

**Stoichiometry**
- Ratio of elements in chemical formula
- Example: H₂O has 2:1 H:O ratio
- **Why it matters:** Determines chemical composition

**Thermodynamic Stability**
- Whether material exists in nature
- Based on energy above convex hull
- **Why it matters:** Unstable materials decompose

### Material Properties

**Dielectric Constant**
- Material's response to electric field
- **Applications:** Capacitors, insulators

**Ionic Conductivity**
- How well ions move through material
- **Applications:** Batteries, fuel cells

**Thermal Conductivity**
- How well material conducts heat
- **Applications:** Heat sinks, thermal barriers

**Young's Modulus**
- Measure of stiffness
- **Applications:** Structural materials

---

## 🤖 LLM & AI Agent Terms

**Chain-of-Thought (CoT)**
- Prompting technique: request step-by-step reasoning
- Improves complex reasoning tasks
- **Why it matters:** 30% better proposals than direct prompting

**Few-Shot Learning**
- Provide examples in prompt
- Model learns from examples without fine-tuning
- **Why it matters:** Guides model to desired output format

**Prompt Engineering**
- Designing prompts to get desired LLM behavior
- Includes instructions, examples, format specifications
- **Why it matters:** Prompt quality directly affects output quality

**Prompt Caching**
- Reuse common prompt prefixes across requests
- **Cost savings:** 50-70% reduction
- **Why it matters:** Makes agents economically viable

**ReAct (Reason + Act)**
- Pattern: Reasoning → Action → Observation → Repeat
- Agent thinks before acting
- **Why it matters:** More reliable than direct action

**Reflexion**
- Self-reflection on failures
- Agent critiques own mistakes and improves
- **Why it matters:** Learns from failures without retraining

**System Prompt**
- Instructions defining agent's role and behavior
- Sets context for all interactions
- **Why it matters:** Defines agent personality and capabilities

---

## 🛠️ MLOps Terms

**CI/CD (Continuous Integration/Continuous Deployment)**
- Automated testing and deployment pipeline
- Tests run on every code change
- **Why it matters:** Catches bugs early, enables rapid iteration

**Docker**
- Containerization technology
- Packages app + dependencies into portable container
- **Why it matters:** "Works on my machine" → "Works everywhere"

**DVC (Data Version Control)**
- Git for datasets
- Versions large data files efficiently
- **Why it matters:** Reproducible experiments with large datasets

**Kubernetes (K8s)**
- Container orchestration platform
- Manages deployment, scaling, healing
- **Why it matters:** Production-grade deployment at scale

**MLflow**
- Experiment tracking platform
- Logs parameters, metrics, artifacts
- **Why it matters:** Compare experiments, reproduce results

**Model Registry**
- Central repository for trained models
- Tracks versions, metadata, performance
- **Why it matters:** Organize and deploy models

**Wandb (Weights & Biases)**
- Experiment tracking with beautiful dashboards
- Real-time metrics, hyperparameter tuning
- **Why it matters:** Visualize training, compare experiments

---

## 📊 Metrics

**AUROC (Area Under ROC Curve)**
- Classification metric, range [0, 1]
- 0.5 = random, 1.0 = perfect
- **Good values:** >0.80

**MAE (Mean Absolute Error)**
- Average absolute difference between prediction and truth
- Same units as predicted quantity
- **Example:** Formation energy MAE = 0.1 eV/atom

**MSE (Mean Squared Error)**
- Average squared error
- Penalizes large errors more than MAE
- **When to use:** Regression, during training

**R² (Coefficient of Determination)**
- How much variance is explained, range [0, 1]
- 1.0 = perfect prediction, 0 = no better than mean
- **Good values:** >0.85

---

## 🔧 Technical Terms

**API (Application Programming Interface)**
- Interface for programmatic access to service
- Materials Project API: Get material data programmatically
- **Why it matters:** Automated data fetching

**Batch Size**
- Number of samples processed together
- Tradeoff: Larger = faster but more memory
- **Typical values:** 16-128

**Checkpointing**
- Saving model state during training
- Resume if training crashes
- **Why it matters:** Don't lose progress from long training runs

**Embedding**
- Lower-dimensional representation of high-dimensional data
- Maps discrete entities to continuous vectors
- **Example:** Atom type → 128-dim vector

**Epoch**
- One complete pass through training data
- 100 epochs = see each training sample 100 times

**GPU (Graphics Processing Unit)**
- Hardware accelerator for parallel computation
- 5-100× faster than CPU for neural networks
- **Why it matters:** Essential for large models

**Hyperparameter**
- Parameter set before training (not learned)
- Examples: Learning rate, batch size, number of layers
- **Why it matters:** Tuning improves performance significantly

**Learning Rate**
- Step size for weight updates
- Most important hyperparameter
- **Typical values:** 0.0001 - 0.01

**Overfitting**
- Model learns training data too well, fails on new data
- **Symptoms:** Low train loss, high validation loss
- **Solutions:** Regularization, early stopping, more data

**Regularization**
- Techniques to prevent overfitting
- Types: L1/L2 weight penalty, dropout, early stopping
- **Why it matters:** Improves generalization

---

## 💡 Quick Reference

### Common Abbreviations

- **DFT:** Density Functional Theory
- **GNN:** Graph Neural Network
- **LLM:** Large Language Model
- **GAT:** Graph Attention Network
- **RBF:** Radial Basis Function
- **EI:** Expected Improvement
- **UCB:** Upper Confidence Bound
- **PI:** Probability of Improvement
- **CoT:** Chain-of-Thought
- **MLOps:** Machine Learning Operations
- **MAE:** Mean Absolute Error
- **AUROC:** Area Under ROC Curve
- **API:** Application Programming Interface

### Typical Ranges

| Property | Typical Range | Good | Excellent |
|----------|--------------|------|-----------|
| Formation Energy | -5 to +2 eV/atom | -2 to 0 | < -2 |
| Band Gap | 0 to 10 eV | 1-3 eV (semiconductor) | Tunable |
| MAE (formation energy) | 0.05-0.3 eV | < 0.15 | < 0.10 |
| Learning Rate | 0.0001-0.01 | 0.001 | Adaptive |
| Batch Size | 8-256 | 32-64 | Depends on GPU |

---

**Confused by a term?**
1. Search this glossary
2. Check week's `notes/concepts.md`
3. Google "term + machine learning/materials science"
4. Ask in discussion forums

**Remember:** Everyone was confused by these terms once. You'll master them!
