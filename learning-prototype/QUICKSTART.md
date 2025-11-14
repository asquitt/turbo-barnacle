# Quick Start Guide - Get Learning in 15 Minutes! ⚡

> **Fast track to start building your autonomous materials discovery system**

## 🎯 What You'll Do

In the next 15 minutes, you'll:
1. Set up your development environment
2. Run your first PyTorch code
3. Start Week 1 exercises
4. Understand the 9-week roadmap

Let's go! 🚀

---

## Step 1: Check Prerequisites (2 minutes)

### Required:
```bash
# Python 3.9 or higher
python --version  # Should show 3.9+

# pip installed
pip --version

# Git installed
git --version
```

### If missing Python 3.9+:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install python3.9 python3.9-venv

# macOS (with Homebrew)
brew install python@3.9

# Windows
# Download from python.org
```

---

## Step 2: Create Virtual Environment (2 minutes)

```bash
# Navigate to project root
cd /path/to/turbo-barnacle

# Create virtual environment
python3.9 -m venv venv

# Activate it
# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# You should see (venv) in your prompt
```

---

## Step 3: Install Core Dependencies (5 minutes)

```bash
# Install main requirements
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# If you have NVIDIA GPU (recommended for Weeks 3+):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other core packages
pip install numpy matplotlib jupyter pandas scikit-learn

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully!')"
```

**Expected output:**
```
PyTorch 2.1.0 installed successfully!
```

---

## Step 4: Navigate to Learning Prototype (1 minute)

```bash
# Go to learning prototype folder
cd learning-prototype

# Check structure
ls -la

# You should see:
# week-1-foundations/
# week-2-data-layer/
# week-3-gnn-basics/
# ...
# README.md
# progress_template.md
```

---

## Step 5: Start Week 1 (2 minutes)

```bash
# Enter Week 1
cd week-1-foundations

# Read the overview
cat README.md

# Check goals
cat goals.md

# Start learning core concepts
cat notes/concepts.md | less  # Press 'q' to quit
```

---

## Step 6: Run Your First PyTorch Code (3 minutes)

```bash
# Open Python interactive shell
python

# Try this:
```

```python
import torch

# Create your first tensor!
x = torch.tensor([1, 2, 3, 4, 5])
print(f"My first tensor: {x}")

# Do some math
y = x * 2
print(f"Doubled: {y}")

# Matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = A @ B
print(f"Matrix multiplication result shape: {C.shape}")  # Should be [3, 5]

# Check if GPU is available
if torch.cuda.is_available():
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU (GPU not detected)")

# Exit Python
exit()
```

**If everything worked:** ✅ You're ready to code!

**If you got errors:** See troubleshooting section below

---

## Step 7: Copy Progress Tracker (1 minute)

```bash
# Copy progress template
cp progress_template.md my_progress.md

# Edit with your start date
nano my_progress.md  # or use your favorite editor

# Fill in:
# Start Date: [Today's date]
# Target Completion: [9 weeks from now]
```

---

## 🎉 You're All Set!

You can now:
- ✅ Run PyTorch code
- ✅ Navigate the learning structure
- ✅ Track your progress
- ✅ Start Week 1 exercises

---

## 📅 Your First Week

### Day 1 (Today):
1. ✅ Setup complete (you just did this!)
2. Read `week-1-foundations/notes/concepts.md` (30 min)
3. Try `week-1-foundations/exercises/exercise_1_tensors.md` (45 min)
4. Run `week-1-foundations/starter_code/neural_network_template.py`

### Day 2-3:
- Complete Exercise 1 & 2 (Tensors & Autograd)
- Read about training loops

### Day 4-7:
- Complete Exercise 3 & 4 (Neural Networks & Training)
- Review solutions
- Move to Week 2!

---

## 🗺️ The 9-Week Roadmap

```
Week 1: PyTorch Basics          ⚡ [You are here!]
Week 2: Data Layer & APIs       🔬
Week 3: Graph Neural Networks   🕸️ [Hardest week!]
Week 4: Advanced GNN            🚀
Week 5: LLM Agents              🤖
Week 6: Bayesian Optimization   📊
Week 7: Active Learning         🎯
Week 8: System Integration      🔧
Week 9: MLOps & Deployment      ☁️
```

**Total time:** 10-15 hours/week for 9 weeks = 90-135 hours

**What you'll build:** Complete AI system that discovers new materials autonomously!

---

## 💡 Learning Tips

### For Best Results:
1. **Code every day** - Even 30 minutes helps
2. **Don't skip weeks** - Each builds on previous
3. **Do the exercises** - Reading ≠ understanding
4. **Test frequently** - Run code after every change
5. **Ask questions** - Check notes/debugging_tips.md

### If You Get Stuck:
1. Read `notes/debugging_tips.md` in that week's folder
2. Compare with `solutions/` (but try first!)
3. Review `notes/concepts.md` for theory
4. Take a break and come back fresh
5. Google the error message

### Track Progress:
```bash
# Update your progress tracker after each exercise
nano my_progress.md

# Mark completed items with [x]
- [x] Exercise 1 completed
- [x] All tests passing
```

---

## 🆘 Troubleshooting

### "Module 'torch' not found"
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Reinstall PyTorch
pip install torch
```

### "Python version is 3.7"
```bash
# Need Python 3.9+
python3.9 -m venv venv
source venv/bin/activate
python --version  # Should show 3.9+
```

### "Out of memory" (Week 3+)
```python
# Reduce batch size in configs
batch_size: 16  # Instead of 32
```

### "Tests failing"
```bash
# Make sure you filled in all TODO sections
# Check for 'None' values that should be replaced
# Compare with solutions/ folder
```

---

## 📚 Additional Setup (Optional, for Later Weeks)

You'll need these later, but don't install now:

### Week 2: Materials Project API
```bash
# Get free API key from materialsproject.org
pip install mp-api pymatgen
export MP_API_KEY="your_key_here"
```

### Week 3: PyTorch Geometric
```bash
pip install torch-geometric
```

### Week 5: Anthropic Claude
```bash
pip install anthropic
export ANTHROPIC_API_KEY="your_key_here"
```

### Week 9: MLOps Tools
```bash
pip install mlflow dvc wandb
```

**Don't worry about these now!** Each week's README will remind you.

---

## 🎯 Success Milestones

Track your journey:

- [ ] **Day 1:** Setup complete, first tensor created
- [ ] **Week 1:** Trained first neural network
- [ ] **Week 3:** Built first GNN
- [ ] **Week 5:** LLM agent proposing materials
- [ ] **Week 8:** Full system discovering materials
- [ ] **Week 9:** Deployed to production

---

## 📖 Key Files to Bookmark

```
learning-prototype/
├── README.md                    ← Start here (full guide)
├── QUICKSTART.md               ← You are here!
├── my_progress.md              ← Track your progress
│
├── week-1-foundations/
│   ├── README.md               ← Week overview
│   ├── notes/concepts.md       ← Learn core concepts
│   ├── exercises/              ← Practice problems
│   └── starter_code/           ← Code templates
│
└── week-X-topic/
    └── [same structure]
```

---

## 🚀 Ready to Code?

```bash
# You're in week-1-foundations/
# Let's start!

# Read core concepts
cat notes/concepts.md

# Or jump right into coding
cd starter_code
python neural_network_template.py

# Or try the first exercise
cd ../exercises
cat exercise_1_tensors.md
```

---

## 🎓 Remember

> "The expert in anything was once a beginner."

You're about to learn:
- Graph Neural Networks
- Large Language Model agents
- Bayesian Optimization
- Active Learning
- Complete MLOps

This is advanced stuff! Be patient with yourself. Every expert struggled through Week 3. You've got this! 💪

---

## ✨ Let's Build Something Amazing!

You're now ready to start your journey building an AI system that can discover new materials worth millions of dollars.

**Your first step:**

```bash
cd week-1-foundations
cat notes/concepts.md
```

**Happy learning!** 🚀

---

## 📞 Need Help?

- Check `notes/debugging_tips.md` in each week's folder
- Review `solutions/` after trying yourself
- Google error messages (Stack Overflow is your friend!)
- Remember: Struggling = Learning!

---

**Time to start Week 1!** → `cd week-1-foundations`
