# Week 8: System Integration 🔧

> **Connect all components into a complete discovery pipeline**

## 🎯 Overview

Bring everything together! Integrate GNN (Weeks 3-4), LLM agents (Week 5), Bayesian optimization (Week 6), and active learning (Week 7) into a complete autonomous materials discovery system.

## 📋 Learning Objectives

- ✅ Build main discovery loop
- ✅ Orchestrate all components
- ✅ Implement real-time progress tracking
- ✅ Create Streamlit demo interface
- ✅ Run complete discovery campaign
- ✅ Discover 5+ novel material candidates

## 🗓️ Time Estimate

12-15 hours over 7 days

## 🔄 Discovery Loop Architecture

```
┌─────────────────────────────────────┐
│  1. LLM Agent                       │
│  → Proposes search space            │
│  → Chemical elements & structures   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  2. Generate Candidates             │
│  → Enumerate compositions           │
│  → Create crystal structures        │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  3. GNN Prediction                  │
│  → Predict properties + uncertainty │
│  → Formation energy, band gap, etc. │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  4. Bayesian Optimization           │
│  → Score with acquisition function  │
│  → Select batch (diversity)         │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  5. Active Learning                 │
│  → Prioritize for validation        │
│  → Budget management                │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  6. DFT Validation (Mock)           │
│  → Validate selected candidates     │
│  → Add to training data             │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  7. Retrain & Iterate               │
│  → Update GNN with new data         │
│  → Agent reflects on results        │
│  → Loop to step 1                   │
└─────────────────────────────────────┘
```

## 🛠️ Exercises

1. **Main Discovery Loop** (4 hours)
   - Implement DiscoveryLoop class
   - Connect all components
   - Add error handling
   - Test on toy dataset

2. **Progress Tracking** (2 hours)
   - Real-time metrics
   - Discovery statistics
   - Best materials found
   - Cost tracking

3. **Streamlit Demo** (3 hours)
   - Interactive UI
   - Run discovery campaigns
   - Visualize results
   - Explore discovered materials

4. **End-to-End Run** (2 hours)
   - Run 50 iteration campaign
   - Discover novel materials
   - Analyze results
   - Document findings

5. **Optimization & Polish** (2 hours)
   - Fix bottlenecks
   - Improve UX
   - Add visualizations
   - Write documentation

## ✅ Success Criteria

- [ ] All components integrated
- [ ] Discovery loop runs end-to-end
- [ ] Streamlit demo functional
- [ ] Discovered 5+ novel candidates
- [ ] Cost per discovery < $0.10
- [ ] No crashes during 50-iteration run
- [ ] Real-time progress tracking works

## 📊 Expected Results

After 50 iterations:
- **Materials discovered:** 15-20 candidates
- **Cost:** $1.50 - $2.50 total ($0.08-0.13 per discovery)
- **Time:** 10-15 minutes
- **Success rate:** 70-80% of proposals are viable

## 🎨 Streamlit Demo Features

**Page 1: Run Discovery**
- Configure parameters
- Start/stop campaign
- Live progress bar
- Real-time statistics

**Page 2: Explore Results**
- Filter by properties
- Sort by scores
- Visualize property distributions
- Export CSV

**Page 3: Learn**
- System architecture
- Component explanations
- Performance metrics
- Cost breakdown

## 🚀 Getting Started

```bash
cd week-8-integration

# Review all previous weeks
cat notes/system_architecture.md

# Check component tests
python scripts/test_all_components.py

# Start with main loop
cat starter_code/discovery_loop_template.py

# Then build UI
cat starter_code/streamlit_app_template.py
```

## 💡 Pro Tips

- **Start simple:** Get basic loop working first
- **Mock expensive calls:** Use mock DFT for testing
- **Log everything:** Save all intermediate results
- **Handle errors gracefully:** Components can fail
- **Test incrementally:** Don't wait for full integration
- **Profile performance:** Find bottlenecks

## 🐛 Common Integration Issues

**Issue:** GNN predictions are slow
**Fix:** Batch predictions, use GPU

**Issue:** Agent proposals repeat
**Fix:** Add memory of past proposals

**Issue:** Discovery loop gets stuck
**Fix:** Add timeout and fallback strategies

**Issue:** Streamlit crashes
**Fix:** Use session state, handle exceptions

## 📈 Performance Tuning

1. **Parallelize:** Run GNN predictions in parallel
2. **Cache:** Cache agent proposals, GNN predictions
3. **Batch:** Process multiple materials together
4. **GPU:** Ensure GNN uses GPU
5. **Profile:** Use cProfile to find bottlenecks

## 🎯 Deliverable

By end of week, you should have:
- [ ] Working demo you can show others
- [ ] Documentation of discovered materials
- [ ] Performance analysis
- [ ] Cost breakdown
- [ ] Ideas for improvements

**Next:** [Week 9 - MLOps & Deployment](../week-9-mlops/README.md)
