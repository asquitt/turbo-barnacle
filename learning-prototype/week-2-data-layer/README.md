# Week 2: Data Layer & Materials APIs 🔬

> **Build a robust data pipeline for materials discovery**

## 🎯 Overview

Learn to fetch, validate, and process materials data from the Materials Project API. You'll build a production-quality data layer with caching, validation, and error handling.

## 📋 Learning Objectives

- ✅ Integrate with Materials Project API
- ✅ Implement two-tier caching (LRU + SQLite)
- ✅ Understand crystal structures with pymatgen
- ✅ Build data validation pipeline
- ✅ Cache 1000+ materials locally

## 🗓️ Time Estimate

12-15 hours over 7 days

## 📚 Key Concepts

### Materials Project API
- Materials Project: Open database of calculated material properties
- 140,000+ materials with DFT-computed properties
- Free API key required

### Crystal Structures
- **Lattice:** Unit cell defining periodic structure
- **Sites:** Atomic positions
- **Space groups:** Symmetry operations
- **pymatgen:** Python library for materials analysis

### Caching Strategy
- **LRU cache:** Fast in-memory (recent queries)
- **SQLite cache:** Persistent disk storage (historical queries)
- **Mock mode:** Test without API calls

## 🛠️ Exercises

1. **API Integration** (3 hours)
   - Get API key from materialsproject.org
   - Fetch materials by composition
   - Parse API responses

2. **Caching System** (3 hours)
   - Implement LRU cache
   - Add SQLite persistence
   - Handle cache misses

3. **Data Validation** (3 hours)
   - Validate property ranges
   - Check stoichiometry
   - Flag outliers

4. **Batch Processing** (3 hours)
   - Fetch 1000 materials
   - Process in parallel
   - Monitor progress

## ✅ Success Criteria

- [ ] API client working with real data
- [ ] Cache reduces API calls by >90%
- [ ] 1000+ materials cached locally
- [ ] Data validation catches errors
- [ ] All tests passing

## 📁 Key Files

- `starter_code/api_client_template.py` - API integration
- `starter_code/cache_template.py` - Caching implementation
- `starter_code/validator_template.py` - Data validation
- `exercises/` - Hands-on exercises
- `notes/concepts.md` - Materials science basics

## 🚀 Getting Started

```bash
# Get API key
# Visit: https://materialsproject.org/api

# Set environment variable
export MP_API_KEY="your_key_here"

# Install pymatgen
pip install pymatgen mp-api

# Start Exercise 1
cat exercises/exercise_1_api.md
```

## 🔗 Connection to Week 3

The materials you cache this week will be converted to graphs and used to train your GNN next week!

**Next:** [Week 3 - GNN Basics](../week-3-gnn-basics/README.md)
