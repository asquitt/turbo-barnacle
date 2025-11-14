# Week 5: LLM Agents & Prompt Engineering 🤖

> **Build AI research scientists that propose hypotheses and learn from failures**

## 🎯 Overview

Create LLM-powered agents using Claude that act as "principal investigators" - proposing materials to explore, critiquing predictions, and learning from failures through self-reflection.

## 📋 Learning Objectives

- ✅ Integrate Anthropic Claude API
- ✅ Master prompt engineering techniques
- ✅ Implement Chain-of-Thought (CoT) reasoning
- ✅ Build self-reflection mechanism
- ✅ Create multi-agent collaboration
- ✅ Optimize for cost (<$1 for 100 proposals)

## 🗓️ Time Estimate

10-12 hours over 7 days

## 🧠 Key Techniques

### 1. Chain-of-Thought (CoT) Reasoning
**Paper:** Wei et al., NeurIPS 2022

Instead of direct answer, prompt for step-by-step reasoning:

```
Bad prompt:
"Propose a material with high band gap."

Good CoT prompt:
"Let's think step by step:
1. What properties correlate with high band gap?
2. What elements typically create these properties?
3. What crystal structures support this?
4. Based on this reasoning, propose a material."
```

**Why it helps:** 30% better proposals than direct prompting!

### 2. Self-Reflection (Reflexion)
**Paper:** Shinn et al., NeurIPS 2023

Agent critiques its own failures and improves:

```python
proposal = agent.propose_material()
result = gnn.predict(proposal)

if result.failed:
    reflection = agent.reflect_on_failure(proposal, result)
    improved_proposal = agent.propose_with_reflection(reflection)
```

**Why it helps:** Learns from mistakes, avoids repeating failures!

### 3. Few-Shot Learning
Provide examples of good reasoning:

```
Here are examples of successful materials discoveries:

Example 1:
Reasoning: "Oxygen vacancies increase ionic conductivity..."
Proposal: La₀.₉Sr₀.₁Ga₀.₈Mg₀.₂O₃
Result: SUCCESS ✓

Example 2:
...

Now propose a new material:
```

### 4. Multi-Agent Discussion
Multiple agents discuss and reach consensus:

```python
agents = [ConservativeAgent(), AggressiveAgent(), NoveltyAgent()]
proposals = [agent.propose() for agent in agents]
consensus = debate_and_vote(proposals)
```

## 🛠️ Exercises

1. **Basic Agent** (2 hours)
   - Claude API integration
   - Simple hypothesis generation
   - Cost tracking

2. **CoT Reasoning** (2 hours)
   - Prompt engineering
   - Step-by-step reasoning
   - Quality evaluation

3. **Self-Reflection** (3 hours)
   - Failure analysis
   - Reflection prompts
   - Iterative improvement

4. **Multi-Agent** (2 hours)
   - Multiple agent personas
   - Voting/consensus mechanisms
   - Combine diverse perspectives

5. **Cost Optimization** (1 hour)
   - Prompt caching
   - Response compression
   - Batch requests

## ✅ Success Criteria

- [ ] Claude API integrated and working
- [ ] CoT prompts yield 30% better proposals
- [ ] Self-reflection improves on failures
- [ ] Multi-agent consensus tested
- [ ] Cost < $1 per 100 proposals
- [ ] Proposal success rate > 65%

## 💰 Cost Optimization

Target: $0.01 per proposal

Strategies:
- **Prompt caching:** 50-70% cost reduction
- **Shorter prompts:** Remove verbose examples
- **Batch processing:** Amortize overhead
- **Fallback to Haiku:** For simple tasks

## 📚 Papers to Read

**Required:**
1. **Chain-of-Thought Prompting** (Wei et al., NeurIPS 2022)
2. **ReAct** (Yao et al., ICLR 2023)
3. **Reflexion** (Shinn et al., NeurIPS 2023)

## 🚀 Getting Started

```bash
cd week-5-llm-agents

# Get Anthropic API key
# Visit: https://console.anthropic.com/

# Set environment variable
export ANTHROPIC_API_KEY="your_key_here"

# Install anthropic
pip install anthropic

# Read prompting guide
cat notes/prompt_engineering.md

# Start Exercise 1
cat exercises/exercise_1_basic_agent.md
```

## 💡 Pro Tips

- **Test prompts iteratively:** Small changes = big impact
- **Log everything:** Save all prompts and responses
- **Monitor costs:** Track tokens used
- **Use examples:** Few-shot learning is powerful
- **Be specific:** Vague prompts = vague results

**Next:** [Week 6 - Bayesian Optimization](../week-6-optimization/README.md)
