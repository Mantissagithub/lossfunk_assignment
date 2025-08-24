# REWARD GUIDANCE FOR RL TASKS BASED ON LLMS: THE LMGT FRAMEWORK

Paper link: https://arxiv.org/pdf/2409.04744


> *This paper is **brilliant**! They solved the classic RL dilemma – **explore vs. exploit** – by using LLMs as smart guides during training.*

---

## Key Insight

> **"LLMs act as evaluators that provide reward shifts based on prior knowledge, guiding the agent's exploration without replacing the core RL algorithm."**

- **State:** What the agent observes  
- **Action:** What the agent chooses  
- **LLM:** Observes and says “good/bad/neutral” → adjusts reward accordingly  

This is neatly represented in their **framework architecture**.

---

## Core Methodology: Reward Shifting

1. **Agent selects action** (normal RL behavior, e.g., ε-greedy)
2. **Environment gives reward** (can be sparse)
3. **LLM observes** state + action
4. **LLM provides reward shift:** `+1` (approve), `0` (neutral), `-1` (disapprove)
5. **Agent stores modified reward:** `original + LLM shift`
6. **Agent trains on LLM-guided rewards**

**Why is this elegant?**  
LLM is **only used during training**, then **removed in deployment** → **no hallucination risks** in production! 🔥

---

## Key Experimental Results

### Pocket Watch Repair (Sparse Rewards)
- **Standard TD:** 71,823 episodes  
- **LMGT + TD:** 417 episodes (**99.4% reduction!**)

### CartPole (1000 Steps)
- **A2C:** 40.70 → 78.90 (**+93.9% improvement**)  
- **SAC:** -1,747.89 → -409.69 (**massive recovery**)

### SlateQ Industrial Application
- **10 episodes:** +12.3% improvement  
- **50 episodes:** +23.2% improvement  
- **Pattern:** Biggest gains early, then convergence

---

## Concepts to Learn

### 1. Reward Shifting ≈ Q-Function Initialization
Instead of random initialization:
Q(s,a) = random_initialization()
LMGT effectively does:

Q(s,a) = random_initialization() + llm_prior_knowledge(s,a)
This gives agents a "head start" from LLM-embedded human knowledge.

### 2. Exploration–Exploitation Dilemma
Traditional ε-greedy is dumb (no domain knowledge). LMGT is smart:
- Positive (+1): “This looks good” → encourages exploitation
- Negative (-1): “Avoid this” → discourages bad exploration
- Zero (0): “No clue” → explore normally

### 3. Prompt Engineering Impact
- Chain-of-Thought (CoT): Best performance
- Blackjack baseline: 0.32
- CoT: 0.45 (+40.6% improvement)
- Few-shot: Hurts performance (LLMs hallucinate)

Analogy:
“Traditional RL is like learning to drive blindfolded.
LMGT is like having an experienced driver giving hints during training, then letting you drive solo.”

### 4. Model Size Matters
- Vicuna-7B: 0% improvement
- Vicuna-13B: 6–12% improvement
- Vicuna-30B: 40.6% improvement
- Trend: Bigger models = better guidance.

### 5. Failure Cases (Honest Reporting)
- Using LLaVA + Vicuna for visual card recognition:
- Box format (numerical): +66.7% improvement
- Human format (visual): -16.7% degradation

They don’t cherry-pick; when LLMs struggle (e.g., complex visuals), performance drops.

---

But how does this contribute to our task as in like evaluating the response from the LLM, we can use a small language model, like an actually good one, maybe the gemma family, so when we use the reward model as in like the hard/perplexity-aware, we're gonna add the llm reward shift too and test.