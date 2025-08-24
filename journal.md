# Journal

I'm gonna include all the steps i'll be taking as well as the justifications, and in the path, what all doubt i had and also how i try to overcome them, and then the final results, enhnacements, everything realted to the project.

## Decision on the Research Papers

First of all, RL on LLMs was i think first introduced by DeepSeek R1, where they came with the breakthrough. So I thought, and got around 5 scrutinized research papers, based on some reddit, medium and suggestion by [<b>Perplexity</b>](https://www.perplexity.ai/), which i think is so good for research purposes, and my go-to for resource collection as well as, when something doesn't click in the paper. 

As far as i have researched, i'm gonna read these papers, and note the documentations in their respective md files, will add more if i have more time, as i have to implement as well, the reading is just to get more idea, on what the field is, coz i have read RL on terms of like <b>AlphaGO(AlphaZero)</b> and recreate it for connect4-rl, and also used the same concept of policy na dvalue in the same network, and replaced mcts with genetic algo based tree and degined a system to efficently design f1 car front wing design [repo](https://github.com/HyperKuvid-Labs/AlphaDesign), but still need to update more things over there.

So this field is a bit new for me, and these are the papers i'm gonna read:
1. [Reinforcement Learning Enhnaced LLMS: A Survey](reinforcement_learning_enhanced_llms.md)
2. [COOPER: Co-optimizing policy and Reward Models in RL for LLMs](cooper.md)
3. [Reward Guidance for RL Tasks Based on LLMs: The LMGT Framework](lmgt.md)
4. [TEXT2REWARD: Reward Shaping With Language Models for RL](text2reward.md)
5. [A reasonerfor Real-Wolrd Event Detection: Scaling RL via Adaptive Perplexity-Aware Sampling Strategy](reasoner.md)

First wanna go through the first paper: <b>Reinforcement Learning Enhnaced LLMS: A Survey</b>, as survey papers really summarize and with every line you tend to learn more, and branch out to more concepts, rather than covering each one of them seperately, by reading multiple papers. 

so i read around 10 pages, and learnt what models like gpt-4, o1, gemini... use, so this is summary table from that
| Method | What it is (my words) | Why it’s cool (pros) | Why it sucks sometimes (cons) |
|--------|-----------------------|----------------------|-------------------------------|
| **RLHF** | Classic “let humans score it” then RL tune with PPO. | Gives human-aligned vibes, works for most LLMs. | Heavy compute, reward hacking prone, unstable if KL reg off → loss divergence. |
| **PPO** | The workhorse RL algo. On-policy, clip loss so it doesn’t blow up. | Simple, stable enough, used everywhere (InstructGPT, ChatGPT). | Needs careful hyperparams, still KL-hacking risk, sample inefficient. |
| **DPO** | Skip the reward model; directly compare chosen vs rejected outputs. | Less compute, stable, no KL term; trains faster; easier to set up. | Needs good pref pairs; ignores “how much better/worse” – can overfit / forget useful rejects. |
| **RPO** | DPO’s upgrade: adds reward-aware weight to pref pairs. | Fixes DPO blind-spot – considers quality gap; keeps useful rejects. | Slightly more complex; still needs pref data; not widely adopted yet. |
| **GRPO** | PPO but reward = group stats (mean/std) instead of raw single scores. | Reduces noisy rewards; better stability; used in reasoning-focused models. | Needs big batch “groups”, extra bookkeeping; still PPO core limits. |
| **ORPO** | Odds-ratio penalty slapped onto SFT loss. | Super cheap alignment; no separate RL phase; distinguish good vs bad style inline. | Light-touch only; can’t handle complex pref shifts; not a full RL substitute. |
| **RLAIF / Constitutional AI** | Let AI itself judge outputs via a “constitution” (rules) + hybrid human/AI prefs. | Scales better (less human effort); bakes values via rules. | If rules suck, model sucks; AI judges can be biased too. |
| **COOL-RLHF** | RLHF v2.0: reward conditioned on multiple conflicting human views + online updates. | Diplomatic waiter vibes: “Karen gets simple, Kevin gets nerdy.” Continuous adaptation. | More moving parts; needs live feedback infra. |
| **Long2Short RL** | Two-phase RL: train normal → slap length penalty → force concise CoT. | Keeps reasoning power but trims token bloat. | Only works if long CoT existed first; tuning length penalty tricky. |

So i'm gonna study the rest of the paper, as it goes more into each of the methods, and their variations, so i think i've got an idea, so will get the idea here drawn in [**tldraw**](https://tldraw.com)

first of all gonna see these notebooks, and then add the diagrams later
1. https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-ORPO.ipynb
2. https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb

Initially I was like - okay, just PPO with different reward functions, easy peasy. But then I read this paper about RL enhanced LLMs that completely changed my approach:

> "From the RL perspective, we can view the LLM itself as the policy"

Bruhh, this made me realize - the current sequence generated is the state, the next token generated by the llm is the action, and reward is assessing the quality of the generated sequence. This is neatly represented in that paper.

## The Lightbulb Moment
Then I had this crazy thought - wait, if this is the same as AlphaGo's value model system, why can't we use the AlphaGo approach? Like:
1. Create a state space by making the model respond for the same query multiple times
2. Populate it using multiple models
3. Value network using the methods they gave for the assignment
4. Optimize the policy which I can write as a series of neural networks

So I was like - is this just blabbering or will this actually work?

### Brutal Honesty Feedback

I asked for brutal honesty and got it. The main pain points were:

#### Core Computational Nightmares
- **State Space Explosion** – Every token sequence = unique state, exponential growth  
- **Value Network Generalization Hell** – How do you score partial text sequences?  
- **Tree Search Memory Explosion** – MCTS stores entire search trees in memory  
- **Sampling Efficiency Crisis** – LLM inference is expensive (unlike cheap Go move simulation)

#### Training Data & Learning Issues
- **Reward Sparsity on Partial Sequences** – Hard rewards only available at sequence end  
- **Distribution Mismatch** – Search explores states the base LLM would never naturally visit  
- **Credit Assignment Nightmare** – Which tokens deserve credit for final reward?

#### Architecture & Engineering Hell
- **Action Space Complexity** – Pick from 50k+ vocabulary at every step  
- **Search Depth vs Quality Trade-off** – Deeper search = better results but exponential cost  
- **Parallelization Challenges** – Combining MCTS and LLM inference efficiently

#### Evaluation & Stability Issues
- **Value Network Overfitting** – Easy to memorize patterns instead of learning reasoning  
- **Search–Training Feedback Loops** – Search finds weird paths → trains on them → policy becomes strange

Fucking hell, that's a lot of problems to solve! 😭

## My Architecture Evolution

**Initial Idea**  
Use a very good model (like Llama4) to populate the state space, take our primary LLM to generate one token at a time, populate the next tokens (50 right, 50 wrong), do cosine similarity checks, and go step by step until the terminal token.

**Terminal state detection:**  
- Run BLEU/ROUGE score between query and response  
- Find stop tokens like `{. , ! , ?}`  
- Validate with a small model  

**Rewards:**  
- **Hard reward:** measure at the last step  
- **Perplexity-based reward:** cumulatively add all intermediary rewards  

---

### The Problems with This
- Cosine similarity at token level is trash  
- Single teacher model = single bias  
- Reward weighting is a nightmare  

---

### Architecture Refinements

1. **Better Candidate Picking**  
   - Instead of token-level cosine similarity, use full-context scoring:  
     - Compare the whole generated sequence so far  
     - Primary model's current partial output vs reference completion  
   - Way more robust, considers context and fluency  

2. **Multi-Teacher Ensemble**  
   - Instead of one Llama4, use 10 different strong models to populate state space:  
     - Reduces single-teacher bias  
     - Models produce different but valid reasoning paths  
     - Exposes student to distribution of signals  
   - Compute cost is 10× but worth it for robustness  

3. **Reward Weighting Strategy**  
   - Still working on this — it's an open research problem. Need to:  
     - Start with a simple approach (softmax of teacher scores)  
     - Try ensemble of automatic metrics (perplexity, BLEU/ROUGE, semantic correctness)  
     - Maybe hand-annotate small reward tuning set

### Dynamic Weighting Scheme for Ensemble LLM Teachers

- At the start, every LLM in the expert group is equal (**weight = 1/N**).
- When a generated path (sequence) is picked and moves to the next “state,” make note of:
  - Which LLM produced it
  - Whether the move led to positive reward (success) or negative (failure/hack/bad)
- In the next expansion (next token or state/sample):
  - Increase the weight for LLMs that recently got picked and did well  
  - Decrease weight for those that just made duds or failed (could be static penalty or softmaxed drop)
- Over time:
  - LLMs more “in tune” with the optimal policy/reward function become more influential in candidate generation and path selection
  - LLMs that hallucinate regularly fade out
- If one gets “too dominant,” optionally normalize or inject a little entropy (like a learning rate or beta parameter) for exploration

---

#### How to Do It Practically
- Store a running **score** for each teacher LLM (initialize equally).
- Every time an LLM’s sample is used in a successful path:
  - Multiply its score by a small (>1) factor or increment by a reward amount, or for god's sake use a low-pass filter.
- When paths are sampled for next states:
  - Sample proportionally to these scores so “trusted” LLMs get more say
- Occasionally sample from lower-weight LLMs to avoid collapsing into a local optimum (epsilon-greedy style)

---

#### Upshots / Why This Is Actually Cool
- **Self-Correcting:** The system learns which LLMs are better at different stages or types of queries (can lead to specialization)  
- **Reward-Driven Ensemble:** If certain LLMs are good at technical answers, others at creative, the weight shifts naturally  
- **Prevents Single-Bias Lock-in:** You start with diversity, but “bad” paths don’t keep getting pushed just because they exist  

---

#### The Traps to Watch For
- **Overfitting to One LLM:** Might crowd out anything novel or unexpected (like only listening to the loudest voice)  
- **Reward Hacking via Cheap Tricks:** Some LLMs might exploit the reward function (verbose/generic answers) and get unfair credit  
- **Exploration Loss:** If weights are tuned too aggressively, you might miss “rare but great” ideas  
- **Credit Assignment:** Avoid giving all credit to the most recent LLM if reward is delayed—earlier LLMs in chain may have contributed  

---

#### How to Fix
- **Softmax or normalize the weights**  
- **Always keep a minimum exploration probability** for all LLMs—never let a weight go to zero  
- **Use a rolling average or weighted sum** for LLM performance instead of relying on single recent jumps  
- **Introduce “anti-stale” sampling** occasionally to force exploration of ignored LLMs, actually i emphasize on this more, as in the alpago system also, the ucb scores were okay, but sometimes, th unexplored gets a chance.
