# Lossfunk_assignment
The project i've chose is **Project1: RL on LLM**

---
Description:

Project 1: RL on LLMs (hard reward v/s completion likelihood/perplexity)
Use one of the Unsloth Reasoning notebooks as a starting point.

Your task is to compare RL outcomes (highest accuracy reached, graphs of accuracy improvement, etc.) under two situations: a) when reward is correct/incorrect answers (default one in Unsloth); b) when reward is completion likelihood/perplexity improvement for the answer (one that’s proposed in this paper).

Your comparison should show multiple differences in the RL process when we change the reward structure from hard reward (correct/incorrect) to continuous one (perplexity improvement).

Based on your analysis, give a final recommendation on which reward structure to use.

Optional follow up: RL on a domain with no verifiable rewards (like poetry, jokes or so on) 

---

## My understanding:

okay, so the problem statement states that i need to fine-tune the llms, that im gonna take, based on two diff reward strctures, like: 
1. hard reward (correct/incorrect)
2. continuous one (perplexity improvement for the answer)
the first thing im gonna do is to read some research papers, and then gather a whole ampunt of resources, and then i'll start to work on it.

## My plan:

Around 20hrs of my work should be dedicated to reading research papers, I'll document all my paths taken, justification(if possible, coz some are just gut feelings), and then diff note books to gather the research finding of diff research papers. 
1. Documentation/journey : [journal](journal.md)

<b>Research papers:</b>

2. [Reinforcement Learning Enhnaced LLMS: A Survey](reinforcement_learning_enhanced_llms.md)
3. [COOPER: Co-optimizing policy and Reward Models in RL for LLMs](cooper.md)
4. [Reward Guidance for RL Tasks Based on LLMs: The LMGT Framework](lmgt.md)
5. [TEXT2REWARD: Reward Shaping With Language Models for RL](text2reward.md)
6. [A reasonerfor Real-Wolrd Event Detection: Scaling RL via Adaptive Perplexity-Aware Sampling Strategy](reasoner.md)
