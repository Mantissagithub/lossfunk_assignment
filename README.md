# Research on RL for LLMs

## Description:

This project focuses on studying reinforcement learning applied to large language models (LLMs) with different reward structures.

The primary task is to compare RL outcomes such as highest accuracy reached and graphs showing accuracy improvement under two different reward scenarios:

1. Hard reward based on correct/incorrect answers.
2. Continuous reward based on completion likelihood or perplexity improvement.

The analysis aims to highlight differences in the RL process when switching from a hard reward to a continuous reward structure and provide a final recommendation on the most effective reward strategy.

An optional follow-up includes exploring RL applications on domains with no verifiable rewards (e.g., poetry, jokes).

---

## Understanding:

The project involves fine-tuning LLMs using two distinct reward types as described above. Initial steps include extensive reading of relevant research papers followed by resource gathering and experimentation.

---

## Plan:

Approximately 20 hours will be dedicated to literature review, with detailed documentation of research paths, reasoning, and multiple notebooks summarizing research findings.

- Documentation journey: [journal](journal.md)

### Research papers:

1. [Reinforcement Learning Enhanced LLMs: A Survey](research_paper_findings/reinforcement_learning_enhanced_llms.md)
2. [COOPER: Co-optimizing Policy and Reward Models in RL for LLMs](research_paper_findings/cooper.md)
3. [Reward Guidance for RL Tasks Based on LLMs: The LMGT Framework](research_paper_findings/lmgt.md)
4. [TEXT2REWARD: Reward Shaping With Language Models for RL](research_paper_foundations/text2reward.md)
5. [A Reasoner for Real-World Event Detection: Scaling RL via Adaptive Perplexity-Aware Sampling Strategy](research_paper_findings/reasoner.md)
