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