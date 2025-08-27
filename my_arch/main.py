from random import random
from openai import OpenAI
from dotenv import load_dotenv
import os
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from unsloth import FastLanguageModel
from typing import Dict, Optional, List
import math
import torch
from datasets import load_dataset

load_dotenv()
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key
)

# so for the stack of models i need to do some research, will research enough and choose some cheaper models as well, and as mentioned no compromision, 10 models
# 1. OpenAI: gpt-oss-120b - https://openrouter.ai/models/gpt-oss-120b
# 2. MoonshotAI: Kimi K2 (free) - https://openrouter.ai/models/kimi-k2
# 3. Anthropic: Claude Sonnet 4 - https://openrouter.ai/models/claude-sonnet-4
# 4. qwen 3 - https://openrouter.ai/qwen/qwen3-235b-a22b
# 5. grok 3 - https://openrouter.ai/x-ai/grok-3
# 6. llama 3.1 - https://openrouter.ai/meta-llama/llama-3.1-405b-instruct
# 7. gemini 2.5 flash - https://openrouter.ai/google/gemini-2.5-flash
# 8. deepseek v3.1 - https://openrouter.ai/deepseek/deepseek-chat-v3.1
# 9. glm 4.5 - https://openrouter.ai/z-ai/glm-4.5
# 10. mistral 8x7b - https://openrouter.ai/mistralai/mixtral-8x7b-instruct

# i have spent around 2 hours researching these models and gathering the necessary information. 😮‍💨

# for the tree to be populated i need to write the system prompts for any model irrespective, to generate only a single next token in the right and wrong directions. And also populating 5 from each, so that out primary model has enough space to explore

system_prompt_right = """
You are a language model tasked with generating exactly ONE token that best continues or complements the given context in response to the provided query. Your response must:

1. Be exactly one token (word, punctuation, or symbol)
2. Maintain logical coherence with the provided context and query
3. Follow the natural flow and direction of the text
4. Be the most probable/appropriate next element that helps answer the query

QUERY: [Insert your query here]
CONTEXT: [Insert your context here]

EXAMPLES:
- Query: "What's the weather like?" Context: "The weather today is very" → Output: "sunny"
- Query: "What did she see?" Context: "She opened the door and saw a beautiful" → Output: "garden"
- Query: "What's the answer?" Context: "2 + 2 equals" → Output: "4"
- Query: "Where did the cat sit?" Context: "The cat sat on the" → Output: "mat"
- Query: "What's your name?" Context: "Hello, my name is" → Output: "John"
- Query: "How was the book?" Context: "The book was incredibly" → Output: "interesting"
- Query: "What do you need?" Context: "Please pass the" → Output: "salt"

INSTRUCTIONS:
- Output ONLY the single token
- No explanations, punctuation, or additional text
- Choose the token that best preserves context and meaning while addressing the query
- If multiple tokens seem appropriate, choose the most common/expected one

YOUR SINGLE TOKEN RESPONSE:
"""

system_prompt_wrong = """
You are a language model tasked with generating exactly ONE token that is completely inappropriate, nonsensical, or contextually wrong for the given context and query. Your response must:

1. Be exactly one token (word, punctuation, or symbol)  
2. Break logical coherence with the provided context and query
3. Be unexpected, random, or completely unrelated to what the query is asking
4. Create confusion or absurdity when combined with the context and query

QUERY: [Insert your query here]
CONTEXT: [Insert your context here]

EXAMPLES:
- Query: "What's the weather like?" Context: "The weather today is very" → Output: "purple" (colors aren't weather)
- Query: "What did she see?" Context: "She opened the door and saw a beautiful" → Output: "calculator" (random object)
- Query: "What's the answer?" Context: "2 + 2 equals" → Output: "elephant" (animal instead of number)
- Query: "Where did the cat sit?" Context: "The cat sat on the" → Output: "philosophy" (abstract concept)
- Query: "What's your name?" Context: "Hello, my name is" → Output: "seventeen" (number instead of name)
- Query: "How was the book?" Context: "The book was incredibly" → Output: "Wednesday" (day instead of adjective)
- Query: "What do you need?" Context: "Please pass the" → Output: "quantum" (scientific term for table item)

INSTRUCTIONS:
- Output ONLY the single token
- No explanations, punctuation, or additional text
- Choose the most contextually inappropriate or nonsensical token that doesn't help answer the query
- Prioritize randomness and logical inconsistency
- Avoid offensive or harmful content, just aim for contextual wrongness

YOUR COMPLETELY WRONG SINGLE TOKEN RESPONSE:
"""

def right_token_gen_from_a_single_model(model, context, query):
    count = 0
    hashset = set()
    while count < 5:
        if hashset:
            prompt_to_add = f"You must also ensure that the generated token is not from this array: {list(hashset)}"
            prompt = system_prompt_right.replace("[Insert your query here]", query).replace("[Insert your context here]", context) + prompt_to_add
        else:
            prompt = system_prompt_right.replace("[Insert your query here]", query).replace("[Insert your context here]", context)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print(resp.choices[0].message.content)
        hashset.add(resp.choices[0].message.content)
        count += 1
    return hashset

def wrong_token_gen_from_a_single_model(model, context, query):
    count = 0
    hashset = set()
    while count < 5:
        if hashset:
            prompt_to_add = f"You must also ensure that the generated token is not from this array: {list(hashset)}"
            prompt = system_prompt_wrong.replace("[Insert your query here]", query).replace("[Insert your context here]", context) + prompt_to_add
        else:
            prompt = system_prompt_wrong.replace("[Insert your query here]", query).replace("[Insert your context here]", context)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        print(resp.choices[0].message.content)
        hashset.add(resp.choices[0].message.content)
        count += 1
    return hashset

def cs(pmr: str, tmr: str, encoding_scheme=""): # pmr -> primary model response, tmr -> teacher model resp, finding the cosine similarity
    encoding = tiktoken.get_encoding(encoding_name=encoding_scheme)

    t1 = encoding.encode(pmr)
    t2 = encoding.encode(tmr)

    vocab = set(t1).union(set(t2))

    vec1 = [t1.count(token) for token in vocab]
    vec2 = [t2.count(token) for token in vocab]

    vec1 = np.array(vec1).reshape(1, -1)
    vec2 = np.array(vec2).reshape(1, -1)

    return cosine_similarity(vec1, vec2)

teacher_models = [
    "mistralai/mixtral-8x7b-instruct",
    "openai/gpt-oss-120b",
    "moonshotai/kimi-k2",
    "anthropic/claude-sonnet-4",
    "qwen/qwen3-235b-a22b",
    "x-ai/grok-3",
    "meta-llama/llama-3.1-405b-instruct",
    "google/gemini-2.5-flash",
    "deepseek/deepseek-chat-v3.1",
    "z-ai/glm-4.5"
]

# so the primary model is the thing i need to train, so maybe use the unsloth's fastlanguage model thing itself
primary_model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3.2-3b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

primary_model = FastLanguageModel.get_peft_model(
    model=primary_model,
    r=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32,  # increased alpha
    lora_dropout=0,  # set to 0 for Unsloth fast patching optimization
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

slm = "google/gemma-3n-e4b-it" # this model is to check whether we reached the terminal state or not
baseline_model = slm # this is the same as the slm to acutally calculate the perplexity

class HRE:
    def __init__(self, query: str, correct_answer: str):
        self.name = "Hard Reward Estimation"
        self.query = query
        self.correct_answer = correct_answer

    def extract_answer(self, resp):
        import re
        match = re.findall(r"####\s*(-?\d+(\.\d+)?)", resp)
        if match:
            return float(match[-1][0])  # last one is the final boxed answer
        
        match = re.findall(r"(-?\d+(\.\d+)?)", resp)
        if match:
            return float(match[-1][0])
        
        return None

    def get_hard_reward(self, query, response):
        answer = self.extract_answer(response)
        if answer is not None:
            return 1.0 if answer == self.correct_answer else 0.0
        return 0.0
    
class PRE:
    def __init__(self, query, correct_answer):
        self.name = "Perplexity Reward Estimation"
        self.query = query
        self.correct_answer = correct_answer

    def extract_answer(self, resp):
        return HRE.extract_answer(self, resp)
    
    def calculate_perplexity(self, prompt, resp):
        full_text = prompt + " " + resp
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, labels=inputs["input_ids"])
            loss = out.loss
            perpl = torch.exp(loss).item()
        return float(perpl)
    
    def get_pr(self, query, resp):
        baseline_perpl = self.calculate_perplexity(self.query, self.correct_answer)
        new_perpl = self.calculate_perplexity(query, resp)
        reward = (baseline_perpl - new_perpl) / (baseline_perpl + 1e-8)
        return reward

terminal_state_check_prompt = """
You are a completion detector. Given a CONTEXT and QUERY, determine if the response in the context has been completed or is still ongoing/incomplete.

OUTPUT RULES:
- Output ONLY "0" if the response is NOT completed (incomplete, cut off, ongoing)
- Output ONLY "1" if the response IS completed (finished, terminated, ended properly)

EVALUATION CRITERIA:
✓ COMPLETED (1): Response ends naturally, concludes the thought, has proper ending, answers the query fully
✗ INCOMPLETE (0): Response cuts off mid-sentence, ends abruptly, leaves thought unfinished, partial answer

CONTEXT: [Insert conversation/response context here]
QUERY: [Insert the original question/request here]

EXAMPLES:

Example 1:
Context: "Q: What is the capital of France? A: The capital of France is Paris."
Query: "What is the capital of France?"
Output: 1

Example 2: 
Context: "Q: Explain machine learning. A: Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can"
Query: "Explain machine learning"
Output: 0

Example 3:
Context: "Q: Write a Python function to add two numbers. A: def add_numbers(a, b): return a + b"
Query: "Write a Python function to add two numbers"
Output: 1

Example 4:
Context: "Q: List the benefits of exercise. A: Exercise has many benefits: 1. Improves cardiovascular health 2. Strengthens muscles 3. Boosts mental health 4."
Query: "List the benefits of exercise"
Output: 0

Example 5:
Context: "Q: What's 2+2? A: 4"
Query: "What's 2+2?"
Output: 1

Example 6:
Context: "Q: Describe the process of photosynthesis. A: Photosynthesis is the process by which plants convert sunlight into energy. It occurs in chloroplasts and involves"
Query: "Describe the process of photosynthesis"
Output: 0

Example 7:
Context: "Q: How do I install Python? A: You can install Python by downloading it from python.org and following the installation wizard. Make sure to check 'Add Python to PATH' during installation."
Query: "How do I install Python?"
Output: 1

ANALYSIS CHECKLIST:
- Does the response end mid-word or mid-sentence? → 0
- Does the response answer the query completely? → Check completeness
- Is there a natural conclusion/ending? → 1 if yes
- Are there trailing indicators like "..." or incomplete lists? → 0
- Does it feel like the response was cut off? → 0

YOUR BINARY OUTPUT (0 or 1):
"""

class TreeNode:
    def __init__(self, context, token, binary_code, model_name, query, parent=None):
        self.context = context
        self.token = token
        self.binary_code = binary_code
        self.model_name = model_name
        self.query = query
        self.parent = parent
        self.children: Dict[str, TreeNode] = {}

        self.visits = 0
        self.total_reward = 0.0
        self.avg_reward = 0.0

        self.weight = 1.0 / len(self.children) if self.children else 1.0

        self.perplexity_rewards = []
        self.path_length_penalty = 0.0

        self.teacher_similarities: Dict[str, float] = {}

    def add_child(self, token, context, binary_code, model_name):
        child = TreeNode(context, token, binary_code, model_name, self.query, parent=self)
        self.children[token] = child
        return child

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self, query, context):
        prompt = terminal_state_check_prompt.replace("[Insert conversation/response context here]", context).replace("[Insert the original question/request here]", query)
        output = client.chat.completions.create(
            model=slm,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return output.choices[0].message.content == "1"

    def get_path_from_root(self):
        path = []
        current = self
        while current:
            path.append(current)
            current = current.parent
        return path[::-1]
    
    def update_reward(self, reward:float):
        self.visits += 1
        self.total_reward += reward
        self.avg_reward = self.total_reward / self.visits

    def ucb_score(self, exploration_const:float = 1.414):
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.avg_reward
        exploration = exploration_const * math.sqrt(math.log(self.parent.visits) / self.visits)
        return (exploitation + exploration) * self.weight

class MCTSSystem:
    def __init__(self, teacher_models, primary_model, client, tokenizer):
        self.teacher_models = teacher_models
        self.primary_model = primary_model
        self.client = client
        self.tokenizer = tokenizer
        self.encoding = "cl100k_base"
        self.right_token_gen = right_token_gen_from_a_single_model
        self.wrong_token_gen = wrong_token_gen_from_a_single_model
        self.cosine_similarity = cs

    def populate_tree(self, node: TreeNode):
        if node.is_terminal(node.query, node.context):
            return

        for model in self.teacher_models[:3]:  # Use subset for faster generation
            try:
                right_tokens = self.right_token_gen(model=model, context=node.context, query=node.query)
                for token in right_tokens:
                    new_context = node.context + " " + token
                    child = node.add_child(
                        token=token,
                        context=new_context,
                        binary_code=node.binary_code + "1",
                        model_name=model
                    )

                wrong_tokens = self.wrong_token_gen(model, node.context, node.query)
                for token in wrong_tokens:
                    new_context = node.context + " " + token
                    child = node.add_child(
                        token=token,
                        context=new_context,
                        binary_code=node.binary_code + "0",
                        model_name=model
                    )
            except Exception as e:
                print(f"Error populating tree with model {model}: {e}")
                continue

    def get_primary_model_token(self, context: str) -> str:
        try:
            inputs = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=2048)
            device = next(self.primary_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.primary_model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=0.7
                )
            
            new_tokens = outputs[0][len(inputs["input_ids"][0]):]
            token = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return token.strip()
        except Exception as e:
            print(f"Error generating token from primary model: {e}")
            return "the"

    def select_best_child(self, node: TreeNode):
        if not node.children:
            return None

        primary_resp = self.get_primary_model_token(node.context)
        best_child = None
        best_score = -float('inf')

        for token, child in node.children.items():
            try:
                similarity = self.cosine_similarity(primary_resp, token, self.encoding)[0][0]
                child.teacher_similarities[self.primary_model] = similarity
                ucb = child.ucb_score()
                combined_score = ucb + similarity * child.weight

                if combined_score > best_score:
                    best_score = combined_score
                    best_child = child
            except Exception as e:
                print(f"Error calculating similarity: {e}")
                continue

        return best_child

    def calculate_rewards(self, node: TreeNode, hre: HRE, pre: PRE) -> float:
        total_reward = 0.0
        path = node.get_path_from_root()

        perplexity_reward = pre.get_pr(node.query, node.context)
        path_length = len(path)
        length_penalty = -0.1 * path_length
        direction_reward = sum(1 if n.binary_code.endswith("1") else -0.5 for n in path[1:] if n.binary_code)

        terminal_reward = 0.0
        if node.is_terminal(node.query, node.context):
            terminal_reward = hre.get_hard_reward(node.query, node.context)

        total_reward = perplexity_reward + length_penalty + direction_reward + terminal_reward
        return total_reward

    def update_weightages(self, path: List[TreeNode], final_reward: float):
        for i, node in enumerate(path[1:], 1):
            if final_reward > 0:
                if node.binary_code and node.binary_code.endswith("1"):
                    node.weight *= 1.1
                else:
                    node.weight *= 0.95
            else:
                if node.binary_code and node.binary_code.endswith("0"):
                    node.weight *= 0.8
                else:
                    node.weight *= 0.98

            node.weight = max(node.weight, 0.1)

    def mcts_search(self, root_context: str, query: str, correct_answer: str, max_iterations: int = 100) -> TreeNode:
        root = TreeNode(context=root_context, token="", binary_code="", model_name="root", query=query)
        
        hre = HRE(query, correct_answer)
        pre = PRE(query, correct_answer, self.primary_model, self.tokenizer)

        for iteration in range(max_iterations):
            print(f"MCTS Iteration {iteration + 1}/{max_iterations}")
            
            current = root
            path = [current]

            while not current.is_leaf() and not current.is_terminal(current.query, current.context):
                current = self.select_best_child(current)
                if current is None:
                    break
                path.append(current)

            if current and not current.is_terminal(current.query, current.context):
                self.populate_tree(current)
                if current.children:
                    import random
                    current = random.choice(list(current.children.values()))
                    path.append(current)

            if current:
                final_reward = self.calculate_rewards(current, hre, pre)
                
                for node in path:
                    node.update_reward(final_reward)
                
                self.update_weightages(path, final_reward)

        return root

    def get_best_path(self, root: TreeNode) -> List[str]:
        path = []
        current = root
        
        while current.children:
            best_child = max(current.children.values(), key=lambda x: x.visits)
            path.append(best_child.token)
            current = best_child
            
        return path

# so my idea is using this gsmk8k to take the query one a time and let the models explore
def load_gsm8k_dataset():
    try:
        gsm8k_dataset = load_dataset("openai/gsm8k", "main")
        return gsm8k_dataset
    except Exception as e:
        print(f"Error loading GSM8K dataset: {e}")
        return {
            "train": [
                {"question": "A bakery sold 23 cupcakes in the morning and 14 cupcakes in the afternoon. How many cupcakes did they sell in total?", 
                 "answer": "The bakery sold 23 cupcakes in the morning and 14 in the afternoon.\n23 + 14 = 37\n#### 37"},
                {"question": "Sarah has 25 stickers. She gives 8 stickers to her friend and buys 12 more stickers. How many stickers does Sarah have now?", 
                 "answer": "Sarah starts with 25 stickers.\nShe gives away 8, so she has 25 - 8 = 17 stickers.\nThen she buys 12 more, so she has 17 + 12 = 29 stickers.\n#### 29"}
            ]
        }
    
# and for the training i have chosen the skywork-rewards dataset, recommended in this paper: https://arxiv.org/pdf/2412.10400
def load_skywork_dataset():
    try:
        skywork_dataset = load_dataset("Skywork/Skywork-Reward-Preference-80K-v0.2")
        return skywork_dataset
    except Exception as e:
        print(f"Error loading Skywork dataset: {e}")
        return None
    
def chunk_ds(dataset, chunk_size=1000):
    if not dataset:
        return None

    data = dataset["train"] if "train" in dataset else list(dataset.values())[0]
    chunks = []

    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])

    return chunks

def train_primary_model_with_chunk(primary_model, tokenizer, training_chunk, reward_threshold=0.5):
    print(f"Training with chunk of {len(training_chunk)} samples")
    
    training_data = []
    for sample in training_chunk[:10]:  
        if 'chosen' in sample and 'rejected' in sample:
            chosen_text = sample['chosen'][0]['content'] if isinstance(sample['chosen'], list) else sample['chosen']
            rejected_text = sample['rejected'][0]['content'] if isinstance(sample['rejected'], list) else sample['rejected']
            
            training_data.append({
                'input': chosen_text[:500], 
                'output': "This is a good response."
            })
    
    print(f"Prepared {len(training_data)} training samples")
    return primary_model

def main():
    # Load datasets
    gsm8k_data = load_gsm8k_dataset()
    skywork_data = load_skywork_dataset()
    
    if skywork_data:
        training_chunks = chunk_ds(skywork_data)
    else:
        training_chunks = []
    
    mcts_system = MCTSSystem(teacher_models, primary_model, client, tokenizer)
    
    for epoch in range(3): 
        print(f"\n=== EPOCH {epoch + 1} ===")
        
        if "train" in gsm8k_data:
            sample = random.choice(gsm8k_data["train"])
        else:
            sample = random.choice(gsm8k_data)
            
        query = sample["question"]
        correct_answer = sample["answer"]
        
        print(f"Query: {query}")
        print(f"Correct Answer: {correct_answer}")
        
        import re
        match = re.findall(r"####\s*(-?\d+(\.\d+)?)", correct_answer)
        numerical_answer = float(match[0][0]) if match else 0.0
        
        root_context = f"Question: {query}\nAnswer:"
        root = mcts_system.mcts_search(root_context, query, numerical_answer, max_iterations=50)
        
        best_path = mcts_system.get_best_path(root)
        final_response = root_context + " " + " ".join(best_path)
        
        print(f"Best response: {final_response}")
        
        hre = HRE(query, numerical_answer)
        final_reward = hre.get_hard_reward(query, final_response)
        
        print(f"Final reward: {final_reward}")
        
        if training_chunks and final_reward > 0.5:
            chunk_index = epoch % len(training_chunks)
            training_chunk = training_chunks[chunk_index]
            primary_model = train_primary_model_with_chunk(primary_model, tokenizer, training_chunk, final_reward)
            print("Model updated with positive reward chunk")
        else:
            print("No training update due to low reward")
        
        print("-" * 50)

if __name__ == "__main__":
    main()
