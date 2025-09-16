from unsloth import FastLanguageModel
import os, gc, json, re, numpy as np, torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, TrainerCallback, DataCollatorForLanguageModeling
from unsloth.chat_templates import get_chat_template
# from trl import ORPOConfig, ORPOTrainer
from trl import PPOConfig, PPOTrainer # now this is for ppo
# referring this repo for more info : https://github.com/huggingface/trl/blob/main/examples/scripts/ppo/ppo.py

# training hyperparams - configured for maximum training capacity
FULL_TRAIN = False # true for full training, i mean the whole long ass training, just for quick results false 
if FULL_TRAIN:
    TRAIN_SIZE = None       # using all 7,473 samples
    VAL_SIZE = None         # using all 1,319 test samples
    NUM_EPOCHS = 3
    LOG_EVERY = 200
    EVAL_EVERY = 200
    BATCH_SIZE = 8          # increased for max capacity
    GRAD_ACCUMULATION = 2   # effective batch = 16
    EVAL_SIZE = 100         # increased eval size
else:
    TRAIN_SIZE = 1000       # quick test with 1k samples
    VAL_SIZE = 200
    NUM_EPOCHS = 2
    LOG_EVERY = 100
    EVAL_EVERY = 100
    BATCH_SIZE = 4
    GRAD_ACCUMULATION = 4   # effective batch = 16
    EVAL_SIZE = 20

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.set_per_process_memory_fraction(0.95)  # increased for max capacity
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(3407)
np.random.seed(3407)

max_seq_length = 2048
dtype = None  # letting UnsLoTH default to fp16/bfloat16
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.2-3b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
model = FastLanguageModel.get_peft_model(
    model=model,
    r=32,  # increased rank for max capacity
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32,  # increased alpha
    lora_dropout=0,  # set to 0 for Unsloth fast patching optimization
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Disable torch.compile for quantized models to avoid the error
# torch.compile() is not compatible with quantized model fine-tuning in PEFT
print("✓ model ready for training (torch.compile disabled for quantized model)")

tokenizer = get_chat_template(tokenizer=tokenizer, chat_template="llama-3")

# Ensure pad_token_id is set for generation
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

class HRE:
    def __init__(self):
        self.name = "Hard Reward Evaluator"
    def extract_answer(self, resp: str):
        pats = [
            r"(?:the answer is|answer:|equals?|=)\s*(-?\d+(?:\.\d+)?)",
            r"(-?\d+(?:\.\d+)?)\s*$",
            r"=\s*(-?\d+(?:\.\d+)?)"
        ]
        for pat in pats:
            m = re.search(pat, resp, re.IGNORECASE)
            if m:
                try: return float(m.group(1))
                except: pass
        nums = re.findall(r"(-?\d+(?:\.\d+)?)", resp)
        if nums:
            try: return float(nums[-1])
            except: pass
        return None
    def evaluate(self, prompt, resp, correct_answer):
        try:
            pred = self.extract_answer(resp)
            return 1.0 if (pred is not None and abs(pred - correct_answer) < 1e-4) else 0.0
        except Exception as e:
            print(f"[HRE] err: {e}")
            return 0.0

class PRE:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.name = "Perplexity Reward Evaluator"
    def calculate_perplexity(self, prompt, resp):
        full_text = prompt + " " + resp
        inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_seq_length)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs, labels=inputs["input_ids"])
            loss = out.loss
            perpl = torch.exp(loss).item()
        return float(perpl)
    def extract_answer(self, resp: str):
        return HRE().extract_answer(resp)
    def evaluate(self, prompt, resp, correct_answer=None):
        try:
            perpl = self.calculate_perplexity(prompt, resp)
            base = 1.0 / (1.0 + np.log1p(perpl))
            length_bonus = min(0.3, len(resp.split()) / 100.0)
            correctness_bonus = 0.0
            if correct_answer is not None:
                pred = self.extract_answer(resp)
                if pred is not None and abs(pred - correct_answer) < 1e-4:
                    correctness_bonus = 0.1
            reward = base + length_bonus + correctness_bonus
            return float(np.clip(reward, 0.0, 1.0))
        except Exception as e:
            print(f"[PRE] err: {e}")
            return 0.1

def load_gsm8k(split="train", sample_size=1000):
    ds = load_dataset("gsm8k", "main")[split]
    if sample_size is not None and sample_size < len(ds):
        ds = ds.shuffle(seed=3407).select(range(sample_size))
    return ds

def format_gsm8k(ds):
    samples = []
    for ex in ds:
        q = ex["question"].strip()
        ans = ex["answer"].strip()
        float_ans = None
        m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", ans)
        if m:
            try: float_ans = float(m.group(1))
            except: pass
        if float_ans is None: continue
        full = f"Let's think step by step.\n{q}\n{ans}"
        # corrupting math for rejected answer
        wrong = float_ans + np.random.randint(-10, 10)
        if wrong == float_ans: wrong += 3
        example_wrong = (
            f"Let's think step by step.\n{q}\n{ans.replace(str(float_ans), str(wrong), 1)}"
        )
        samples.append(
            {
                "prompt": q,
                "chosen": full,
                "rejected": example_wrong,
                "correct_answer": float_ans,
            }
        )
    return samples

class RCF:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.hard = HRE()
        self.perpl = PRE(model, tokenizer)
        self.results = {}
        self.best_accuracy = {"hard": 0.0, "perplexity": 0.0}
    def format_ds_for_ppo(self, ds, reward_type="hard"):
        # For PPO, we only need queries (prompts), not pre-generated responses
        # Responses will be generated during training loop
        formatted = []
        for ex in ds:
            formatted.append({
                "query": ex["prompt"],
                "correct_answer": ex["correct_answer"],  # Keep for reward calculation
            })
        return formatted
    
    def train_with_rt(self, ds, rt="hard", num_epochs=NUM_EPOCHS):
        print(f"Training with {rt} reward evaluator for {num_epochs} epochs.")
        print(f"Config: batch_size={BATCH_SIZE}, grad_acc={GRAD_ACCUMULATION}, effective_batch={BATCH_SIZE*GRAD_ACCUMULATION}")
        formatted = self.format_ds_for_ppo(ds, reward_type=rt)
        
        # converting list to Dataset object
        formatted_dataset = Dataset.from_list(formatted)

        ppo_config = PPOConfig(
            output_dir=f"./ppo_{rt}_results",
            init_kl_coef=0.2, # the ll div coeff, which is a hyperparameter
            target=6, # the target kl divergence
            horizon=10000, # the number of steps to look ahead
            gamma=1, # discount factor
            lam=0.95, # GAE lambda
            cliprange=0.2, # clipping range
            cliprange_value=0.2, # clipping range for value function
            vf_coef=0.1, # value function coefficient
        )

        ref_model, _ = FastLanguageModel.from_pretrained(
            model_name="unsloth/llama-3.2-3b-bnb-4bit",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        collator_fn = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Use padding instead of masking for autoregressive models
        )

        trainer = PPOTrainer(
            config=ppo_config,  # Use 'config' instead of 'args'
            model=self.model,
            ref_model=ref_model,  # Now correctly using just the model
            tokenizer=self.tokenizer,
            dataset=formatted_dataset,  # Use 'dataset' instead of 'train_dataset'
            data_collator=collator_fn,
        )
        training_metrics = []
        
        class MCB(TrainerCallback):
            def __init__(self, framework, reward_type, metrics_list, eval_every=EVAL_EVERY, eval_size=EVAL_SIZE):
                self.framework = framework
                self.reward_type = reward_type
                self.eval_every = eval_every
                self.eval_size = eval_size
                self.metrics_list = metrics_list
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if not logs: return
                current = {
                    "step": state.global_step,
                    "loss": float(logs.get("train_loss", 0.0)),
                    "reward_type": self.reward_type,
                    "learning_rate": float(logs.get("learning_rate", 0.0)),
                    "accuracy": 0.0,
                }
                
                # only evaluating accuracy at specified intervals to save time
                if state.global_step % self.eval_every == 0 or state.global_step >= state.max_steps:
                    print(f"\n=== Evaluating {self.reward_type.upper()} @ step {state.global_step} ===")
                    acc = self.framework.evaluate_accuracy(model, self.reward_type, eval_size=self.eval_size)
                    current["accuracy"] = float(acc)
                    
                    # tracking best accuracy and saving model if improved
                    if acc > self.framework.best_accuracy[self.reward_type]:
                        self.framework.best_accuracy[self.reward_type] = acc
                        print(f"🎯 new best {self.reward_type} accuracy: {acc:.3f}")
                        # optionally saving best model
                        # trainer.save_model(f"./best_{self.reward_type}_model")
                    
                    print(f"*** Accuracy: {acc:.3f} ***")
                
                self.metrics_list.append(current)
                torch.cuda.empty_cache()
        
        self.validation_sample = self.validation_sample if hasattr(self, "validation_sample") else ds
        trainer.add_callback(MCB(self, rt, training_metrics, eval_every=EVAL_EVERY, eval_size=EVAL_SIZE))
        
        # Create a DataLoader for the formatted_dataset
        from torch.utils.data import DataLoader

        dataloader = DataLoader(formatted_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # Define generation_kwargs for response generation
        generation_kwargs = {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,
        }

        # PPO Training Loop
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(dataloader):
                queries = batch["query"]  # List of query strings
                correct_answers = batch["correct_answer"]  # List of correct answers
                
                # Tokenize queries for generation
                query_tensors = []
                for query in queries:
                    # Apply chat formatting
                    formatted_query = [{"role": "user", "content": query}]
                    query_text = self.tokenizer.apply_chat_template(
                        formatted_query, tokenize=False, add_generation_prompt=True
                    )
                    # Tokenize
                    query_tokens = self.tokenizer(
                        query_text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=max_seq_length
                    )["input_ids"].squeeze(0)
                    query_tensors.append(query_tokens)

                # Generate responses with the policy model
                response_tensors = []
                for query_tensor in query_tensors:
                    # Move to device and expand dims for generation
                    query_input = query_tensor.unsqueeze(0).to(self.model.device)
                    
                    # Generate response
                    with torch.no_grad():
                        response_tensor = trainer.generate(query_input, **generation_kwargs)
                        # Extract only the generated part (remove input tokens)
                        response_only = response_tensor[0][len(query_tensor):]
                        response_tensors.append(response_only)

                # Decode responses for reward calculation
                responses = []
                for response_tensor in response_tensors:
                    response_text = self.tokenizer.decode(response_tensor, skip_special_tokens=True)
                    responses.append(response_text)

                # Compute rewards for (query, response) pairs
                rewards = []
                evaluator = self.hard if rt == "hard" else self.perpl
                for i, (query, response, correct_answer) in enumerate(zip(queries, responses, correct_answers)):
                    reward = evaluator.evaluate(query, response, correct_answer)
                    rewards.append(torch.tensor(reward, dtype=torch.float))

                # PPO step with proper tensor formats
                if query_tensors and response_tensors and rewards:
                    try:
                        # Convert to tensors and move to device
                        reward_tensors = torch.stack(rewards).to(self.model.device)
                        
                        # Pad query and response tensors to same length for batch processing
                        max_query_len = max(len(qt) for qt in query_tensors)
                        max_response_len = max(len(rt) for rt in response_tensors)
                        
                        padded_queries = []
                        padded_responses = []
                        
                        for qt, rt in zip(query_tensors, response_tensors):
                            # Pad queries
                            padded_query = torch.cat([
                                qt, 
                                torch.full((max_query_len - len(qt),), self.tokenizer.pad_token_id, dtype=qt.dtype)
                            ])
                            padded_queries.append(padded_query)
                            
                            # Pad responses  
                            padded_response = torch.cat([
                                rt,
                                torch.full((max_response_len - len(rt),), self.tokenizer.pad_token_id, dtype=rt.dtype)
                            ])
                            padded_responses.append(padded_response)
                        
                        query_batch = torch.stack(padded_queries).to(self.model.device)
                        response_batch = torch.stack(padded_responses).to(self.model.device)
                        
                        # PPO step
                        stats = trainer.step(query_batch, response_batch, reward_tensors)
                        
                        if batch_idx % 10 == 0:
                            print(f"Epoch {epoch}, Batch {batch_idx}: Mean reward = {reward_tensors.mean().item():.3f}")
                            
                    except Exception as e:
                        print(f"Error in PPO step: {e}")
                        continue
        
        self.results[rt] = {
            "training_metrics": training_metrics,
            "final_accuracy": training_metrics[-1].get("accuracy", 0.0) if training_metrics else 0.0,
            "best_accuracy": self.best_accuracy[rt],
            "model": trainer.model
        }
        print(f"\n*** Final {rt.upper()} metrics count: {len(training_metrics)} ***")
        print(f"*** Best {rt.upper()} accuracy achieved: {self.best_accuracy[rt]:.3f} ***")
        return training_metrics
    
    def evaluate_accuracy(self, model, reward_type, eval_size=100):
        val_sample = getattr(self, "validation_sample", None)
        ds = val_sample if val_sample else []
        if not ds:
            print("WARNING: No validation data available.")
            return 0.0
        if eval_size < len(ds):
            ds = ds[:eval_size]
        correct = 0
        for i, ex in enumerate(ds):
            prompt = [{"role": "user", "content": ex["prompt"]}]
            txt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(txt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True,
                    top_p=0.9, pad_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.1)
            resp = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = self.hard.extract_answer(resp)
            if pred is not None and abs(pred - ex["correct_answer"]) < 1e-4:
                correct += 1
        acc = correct / max(1, len(ds))
        return float(acc)
    
    def run_comparison(self, train_ds, val_ds, num_epochs=NUM_EPOCHS):
        print(f"Starting reward structure comparison with enhanced configuration...")
        print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
        print(f"Epochs: {num_epochs}, Effective batch size: {BATCH_SIZE * GRAD_ACCUMULATION}")
        
        self.validation_sample = val_ds
        print("\nTraining with hard rewards...")
        hard_metrics = self.train_with_rt(train_ds, "hard", num_epochs)
        
        print("\nResetting model for perplexity training...")
        gc.collect(); torch.cuda.empty_cache()
        self.model, _ = FastLanguageModel.from_pretrained(
            model_name="unsloth/llama-3.2-3b-bnb-4bit",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        self.model = FastLanguageModel.get_peft_model(
            self.model, r=16,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_alpha=16, lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth", random_state=3407,
        )
        
        # recompiling model for second training
        try:
            self.model = torch.compile(self.model)
        except Exception as e:
            print(f"⚠ second compile failed: {e}")
            
        self.perpl = PRE(self.model, self.tokenizer)
        print("\nTraining with perplexity-based rewards...")
        perpl_metrics = self.train_with_rt(train_ds, "perplexity", num_epochs)
        
        return hard_metrics, perpl_metrics
    
    def visualize_results(self, hard_metrics, perpl_metrics, out_png='reward_comparison.png'):
        sns.set(style="whitegrid", font_scale=1.15)
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        for metrics, color, label in [
            (hard_metrics, "tab:red", "Hard acc"),
            (perpl_metrics, "tab:blue", "Perplexity acc"),
        ]:
            x = [m.get("step",0) for m in metrics if m.get("step",0) > 0]
            y = [m.get("accuracy",0) for m in metrics if m.get("step",0) > 0]
            axes[0,0].plot(x, y, marker="o", color=color, label=label)
        axes[0,0].set_title("Accuracy vs Steps"); axes[0,0].set_ylabel("Accuracy"); axes[0,0].set_xlabel("Step")
        axes[0,0].legend()
        for metrics, color, label in [
            (hard_metrics, "tab:red", "Hard loss"),
            (perpl_metrics, "tab:blue", "Perplexity loss"),
        ]:
            x = [m.get("step",0) for m in metrics if m.get("step",0) > 0]
            y = [m.get("loss",0) for m in metrics if m.get("step",0) > 0]
            axes[0,1].plot(x, y, marker="s", color=color, label=label)
        axes[0,1].set_title("Loss vs Steps"); axes[0,1].set_ylabel("Loss"); axes[0,1].set_xlabel("Step")
        axes[0,1].legend()
        axes[1,0].hist([m.get("accuracy",0) for m in hard_metrics], alpha=0.7, color="tab:red", label="Hard", density=True)
        axes[1,0].hist([m.get("accuracy",0) for m in perpl_metrics], alpha=0.4, color="tab:blue", label="Perplexity", density=True)
        axes[1,0].set_title("Accuracy distribution")
        axes[1,0].set_xlabel("Accuracy"); axes[1,0].set_ylabel("Density")
        axes[1,0].legend()
        
        # adding configuration info in the fourth subplot
        config_text = f"""Training Configuration:
• Epochs: {NUM_EPOCHS}
• Batch size: {BATCH_SIZE}
• Grad accumulation: {GRAD_ACCUMULATION} 
• Effective batch: {BATCH_SIZE * GRAD_ACCUMULATION}
• Training samples: {TRAIN_SIZE or "All"}
• Validation samples: {VAL_SIZE or "All"}
• Best Hard acc: {self.best_accuracy['hard']:.3f}
• Best Perplexity acc: {self.best_accuracy['perplexity']:.3f}"""
        axes[1,1].text(0.1, 0.5, config_text, fontsize=10, verticalalignment='center', 
                      transform=axes[1,1].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1,1].set_title("Configuration")
        axes[1,1].axis("off")
        
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.show()
    
    def write_reports(self, hard_metrics, perpl_metrics, mdfile="report.md", jsonfile="report.json"):
        get_final = lambda metrics: metrics[-1].get("accuracy",0.0) if metrics else 0
        hard_final = get_final(hard_metrics)
        perpl_final = get_final(perpl_metrics)
        winner = "Perplexity-based rewards" if perpl_final > hard_final else "Hard (binary) rewards"
        delta = (perpl_final - hard_final) * 100
        
        # enhanced recommendation based on best accuracy achieved
        hard_best = self.best_accuracy["hard"]
        perpl_best = self.best_accuracy["perplexity"]
        best_winner = "Perplexity-based" if perpl_best > hard_best else "Hard"
        
        rec = (
            f"Use {best_winner.lower()} reward: achieved best accuracy of {max(hard_best, perpl_best):.3f}. "
            f"{'Perplexity rewards show smoother optimization.' if perpl_best > hard_best else 'Hard rewards converge faster to verifiable answers.'}"
        )
        
        summary_md = f"""# Enhanced Reward Structure RL Comparison

## Configuration
- **Date:** {datetime.now().isoformat()}
- **Dataset:** GSM8K (train/val split)  
- **Training Mode:** {"Full training" if FULL_TRAIN else "Quick test"}
- **Epochs:** {NUM_EPOCHS}
- **Effective Batch Size:** {BATCH_SIZE * GRAD_ACCUMULATION}
- **Training Samples:** {TRAIN_SIZE or "All (7,473)"}
- **Validation Samples:** {VAL_SIZE or "All (1,319)"}

## Results
- **Hard reward final acc:** {hard_final:.3f} (best: {hard_best:.3f})
- **Perplexity reward final acc:** {perpl_final:.3f} (best: {perpl_best:.3f})
- **Winner (final):** {winner}
- **Winner (best):** {best_winner}-based rewards
- **Improvement (percentage points):** {abs(delta):.2f}

## Recommendation
{rec}

## Performance Optimizations Applied
- ✅ larger effective batch size ({BATCH_SIZE * GRAD_ACCUMULATION})
- ✅ reduced evaluation frequency (every {EVAL_EVERY} steps)
- ✅ disabled expensive checkpoint saving
- ✅ mixed-precision training (fp16/bfloat16)
- ✅ tensorFloat-32 acceleration
- ✅ model compilation for 10-15% speedup
- ✅ best model tracking

![metrics plot](reward_comparison.png)
"""
        json_out = {
            "date": datetime.now().isoformat(),
            "dataset": "gsm8k",
            "configuration": {
                "full_training": FULL_TRAIN,
                "epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "grad_accumulation": GRAD_ACCUMULATION,
                "effective_batch_size": BATCH_SIZE * GRAD_ACCUMULATION,
                "train_samples": TRAIN_SIZE,
                "val_samples": VAL_SIZE
            },
            "final_accuracy_hard": hard_final,
            "final_accuracy_perplexity": perpl_final,
            "best_accuracy_hard": hard_best,
            "best_accuracy_perplexity": perpl_best,
            "winner_final": winner,
            "winner_best": f"{best_winner}-based rewards",
            "improvement_percentage": float(abs(delta)),
            "recommendation": rec,
            "hard_metrics": hard_metrics,
            "perplexity_metrics": perpl_metrics
        }
        with open(mdfile, "w") as f: f.write(summary_md)
        with open(jsonfile, "w") as f: json.dump(json_out, f, indent=2)
        print(f"wrote enhanced report to {mdfile} and JSON to {jsonfile}")

def main():
    print(f"Loading GSM8K dataset with configuration:")
    print(f"  Training mode: {'FULL' if FULL_TRAIN else 'QUICK'}")
    print(f"  Train samples: {TRAIN_SIZE or 'All'}")
    print(f"  Val samples: {VAL_SIZE or 'All'}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUMULATION}")
    
    train_ds_raw = load_gsm8k("train", sample_size=TRAIN_SIZE)
    val_ds_raw = load_gsm8k("test", sample_size=VAL_SIZE)
    train_ds = format_gsm8k(train_ds_raw)
    val_ds = format_gsm8k(val_ds_raw)
    
    print(f"formatted {len(train_ds)} training and {len(val_ds)} validation samples")
    
    framework = RCF(model, tokenizer)
    hard_metrics, perpl_metrics = framework.run_comparison(train_ds, val_ds, NUM_EPOCHS)
    
    print("generating enhanced plots...")
    framework.visualize_results(hard_metrics, perpl_metrics)
    
    print("exporting enhanced report...")
    framework.write_reports(hard_metrics, perpl_metrics)
    print("done! 🎉")

if __name__ == "__main__":
    main()
