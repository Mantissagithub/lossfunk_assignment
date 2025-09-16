from unsloth import FastLanguageModel
import os, gc, json, re, numpy as np, torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, TrainerCallback, Trainer, TrainingArguments
from unsloth.chat_templates import get_chat_template
from trl import ORPOConfig, ORPOTrainer
from torch.utils.tensorboard import SummaryWriter

#half model config form the orpo script
TRAIN_SIZE = 1000
VAL_SIZE = 200
NUM_EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUMULATION = 4
EVAL_EVERY = 100
EVAL_SIZE = 50

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.set_per_process_memory_fraction(0.95)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(3407)
np.random.seed(3407)

max_seq_length = 1024  
dtype = None
load_in_4bit = True

def load_baseline_model():
    print("🔄 Loading baseline model (Llama-3.2-3B)...")
    baseline_model, baseline_tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3.2-3b-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    baseline_model.eval()
    for param in baseline_model.parameters():
        param.requires_grad = False
    
    print("✅ Baseline model loaded and frozen")
    return baseline_model, baseline_tokenizer

def load_training_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-3-270m-it",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=16,  
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    tokenizer = get_chat_template(tokenizer=tokenizer, chat_template="gemma")
    return model, tokenizer

def preprocess_gsm8k_for_sft(dataset, tokenizer):
    def tokenize_function(examples):
        texts = []
        for question, answer in zip(examples['question'], examples['answer']):
            # Format for Gemma
            text = f"<bos><start_of_turn>user\n{question.strip()}<end_of_turn>\n<start_of_turn>model\n{answer.strip()}<end_of_turn><eos>"
            texts.append(text)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

class HardRewardEvaluator:
    def extract_answer(self, resp):
        # Extract #### answer format
        match = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", resp)
        if match:
            return float(match[-1])
        
        # fallback: last number
        match = re.findall(r"(-?\d+(?:\.\d+)?)", resp)
        if match:
            return float(match[-1])
        return None
    
    def evaluate(self, prompt, resp, correct_answer):
        try:
            pred = self.extract_answer(resp)
            return 1.0 if (pred is not None and abs(pred - correct_answer) < 1e-4) else 0.0
        except:
            return 0.0

class PerplexityRewardEvaluator:
    def __init__(self, baseline_model, baseline_tokenizer, training_model, training_tokenizer):
        self.baseline_model = baseline_model
        self.baseline_tokenizer = baseline_tokenizer
        self.training_model = training_model
        self.training_tokenizer = training_tokenizer
        
    def calculate_perplexity(self, model, tokenizer, prompt, resp):
        full_text = f"Question: {prompt}\nAnswer: {resp}"
        
        inputs = tokenizer(
            full_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_seq_length
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()
        
        return float(perplexity)
    
    def evaluate(self, prompt, resp, correct_answer=None):
        try:
            baseline_perpl = self.calculate_perplexity(
                self.baseline_model, self.baseline_tokenizer, prompt, resp
            )
            training_perpl = self.calculate_perplexity(
                self.training_model, self.training_tokenizer, prompt, resp
            )
            
            # reward = improvement (lower perplexity is better)
            # ff training model has lower perplexity than baseline, positive reward
            reward = (baseline_perpl - training_perpl) / (baseline_perpl + 1e-8)
            
            if correct_answer is not None:
                hard_eval = HardRewardEvaluator()
                if hard_eval.evaluate(prompt, resp, correct_answer) > 0.5:
                    reward += 0.1
            
            return float(np.clip(reward, 0.0, 1.0))
            
        except Exception as e:
            print(f"[PERPLEXITY REWARD] Error: {e}")
            return 0.1

def evaluate_model_accuracy(model, tokenizer, eval_dataset, evaluator):
    correct = 0
    total = 0
    
    model.eval()
    
    for example in eval_dataset:
        prompt = example['question']
        correct_answer = float(re.findall(r"####\s*(-?\d+(?:\.\d+)?)", example['answer'])[-1])
        
        inputs = tokenizer(
            f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        score = evaluator.evaluate(prompt, response, correct_answer)
        correct += score
        total += 1
        
        if total >= EVAL_SIZE:
            break
    
    return correct / total if total > 0 else 0.0

def train_sft_baseline(model, tokenizer, train_dataset, val_dataset):
    print("🚀 Starting SFT Baseline Training...")
    
    train_tokenized = preprocess_gsm8k_for_sft(train_dataset, tokenizer)
    val_tokenized = preprocess_gsm8k_for_sft(val_dataset, tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./sft_baseline_gemma",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        evaluation_strategy="steps",
        eval_steps=EVAL_EVERY,
        logging_steps=50,
        learning_rate=5e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        run_name="sft_baseline_gemma"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    evaluator = HardRewardEvaluator()
    accuracy = evaluate_model_accuracy(model, tokenizer, val_dataset, evaluator)
    print(f"✅ SFT Baseline Final Accuracy: {accuracy:.3f}")
    
    return model, accuracy

def format_gsm8k_for_orpo(dataset):
    samples = []
    for ex in dataset:
        q = ex["question"].strip()
        ans = ex["answer"].strip()
        
        match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", ans)
        if not match:
            continue
            
        float_ans = float(match.group(1))
        
        # Create correct response
        correct_response = f"Let's solve this step by step.\n\n{ans}"
        
        # Create incorrect response (corrupt the final answer)
        wrong_ans = float_ans + np.random.randint(-10, 10)
        if wrong_ans == float_ans:
            wrong_ans += 3
        
        wrong_response = ans.replace(str(float_ans), str(wrong_ans), 1)
        wrong_response = f"Let's solve this step by step.\n\n{wrong_response}"
        
        samples.append({
            "prompt": q,
            "chosen": correct_response,
            "rejected": wrong_response,
            "correct_answer": float_ans,
        })
    
    return samples

class ORPOExperiment:
    def __init__(self, training_model, training_tokenizer, baseline_model, baseline_tokenizer):
        self.training_model = training_model
        self.training_tokenizer = training_tokenizer
        self.baseline_model = baseline_model
        self.baseline_tokenizer = baseline_tokenizer
        self.hard_evaluator = HardRewardEvaluator()
        self.perplexity_evaluator = PerplexityRewardEvaluator(
            baseline_model, baseline_tokenizer, training_model, training_tokenizer
        )
        self.writer = SummaryWriter(f"runs/gemma_orpo_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
    def format_dataset_for_orpo(self, samples, reward_type="hard"):
        formatted = []
        print(f"Formatting {len(samples)} samples for {reward_type} rewards...")
        
        for i, sample in enumerate(samples):
            if i % 50 == 0:
                print(f"Progress: {i}/{len(samples)} ({i/len(samples)*100:.1f}%)")
            
            if reward_type == "hard":
                # Hard rewards: correct gets 1.0, incorrect gets 0.0
                chosen_reward = 1.0
                rejected_reward = 0.0
            else:
                # Perplexity-based rewards
                chosen_reward = self.perplexity_evaluator.evaluate(
                    sample["prompt"], sample["chosen"], sample["correct_answer"]
                )
                rejected_reward = self.perplexity_evaluator.evaluate(
                    sample["prompt"], sample["rejected"], sample["correct_answer"]
                )
            
            formatted.append({
                "prompt": [{"role": "user", "content": sample["prompt"]}],
                "chosen": [{"role": "assistant", "content": sample["chosen"]}],
                "rejected": [{"role": "assistant", "content": sample["rejected"]}],
                "chosen_reward": chosen_reward,
                "rejected_reward": rejected_reward,
            })
        
        return Dataset.from_list(formatted)
    
    def reset_training_model(self):
        model, tokenizer = load_training_model()
        self.training_model = model
        self.training_tokenizer = tokenizer
        self.perplexity_evaluator.training_model = model
        self.perplexity_evaluator.training_tokenizer = tokenizer
        return model
    
    def train_orpo(self, train_samples, val_dataset, reward_type="hard"):
        print(f"Training ORPO with {reward_type} rewards...")
        
        formatted_dataset = self.format_dataset_for_orpo(train_samples, reward_type)
        
        model = self.reset_training_model()
        
        orpo_config = ORPOConfig(
            output_dir=f"./orpo_gemma_{reward_type}_results",
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUMULATION,
            learning_rate=8e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            max_length=max_seq_length,
            beta=0.1,
            logging_steps=50,
            eval_strategy="no",
            save_strategy="no",
            bf16=True,
            report_to=["tensorboard"],
            run_name=f"orpo_gemma_{reward_type}"
        )
        
        trainer = ORPOTrainer(
            model=model,
            args=orpo_config,
            train_dataset=formatted_dataset,
            tokenizer=self.training_tokenizer
        )
        
        metrics = []
        
        class MetricsCallback(TrainerCallback):
            def __init__(self, experiment, reward_type, metrics_list):
                self.experiment = experiment
                self.reward_type = reward_type
                self.metrics_list = metrics_list
                
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if not logs or state.global_step % EVAL_EVERY != 0:
                    return
                
                print(f"Evaluating {self.reward_type} at step {state.global_step}...")
                accuracy = evaluate_model_accuracy(
                    model, self.experiment.training_tokenizer, val_dataset, 
                    self.experiment.hard_evaluator
                )
                
                metric = {
                    "step": state.global_step,
                    "loss": logs.get("train_loss", 0.0),
                    "accuracy": accuracy,
                    "reward_type": self.reward_type
                }
                
                self.metrics_list.append(metric)
                print(f"Step {state.global_step}: Accuracy = {accuracy:.3f}")
                
                self.experiment.writer.add_scalar(f"{self.reward_type}/accuracy", accuracy, state.global_step)
                self.experiment.writer.add_scalar(f"{self.reward_type}/loss", metric["loss"], state.global_step)
        
        trainer.add_callback(MetricsCallback(self, reward_type, metrics))
        
        trainer.train()
        
        final_accuracy = evaluate_model_accuracy(trainer.model, self.training_tokenizer, val_dataset, self.hard_evaluator)
        print(f"{reward_type.upper()} Final Accuracy: {final_accuracy:.3f}")
        
        return metrics, final_accuracy
    
    def visualize_results(self, hard_metrics, perplexity_metrics, sft_accuracy):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        hard_steps = [m["step"] for m in hard_metrics]
        hard_acc = [m["accuracy"] for m in hard_metrics]
        perp_steps = [m["step"] for m in perplexity_metrics]
        perp_acc = [m["accuracy"] for m in perplexity_metrics]
        
        axes[0].plot(hard_steps, hard_acc, 'r-o', label='Hard Rewards', linewidth=2)
        axes[0].plot(perp_steps, perp_acc, 'b-s', label='Perplexity Rewards', linewidth=2)
        axes[0].axhline(y=sft_accuracy, color='g', linestyle='--', label=f'SFT Baseline ({sft_accuracy:.3f})')
        axes[0].set_xlabel('Training Steps')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Gemma-270M: Training Progress Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        methods = ['SFT Baseline', 'Hard Rewards', 'Perplexity Rewards']
        final_accs = [sft_accuracy, hard_acc[-1] if hard_acc else 0, perp_acc[-1] if perp_acc else 0]
        colors = ['green', 'red', 'blue']
        
        bars = axes[1].bar(methods, final_accs, color=colors, alpha=0.7)
        axes[1].set_ylabel('Final Accuracy')
        axes[1].set_title('Final Accuracy Comparison')
        axes[1].set_ylim(0, max(final_accs) * 1.2)
        
        for bar, acc in zip(bars, final_accs):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('gemma_sft_orpo_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    print("Starting Gemma-270M + Llama-3B Baseline Experiment")
    print(f"Configuration: {TRAIN_SIZE} train samples, {VAL_SIZE} val samples, {NUM_EPOCHS} epochs")
    
    print("Loading GSM8K dataset...")
    train_ds_raw = load_dataset("gsm8k", "main")["train"].shuffle(seed=3407).select(range(TRAIN_SIZE)) # seed over here means, the one controlling the randomness of the shuffle
    val_ds_raw = load_dataset("gsm8k", "main")["test"].shuffle(seed=3407).select(range(VAL_SIZE))
    
    print("Loading models...")
    baseline_model, baseline_tokenizer = load_baseline_model()
    training_model, training_tokenizer = load_training_model()
    
    #1. training first with sft
    print("\n" + "="*50)
    print("PHASE 1: SFT BASELINE TRAINING (GEMMA-270M)")
    print("="*50)
    sft_model, sft_accuracy = train_sft_baseline(training_model, training_tokenizer, train_ds_raw, val_ds_raw)
    
    #2. ORPO Experiments
    print("\n" + "="*50)
    print("PHASE 2: ORPO REWARD COMPARISON")
    print("="*50)
    
    train_samples = format_gsm8k_for_orpo(train_ds_raw)
    print(f"📝 Formatted {len(train_samples)} samples for ORPO training")
    
    experiment = ORPOExperiment(training_model, training_tokenizer, baseline_model, baseline_tokenizer)
    
    hard_metrics, hard_final = experiment.train_orpo(train_samples, val_ds_raw, "hard")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    perp_metrics, perp_final = experiment.train_orpo(train_samples, val_ds_raw, "perplexity")
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"SFT Baseline:        {sft_accuracy:.3f}")
    print(f"Hard Rewards:        {hard_final:.3f}")
    print(f"Perplexity Rewards:  {perp_final:.3f}")
    
    best = max(sft_accuracy, hard_final, perp_final)
    if best == sft_accuracy:
        print("Winner: SFT Baseline")
    elif best == hard_final:
        print("Winner: Hard Rewards")
    else:
        print("Winner: Perplexity Rewards")
    
    experiment.visualize_results(hard_metrics, perp_metrics, sft_accuracy)
    
    results = {
        "models": {
            "training": "unsloth/gemma-3-270m-it-GGUF",
            "baseline": "unsloth/llama-3.2-3b-bnb-4bit"
        },
        "sft_accuracy": float(sft_accuracy),
        "hard_final_accuracy": float(hard_final),
        "perplexity_final_accuracy": float(perp_final),
        "hard_metrics": hard_metrics,
        "perplexity_metrics": perp_metrics,
        "config": {
            "train_size": TRAIN_SIZE,
            "val_size": VAL_SIZE,
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE
        }
    }
    
    with open("gemma_orpo_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    experiment.writer.close()
    print("\nExperiment completed! Check 'gemma_sft_orpo_comparison.png' and 'gemma_orpo_results.json'")

if __name__ == "__main__":
    main()
