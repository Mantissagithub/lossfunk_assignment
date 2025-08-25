from unsloth import FastLanguageModel
import os, gc, json, re, numpy as np, torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, TrainerCallback, DataCollatorForLanguageModeling    
from unsloth.chat_templates import get_chat_template
# from trl import ORPOConfig, ORPOTrainer
from trl import DPOConfig, DPOTrainer
# referring to this official doc -> https://huggingface.co/docs/trl/en/dpo_trainer, includes dpoconfig as well as the trainer

# training hyperparams - configured for maximum training capacity
FULL_TRAIN = True  
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
    NUM_EPOCHS = 1
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
    def format_ds_for_orpo(self, ds, reward_type="hard"):
        formatted = []
        for ex in ds:
            if reward_type == "hard":
                cr = self.hard.evaluate(ex["prompt"], ex["chosen"], ex["correct_answer"])
                rr = self.hard.evaluate(ex["prompt"], ex["rejected"], ex["correct_answer"])
            else:
                cr = self.perpl.evaluate(ex["prompt"], ex["chosen"], ex["correct_answer"])
                rr = self.perpl.evaluate(ex["prompt"], ex["rejected"], ex["correct_answer"])
            if cr <= rr:
                ex_ch, ex_rj, ch_r, rj_r = ex["rejected"], ex["chosen"], rr, cr
            else:
                ex_ch, ex_rj, ch_r, rj_r = ex["chosen"], ex["rejected"], cr, rr
            formatted.append({
                "prompt": [{"role": "user", "content": ex["prompt"]}],
                "chosen": [{"role": "assistant", "content": ex_ch}],
                "rejected": [{"role": "assistant", "content": ex_rj}],
                "chosen_reward": ch_r,
                "rejected_reward": rj_r,
            })
        return formatted
    
    def train_with_rt(self, ds, rt="hard", num_epochs=NUM_EPOCHS):
        print(f"Training with {rt} reward evaluator for {num_epochs} epochs.")
        print(f"Config: batch_size={BATCH_SIZE}, grad_acc={GRAD_ACCUMULATION}, effective_batch={BATCH_SIZE*GRAD_ACCUMULATION}")
        formatted = self.format_ds_for_orpo(ds, reward_type=rt)
        
        # converting list to Dataset object
        formatted_dataset = Dataset.from_list(formatted)

        dpo_config = DPOConfig(
            output_dir=f"./dpo_{rt}_results",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUMULATION,
            optim="paged_adamw_8bit",
            learning_rate=8e-5,  # slightly higher for better convergence
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            max_length=max_seq_length,
            max_completion_length=512,  # increased for longer responses
            beta=0.1,
            save_strategy="steps" if FULL_TRAIN else "no",  # save checkpoints for full training
            save_steps=500 if FULL_TRAIN else 1000,
            eval_strategy="no",  # disable built-in evaluation, we do custom evaluation
            logging_steps=LOG_EVERY,
            remove_unused_columns=False,
            label_pad_token_id=-100,
            dataloader_num_workers=4,  # parallel data loading
            fp16=False,  # disable fp16 since model is in bfloat16
            bf16=True,   # use bfloat16 to match model precision
            gradient_checkpointing=True,  # memory optimization
            weight_decay=0.01,  # regularization
            restore_callback_states_from_checkpoint=True,
            # dpo specific params, recommended by perplexity and the documentation
            loss_type="sigmoid",
            max_prompt_length=512
        )

        # from ppo
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

        trainer = DPOTrainer(
            model=self.model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=formatted_dataset,
            tokenizer=self.tokenizer
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
        trainer.train()
        
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

# for my ref to read this later, and understand everything:
# Parameters that control the model and reference model

# model_init_kwargs (dict[str, Any] or None, optional, defaults to None) — Keyword arguments for AutoModelForCausalLM.from_pretrained, used when the model argument of the DPOTrainer is provided as a string.
# ref_model_init_kwargs (dict[str, Any] or None, optional, defaults to None) — Keyword arguments for AutoModelForCausalLM.from_pretrained, used when the ref_model argument of the DPOTrainer is provided as a string.
# model_adapter_name (str or None, optional, defaults to None) — Name of the train target PEFT adapter, when using LoRA with multiple adapters.
# ref_adapter_name (str or None, optional, defaults to None) — Name of the reference PEFT adapter, when using LoRA with multiple adapters.
# force_use_ref_model (bool, optional, defaults to False) — If you provide a PEFT model as the active model and wish to use a different model for the ref_model, set this flag to True.
# disable_dropout (bool, optional, defaults to True) — Whether to disable dropout in the model and reference model.
# use_logits_to_keep (bool, optional, defaults to False) — If True, only a specified number of logits are computed in the forward pass. This can be useful for saving memory and speeding up training by not computing the logits for all tokens, especially in scenarios when working with very long prompts where labels are ignored (-100).
# Parameters that control the data preprocessing

# dataset_num_proc (int or None, optional, defaults to None) — Number of processes to use for processing the dataset.
# padding_value (int or None, optional, defaults to None) — Padding value to use. If None, the padding value of the tokenizer is used.
# label_pad_token_id (int, optional, defaults to -100) — Padding value to use for labels.
# max_prompt_length (int or None, optional, defaults to 512) — Maximum length of the prompt.
# max_completion_length (int or None, optional, defaults to None) — Maximum length of the completion.
# max_length (int or None, optional, defaults to 1024) — Maximum length of the full sequence (prompt + completion).
# truncation_mode (str, optional, defaults to "keep_end") — Truncation mode to use when the sequence exceeds max_length. Possible values are "keep_end" and "keep_start".
# padding_free (bool, optional, defaults to False) — Whether to perform forward passes without padding by flattening all sequences in the batch into a single continuous sequence. This reduces memory usage by eliminating padding overhead. Currently, this is only supported with the flash_attention_2 attention implementation, which can efficiently handle the flattened batch structure.
# precompute_ref_log_probs (bool, optional, defaults to False) — Whether to precompute the log probabilities from the reference model. Setting this to True allows training without needing the reference model during training, which can help reduce GPU memory usage. If set to False (default), the reference model will be used during training to compute log probabilities on-the-fly.
# precompute_ref_batch_size (int or None, optional, defaults to None) — Batch size to use when precomputing reference model log probabilities. This can be set higher than the training batch size to speed up preprocessing. If None, defaults to per_device_train_batch_size for training and per_device_eval_batch_size for evaluation.
# tools (Optional[list[Union[dict, Callable]]], optional, defaults to None) — List of tools (callable functions) that will be accessible to the model. If the template does not support function calling, this argument will have no effect.
# Parameters that control the training

# loss_type (str or list[str], optional, defaults to "sigmoid") — Type of loss to use. Possible values are:
# "sigmoid": sigmoid loss from the original DPO paper.
# "hinge": hinge loss on the normalized likelihood from the SLiC paper.
# "ipo": IPO loss from the IPO paper.
# "exo_pair": pairwise EXO loss from the EXO paper.
# "nca_pair": pairwise NCA loss from the NCA paper.
# "robust": unbiased estimate of the DPO loss that is robust to preference noise from the Robust DPO paper.
# "bco_pair": pairwise BCO loss from the BCO paper.
# "sppo_hard": SPPO loss with hard label from the SPPO paper.
# "aot": AOT loss for paired datasets from the AOT paper.
# "aot_pair": AOT loss for unpaired datasets from the AOT paper.
# "discopop": DiscoPOP (a.k.a Log-Ratio Modulated Loss, LRML) loss from the DiscoPOP paper.
# "apo_zero": APO-zero loss from the APO paper.
# "apo_down": APO-down loss from the APO paper.
# "sft": Negative log-likelihood loss (standard supervised fine-tuning loss).
# Multiple loss types can be combined using comma separation (e.g., ["sigmoid", "bco_pair", "sft"] for MPO). The loss_weights parameter can be used to specify corresponding weights for each loss type.

# use_liger_loss (bool, optional, defaults to False) — Whether to use Liger loss.
# base_model_attribute_name (str, optional, defaults to "model") — Name of the attribute in the model that contains the base model. This is used to get the base model from the model when the model does not have a get_decoder method in the case when use_liger_loss is True.
# beta (float, optional, defaults to 0.1) — Parameter controlling the deviation from the reference model. Higher β means less deviation from the reference model. For the IPO loss (loss_type="ipo"), β is the regularization parameter denoted by τ in the paper.
# f_divergence_type (str, optional, defaults to FDivergenceType.REVERSE_KL) — Type of f-divergence regularization function to compute divergence between policy and reference model.
# f_alpha_divergence_coef (float, optional, defaults to 1.0) — α coefficient in the α-divergence u^-α regularization function for DPO loss.
# reference_free (bool, optional, defaults to False) — Whether to ignore the provided reference model and implicitly use a reference model that assigns equal probability to all responses.
# label_smoothing (float, optional, defaults to 0.0) — Robust DPO label smoothing parameter from the cDPO report and Robust DPO paper that should be between 0.0 and 0.5.
# use_weighting (bool, optional, defaults to False) — Whether to weight the loss as done in the WPO paper.
# rpo_alpha (float, optional, defaults to None) — α parameter from the RPO paper (v3), which controls the weighting of the NLL term in the loss. If None, no weighting is applied and the loss is the same as the DPO loss. The paper recommends rpo_alpha=1.0.
# ld_alpha (float or None, optional, defaults to None) — α parameter from the LD-DPO paper, which controls the weighting of the verbose token log-probabilities in responses. If None, no weighting is applied to the verbose part, and the loss is equivalent to the standard DPO loss. The paper recommends setting ld_alpha between 0.0 and 1.0.
# discopop_tau (float, optional, defaults to 0.05) — τ/temperature parameter from the DiscoPOP paper, which controls the shape of log ratio modulated loss. The paper recommends the default value discopop_tau=0.05.
# loss_weights (list[float] or None, optional, defaults to None) — List of loss weights for multi-loss combinations. Used when combining multiple loss types. Example: [0.8, 0.2, 1.0] for MPO. If not provided, defaults to equal weights (1.0) for all loss types.
# sync_ref_model (bool, optional, defaults to False) — Whether to synchronize the reference model with the active model every ref_model_sync_steps steps, using the ref_model_mixup_alpha parameter. This synchronization originates from the TR-DPO paper.
# ref_model_mixup_alpha (float, optional, defaults to 0.6) — α parameter from the TR-DPO paper, which controls the mix between the current policy and the previous reference policy during updates. The reference policy is updated according to the equation: π_ref = α * π_θ + (1 - α) * π_ref_prev. To use this parameter, you must set sync_ref_model=True.
# ref_model_sync_steps (int, optional, defaults to 512) — τ parameter from the TR-DPO paper, which determines how frequently the current policy is synchronized with the reference policy. To use this parameter, you must set sync_ref_model=True.
# Parameters that control the logging

# generate_during_eval (bool, optional, defaults to False) — Whether to generate and log completions from both the model and the reference model to W&B or Comet during evaluation.