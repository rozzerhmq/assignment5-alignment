import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import wandb
import os

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
NEW_MODEL_NAME = "Qwen-Math-1.5B-SFT-Adapter"
WANDB_PROJECT = "qwen-math-sft"
OUTPUT_DIR = "./results"

# 1. Initialize WandB
wandb.init(project=WANDB_PROJECT, name="run-qwen-1.5b-qlora")

# 2. Quantization Config (The "4-bit" magic)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # Normalized Float 4 (optimized for weights)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16 for speed
    bnb_4bit_use_double_quant=True,  # Compress the quantization constants
)

# 3. Load Base Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False  # Silence warnings during training

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Qwen has no pad token by default

# 4. LoRA Configuration
# We target "all-linear" layers for maximum performance.
peft_config = LoraConfig(
    r=64,  # Rank (Higher = more parameters to train)
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",  # Targets q_proj, k_proj, v_proj, o_proj, etc.
)

# 5. Dataset (Example: Using a small math reasoning subset)
# In reality, replace this with your own JSONL file: data_files="my_data.jsonl"
dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train[:500]")

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,  # Increase to 8 or 16 if VRAM allows
    gradient_accumulation_steps=4,  # Simulates batch size of 4*4=16
    learning_rate=2e-4,
    logging_steps=1,
    max_steps=50,  # Short run for demo (remove for full training)
    save_strategy="no",  # Don't save checkpoints every step
    optim="paged_adamw_8bit",  # Saves memory!
    fp16=False,  # Use BF16 (Amere/Ada GPUs support this)
    bf16=True,
    report_to="wandb",  # Crucial: Logs metrics to WandB
)

# 7. Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="question",  # The column name in your dataset
    max_seq_length=2048,
    tokenizer=tokenizer,
    args=training_args,
    packing=False,
)

print("Starting training...")
trainer.train()

print(f"Saving adapter to {NEW_MODEL_NAME}...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)
wandb.finish()
