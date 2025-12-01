import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from cs336_alignment.prompt_utils import format_r1_zero_example
import os
import argparse

# --- Configuration ---
# Qwen 2.5 32B fits on a 3090 (24GB) in 4-bit quantization.
# Ideally we would use a Math model, but 32B only comes in Instruct/Base.
MODEL_NAME = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
NEW_MODEL_NAME = "qwen3-14b-math-sft-adapter"
OUTPUT_DIR = "checkpoints/qwen3-14b-math-sft-temp"
MAX_SEQ_LENGTH = 2048
DTYPE = (
    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)  # Set DTYPE explicitly based on BF16 support
LOAD_IN_4BIT = True  # Use 4bit quantization to reduce memory usage.


def train(args):
    # 1. Load Model and Tokenizer using Unsloth
    print(f"Loading Unsloth model: {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # 2. Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Rank
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    # 3. Load and Format Dataset
    print("Loading and formatting dataset...")
    data_dir = "data/gsm8k"
    dataset = load_dataset("json", data_dir=data_dir, split="train")

    if args.limit:
        print(f"Limiting dataset to first {args.limit} examples for testing.")
        dataset = dataset.select(range(args.limit))

    def formatting_prompts_func(examples):
        questions = examples["question"]
        answers = examples["answer"]
        texts = []
        for q, a in zip(questions, answers):
            # Use the project's standardized formatting
            prompt, answer = format_r1_zero_example(q, a)
            # SFT training involves calculating loss on the full sequence.
            # We add the EOS token to ensure the model learns to stop.
            text = prompt + answer + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}

    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

    # 4. Training Arguments
    # 32B Model on 24GB VRAM requires strict memory management.
    # Batch size 1 + Gradient Accumulation 8 = Effective Batch Size 8.
    training_args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        max_steps=60 if not args.limit else (args.limit // 2),  # Adjust steps
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",  # Crucial for memory
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="wandb",
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    print("Starting Unsloth QLoRA Training (Qwen 2.5 32B)...")
    trainer.train()

    print(f"Saving adapter to {NEW_MODEL_NAME}...")
    model.save_pretrained(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, help="Limit number of training examples for testing"
    )
    args = parser.parse_args()
    train(args)
