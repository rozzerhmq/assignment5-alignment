import torch
import json
import os
import wandb
import argparse
import subprocess
import re
import gc
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from tests.adapters import (
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_sft_microbatch_train_step,
)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="SFT Training Script")
parser.add_argument(
    "--run_eval",
    action="store_true",
    help="Run evaluation and statistics after training.",
)
args = parser.parse_args()

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Hyperparameters for the training process
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
TRAINING_SET_SIZE = 2048
TRAINING_STEPS = TRAINING_SET_SIZE // MICRO_BATCH_SIZE  # Use integer division
LEARNING_RATE = 2e-5
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
WANDB_PROJECT = "cs336-sft"
ADAPTER_SAVE_PATH = f"qwen-math-1.5b-sft-{TRAINING_SET_SIZE}-samples-lr{LEARNING_RATE}-mb{MICRO_BATCH_SIZE}-ga{GRADIENT_ACCUMULATION_STEPS}"


# --- Device Setup ---
# Set the device to CUDA (GPU) if available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Weights & Biases Initialization ---
# Try to log in from an environment variable. If not found, fall back to anonymous logging.
wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    print("WANDB_API_KEY not found, logging anonymously.")
    wandb.login(anonymous="allow")

hyperparameters = {
    "learning_rate": LEARNING_RATE,
    "micro_batch_size": MICRO_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "effective_batch_size": MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
    "training_set_size": TRAINING_SET_SIZE,
    "training_steps": TRAINING_STEPS,
    "model_name": MODEL_NAME,
}
wandb.init(
    project=WANDB_PROJECT,
    config=hyperparameters,
    name=f"sft-run-{wandb.util.generate_id()}",
)


# --- Model and Tokenizer Setup ---
# Load the pre-trained model and move it to the selected device.
# Using bfloat16 for mixed-precision training and flash_attention_2 for efficiency on supported GPUs.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
).to(
    device
)  # Move model to GPU

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()
model.config.use_cache = False

# Load the tokenizer associated with the model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# --- Data Loading and Preparation ---
# Construct absolute paths to data files to ensure the script is runnable from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data", "gsm8k")

# Manually load the JSONL data
with open(os.path.join(data_dir, "train.jsonl"), "r") as f:
    train_data_json = [json.loads(line) for line in f]
with open(os.path.join(data_dir, "test.jsonl"), "r") as f:
    test_data_json = [json.loads(line) for line in f]

# Create Hugging Face Dataset objects
train_dataset = Dataset.from_list(train_data_json)
test_dataset = Dataset.from_list(test_data_json)

# Combine into a DatasetDict
dataset = DatasetDict(
    {
        "train": train_dataset,
        "test": test_dataset,
    }
)

# Create a smaller, shuffled subset of the training data for fine-tuning
training_data = dataset["train"].shuffle(seed=42).select(range(TRAINING_SET_SIZE))

# Extract prompts and answers from the dataset
prompts = [example["question"] for example in training_data]
answers = [example["answer"] for example in training_data]

# Tokenize the prompts and answers using the adapter function
tokenized_data = run_tokenize_prompt_and_output(prompts, answers, tokenizer)

# Convert the dictionary of tokenized data into a PyTorch TensorDataset
input_ids = tokenized_data["input_ids"]
labels = tokenized_data["labels"]
response_mask = tokenized_data["response_mask"]
tensor_dataset = TensorDataset(input_ids, labels, response_mask)

# Create a DataLoader to handle batching and shuffling
train_loader = DataLoader(tensor_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True)

# --- Optimizer Setup ---
# Initialize the AdamW optimizer with the model's parameters and a learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
# Set the model to training mode
model.train()
global_step = 0
batch_loss = 0
batch_token_entropy = 0
# Track total valid tokens to do a weighted average (More mathematically precise)
batch_accumulated_tokens = 0
for step, (batch_inputs, batch_labels, batch_response_masks) in enumerate(train_loader):
    if step >= TRAINING_STEPS:  # Ensure loop stops after one epoch
        break
    print(f"Step {step+1}/{TRAINING_STEPS}")

    # Move batch tensors to the GPU
    batch_inputs = batch_inputs.to(device)
    batch_labels = batch_labels.to(device)
    batch_response_masks = batch_response_masks.to(device)

    # Get the log probabilities of the correct tokens in the response
    response_log_p = run_get_response_log_probs(
        model, batch_inputs, batch_labels, return_token_entropy=True
    )

    if batch_response_masks.sum().item() > 0:
        batch_accumulated_tokens += batch_response_masks.sum().item()
        token_entropy = response_log_p["token_entropy"]
        batch_token_entropy += (token_entropy * batch_response_masks).sum()

    # Perform the SFT training step (forward pass, loss calculation, and backward pass)
    # The loss is normalized by the number of response tokens to get the mean loss per token.
    loss, metadata = run_sft_microbatch_train_step(
        response_log_p["log_probs"],
        batch_response_masks,
        GRADIENT_ACCUMULATION_STEPS,
        batch_response_masks.sum(),  # normalize_constant is the sum of tokens for mean loss
    )

    # Accumulate raw loss (mean per token)
    batch_loss += loss.item()

    # Perform an optimizer step after accumulating gradients for the specified number of steps
    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
        global_step += 1

        # Log metrics to Weights & Biases
        wandb.log(
            {
                "train/loss": batch_loss,
                "train/token_entropy": batch_token_entropy / batch_accumulated_tokens,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/grad_norm": torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                ),
            },
            step=global_step,
        )

        # Step the optimizer and clear gradients
        optimizer.step()
        optimizer.zero_grad()

        batch_loss = 0  # Reset batch loss for the next accumulation
        batch_token_entropy = 0  # Reset token entropy for the next accumulation
        batch_accumulated_tokens = 0  # Reset token count for the next accumulation


# --- Save Model ---
# Save the fine-tuned model and tokenizer to a directory
print(f"Saving model to {ADAPTER_SAVE_PATH}...")
model.save_pretrained(ADAPTER_SAVE_PATH)
tokenizer.save_pretrained(ADAPTER_SAVE_PATH)

# Free up memory for evaluation
del model
del optimizer
del tokenizer
gc.collect()
torch.cuda.empty_cache()

if args.run_eval:
    print("Running evaluation...")
    try:
        # 1. Run evaluation
        # Note: We use subprocess to run in a separate process to avoid vLLM/PyTorch context issues
        # and to match how we run it from the command line.
        eval_cmd = [
            "uv",
            "run",
            "python",
            "cs336_alignment/eval.py",
            "--model",
            ADAPTER_SAVE_PATH,
        ]
        subprocess.run(eval_cmd, check=True)

        # 2. Run stats calculation
        results_file = f"{ADAPTER_SAVE_PATH}_evaluation_results.jsonl"
        stats_cmd = [
            "uv",
            "run",
            "python",
            "cs336_alignment/eval_stats.py",
            results_file,
        ]
        # Capture output to parse metrics
        result = subprocess.run(stats_cmd, capture_output=True, text=True, check=True)
        print(result.stdout)

        # 3. Parse metrics from stdout
        # Expected output format:
        # Average format reward: 0.9689...
        # Average answer reward: 0.5003...
        format_reward_match = re.search(
            r"Average format reward: ([\d.]+)", result.stdout
        )
        answer_reward_match = re.search(
            r"Average answer reward: ([\d.]+)", result.stdout
        )

        metrics = {}
        if format_reward_match:
            metrics["eval/format_reward"] = float(format_reward_match.group(1))
        if answer_reward_match:
            metrics["eval/answer_reward"] = float(answer_reward_match.group(1))

        if metrics:
            print(f"Logging eval metrics to wandb: {metrics}")
            wandb.log(metrics)
        else:
            print("Warning: Could not parse evaluation metrics from output.")

    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during evaluation: {e}")

wandb.finish()
print("Done.")
