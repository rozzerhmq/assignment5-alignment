import torch
import json
import os
import wandb
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from tests.adapters import (
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_sft_microbatch_train_step,
)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Hyperparameters for the training process
MICRO_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
TRAINING_SET_SIZE = 2048
TRAINING_STEPS = TRAINING_SET_SIZE // MICRO_BATCH_SIZE  # Use integer division
LEARNING_RATE = 3e-4
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
WANDB_PROJECT = "cs336-sft"


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

        # Step the optimizer and clear gradients
        optimizer.step()
        optimizer.zero_grad()

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
        batch_loss = 0  # Reset batch loss for the next accumulation
        batch_token_entropy = 0  # Reset token entropy for the next accumulation
        batch_accumulated_tokens = 0  # Reset token count for the next accumulation


# --- Save Model ---
# Save the fine-tuned model and tokenizer to a directory
print("Saving model to qwen-math-1.5b-sft-adapter...")
model.save_pretrained("qwen-math-1.5b-sft-adapter")
tokenizer.save_pretrained("qwen-math-1.5b-sft-adapter")
wandb.finish()
print("Done.")
