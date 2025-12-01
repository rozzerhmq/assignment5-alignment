import torch
import os
import wandb
import argparse
import gc
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tests.adapters import (
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_sft_microbatch_train_step,
)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from cs336_alignment.eval import run_eval
from cs336_alignment.eval_stats import calculate_stats
from cs336_alignment.prompt_utils import format_r1_zero_example

# --- Configuration ---
# Hyperparameters for the training process
MICRO_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
TRAINING_SET_SIZE = 2048
TRAINING_STEPS = TRAINING_SET_SIZE // MICRO_BATCH_SIZE  # Use integer division
LEARNING_RATE = 2e-5
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
WANDB_PROJECT = "cs336-sft"
ADAPTER_SAVE_PATH = f"qwen-math-1.5b-cot-sft-{TRAINING_SET_SIZE}-samples-lr{LEARNING_RATE}-mb{MICRO_BATCH_SIZE}-ga{GRADIENT_ACCUMULATION_STEPS}"


def prepare_training_data(dataset: Dataset, tokenizer: AutoTokenizer) -> TensorDataset:
    """
    Prepares the training dataset for Supervised Fine-Tuning (SFT).

    This function performs the following steps:
    1. Shuffles the dataset and selects a subset of the specified size.
    2. Formats each example into a prompt and answer pair using the R1 Zero template.
    3. Tokenizes the prompts and answers to create input IDs, labels, and response masks.
    4. Wraps the tokenized data into a PyTorch TensorDataset for efficient loading.

    Args:
        dataset (Dataset): The raw Hugging Face dataset.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding text.

    Returns:
        TensorDataset: A dataset containing input_ids, labels, and response_mask tensors.
    """
    # Create a smaller, shuffled subset of the training data for fine-tuning
    training_data = dataset.shuffle(seed=42).select(range(TRAINING_SET_SIZE))

    # Extract and format prompts and answers from the dataset
    prompts = []
    answers = []
    for example in training_data:
        question = example["question"]
        raw_answer = example["answer"]

        # Format the question and answer using the specific template (e.g., R1 Zero)
        prompt, answer = format_r1_zero_example(question, raw_answer)

        prompts.append(prompt)
        answers.append(answer)

    # Tokenize the formatted prompts and answers.
    # This helper function handles the creation of 'labels' (shifted input_ids)
    # and 'response_mask' (1 for response tokens, 0 otherwise).
    tokenized_data = run_tokenize_prompt_and_output(prompts, answers, tokenizer)

    return TensorDataset(
        tokenized_data["input_ids"],
        tokenized_data["labels"],
        tokenized_data["response_mask"],
    )


def train_model(train_dataset: Dataset) -> str:
    """
    Runs the Supervised Fine-Tuning (SFT) training loop.

    Args:
        train_dataset (Dataset): The dataset to use for training.

    Returns:
        str: The path where the fine-tuned model is saved.
    """
    # --- Device Setup ---
    # Set the device to CUDA (GPU) if available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Model and Tokenizer Setup ---
    # Load the pre-trained model.
    # We use bfloat16 for mixed-precision training which improves memory efficiency and speed
    # on compatible GPUs (e.g., Ampere architecture), and Flash Attention 2 for faster attention calculation.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    # Enable gradient checkpointing to reduce memory usage by trading compute for memory.
    # This allows training larger models or using larger batch sizes.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # Disable KV cache as it's not needed for training

    # Load the tokenizer associated with the model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- Data Preparation ---
    # Process raw data into tensor format ready for the model
    tensor_dataset = prepare_training_data(train_dataset, tokenizer)

    # Create a DataLoader to handle batching and shuffling during training
    train_loader = DataLoader(tensor_dataset, batch_size=MICRO_BATCH_SIZE, shuffle=True)

    # --- Optimizer & Scheduler Setup ---
    # Initialize the AdamW optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Calculate total optimization steps.
    # We divide by GRADIENT_ACCUMULATION_STEPS because the optimizer only steps once
    # per accumulated batch, not per micro-batch.
    num_update_steps = TRAINING_STEPS // GRADIENT_ACCUMULATION_STEPS

    scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_update_steps),  # 10% warmup steps
        num_training_steps=num_update_steps,
    )

    # --- Training Loop ---
    model.train()
    global_step = 0
    batch_loss = 0
    batch_token_entropy = 0
    # Track total valid tokens to calculate a weighted average of entropy, which is mathematically more precise
    batch_accumulated_tokens = 0

    for step, (batch_inputs, batch_labels, batch_response_masks) in enumerate(
        train_loader
    ):
        if step >= TRAINING_STEPS:
            break
        print(f"Step {step+1}/{TRAINING_STEPS}")

        # Move batch tensors to the GPU
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        batch_response_masks = batch_response_masks.to(device)

        # Get the log probabilities of the correct tokens in the response.
        # We also calculate token entropy to monitor the model's uncertainty.
        response_log_p = run_get_response_log_probs(
            model, batch_inputs, batch_labels, return_token_entropy=True
        )

        # Accumulate metrics for logging
        if batch_response_masks.sum().item() > 0:
            batch_accumulated_tokens += batch_response_masks.sum().item()
            token_entropy = response_log_p["token_entropy"]
            # Detach entropy to prevent graph retention and memory leaks
            batch_token_entropy += (token_entropy * batch_response_masks).sum().item()

        # Perform the SFT training step (forward pass, loss calculation, and backward pass).
        # The loss returned is scaled by 1/GRADIENT_ACCUMULATION_STEPS inside the function
        # to ensure the magnitude of the gradients is correct when we step the optimizer.
        loss, metadata = run_sft_microbatch_train_step(
            response_log_p["log_probs"],
            batch_response_masks,
            GRADIENT_ACCUMULATION_STEPS,
            batch_response_masks.sum(),  # normalize_constant is the sum of tokens for mean loss
        )

        # Accumulate raw loss (mean per token) for logging purposes
        batch_loss += loss.item()

        # Perform an optimizer step only after accumulating gradients for the specified number of micro-batches
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            global_step += 1

            # Log metrics to Weights & Biases
            wandb.log(
                {
                    "train/loss": batch_loss,
                    "train/token_entropy": batch_token_entropy
                    / batch_accumulated_tokens,
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    ),
                },
                step=global_step,
            )

            # Step the optimizer and clear gradients for the next accumulation cycle
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Reset accumulation variables
            batch_loss = 0
            batch_token_entropy = 0
            batch_accumulated_tokens = 0

    # --- Save Model ---
    # Save the fine-tuned model and tokenizer to disk for later evaluation or usage
    print(f"Saving model to {ADAPTER_SAVE_PATH}...")
    model.save_pretrained(ADAPTER_SAVE_PATH)
    tokenizer.save_pretrained(ADAPTER_SAVE_PATH)

    # Free up memory explicitly to avoid OOM issues during subsequent evaluation
    del model
    del optimizer
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return ADAPTER_SAVE_PATH


def evaluate_model(adapter_save_path: str, test_dataset: Dataset):
    print("Running evaluation...")
    try:
        # 1. Run evaluation (Modular)
        results_filename = run_eval(adapter_save_path, dataset=test_dataset)

        # 2. Run stats calculation (Modular)
        stats = calculate_stats(results_filename)

        print(f"Evaluation Stats: {stats}")

        # 3. Log to WandB
        wandb.log(
            {
                "eval/format_reward": stats["avg_format_reward"],
                "eval/answer_reward": stats["avg_answer_reward"],
                "eval/average_reward": stats["avg_reward"],
            }
        )

    except Exception as e:
        print(f"An unexpected error occurred during evaluation: {e}")


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="SFT Training Script")
    parser.add_argument(
        "--run_train",
        action="store_true",
        help="Run the SFT training loop.",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Run evaluation and statistics after training.",
    )
    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # --- Weights & Biases Initialization ---
    if args.run_train or args.run_eval:
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

    # --- Data Loading ---
    # Assume the script is run from the project root
    data_dir = "data/gsm8k"
    dataset = load_dataset("json", data_dir=data_dir)

    # --- Orchestration ---
    adapter_save_path_from_train = None
    if args.run_train:
        adapter_save_path_from_train = train_model(dataset["train"])

    if args.run_eval:
        # If train was run, use the path from train_model, otherwise use the globally defined path
        eval_model_path = (
            adapter_save_path_from_train
            if adapter_save_path_from_train
            else ADAPTER_SAVE_PATH
        )
        evaluate_model(eval_model_path, dataset["test"])

    if args.run_train or args.run_eval:
        wandb.finish()
    print("Done.")


if __name__ == "__main__":
    main()
