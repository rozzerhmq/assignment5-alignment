import torch
import json
import os
from transformers import AutoTokenizer
from tests.adapters import run_tokenize_prompt_and_output

def create_eos_checker():
    """
    This utility checks if the tokenized response sequences in a dataset
    correctly end with the End-of-Sequence (EOS) token.
    """
    print("--- EOS Sanity Checker ---")

    # 1. Load Tokenizer
    print("Loading tokenizer: Qwen/Qwen2.5-Math-1.5B...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_eos_id = tokenizer.eos_token_id
    print(f"EOS Token ID: {tokenizer_eos_id}")

    # 2. Load Data
    print("Loading data: data/gsm8k/train.jsonl...")
    with open("data/gsm8k/train.jsonl", "r") as f:
        train_data_json = [json.loads(line) for line in f]
    
    prompts = [example["question"] for example in train_data_json]
    answers = [example["answer"] for example in train_data_json]
    
    # 3. Tokenize Data
    print("Tokenizing all training data...")
    tokenized_data = run_tokenize_prompt_and_output(prompts, answers, tokenizer)
    all_labels = tokenized_data["labels"]
    all_response_masks = tokenized_data["response_mask"]
    print(f"Shape of labels: {all_labels.shape}")
    print("Tokenization complete.")

    # 4. Define and Run the Sanity Check Logic
    errors_found = 0
    total_checked = 0

    print("\nRunning EOS sanity check on all sequences...")
    # Iterate through the labels in the batch
    for i, label_row in enumerate(all_labels):
        total_checked += 1
        response_mask_row = all_response_masks[i]
        
        # Find the indices of the response tokens
        response_indices = response_mask_row.nonzero(as_tuple=True)[0]
        
        if len(response_indices) == 0:
            print(f"⚠️ Row {i}: Has no response tokens!")
            continue
            
        # Get the index of the last response token
        last_response_idx = response_indices[-1]
        last_response_token = label_row[last_response_idx]
        
        if last_response_token != tokenizer_eos_id:
            if errors_found < 5: # Print details for the first 5 errors for debugging
                print(f"❌ Error in Row {i}: Last response token is {last_response_token} ('{tokenizer.decode([last_response_token])}'), expected EOS ({tokenizer_eos_id})")
                print(f"   Last 5 Labels: {label_row[last_response_idx-4:last_response_idx+1].tolist()}")
                print(f"   Last 5 Decoded: {tokenizer.decode(label_row[last_response_idx-4:last_response_idx+1])}")
                print(f"   Mask for last 5: {response_mask_row[last_response_idx-4:last_response_idx+1].tolist()}")

            errors_found += 1
    
    # 5. Print Summary
    print("\n--- Check Complete ---")
    if errors_found == 0:
        print(f"✅ Sanity check passed! All {total_checked} sequences correctly end with the EOS token.")
    else:
        print(f"❌ Sanity check failed. Found {errors_found} out of {total_checked} sequences that do not end with the EOS token.")
        if errors_found == total_checked:
             print("This may indicate a systematic issue in the tokenization or slicing logic.")

if __name__ == "__main__":
    create_eos_checker()