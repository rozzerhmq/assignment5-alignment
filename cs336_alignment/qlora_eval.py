from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import argparse
from datasets import load_dataset
from cs336_alignment.prompt_utils import load_r1_zero_templates, parse_gsm8k_answer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.evaluation_result import EvaluationResult
import json
import re
import os

# --- Configuration ---
# Must match the base model used in training (or be compatible)
BASE_MODEL = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
ADAPTER_PATH = "qwen3-14b-math-sft-adapter"  # Path to saved adapter
ADAPTER_NAME = "math-adapter"


def evaluate(args):
    # 1. Initialize vLLM with LoRA enabled
    # gpu_memory_utilization=0.8 leaves VRAM buffer
    if args.base_only:
        print(f"Loading base model: {BASE_MODEL} (No Adapter)")
        enable_lora = False
    else:
        print(f"Loading base model: {BASE_MODEL} with adapter from: {ADAPTER_PATH}")
        if not os.path.exists(ADAPTER_PATH):
            print(
                f"Error: Adapter path '{ADAPTER_PATH}' not found. Run qlora_sft.py first."
            )
            return
        enable_lora = True

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=enable_lora,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        max_lora_rank=16,  # Must match training rank
        max_model_len=2048,  # Limit max sequence length for VLLM
    )

    # 2. Define Sampling Parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        max_tokens=2048,
        stop=["</answer>"],  # Stop at end of answer
    )

    # 3. Load Data
    print("Loading GSM8K test dataset...")
    data_dir = "data/gsm8k"
    dataset = load_dataset("json", data_dir=data_dir, split="test")

    # Optional: subset for quick testing
    if args.limit:
        print(f"Limiting to first {args.limit} examples.")
        dataset = dataset.select(range(args.limit))

    # 4. Prepare Prompts
    r1_zero_prompt, r1_zero_answer_template = load_r1_zero_templates()

    prompts = [
        re.sub(r"\{question\}", example["question"], r1_zero_prompt)
        for example in dataset
    ]

    print(f"Running inference on {len(prompts)} examples...")

    # 5. Generate with Adapter
    # We load the adapter dynamically using LoRARequest
    if args.base_only:
        outputs = llm.generate(prompts, sampling_params)
        output_filename = "qlora_base_evaluation_results.jsonl"
    else:
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=LoRARequest(ADAPTER_NAME, 1, ADAPTER_PATH),
        )
        output_filename = "qlora_evaluation_results.jsonl"

    # 6. Process and Save Results
    with open(output_filename, "w") as outfile:
        for example, output in zip(dataset, outputs):
            response_text = output.outputs[0].text

            # Score
            reasoning, final_answer = parse_gsm8k_answer(example["answer"])
            golden_answer = r1_zero_answer_template.format(
                reasoning=reasoning, final_answer=final_answer
            )
            reward = r1_zero_reward_fn(response_text, golden_answer)

            result = EvaluationResult(
                prompt=output.prompt,
                response=response_text,
                golden=golden_answer,
                rewards=reward,
            )
            outfile.write(json.dumps(result.__dict__) + "\n")

    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, help="Limit number of examples for quick testing"
    )
    parser.add_argument(
        "--base_only", action="store_true", help="Evaluate the base model only"
    )
    args = parser.parse_args()
    evaluate(args)
