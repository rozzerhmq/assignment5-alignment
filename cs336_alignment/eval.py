from typing import Callable
import argparse
import re
import json
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams
from cs336_alignment.evaluation_result import EvaluationResult
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.prompt_utils import load_r1_zero_templates, parse_gsm8k_answer


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    eval_sampling_params: SamplingParams,
    dataset: Dataset,
) -> str:
    """
    Evaluates a VLLM model on a dataset, computes rewards, and saves results.

    Args:
        vllm_model (LLM): The VLLM model instance.
        reward_fn (Callable): A function to compute rewards (e.g., correctness).
        eval_sampling_params (SamplingParams): Parameters for text generation.
        dataset (Dataset): The dataset to evaluate on.

    Returns:
        str: The filename of the saved evaluation results.
    """
    model_name = vllm_model.llm_engine.model_config.model
    # Create a filename based on the model name to avoid overwriting
    output_filename = f"{model_name.replace('/', '_')}_evaluation_results.jsonl"

    # Load the R1 Zero prompt template to format questions
    r1_zero_prompt, r1_zero_answer_template = load_r1_zero_templates()

    print(
        f"Starting evaluation on {len(dataset)} examples. Results will be saved to {output_filename}."
    )

    with open(output_filename, "w") as outfile:
        # Process the dataset in batches for efficiency
        batch_size = 128
        total_batches = (len(dataset) + batch_size - 1) // batch_size

        for i in range(0, len(dataset), batch_size):
            print(f"Processing batch {i // batch_size + 1}/{total_batches}")

            # Select the current batch of examples
            batch_examples = dataset.select(range(i, min(i + batch_size, len(dataset))))

            # Format prompts: Insert the question into the R1 Zero template
            model_prompts = [
                re.sub(r"\{question\}", example["question"], r1_zero_prompt)
                for example in batch_examples
            ]

            # Generate responses using VLLM
            response_outputs = vllm_model.generate(
                model_prompts, sampling_params=eval_sampling_params
            )

            assert len(response_outputs) == len(
                batch_examples
            ), "Mismatch between prompts and outputs"

            # Process each generated response
            for example, response_output in zip(batch_examples, response_outputs):
                response_text = response_output.outputs[0].text

                # Parse the ground truth reasoning and final answer from the dataset
                reasoning, final_answer = parse_gsm8k_answer(example["answer"])

                # Construct the "golden" (reference) answer in the expected format
                golden_answer = r1_zero_answer_template.format(
                    reasoning=reasoning, final_answer=final_answer
                )

                # Calculate rewards (e.g., did the model get the correct answer?)
                reward = reward_fn(response_text, golden_answer)

                # Create a result object
                result = EvaluationResult(
                    prompt=response_output.prompt,
                    response=response_text,
                    golden=golden_answer,
                    rewards=reward,
                )

                # Write the result to the JSONL file immediately
                outfile.write(json.dumps(result.__dict__) + "\n")

    return output_filename


def run_eval(model_name_or_path: str, dataset: Dataset = None) -> str:
    """
    Initializes the model and runs the evaluation loop.

    Args:
        model_name_or_path (str): Path or name of the model.
        dataset (Dataset): The dataset to evaluate.

    Returns:
        str: Path to the results file.
    """
    print(f"Loading model: {model_name_or_path}")

    # Initialize VLLM.
    # gpu_memory_utilization is set to 0.85 to leave some room for overhead.
    llm = LLM(
        model=model_name_or_path,
        gpu_memory_utilization=0.85,
        max_model_len=1024,
    )

    # Configure sampling parameters.
    # stop=["</answer>"] ensures the model stops generating after producing the answer.
    sampling_params = SamplingParams(
        temperature=1.0, top_p=0.95, max_tokens=1024, stop=["</answer>"]
    )
    sampling_params.include_stop_str_in_output = True

    return evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        eval_sampling_params=sampling_params,
        dataset=dataset,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a VLLM model on GSM8K.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="The name or path of the model to evaluate.",
    )
    args = parser.parse_args()

    # Load the test split of the GSM8K dataset
    print("Loading GSM8K test dataset...")
    data_dir = "data/gsm8k"
    dataset = load_dataset("json", data_dir=data_dir, split="test")

    run_eval(args.model, dataset=dataset)


if __name__ == "__main__":
    main()
