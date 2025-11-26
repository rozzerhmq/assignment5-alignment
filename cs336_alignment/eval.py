from typing import Callable, List
import argparse
import os
import re
import json

from vllm import LLM, SamplingParams
from cs336_alignment.evaluation_result import EvaluationResult
from cs336_alignment.drgrpo_grader import question_only_reward_fn


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.

    This script should (1) load the MATH validation examples from /data/a5-alignment/MATH/validation.jsonl,
    (2) format them as string prompts to the language model using the r1_zero prompt, and (3) gen-
    erate outputs for each example. This script should also (4) calculate evaluation metrics and
    (5) serialize the examples, model generations, and corresponding evaluation scores to disk for
    analysis in subsequent problems.

    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", "question_only.prompt")
    test_file_path = os.path.join(script_dir, "..", "data", "gsm8k", "test.jsonl")

    model_name = vllm_model.llm_engine.model_config.model
    output_filename = f"{model_name.replace('/', '_')}_evaluation_results.jsonl"
    with (
        open(prompt_path, "r") as prompt_file,
        open(test_file_path, "r") as test_file,
        open(output_filename, "w") as outfile,
    ):
        r1_zero_prompt = prompt_file.read()

        examples = [json.loads(line.strip()) for line in test_file]

        batch_size = 128
        total_batches = (len(examples) + batch_size - 1) // batch_size
        for i in range(0, len(examples), batch_size):
            print(f"Processing batch {i // batch_size + 1}/{total_batches}")
            batch_examples = examples[i : i + batch_size]
            model_prompts = [
                re.sub(r"\{question\}", example["question"], r1_zero_prompt)
                for example in batch_examples
            ]

            response_outputs = vllm_model.generate(
                model_prompts, sampling_params=eval_sampling_params
            )

            assert len(response_outputs) == len(batch_examples)

            for example, response_output in zip(batch_examples, response_outputs):
                response_text = response_output.outputs[0].text
                reward = reward_fn(response_text, example["answer"])

                result = EvaluationResult(
                    prompt=response_output.prompt,
                    response=response_text,
                    golden=example["answer"],
                    rewards=reward,
                )

                outfile.write(json.dumps(result.__dict__) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a VLLM model.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="The name or path of the model to evaluate.",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=0.85,
        max_model_len=1024,
    )

    sampling_params = SamplingParams(
        temperature=1.0, top_p=0.95, max_tokens=1024, stop=["</answer>"]
    )
    sampling_params.include_stop_str_in_output = True

    evaluate_vllm(
        vllm_model=llm,
        reward_fn=question_only_reward_fn,
        prompts=None,
        eval_sampling_params=sampling_params,
    )


if __name__ == "__main__":
    main()
