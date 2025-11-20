from typing import Callable, List

from vllm import LLM, SamplingParams
from cs336_alignment import EvaluationResult
from drgrpo_grader import r1_zero_reward_fn

import re, json


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
    model_name = vllm_model.llm_engine.model_config.model
    output_filename = f"{model_name.replace('/', '_')}_evaluation_results.jsonl"
    with (
        open("cs336_alignment/prompts/r1_zero.prompt", "r") as prompt_file,
        open("data/gsm8k/test.jsonl", "r") as test_file,
        open(output_filename, "w") as outfile,
    ):
        r1_zero_prompt = prompt_file.read()

        examples = [json.loads(line.strip()) for line in test_file]

        batch_size = 20
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
                reward = r1_zero_reward_fn(response_text, example["answer"])

                result = EvaluationResult(
                    prompt=response_output.prompt,
                    response=response_text,
                    golden=example["answer"],
                    rewards=reward,
                )

                outfile.write(json.dumps(result.__dict__) + "\n")

def main():
    # llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    llm = LLM(
        model="Qwen/Qwen3-4B-Base",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )

    prompts = [
        "What is 1+1?",
        "Tom gets 4 car washes a month.  If each car wash costs $15 how much does he pay in a year?",
    ]

    sampling_params = SamplingParams(
        temperature=1.0, top_p=0.95, max_tokens=1024, stop=["</answer>"]
    )
    sampling_params.include_stop_str_in_output = True

    evaluate_vllm(
        vllm_model=llm,
        reward_fn=None,
        prompts=prompts,
        eval_sampling_params=sampling_params,
    )

if __name__ == "__main__":
    main()