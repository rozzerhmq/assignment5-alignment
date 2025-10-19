from typing import Callable, List

from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn

import re, json


class EvaluationResult:
    """
    A class to store the results of an evaluation.
    """

    def __init__(
        self, prompt: str, response: str, golden: str, rewards: dict[str, float]
    ):
        self.prompt = prompt
        self.response = response
        self.golden = golden
        self.rewards = rewards


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
    results: List[EvaluationResult] = []

    with (
        open("cs336_alignment/prompts/r1_zero.prompt", "r") as prompt_file,
        open("data/gsm8k/test.jsonl", "r") as test_file,
    ):
        r1_zero_prompt = prompt_file.read()

        examples = [json.loads(line.strip()) for line in test_file]

        model_prompts = [
            re.sub(r"\{question\}", example["question"], r1_zero_prompt)
            for example in examples
        ]

        response_outputs = vllm_model.generate(
            model_prompts, sampling_params=eval_sampling_params
        )

        assert len(response_outputs) == len(examples)

        for example, response_output in zip(examples, response_outputs):
            response_text = response_output.outputs[0].text
            reward = r1_zero_reward_fn(response_text, example["answer"])

            result = EvaluationResult(
                prompt=response_output.prompt,
                response=response_text,
                golden=example["answer"],
                rewards=reward,
            )

            results.append(result)

    with open("evaluation_results.jsonl", "w") as outfile:
        for result in results:
            outfile.write(json.dumps(result.__dict__) + "\n")


if __name__ == "__main__":
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")

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
