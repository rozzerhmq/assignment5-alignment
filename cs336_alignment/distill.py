from datasets import load_dataset
from vllm import LLM, SamplingParams
from prompt_utils import format_distill_example
import json

EPOCH = 1
SAMPLE_D = 16
GENERATE_G = 2


def main():
    # --- Data Loading ---
    # Assume the script is run from the project root
    data_dir = "data/gsm8k"
    dataset = load_dataset("json", data_dir=data_dir)

    model_name = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
    # Expert Iteration Loop
    # 1. Sample D examples from the dataset
    # 2. Generate G repsonses per example using the old policy model with vLLM
    # 3. Evaluate the generated responses using the reward function
    # 4. Filter out wrong answers, and keep the correct ones
    # 5. Run the SFT iteration using the filtered correct responses
    print(f"Loading model: {model_name}")

    # Initialize VLLM.
    # gpu_memory_utilization is set to 0.85 to leave some room for overhead.
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.85,
        max_model_len=1024,
    )

    # Define Sampling Parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.95,
        n=GENERATE_G,
        max_tokens=2048,
        stop=["####END"],  # Stop at end of answer
    )

    for epoch in range(EPOCH):
        print(f"Starting epoch {epoch + 1}/{EPOCH}...")
        dataset_sampled = dataset["train"].shuffle(seed=42).select(range(SAMPLE_D))
        prompts = [
            format_distill_example(example["question"]) for example in dataset_sampled
        ]

        vllm_outputs = llm.generate(prompts, sampling_params)

    with open("distill_output.jsonl", "w") as outfile:
        for i, output in enumerate(vllm_outputs):
            example = dataset_sampled[i]
            for j, response in enumerate(output.outputs):
                result = {
                    "question": example["question"],
                    "answer": response.text.split("####END")[0].strip(),
                }
                outfile.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
