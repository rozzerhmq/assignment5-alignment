from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# --- Configuration ---
BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B"
ADAPTER_PATH = "./Qwen-Math-1.5B-SFT-Adapter"  # Path to saved adapter
ADAPTER_NAME = "math-adapter"

# 1. Initialize vLLM with LoRA enabled
# gpu_memory_utilization=0.8 leaves 20% VRAM buffer just in case
llm = LLM(
    model=BASE_MODEL,
    enable_lora=True,
    gpu_memory_utilization=0.8,
    trust_remote_code=True,
)

# 2. Define Sampling Parameters (Temperature 0 for strict math)
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

# 3. Test Prompts
prompts = [
    "Calculate the integral of x^2 from 0 to 3.",
    "If a train travels 300 miles in 4 hours, what is its average speed?",
    "Solve for x: 3x + 5 = 20",
]

print("--- Running Inference with Base Model ONLY (for comparison) ---")
# Passing lora_request=None uses the base model
outputs_base = llm.generate(prompts, sampling_params, lora_request=None)

for prompt, output in zip(prompts, outputs_base):
    print(f"\nPrompt: {prompt}")
    print(f"Base Output: {output.outputs[0].text}")

print("\n\n--- Running Inference with YOUR SFT ADAPTER ---")
# We load the adapter dynamically
outputs_adapter = llm.generate(
    prompts, sampling_params, lora_request=LoRARequest(ADAPTER_NAME, 1, ADAPTER_PATH)
)

for prompt, output in zip(prompts, outputs_adapter):
    print(f"\nPrompt: {prompt}")
    print(f"Adapter Output: {output.outputs[0].text}")
