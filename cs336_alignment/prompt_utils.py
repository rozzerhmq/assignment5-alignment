import os


def load_distill_templates():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", "distill.prompt")

    with open(prompt_path, "r") as f:
        distill_prompt_template = f.read()

    return distill_prompt_template


def format_distill_example(question: str):
    """
    Formats a single GSM8K example into the distillation prompt format.

    This function combines the question with the distillation prompt template.

    Args:
        question (str): The input question.

    Returns:
        str: The fully formatted prompt string.
    """
    prompt_template = load_distill_templates()
    return prompt_template.format(question=question)


def load_r1_zero_templates():
    """
    Loads the R1 Zero prompt template from the file system and defines the answer template.

    The prompt template is loaded from a file named 'r1_zero.prompt' located in the 'prompts'
    subdirectory relative to this script. The answer template is hardcoded to include standard boxing.

    Returns:
        tuple[str, str]: A tuple containing:
            - r1_zero_prompt_template (str): The content of the prompt template file.
            - r1_zero_answer_template (str): The format string for the answer, including reasoning and LaTeX boxing.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "prompts", "r1_zero.prompt")

    with open(prompt_path, "r") as f:
        r1_zero_prompt_template = f.read()

    # The prompt ends with <think>, so we start with the reasoning content.
    # We use double backslash for literal \ in the f-string result for the LaTeX command.
    r1_zero_answer_template = (
        " {reasoning} </think> <answer> \\boxed{{{final_answer}}} </answer>"
    )
    return r1_zero_prompt_template, r1_zero_answer_template


def parse_gsm8k_answer(raw_answer: str):
    """
    Parses a raw GSM8K answer string into its reasoning and final answer components.

    GSM8K answers typically separate the reasoning from the final numerical answer using '####'.

    Args:
        raw_answer (str): The raw answer string from the dataset.

    Returns:
        tuple[str, str]: A tuple containing:
            - reasoning (str): The reasoning steps extracted from the answer.
            - final_answer (str): The final numerical answer extracted from the answer.
    """
    if "####" in raw_answer:
        reasoning, final_answer = raw_answer.split("####", 1)
        reasoning = reasoning.strip()
        final_answer = final_answer.strip()
    else:
        reasoning = raw_answer.strip()
        final_answer = ""
    return reasoning, final_answer


def format_r1_zero_example(question: str, raw_answer: str):
    """
    Formats a single GSM8K example into the SFT prompt and answer format.

    This function combines the question with the R1 Zero prompt template and formats the
    raw answer into the expected structure with reasoning and a boxed final answer.

    Args:
        question (str): The input question.
        raw_answer (str): The raw GSM8K answer string.

    Returns:
        tuple[str, str]: A tuple containing:
            - formatted_prompt (str): The fully formatted prompt string.
            - formatted_answer (str): The fully formatted answer string.
    """
    prompt_template, answer_template = load_r1_zero_templates()

    reasoning, final_answer = parse_gsm8k_answer(raw_answer)

    formatted_prompt = prompt_template.format(question=question)
    formatted_answer = answer_template.format(
        reasoning=reasoning, final_answer=final_answer
    )

    return formatted_prompt, formatted_answer
