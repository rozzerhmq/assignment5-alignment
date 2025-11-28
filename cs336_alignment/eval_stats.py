import pandas as pd
import sys

format_reward_column = "format_reward"
answer_reward_column = "answer_reward"
reward_column = "reward"


def calculate_stats(filename):
    df = pd.read_json(filename, lines=True)

    rewards_df = pd.json_normalize(df["rewards"])
    df = pd.concat([df, rewards_df], axis=1)
    df = df.drop(columns=["rewards"])

    metrics = {
        "num_examples": len(df),
        "num_format_rewards": int(df[format_reward_column].sum()),
        "num_answer_rewards": int(df[answer_reward_column].sum()),
        "avg_format_reward": float(df[format_reward_column].mean()),
        "avg_answer_reward": float(df[answer_reward_column].mean()),
        "avg_reward": float(df[reward_column].mean()),
    }
    return metrics


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "evaluation_results.jsonl"
    
    metrics = calculate_stats(filename)
    
    print(f"Number of examples: {metrics['num_examples']}")
    print(f"Number foramt rewards: {metrics['num_format_rewards']}")
    print(f"Number answer rewards: {metrics['num_answer_rewards']}")
    print(f"Average format reward: {metrics['avg_format_reward']}")
    print(f"Average answer reward: {metrics['avg_answer_reward']}")
    print(f"Average reward: {metrics['avg_reward']}")

if __name__ == "__main__":
    main()