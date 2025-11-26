import pandas as pd
import sys

format_reward_column = "format_reward"
answer_reward_column = "answer_reward"
reward_column = "reward"


def read_results(filename):
    df = pd.read_json(filename, lines=True)

    rewards_df = pd.json_normalize(df["rewards"])
    df = pd.concat([df, rewards_df], axis=1)
    df = df.drop(columns=["rewards"])

    print(df)
    print(f"Number of examples: {len(df)}")
    print(f"Number foramt rewards: {df[format_reward_column].sum()}")
    print(f"Number answer rewards: {df[answer_reward_column].sum()}")
    print(f"Average format reward: {df[format_reward_column].mean()}")
    print(f"Average answer reward: {df[answer_reward_column].mean()}")
    print(f"Average reward: {df[reward_column].mean()}")


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "evaluation_results.jsonl"
    read_results(filename)

if __name__ == "__main__":
    main()