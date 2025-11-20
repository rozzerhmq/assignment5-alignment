import json
import sys
from cs336_alignment import EvaluationResult
from drgrpo_grader import r1_zero_reward_fn

def rerun_grader(file_path: str):
    """
    Reads a file with EvaluationResult objects, re-runs the grader,
    and updates the rewards in the file.
    """
    updated_results = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            # Re-create the EvaluationResult object to ensure structure
            result = EvaluationResult(
                prompt=data["prompt"],
                response=data["response"],
                golden=data["golden"],
                rewards=data["rewards"],
            )
            
            # Rerun the reward function
            new_rewards = r1_zero_reward_fn(result.response, result.golden)
            result.rewards = new_rewards
            
            updated_results.append(result)

    with open(file_path, "w") as f:
        for result in updated_results:
            f.write(json.dumps(result.__dict__) + "\n")

def main(file_path: str):
    rerun_grader(file_path)
    print(f"Successfully re-graded and updated {file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cs336_alignment/rerun_grader.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    main(file_path)