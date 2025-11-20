
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
