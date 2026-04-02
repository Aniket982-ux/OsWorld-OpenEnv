"""
Reward calculator for the OsWorld Data Cleaning Environment.

R = step_penalty + ΔΦ + regression_penalty (if score drops) + terminal_bonus
"""


class RewardCalculator:
    """Simple, consistent reward shaping. The grader does the heavy lifting."""

    def __init__(
        self,
        step_penalty: float = -0.03,
        error_penalty: float = -0.2,
        regression_penalty: float = -0.1,
        terminal_reward: float = 10.0,
    ):
        self.step_penalty = step_penalty
        self.error_penalty = error_penalty
        self.regression_penalty = regression_penalty
        self.terminal_reward = terminal_reward

    def calculate(
        self,
        old_score: float,
        new_score: float,
        done: bool,
        is_error: bool = False,
        is_unknown: bool = False,
    ) -> float:
        """Calculate reward from score transition."""
        reward = self.step_penalty

        # Error / unknown action penalty
        if is_error or is_unknown:
            reward += self.error_penalty

        # ΔΦ shaping
        delta = new_score - old_score
        reward += delta

        # Extra regression penalty if score dropped
        if delta < 0:
            reward += self.regression_penalty

        # Terminal bonus
        if done and new_score >= 1.0:
            reward += self.terminal_reward

        return reward
