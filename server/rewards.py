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
        destructive_penalty: float = -0.5,
        terminal_reward: float = 1.0,
        inspect_first_bonus: float = 0.05,
    ):
        self.step_penalty = step_penalty
        self.error_penalty = error_penalty
        self.regression_penalty = regression_penalty
        self.destructive_penalty = destructive_penalty
        self.terminal_reward = terminal_reward
        self.inspect_first_bonus = inspect_first_bonus

    def calculate(
        self,
        old_score: float,
        new_score: float,
        done: bool,
        step_count: int,
        optimal_steps: int,
        first_action_type: str | None = None,
        is_error: bool = False,
        is_unknown: bool = False,
        is_destructive: bool = False,
    ) -> float:
        """Calculate reward from score transition."""
        reward = self.step_penalty

        # Inspect first bonus immediately applied on step 1
        if step_count == 1 and first_action_type in ["inspect_schema", "view_head", "read_file"]:
            reward += self.inspect_first_bonus

        # Error / unknown action penalty
        if is_error or is_unknown:
            reward += self.error_penalty

        # Destructive action penalty
        if is_destructive:
            reward += self.destructive_penalty

        # ΔΦ shaping
        delta = new_score - old_score
        reward += delta

        # Extra regression penalty if score dropped
        if delta < 0:
            reward += self.regression_penalty

        # Terminal bonus (efficiency scaled)
        if done and new_score >= 1.0:
            efficiency_ratio = min(1.0, optimal_steps / max(1, step_count))
            # Floor at 20% of terminal reward so they still get something for succeeding
            efficiency_ratio = max(0.2, efficiency_ratio)
            reward += self.terminal_reward * efficiency_ratio

        return reward
