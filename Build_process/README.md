# OsWorld Data Cleaning Environment Architecture

This `Build_process` directory contains details describing the environment's internal mechanics. This system represents an OpenEnv-compliant reinforcement learning and heuristic evaluation wrapper targeting automated programmatic data cleaning tasks.

### Documentation Index
- [`01_scenarios_and_difficulties.md`](./01_scenarios_and_difficulties.md): Breakdowns of the 15 task variants across Easy, Medium, and Hard tiers, including start/end states and agent constraints.
- [`02_grading_mechanics.md`](./02_grading_mechanics.md): The multi-component semantic grader ($\Phi$) combining content F1, schema correctness, validity checks, and constraint satisfaction with anti-cheat protections.
- [`03_reward_shaping.md`](./03_reward_shaping.md): Potential-based $\Delta\Phi$ reward shaping with step penalties, regression penalties, and terminal bonuses.