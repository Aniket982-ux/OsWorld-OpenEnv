# Reward Functions and Reinforcement Mechanisms

The OsWorld environment uses a scientifically-grounded Reward Shaping matrix to properly guide reinforcement learning and LLM-agents alike. Pure sparse rewards (only receiving points at the very end of an episode) are notoriously difficult for agents to learn from. 

We utilize a **Potential-Based Reward Shaping** architecture with regression penalties. The grader does the heavy lifting; the reward stays simple and consistent.

## Core Formula

$$R = \text{step\_penalty} + \Delta\Phi + \text{regression\_penalty} + \text{terminal\_bonus}$$

Where $\Phi$ is the multi-component grader score (see `02_grading_mechanics.md`).

## 1. Differential Rewards ($\Delta\Phi$)

The environment tracks the grader score step-over-step and issues points correlating to exact *progress* made:

$$R_{diff} = \Phi(S_{current}) - \Phi(S_{previous})$$

If the agent takes an action that increases the score by 20%, they immediately receive $+0.20$ as reward. If they damage the file, the score drops, issuing a *negative* differential reward.

## 2. Step Penalty

Every action incurs a flat penalty of **-0.03**. This mathematically forces the agent to identify the *shortest possible path* to success rather than taking redundant actions.

## 3. Regression Penalty

If an agent's action causes the score to *decrease* ($\Delta\Phi < 0$), an additional **-0.10** penalty is applied on top of the natural negative $\Delta\Phi$. This strongly discourages destructive actions and "try random things" strategies.

## 4. Execution & Syntax Penalties

When the agent generates invalid Python logic (SyntaxError, runtime errors), the server traps the `traceback` and assesses an **-0.20** penalty. The error text is fed back into the observation's `screen_text` so the LLM can self-correct.

This same penalty applies to unknown action types.

## 5. Terminal Bonus

When $\Phi \geq 1.0$ (score reaches perfect clean status), the environment issues a **+10.0 Terminal Reward Bonus** and flips `done = True`, concluding the episode.

## Summary Episode Flow

1. **Start:** Score is $0.0$, Reward is $0.0$
2. **Action 1 (Fix Schema):** Score becomes $0.40$. Reward = $+0.40$ (Differential) $- 0.03$ (Step) = $+0.37$
3. **Action 2 (Syntax Error):** Score stays $0.40$. Reward = $-0.03$ (Step) $-0.20$ (Error) = $-0.23$
4. **Action 3 (Bad Fix, Regresses):** Score drops to $0.30$. Reward = $-0.10$ (Diff) $-0.03$ (Step) $-0.10$ (Regression) = $-0.23$
5. **Action 4 (Fix All Remaining):** Score hits $1.0$. Reward = $+0.70$ (Diff) $-0.03$ (Step) $+10.0$ (Terminal) = $+10.67$

**Total Episode Cumulative Return** evaluates how cleanly, efficiently, and intelligently the dataset was restored.

## Reward Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Step Penalty | -0.03 | Encourages efficiency |
| Error Penalty | -0.20 | Punishes syntax/runtime errors |
| Regression Penalty | -0.10 | Discourages destructive actions |
| Terminal Bonus | +10.0 | Strong completion incentive |

## Design Principle

The reward is intentionally **simple and consistent**. The multi-component grader ($\Phi$) does all the hard work of evaluating data quality. The reward function just converts $\Delta\Phi$ into learning signal with light penalties. This separation ensures that improving the grader automatically improves the reward signal without needing to redesign the reward function.