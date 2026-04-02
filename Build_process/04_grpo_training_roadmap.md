# Roadmap: Preparing OsWorld-OpenEnv for GRPO Training

If you plan to use an LLM-based RL algorithm like **GRPO (Group Relative Policy Optimization)**, the environment needs to be tailored to how GRPO learns. GRPO generates a *group* of multiple trajectories (e.g., 8 different python scripts) for the exact same prompt, evaluates all of them, and updates the model to favor the trajectories that scored higher relative to the group's average. 

To achieve high **reliability** and **novelty** (in terms of generalizable agent capability), here is the one massive shift you must make, followed by how the components must adapt.

---

## 1. The Core Reliability Shift: Procedural Task Generation
Because GRPO evaluates multiple samples simultaneously, if your tasks are static (always the same 6 rows of "Alice, Bob, Charlie"), the model will rapidly overfit. It will learn to hardcode `"alice"`, rather than learning to write a generalized `.str.strip().str.lower()` script.

**What must change:**
You must update `tasks.py` to become a factory rather than a static dictionary. 
* Instead of `expected_df = pd.DataFrame({'id':[1,2], 'name':['alice', 'bob']})`
* Use random data generation on `reset()`: 
  ```python
  num_rows = random.randint(10, 50)
  names = [faker.name() for _ in range(num_rows)]
  # Randomly inject corruptions like extra spaces to 20% of rows
  ```
This guarantees that **every episode is computationally unique**, forcing the agent to rely on programmatic logic rather than memory.

---

## 2. Updates to Tasks (`server/tasks.py`)
Apart from procedural generation, tasks need to support "Self-Verification" (a cornerstone of modern RL reasoning like DeepSeek-R1). 
* **Add Hidden Edge Cases:** Inject "trap" data that looks clean but isn't (e.g., a tab character instead of a space). 
* **Prompt Engineering for Reasoning:** Instruct the task description to require a `<think>` block before writing the code, where the LLM can output its chain of thought. GRPO learns exceptionally well when allowed to explore latent reasoning tokens before taking an environment action.

---

## 3. Updates to the Grader (`server/graders.py`)
GRPO relies on fine-grained, highly accurate scalar rankings to figure out which trajectory in the group was "best".
* **Resolution is King:** The current `SemanticGrader` is actually in an excellent state for GRPO because it uses continuous formulas (like F1 score and Jaccard similarity) instead of binary Pass/Fail logic. 
* **Efficiency Metrics:** Introduce a new component to the grader that assesses the *elegance* of the code. For instance, if two scripts in the group both clean the data perfectly, the one that used vectorized pandas operations should grade slightly higher than the one that used `.iterrows()`.

---

## 4. Updates to the Reward Function (`server/rewards.py`)
Currently, you use dense, potential-based reward shaping (giving `+0.27` for small improvements and `-0.03` for taking steps).
* **Terminal-Heavy (Outcome-Based) Rewards:** GRPO often performs better with outcome-based rewards rather than dense per-step shaping. Since GRPO optimizes relative to a baseline, dense inner rewards can sometimes cause reward hacking. 
* **Recommendation:** You may want to modify `rewards.py` to only calculate the score at `done=True`. If the agent fixes the dataset perfectly in 1 step, reward = `+1.0`. If it takes 5 steps, reward = `+0.8` (penalizing length). This sparse setup forces the GRPO algorithm to figure out the path itself, which leads to stronger, more novel "Aha!" moments in reasoning models.
