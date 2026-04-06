# OsWorld Data Cleaning Environment

This is the **Data Cleaning Environment** built on the OpenEnv framework. It presents programmatic challenges of varying complexity where AI agents must read, diagnose, and clean data artifacts (`data.csv` files). 

The environment uses structured Pydantic inputs, sandboxed Python code execution, a **multi-component semantic grader** (content F1 + schema + validity + constraints), potential-based reward shaping, and anti-cheat protections.

## Quick Start & Setup

The project is built using Python `uv` for lightning-fast dependency management.

1. **Install uv** (if you haven't already):
   ```bash
   pip install uv
   ```

2. **Clone and Install Dependencies**:
   ```bash
   # From the project root
   uv sync
   ```

## Environment Variables (.env)

The `baseline.py` script relies on an LLM to interact with the environment. It defaults to using OpenRouter (specifically the `gpt-4o-mini` model). 

You must create a `.env` file in the root of the project with your API key:

1. Create a file named `.env`:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   
   # If you adjust baseline.py to use standard OpenAI:
   # OPENAI_API_KEY=your_openai_api_key_here
   ```

*(Note: The environment server itself does not require an API key to run, only the baseline agent script requires it).*

## Action and Observation Space Definitions

### Observation Space (`OsworldObservation`)
- `screen_text` (str): Terminal or textual output from the executed action.
- `files` (Dict[str, str]): Key-value map of workspace file states (e.g., `'data.csv'` -> content).
- `current_task` (str): Objective to accomplish.
- `score` (float): Current normalized completion score from 0.0 to 1.0.
- `done` (bool): Indicates episode termination.
- `reward` (float): Calculated Delta-Phi shaped reward.

### Action Space (`OsworldAction`)
Actions strictly conform to a Pydantic structure utilizing `action_type` (str) and `payload` (Dict):
- `action_type`: One of `"inspect_schema"`, `"view_head"`, `"read_file"`, `"preview_changes"`, `"execute_python"`, `"remove_duplicates"`, or `"fill_nulls"`.
- `payload`: Contains parameters such as `"code"` (str, executed inside sandbox), `"filename"` (str), or `"n"` (int).

## Task Structure
The environment includes **12 task variants** across 3 difficulty tiers:

### Easy (4 variants)
| Variant | Description |
|---------|-------------|
| **Duplicate Removal** | Standardize column names. Remove duplicate rows. |
| **Format Normalization** | Standardize names to lowercase without leading/trailing whitespace. |
| **Type Coercion** | Convert ages to int, Yes/No to booleans, standardize columns. |
| **Column Rename Only** | Properly rename column headers. |

### Medium (5 variants)
| Variant | Description |
|---------|-------------|
| **Missing Value Imputation** | Drop extra columns, standardize names, fill missing with 0. |
| **Schema Repair** | Standardize weird column names and strip extra flags. |
| **Constraint Enforcement** | Deduplicate, enforce bounds, standardize names. |
| **Multi-File Join** | Clean secondary file, join on user_id, save as new csv. |
| **JSON Normalization** | Flatten deeply nested JSON into standard tabular dataframe. |

### Hard (3 variants)
| Variant | Description |
|---------|-------------|
| **Corrupted Pipeline Recovery** | Corrupted rows, bounds enforcement, str cleaning, dedup, fill. |
| **Adversarial Corruption** | Syntactically intact but semantically impossible constraints. |
| **Cascading Pipeline** | Multi-file dependency. Extract rates, compute new columns, fill bounds. |

Tasks cycle automatically on each `reset()` call.

## Grading System

The environment uses a **multi-component semantic grader** (not simple string matching):

```
Score = 0.4 * content_score    (F1: precision + recall via merge)
      + 0.2 * schema_score     (column name Jaccard + order bonus)
      + 0.2 * validity_score   (nulls, types, formatting)
      + 0.2 * constraint_score (uniqueness, ranges)
      - extra_row_penalty      (anti-cheat)
```

## Running the Project

### 1. Run the Benchmarking Inference Script (Hackathon)
To run a strictly validated evaluation producing OpenEnv standard `[START]`, `[STEP]`, `[END]` outputs:
```bash
uv run inference.py
```
**Baseline Scores:**
Using our default `Qwen/Qwen2.5-72B-Instruct` via a remote API endpoint reliably nets a normalized `score` of **~0.90** on Easy Tasks, and **~0.60** on Hard tasks. The soft multi-component grader guarantees meaningful signal and partial rewards!

### 2. Standalone Server Mode
If you are developing your own agent, run the server separately:
```bash
uvicorn server.app:app --port 8000 --reload
```

And in your agent script, connect to it:
```python
from client import OsworldEnv
env = OsworldEnv(url="ws://localhost:8000")
env.reset()  # Cycles through tasks automatically
```

### 3. Run Evaluation Tests
Verify grader, rewards, and anti-cheat protections:
```bash
uv run python eval.py
```

## Documentation

For deep dives into how the underlying architecture works, see `Build_process/`:

- [**Scenarios and Difficulties**](Build_process/01_scenarios_and_difficulties.md): Details on the 6 task variants across Easy, Medium, and Hard tiers.
- [**Semantic Grading Mechanics**](Build_process/02_grading_mechanics.md): How the multi-component grader eliminates the vulnerabilities of string-matching and merge-only scoring.
- [**Reward Shaping & Scoring**](Build_process/03_reward_shaping.md): How potential-based reward shaping with regression penalties enforces optimal reasoning paths.

## Project Structure
```text
OsWorld/
├── Build_process/         # Architectural documentation
├── __init__.py            # Module exports
├── client.py              # OsworldEnv client
├── models.py              # Strict Action and Observation Pydantic models
├── baseline.py            # Reference agent using OpenRouter LLM
├── eval.py                # Grader, reward, and anti-exploit tests
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Requirements
└── server/
    ├── OsWorld_environment.py  # Core environment logic
    ├── tasks.py           # 6 task variants with expected states + constraints
    ├── graders.py         # Multi-component semantic grader (Phi)
    ├── rewards.py         # Reward shaping calculator
    └── app.py             # FastAPI App
```