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

## Task Structure

The environment includes **6 task variants** across 3 difficulty tiers:

### Easy (2 issues per task)
| Variant | Description |
|---------|-------------|
| **Duplicate Removal** | Wrong column casing + Duplicate rows |
| **Format Normalization** | Wrong column casing + Inconsistent formatting (whitespace/casing) |

### Medium (3-5 issues per task)
| Variant | Description |
|---------|-------------|
| **Missing Value Imputation** | Wrong column casing + Extra column + Null imputation |
| **Schema Repair** | Non-standard column names + Extra column |
| **Constraint Enforcement** | Wrong column casing + Extra column + Duplicated IDs + Range clamps |

### Hard (7-9 issues combined)
| Variant | Description |
|---------|-------------|
| **Corrupted Pipeline Recovery** | Schema + extra col + duplicates + nulls + formatting + range clamps |

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

### 1. Run the Baseline Agent (End-to-End)
This executes the LLM agent across all 6 task variants. It spins up the server dynamically.
```bash
uv run baseline.py
```

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