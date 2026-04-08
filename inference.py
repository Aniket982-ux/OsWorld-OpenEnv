import asyncio
import os
import json
import textwrap
from typing import Dict, Any, List, Optional
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from client import OsworldEnv
from models import OsworldAction

# --- ANSI Colors for Premium Dashboard ---
C_BOLD = "\033[1m"
C_BLUE = "\033[94m"
C_CYAN = "\033[96m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_RESET = "\033[0m"

def draw_phi_bar(phi: float, width: int = 20) -> str:
    """Visual progress bar for the Phi score (0.0 to 1.0)."""
    filled = int(phi * width)
    bar = "█" * filled + "░" * (width - filled)
    color = C_GREEN if phi >= 1.0 else C_YELLOW if phi > 0.5 else C_CYAN
    return f"{color}[{bar}] {phi:.2f}{C_RESET}"

class Payload(BaseModel):
    filename: str | None = None
    n: int | None = None
    column: str | None = None
    value: str | None = None
    code: str | None = None

class LLMAction(BaseModel):
    action_type: str
    payload: Payload

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "dummy_key_if_local")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_NAME = os.getenv("OSWORLD_TASK", "data-cleaning")
BENCHMARK = os.getenv("OSWORLD_BENCHMARK", "osworld")

MAX_STEPS = 10
NUM_EPISODES = 3

def sanitize_payload(payload_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for k, v in payload_dict.items():
        if isinstance(v, str) and v:
            if k == "code":
                v = v.strip()
                if v.startswith("```python"):
                    v = v[len("```python"):].strip()
                elif v.startswith("```"):
                    v = v[len("```"):].strip()
                if v.endswith("```"):
                    v = v[:-3].strip()
            elif k in ["filename", "column", "value"]:
                v = v.split('}')[0].split(']')[0].strip().strip('"`')
        cleaned[k] = v
    return cleaned

def log_start(task: str, env: str, model: str, ep: int, total: int) -> None:
    print(f"\n{C_BLUE}{C_BOLD}", flush=True)
    print(f" EPISODE {ep}/{total} | TASK: {task} | ENV: {env}", flush=True)
    print(f" MODEL: {model}", flush=True)
    print(f"{C_RESET}", flush=True)
    # The [START] tag must be exactly as required for the automated grader
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str], score: float) -> None:
    # Human-friendly report
    status = f"{C_GREEN}SUCCESS{C_RESET}" if done and score >= 1.0 else f"{C_YELLOW}STEPPING{C_RESET}"
    if error:
        status = f"{C_RED}ERROR{C_RESET}"
    
    print(f" {C_CYAN}Step {step:02d}{C_RESET} | {C_BOLD}{action:15}{C_RESET} | {draw_phi_bar(score)} | Reward: {reward:+.2f} | {status}", flush=True)
    
    # The [STEP] tag must be exactly as required for the automated grader
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_str = action.replace('\n', '\\n').replace('"', "'")
    print(
        f"[STEP] step={step} action=\"{action_str}\" reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    result_color = C_GREEN if success else C_RED
    result_str = "PASSED" if success else "FAILED"
    total_reward = sum(rewards)
    
    print(f"{C_BOLD}--- Episode Statistics ---{C_RESET}", flush=True)
    print(f"Result: {result_color}{result_str}{C_RESET} | Steps: {steps} | Phi Score: {score:.3f} | Total Reward: {total_reward:+.2f}", flush=True)
    
    # The [END] tag must be exactly as required for the automated grader
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}\n", flush=True)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Allow containerized mode or local server mode
    if LOCAL_IMAGE_NAME:
        env_client = await OsworldEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        # Fallback to sync local server if not using docker
        env_client = OsworldEnv(base_url="http://localhost:8000").sync()

    for episode in range(1, NUM_EPISODES + 1):
        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        success = False
        final_score = 0.0

        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME, ep=episode, total=NUM_EPISODES)

        try:
            # Check if env is async or sync wrapper
            if hasattr(env_client, "reset_async"):
                result = await env_client.reset_async()
            elif asyncio.iscoroutinefunction(env_client.reset):
                result = await env_client.reset()
            else:
                result = env_client.reset()

            obs = result.observation
            done = result.done

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break
                
                obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict()
                prompt = f"""
You are a specialized Data Engineering Agent. You must resolve data quality issues programmatically.
Here is your current observation:
{json.dumps(obs_dict, indent=2)}

Your goal is to transform the provided datasets to match a rigorous clean structure. You are evaluated on:
- Content Accuracy: Ensuring precision and recall of the data records.
- Schema Integrity: Validating column names, types, and strict ordering.
- Data Validity: Eliminating nulls, standardizing formats, and correcting types.
- Constraint Satisfaction: Enforcing uniqueness and logical range boundaries.

ENVIRONMENT REWARDS & REASONING:
- This is a Reinforcement Learning environment. You receive positive rewards for incremental progress toward the clean state.
- Efficiency is critical. Your evaluation score is scaled by how few actions you take to reach the goal.
- Catastrophic data loss (e.g., dropping the majority of rows accidentally) results in a heavy negative penalty.
- Professional methodology is incentivized. It is highly recommended to inspect the data and schema before committing to permanent changes.

Available Action Types:
1. "inspect_schema": Check column names and types. Use "filename" in payload.
2. "view_head": Look at the first N rows. Use "filename" and "n" (default 5) in payload.
3. "read_file": Read the entire file content. Use "filename" in payload.
4. "preview_changes": Test your "code" without saving changes. (Zero risk, high transparency).
5. "execute_python": Perform permanent file mutations via Python code.
6. "remove_duplicates": Utility to deduplicate a file. Use "filename".
7. "fill_nulls": Utility to fill missing values. Use "filename" and "value".

PYTHON EXECUTION RULES:
For `execute_python` and `preview_changes`, your code runs in a sandboxed `exec()` environment.
- You have access to a `files` dictionary containing the file strings.
- You MUST read and write the CSV via this dict: `files["data.csv"]`
- You MUST put your python code inside the 'code' field of the JSON payload.
- Example JSON pattern for execute_python:
  {{
    "action_type": "execute_python",
    "payload": {{
      "code": "import pandas as pd\\nimport io\\ndf = pd.read_csv(io.StringIO(files['data.csv']))\\nfiles['data.csv'] = df.to_csv(index=False)\\n"
    }}
  }}

Decide on the next action. Your response must be a single, valid JSON object matching the schema exactly.
"""
                user_msg = {"role": "user", "content": prompt}
                messages = [{"role": "system", "content": "You are a professional data cleaning engineer."}]
                # We limit history to recent to avoid massive token bloat
                if len(history) > 4:
                    messages.extend(history[-4:])
                else:
                    messages.extend(history)
                messages.append(user_msg)

                action_str = ""
                error = None
                action = None

                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        response_format={"type": "json_object"},
                    )
                    content = response.choices[0].message.content
                    history.append(user_msg)
                    history.append({"role": "assistant", "content": content})

                    try:
                        parsed_json = json.loads(content)
                        # Map to our models to validate
                        llm_action = LLMAction(**parsed_json)
                        payload_dict = llm_action.payload.model_dump(exclude_none=True)
                        payload_dict = sanitize_payload(payload_dict)
                        
                        action_str = llm_action.action_type
                        action = OsworldAction(
                            action_type=llm_action.action_type,
                            payload=payload_dict
                        )
                    except json.JSONDecodeError:
                        error = "Model generated invalid JSON"
                        action_str = "pass"
                        action = OsworldAction(action_type="pass", payload={})

                except Exception as e:
                    error_msg = str(e)
                    if "402" in error_msg or "credits" in error_msg:
                        error = "Credits Depleted (402)"
                        print(f"  {C_RED} Error: API Credits Depleted. Stopping episode.{C_RESET}", flush=True)
                    else:
                        error = f"Model query error: {error_msg[:50]}..."
                    action_str = "pass"
                    action = OsworldAction(action_type="pass", payload={})

                # Step in environment
                if hasattr(env_client, "step_async"):
                    result = await env_client.step_async(action)
                elif asyncio.iscoroutinefunction(env_client.step):
                    result = await env_client.step(action)
                else:
                    result = env_client.step(action)

                obs = result.observation
                done = result.done
                reward = result.reward

                if "Error" in obs.screen_text:
                    error = "Execution Error"

                rewards.append(reward)
                steps_taken = step
                final_score = getattr(obs, "score", 0.0)

                log_step(step=step, action=action_str, reward=reward, done=done, error=error, score=final_score)

            success = final_score >= 1.0

        except Exception as ep_error:
            # If the episode fatally crashed
            import traceback
            traceback.print_exc()
        finally:
            log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    if hasattr(env_client, "close"):
        if asyncio.iscoroutinefunction(env_client.close):
            await env_client.close()
        else:
            env_client.close()

if __name__ == "__main__":
    asyncio.run(main())
