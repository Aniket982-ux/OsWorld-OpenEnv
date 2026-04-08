import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from client import OsworldEnv
from models import OsworldAction

load_dotenv()


class Payload(BaseModel):
    filename: str | None = None
    n: int | None = None
    column: str | None = None
    value: str | None = None
    code: str | None = None


class LLMAction(BaseModel):
    action_type: str
    payload: Payload = Payload()


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_NAME = os.getenv("OSWORLD_TASK", "data-cleaning")
BENCHMARK = os.getenv("OSWORLD_BENCHMARK", "osworld")

NUM_EPISODES = int(os.getenv("NUM_EPISODES", 15))

MAX_STEPS = 10
TEMPERATURE = 0.0
MAX_TOKENS = 2048


def sanitize_payload(payload_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for k, v in payload_dict.items():
        if isinstance(v, str):
            v = v.strip()
            if k == "code":
                if v.startswith("```python"):
                    v = v[len("```python"):].strip()
                elif v.startswith("```"):
                    v = v[len("```"):].strip()
                if v.endswith("```"):
                    v = v[:-3].strip()
            elif k in {"filename", "column", "value"}:
                v = v.split("}")[0].split("]")[0].strip().strip('"`')
        cleaned[k] = v
    return cleaned


def compact_action_string(action_type: str, payload: Dict[str, Any]) -> str:
    obj = {"action_type": action_type, "payload": payload}
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error is not None else "null"
    action_str = action.replace("\n", "\\n")
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def extract_env_error(result: Any) -> Optional[str]:
    for attr in ("last_action_error", "error", "message"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if val:
                return str(val)

    obs = getattr(result, "observation", None)
    if obs is not None:
        for attr in ("last_action_error", "error"):
            if hasattr(obs, attr):
                val = getattr(obs, attr)
                if val:
                    return str(val)

        screen_text = getattr(obs, "screen_text", None)
        if isinstance(screen_text, str) and "Error" in screen_text:
            return screen_text

    return None


def build_prompt(obs_dict: Dict[str, Any], history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        You are a specialized Data Engineering Agent. You must resolve data quality issues programmatically.
        
        Here is your current observation:
        {json.dumps(obs_dict, indent=2, ensure_ascii=False)}

        Previous actions:
        {history_block}

        Your goal is to transform the provided datasets to match a rigorous clean structure. You are evaluated on:
        - Content Accuracy: Ensuring precision and recall of the data records.
        - Schema Integrity: Validating column names, types, and strict ordering.
        - Data Validity: Eliminating nulls, standardizing formats, and correcting types.
        - Constraint Satisfaction: Enforcing uniqueness and logical range boundaries.

        ENVIRONMENT REWARDS & REASONING:
        - This is a Reinforcement Learning environment. You receive positive rewards for incremental progress toward the clean state.
        - Efficiency is critical. Your final evaluation score is scaled by how few actions you take to reach the goal.
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
        - Example Pattern:
          ```python
          import pandas as pd
          import io
          df = pd.read_csv(io.StringIO(files['data.csv']))
          # apply fixes...
          files['data.csv'] = df.to_csv(index=False)
          ```

        Decide the next action. Return exactly one valid JSON object with this schema:
        {{
          "action_type": "<string>",
          "payload": {{
            "filename": "<string or null>",
            "n": <int or null>,
            "column": "<string or null>",
            "value": "<string or null>",
            "code": "<string or null>"
          }}
        }}
        """
    ).strip()


def get_model_action(client: OpenAI, obs_dict: Dict[str, Any], history: List[str]) -> OsworldAction:
    prompt = build_prompt(obs_dict, history)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a professional data cleaning engineer."},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content or "{}"
    
    # Strip markdown fences if the model included them
    content = content.strip()
    if content.startswith("```"):
        # Remove the first line (e.g. ```json)
        content = content.split("\n", 1)[-1]
    if content.endswith("```"):
        # Remove the last line
        content = content.rsplit("\n", 1)[0]
    content = content.strip()
    
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fallback if parsing completely fails
        parsed = {"action_type": "pass", "payload": {}}

    llm_action = LLMAction(**parsed)
    
    if hasattr(llm_action.payload, "model_dump"):
        raw_payload = llm_action.payload.model_dump(exclude_none=True)
    elif hasattr(llm_action.payload, "dict"):
        raw_payload = llm_action.payload.dict(exclude_none=True)
    elif isinstance(llm_action.payload, dict):
        raw_payload = {k: v for k, v in llm_action.payload.items() if v is not None}
    else:
        raw_payload = llm_action.payload
        
    payload_dict = sanitize_payload(raw_payload)

    return OsworldAction(
        action_type=llm_action.action_type,
        payload=payload_dict,
    )


async def make_env() -> Any:
    if LOCAL_IMAGE_NAME:
        return await OsworldEnv.from_docker_image(LOCAL_IMAGE_NAME)

    env_url = os.getenv("OSWORLD_API_URL", "https://aniket2886-osworld.hf.space")
    env = OsworldEnv(base_url=env_url)
    if hasattr(env, "sync"):
        return env.sync()
    return env


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = None
    try:
        env = await make_env()

        for episode in range(1, NUM_EPISODES + 1):
            history: List[str] = []
            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False

            log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
            
            try:
                if hasattr(env, "reset_async"):
                    result = await env.reset_async()
                else:
                    reset_fn = env.reset
                    result = await reset_fn() if asyncio.iscoroutinefunction(reset_fn) else reset_fn()

                done = bool(getattr(result, "done", False))
                obs = getattr(result, "observation", None)

                for step in range(1, MAX_STEPS + 1):
                    if done:
                        break

                    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
                    try:
                        action = get_model_action(client, obs_dict, history)
                        action_type = action.action_type
                        
                        payload_obj = action.payload
                        if hasattr(payload_obj, "model_dump"):
                            payload_dict = payload_obj.model_dump(exclude_none=True)
                        elif hasattr(payload_obj, "dict"):
                            payload_dict = payload_obj.dict(exclude_none=True)
                        else:
                            payload_dict = payload_obj
                        
                        payload_dict = sanitize_payload(payload_dict)
                        env_action = OsworldAction(action_type=action_type, payload=payload_dict)
                        action_str = compact_action_string(action_type, payload_dict)
                    except Exception as e:
                        env_action = OsworldAction(action_type="pass", payload={})
                        action_str = compact_action_string("pass", {})

                    if hasattr(env, "step_async"):
                        result = await env.step_async(env_action)
                    else:
                        step_fn = env.step
                        result = await step_fn(env_action) if asyncio.iscoroutinefunction(step_fn) else step_fn(env_action)

                    obs = getattr(result, "observation", None)
                    done = bool(getattr(result, "done", False))
                    reward = float(getattr(result, "reward", 0.0) or 0.0)

                    rewards.append(reward)
                    steps_taken = step

                    error = extract_env_error(result)

                    if obs is not None and hasattr(obs, "screen_text") and isinstance(obs.screen_text, str) and "Error" in obs.screen_text:
                        error = error or obs.screen_text

                    score = float(getattr(obs, "score", score) or score)
                    score = max(0.0, min(1.0, score))

                    log_step(step=step, action=action_str, reward=reward, done=done, error=error)

                    history.append(f"Step {step}: {action_str} -> reward {reward:.2f}")

                    if done:
                        break

                score = max(0.0, min(1.0, float(score)))
                success = score >= 1.0

            except Exception as exc:
                import sys
                print(f"[SYSTEM ERROR] {exc}", file=sys.stderr)
                success = False
            finally:
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    except Exception as exc:
        import sys
        print(f"[SYSTEM ERROR] {exc}", file=sys.stderr)
    finally:
        try:
            if env is not None and hasattr(env, "close"):
                close_fn = env.close
                if asyncio.iscoroutinefunction(close_fn):
                    await close_fn()
                else:
                    close_fn()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())