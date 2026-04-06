import os
import json
import openai
from client import OsworldEnv
from models import OsworldAction
from pydantic import BaseModel
from typing import Dict, Any

class Payload(BaseModel):
    # Defining a specific schema instead of Dict[str, Any] which breaks strict structured parsing
    filename: str | None = None
    n: int | None = None
    column: str | None = None
    value: str | None = None
    code: str | None = None

def sanitize_payload(payload_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip trailing JSON syntax or markdown junk from string fields."""
    cleaned = {}
    for k, v in payload_dict.items():
        if isinstance(v, str) and v:
            # If it's the code field, firmly strip markdown fences
            if k == "code":
                v = v.strip()
                if v.startswith("```python"):
                    v = v[len("```python"):].strip()
                elif v.startswith("```"):
                    v = v[len("```"):].strip()
                if v.endswith("```"):
                    v = v[:-3].strip()
            # For purely string filenames, strip hallucinated JSON closures
            elif k in ["filename", "column", "value"]:
                v = v.split('}')[0].split(']')[0].strip().strip('"`')
        cleaned[k] = v
    return cleaned

class LLMAction(BaseModel):
    action_type: str
    payload: Payload

# See `baseline.py` for episode logic
NUM_EPISODES = 15

# Change this to the Ollama model you decided to download
OLLAMA_MODEL = "qwen2.5-coder:7b"

def main():
    # Initialize the OpenAI client to point to Local Ollama Instance
    try:
        openai_client = openai.OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama", # Local Ollama does not require an API key, but the client expects a non-empty string.
        )
    except openai.OpenAIError as e:
        print(f"Client initialization error: {e}")
        return

    # Connect to the local environment server globally
    env = OsworldEnv(base_url="http://localhost:8000").sync()

    for episode in range(1, NUM_EPISODES + 1):
        print(f"\n==========================================")
        print(f"Episode {episode} / {NUM_EPISODES}")
        print(f"==========================================")
        
        result = env.reset()
        obs = result.observation
        done = result.done

        print(f"Task: {obs.current_task}")
        print(f"Initial Score: {obs.score:.4f}")

        step = 0
        history = []
        
        while not done:
            obs_dict = obs.model_dump() if hasattr(obs, 'model_dump') else obs.dict()
            prompt = f"""
You are an expert data cleaning bot. Here is your current observation:
{json.dumps(obs_dict, indent=2)}

You must solve the current_task by writing Python code to fix the dataset.
Your goal is to transform the CSV semantically to match clean structures exactly.

The environment scores you on 4 components:
- Content (40%): Are the correct rows present? (F1: precision + recall)
- Schema (20%): Are column names correct and in the right order?
- Validity (20%): No nulls in required fields, correct types, clean formatting
- Constraints (20%): Unique IDs where required, values in valid ranges

Available Action Types:
1. "inspect_schema": Check column names and types. Use "filename" in payload.
2. "view_head": Look at the first N rows. Use "filename" and "n" (default 5) in payload.
3. "read_file": Read the entire file content. Use "filename" in payload.
4. "preview_changes": Test your "code" without saving changes. High transparency, zero risk.
5. "execute_python": Run your "code" and PERMANENTLY update the files.
6. "remove_duplicates": Quick tool for row deduplication. Use "filename".
7. "fill_nulls": Quick tool to fill missing values. Use "filename" and "value".

PYTHON EXECUTION RULES:
For `execute_python` and `preview_changes`, your code runs in a sandboxed `exec()` environment.
- You have access to a `files` dictionary containing the file strings.
- You MUST read and write the CSV via this dict: `files["data.csv"]`
- You MUST put your python code inside the 'code' field of the JSON payload.
- Example JSON pattern for execute_python:
  {{
    "action_type": "execute_python",
    "payload": {{
      "code": "import pandas as pd\nimport io\ndf = pd.read_csv(io.StringIO(files['data.csv']))\nfiles['data.csv'] = df.to_csv(index=False)\n"
    }}
  }}

Strategic Tips:
- Use "inspect_schema" or "view_head" FIRST to diagnose the problem. There is an "Inspect-First Bonus" (+0.05) to your reward for doing this.
- Use "preview_changes" if you are unsure about your Pandas logic.
- When ready, use "execute_python" with clear, vectorized Pandas code to solve the task.

Decide on the next action to progress data cleaning.

CRITICAL: Your response must be a single, valid JSON object matching the requested schema. 
Do not include any words, markdown backticks, or thoughts outside the JSON.
"""
            user_msg = {"role": "user", "content": prompt}
            messages = [{"role": "system", "content": "You are a professional data cleaning engineer."}]
            messages.extend(history)
            messages.append(user_msg)

            try:
                # Using the OpenAI python SDK's built-in structured output parsing
                # Ollama supports this format seamlessly on newer versions!
                response = openai_client.beta.chat.completions.parse(
                    model=OLLAMA_MODEL,
                    messages=messages,
                    response_format=LLMAction,
                )
                
                history.append(user_msg)
                history.append({"role": "assistant", "content": response.choices[0].message.content})

                llm_action = response.choices[0].message.parsed
                payload_dict = llm_action.payload.model_dump(exclude_none=True)
                payload_dict = sanitize_payload(payload_dict)
                
                action = OsworldAction(
                    action_type=llm_action.action_type,
                    payload=payload_dict
                )
            except Exception as e:
                print(f"Failed to query model: {e}")
                action = OsworldAction(
                    action_type="pass",
                    payload={}
                )
                
            step_result = env.step(action)
            obs = step_result.observation
            done = step_result.done
            reward = step_result.reward
            step += 1
            
            print(f"Step {step} | Action: {action.action_type} | Reward: {reward:+.4f} | Score: {obs.score:.4f}")
            
            if done:
                final_score = getattr(obs, "score", 0.0)
                print(f"\n--- Episode {episode} Finished ---")
                print(f"Final Score: {final_score:.4f}")

    if hasattr(env, "close"):
        env.close()

if __name__ == "__main__":
    main()
