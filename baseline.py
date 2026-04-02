import os
import json
from dotenv import load_dotenv
import openai
from client import OsworldEnv
from models import OsworldAction
from pydantic import BaseModel
from typing import Dict, Any

class Payload(BaseModel):
    # Defining a specific schema instead of Dict[str, Any] which breaks strict structured parsing
    column: str | None = None
    value: str | None = None
    code: str | None = None

class LLMAction(BaseModel):
    action_type: str
    payload: Payload

load_dotenv()

# The environment auto-cycles through 6 task variants across 3 difficulty tiers:
#   Reset 1 -> Easy   v0 (2 issues: wrong col casing + duplicate rows)
#   Reset 2 -> Medium v0 (3 issues: wrong col casing + extra col + null imputation)
#   Reset 3 -> Hard   v0 (9 issues: rename 3 cols + drop extra + dedup + name fmt + null + range x2)
#   Reset 4 -> Easy   v1 (2 issues: wrong col casing + string whitespace/casing in values)
#   Reset 5 -> Medium v1 (3 issues: non-standard col names + extra col)
#   Reset 6 -> Hard   v0 (Corrupted Pipeline, wraps)
# We run 6 episodes to cover all unique variants.
NUM_EPISODES = 6

def main():
    # Initialize the OpenAI client to point to OpenRouter
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY not found in environment.")
        return

    try:
        openai_client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )
    except openai.OpenAIError as e:
        print(f"Client initialization error: {e}")
        return

    # Connect to the local environment server globally, so reset_count increments
    env = OsworldEnv(base_url="http://localhost:8000").sync()

    for episode in range(1, NUM_EPISODES + 1):
        print(f"\n==========================================")
        print(f"Episode {episode} / {NUM_EPISODES}")
        print(f"==========================================")
        
        # Reset cycles automatically through variants
        result = env.reset()
        obs = result.observation
        done = result.done

        # Print what task we got
        print(f"Task: {obs.current_task}")
        print(f"Initial Score: {obs.score:.4f}")

        step = 0
        
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

Use action_type "execute_python" with a "code" field in the payload.
Your code has access to a `files` dict. Read/write CSV via files["data.csv"].
Example:
  action_type: "execute_python"
  payload: {{"code": "import pandas as pd\\nimport io\\ndf = pd.read_csv(io.StringIO(files['data.csv']))\\n# ... clean ...\\nfiles['data.csv'] = df.to_csv(index=False)"}}

Decide on the next action to progress data cleaning. Be precise.
"""
            
            try:
                response = openai_client.beta.chat.completions.parse(
                    model="openai/gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional data cleaning engineer."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=LLMAction,
                )
                llm_action = response.choices[0].message.parsed
                payload_dict = llm_action.payload.model_dump(exclude_none=True)
                
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

    # Cleanly close the environment client after all episodes
    if hasattr(env, "close"):
        env.close()

if __name__ == "__main__":
    main()
