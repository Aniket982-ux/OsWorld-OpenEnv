"""
Task definitions for the OsWorld Data Cleaning Environment.

12 Procedurally Generated Tasks:
- Easy (4): Duplicate Removal, Format Normalization, Type Coercion, Column Rename Only
- Medium (5): Missing Value, Schema Repair, Constraint Enforcement, Multi-File Join, JSON Normalization
- Hard (3): Pipeline Recovery, Adversarial Corruption, Cascading Pipeline
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
from faker import Faker
import random

try:
    from ..models import TaskLevel
except ImportError:
    from models import TaskLevel

@dataclass
class TaskConfig:
    """Configuration for a single task variant."""
    files: Dict[str, str]
    screen_text: str
    task_description: str
    expected_df: pd.DataFrame
    constraints: Dict[str, Any]
    optimal_steps: int = 4

# 
# EASY TASKS
# 

def gen_duplicate_removal(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    random.seed(seed)

    rows, expected_rows = [], []
    for i in range(1, 5):
        name = fake.first_name().lower()
        rows.append({"ID": i, "Name": name})
        expected_rows.append({"id": i, "name": name})

    rows.append(rows[1])
    rows.append(rows[2])
    df_dirty = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return TaskConfig(
        files={"data.csv": df_dirty.to_csv(index=False)},
        screen_text="Easy task loaded. Inspect data.csv and clean it.",
        task_description="Standardize column names to 'id' and 'name'. Remove duplicate rows.",
        expected_df=pd.DataFrame(expected_rows),
        constraints={"expected_cols": ["id", "name"], "expected_col_order": True, "unique_cols": ["id"], "no_null_cols": ["id", "name"]},
        optimal_steps=2
    )

def gen_format_normalization(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    expected_rows, dirty_rows = [], []
    
    for i in range(1, 6):
        name = fake.first_name().lower()
        expected_rows.append({"id": i, "name": name})
        corrupt_name = name.upper() if i % 2 == 0 else f"  {name} " if i % 3 == 0 else name.capitalize() if i % 4 == 0 else name
        dirty_rows.append({"ID": i, "Name": corrupt_name})

    return TaskConfig(
        files={"data.csv": pd.DataFrame(dirty_rows).to_csv(index=False)},
        screen_text="Easy task loaded. Inspect data.csv and clean it.",
        task_description="Standardize column names to 'id' and 'name'. Format names to lowercase without leading/trailing whitespace.",
        expected_df=pd.DataFrame(expected_rows),
        constraints={"expected_cols": ["id", "name"], "expected_col_order": True, "unique_cols": ["id"], "no_null_cols": ["id", "name"]},
        optimal_steps=2
    )

def gen_type_coercion(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    expected_rows, dirty_rows = [], []
    
    for i in range(1, 6):
        name = fake.first_name().lower()
        age = fake.random_int(min=18, max=65)
        active = bool(fake.random_int(min=0, max=1))
        expected_rows.append({"id": i, "name": name, "age": age, "is_active": active})
        dirty_rows.append({"Identifier": i, "FullName": name, "age": f"{age} yrs", "is_active": "Yes" if active else "No"})

    return TaskConfig(
        files={"data.csv": pd.DataFrame(dirty_rows).to_csv(index=False)},
        screen_text="Easy task loaded. Inspect data.csv and clean it.",
        task_description="Standardize column names to 'id', 'name', 'age', 'is_active'. Convert 'age' out of 'X yrs' into an integer, and 'is_active' from Yes/No to standard booleans.",
        expected_df=pd.DataFrame(expected_rows),
        constraints={"expected_cols": ["id", "name", "age", "is_active"], "expected_col_order": True, "unique_cols": ["id"], "no_null_cols": ["id", "name", "age", "is_active"]},
        optimal_steps=3
    )

def gen_column_rename_only(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    expected_rows, dirty_rows = [], []
    
    for i in range(1, 6):
        name = fake.first_name().lower()
        score = fake.random_int(min=0, max=100)
        expected_rows.append({"id": i, "name": name, "score": score})
        dirty_rows.append({"Identifier": i, "StudentName": name, "TestScore": score})

    return TaskConfig(
        files={"data.csv": pd.DataFrame(dirty_rows).to_csv(index=False)},
        screen_text="Easy task loaded. Inspect data.csv and clean it.",
        task_description="This dataset is perfectly clean except for the column names. Rename 'Identifier' to 'id', 'StudentName' to 'name', and 'TestScore' to 'score'. Do not over-engineer.",
        expected_df=pd.DataFrame(expected_rows),
        constraints={"expected_cols": ["id", "name", "score"], "expected_col_order": True, "unique_cols": ["id"], "no_null_cols": ["id", "name", "score"]},
        optimal_steps=2
    )

# 
# MEDIUM TASKS
# 

def gen_missing_value_imputation(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    expected_rows, dirty_rows = [], []
    
    for i in range(1, 7):
        val = fake.random_int(min=10, max=100)
        expected_val = 0 if i in [2, 4] else val
        dirty_val = None if i in [2, 4] else val
        expected_rows.append({"id": i, "val": expected_val})
        dirty_rows.append({"Id": i, "Val": dirty_val, "extra": "junk"})

    return TaskConfig(
        files={"data.csv": pd.DataFrame(dirty_rows).to_csv(index=False)},
        screen_text="Medium task loaded. Inspect data.csv and clean it.",
        task_description="There are missing values and a redundant column 'extra'. Drop 'extra', standardise column names to 'id' and 'val', and fill missing numeric values with 0.",
        expected_df=pd.DataFrame(expected_rows),
        constraints={"expected_cols": ["id", "val"], "expected_col_order": True, "unique_cols": ["id"], "no_null_cols": ["id", "val"]},
        optimal_steps=3
    )

def gen_schema_repair(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    expected_rows, dirty_rows = [], []
    
    for i in range(1, 6):
        val = fake.random_int(min=10, max=100)
        expected_rows.append({"id": i, "val": val})
        dirty_rows.append({"IDENTIFIER": i, "VALUE": val, "flag": fake.random_element(elements=("y", "n"))})

    return TaskConfig(
        files={"data.csv": pd.DataFrame(dirty_rows).to_csv(index=False)},
        screen_text="Medium task loaded. Inspect data.csv and clean it.",
        task_description="Extraneous columns and terrible names. Standardize names to 'id' and 'val', stripping the rest.",
        expected_df=pd.DataFrame(expected_rows),
        constraints={"expected_cols": ["id", "val"], "expected_col_order": True, "unique_cols": ["id"], "no_null_cols": ["id", "val"]},
        optimal_steps=3
    )

def gen_constraint_enforcement(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    expected_rows, dirty_rows = [], []
    
    dirty_rows.append({"Id": 1, "Val": 10, "extra": "junk"})
    dirty_rows.append({"Id": 1, "Val": 25, "extra": "junk"})
    dirty_rows.append({"Id": 2, "Val": 150, "extra": "junk"})
    dirty_rows.append({"Id": 3, "Val": -10, "extra": "junk"})
    dirty_rows.append({"Id": 4, "Val": 80, "extra": "junk"})
    dirty_rows.append({"Id": 5, "Val": 200, "extra": "junk"})
    dirty_rows.append({"Id": 2, "Val": 50, "extra": "junk"})
    
    for i in range(6, 12):
        val = fake.random_int(min=0, max=100)
        dirty_rows.append({"Id": i, "Val": val, "extra": "junk"})
        
    df_dirty = pd.DataFrame(dirty_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    seen_ids = set()
    for _, row in df_dirty.iterrows():
        if row["Id"] not in seen_ids:
            seen_ids.add(row["Id"])
            expected_rows.append({"id": row["Id"], "val": max(0, min(100, row["Val"]))})

    df_expected = pd.DataFrame(expected_rows).sort_values("id").reset_index(drop=True)
    return TaskConfig(
        files={"data.csv": df_dirty.to_csv(index=False)},
        screen_text="Medium task loaded. Inspect data.csv and clean it.",
        task_description="Duplicates, out-of-bounds metrics, and extra columns. Deduplicate (keep first), bound 'val' to [0, 100], and standardize names to 'id' and 'val'.",
        expected_df=df_expected,
        constraints={"expected_cols": ["id", "val"], "expected_col_order": True, "unique_cols": ["id"], "no_null_cols": ["id", "val"], "range_constraints": {"val": (0, 100)}},
        optimal_steps=4
    )

def gen_multi_file_join(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    users, orders, expected_rows = [], [], []
    
    for i in range(1, 6):
        name = fake.first_name().lower()
        users.append({"user_id": i, "name": name})
        order_val = fake.random_int(min=10, max=100)
        orders.append({"UID": i, "order_value": order_val, "junk_col": "x"})
        expected_rows.append({"user_id": i, "name": name, "order_value": order_val})
        
    orders.append({"UID": 99, "order_value": 50, "junk_col": "y"}) # orphaned
    
    return TaskConfig(
        files={"users.csv": pd.DataFrame(users).to_csv(index=False), "orders.csv": pd.DataFrame(orders).to_csv(index=False)},
        screen_text="Medium task loaded. Two files: users.csv and orders.csv.",
        task_description="Clean orders.csv schema (UID -> user_id, drop junk_col). Join with users.csv on user_id (inner join). Save result as 'merged.csv'.",
        expected_df=pd.DataFrame(expected_rows),
        constraints={"expected_cols": ["user_id", "name", "order_value"], "expected_col_order": True, "unique_cols": ["user_id"], "no_null_cols": ["user_id", "name", "order_value"], "target_file": "merged.csv"},
        optimal_steps=5
    )

def gen_json_normalization(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    json_data = []
    expected_rows = []

    for i in range(1, 6):
        name = fake.first_name().lower()
        city = fake.city()
        val = fake.random_int(min=0, max=100)
        
        # Add deep nesting and ragged structure
        person_data = {"name": name, "location": {"city": city, "unused_meta": fake.zipcode()}}
        if i % 2 == 0:
            person_data["extra_tag"] = "test"
            
        json_data.append({
            "metadata": {"identifier": i}, 
            "profile": person_data, 
            "metrics": {"val": val}
        })
        expected_rows.append({"id": i, "name": name, "city": city, "val": val})

    return TaskConfig(
        files={"data.json": json.dumps(json_data)},
        screen_text="Medium task loaded. A deeply nested JSON file needs flattening.",
        task_description="You have 'data.json'. Flatten the nested dictionaries into a tabular dataframe. Standardize column names strictly to 'id', 'name', 'city', and 'val'. Clean out all extra metadata fields. Save to 'data.csv'.",
        expected_df=pd.DataFrame(expected_rows),
        constraints={"expected_cols": ["id", "name", "city", "val"], "expected_col_order": True, "unique_cols": ["id"], "no_null_cols": ["id", "name", "city", "val"], "target_file": "data.csv"},
        optimal_steps=4
    )

# 
# HARD TASKS
# 

def gen_pipeline_recovery(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    dirty_rows = []
    
    for i in range(1, 8):
        name = fake.first_name().lower()
        val = fake.random_int(min=0, max=100)
        
        corrupt_name = f" {name.upper()} " if i % 2 == 0 else name
        dirty_val = None if i == 4 else val + 150 if i == 5 else val - 50 if i == 6 else val
        dirty_rows.append({"ID": i, "Name": corrupt_name, "Value": dirty_val, "extra": "junk"})

    # Intentionally create a duplicate that will be identical ONLY after normalization
    dirty_rows.append({"ID": 1, "Name": f" {dirty_rows[0]['Name'].upper().strip()} ", "Value": dirty_rows[0]["Value"], "extra": "junk"})
    
    df_dirty = pd.DataFrame(dirty_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Generate expected_df by applying the exact, ideal operations on the dirty dataset
    df_expected = df_dirty.copy()
    df_expected = df_expected.drop(columns=["extra"])
    df_expected.rename(columns={"ID": "id", "Name": "name", "Value": "val"}, inplace=True)
    
    # Lowercase & strip first
    df_expected["name"] = df_expected["name"].str.strip().str.lower()
    
    # Then deduplicate based on ID
    df_expected = df_expected.drop_duplicates(subset=["id"], keep="first")
    
    # Then bounding / filling
    df_expected["val"] = df_expected["val"].fillna(0).clip(0, 100)
    
    # Sort to enforce deterministic order
    df_expected = df_expected.sort_values("id").reset_index(drop=True)

    return TaskConfig(
        files={"data.csv": df_dirty.to_csv(index=False)},
        screen_text="Hard task loaded. All dimensions broken.",
        task_description="Pipeline collapse. Fix columns (id, name, val), deduplicate, string cleaning, bounds enforcement [0,100], and null fill (0).",
        expected_df=df_expected,
        constraints={"expected_cols": ["id", "name", "val"], "expected_col_order": True, "unique_cols": ["id"], "no_null_cols": ["id", "name", "val"], "range_constraints": {"val": (0, 100)}},
        optimal_steps=6
    )

def gen_adversarial_corruption(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    expected_rows, dirty_rows = [], []

    for i in range(1, 8):
        name = fake.first_name().lower()
        age = fake.random_int(min=20, max=50)
        score = fake.random_int(min=60, max=90)
        
        expected_rows.append({"id": i, "name": name, "age": age, "score": score})
        
        # Adversarial logic: structural logic intact (no nulls, standard dtypes) but SEMANTICALLY impossible
        dirty_age = age
        dirty_score = score
        if i == 3: dirty_age = 150  # Biologically impossible for a normal dataset, clip to 100
        if i == 4: dirty_score = 105 # Out of 100 limit, clip to 100
        if i == 5: dirty_age = -5 # Negative age, clip to 0
        
        dirty_rows.append({"id": i, "name": name, "age": dirty_age, "score": dirty_score})

    df_expected = pd.DataFrame(expected_rows)
    for i, row in df_expected.iterrows():
        if i == 2: df_expected.at[i, "age"] = 100
        if i == 3: df_expected.at[i, "score"] = 100
        if i == 4: df_expected.at[i, "age"] = 0

    df_dirty = pd.DataFrame(dirty_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df_expected = df_expected.sort_values("id").reset_index(drop=True)

    return TaskConfig(
        files={"data.csv": df_dirty.to_csv(index=False)},
        screen_text="Hard task loaded. Adversarial Data context.",
        task_description="Dtypes are perfect, no duplicates. But some values are logically impossible. 'score' maximum is 100. 'age' limits are 0 to 100. Clip invalid values.",
        expected_df=df_expected,
        constraints={"expected_cols": ["id", "name", "age", "score"], "expected_col_order": True, "unique_cols": ["id"], "no_null_cols": ["id", "name", "age", "score"], "range_constraints": {"age": (0, 100), "score": (0, 100)}},
        optimal_steps=5
    )

def gen_cascading_pipeline(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    inventory, conversions, expected_rows = [], [], []

    # Requires multiple steps where step A's output impacts step B
    for i in range(1, 6):
        item = fake.word().lower()
        price_gbp = fake.random_int(min=10, max=100)
        inventory.append({"item_id": i, "item_name": f" {item.upper()} ", "price_gbp": None if i == 3 else price_gbp})
        
        expected_gbp = 0 if i == 3 else price_gbp
        expected_usd = expected_gbp * 1.5  # 1.5 conversion
        expected_rows.append({"item_id": i, "item_name": item, "price_gbp": expected_gbp, "price_usd": expected_usd})

    conversions.append({"currency": "USD", "rate": 1.5})
    
    return TaskConfig(
        files={"inventory.csv": pd.DataFrame(inventory).to_csv(index=False), "rates.csv": pd.DataFrame(conversions).to_csv(index=False)},
        screen_text="Hard task loaded. Dependent operations.",
        task_description="Clean 'item_name' string formatting. Fill null 'price_gbp' with 0. Extract conversion rate from rates.csv and apply it to create a new column 'price_usd'. Save to 'final.csv'.",
        expected_df=pd.DataFrame(expected_rows),
        constraints={"expected_cols": ["item_id", "item_name", "price_gbp", "price_usd"], "expected_col_order": True, "unique_cols": ["item_id"], "no_null_cols": ["item_id", "item_name", "price_gbp", "price_usd"], "target_file": "final.csv"},
        optimal_steps=6
    )

# 
# ETL / TEXT-BASED MULTI-FORMAT TASKS
# 

def gen_sql_extraction(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    expected_rows = []
    
    sql_lines = [
        "CREATE TABLE users (user_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT);",
        "CREATE TABLE purchases (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL);"
    ]
    
    for i in range(1, 6):
        fname = fake.first_name().lower()
        lname = fake.last_name().lower()
        amt = round(fake.random.uniform(10.5, 99.9), 2)
        
        sql_lines.append(f"INSERT INTO users (user_id, first_name, last_name) VALUES ({i}, '{fname}', '{lname}');")
        sql_lines.append(f"INSERT INTO purchases (id, user_id, amount) VALUES ({i}, {i}, {amt});")
        expected_rows.append({"id": i, "full_name": f"{fname} {lname}", "amount": amt})
        
    # Extra rogue data that shouldn't join
    sql_lines.append(f"INSERT INTO purchases (id, user_id, amount) VALUES (99, 99, 50.0);")
    
    return TaskConfig(
        files={"data.sql": "\n".join(sql_lines)},
        screen_text="Medium task loaded. SQL Dump provided.",
        task_description="You have 'data.sql'. It contains schema and inserts for 'users' and 'purchases'. Load it into an in-memory SQLite database, join the tables on user_id, combine first_name and last_name into 'full_name' (space separated), and save the result to 'data.csv'. Drop orphaned purchases.",
        expected_df=pd.DataFrame(expected_rows),
        constraints={"expected_cols": ["id", "full_name", "amount"], "expected_col_order": True, "unique_cols": ["id"], "no_null_cols": ["id", "full_name", "amount"], "target_file": "data.csv"},
        optimal_steps=4
    )

def gen_html_scraping(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    expected_rows = []
    
    html = "<html><body><h1>User Data</h1><table id='users'><tr><th>IDENTIFIER</th><th>USER NAME</th><th>AGE</th></tr>"
    for i in range(1, 6):
        name = fake.first_name().lower()
        age = fake.random_int(min=20, max=60)
        html += f"<tr><td>{i}</td><td>  {name}  </td><td>{age}</td></tr>"
        expected_rows.append({"id": i, "user_name": name, "age": age})
        
    html += "</table><div class='junk'>ignored data</div></body></html>"
    
    return TaskConfig(
        files={"data.html": html},
        screen_text="Medium task loaded. HTML Scraping task.",
        task_description="You have 'data.html' containing a messy table. Parse it (e.g. using pandas.read_html), clean up whitespace in 'USER NAME', standardize columns to 'id', 'user_name', and 'age', and convert 'id' and 'age' to integers. Save to 'data.csv'.",
        expected_df=pd.DataFrame(expected_rows),
        constraints={"expected_cols": ["id", "user_name", "age"], "expected_col_order": True, "unique_cols": ["id"], "no_null_cols": ["id", "user_name", "age"], "target_file": "data.csv"},
        optimal_steps=4
    )

def gen_log_parsing(seed: int) -> TaskConfig:
    fake = Faker()
    fake.seed_instance(seed)
    expected_rows = []
    
    logs = [
        "[INFO] System started normally.",
        "[DEBUG] Memory check OK."
    ]
    
    for i in range(1, 8):
        name = fake.first_name().lower()
        score = fake.random_int(min=0, max=100)
        logs.append(f"[METRIC] | id={i} | user={name} | score={score} | latency={fake.random_int(min=10, max=90)}ms")
        expected_rows.append({"id": i, "user": name, "score": score})
        
    logs.append("[ERROR] id=NaN user=NULL score=NaN")
    logs.append("[METRIC] | id=99 | user=rogue | score=missing | latency=10ms") # Invalid score
    random.shuffle(logs)
    
    df_expected = pd.DataFrame(expected_rows).sort_values("id").reset_index(drop=True)
    return TaskConfig(
        files={"server.log": "\n".join(logs)},
        screen_text="Hard task loaded. Unstructured log parsing.",
        task_description="Extract structured records from 'server.log'. Only process lines starting with '[METRIC]'. Extract 'id', 'user', and 'score'. Ignore lines where 'score' is not a valid integer. Save the clean tabular data to 'data.csv'.",
        expected_df=df_expected,
        constraints={"expected_cols": ["id", "user", "score"], "expected_col_order": True, "unique_cols": ["id"], "no_null_cols": ["id", "user", "score"], "target_file": "data.csv"},
        optimal_steps=5
    )

# 
# Task Registry & Accessors
# 

TASK_REGISTRY: Dict[TaskLevel, List[Callable[[int], TaskConfig]]] = {
    TaskLevel.EASY: [gen_duplicate_removal, gen_format_normalization, gen_type_coercion, gen_column_rename_only],
    TaskLevel.MEDIUM: [gen_missing_value_imputation, gen_schema_repair, gen_constraint_enforcement, gen_multi_file_join, gen_json_normalization, gen_sql_extraction, gen_html_scraping],
    TaskLevel.HARD: [gen_pipeline_recovery, gen_adversarial_corruption, gen_cascading_pipeline, gen_log_parsing],
}

def get_task_setup(level: TaskLevel, seed: int, reset_count: int = 0) -> TaskConfig:
    """
    Returns a dynamically generated TaskConfig for a given level.
    """
    tasks = TASK_REGISTRY[level]
    variant_index = ((reset_count - 1) // 3) % len(tasks)
    return tasks[variant_index](seed)

def get_next_level(reset_count: int) -> TaskLevel:
    """
    Cycles: EASY -> MEDIUM -> HARD -> EASY...
    """
    levels = [TaskLevel.EASY, TaskLevel.MEDIUM, TaskLevel.HARD]
    return levels[(reset_count - 1) % 3]
