"""
Task definitions for the OsWorld Data Cleaning Environment.

6 task variants grouped into 3 difficulty tiers:
- Easy   (2 issues): Duplicate Removal, Format Normalization
- Medium (3-5 issues): Missing Values, Schema Repair, Constraint Enforcement
- Hard   (7+ issues): Corrupted Pipeline Recovery

Difficulty reflects the NUMBER and COMPLEXITY of distinct operations the agent
must perform. All tasks start with dirty score ~0 because wrong column names
cause every grader component to fail — forcing the agent to earn every point.

IMPORTANT: Task descriptions hint at SYMPTOMS only — the agent must diagnose
and fix issues itself using execute_python actions.
"""

import pandas as pd
from typing import Dict, Any, List
from dataclasses import dataclass

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


# ─────────────────────────────────────────────
# EASY TASKS (2 issues each, 5-6 rows)
# Primary challenge: fix columns + ONE content issue
# ─────────────────────────────────────────────

EASY_TASKS: List[TaskConfig] = [
    # Variant 0: Duplicate Removal
    # Issues: (1) column casing wrong: ID->id, Name->name
    #         (2) two duplicate rows that must be removed
    # Agent must: rename columns lowercase, drop duplicates
    TaskConfig(
        files={"data.csv": (
            "ID,Name\n"
            "1,alice\n"
            "2,bob\n"
            "3,charlie\n"
            "4,dave\n"
            "2,bob\n"
            "3,charlie\n"
        )},
        screen_text="Easy task loaded. Inspect data.csv and clean it.",
        task_description=(
            "The file data.csv has data quality problems. "
            "Some records appear to be repeated. "
            "Inspect it and produce a clean, deduplicated version. "
            "Standardize column names to lowercase 'id' and 'name'."
        ),
        expected_df=pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["alice", "bob", "charlie", "dave"],
        }),
        constraints={
            "expected_cols": ["id", "name"],
            "expected_col_order": True,
            "unique_cols": ["id"],
            "no_null_cols": ["id", "name"],
        },
    ),

    # Variant 1: Format Normalization
    # Issues: (1) column casing wrong: ID->id, Name->name
    #         (2) name values have inconsistent whitespace and letter-casing
    # Agent must: rename columns lowercase, strip whitespace and lowercase all names
    TaskConfig(
        files={"data.csv": (
            "ID,Name\n"
            "1, Alice\n"
            "2,BOB\n"
            "3, Charlie\n"
            "4,  dave\n"
            "5,Eve \n"
        )},
        screen_text="Easy task loaded. Inspect data.csv and clean it.",
        task_description=(
            "The file data.csv has data quality problems. "
            "Names appear to have inconsistent formatting. "
            "Inspect it and produce a uniformly formatted, clean version. "
            "Standardize column names to lowercase 'id' and 'name'."
        ),
        expected_df=pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["alice", "bob", "charlie", "dave", "eve"],
        }),
        constraints={
            "expected_cols": ["id", "name"],
            "expected_col_order": True,
            "unique_cols": ["id"],
            "no_null_cols": ["id", "name"],
        },
    ),
]

# ─────────────────────────────────────────────
# MEDIUM TASKS (3-5 issues each, 5-6 rows)
# Primary challenge: fix columns + extra col + ONE structural/content issue
# ─────────────────────────────────────────────

MEDIUM_TASKS: List[TaskConfig] = [
    # Variant 0: Missing Value Imputation
    # Issues: (1) column casing wrong: Id->id, Val->val
    #         (2) extra column 'extra' must be dropped
    #         (3) missing values in 'val' -- fill with 0 (default for missing count)
    # Agent must: rename columns, drop extra col, fill nulls with 0
    TaskConfig(
        files={"data.csv": (
            "Id,Val,extra\n"
            "1,10,junk\n"
            "2,,junk\n"
            "3,30,junk\n"
            "4,,junk\n"
            "5,50,junk\n"
            "6,60,junk\n"
        )},
        screen_text="Medium task loaded. Inspect data.csv and clean it.",
        task_description=(
            "The file data.csv has data quality problems. "
            "Some values are missing and there is a redundant column. "
            "Treat any missing numeric values as 0. "
            "Inspect it and produce a clean, well-typed version. "
            "Standardize column names to lowercase 'id' and 'val'."
        ),
        expected_df=pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "val": [10, 0, 30, 0, 50, 60],
        }),
        constraints={
            "expected_cols": ["id", "val"],
            "expected_col_order": True,
            "unique_cols": ["id"],
            "no_null_cols": ["id", "val"],
        },
    ),

    # Variant 1: Schema Repair
    # Issues: (1) IDENTIFIER -> id  (completely non-standard column name)
    #         (2) VALUE -> val
    #         (3) extra column 'flag' must be dropped
    # Agent must: rename all columns to match schema, drop extra
    TaskConfig(
        files={"data.csv": (
            "IDENTIFIER,VALUE,flag\n"
            "1,10,y\n"
            "2,20,y\n"
            "3,30,y\n"
            "4,40,y\n"
            "5,50,y\n"
        )},
        screen_text="Medium task loaded. Inspect data.csv and clean it.",
        task_description=(
            "The file data.csv has data quality problems. "
            "Columns use non-standard names and there is an extraneous column. "
            "Inspect it and produce a clean, well-structured version. "
            "Standardize column names to lowercase 'id' and 'val'."
        ),
        expected_df=pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "val": [10, 20, 30, 40, 50],
        }),
        constraints={
            "expected_cols": ["id", "val"],
            "expected_col_order": True,
            "unique_cols": ["id"],
            "no_null_cols": ["id", "val"],
        },
    ),

    # Variant 2: Constraint Enforcement
    # Issues: (1) column casing wrong: Id->id, Val->val
    #         (2) extra column 'extra' must be dropped
    #         (3) duplicate id=1 -- keep first occurrence (val=10)
    #         (4) val=150 violates upper bound -- clamp to 100
    #         (5) val=-10 violates lower bound -- clamp to 0
    #         (6) val=200 violates upper bound -- clamp to 100
    # Agent must: rename, drop extra col, dedup (keep first), clamp val to [0,100]
    TaskConfig(
        files={"data.csv": (
            "Id,Val,extra\n"
            "1,10,junk\n"
            "1,25,junk\n"
            "2,150,junk\n"
            "3,-10,junk\n"
            "4,80,junk\n"
            "5,200,junk\n"
        )},
        screen_text="Medium task loaded. Inspect data.csv and clean it.",
        task_description=(
            "The file data.csv has data quality problems. "
            "There are duplicate records, out-of-range values, and a redundant column. "
            "Valid values for 'val' must be in the range [0, 100]. "
            "Inspect it and produce a clean, constraint-valid version. "
            "Standardize column names to lowercase 'id' and 'val'."
        ),
        expected_df=pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "val": [10, 100, 0, 80, 100],
        }),
        constraints={
            "expected_cols": ["id", "val"],
            "expected_col_order": True,
            "unique_cols": ["id"],
            "no_null_cols": ["id", "val"],
            "range_constraints": {"val": (0, 100)},
        },
    ),
]

# ─────────────────────────────────────────────
# HARD TASKS (7+ issues combined, 8 rows)
# All dimensions broken simultaneously
# ─────────────────────────────────────────────

HARD_TASKS: List[TaskConfig] = [
    # Variant 0: Corrupted Pipeline Recovery
    # Issues: (1) ID -> id   (column rename)
    #         (2) Name -> name (column rename)
    #         (3) Value -> val (column rename)
    #         (4) extra column 'extra' must be dropped
    #         (5) duplicate id=1 (Alice and ALICE are same after normalization)
    #         (6) all name values have wrong whitespace and letter-casing
    #         (7) val=None for id=4 -- fill with 0
    #         (8) val=150 for id=3 -- clamp to 100
    #         (9) val=-5 for id=5  -- clamp to 0
    # Agent must: rename 3 cols, drop extra, dedup, fix name formatting,
    #             fill null val, clamp out-of-range vals
    TaskConfig(
        files={
            "data.csv": (
                "ID,Name,Value,extra\n"
                "1, Alice ,10,junk\n"
                "1, ALICE ,10,junk\n"
                "2, Bob ,40,junk\n"
                "3,CHARLIE ,150,junk\n"
                "4, dave,,junk\n"
                "5,  EVE,-5,junk\n"
                "6,frank ,80,junk\n"
                "7, Grace ,90,junk\n"
            )
        },
        screen_text="Hard task loaded. Inspect data.csv and clean it.",
        task_description=(
            "The file data.csv has data quality problems. "
            "Inspect it thoroughly — there are multiple structural, content, "
            "and constraint issues. Valid values for 'val' must be in [0, 100]. "
            "Missing numeric values should default to 0. "
            "Produce a fully clean, well-structured version. "
            "Standardize column names to lowercase 'id', 'name', and 'val'."
        ),
        expected_df=pd.DataFrame({
            "id":   [1,       2,     3,         4,      5,     6,       7],
            "name": ["alice", "bob", "charlie", "dave", "eve", "frank", "grace"],
            "val":  [10,      40,    100,        0,      0,    80,      90],
        }),
        constraints={
            "expected_cols": ["id", "name", "val"],
            "expected_col_order": True,
            "unique_cols": ["id"],
            "no_null_cols": ["id", "name", "val"],
            "range_constraints": {"val": (0, 100)},
        },
    ),
]

# ─────────────────────────────────────────────
# Task Registry & Accessors
# ─────────────────────────────────────────────

TASK_REGISTRY: Dict[TaskLevel, List[TaskConfig]] = {
    TaskLevel.EASY: EASY_TASKS,
    TaskLevel.MEDIUM: MEDIUM_TASKS,
    TaskLevel.HARD: HARD_TASKS,
}


def get_task_setup(level: TaskLevel, reset_count: int = 0) -> TaskConfig:
    """
    Returns the TaskConfig for a given level.
    Cycles through variants within each tier based on reset_count.
    Uses (reset_count - 1) so first episode maps to variant 0.
    """
    tasks = TASK_REGISTRY[level]
    variant_index = ((reset_count - 1) // 3) % len(tasks)
    return tasks[variant_index]


def get_next_level(reset_count: int) -> TaskLevel:
    """
    Cycles through task levels in order: EASY -> MEDIUM -> HARD -> EASY...
    Uses (reset_count - 1) so first episode is EASY.
    """
    levels = [TaskLevel.EASY, TaskLevel.MEDIUM, TaskLevel.HARD]
    return levels[(reset_count - 1) % 3]
