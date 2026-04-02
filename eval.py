"""
Evaluation script for the OsWorld Data Cleaning Environment.

Tests:
1. Grader sanity    — perfect/partial/wrong scores
2. Anti-exploit     — delete rows, add junk, wrong schema
3. Reward behavior  — improvement/no-op/regression
4. Difficulty order — easy < medium < hard gap
"""

import sys
import os

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "server"))

from tasks import TASK_REGISTRY, TaskConfig
from graders import SemanticGrader
from rewards import RewardCalculator
from models import TaskLevel

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    icon = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {icon} {label}  {detail}")


# ─────────────────────────────────────────────
# 1. GRADER SANITY
# ─────────────────────────────────────────────
def test_grader_sanity():
    grader = SemanticGrader()
    print("\n====== GRADER SANITY ======\n")

    for level in TaskLevel:
        tasks = TASK_REGISTRY[level]
        for i, task in enumerate(tasks):
            tag = task.screen_text.split(": ")[-1].rstrip(".")
            print(f"[{level.value} v{i}] {tag}")

            # Perfect
            perfect_csv = task.expected_df.to_csv(index=False)
            ps = grader.get_score({"data.csv": perfect_csv}, task.expected_df, task.constraints)
            check("Perfect score", ps >= 0.95, f"Φ={ps:.4f}")

            # Dirty (initial)
            ds = grader.get_score(task.files, task.expected_df, task.constraints)
            check("Dirty < perfect", ds < ps, f"Φ={ds:.4f}")

            # Empty
            es = grader.get_score({"data.csv": ""}, task.expected_df, task.constraints)
            check("Empty ≈ 0", es <= 0.05, f"Φ={es:.4f}")
            print()


# ─────────────────────────────────────────────
# 2. ANTI-EXPLOIT
# ─────────────────────────────────────────────
def test_anti_exploit():
    grader = SemanticGrader()
    task = TASK_REGISTRY[TaskLevel.EASY][0]  # duplicate removal (4 expected rows)
    print("====== ANTI-EXPLOIT (easy dup task) ======\n")

    # Headers only
    s = grader.get_score({"data.csv": "id,name\n"}, task.expected_df, task.constraints)
    check("Delete all rows", s <= 0.25, f"Φ={s:.4f}")

    # Junk rows
    s = grader.get_score(
        {"data.csv": "id,name\n1,alice\n2,bob\n99,fake\n98,faker\n97,fakest\n"},
        task.expected_df, task.constraints,
    )
    check("Add junk rows", s < 0.95, f"Φ={s:.4f}")

    # Wrong schema
    s = grader.get_score(
        {"data.csv": "x,y\n1,alice\n2,bob\n"},
        task.expected_df, task.constraints,
    )
    check("Wrong schema", s <= 0.4, f"Φ={s:.4f}")

    # Partial output
    s = grader.get_score(
        {"data.csv": "id,name\n1,alice\n"},
        task.expected_df, task.constraints,
    )
    check("Partial (1/4 rows)", 0.3 < s < 0.95, f"Phi={s:.4f}")

    # Duplicated correct rows
    s = grader.get_score(
        {"data.csv": "id,name\n1,alice\n1,alice\n2,bob\n2,bob\n"},
        task.expected_df, task.constraints,
    )
    check("Dup correct rows", s < 0.95, f"Φ={s:.4f}")
    print()


# ─────────────────────────────────────────────
# 3. REWARD BEHAVIOR
# ─────────────────────────────────────────────
def test_reward_behavior():
    calc = RewardCalculator()
    print("====== REWARD BEHAVIOR ======\n")

    r = calc.calculate(0.3, 0.6, done=False)
    check("Improvement +", r > 0, f"R={r:+.4f}")

    r = calc.calculate(0.5, 0.5, done=False)
    check("No-op negative", r < 0, f"R={r:+.4f}")

    r = calc.calculate(0.6, 0.3, done=False)
    check("Regression <<0", r < -0.3, f"R={r:+.4f}")

    r = calc.calculate(0.8, 1.0, done=True)
    check("Terminal bonus", r > 5.0, f"R={r:+.4f}")

    r = calc.calculate(0.5, 0.5, done=False, is_error=True)
    check("Error penalty", r < -0.2, f"R={r:+.4f}")
    print()


# ─────────────────────────────────────────────
# 4. DIFFICULTY ORDERING
# ─────────────────────────────────────────────
def test_difficulty_ordering():
    grader = SemanticGrader()
    print("====== DIFFICULTY ORDERING ======\n")

    avg_gaps = {}
    for level in TaskLevel:
        tasks = TASK_REGISTRY[level]
        gaps = []
        for i, task in enumerate(tasks):
            initial = grader.get_score(task.files, task.expected_df, task.constraints)
            perfect_csv = task.expected_df.to_csv(index=False)
            perfect = grader.get_score({"data.csv": perfect_csv}, task.expected_df, task.constraints)
            gap = perfect - initial
            gaps.append(gap)
            print(f"  {level.value:8s} v{i}: initial={initial:.4f}  perfect={perfect:.4f}  gap={gap:.4f}")
        avg_gaps[level] = sum(gaps) / len(gaps)

    print()
    check(
        "Easy gap ≤ Medium gap",
        avg_gaps[TaskLevel.EASY] <= avg_gaps[TaskLevel.MEDIUM] + 0.05,
        f"E={avg_gaps[TaskLevel.EASY]:.3f} M={avg_gaps[TaskLevel.MEDIUM]:.3f}",
    )
    check(
        "Medium gap ≤ Hard gap",
        avg_gaps[TaskLevel.MEDIUM] <= avg_gaps[TaskLevel.HARD] + 0.05,
        f"M={avg_gaps[TaskLevel.MEDIUM]:.3f} H={avg_gaps[TaskLevel.HARD]:.3f}",
    )
    print()


# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_grader_sanity()
    test_anti_exploit()
    test_reward_behavior()
    test_difficulty_ordering()

    total = PASS + FAIL
    print(f"DONE: {PASS}/{total} passed, {FAIL} failed.\n")
    sys.exit(1 if FAIL > 0 else 0)
