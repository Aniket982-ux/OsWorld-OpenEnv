# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
OsWorld Environment Implementation.

Data Cleaning environment with multi-component grading,
6 task variants across 3 difficulty tiers, and reward shaping.
"""

import contextlib
import io
from typing import Dict
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import OsworldAction, OsworldObservation, TaskLevel
except ImportError:
    from models import OsworldAction, OsworldObservation, TaskLevel

try:
    from .tasks import TaskConfig, get_task_setup, get_next_level
    from .graders import SemanticGrader
    from .rewards import RewardCalculator
except ImportError:
    from tasks import TaskConfig, get_task_setup, get_next_level
    from graders import SemanticGrader
    from rewards import RewardCalculator


class OsworldEnvironment(Environment):
    """
    Data Cleaning Environment with multi-component grading.

    Supports 6 task variants across Easy/Medium/Hard tiers,
    step-wise reward shaping, and anti-cheat grading.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the Data Cleaning environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.files: Dict[str, str] = {}
        self.task_level: TaskLevel = TaskLevel.EASY
        self.task_config: TaskConfig = None
        self.screen_text = ""
        self.max_steps = 10

        # Modular components
        self.grader = SemanticGrader()
        self.reward_calculator = RewardCalculator()

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    def _current_score(self) -> float:
        """Get current Φ score using the grader."""
        return self.grader.get_score(
            self.files,
            self.task_config.expected_df,
            self.task_config.constraints,
        )

    def reset(self, options: Dict = None) -> OsworldObservation:  # type: ignore[override]
        """Reset the environment and set up a new task.

        Args:
            options: Optional dict for curriculum control.
                     Supports ``{"difficulty": "easy" | "medium" | "hard"}``
                     to override the automatic cycling schedule.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        # Allow external trainer to override difficulty via options
        if options and "difficulty" in options:
            difficulty_map = {
                "easy": TaskLevel.EASY,
                "medium": TaskLevel.MEDIUM,
                "hard": TaskLevel.HARD,
            }
            requested = options["difficulty"].lower()
            self.task_level = difficulty_map.get(requested, get_next_level(self._reset_count))
        else:
            # Automatic curriculum: EASY → MEDIUM → HARD → EASY …
            self.task_level = get_next_level(self._reset_count)

        # Get task variant (cycles within each tier)
        self.task_config = get_task_setup(self.task_level, self._reset_count)
        # Copy files so we don't mutate the task definition
        self.files = dict(self.task_config.files)
        self.screen_text = self.task_config.screen_text

        return OsworldObservation(
            screen_text=self.screen_text,
            files=self.files,
            current_task=self.task_config.task_description,
            done=False,
            reward=0.0,
            score=self._current_score(),
        )

    def step(self, action: OsworldAction) -> OsworldObservation:  # type: ignore[override]
        """Execute an action and return the observation."""
        self._state.step_count += 1
        self.screen_text = f"Executed {action.action_type}."

        is_error = False
        is_unknown = False

        old_score = self._current_score()

        # ── Action Execution ──
        if action.action_type == "execute_python":
            code = action.payload.get("code", "")
            # Snapshot files before exec so we can roll back on failure
            file_snapshot = dict(self.files)
            f = io.StringIO()
            try:
                with contextlib.redirect_stdout(f):
                    local_vars = {"files": self.files}
                    exec(
                        "import pandas as pd\nimport io\nimport traceback\n" + code,
                        {},
                        local_vars,
                    )
                self.screen_text = (
                    f.getvalue() or f"Successfully executed: {len(code)} chars."
                )
            except Exception:
                import traceback
                # Roll back any partial file mutations made before the crash
                self.files = file_snapshot
                self.screen_text = (
                    f"Python Execution Error:\n{traceback.format_exc()}"
                )
                is_error = True

        elif action.action_type == "remove_duplicates":
            filename = action.payload.get("filename", "data.csv")
            if filename in self.files:
                lines = self.files[filename].splitlines(keepends=True)
                seen = set()
                new_lines = []
                for line in lines:
                    if line not in seen:
                        seen.add(line)
                        new_lines.append(line)
                self.files[filename] = "".join(new_lines)

        elif action.action_type == "fill_nulls":
            filename = action.payload.get("filename", "data.csv")
            fill_val = action.payload.get("value", "0")
            if filename in self.files:
                self.files[filename] = self.files[filename].replace(
                    ",\n", f",{fill_val}\n"
                )

        else:
            is_unknown = True
            self.screen_text = f"Unknown action: {action.action_type}"

        # ── Grading & Reward ──
        new_score = self._current_score()
        done = self._state.step_count >= self.max_steps or new_score >= 1.0

        reward = self.reward_calculator.calculate(
            old_score=old_score,
            new_score=new_score,
            done=done,
            is_error=is_error,
            is_unknown=is_unknown,
        )

        return OsworldObservation(
            screen_text=self.screen_text,
            files=self.files,
            current_task=self.task_config.task_description,
            done=done,
            reward=reward,
            score=new_score,
        )