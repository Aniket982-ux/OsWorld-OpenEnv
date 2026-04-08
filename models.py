# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Osworld Environment.

The OsWorld environment is a simple test environment that echoes back messages.
"""

from enum import Enum
from typing import Any, Dict

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class TaskLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class OsworldAction(Action):
    """Action for the Data Cleaning environment."""

    action_type: str = Field(..., description="Type of action: inspect_schema, view_head, preview_changes, read_file, execute_python")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the action")


class OsworldObservation(Observation):
    """Observation from the Data Cleaning environment."""

    screen_text: str = Field(default="", description="Current console output or screen representation")
    files: Dict[str, str] = Field(default_factory=dict, description="Dictionary of current file states and contents")
    current_task: str = Field(default="", description="Description of the active task")
    score: float = Field(default=0.0, description="Current evaluation score from 0.0 to 1.0")
    done: bool = Field(default=False, description="Whether the episode is finished")
    reward: float = Field(default=0.0, description="Reward from the last step")
