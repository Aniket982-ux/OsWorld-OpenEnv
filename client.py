# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Osworld Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import OsworldAction, OsworldObservation
except ImportError:
    from models import OsworldAction, OsworldObservation


class OsworldEnv(
    EnvClient[OsworldAction, OsworldObservation, State]
):
    """
    Client for the Osworld Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with OsworldEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(OsworldAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = OsworldEnv.from_docker_image("OsWorld-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(OsworldAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: OsworldAction) -> Dict:
        """
        Convert OsworldAction to JSON payload for step message.
        """
        return {
            "action_type": action.action_type,
            "payload": action.payload,
        }

    def _parse_result(self, payload: Dict) -> StepResult[OsworldObservation]:
        """
        Parse server response into StepResult[OsworldObservation].
        """
        # Formatted server response summary (instead of raw payload dump)
        obs_preview = payload.get("observation", {})
        screen = obs_preview.get("screen_text", "")
        screen_short = (screen[:120] + "...") if len(screen) > 120 else screen
        print(
            f"  Server → score={obs_preview.get('score', 0.0):.4f} | "
            f"reward={payload.get('reward', 0.0):+.4f} | "
            f"done={payload.get('done', False)} | "
            f"screen: {screen_short}"
        )
        obs_data = payload.get("observation", {})
        observation = OsworldObservation(
            screen_text=obs_data.get("screen_text", ""),
            files=obs_data.get("files", {}),
            current_task=obs_data.get("current_task", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            score=obs_data.get("score", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
