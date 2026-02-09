# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cosmos WM environment with GPU offload support (state serialization)."""

import io

import numpy as np
import torch

from rlinf.envs.env_manager import EnvOffloadMixin
from rlinf.envs.maniskill.utils import recursive_to_device
from rlinf.envs.world_model.world_model_cosmos_env import CosmosEnv

__all__ = ["CosmosOffloadEnv"]


class CosmosOffloadEnv(CosmosEnv, EnvOffloadMixin):
    """CosmosEnv variant that can serialize / deserialize its state to bytes.

    This is used by the hybrid-engine scheduler so that the Cosmos model's
    GPU memory can be reclaimed while the policy is running, and later
    restored when the environment is needed again.
    """

    def get_state(self) -> bytes:
        """Serialize environment state to bytes buffer."""
        env_state = {
            "current_obs": recursive_to_device(self.current_obs, "cpu")
            if self.current_obs is not None
            else None,
            "task_descriptions": self.task_descriptions,
            "init_ee_poses": self.init_ee_poses,
            "elapsed_steps": self.elapsed_steps,
            "prev_step_reward": self.prev_step_reward.cpu(),
            "_is_start": self._is_start,
            "video_cnt": self.video_cnt,
            "render_images": self.render_images,
            "render_rgb": getattr(self, "render_rgb", None),
            "render_actions": getattr(self, "render_actions", None),
            "render_rewards": getattr(self, "render_rewards", None),
            "reset_state_ids": self.reset_state_ids.cpu(),
            "generator_state": self._generator.get_state(),
        }

        # Save frame_queue (list of deques containing uint8 numpy arrays)
        frame_queue_state = []
        for env_idx in range(self.num_envs):
            queue_frames = []
            for frame in self.frame_queue[env_idx]:
                # frame is an np.ndarray (H, W, 3) uint8
                queue_frames.append(frame.copy())
            frame_queue_state.append(queue_frames)
        env_state["frame_queue"] = frame_queue_state

        # Save metrics if recording
        if self.record_metrics:
            env_state.update(
                {
                    "success_once": self.success_once.cpu(),
                    "returns": self.returns.cpu(),
                }
            )

        # Serialize to bytes
        buffer = io.BytesIO()
        torch.save(env_state, buffer)
        return buffer.getvalue()

    def load_state(self, state_buffer: bytes):
        """Load environment state from bytes buffer."""
        buffer = io.BytesIO(state_buffer)
        state = torch.load(buffer, map_location="cpu", weights_only=False)

        # Restore basic state
        self.current_obs = (
            recursive_to_device(state["current_obs"], self.device)
            if state["current_obs"] is not None
            else None
        )
        self.task_descriptions = state["task_descriptions"]
        self.init_ee_poses = state["init_ee_poses"]
        self.elapsed_steps = state["elapsed_steps"]
        self.prev_step_reward = state["prev_step_reward"].to(self.device)
        self._is_start = state["_is_start"]
        self.video_cnt = state["video_cnt"]
        self.render_images = state["render_images"]
        self.render_rgb = state.get("render_rgb", None)
        self.render_actions = state.get("render_actions", None)
        self.render_rewards = state.get("render_rewards", None)

        # Restore reset state management
        self.reset_state_ids = state["reset_state_ids"].to(self.device)
        self._generator.set_state(state["generator_state"])

        # Restore frame_queue
        frame_queue_state = state["frame_queue"]
        for env_idx in range(self.num_envs):
            self.frame_queue[env_idx].clear()
            for frame in frame_queue_state[env_idx]:
                self.frame_queue[env_idx].append(frame.copy())

        # Restore metrics if recording
        if self.record_metrics and "success_once" in state:
            self.success_once = state["success_once"].to(self.device)
            self.returns = state["returns"].to(self.device)
