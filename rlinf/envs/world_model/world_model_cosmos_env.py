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

"""
Cosmos Predict 2.5 action-conditioned world model environment.

This wraps NVIDIA's Cosmos Predict 2.5 Video2WorldActionConditionedPipeline
as an RLinf environment, replacing the OpenSora world model.
"""

import math
import os
from collections import deque
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.data.datasets.world_model import NpyTrajectoryDatasetWrapper
from rlinf.envs.world_model.base_world_env import BaseWorldEnv
from rlinf.models.reward_model import ResnetRM
from omegaconf import OmegaConf

__all__ = ["CosmosEnv"]


class CosmosEnv(BaseWorldEnv):
    """World model environment backed by Cosmos Predict 2.5.

    Key differences from OpenSoraEnv:
    - Uses ``Video2WorldActionConditionedPipeline`` from cosmos_predict2 for
      video generation (no manual VAE encode/decode or scheduler calls).
    - Keeps a sliding window of *pixel-space* frames instead of latent
      frames, because the Cosmos pipeline handles latent encoding
      internally.
    - Uses a standalone ``ResnetRM`` reward model for reward prediction.
    """

    def __init__(
        self, cfg, num_envs, seed_offset, total_num_processes, record_metrics=True
    ):
        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, record_metrics
        )
        self.world_model_cfg = self.cfg.world_model_cfg
        self.inference_dtype = torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Reset state management
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size

        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

        # ---------- Cosmos-specific hyperparams ----------
        cosmos_cfg = self.world_model_cfg.cosmos
        self.chunk = self.world_model_cfg.chunk  # Action chunk size (8)
        self.condition_frame_length = self.world_model_cfg.condition_frame_length
        self.image_size = tuple(self.world_model_cfg.image_size)  # (256, 320) for Cosmos

        # Cosmos inference params
        self.cosmos_ckpt_path = cosmos_cfg.ckpt_path
        self.cosmos_tokenizer_path = cosmos_cfg.get("tokenizer_path", None)
        self.cosmos_guidance = cosmos_cfg.get("guidance", 7)
        self.cosmos_num_steps = cosmos_cfg.get("num_steps", 35)
        self.cosmos_action_scaler = cosmos_cfg.get("action_scaler", 20.0)

        # Number of predicted frames per chunk.
        # Cosmos natively predicts ``chunk_size`` frames (default 12).
        # The actual VLA chunk is ``self.chunk`` (8). We generate
        # ``chunk_size`` frames and keep the first ``self.chunk``.
        self.cosmos_chunk_size = cosmos_cfg.get("chunk_size", 12)

        # Micro-batch size for batched Cosmos inference.
        # All num_envs are processed in micro-batches of this size to
        # avoid GPU OOM while maximising throughput.
        self.cosmos_batch_size = cosmos_cfg.get("batch_size", 16)

        # ---------- Load Cosmos model ----------
        self.cosmos_model = self._load_cosmos_model()

        # ---------- Load ResnetRM reward model ----------
        self.reward_model = self._load_reward_model().eval().to(self.device)

        # ---------- State tracking ----------
        # Keep decoded pixel frames as tensors in [-1, 1].
        # current_obs: [num_envs, 3, 1, T, H, W]
        self.current_obs = None
        self.task_descriptions = [""] * self.num_envs
        self.init_ee_poses = [None] * self.num_envs

        # Sliding window of *pixel* condition frames per env
        # Each entry is an np.ndarray (H, W, 3) uint8
        self.frame_queue = [
            deque(maxlen=self.condition_frame_length) for _ in range(self.num_envs)
        ]

        self._init_metrics()

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------

    def _build_dataset(self, cfg):
        return NpyTrajectoryDatasetWrapper(cfg.initial_image_path)

    def _load_cosmos_model(self):
        """Load the Cosmos Video2WorldActionConditionedPipeline."""
        from cosmos_predict2.configs.action_conditioned.config import (
            get_cosmos_predict2_action_conditioned_pipeline,
        )
        from cosmos_predict2.pipelines.video2world_action import (
            Video2WorldActionConditionedPipeline,
        )

        config = get_cosmos_predict2_action_conditioned_pipeline(
            model_size="2B", resolution="480", fps=4,
        )
        # Disable guardrail & prompt refiner for RL inference
        config.guardrail_config.enabled = False
        config.prompt_refiner_config.enabled = False
        # Disable online resize — we feed a single conditioning frame, not a
        # multi-frame video, so temporal_sample would fail.
        config.resize_online = False

        # Override tokenizer path when explicitly provided so that
        # the tokenizer is loaded from the self-contained model
        # directory rather than the default HF cache location.
        if self.cosmos_tokenizer_path is not None:
            config.tokenizer.vae_pth = self.cosmos_tokenizer_path

        pipe = Video2WorldActionConditionedPipeline.from_config(
            config=config,
            dit_path=self.cosmos_ckpt_path,
            use_text_encoder=False,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_prompt_refiner=False,
        )
        return pipe

    def _load_reward_model(self):
        """Load the standalone ResnetRM reward model."""
        rm_cfg = OmegaConf.to_container(
            self.world_model_cfg.reward_model, resolve=True
        )
        rew_model = ResnetRM(from_pretrained=rm_cfg["from_pretrained"])
        return rew_model

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0
            self._elapsed_steps = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        if isinstance(terminations, torch.Tensor):
            self.success_once = self.success_once | terminations
        else:
            terminations_tensor = torch.tensor(
                terminations, device=self.device, dtype=torch.bool
            )
            self.success_once = self.success_once | terminations_tensor
        episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = torch.full(
            (self.num_envs,),
            self.elapsed_steps,
            dtype=torch.float32,
            device=self.device,
        )
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _calc_step_reward(self, chunk_rewards):
        """Calculate (optionally relative) step reward."""
        reward_diffs = torch.zeros(
            (self.num_envs, self.chunk), dtype=torch.float32, device=self.device
        )
        for i in range(self.chunk):
            reward_diffs[:, i] = (
                self.cfg.reward_coef * chunk_rewards[:, i] - self.prev_step_reward
            )
            self.prev_step_reward = self.cfg.reward_coef * chunk_rewards[:, i]

        if self.use_rel_reward:
            return reward_diffs
        else:
            return chunk_rewards

    def _estimate_success_from_rewards(self, chunk_rewards):
        success_threshold = getattr(self.cfg, "success_reward_threshold", 0.9)
        max_reward_in_chunk = chunk_rewards.max(dim=1)[0]
        return (max_reward_in_chunk >= success_threshold).to(self.device)

    # ------------------------------------------------------------------
    # Reset state management
    # ------------------------------------------------------------------

    def update_reset_state_ids(self):
        total_num_episodes = len(self.dataset)
        reset_state_ids = torch.randint(
            low=0,
            high=total_num_episodes,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        ).to(self.device)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
        episode_indices: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        self.elapsed_steps = 0

        if self.is_start:
            if self.use_fixed_reset_state_ids:
                episode_indices = self.reset_state_ids
            self._is_start = False

        num_envs = self.num_envs
        if len(self.dataset) < num_envs:
            raise ValueError(
                f"Not enough episodes. Found {len(self.dataset)}, need {num_envs}"
            )

        if episode_indices is None:
            if seed is not None:
                np.random.seed(seed[0] if isinstance(seed, list) else seed)
            episode_indices = np.random.choice(
                len(self.dataset), size=num_envs, replace=False
            )
        else:
            if isinstance(episode_indices, torch.Tensor):
                episode_indices = episode_indices.cpu().numpy()

        # Load first frames
        img_tensors = []
        task_descriptions = []
        init_ee_poses = []

        for env_idx, episode_idx in enumerate(episode_indices):
            episode_data = self.dataset[episode_idx]
            if len(episode_data["start_items"]) == 0:
                raise ValueError(f"Empty start_items for episode {episode_idx}")

            first_frame = episode_data["start_items"][0]
            task_desc = episode_data.get("task", "")
            task_descriptions.append(str(task_desc))

            img_tensor = first_frame["image"]  # [3, H, W], float [0, 1]

            if "observation.state" in first_frame:
                init_ee_poses.append(first_frame["observation.state"].numpy())
            else:
                init_ee_poses.append(None)

            # Resize if needed
            if img_tensor.shape[1:] != self.image_size:
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            # Normalize to [-1, 1]
            img_tensor = img_tensor * 2.0 - 1.0

            # Repeat for condition frames: [3, cond_len, H, W]
            img_tensor = img_tensor.unsqueeze(1).repeat(
                1, self.condition_frame_length, 1, 1
            )
            img_tensors.append(img_tensor)

            # Fill pixel frame queue with uint8 numpy frames
            frame_uint8 = first_frame["image"]  # [3, H, W] float [0,1]
            frame_np = (frame_uint8.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # Resize if needed
            if frame_np.shape[:2] != self.image_size:
                from PIL import Image

                frame_pil = Image.fromarray(frame_np).resize(
                    (self.image_size[1], self.image_size[0]), Image.BILINEAR
                )
                frame_np = np.array(frame_pil)
            self.frame_queue[env_idx].clear()
            for _ in range(self.condition_frame_length):
                self.frame_queue[env_idx].append(frame_np.copy())

        # Stack: [num_envs, 3, cond_len, H, W]
        stacked_imgs = torch.stack(img_tensors, dim=0).to(self.device)
        self.current_obs = stacked_imgs.unsqueeze(2).to(self.device)
        # Shape: [num_envs, 3, 1, cond_len, H, W]

        self.task_descriptions = task_descriptions
        self.init_ee_poses = init_ee_poses

        self._is_start = False
        self._reset_metrics()

        extracted_obs = self._wrap_obs()
        return extracted_obs, {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, actions=None, auto_reset=True):
        raise NotImplementedError(
            "step in CosmosEnv is not implemented, use chunk_step instead"
        )

    # ------------------------------------------------------------------
    # Inference: generate next chunk of frames via Cosmos
    # ------------------------------------------------------------------

    def _scale_actions(self, actions_np: np.ndarray) -> np.ndarray:
        """Scale actions by cosmos_action_scaler (matching Cosmos's convention)."""
        return actions_np * self.cosmos_action_scaler

    def _infer_next_chunk_frames(self, actions):
        """Generate next chunk of frames using Cosmos Predict 2.5.

        Uses batched inference through ``generate_batch`` to process multiple
        environments in a single forward pass through the DiT, with
        configurable micro-batch size to stay within GPU memory.
        """
        num_envs = self.num_envs
        actions_np = (
            actions if isinstance(actions, np.ndarray) else actions.cpu().numpy()
        )

        # actions_np: [num_envs, chunk, action_dim]
        actions_np = actions_np.reshape(num_envs, -1, actions_np.shape[-1])

        # ---- Prepare all frames & actions upfront ----
        all_frames = []
        all_actions = []
        for env_idx in range(num_envs):
            current_frame = self.frame_queue[env_idx][-1]  # (H, W, 3) uint8
            env_actions = actions_np[env_idx]

            # Pad/truncate to cosmos_chunk_size
            if env_actions.shape[0] < self.cosmos_chunk_size:
                pad = np.zeros(
                    (self.cosmos_chunk_size - env_actions.shape[0], env_actions.shape[1]),
                    dtype=env_actions.dtype,
                )
                env_actions = np.concatenate([env_actions, pad], axis=0)
            else:
                env_actions = env_actions[: self.cosmos_chunk_size]

            all_frames.append(current_frame)
            all_actions.append(self._scale_actions(env_actions))

        # ---- Batched inference with micro-batching ----
        batch_size = self.cosmos_batch_size
        all_videos = []
        for start in range(0, num_envs, batch_size):
            end = min(start + batch_size, num_envs)
            videos = self.cosmos_model.generate_batch(
                first_frames=all_frames[start:end],
                actions_list=all_actions[start:end],
                num_conditional_frames=1,
                guidance=self.cosmos_guidance,
                num_sampling_step=self.cosmos_num_steps,
                seed=self.seed + self.elapsed_steps + start,
            )
            all_videos.append(videos)

        # (num_envs, 3, T, H, W) in [-1, 1]
        all_videos = torch.cat(all_videos, dim=0)

        assert all_videos.shape[2] > self.chunk, (
            f"Cosmos returned only {all_videos.shape[2]} frames, need >{self.chunk}."
        )

        # ---- Extract generated frames & update queues ----
        # Skip conditioning frame (index 0), take next `chunk` frames
        generated = all_videos[:, :, 1 : 1 + self.chunk]  # (num_envs, 3, chunk, H, W)

        # Update frame queues with last generated frame per env
        last_frames = all_videos[:, :, min(self.chunk, all_videos.shape[2] - 1)]  # (num_envs, 3, H, W)
        last_frames_np = (
            (last_frames.permute(0, 2, 3, 1).cpu().float() / 2 + 0.5)
            .clamp(0, 1)
            .mul(255)
            .to(torch.uint8)
            .numpy()
        )
        for env_idx in range(num_envs):
            self.frame_queue[env_idx].append(last_frames_np[env_idx])

        # Add view dim: [num_envs, 3, 1, chunk, H, W]
        x_samples = generated.unsqueeze(2).to(self.device).float()

        # Append to current_obs
        self.current_obs = torch.cat([self.current_obs, x_samples], dim=3)

        # Keep sliding window
        max_frames = self.condition_frame_length + self.chunk * 2
        if self.current_obs.shape[3] > max_frames:
            self.current_obs = self.current_obs[:, :, :, -max_frames:, :, :]

    # ------------------------------------------------------------------
    # Reward inference
    # ------------------------------------------------------------------

    def _infer_next_chunk_rewards(self):
        """Predict rewards using the ResnetRM model on decoded frames."""
        if self.reward_model is None:
            raise ValueError("Reward model is not loaded")

        num_envs, c, v, t, h, w = self.current_obs.shape
        extract_chunk_obs = self.current_obs.permute(0, 3, 1, 2, 4, 5)
        # [num_envs, T, 3, v, h, w]

        if self.cfg.world_model_cfg.reward_model.type == "ResnetRM":
            extract_chunk_obs = extract_chunk_obs[:, -self.chunk :, :, :, :, :]
            extract_chunk_obs = extract_chunk_obs.reshape(
                self.num_envs * self.chunk, c, v, h, w
            )
            extract_chunk_obs = extract_chunk_obs.squeeze(2)  # [N*chunk, 3, h, w]

            # ResnetRM expects images in [-1, 1] range on GPU
            extract_chunk_obs = extract_chunk_obs.to(self.device)
            rewards = self.reward_model.predict_rew(extract_chunk_obs)
            rewards = rewards.reshape(self.num_envs, self.chunk)
        else:
            raise ValueError(
                f"Unknown reward model type: {self.cfg.world_model_cfg.reward_model.type}"
            )

        return rewards

    # ------------------------------------------------------------------
    # Observation wrapping
    # ------------------------------------------------------------------

    def _wrap_obs(self):
        """Format observations to match libero_env interface."""
        num_envs = self.num_envs
        b, c, v, t, h, w = self.current_obs.shape

        last_frame = self.current_obs[:, :, 0, -1, :, :]  # [num_envs, 3, H, W]
        full_image = last_frame.permute(0, 2, 3, 1)  # [num_envs, H, W, 3]
        full_image = (full_image + 1.0) / 2.0 * 255.0
        full_image = torch.clamp(full_image, 0, 255).to(torch.uint8)

        states = torch.zeros((num_envs, 16), device=self.device, dtype=torch.float32)

        obs = {
            "main_images": full_image,
            "wrist_images": None,
            "states": states,
            "task_descriptions": self.task_descriptions,
        }
        return obs

    # ------------------------------------------------------------------
    # Auto-reset
    # ------------------------------------------------------------------

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs
        final_info = infos

        extracted_obs, infos = self.reset()

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones

        return extracted_obs, infos

    # ------------------------------------------------------------------
    # chunk_step  – main rollout entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def chunk_step(self, policy_output_action):
        """Execute a chunk of actions through the Cosmos world model."""
        with torch.amp.autocast(device_type="cuda", dtype=self.inference_dtype):
            self._infer_next_chunk_frames(policy_output_action)

        self.elapsed_steps += self.chunk

        extracted_obs = self._wrap_obs()

        chunk_rewards = self._infer_next_chunk_rewards()
        chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)

        estimated_success = self._estimate_success_from_rewards(chunk_rewards)

        raw_chunk_terminations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        raw_chunk_terminations[:, -1] = estimated_success

        raw_chunk_truncations = torch.zeros(
            self.num_envs, self.chunk, dtype=torch.bool, device=self.device
        )
        truncations = torch.tensor(
            self.elapsed_steps >= self.cfg.max_episode_steps
        ).to(self.device)

        if truncations.any():
            raw_chunk_truncations[:, -1] = truncations

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, {}
            )
        else:
            infos = {}

        infos = self._record_metrics(
            chunk_rewards_tensors.sum(dim=1), past_terminations, infos
        )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        self.add_new_frames()

        return (
            extracted_obs,
            chunk_rewards_tensors,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    # ------------------------------------------------------------------
    # Video rendering
    # ------------------------------------------------------------------

    def add_new_frames(self):
        """Append latest chunk frames to the render buffer."""
        if self.current_obs is None:
            return

        num_envs, channels, num_views, num_steps, height, width = (
            self.current_obs.shape
        )
        view_idx = 0
        chunk_len = min(self.chunk, num_steps)
        start_step = num_steps - chunk_len

        for step_idx in range(chunk_len):
            images = []
            for env_idx in range(num_envs):
                frame_tensor = self.current_obs[
                    env_idx, :, view_idx, start_step + step_idx, :, :
                ]
                frame_np = frame_tensor.detach().cpu().permute(1, 2, 0).numpy()
                frame_np = (frame_np + 1.0) / 2.0 * 255.0
                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                images.append(frame_np)

            tiled = self._tile_images(images)
            if tiled is not None:
                self.render_images.append(tiled)

    def _tile_images(self, images, nrows: Optional[int] = None):
        if not images:
            return None

        num_images = len(images)
        height, width, channels = images[0].shape
        rows = nrows or max(1, int(math.sqrt(num_images)))
        cols = int(math.ceil(num_images / rows))

        canvas = np.zeros(
            (rows * height, cols * width, channels), dtype=images[0].dtype
        )
        for idx, image in enumerate(images):
            row = idx // cols
            col = idx % cols
            y0, y1 = row * height, (row + 1) * height
            x0, x1 = col * width, (col + 1) * width
            canvas[y0:y1, x0:x1] = image

        return canvas

    def flush_video(self, video_sub_dir: Optional[str] = None):
        if len(self.render_images) == 0:
            return

        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")

        os.makedirs(output_dir, exist_ok=True)

        from mani_skill.utils.visualization.misc import images_to_video

        images_to_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
            fps=self.cfg.get("fps", 10),
            verbose=False,
        )

        self.video_cnt += 1
        self.render_images = []
