import logging
import os
import time
from typing import Dict
import cv2
import numpy as np
from pathlib import Path
from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy
from molmo_spaces.policy.learned_policy.utils import PromptSampler, resize_with_pad
from starVLA.model.tools import read_mode_config
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class STARVLA_Policy(InferencePolicy):
    def __init__(
        self,
        exp_config: MlSpacesExpConfig,
        task_type: str,
    ) -> None:
        super().__init__(exp_config, exp_config.task_type)
        self.remote_config = exp_config.policy_config.remote_config
        self.prompt_sampler = PromptSampler(
            task_type=exp_config.task_type,
            prompt_templates=exp_config.policy_config.prompt_templates,
            prompt_object_word_num=exp_config.policy_config.prompt_object_word_num,
        )
        self.checkpoint_path = exp_config.policy_config.checkpoint_path
        self.grasping_type = exp_config.policy_config.grasping_type
        # self.chunk_size = exp_config.policy_config.chunk_size
        self.chunk_size = self.get_action_chunk_size(policy_ckpt_path=self.checkpoint_path)
        self.grasping_threshold = exp_config.policy_config.grasping_threshold
        self.model = None  # don't init model till inference to allow multiprocessing
        self.unnorm_key = None
        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=self.checkpoint_path)

    def get_action_chunk_size(self, policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        # import ipdb; ipdb.set_trace()
        return model_config['framework']['action_model']['future_action_window_size'] + 1


    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.prompt_sampler.next()
        self.starting_time = None

    def prepare_model(self):
        self.model_name = os.path.basename(self.checkpoint_path)
        if self.remote_config is not None:
            self._prepare_remote_model(self.checkpoint_path)
        else:
            self._prepare_local_model(self.checkpoint_path)
        # self.reset()

    def _prepare_local_model(self, checkpoint_path: str):
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
        self.config = _config.get_config(os.path.basename(checkpoint_path))
        self.model = _policy_config.create_trained_policy(self.config, checkpoint_path)

    def _prepare_remote_model(self, checkpoint_path: str):
        # try:
        #     from openpi_client import websocket_client_policy
        # except ImportError as e:
        #     log.warning(
        #         "openpi_client package is required for remote model inference. "
        #         "Install it with: pip install openpi-client"
        #     )
        #     raise e

        host = self.remote_config.get("host", "localhost")
        port = self.remote_config.get("port", 8000)
        self.checkpoint_path = checkpoint_path

        max_retries = 5
        for attempt in range(max_retries):
            try:
                self.model = WebsocketClientPolicy(
                    host=host,
                    port=port,
                )
                log.info(f"Successfully connected to remote model at {host}:{port}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    log.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    log.error(f"Failed to connect to remote model after {max_retries} attempts")
                    raise

    def render(self, obs):
        views = np.concatenate([obs["wrist_camera"], obs["exo_camera_1"]], axis=1)
        cv2.imshow("views", cv2.cvtColor(views, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

    def obs_to_model_input(self, obs):
        # self.render(obs)
        prompt = self.prompt_sampler.get_prompt(self.task).lower()

        grip = np.clip(obs["qpos"]["gripper"][0] / 0.824033, 0, 1)
        exo_camera_key = "droid_shoulder_light_randomization" if "droid_shoulder_light_randomization" in obs else "exo_camera_1"
        wrist_camera_key = "wrist_camera_zed_mini" if "wrist_camera_zed_mini" in obs else "wrist_camera"
        # model_input = {
        #     "observation/exterior_image_1_left": resize_with_pad(obs[exo_camera_key], 224, 224),
        #     "observation/wrist_image_left": resize_with_pad(obs[wrist_camera_key], 224, 224),
        #     "observation/joint_position": np.array(obs["qpos"]["arm"][:7]).reshape(
        #         7,
        #     ),
        #     "observation/gripper_position": np.array(grip).reshape(
        #         1,
        #     ),
        #     "prompt": prompt,
        # }
        example = {}
        images = [resize_with_pad(obs[exo_camera_key], 224, 224),resize_with_pad(obs[wrist_camera_key], 224, 224)]                    
        example["image"] = images
        example["lang"] = prompt
        model_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": True,
            "num_ddim_steps": 10,
        }
        return model_input
    def get_action_stats(self, unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    def _check_unnorm_key(self, norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def unnormalize_actions(self, normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        if self.grasping_type == "continuous":
            normalized_actions[-1] = normalized_actions[-1] * np.array([1.0])
        else:  # binary
            normalized_actions[-1] = (
                np.array([1.0]) if normalized_actions[-1] > self.grasping_threshold else np.array([0.0])
            )
        # normalized_actions[ -1] = np.where(normalized_actions[:, -1] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        actions[-1] *= 255.0
        return actions
        
    def inference_model(self, model_input):
        if self.model is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()
        if self.actions_buffer is None or self.current_buffer_index >= self.chunk_size:
            self.actions_buffer = self.model.predict_action(model_input)["data"]["normalized_actions"][0]
            self.current_buffer_index = 0
        model_output = self.actions_buffer[self.current_buffer_index]
        model_output = self.unnormalize_actions(normalized_actions=model_output, action_norm_stats=self.action_norm_stats)
        self.current_buffer_index += 1
        return model_output

    def model_output_to_action(self, model_output):
        # if self.grasping_type == "continuous":
        #     gripper_pos = model_output[7] * np.array([255.0])
        # else:  # binary
        #     gripper_pos = (
        #         np.array([255.0]) if model_output[7] > self.grasping_threshold else np.array([0.0])
        #     )
        gripper_pos = model_output[7]
        arm_output = model_output[:7].reshape(
            7,
        )
        action = {
            "arm": arm_output,
            "gripper": gripper_pos,
        }
        return action

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "starvla"
        info["policy_checkpoint"] = self.model_name
        info["policy_buffer_length"] = self.chunk_size
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_grasping_type"] = self.grasping_type
        info["prompt"] = self.prompt_sampler.get_prompt(self.task)
        # info["session_id"] = self.session_id
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return info
