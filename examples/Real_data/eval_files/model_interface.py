from collections import deque
from typing import Optional, Sequence
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
from typing import Dict
import numpy as np
from pathlib import Path
from PIL import Image
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import read_mode_config

def save_numpy_as_mp4(video_array, output_path, fps=25):
    """
    将 NumPy 数组保存为 MP4 视频
    
    Args:
        video_array (np.ndarray): 形状为 (T, H, W, 3) 的 uint8 RGB 数组
        output_path (str): 输出 MP4 文件路径（如 "output.mp4"）
        fps (int): 帧率，默认 25
    """
    video_array = np.array(video_array)
    if video_array.dtype != np.uint8:
        # 自动处理 float [0,1] -> uint8 [0,255]
        if video_array.max() <= 1.0:
            video_array = (video_array * 255).astype(np.uint8)
        else:
            video_array = video_array.astype(np.uint8)

    T, H, W, _ = video_array.shape

    # 创建 VideoWriter
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
    out = cv.VideoWriter(output_path, fourcc, fps, (W, H))

    if not out.isOpened():
        print("saving video fail.")

    try:
        for t in range(T):
            # RGB -> BGR（OpenCV 使用 BGR）
            frame_bgr = cv.cvtColor(video_array[t], cv.COLOR_RGB2BGR)
            out.write(frame_bgr)
    finally:
        out.release()

def save_pil_deque_to_mp4(pil_deque, output_path, fps=25):
    """
    将 deque 中的 PIL Image 保存为 MP4 视频
    
    Args:
        pil_deque (deque): 每个元素是 PIL Image (RGB)
        output_path (str): 输出 MP4 文件路径
        fps (int): 帧率，默认 25
    """
    if not pil_deque:
        raise ValueError("Deque is empty")

    # 获取第一帧尺寸
    first_img = pil_deque[0]
    width, height = first_img.size

    # 创建 VideoWriter
    # 使用 'mp4v' 编码器（H.264 需要安装额外 codec，'mp4v' 更通用）
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for pil_img in pil_deque:
            # 确保图像是 RGB 模式
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # PIL (RGB) -> OpenCV (BGR)
            frame = np.array(pil_img)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            video_writer.write(frame_bgr)
        video_writer.release()
    except:
        print(f"Failed to open video writer for {output_path}")

        

def numpy_to_pil(img_array):
    # 确保数组是 uint8 类型（PIL 要求）
    if img_array.dtype != np.uint8:
        # 如果是 float [0,1]，先转到 [0,255]
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
    
    # 转为 PIL Image
    pil_img = Image.fromarray(img_array)
    
    return pil_img


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        output_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 1000,
        action_ensemble = True,
        action_ensemble_horizon: Optional[int] = 3, # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10095,
    ) -> None:
        
        # build client to connect server policy
        vla = baseframework.from_pretrained( # TODO should auto detect framework from model path
            policy_ckpt_path,
        )
        vla = vla.to(torch.bfloat16)
        vla = vla.to("cuda").eval()
        self.policy = vla
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = "Pick up the three stuffed animals on the table one by one and place them into the bin." if "clean_the_table" in policy_ckpt_path else "Open the drawer, place the toy inside, and then close the drawer."
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)

        self.obs_deque = deque([], maxlen=8)
        self.reset_count = 0
    
    def get_latest_obs(self):  
        return self.obs_deque[-1]

    def process_and_update_obs(self, obs):

        example = self.prepare_observation(obs)
        self.obs_deque.append(example)
        self._add_image_to_history(example['image'][0])
        print('observation updated!')

    def prepare_observation(self, obs):
        """Prepare observation for policy input."""
        # Get preprocessed images
        img = obs['main_camera_rgb']
        wrist_img = obs['wrist_camera_rgb']

        example = {
            "image": [img, wrist_img],
            'state': [obs['robot_endpose']],
            "lang": self.task_description,
        }

        return example

    def _add_image_to_history(self, image) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self) -> None:
        save_numpy_as_mp4(self.image_history, os.path.join(self.output_path, f"{self.reset_count}.mp4"))
        self.reset_count += 1
        self.obs_deque.clear()
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None


    def predict_actions(
        self, 
        obs
    ):
        """
        Perform one step of inference
        :param image: Input image in the format (H, W, 3), type uint8
        :param task_description: Task description text
        :return: (raw action, processed action)
        """
       
        self.process_and_update_obs(obs)
        example = self.get_latest_obs()
        
        task_description = self.task_description
        images = example["image"]  # list of images for history
                
        images = [self._resize_image(image) for image in images]
        example["image"] = images
        
        action_chunk_size = self.action_chunk_size

        response = self.policy.predict_action([example])
        normalized_actions = response["normalized_actions"] # B, chunk, D        
        
        normalized_actions = normalized_actions[0]    
        actions = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)

        return {"actions": actions}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 7] = np.where(normalized_actions[:, 7] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        # import ipdb; ipdb.set_trace()
        return model_config['framework']['action_model']['future_action_window_size'] + 1


    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
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
