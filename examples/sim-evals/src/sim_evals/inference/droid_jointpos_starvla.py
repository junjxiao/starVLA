import tyro
import numpy as np
from PIL import Image
from typing import Dict
from pathlib import Path
# from openpi_client import websocket_client_policy, image_tools
# from .starvla_client import ModelClient
from .abstract_client import InferenceClient
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config

def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image

class Client(InferenceClient):
    def __init__(self, 
                remote_host:str = "localhost", 
                remote_port:int = 8000,
                open_loop_horizon:int = 8,
                ckpt=None,
                unnorm_key=None,
                 ) -> None:
        # self.open_loop_horizon = open_loop_horizon
        self.open_loop_horizon = self.get_action_chunk_size(policy_ckpt_path=ckpt)
        self.client = WebsocketClientPolicy(
            remote_host, remote_port
        )
        self.use_ddim = True
        self.num_ddim_steps = 10
        self.unnorm_key = unnorm_key
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=ckpt)

    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        # import ipdb; ipdb.set_trace()
        return model_config['framework']['action_model']['future_action_window_size'] + 1


    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

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

    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, -1] = np.where(normalized_actions[:, -1] < 0.5, 0, 1) 
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        base_img = resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = resize_with_pad(curr_obs["wrist_image"], 224, 224)
        combined = np.concatenate([base_img, wrist_img], axis=1)
        return combined

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def infer(self, obs: dict, instruction: str) -> dict:
        """
        Infer the next action from the policy in a server-client setup
        """
        import ipdb
        ipdb.set_trace()
        curr_obs = self._extract_observation(obs)
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            # self.actions_from_chunk_completed = 0
            # request_data = {
            #     "observation/exterior_image_1_left": resize_with_pad(
            #         curr_obs["right_image"], 224, 224
            #     ),
            #     "observation/wrist_image_left": resize_with_pad(
            #         curr_obs["wrist_image"], 224, 224
            #     ),
            #     "observation/joint_position": curr_obs["joint_position"],
            #     "observation/gripper_position": curr_obs["gripper_position"],
            #     "prompt": instruction,
            # }
            task_description = instruction
            images = [resize_with_pad(curr_obs["right_image"], 224, 224),resize_with_pad(curr_obs["wrist_image"], 224, 224)]                    
            # images = [self._resize_image(image) for image in images]
            example["image"] = images
            vla_input = {
                "examples": [example],
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"] # B, chunk, D        
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(f"Key 'normalized_actions' not found in response data: {response['data'].keys()}")
            
            normalized_actions = normalized_actions[0]    
            self.pred_action_chunk = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)
            # self.pred_action_chunk = self.client.infer(request_data)["actions"]

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # binarize gripper action
        # if action[-1].item() > 0.5:
        #     action = np.concatenate([action[:-1], np.ones((1,))])
        # else:
        #     action = np.concatenate([action[:-1], np.zeros((1,))])

        img1 = resize_with_pad(curr_obs["right_image"], 224, 224)
        img2 = resize_with_pad(curr_obs["wrist_image"], 224, 224)
        both = np.concatenate([img1, img2], axis=1)

        return {"action": action, "viz": both}

    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        # Assign images
        right_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()

        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

if __name__ == "__main__":
    import torch
    args = tyro.cli(Args)
    client = Client(args)
    fake_obs = {
        "splat": {
            "right_cam": np.zeros((224, 224, 3), dtype=np.uint8),
            "wrist_cam": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        "policy": {
            "arm_joint_pos": torch.zeros((7,), dtype=torch.float32),
            "gripper_pos": torch.zeros((1,), dtype=torch.float32),

        },
    }
    fake_instruction = "pick up the object"

    import time

    start = time.time()
    client.infer(fake_obs, fake_instruction) # warm up
    num = 20
    for i in range(num):
        ret = client.infer(fake_obs, fake_instruction)
        print(ret["action"].shape)
    end = time.time()

    print(f"Average inference time: {(end - start) / num}")
