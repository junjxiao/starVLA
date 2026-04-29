import os
import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted
import glob
import cv2
import numpy as np

def center_crop_and_resize(image: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    1. Center Crop: 根据原始图像的最短边进行正方形裁剪。
    2. Resize:将裁剪后的正方形图像缩放到目标尺寸。
    
    Args:
        image: Input image (H, W, C) uint8.
        target_size: Target size (Width, Height). e.g., (256, 256).
        
    Returns:
        Resized image (Target_H, Target_W, C) uint8.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # 1. Center Crop (Square Crop based on shortest side)
    if h < w:
        # 高度较短，以高度为基准裁剪宽度
        start_x = (w - h) // 2
        end_x = start_x + h
        cropped = image[:, start_x:end_x, :]
    else:
        # 宽度较短，以宽度为基准裁剪高度
        start_y = (h - w) // 2
        end_y = start_y + w
        cropped = image[start_y:end_y, :, :]
        
    # 2. Resize to target size
    # cv2.resize expects dsize=(width, height)
    resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # Ensure 3 channels if needed (e.g., if input was grayscale, though unlikely here)
    if len(resized.shape) == 2:
        resized = np.stack([resized] * 3, axis=-1)
        
    return resized.astype(np.uint8)

TASK_INSTRUCTIONS = {
    "place_cup": "Place the green cup on the red block.",
    "insert_cube": "Insert the pink cube into the red cup.",
    "place_cylinder": "Place the cylinder on the block.",
    "stack_block": "Stack the blue block on the red block."
}
task = "place_cup"
# === 配置 ===
INPUT_ROOT = Path(f"/mnt/xlab-nas-1/junjin/dataset/vla_dataset/real_vladata/{task}/data")
OUTPUT_ROOT = Path(f"/mnt/xlab-nas-1/junjin/dataset/real_vla_lerobot_v21/{task}")

# === 常量 ===
DEFAULT_CHUNK_SIZE = 1000
VIDEO_HEIGHT, VIDEO_WIDTH = 256, 256
FPS = 20
TASK_NAME = TASK_INSTRUCTIONS[task]  # Open the drawer, place the toy inside, and then close the drawer.    Pick up the three stuffed animals on the table one by one and place them into the bin.


def compute_video_stats(video_frames: list) -> dict:
    """计算视频通道统计（符合 LeRobot 格式：[[[mean_ch0]], [[mean_ch1]], [[mean_ch2]]]）"""
    if not video_frames:
        return {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.0]], [[0.0]], [[0.0]]],
            "std": [[[0.0]], [[0.0]], [[0.0]]],
            "count": [0]
        }
    
    # Stack frames: (T, H, W, 3) -> (3, T*H*W)
    frames = np.stack(video_frames)  # uint8
    frames_float = frames.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    stats = {}
    for i, name in enumerate(["min", "max", "mean", "std"]):
        if name == "min":
            val = frames_float.min(axis=(0, 1, 2))  # (3,)
        elif name == "max":
            val = frames_float.max(axis=(0, 1, 2))
        elif name == "mean":
            val = frames_float.mean(axis=(0, 1, 2))
        elif name == "std":
            val = frames_float.std(axis=(0, 1, 2))
        
        # 转换为 [[[val0]], [[val1]], [[val2]]]
        stats[name] = [[[float(v)]] for v in val]
    
    stats["count"] = [len(frames)]
    return stats


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # 创建目录
    data_dir = OUTPUT_ROOT / "data"
    meta_dir = OUTPUT_ROOT / "meta"
    videos_main_dir = OUTPUT_ROOT / "videos" #/ "observation.images.image"
    # videos_wrist_dir = OUTPUT_ROOT / "videos" #/ "observation.images.wrist_image"

    for d in [data_dir, meta_dir, videos_main_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 获取轨迹
    traj_dirs = [d for d in INPUT_ROOT.iterdir() if d.is_dir()]
    print(f"Found {len(traj_dirs)} trajectories.")

    all_episodes = []
    global_actions = []
    global_states = []
    all_main_videos = []
    # all_wrist_videos = []

    for ep_idx, traj_dir in enumerate(tqdm(traj_dirs, desc="Processing trajectories")):
        npz_files = natsorted(glob.glob(str(traj_dir / "*.npz")))
        if not npz_files:
            continue

        frames_main = []
        # frames_wrist = []
        states = []
        joints = []

        for npz_file in npz_files:
            data = np.load(npz_file)
            # import ipdb
            # ipdb.set_trace()
            main_rgb = data['left_camera_rgb']
            # wrist_rgb = data['wrist_camera_rgb']
            endpose = data['robot_endpose']  # (8,)
            joint = data['robot_joints_angle']
            main_resized = center_crop_and_resize(main_rgb, (VIDEO_HEIGHT, VIDEO_WIDTH))
            # wrist_resized = center_crop_and_resize(wrist_rgb, main_resized)

            frames_main.append(main_resized.astype(np.uint8))
            # frames_wrist.append(wrist_resized.astype(np.uint8))
            states.append(endpose.astype(np.float32))
            joints.append(joint.astype(np.float32))

        states = np.array(states)  # (T, 8)
        joints = np.array(joints)
        T = len(joints)

        # 构造 state: state[t] = action[t-1], state[0] = action[0]
        actions = np.zeros_like(joints)
        actions[:-1] = joints[1:]
        actions[-1] = joints[-1]

        # 保存 parquet
        chunk_id = ep_idx // DEFAULT_CHUNK_SIZE
        parquet_path = data_dir / f"chunk-{chunk_id:03d}" / f"episode_{ep_idx:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存视频
        main_video_path = videos_main_dir / f"chunk-{chunk_id:03d}" / "observation.images.image" / f"episode_{ep_idx:06d}.mp4"
        main_video_path.parent.mkdir(parents=True, exist_ok=True)
        save_video(frames_main, str(main_video_path))

        # wrist_video_path = videos_wrist_dir/ f"chunk-{chunk_id:03d}" / "observation.images.wrist_image" / f"episode_{ep_idx:06d}.mp4"
        # wrist_video_path.parent.mkdir(parents=True, exist_ok=True)
        # save_video(frames_wrist, str(wrist_video_path))

        

        df = pd.DataFrame({
            "observation.state": list(states),          # (T, 8)
            "action": list(actions),      
            "timestamp": (np.arange(T) / FPS).tolist(), # seconds
            "frame_index": np.arange(T).tolist(),
            "episode_index": [ep_idx] * T,
            "index": np.arange(T).tolist(),
            "task_index": [0] * T,
        })
        df.to_parquet(parquet_path, index=False)

        # 记录 episode
        all_episodes.append({
            "episode_index": ep_idx,
            "tasks": [TASK_NAME],  # 注意: 是列表!
            "length": T,
        })

        # 收集全局统计
        global_actions.extend(actions)
        global_states.extend(states)
        all_main_videos.extend(frames_main)
        # all_wrist_videos.extend(frames_wrist)

    # === 计算全局统计 ===
    global_actions = np.array(global_actions)  # (N, 7)
    global_states = np.array(global_states)    # (N, 8)

    action_stats = {
        "min": global_actions.min(axis=0).tolist(),
        "max": global_actions.max(axis=0).tolist(),
        "mean": global_actions.mean(axis=0).tolist(),
        "std": global_actions.std(axis=0).tolist(),
        "count": [len(global_actions)]
    }

    state_stats = {
        "min": global_states.min(axis=0).tolist(),
        "max": global_states.max(axis=0).tolist(),
        "mean": global_states.mean(axis=0).tolist(),
        "std": global_states.std(axis=0).tolist(),
        "count": [len(global_states)]
    }

    main_video_stats = compute_video_stats(all_main_videos)
    # wrist_video_stats = compute_video_stats(all_wrist_videos)

    # === 写入 meta/episodes_stats.jsonl ===
    with open(meta_dir / "episodes_stats.jsonl", 'w') as f:
        for ep in all_episodes:
            # 每个 episode 使用全局统计（简化）
            stats = {
                "episode_index": ep["episode_index"],
                "stats": {
                    "observation.images.image": main_video_stats,
                    # "observation.images.wrist_image": wrist_video_stats,
                    "observation.state": state_stats,
                    "action": action_stats,
                    "timestamp": {
                        "min": [0.0],
                        "max": [(ep["length"] - 1) / FPS],
                        "mean": [(ep["length"] - 1) / (2 * FPS)],
                        "std": [(ep["length"] - 1) / (2 * FPS * np.sqrt(3))],
                        "count": [ep["length"]]
                    },
                    "frame_index": {
                        "min": [0],
                        "max": [ep["length"] - 1],
                        "mean": [(ep["length"] - 1) / 2.0],
                        "std": [(ep["length"] - 1) / (2 * np.sqrt(3))],
                        "count": [ep["length"]]
                    },
                    "episode_index": {
                        "min": [ep["episode_index"]],
                        "max": [ep["episode_index"]],
                        "mean": [float(ep["episode_index"])],
                        "std": [0.0],
                        "count": [ep["length"]]
                    },
                    "index": {
                        "min": [0],
                        "max": [ep["length"] - 1],
                        "mean": [(ep["length"] - 1) / 2.0],
                        "std": [(ep["length"] - 1) / (2 * np.sqrt(3))],
                        "count": [ep["length"]]
                    },
                    "task_index": {
                        "min": [0],
                        "max": [0],
                        "mean": [0.0],
                        "std": [0.0],
                        "count": [ep["length"]]
                    }
                }
            }
            f.write(json.dumps(stats) + "\n")

    # === 写入 meta/episodes.jsonl ===
    with open(meta_dir / "episodes.jsonl", 'w') as f:
        for ep in all_episodes:
            f.write(json.dumps(ep) + "\n")

    # === 写入 meta/tasks.jsonl ===
    with open(meta_dir / "tasks.jsonl", 'w') as f:
        task_entry = {
            "task_index": 0,
            "task": TASK_NAME  # 你可以替换为你的实际任务描述
        }
        f.write(json.dumps(task_entry) + "\n")

    # === 写入 meta/info.json ===
    info = {
        "codebase_version": "v2.1",
        "robot_type": "franka",
        "total_episodes": len(all_episodes),
        "total_frames": sum(ep["length"] for ep in all_episodes),
        "total_tasks": 1,
        "total_videos": len(all_episodes) * 2,
        "total_chunks": (len(all_episodes) + DEFAULT_CHUNK_SIZE - 1) // DEFAULT_CHUNK_SIZE,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{len(all_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.image": {
                "dtype": "video",
                "shape": [VIDEO_HEIGHT, VIDEO_WIDTH, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.height": VIDEO_HEIGHT,
                    "video.width": VIDEO_WIDTH,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": FPS,
                    "video.channels": 3,
                    "has_audio": False
                }
            },
            # "observation.images.wrist_image": {
            #     "dtype": "video",
            #     "shape": [VIDEO_HEIGHT, VIDEO_WIDTH, 3],
            #     "names": ["height", "width", "rgb"],
            #     "info": {
            #         "video.height": VIDEO_HEIGHT,
            #         "video.width": VIDEO_WIDTH,
            #         "video.codec": "av1",
            #         "video.pix_fmt": "yuv420p",
            #         "video.is_depth_map": False,
            #         "video.fps": FPS,
            #         "video.channels": 3,
            #         "has_audio": False
            #     }
            # },
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": {
                    "motors": ["x", "y", "z", "qx", "qy", "qx", "qw", "gripper"]
                }
            },
            "action": {
                "dtype": "float32",
                "shape": [8],
                "names": {
                    "motors": ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]
                }
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None}
        }
    }
    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"✅ Conversion complete! Output saved to: {OUTPUT_ROOT}")


def save_video(frames, output_path):
    if not frames:
        return
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, FPS, (width, height))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()


if __name__ == "__main__":
    main()
