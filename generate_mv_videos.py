# generate_lr_distributed_video.py
import os
import argparse
from PIL import Image
import torch
import torch.distributed as dist
# from diffusers import QwenImageEditPlusPipeline
from tqdm import tqdm
import gc
import numpy as np

# ===== 仅使用 imageio =====
import imageio.v3 as iio
# =========================
from starVLA.model.modules.longcat_image_edit_model import LongCatImageEditModel

# 提示词
LPROMPT = "Rotate the camera view to the left."
RPROMPT = "Rotate the camera view to the right."

def extract_frames_and_metadata(video_path):
    """
    使用 imageio 安全读取视频帧和元数据（兼容新旧版本）
    返回: (frames: List[PIL.Image], fps: float, width: int, height: int)
    失败时返回 (None, None, None, None)
    """
    try:
        frames_np = []
        reader = iio.imopen(video_path, "r", plugin="pyav")
        meta = reader.metadata()
        fps = meta.get("fps", 30.0)
        if fps <= 0:
            fps = 30.0
        
        for frame in reader.iter():
            frames_np.append(frame)
        
        reader.close()
        
        if not frames_np:
            raise ValueError("No frames read")
            
        height, width = frames_np[0].shape[0], frames_np[0].shape[1]
        frames_pil = [Image.fromarray(frame) for frame in frames_np]
        return frames_pil, fps, width, height
        
    except Exception as e:
        print(f"❌ Failed to read {os.path.basename(video_path)} with imageio: {e}")
        return None, None, None, None


def write_video_from_frames(frames, output_path, fps, size):
    """
    将 PIL 图像列表写入 H.264 视频
    frames: List[PIL.Image]
    size: (width, height)
    """
    video_array = []
    for img in frames:
        # 先 resize 再转 array（避免大图内存爆炸）
        resized_img = img.resize(size, Image.Resampling.LANCZOS)
        video_array.append(np.array(resized_img))
    
    # 写入视频（关键：yuv420p 保证兼容性）
    iio.imwrite(
        output_path,
        video_array,
        fps=fps,
        codec="libx264",
        quality=8                # 7-10 较好（10 最高质量）
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="libero_10_no_noops", help="e.g., libero_10_noops")
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()

    # 初始化分布式
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))

    torch.cuda.set_device(local_rank)
    print(f"🌍 Rank {rank}/{world_size} | GPU: {local_rank}")

    # 路径
    BASE_DIR = f"/mnt/xlab-nas-1/junjin/dataset/libero_mv_feats/{args.dataset_name}"
    ORIG_DIR = os.path.join('/mnt/nas-data-3/yangyandan/libero', f'{args.dataset_name}', 'videos/chunk-000/observation.images.image')
    L_DIR = os.path.join(BASE_DIR, "limages")
    R_DIR = os.path.join(BASE_DIR, "rimages")
    os.makedirs(L_DIR, exist_ok=True)
    os.makedirs(R_DIR, exist_ok=True)

    # 获取所有视频文件（过滤掉明显无效的小文件）
    all_videos = []
    for f in os.listdir(ORIG_DIR):
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            full_path = os.path.join(ORIG_DIR, f)
            if os.path.getsize(full_path) > 1024:  # 至少 1KB
                all_videos.append(f)
    all_videos = sorted(all_videos)

    num_videos = len(all_videos)
    videos_per_rank = num_videos // world_size

    # if rank == world_size - 1:
    #     my_videos = all_videos[rank * videos_per_rank:]
    # else:
    #     start = rank * videos_per_rank
    #     end = start + videos_per_rank
    #      my_videos = all_videos[start:end]
    my_videos = all_videos[args.start:args.end]

    print(f"Rank {rank}: processing {len(my_videos)} videos")
    # import ipdb
    # ipdb.set_trace()
    # video_path = os.path.join(ORIG_DIR, my_videos[0])
    # frames, fps, width, height = extract_frames_and_metadata(video_path)
    # 加载模型
    # model_path = "/mnt/nas-data-5/junjin/pretrained_models/Qwen-Image-Edit-2509"
    # pipeline = QwenImageEditPlusPipeline.from_pretrained(
    #     model_path,
    #     torch_dtype=torch.bfloat16,
    #     use_safetensors=True,
    # ).to("cuda")
    # pipeline.load_lora_weights("/mnt/nas-data-5/junjin/pretrained_models/Qwen-Edit-2509-Multiple-angles")

    pipeline = LongCatImageEditModel.from_pretrained("/mnt/xlab-nas-1/junjin/pretrained_models/LongCat-Image-Edit", lora_path='/mnt/workspace/junjin/code/LongCat-Image/output/edit_lora_model_10000step/checkpoints-9000',torch_dtype=torch.bfloat16)
    pipeline.to('cuda')

    generator = torch.Generator(device="cuda").manual_seed(42)

    # 处理每个视频
    for vname in tqdm(my_videos, desc=f"Rank {rank}"):
        

        # if os.path.exists(l_path) and os.path.exists(r_path):
        #     continue

        video_path = os.path.join(ORIG_DIR, vname)
        video_name =os.path.splitext(vname)[0]
        l_path = os.path.join(L_DIR, video_name)
        r_path = os.path.join(R_DIR, video_name)
        os.makedirs(l_path, exist_ok=True)
        os.makedirs(r_path, exist_ok=True)
        # 使用 imageio 一次性读取帧和元数据
        frames, fps, width, height = extract_frames_and_metadata(video_path)
        if frames is None:
            print(f"⏭️ Skipping {vname} due to read failure.")
            continue

        l_frames = []
        r_frames = []

        # 逐帧处理
        for i, img in enumerate(frames):
            # Left view
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                prompt = 'Rotate the camera view to the left'
                inputs = {
                    "images": [img],
                    "prompts": [prompt],
                    "generator": torch.Generator("cuda").manual_seed(43),
                    "num_inference_steps": 8,
                    "guidance_scale": 1.0,
                    "output_type": "latent", #latent
                    "device": 'cuda',
                    'width': 256,
                    'height': 256
                }
                output = pipeline(**inputs)
                # output[0].save(os.path.join(l_path, f"{i}.jpg"))
                # import ipdb
                # ipdb.set_trace()
                np.save(os.path.join(l_path, f"{i}.npy"), output.detach().cpu().to(torch.float32).numpy())



            # Right view
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                prompt = 'Rotate the camera view to the right'
                inputs = {
                    "images": [img],
                    "prompts": [prompt],
                    "generator": torch.Generator("cuda").manual_seed(43),
                    "num_inference_steps": 8,
                    "guidance_scale": 1.0,
                    "output_type": "latent", #latent
                    "device": 'cuda',
                    'width': 256,
                    'height': 256
                }
                output = pipeline(**inputs)
                np.save(os.path.join(r_path, f"{i}.npy"), output[0].detach().cpu().to(torch.float32).numpy())
                # output[0].save(os.path.join(r_path, f"{i}.jpg"))

            # 每 20 帧清理缓存防止 OOM
            # if (i + 1) % 20 == 0:
            #     torch.cuda.empty_cache()
            #     gc.collect()

        # 写入视频
        # try:
        #     write_video_from_frames(l_frames, l_path, fps, (width, height))
        #     write_video_from_frames(r_frames, r_path, fps, (width, height))
        # except Exception as e:
        #     print(f"❌ Failed to write video {vname}: {e}")
        #     # 清理可能损坏的文件
        #     if os.path.exists(l_path):
        #         os.remove(l_path)
        #     if os.path.exists(r_path):
        #         os.remove(r_path)

        # # 清理内存
        # del frames, l_frames, r_frames
        # torch.cuda.empty_cache()
        # gc.collect()

    print(f"✅ Rank {rank} finished.")

if __name__ == "__main__":
    main()
