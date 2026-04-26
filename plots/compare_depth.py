import os
import sys
import pathlib
import numpy as np
import torch
import torchvision.io as tv_io
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import tqdm
import matplotlib.cm as cm
from typing import List, Dict

# ==========================================
# 1. 环境配置与模型加载
# ==========================================

# 添加 StarVLA 根目录到路径
STARVLA_ROOT = pathlib.Path("/mnt/workspace/junjin/code/starVLA")
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 模型Checkpoint路径
policy_ckpt_path = "/mnt/workspace/junjin/code/starVLA/checkpoints/0423_libero10_dpt_ours/checkpoints/steps_4000_pytorch_model.pt"
# import ipdb
# ipdb.set_trace()
print(f"Loading Model from {policy_ckpt_path} ...")

from starVLA.model.framework.base_framework import baseframework

# 按照要求的方法加载
vla = baseframework.from_pretrained(policy_ckpt_path)

# vla = vla.to(torch.bfloat16)
vla = vla.to("cuda").eval()
print("✅ Model Loaded Successfully.")


# ==========================================
# 2. 辅助函数
# ==========================================

def get_first_video_path(base_dir: str, sub_dir_name: str) -> str:
    """获取指定子目录下第一个视频文件的路径"""
    dir_path = os.path.join(base_dir, sub_dir_name)
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))])
    if not files:
        raise FileNotFoundError(f"No video files found in {dir_path}")
    
    return os.path.join(dir_path, files[0])

def compute_depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray = None) -> Dict[str, float]:
    """
    计算深度估计指标
    """
    if mask is None:
        mask = np.ones_like(gt_depth, dtype=bool)
    
    valid_mask = mask & (gt_depth > 1e-6) & np.isfinite(gt_depth) & np.isfinite(pred_depth)
    
    if valid_mask.sum() < 10:
        return {
            "abs_rel": np.nan, "rmse": np.nan, "log10": np.nan,
            "delta1": np.nan, "delta2": np.nan, "delta3": np.nan
        }

    p = pred_depth[valid_mask]
    g = gt_depth[valid_mask]

    # 1. AbsRel
    abs_rel = np.mean(np.abs(p - g) / g)

    # 2. RMSE
    rmse = np.sqrt(np.mean((p - g) ** 2))

    # 3. log10 RMSE
    log_p = np.log10(np.clip(p, 1e-8, None))
    log_g = np.log10(np.clip(g, 1e-8, None))
    log10_rmse = np.sqrt(np.mean((log_p - log_g) ** 2))

    # 4. Delta Thresholds
    max_ratio = np.maximum(p / g, g / p)
    
    delta1 = np.mean(max_ratio < 1.25)
    delta2 = np.mean(max_ratio < 1.25 ** 2)
    delta3 = np.mean(max_ratio < 1.25 ** 3)
    
    return {
        "abs_rel": float(abs_rel),
        "rmse": float(rmse),
        "log10": float(log10_rmse),
        "delta1": float(delta1),
        "delta2": float(delta2),
        "delta3": float(delta3)
    }

def save_colored_depth(depth_np: np.ndarray, save_path: str, size: tuple = (224, 224)):
    """
    将深度图 Resize 到指定大小，归一化，应用 Colormap，并保存为彩色 PNG
    """
    # 1. Resize to target size
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    depth_resized = TF.resize(depth_tensor, size=size, interpolation=TF.InterpolationMode.BILINEAR).squeeze().numpy()
    
    # 2. Normalize to 0-1
    d_min = depth_resized.min()
    d_max = depth_resized.max()
    
    if d_max - d_min > 1e-8:
        depth_norm = (depth_resized - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_resized)
    rgb_colored = cv2.applyColorMap(np.uint8(255 * depth_norm), cv2.COLORMAP_INFERNO)
    # rgb_colored = cv2.applyColorMap(np.uint8(255 * depth_resized), cv2.COLORMAP_JET)
    # rgb_colored = cv2.cvtColor(rgb_colored, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, rgb_colored)
    # 5. Save
    # img = Image.fromarray(rgb_colored, mode='RGB')
    # img.save(save_path)

def save_rgb_image_from_pil(pil_img: Image.Image, save_path: str, size: tuple = (224, 224)):
    """
    Resize PIL image and save
    """
    pil_img_resized = pil_img.resize(size, Image.Resampling.BICUBIC)
    pil_img_resized.save(save_path)

# ==========================================
# 3. 主测试流程
# ==========================================

def main():
    # --- 配置路径 ---
    data_root = "/mnt/xlab-nas-1/junjin/dataset/libero_no_noops_1.0.0_lerobot/libero_10_no_noops_1.0.0_lerobot/videos/chunk-000"
    output_dir = "./plots/my_figs/exp/depth/ours"
    
    # 创建子目录
    dir_input = os.path.join(output_dir, "input_rgb")
    dir_gt = os.path.join(output_dir, "gt_depth")
    dir_pred = os.path.join(output_dir, "pred_depth")
    
    os.makedirs(dir_input, exist_ok=True)
    os.makedirs(dir_gt, exist_ok=True)
    os.makedirs(dir_pred, exist_ok=True)
    
    sub_dirs = {
        "image": "observation.images.image",
        "wrist_image": "observation.images.wrist_image",
        "depth": "observation.depth.depth",
    }

    # 1. 获取视频路径
    print("Locating video files...")
    try:
        img_video_path = get_first_video_path(data_root, sub_dirs["image"])
        wrist_img_video_path = get_first_video_path(data_root, sub_dirs["wrist_image"])
        depth_video_path = get_first_video_path(data_root, sub_dirs["depth"])
        
        print(f"📹 Main Cam:   {os.path.basename(img_video_path)}")
        print(f"📹 Wrist Cam:  {os.path.basename(wrist_img_video_path)}")
        print(f"📹 Depth GT:   {os.path.basename(depth_video_path)}")
    except FileNotFoundError as e:
        print(e)
        return

    # 2. 读取视频
    print("Reading videos with torchvision...")
    try:
        # read_video 返回: video (T, H, W, C) uint8 [0, 255], audio, info
        # 注意：某些版本可能是 (T, C, H, W)，根据之前的 debug 输出，这里是 (T, H, W, C)
        img_tensor, _, _ = tv_io.read_video(img_video_path, pts_unit='sec')
        wrist_img_tensor, _, _ = tv_io.read_video(wrist_img_video_path, pts_unit='sec')
        depth_tensor_raw, _, _ = tv_io.read_video(depth_video_path, pts_unit='sec')
    except Exception as e:
        print(f"Error reading videos: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 对齐帧数
    min_frames = min(img_tensor.shape[0], wrist_img_tensor.shape[0], depth_tensor_raw.shape[0])
    if min_frames == 0:
        print("No frames loaded.")
        return
        
    img_tensor = img_tensor[:min_frames]       # [T, H, W, C]
    wrist_img_tensor = wrist_img_tensor[:min_frames] # [T, H, W, C]
    depth_tensor_raw = depth_tensor_raw[:min_frames]   # [T, H, W, C]
    
    print(f"Loaded {min_frames} frames. Shape: {img_tensor.shape}")

    # 【关键修复】处理深度图数据
    # 形状是 [T, H, W, C]，取第0个通道作为深度，并转为 float [0, 1]
    # depth_vals_np shape: [T, H, W]
    depth_vals_np = (depth_tensor_raw[:, :, :, 0].float() / 255.0).cpu().numpy()

    metrics_list = []
    target_size = (224, 224)
    # import ipdb
    # ipdb.set_trace()
    with torch.no_grad():
        for i in tqdm.tqdm(range(min_frames), desc="Evaluating Frames"):
            
            # --- A. 准备输入数据 (Resize to 224 BEFORE Model) ---
            
            # 1. 提取当前帧 Tensor [H, W, C] (uint8)
            curr_img_t = img_tensor[i]      # [256, 256, 3]
            curr_wrist_t = wrist_img_tensor[i] # [256, 256, 3]
            
            # 2. 转换为 PIL Image
            try:
                # From array expects (H, W, C) for RGB, which matches our tensor
                pil_rgb_raw = Image.fromarray(curr_img_t.cpu().numpy())
                pil_wrist_raw = Image.fromarray(curr_wrist_t.cpu().numpy())
                
                # 确保模式正确
                if pil_rgb_raw.mode != 'RGB':
                    pil_rgb_raw = pil_rgb_raw.convert('RGB')
                if pil_wrist_raw.mode != 'RGB':
                    pil_wrist_raw = pil_wrist_raw.convert('RGB')
                    
            except Exception as e:
                print(f"Frame {i} PIL creation error: {e}")
                continue

            # 3. 【核心要求】 Resize 输入图像到 224x224
            pil_rgb = pil_rgb_raw.resize(target_size, Image.Resampling.BICUBIC)
            pil_wrist = pil_wrist_raw.resize(target_size, Image.Resampling.BICUBIC)
            
            # 用于保存的 Numpy (已经是 224x224)
            rgb_np_for_save = np.array(pil_rgb)
            
            # 4. 获取 GT Depth (原始分辨率)
            gt_depth_val = depth_vals_np[i] # [H, W]
            
            # 5. 构建 Example (传入的是 224x224 的 PIL)
            example = {
                "image": [pil_rgb],  #, pil_wrist
                "lang": "pick up the object", 
            }
            
            # --- B. 模型推理 ---
            
            output = vla.predict_action(examples=[example])
            pred_depth_tensor = output["depth"][0]
            
            # 处理输出形状
            if pred_depth_tensor.dim() == 4:
                pred_depth_tensor = pred_depth_tensor.squeeze(0).squeeze(-1)
            elif pred_depth_tensor.dim() == 3:
                pred_depth_tensor = pred_depth_tensor.squeeze(-1)
            
            pred_depth_np = pred_depth_tensor.cpu().numpy()
            
            # --- C. 计算指标 ---
            # 为了公平比较，将 Pred (224x224) Resize 回 GT 的原始分辨率
            pred_depth_for_metric = TF.resize(
                pred_depth_tensor.unsqueeze(0).unsqueeze(0), # [1, 1, 224, 224]
                size=gt_depth_val.shape, 
                interpolation=TF.InterpolationMode.BILINEAR
            ).squeeze(0).squeeze(0).cpu().numpy()
            
            mask = gt_depth_val > 1e-6 
            metrics = compute_depth_metrics(pred_depth_for_metric, gt_depth_val, mask)
            metrics_list.append(metrics)
            
            # --- D. 保存图像 (全部已经是或调整为 224x224) ---
            # import ipdb
            # ipdb.set_trace()
            # 1. Save Input RGB (Already 224x224)
            save_rgb_image_from_pil(pil_rgb, os.path.join(dir_input, f"frame_{i:04d}.png"), size=target_size)
            
            # 2. Save GT Depth (Colored, Resized to 224x224)
            save_colored_depth(gt_depth_val, os.path.join(dir_gt, f"frame_{i:04d}.png"), size=target_size)
            
            # 3. Save Pred Depth (Colored, Resized to 224x224)
            # pred_depth_np is already 224x224 from model output usually, but ensure via save func
            save_colored_depth(pred_depth_np, os.path.join(dir_pred, f"frame_{i:04d}.png"), size=target_size)

            

    # 5. 汇总并保存最终结果
    if metrics_list:
        valid_metrics = [m for m in metrics_list if not np.isnan(m['rmse'])]
        
        if valid_metrics:
            avg_abs_rel = np.mean([m['abs_rel'] for m in valid_metrics])
            avg_rmse = np.mean([m['rmse'] for m in valid_metrics])
            avg_log10 = np.mean([m['log10'] for m in valid_metrics])
            avg_delta1 = np.mean([m['delta1'] for m in valid_metrics])
            avg_delta2 = np.mean([m['delta2'] for m in valid_metrics])
            avg_delta3 = np.mean([m['delta3'] for m in valid_metrics])
            
            summary_text = (
                f"Depth Estimation Evaluation Summary\n"
                f"-----------------------------------\n"
                f"Total Frames Evaluated: {len(valid_metrics)}\n"
                f"\n"
                f"Avg AbsRel:       {avg_abs_rel:.4f}\n"
                f"Avg RMSE:         {avg_rmse:.4f}\n"
                f"Avg log10 RMSE:   {avg_log10:.4f}\n"
                f"\n"
                f"Avg Delta < 1.25:  {avg_delta1:.4f}\n"
                f"Avg Delta < 1.25^2: {avg_delta2:.4f}\n"
                f"Avg Delta < 1.25^3: {avg_delta3:.4f}\n"
            )
            
            print("\n" + summary_text)
            
            summary_path = os.path.join(output_dir, "metrics_summary.txt")
            with open(summary_path, "w") as f:
                f.write(summary_text)
            print(f"Summary saved to {summary_path}")
        else:
            print("No valid metrics to summarize.")
    else:
        print("Evaluation failed or no frames processed.")

if __name__ == "__main__":
    main()
