import os
import cv2
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import numpy as np
from tqdm import tqdm
import argparse
import glob

def get_image_paths(input_path):
    """收集所有图片路径"""
    image_files = []
    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            image_files.append(input_path)
        else:
            print(f"Warning: {input_path} does not look like an image file.")
    elif os.path.isdir(input_path):
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
            image_files.extend(glob.glob(os.path.join(input_path, "**", ext), recursive=True))
        image_files = sorted(list(set(image_files)))
    else:
        raise FileNotFoundError(f"Input path {input_path} does not exist.")
    return image_files

def process_images(args):
    input_path = args.input_path
    output_dir = args.output_dir
    model_cfg = args.model_config
    
    # 1. 设置设备
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU setup.")
    device = torch.device('cuda:0')
    print(f"Using device: {device}")

    # 2. 收集所有需要处理的图片路径
    print("Scanning for images...")
    image_files = get_image_paths(input_path)

    if len(image_files) == 0:
        print("No images found!")
        return

    print(f"Found {len(image_files)} images.")

    # 3. 初始化模型
    print("Loading VGGT model...")
    # 注意：如果是官方预训练模型，通常用 from_pretrained。
    # 如果是本地配置文件，请改回 VGGT.from_config(model_cfg)
    try:
        model = VGGT.from_pretrained(model_cfg).to(device)
    except Exception:
        #  fallback to from_config if from_pretrained fails or if model_cfg is a config dict/path
        model = VGGT.from_config(model_cfg).to(device)
        
    model.eval()
    dtype = torch.float16  # 使用半精度推理

    os.makedirs(output_dir, exist_ok=True)

    # 4. 分批处理配置
    # 重要：load_and_preprocess_images 会将所有图片 resize 到相同大小并堆叠。
    # 如果图片太多，一次性放入显存会 OOM。这里设置一个批次大小。
    BATCH_SIZE = 10  # 根据显存大小调整，显存大可以设大一点，比如 32 或 64
    
    total_batches = (len(image_files) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Start inference in {total_batches} batches (Batch Size: {BATCH_SIZE})...")

    with torch.no_grad():
        for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Processing Batches"):
            batch_paths = image_files[i : i + BATCH_SIZE]
            batch_names = [os.path.splitext(os.path.basename(p))[0] for p in batch_paths]
            
            # 检查断点续传：如果当前批次所有图片都已处理，则跳过
            all_exist = True
            for name in batch_names:
                if not (os.path.exists(os.path.join(output_dir, f"{name}_depth.npy")) and 
                        os.path.exists(os.path.join(output_dir, f"{name}_depth_vis.png"))):
                    all_exist = False
                    break
            if all_exist:
                continue

            try:
                # --- 核心修改：直接将路径列表送入预处理函数 ---
                # load_and_preprocess_images 内部会读取图片、Resize、Normalize 并转为 Tensor
                # 返回形状通常为: [N, 3, H, W]
                images_tensor = load_and_preprocess_images(batch_paths)
                images_tensor = images_tensor[None].to(device)

                with torch.cuda.amp.autocast(dtype=dtype):
                    # VGGT 推理步骤
                    # aggregator 期望输入 [B, 3, H, W]
                    aggregated_tokens_list, ps_idx = model.aggregator(images_tensor)
                    
                    # depth_head 期望输入 tokens 和原始图像
                    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_tensor, ps_idx)
                    # depth_map shape: [N, 1, H_out, W_out]
                
                # 获取原始图片尺寸用于上采样
                # 注意：load_and_preprocess_images 可能已经改变了图片尺寸。
                # 如果需要还原到原始文件尺寸，需要在这里重新读取一次图片获取 H, W，
                # 或者假设 load_and_preprocess_images 输出的尺寸即为标准尺寸。
                # 这里为了简化，我们假设输出深度图的分辨率就是模型输出的分辨率，
                # 或者我们可以从 images_tensor 获取当前的 H, W。
                
                # curr_h, curr_w = images_tensor.shape[2], images_tensor.shape[3]
                
                # 如果模型输出分辨率与输入不一致，进行上采样
                # 注意：VGGT 的输出分辨率取决于 patch size 和模型架构。
                # 这里假设 depth_map 的空间分辨率可能与 input 不同，需要插值
                # 如果 depth_map 已经是 [N, 1, H, W] 且 H,W 与 input 一致，则 interpolate 可省略或保持现状
                
                # 为了确保深度图与输入图像像素对齐，通常插值回输入尺寸
                # depth_map_up = torch.nn.functional.interpolate(
                #     depth_map, size=(curr_h, curr_w), mode='bilinear', align_corners=False
                # )
                
                # 转换为 CPU numpy: [N, 1, H, W] -> List of [H, W]
                depth_maps_np = depth_map[0].squeeze(1).cpu().numpy() # [N, H, W]

                # 逐个保存
                for j, img_name in enumerate(batch_names):
                    output_npy_path = os.path.join(output_dir, f"{img_name}_depth.npy")
                    output_png_path = os.path.join(output_dir, f"{img_name}_depth_vis.png")
                    
                    # 再次检查单个文件是否存在（防止批次中部分存在部分不存在）
                    if os.path.exists(output_npy_path) and os.path.exists(output_png_path):
                        continue

                    depth_val = depth_maps_np[j] # [H, W]

                    # 1. 保存原始深度值 (.npy)
                    # np.save(output_npy_path, depth_val)

                    # 2. 保存可视化深度图 (.png)
                    d_min = depth_val.min()
                    d_max = depth_val.max()
                    
                    if d_max - d_min > 1e-8:
                        depth_vis = (depth_val - d_min) / (d_max - d_min) * 255.0
                    else:
                        depth_vis = np.zeros_like(depth_val)
                    
                    depth_vis = depth_vis.astype(np.uint8)
                    depth_vis_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                    cv2.imwrite(output_png_path, depth_vis_colored)

            except Exception as e:
                print(f"Error processing batch starting with {batch_names[0]}: {e}")
                import traceback
                traceback.print_exc()

    print(f"All done! Results saved to {output_dir}")


# -----------------------------
# 入口函数
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Single GPU Depth Prediction using VGGT")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to a single image OR a directory containing images.")
    parser.add_argument("--output_dir", type=str, default="./depth_output",
                        help="Directory to save output depth maps.")
    parser.add_argument("--model_config", type=str, default="facebook/vggt-small", # 默认改为一个常见的 pretrained id 或 local path
                        help="VGGT model configuration or pretrained path.")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of images to process in one forward pass to save memory.")
    
    args = parser.parse_args()
    
    # 更新全局 BATCH_SIZE
    global BATCH_SIZE
    # 由于 Python 作用域规则，最好在函数内处理，这里简单起见直接传给 process_images 或修改逻辑
    # 为了代码简洁，我直接在 process_images 里硬编码了 BATCH_SIZE=10，
    # 如果你想用命令行参数，可以将 args.batch_size 传入 process_images
    
    process_images(args)


if __name__ == "__main__":
    main()


