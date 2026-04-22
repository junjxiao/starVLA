import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import json
import argparse
from tqdm import tqdm
import torchvision.transforms.functional as TF

# ==========================================
# 1. 路径配置与依赖导入
# ==========================================

# 添加 StarVLA 根目录到路径，以便导入模型
STARVLA_ROOT = Path("/mnt/workspace/junjin/code/starVLA")
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

# 导入必要的模块
from omegaconf import OmegaConf
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from vggt.models.vggt import VGGT
from starVLA.model.framework.qwen_gr00t_spatial_aml import Qwen_GR00TSpatialAML, preprocess_images
from starVLA.model.modules.longcat_image_edit_model import LongCatImageEditModel

# 假设 DPT 结构，如果 starVLA 内部有 DPT 实现请替换，这里使用一个简化的 Transformer Decoder 模拟 DPT 行为
class SimpleDPTHead(nn.Module):
    """
    简化版的 DPT Head，用于将 Token 特征映射回深度图。
    实际实验中建议替换为真实的 DPT 实现 (e.g., from transformers import DPTForDepthEstimation)
    """
    def __init__(self, input_dim=2560, output_dim=1, img_size=256, patch_size=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # 简单的 MLP 映射 + Upsample
        self.proj = nn.Linear(input_dim, patch_size * patch_size * output_dim)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(output_dim, output_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(output_dim, output_dim, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, tokens):
        # tokens: [B, N, C]
        B, N, C = tokens.shape
        # 假设 tokens 已经是对应 spatial grid 的特征
        # 如果 N != num_patches，可能需要插值或调整，这里假设 VGGT 输出 token 数与 grid 对应
        # 注意：VGGT 输出的 token 数可能不等于 (H/P)*(W/P)，需要根据实际 VGGT 输出调整
        # 这里做一个通用的 reshape 假设，实际需根据 VGGT 输出的 ps_idx 调整
        
        # 为了演示，我们假设输入 tokens 已经是 spatial map 的 flatten 形式
        # 如果 VGGT 输出是 [B, L, C]，我们需要知道 L 对应的 H, W
        # 此处简化处理：直接线性投影后 reshape
        x = self.proj(tokens) # [B, N, P*P*D]
        
        # 这里的 reshape 逻辑高度依赖 VGGT 的具体输出结构
        # 假设 VGGT 输出的是 16x16 或 32x32 的 grid
        h_w = int(np.sqrt(N))
        if h_w * h_w != N:
             # 如果不是正方形，尝试寻找最接近的因子或报错
             # 临时方案：强制 interpolate 到固定大小
             x = x.transpose(1, 2).view(B, -1, h_w, h_w) 
             x = torch.nn.functional.interpolate(x, size=(self.img_size//4, self.img_size//4), mode='bilinear')
        else:
            x = x.transpose(1, 2).view(B, -1, h_w, h_w)

        depth_low_res = x[:, :1, :, :] # 取第一个通道作为深度
        depth_high_res = self.upsample(depth_low_res)
        
        # Crop or Pad to exact img_size if needed
        if depth_high_res.shape[-1] != self.img_size:
            depth_high_res = torch.nn.functional.interpolate(depth_high_res, size=(self.img_size, self.img_size), mode='bilinear')
            
        return depth_high_res.squeeze(1) # [B, H, W]


# ==========================================
# 2. 数据集定义
# ==========================================

class LiberoDepthDataset(Dataset):
    def __init__(self, dataset_path, split="train", max_samples=None):
        """
        加载 LeRobot 数据集，提取 RGB 和 Depth
        """
        print(f"Loading dataset from {dataset_path}...")
        self.dataset = LeRobotDataset.from_pretrained(dataset_path)
        
        # 获取 split 索引
        if split == "train":
            start, end = map(int, self.dataset.meta.info["splits"]["train"].split(":"))
            self.indices = list(range(start, end))
        else:
            self.indices = list(range(len(self.dataset)))
            
        if max_samples:
            self.indices = self.indices[:max_samples]
            
        print(f"Loaded {len(self.indices)} samples for {split}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.dataset[real_idx]
        
        # 1. 获取 RGB 图像 (observation.images.image)
        # LeRobot 返回的是 Tensor [C, H, W] 或 Video 片段，这里取第一帧或当前帧
        # 注意：LeRobot item 结构可能因版本而异，通常 image 是 Tensor
        rgb_tensor = item["observation.images.image"] # [C, H, W]
        if rgb_tensor.dim() == 4: # 如果是视频片段 [T, C, H, W]，取中间帧
            rgb_tensor = rgb_tensor[rgb_tensor.shape[0] // 2]
        
        # 转换为 PIL Image 供模型预处理使用 (因为 preprocess_images 期望 PIL)
        rgb_np = (rgb_tensor.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        rgb_pil = Image.fromarray(rgb_np)
        
        # 2. 获取 GT Depth (observation.depth.depth)
        # 注意：info.json 中显示 depth 也是 video 类型且 is_depth_map=false? 
        # 这通常意味着它存储的是编码后的图像或者需要特殊解码。
        # 假设它存储的是归一化的深度值或者可以直接使用的深度图。
        # 如果 is_depth_map=false 且 dtype=video，它可能只是看起来像深度的 RGB 图。
        # **关键假设**：这里假设 observation.depth.depth 存储的是单通道深度值或者需要转换。
        # 如果它是 [3, H, W] 且是伪彩色，你需要先转灰度或解码。
        # 这里假设它是 [1, H, W] 或 [3, H, W] 的真实深度值 (米)
        depth_tensor = item["observation.depth.depth"]
        if depth_tensor.dim() == 4:
            depth_tensor = depth_tensor[depth_tensor.shape[0] // 2]
        
        # 如果 depth 是 3 通道，取均值或第一个通道作为深度
        if depth_tensor.shape[0] == 3:
            depth_val = depth_tensor.mean(dim=0, keepdim=True)
        else:
            depth_val = depth_tensor
            
        # 归一化深度到 0-1 或保持原值，取决于 Loss 函数
        # 这里简单归一化到 0-1 用于 MSE Loss 稳定性
        d_min, d_max = depth_val.min(), depth_val.max()
        if d_max - d_min > 1e-6:
            depth_norm = (depth_val - d_min) / (d_max - d_min)
        else:
            depth_norm = torch.zeros_like(depth_val)
            
        return {
            "rgb_pil": rgb_pil,
            "gt_depth": depth_norm.squeeze(0), # [H, W]
            "original_depth_raw": depth_val.squeeze(0) # 用于评估
        }

# ==========================================
# 3. 主实验类
# ==========================================

class DepthComparisonExperiment:
    def __init__(self, config_path, checkpoint_path, dataset_path, device="cuda"):
        self.device = torch.device(device)
        self.dataset_path = dataset_path
        
        # 1. 加载配置
        print("Loading Config...")
        self.cfg = OmegaConf.load(config_path)
        
        # 修改配置以匹配实验需求 (如果需要)
        # 例如，确保 image_edit_model 启用
        if not hasattr(self.cfg.framework, 'image_edit_model'):
             raise ValueError("Config must have image_edit_model section")
        
        # 2. 初始化主模型 (StarVLA)
        print("Initializing StarVLA Model...")
        self.model = Qwen_GR00TSpatialAML(config=self.cfg)
        
        # 3. 加载预训练权重
        print(f"Loading Checkpoint from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # 处理可能的 key 前缀 (e.g., "model." or "module.")
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        
        self.model.to(self.device)
        self.model.eval() # 设置为评估模式，冻结 BN/Dropout
        
        # 4. 冻结主模型参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 5. 初始化两个 DPT Heads
        # 假设 VGGT 输出维度为 2560 (根据代码中的 spatial_projector output_dim)
        # 注意：需要确认 VGGT 输出 token 的数量和空间分辨率
        self.dpt_baseline = SimpleDPTHead(input_dim=2560, img_size=256).to(self.device)
        self.dpt_fused = SimpleDPTHead(input_dim=2560, img_size=256).to(self.device)
        
        # 6. 优化器 (只优化 DPT Heads)
        self.optimizer = optim.AdamW(
            list(self.dpt_baseline.parameters()) + list(self.dpt_fused.parameters()),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        # 7. 损失函数
        self.criterion = nn.MSELoss()

    def extract_features(self, batch_rgb_pils):
        """
        运行模型前向传播，提取 Baseline 和 Fused 特征
        """
        # 准备输入格式
        # Qwen_GR00TSpatialAML 期望 images 是 List[List[PIL.Image]]
        # batch_rgb_pils is List[PIL.Image]
        batch_images = [[img] for img in batch_rgb_pils]
        instructions = [""] * len(batch_rgb_pils) # Dummy instruction
        
        # 1. 预处理图像
        # 使用模型内部的 preprocess_images
        # 注意：preprocess_images 返回 Tensor [B, C, H, W]
        target_size = 256 # 根据数据集分辨率
        processed_imgs = preprocess_images(batch_images, target_size=target_size).to(self.device)
        
        with torch.no_grad():
            # 2. 获取 VGGT 空间特征 (Baseline)
            # 参考 forward_pass_VLM 中的逻辑
            aggregated_tokens_list, ps_idx = self.model.spatial_model.aggregator(processed_imgs)
            # 获取最后一层的 spatial tokens
            # aggregated_tokens_list[-1] shape: [B, Layers?, N, C]? 
            # 根据代码: spatial_tokens = aggregated_tokens_list[-1][:,0,ps_idx:,:]
            # 假设 aggregated_tokens_list 是 list of tensors
            raw_spatial_tokens = aggregated_tokens_list[-1][:, 0, ps_idx:, :] # [B, N_spatial, C_vggt]
            
            # 3. 获取 Image Edit 特征 (Multi-view Latents)
            # 参考 forward_pass_image_edit_model
            primary_images = [img[0] for img in batch_images]
            # 调用 image edit model 获取 latents
            # 注意：forward_pass_image_edit_model 内部有 autocast 和 no_grad
            extra_latents = self.model.forward_pass_image_edit_model(primary_images, render_mv_img=False)
            # extra_latents shape: [B, N_mv * N_tokens, C_latent] (C_latent=64 usually)
            
            # 4. 投影特征
            spatial_tokens_proj = self.model.spatial_projector(raw_spatial_tokens) # [B, N_spatial, D_proj]
            extra_latents_proj = self.model.image_edit_projector(extra_latents) # [B, N_mv*N_tok, D_proj]
            
            # 5. 融合特征 (Fused)
            # 参考 forward_pass_VLM 中的 fusion logic
            # 这里复现 'cross_attention' 或 'mmdit' 等融合逻辑
            # 假设使用 cross_attention fuser
            fused_spatial_tokens = self.model.spatial_fuser(spatial_tokens_proj, extra_latents_proj)
            # 注意：不同的 fuser_type 返回形状不同。
            # 如果是 CrossAttention(Query=Spatial, Key/Value=MV), 输出形状同 Spatial [B, N_spatial, D_proj]
            
        return raw_spatial_tokens, fused_spatial_tokens

    def train_epoch(self, dataloader, epoch):
        self.dpt_baseline.train()
        self.dpt_fused.train()
        
        total_loss_base = 0
        total_loss_fused = 0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            rgb_pils = batch["rgb_pil"]
            gt_depths = batch["gt_depth"].to(self.device) # [B, H, W]
            
            # 1. 提取特征
            try:
                base_feats, fused_feats = self.extract_features(rgb_pils)
            except Exception as e:
                print(f"Error extracting features: {e}")
                continue
                
            # 2. 预测深度
            pred_depth_base = self.dpt_baseline(base_feats)
            pred_depth_fused = self.dpt_fused(fused_feats)
            
            # 3. 计算 Loss
            # 确保预测值和 GT 形状一致
            if pred_depth_base.shape[-2:] != gt_depths.shape[-2:]:
                pred_depth_base = torch.nn.functional.interpolate(pred_depth_base.unsqueeze(1), size=gt_depths.shape[-2:], mode='bilinear').squeeze(1)
                pred_depth_fused = torch.nn.functional.interpolate(pred_depth_fused.unsqueeze(1), size=gt_depths.shape[-2:], mode='bilinear').squeeze(1)

            loss_base = self.criterion(pred_depth_base, gt_depths)
            loss_fused = self.criterion(pred_depth_fused, gt_depths)
            
            # 总 Loss (可以同时优化，也可以加权)
            total_loss = loss_base + loss_fused
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            total_loss_base += loss_base.item()
            total_loss_fused += loss_fused.item()
            n_batches += 1
            
            pbar.set_postfix({
                "L_Base": f"{loss_base.item():.4f}",
                "L_Fused": f"{loss_fused.item():.4f}"
            })
            
        avg_loss_base = total_loss_base / n_batches if n_batches > 0 else 0
        avg_loss_fused = total_loss_fused / n_batches if n_batches > 0 else 0
        
        print(f"Epoch {epoch} Avg Loss - Baseline: {avg_loss_base:.4f}, Fused: {avg_loss_fused:.4f}")
        return avg_loss_base, avg_loss_fused

    def run(self, num_epochs=10, batch_size=4, num_workers=4):
        # 创建数据集
        dataset = LiberoDepthDataset(self.dataset_path, split="train")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=lambda x: {
            "rgb_pil": [d["rgb_pil"] for d in x],
            "gt_depth": torch.stack([d["gt_depth"] for d in x]),
            "original_depth_raw": torch.stack([d["original_depth_raw"] for d in x])
        })
        
        results = []
        for epoch in range(num_epochs):
            l_base, l_fused = self.train_epoch(dataloader, epoch)
            results.append({"epoch": epoch, "loss_base": l_base, "loss_fused": l_fused})
            
            # 保存中间检查点
            if (epoch + 1) % 5 == 0:
                torch.save({
                    "dpt_baseline": self.dpt_baseline.state_dict(),
                    "dpt_fused": self.dpt_fused.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                }, f"./checkpoints/depth_exp_epoch_{epoch+1}.pth")
                
        # 最终比较
        print("\n--- Final Comparison ---")
        print(f"Average Baseline Loss: {np.mean([r['loss_base'] for r in results]):.4f}")
        print(f"Average Fused Loss:    {np.mean([r['loss_fused'] for r in results]):.4f}")
        
        if np.mean([r['loss_fused'] for r in results]) < np.mean([r['loss_base'] for r in results]):
            print("✅ Conclusion: Multi-view Fusion improves depth prediction.")
        else:
            print("❌ Conclusion: Multi-view Fusion did NOT improve depth prediction in this experiment.")

# ==========================================
# 4. 启动入口
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint .pt file")
    parser.add_argument("--dataset", type=str, default="/mnt/xlab-nas-1/junjin/dataset/libero_no_noops_1.0.0_lerobot/libero_10_no_noops_1.0.0_lerobot/libero_10_no_noops_lerobot", help="Path to LeRobot dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2) # Small batch size due to large model
    args = parser.parse_args()
    
    # 创建检查点目录
    os.makedirs("./checkpoints", exist_ok=True)
    
    exp = DepthComparisonExperiment(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset
    )
    
    exp.run(num_epochs=args.epochs, batch_size=args.batch_size)
