import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from io import BytesIO
import requests
import torch
import torch.nn as nn

pipeline = QwenImageEditPlusPipeline.from_pretrained("/mnt/xlab-nas-1/junjin/pretrained_models/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
# del pipeline.vae.decoder
# del pipeline.vae.post_quant_conv
# pipeline.vae.decoder = None
# pipeline.vae.post_quant_conv = None
print("pipeline loaded")
# pipeline.enable_model_cpu_offload()

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open('a.jpg')
image2 = Image.open('b.jpg')
# prompt = "保持场景和物体的几何、布局等不变，将相机的视角向左旋转10度。"
# inputs = {
#     "image": [image1],
#     "prompt": prompt,
#     "generator": torch.manual_seed(0),
#     "true_cfg_scale": 4.0,
#     "height": 256,
#     "width": 256,
#     "negative_prompt": " ",
#     "num_inference_steps": 40,
#     "guidance_scale": 1.0,
#     "num_images_per_prompt": 1,
# }
# with torch.inference_mode():
#     output = pipeline(**inputs)
#     output_image = output.images[0]
#     output_image.save("output_image_edit_2511.png")
#     print("image saved at", os.path.abspath("output_image_edit_2511_left.png"))

# prompt = "保持场景和物体的几何、布局等不变，将相机的视角向右旋转10度。"
# inputs = {
#     "image": [image1],
#     "prompt": prompt,
#     "generator": torch.manual_seed(0),
#     "true_cfg_scale": 4.0,
#     "height": 256,
#     "width": 256,
#     "negative_prompt": " ",
#     "num_inference_steps": 40,
#     "guidance_scale": 1.0,
#     "num_images_per_prompt": 1,
# }
# with torch.inference_mode():
#     output = pipeline(**inputs)
#     output_image = output.images[0]
#     output_image.save("output_image_edit_2511.png")
#     print("image saved at", os.path.abspath("output_image_edit_2511_left.png"))



class TokenDownsampler(nn.Module):
    def __init__(
        self, 
        input_tokens: int = 4096,
        input_dim: int = 64,
        target_tokens: int = 256,      # 你可以设为 1024, 256, 64 等
        output_dim: int = 2560,
        use_residual: bool = True
    ):
        super().__init__()
        
        # 验证输入
        # assert input_tokens == 4096, "Input tokens must be 4096"
        # assert input_dim == 16, "Input dimension must be 16"
        
        # 计算空间尺寸
        input_h = input_w = int(input_tokens ** 0.5)  # 64
        target_h = target_w = int(target_tokens ** 0.5)
        
        assert input_h * input_w == input_tokens, "Input tokens must be a perfect square"
        assert target_h * target_w == target_tokens, "Target tokens must be a perfect square"
        assert input_h % target_h == 0, "Input size must be divisible by target size"
        
        self.input_h = input_h
        self.input_w = input_w
        self.target_h = target_h
        self.target_w = target_w
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_residual = use_residual
        
        # 计算下采样倍数
        downsample_factor = input_h // target_h  # e.g., 64//8 = 8
        
        # 构建下采样网络
        layers = []
        current_dim = input_dim
        
        if downsample_factor >= 8:
            # 64 -> 32 -> 16 -> 8 (3 steps for 8x downsample)
            layers.extend([
                nn.Conv2d(current_dim, 64, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(inplace=True),
            ])
            current_dim = 64
            
            layers.extend([
                nn.Conv2d(current_dim, 256, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(16, 256),
                nn.ReLU(inplace=True),
            ])
            current_dim = 256
            
            layers.extend([
                nn.Conv2d(current_dim, output_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(min(32, output_dim // 8), output_dim),
                nn.ReLU(inplace=True),
            ])
            
        elif downsample_factor >= 4:
            # 64 -> 32 -> 16 (2 steps for 4x downsample)
            layers.extend([
                nn.Conv2d(current_dim, 128, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 128),
                nn.ReLU(inplace=True),
            ])
            current_dim = 128
            
            layers.extend([
                nn.Conv2d(current_dim, output_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(min(32, output_dim // 8), output_dim),
                nn.ReLU(inplace=True),
            ])
            
        elif downsample_factor >= 2:
            # 64 -> 32 (1 step for 2x downsample)
            layers.extend([
                nn.Conv2d(current_dim, output_dim, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(min(32, output_dim // 8), output_dim),
                nn.ReLU(inplace=True),
            ])
        else:
            # No downsampling, just projection
            layers.extend([
                nn.Conv2d(current_dim, output_dim, kernel_size=1),
                nn.GroupNorm(min(32, output_dim // 8), output_dim),
                nn.ReLU(inplace=True),
            ])
        
        self.downsample_proj = nn.Sequential(*layers)
        
        # 可选：残差连接（如果输入和输出空间尺寸相同且维度匹配）
        if use_residual and downsample_factor == 1 and input_dim == output_dim:
            self.residual_proj = nn.Identity()
        elif use_residual and downsample_factor == 1:
            self.residual_proj = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        else:
            self.residual_proj = None
            
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, 4096, 16) - input tokens
            
        Returns:
            output_tokens: (B, target_tokens, 2560) - downsampled and expanded tokens
        """
        B = tokens.shape[0]
        
        # Reshape to spatial format: (B, 16, 64, 64)
        x = tokens.transpose(1, 2).reshape(B, self.input_dim, self.input_h, self.input_w)
        
        # Save for residual if needed
        if self.residual_proj is not None:
            x_res = self.residual_proj(x)
        else:
            x_res = None
        
        # Apply downsampling and expansion
        x = self.downsample_proj(x)  # (B, 2560, target_h, target_w)
        
        # Add residual connection if available
        if x_res is not None:
            # Ensure spatial dimensions match
            if x_res.shape[2:] != x.shape[2:]:
                x_res = torch.nn.functional.interpolate(
                    x_res, size=x.shape[2:], mode='nearest'
                )
            x = x + x_res
        
        # Reshape back to tokens: (B, 2560, target_tokens) -> (B, target_tokens, 2560)
        output_tokens = x.reshape(B, self.output_dim, -1).transpose(1, 2)
        
        return output_tokens
prompt = "Preserve scene layout, object positions, and spatial relationships; only slightly adjust camera viewpoint, background color, and lighting."
inputs = {
    "image": [image1,image2],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 2,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 2,
    "output_type": "latent"
}
import ipdb
ipdb.set_trace()
projector = TokenDownsampler().to('cuda',torch.bfloat16)

with torch.inference_mode():
    output = pipeline(**inputs)
    
    latents = projector(output.images)
    output_image = output.images[0]
    output_image.save("output_image_edit_2511_1.png")
    output_image = output.images[1]
    output_image.save("output_image_edit_2511_2.png")
