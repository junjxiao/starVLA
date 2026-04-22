import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path

# ==============================================================================
# 1. 环境配置与导入
# ==============================================================================
# 假设你的项目根目录在上级目录，确保能导入 starVLA
_workspace_root = Path(__file__).parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from starVLA.model.framework.base_framework import baseframework

# ==============================================================================
# 2. 定义 Hook 类来捕获 Gate Map
# ==============================================================================
class GateMapExtractor:
    def __init__(self):
        self.gate_map = None
        
    def hook_fn(self, module, input, output):
        """
        捕获 view_selector 的输出。
        对于 mlp_gated_tranformer:
        self.view_selector = nn.Sequential(
            Linear(D*2, D), GELU(), Linear(D, 1), Sigmoid()
        )
        Input: Concatenated features from two views (B, L_view, D*2)
        Output: Gate weights (B, L_view, 1) -> Values between 0 and 1
        """
        # output shape: (B, L_view, 1)
        self.gate_map = output.detach().cpu() 

def visualize_gate_map(
    image_path: str, 
    instruction: str, 
    checkpoint_path: str, 
    output_dir: str = "./gate_vis",
    use_bf16: bool = True
):
    """
    加载模型，运行前向传播，提取 Gate Map 并可视化。
    专门针对 fuser_type == 'mlp_gated_tranformer'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Loading model from {checkpoint_path}...")
    
    # --- 加载模型 ---
    try:
        vla = baseframework.from_pretrained(checkpoint_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    if use_bf16:
        vla = vla.to(torch.bfloat16)
    
    vla = vla.to("cuda").eval()
    print("✅ Model loaded.")

    # --- 检查配置是否支持 Gate Extraction ---
    # 目标模块是 view_selector
    target_module_name = "view_selector"
    
    if not hasattr(vla, target_module_name):
        print(f"❌ Error: Module '{target_module_name}' not found in the model.")
        print("Please ensure the model was trained with fuser_type='mlp_gated_tranformer'.")
        return
        
    target_module = getattr(vla, target_module_name)
    
    # 简单检查是否是 Sequential 且以 Sigmoid 结尾
    if not isinstance(target_module, torch.nn.Sequential):
        print(f"⚠️ Warning: {target_module_name} is not a Sequential module. Hook might behave unexpectedly.")
    
    # --- 注册 Hook ---
    extractor = GateMapExtractor()
    handle = target_module.register_forward_hook(extractor.hook_fn)
    print(f"🪝 Hook registered on: {target_module_name}")

    # --- 准备输入数据 ---
    print(f"📷 Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224), Image.LANCZOS)

    # 构建符合模型输入的 sample 格式
    # 注意：mlp_gated_tranformer 通常期望 view_num=2
    # 如果只有一张图，forward_pass_VLM 可能会通过 image_edit_model 生成第二视角
    # 或者你需要提供 mv_feat
    sample = {
        "image": [image],  
        "lang": instruction,
    }
    
    # --- 执行前向传播 ---
    print("⚙️ Running inference...")
    with torch.no_grad():

        # 触发 forward

        output = vla.predict_action(examples=[sample], render_mv_img=True)
        mv_imgs = output['mv_imgs']
        gate_map = output['gate']
        import ipdb
        ipdb.set_trace()
            
            
        
    # --- 后处理与可视化 ---
    print("🎨 Visualizing results...")
    
    B, L_view, C = gate_map.shape
    
    # 1. 确定 Grid 尺寸
    # mlp_gated_tranformer 中，L_view 是单个视角的 token 数量
    # 假设它是正方形网格
    grid_side = int(np.sqrt(L_view))
    if grid_side * grid_side != L_view:
        print(f"⚠️ Token length {L_view} is not a perfect square. Approximating grid.")
        # 这里需要根据你的 VGGT/Projector 实际输出调整
        # 常见 ViT-L/14 @ 518px -> 37x37 = 1369
        # 常见 ViT-B/16 @ 224px -> 14x14 = 196
        # 请根据实际情况手动设置以下两行，如果自动计算错误：
        # grid_h, grid_w = 37, 37 
        grid_h = grid_side
        grid_w = L_view // grid_side
    else:
        grid_h = grid_side
        grid_w = grid_side
        
    # 2. Reshape Gate Map
    # gate_map_tensor: (1, L_view, 1) -> (H, W)
    heatmap = gate_map[0].detach().cpu().squeeze(-1).reshape(grid_h, grid_w).numpy()
    
    # Normalize heatmap to 0-1
    h_min, h_max = heatmap.min(), heatmap.max()
    if h_max - h_min > 1e-8:
        heatmap = (heatmap - h_min) / (h_max - h_min)
    else:
        heatmap = np.zeros_like(heatmap)
    lheatmap = heatmap
    rheatmap = 255 - lheatmap
    # 3. Resize to Original Image Size
    orig_w, orig_h = image.size
    lheatmap_resized = cv2.resize(lheatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    rheatmap_resized = cv2.resize(rheatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    
    # 4. Create Overlay
    l_img_np = np.array(mv_imgs[0][0].resize((224,224)))
    r_img_np = np.array(mv_imgs[1][0].resize((224,224)))
    # Apply Colormap (Jet: Blue=Low, Red=High)
    l_heatmap_colored = cv2.applyColorMap(np.uint8(255 * lheatmap_resized), cv2.COLORMAP_JET)
    l_heatmap_colored = cv2.cvtColor(l_heatmap_colored, cv2.COLOR_BGR2RGB)
    r_heatmap_colored = cv2.applyColorMap(np.uint8(255 * rheatmap_resized), cv2.COLORMAP_JET)
    r_heatmap_colored = cv2.cvtColor(r_heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend: 0.6 Original + 0.4 Heatmap
    loverlayed_img = cv2.addWeighted(l_img_np, 0.8, l_heatmap_colored, 0.2, 0)
    roverlayed_img = cv2.addWeighted(r_img_np, 0.6, r_heatmap_colored, 0.4, 0)
    
    # --- 保存结果 ---
    base_name = Path(image_path).stem
    lsave_path_overlay = os.path.join(output_dir, f"{base_name}_l_gate_overlay.png")
    lsave_path_heatmap = os.path.join(output_dir, f"{base_name}_l_gate_heatmap.png")
    lsave_path_original = os.path.join(output_dir, f"{base_name}_l_original.png")

    rsave_path_overlay = os.path.join(output_dir, f"{base_name}_r_gate_overlay.png")
    rsave_path_heatmap = os.path.join(output_dir, f"{base_name}_r_gate_heatmap.png")
    rsave_path_original = os.path.join(output_dir, f"{base_name}_r_original.png")
    cv2.imwrite(lsave_path_overlay, loverlayed_img[:,:,[2,1,0]]) 
    cv2.imwrite(lsave_path_heatmap, l_heatmap_colored[:,:,[2,1,0]]) 
    cv2.imwrite(lsave_path_original, l_img_np[:,:,[2,1,0]]) 
    cv2.imwrite(rsave_path_overlay, roverlayed_img[:,:,[2,1,0]]) 
    cv2.imwrite(rsave_path_heatmap, r_heatmap_colored[:,:,[2,1,0]]) 
    cv2.imwrite(rsave_path_original, r_img_np[:,:,[2,1,0]]) 
    
    
    
    print(f"✅ Saved visualizations to {output_dir}")
    
    # 清理 Hook
    handle.remove()

# ==============================================================================
# 3. 主执行入口
# ==============================================================================
if __name__ == "__main__":
    # === 用户配置区域 ===
    # 请替换为你的实际路径
    CHECKPOINT_PATH = "/mnt/workspace/junjin/code/starVLA/checkpoints/0416_liberoall_Qwen3vlGR00TAML_vggt_longcat_view2_cross_mlp_gated_tranformer_ck10_JAT2048_14k_bs16_4gpus/checkpoints/steps_40000_pytorch_model.pt"  
    IMAGE_PATH = "/mnt/workspace/junjin/code/starVLA/plots/my_figs/exp/gate/input.jpg"                       
    INSTRUCTION = "pick up the black bowl on the stove and place it on the plate"                
    # ====================

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint path {CHECKPOINT_PATH} does not exist.")
    elif not os.path.exists(IMAGE_PATH):
        print(f"Error: Image path {IMAGE_PATH} does not exist.")
    else:
        visualize_gate_map(
            image_path=IMAGE_PATH,
            instruction=INSTRUCTION,
            checkpoint_path=CHECKPOINT_PATH,
            output_dir="./plots/my_figs/exp/gate/"
        )
