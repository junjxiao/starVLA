# # Copyright 2025 starVLA community. All rights reserved.
# # Licensed under the MIT License, Version 1.0 (the "License");
# # Implemented by [Junqiu YU / Fudan University] in [2025]. 
# # Design and Merged by [Jinhui YE / HKUST University] in [2025].
# """
# Qwen-GR00T Framework
# A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
# Flow-matching header is copyright from GR00T N1.5,
# """
# import sys
# from pathlib import Path

# # Add workspace root to Python path if not already there
# _workspace_root = Path(__file__).parent.parent.parent.parent
# if str(_workspace_root) not in sys.path:
#     sys.path.insert(0, str(_workspace_root))
# import os
# CHECKPOINT_BASEDIR = os.getenv('CHECKPOINT_BASEDIR', None)
# from typing import List
# from tqdm import tqdm
# from typing import List, Optional, Tuple
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from PIL import Image
from torchvision import transforms as TF
# import cv2


# from starVLA.training.trainer_utils import initialize_overwatch
# from deployment.model_server.tools.image_tools import to_pil_preserve

# logger = initialize_overwatch(__name__)

# # HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
# IGNORE_INDEX = -100

# from starVLA.model.framework.base_framework import baseframework
# from starVLA.model.modules.vlm import get_vlm_model
# from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
# from starVLA.training.trainer_utils.trainer_tools import resize_images
# from starVLA.model.tools import FRAMEWORK_REGISTRY

# from vggt.models.vggt import VGGT
# from diffusers import QwenImageEditPlusPipeline
# from starVLA.model.modules.projector.QFormer import get_layerwise_qformer
# from vggt.heads.dpt_head import DPTHead

# class CrossAttention(nn.Module):
#     def __init__(
#         self,
#         d_model: int,
#         d_hidden: int,
#         nhead: int = 8,
#         dropout: float = 0.0,
#         kv_dim: int = 2048
#     ):
#         super().__init__()
#         self.d_model = d_model
#         self.d_hidden = d_hidden if d_hidden is not None else d_model
#         self.nhead = nhead
#         self.head_dim = self.d_hidden // nhead
#         assert self.d_hidden % nhead == 0, "d_hidden must be divisible by nhead"

#         # Projections
#         self.q_proj = nn.Linear(d_model, self.d_hidden)
#         self.k_proj = nn.Linear(kv_dim, self.d_hidden)
#         self.v_proj = nn.Linear(kv_dim, self.d_hidden)
#         self.out_proj = nn.Linear(self.d_hidden, d_model)

#         self.dropout_attn = nn.Dropout(dropout)
#         self.dropout_out = nn.Dropout(dropout)
#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, image_feature: torch.Tensor, spatial_feature: torch.Tensor):
#         """
#         Args:
#             image_feature: (B, N_img, d_model) — Query
#             vggt_feature:   (B, N_vggt, kv_dim) — Key and Value

#         Returns:
#             fused_image_feature: (B, N_img, d_model)
#         """
#         B, N_img, _ = image_feature.shape
#         _, N_spatial, _ = spatial_feature.shape

#         # Project to d_hidden
#         q = self.q_proj(image_feature)   # (B, N_img, d_hidden)
#         k = self.k_proj(spatial_feature)     # (B, N_vggt, d_hidden)
#         v = self.v_proj(spatial_feature)     # (B, N_vggt, d_hidden)

#         # Reshape for multi-head: (B, N, d_hidden) -> (B, N, nhead, head_dim) -> (B, nhead, N, head_dim)
#         q = q.view(B, N_img, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_img, head_dim)
#         k = k.view(B, N_spatial, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_vggt, head_dim)
#         v = v.view(B, N_spatial, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_vggt, head_dim)

#         # Scaled Dot-Product Attention
#         scale = self.head_dim ** -0.5
#         attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, nhead, N_img, N_vggt)
#         attn_weights = F.softmax(attn_weights, dim=-1)
#         attn_weights = self.dropout_attn(attn_weights)

#         # Weighted sum over values
#         attn_output = torch.matmul(attn_weights, v)  # (B, nhead, N_img, head_dim)

#         # Concatenate heads and project back
#         attn_output = attn_output.transpose(1, 2).contiguous()  # (B, N_img, nhead, head_dim)
#         attn_output = attn_output.view(B, N_img, self.d_hidden)  # (B, N_img, d_hidden)

#         # Final projection to d_model
#         output = self.out_proj(attn_output)  # (B, N_img, d_model)
#         output = self.dropout_out(output)

#         # Residual connection + LayerNorm
#         output = self.norm(image_feature + output)

#         return output

# def preprocess_images(image_list, target_size, mode='crop'): #  [B，[PLT]]
#     batch_images = []
#     shapes = set()
#     to_tensor = TF.ToTensor()
#     # target_size = 518

#     # First process all images and collect their shapes
#     for imgs in image_list:
#         epi_images = []
#         for img in imgs:
#             width, height = img.size

#             if mode == "pad":
#                 # Make the largest dimension 518px while maintaining aspect ratio
#                 if width >= height:
#                     new_width = target_size
#                     new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
#                 else:
#                     new_height = target_size
#                     new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
#             else:  # mode == "crop"
#                 # Original behavior: set width to 518px
#                 new_width = target_size
#                 # Calculate height maintaining aspect ratio, divisible by 14
#                 new_height = round(height * (new_width / width) / 14) * 14

#             # Resize with new dimensions (width, height)
#             # img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
#             img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
#             img = to_tensor(img)  # Convert to tensor (0, 1)

#             # Center crop height if it's larger than 518 (only in crop mode)
#             if mode == "crop" and new_height > target_size:
#                 start_y = (new_height - target_size) // 2
#                 img = img[:, start_y : start_y + target_size, :]

#             # For pad mode, pad to make a square of target_size x target_size
#             if mode == "pad":
#                 h_padding = target_size - img.shape[1]
#                 w_padding = target_size - img.shape[2]

#                 if h_padding > 0 or w_padding > 0:
#                     pad_top = h_padding // 2
#                     pad_bottom = h_padding - pad_top
#                     pad_left = w_padding // 2
#                     pad_right = w_padding - pad_left

#                     # Pad with white (value=1.0)
#                     img = torch.nn.functional.pad(
#                         img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
#                     )

#             shapes.add((img.shape[1], img.shape[2]))
#             epi_images.append(img)
#         batch_images.append(torch.stack(epi_images))

#     # Check if we have different shapes
#     # In theory our model can also work well with different shapes
#     if len(shapes) > 1:
#         print(f"Warning: Found images with different shapes: {shapes}")
#         # Find maximum dimensions
#         max_height = max(shape[0] for shape in shapes)
#         max_width = max(shape[1] for shape in shapes)

#         # Pad images if necessary
#         padded_images = []
#         for img in batch_images:
#             h_padding = max_height - img.shape[1]
#             w_padding = max_width - img.shape[2]

#             if h_padding > 0 or w_padding > 0:
#                 pad_top = h_padding // 2
#                 pad_bottom = h_padding - pad_top
#                 pad_left = w_padding // 2
#                 pad_right = w_padding - pad_left

#                 img = torch.nn.functional.pad(
#                     img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
#                 )
#             padded_images.append(img)
#         batch_images = padded_images

#     batch_images = torch.stack(batch_images)  # concatenate images

#     # Ensure correct shape when single image
#     if len(image_list) == 1:
#         # Verify shape is (1, C, H, W)
#         if batch_images.dim() == 3:
#             batch_images = batch_images.unsqueeze(0)
#     return batch_images

# def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
#     """
#     Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
    
#     Args:
#         input_tensor (torch.Tensor): The loss tensor to check and fix.
#         loss_name (str): Name of the loss (for diagnostic prints).
#         hard_max (float, optional): Maximum absolute value allowed. Values outside 
#                                   [-hard_max, hard_max] will be clamped. If None, 
#                                   no clamping is performed. Defaults to 100.
#     """
#     if input_tensor is None:
#         return input_tensor
    
#     # Check for inf/nan values
#     has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
#     if has_inf_nan:
#         logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
#         input_tensor = torch.where(
#             torch.isnan(input_tensor) | torch.isinf(input_tensor),
#             torch.zeros_like(input_tensor),
#             input_tensor
#         )

#     # Apply hard clamping if specified
#     if hard_max is not None:
#         input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

#     return input_tensor

# def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = '', valid_range=0.98, **kwargs):
#     """
#     Compute depth loss.
    
#     Args:
#         predictions: Dict containing 'depth' and 'depth_conf'
#         batch: Dict containing ground truth 'depths' and 'point_masks'
#         gamma: Weight for confidence loss
#         alpha: Weight for confidence regularization
#         gradient_loss_fn: Type of gradient loss to apply
#         valid_range: Quantile range for outlier filtering
#     """
#     pred_depth = predictions['depth']
#     pred_depth_conf = predictions['depth_conf']

#     gt_depth = batch['depths']
#     gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
#     gt_depth = gt_depth[..., None]              # (B, H, W, 1)
#     gt_depth_mask = batch['point_masks'].clone()   # 3D points derived from depth map, so we use the same mask

#     if gt_depth_mask.sum() < 100:
#         # If there are less than 100 valid points, skip this batch
#         dummy_loss = (0.0 * pred_depth).mean()
#         loss_dict = {f"loss_conf_depth": dummy_loss,
#                     f"loss_reg_depth": dummy_loss,
#                     f"loss_grad_depth": dummy_loss,}
#         return loss_dict

#     # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
#     # this is hacky, but very easier to implement
#     loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
#                                              gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

#     loss_dict = {
#         f"loss_conf_depth": loss_conf,
#         f"loss_reg_depth": loss_reg,    
#         f"loss_grad_depth": loss_grad,
#     }

#     return loss_dict

# def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
#     """
#     Filter loss tensor by keeping only values below a certain quantile threshold.
    
#     This helps remove outliers that could destabilize training.
    
#     Args:
#         loss_tensor: Tensor containing loss values
#         valid_range: Float between 0 and 1 indicating the quantile threshold
#         min_elements: Minimum number of elements required to apply filtering
#         hard_max: Maximum allowed value for any individual loss
    
#     Returns:
#         Filtered and clamped loss tensor
#     """
#     if loss_tensor.numel() <= min_elements:
#         # Too few elements, just return as-is
#         return loss_tensor

#     # Randomly sample if tensor is too large to avoid memory issues
#     if loss_tensor.numel() > 100000000:
#         # Flatten and randomly select 1M elements
#         indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
#         loss_tensor = loss_tensor.view(-1)[indices]

#     # First clamp individual values to prevent extreme outliers
#     loss_tensor = loss_tensor.clamp(max=hard_max)

#     # Compute quantile threshold
#     quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
#     quantile_thresh = min(quantile_thresh, hard_max)

#     # Apply quantile filtering if enough elements remain
#     quantile_mask = loss_tensor < quantile_thresh
#     if quantile_mask.sum() > min_elements:
#         return loss_tensor[quantile_mask]
#     return loss_tensor


# def torch_quantile(
#     input,
#     q,
#     dim = None,
#     keepdim: bool = False,
#     *,
#     interpolation: str = "nearest",
#     out: torch.Tensor = None,
# ) -> torch.Tensor:
#     """Better torch.quantile for one SCALAR quantile.

#     Using torch.kthvalue. Better than torch.quantile because:
#         - No 2**24 input size limit (pytorch/issues/67592),
#         - Much faster, at least on big input sizes.

#     Arguments:
#         input (torch.Tensor): See torch.quantile.
#         q (float): See torch.quantile. Supports only scalar input
#             currently.
#         dim (int | None): See torch.quantile.
#         keepdim (bool): See torch.quantile. Supports only False
#             currently.
#         interpolation: {"nearest", "lower", "higher"}
#             See torch.quantile.
#         out (torch.Tensor | None): See torch.quantile. Supports only
#             None currently.
#     """
#     # https://github.com/pytorch/pytorch/issues/64947
#     # Sanitization: q
#     try:
#         q = float(q)
#         assert 0 <= q <= 1
#     except Exception:
#         raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

#     # Handle dim=None case
#     if dim_was_none := dim is None:
#         dim = 0
#         input = input.reshape((-1,) + (1,) * (input.ndim - 1))

#     # Set interpolation method
#     if interpolation == "nearest":
#         inter = round
#     elif interpolation == "lower":
#         inter = floor
#     elif interpolation == "higher":
#         inter = ceil
#     else:
#         raise ValueError(
#             "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
#             f"(got '{interpolation}')!"
#         )

#     # Validate out parameter
#     if out is not None:
#         raise ValueError(f"Only None value is currently supported for out (got {out})!")

#     # Compute k-th value
#     k = inter(q * (input.shape[dim] - 1)) + 1
#     out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

#     # Handle keepdim and dim=None cases
#     if keepdim:
#         return out
#     if dim_was_none:
#         return out.squeeze()
#     else:
#         return out.squeeze(dim)

#     return out

# def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn='', gamma=1.0, alpha=0.2, valid_range=-1):
#     """
#     Core regression loss function with confidence weighting and optional gradient loss.
    
#     Computes:
#     1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
#     2. Optional gradient loss
    
#     Args:
#         pred: (B, S, H, W, C) predicted values
#         gt: (B, S, H, W, C) ground truth values
#         mask: (B, S, H, W) valid pixel mask
#         conf: (B, S, H, W) confidence weights (optional)
#         gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
#         gamma: Weight for confidence loss
#         alpha: Weight for confidence regularization
#         valid_range: Quantile range for outlier filtering
    
#     Returns:
#         loss_conf: Confidence-weighted loss
#         loss_grad: Gradient loss (0 if not specified)
#         loss_reg: Regular L2 loss
#     """
#     bb, ss, hh, ww, nc = pred.shape

#     # Compute L2 distance between predicted and ground truth points
#     loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
#     loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

#     # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
#     # This encourages the model to be confident on easy examples and less confident on hard ones
#     loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
#     loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
#     # Initialize gradient loss
#     loss_grad = 0

#     # Prepare confidence for gradient loss if needed
#     if "conf" in gradient_loss_fn:
#         to_feed_conf = conf.reshape(bb*ss, hh, ww)
#     else:
#         to_feed_conf = None

#     # Compute gradient loss if specified for spatial smoothness
#     if "normal" in gradient_loss_fn:
#         # Surface normal-based gradient loss
#         loss_grad = gradient_loss_multi_scale_wrapper(
#             pred.reshape(bb*ss, hh, ww, nc),
#             gt.reshape(bb*ss, hh, ww, nc),
#             mask.reshape(bb*ss, hh, ww),
#             gradient_loss_fn=normal_loss,
#             scales=3,
#             conf=to_feed_conf,
#         )
#     elif "grad" in gradient_loss_fn:
#         # Standard gradient-based loss
#         loss_grad = gradient_loss_multi_scale_wrapper(
#             pred.reshape(bb*ss, hh, ww, nc),
#             gt.reshape(bb*ss, hh, ww, nc),
#             mask.reshape(bb*ss, hh, ww),
#             gradient_loss_fn=gradient_loss,
#             conf=to_feed_conf,
#         )

#     # Process confidence-weighted loss
#     if loss_conf.numel() > 0:
#         # Filter out outliers using quantile-based thresholding
#         if valid_range>0:
#             loss_conf = filter_by_quantile(loss_conf, valid_range)

#         loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
#         loss_conf = loss_conf.mean()
#     else:
#         loss_conf = (0.0 * pred).mean()

#     # Process regular regression loss
#     if loss_reg.numel() > 0:
#         # Filter out outliers using quantile-based thresholding
#         if valid_range>0:
#             loss_reg = filter_by_quantile(loss_reg, valid_range)

#         loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
#         loss_reg = loss_reg.mean()
#     else:
#         loss_reg = (0.0 * pred).mean()

#     return loss_conf, loss_grad, loss_reg


# @FRAMEWORK_REGISTRY.register("QwenGR00TDPT")
# class Qwen_GR00TDPT(baseframework):
#     """
#     Multimodal vision-language-action model.

#     Components:
#       - Qwen2.5 VL interface for fused language/vision token embeddings
#       - Layer-wise QFormer for multi-layer feature aggregation
#       - DINO encoder for dense multi-view spatial tokens
#       - DiT diffusion head for future action sequence modeling

#     Focus: Predict future continuous actions conditioned on images + instruction.
#     """

#     def __init__(
#         self,
#         config: Optional[dict] = None,
#         **kwargs,
#     ) -> None:
#         """
#         Construct all submodules and cache key configuration values.

#         Args:
#             config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
#             **kwargs: Reserved for future overrides (unused).
#         """
#         super().__init__()
#         self.config = config
#         self.qwen_vl_interface = get_vlm_model(config=self.config)
#         # align dims --> we should put them to config or no?
#         self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

#         self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

#         self.future_action_window_size = config.framework.action_model.future_action_window_size
#         self.past_action_window_size = config.framework.action_model.past_action_window_size
#         self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
#         if config.framework.spatial_model is not None:
#             self.spatial_model = self.get_spatial_model(config)
#             self.spatial_projector = self.get_spatial_projector(config)
        
#             self.config.framework.fuser = {'type':'cross_attention'}
#             print(self.config.framework.fuser.type)
#             if self.config.framework.fuser.type == 'cross_attention':
#                 self.fuser = self.get_cross_attention(config)
#             else:
#                 raise NotImplementedError
#         self.depth_head = DPTHead(dim_in=self.qwen_vl_interface.model.config.hidden_size//4, output_dim=2, activation="exp", conf_activation="expp1",intermediate_layer_idx=[0,1,2,3])

#     def get_cross_attention(self, config):
#         model = CrossAttention(d_model=config.framework.spatial_projector.output_dim,d_hidden=config.framework.spatial_projector.output_dim,kv_dim=config.framework.spatial_projector.output_dim)
#         return model
        
#     def get_spatial_model(self, config):
#         spatial_model_cfg = config.framework.spatial_model
#         if "vggt" in spatial_model_cfg.model_name_or_path:
#             self.spatial_type = "vggt"
#             spatial_model = VGGT.from_pretrained(spatial_model_cfg.model_name_or_path)
#         else:
#             raise NotImplementedError
#         return spatial_model

#     def get_spatial_projector(self, config):
#         spatial_projector_cfg = config.framework.spatial_projector
#         projector = nn.Linear(config.framework.spatial_model.output_dim, spatial_projector_cfg.output_dim)
#         return projector

#     def forward_pass_qwen_image_edit_model(self, images, prompt=None):
#         if prompt is None:
#             prompt = "Preserve scene layout, object positions, and spatial relationships; only slightly adjust camera viewpoint, background color, and lighting."
#         inputs = {
#             "image": images,
#             "prompt": prompt,
#             "generator": torch.manual_seed(0),
#             "true_cfg_scale": 4.0,
#             "negative_prompt": " ",
#             "num_inference_steps": 2,
#             "guidance_scale": 1.0,
#             "num_images_per_prompt": len(images),
#             "output_type": "latent"
#         }
#         with torch.inference_mode():
#             output = self.qwen_image_edit_model(**inputs)
#             output_image = output.images
#         qwen_feat = self.qwen_projector(output_image.clone())
#         return qwen_feat
        
            

#     def forward(
#         self,
#         examples: List[dict] = None,
#         **kwargs,
#     ) -> Tuple:
#         """

#         """
        
#         batch_images = [example["image"][:2] for example in examples]  #  [B，[PLT]]
#         batch_depth = [example["image"][2] for example in examples]
#         instructions = [example["lang"] for example in examples]  # [B, str]
#         actions = [example["action"] for example in examples]  # label [B， len, 7]
        
#         state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
#         has_spatial_model = self.config.framework.spatial_model is not None
#         with torch.no_grad():
#             with torch.cuda.amp.autocast(dtype=torch.bfloat16):  
#                 # Step 1: QWenVL input format
#                 qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

#                 qwenvl_outputs = self.qwen_vl_interface(
#                     **qwen_inputs,
#                     output_attentions=False,
#                     output_hidden_states=True,
#                     return_dict=True,
#                 )    
#                 spatial_input = preprocess_images(batch_images, batch_images[0][0].size[0]).to(qwen_inputs['pixel_values'].device)   
#                 # step 2: encode spatial feature
#                 if has_spatial_model:
#                     if self.spatial_type == "vggt":    
                        
#                         aggregated_tokens_list, ps_idx = self.spatial_model.aggregator(spatial_input)
                        
#                     elif self.spatial_type == "depthanything3":
#                         denorm_image = (denorm_image - self._resnet_mean.to(denorm_image.device)) / self._resnet_std.to(denorm_image.device)
#                         feats, aux_feats = self.spatial_model.model.da3.backbone(denorm_image.unsqueeze(1).to(torch.bfloat16),cam_token=None, export_feat_layers=[-1], ref_view_strategy="saddle_balanced")
#                         Bs, S, N, C = feats[0][0].shape
#                         spatial_tokens = feats[-1][0].reshape(Bs*S, N, C)

#                     # step 3: fuse spatial tokens and qwen tokens
#                     num_layers = 4
#                     qwenvl_interval = len(qwenvl_outputs.hidden_states) // num_layers  
#                     qwenvl_index = [i * qwenvl_interval for i in range(1, num_layers)] + [-1]
#                     qwenvl_hidden_states = torch.stack([qwenvl_outputs.hidden_states[i] for i in qwenvl_index])
#                     spatial_interval = len(aggregated_tokens_list) // num_layers
#                     spatial_index = [spatial_interval * i for i in range(1, num_layers)] + [-1]
#                     spatial_hidden_states = torch.stack([aggregated_tokens_list[i][:,0,ps_idx:,:] for i in spatial_index])
#                     spatial_hidden_states = spatial_hidden_states.to(self.spatial_projector.weight.dtype)
#                     spatial_hidden_states = self.spatial_projector(spatial_hidden_states)
                    
#                     hiddens = []
#                     for i in range(num_layers):
#                         last_hidden = self.fuser(qwenvl_hidden_states[i], spatial_hidden_states[i])
#                         hiddens.append(last_hidden)
#                 else:
#                     num_layers = 4
#                     qwenvl_interval = len(qwenvl_outputs.hidden_states) // num_layers  
#                     qwenvl_index = [i * qwenvl_interval for i in range(1, num_layers)] + [-1]
#                     hiddens = [qwenvl_outputs.hidden_states[i] for i in qwenvl_index]


        
#         # TODO：根据inputid取出图像token
#         bs = qwen_inputs['input_ids'].shape[0]
#         primary_img_idx = qwen_inputs['input_ids'] == 151655
#         primary_img_tokens = [hidden[primary_img_idx].view(bs,primary_img_idx[0].sum(), -1) for hidden in hiddens]
#         token_len = len(primary_img_tokens[0][1])
#         primary_img_tokens = [hidden[:,:token_len//2].view(bs, token_len*2, -1) for hidden in primary_img_tokens]
        
#         depth, dep_conf = self.depth_head(primary_img_tokens, spatial_input[:,:1], patch_start_idx=0)

#         predictions = {
#             "depth": depth,
#             "depth_conf": dep_conf
#         }
#         to_tensor = TF.ToTensor()
#         gt_depth = torch.stack([to_tensor(depth_img)[:1].to(depth.device) for depth_img in batch_depth])
#         batch = {
#             "depths": gt_depth,
#             "point_masks": torch.ones_like(gt_depth).to(torch.bool)
#         }

#         loss_dict = compute_depth_loss(predictions, batch)

#         depth_loss = loss_dict["loss_reg_depth"] +loss_dict["loss_conf_depth"] #+loss_dict["loss_grad_depth"] 
#         return {"action_loss": depth_loss}

#     @torch.inference_mode()
#     def predict_action(
#         self,
#         examples: List[dict],
#         **kwargs: str,
#     ) -> np.ndarray:
#         """
#         Steps:
#           1. Resize images to training resolution (if specified)
#           2. Encode with QwenVL (hidden states retained)
#           6. Return normalized action trajectory
#         Returns:
#             dict:
#                 normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
#         """
#         if type(examples) is not list:
#             examples = [examples]
#         batch_images = [to_pil_preserve(example["image"][:2]) for example in examples]  #  [B，[PLT]]
#         instructions = [example["lang"] for example in examples]  # [B, str]
    
#         state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
#         has_spatial_model = self.config.framework.spatial_model is not None
#         train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
#         if train_obs_image_size:
#             batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
#         with torch.cuda.amp.autocast(dtype=torch.bfloat16):  
#             # Step 1: QWenVL input format
#             qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

#             qwenvl_outputs = self.qwen_vl_interface(
#                 **qwen_inputs,
#                 output_attentions=False,
#                 output_hidden_states=True,
#                 return_dict=True,
#             )    
#             spatial_input = preprocess_images(batch_images, batch_images[0][0].size[0]).to(qwen_inputs['pixel_values'].device)   
#             # step 2: encode spatial feature
#             if has_spatial_model:
#                 if self.spatial_type == "vggt":    
#                     spatial_input = preprocess_images(batch_images, batch_images[0][0].size[0]).to(qwen_inputs['pixel_values'].device)   
#                     aggregated_tokens_list, ps_idx = self.spatial_model.aggregator(spatial_input)
                    
#                 elif self.spatial_type == "depthanything3":
#                     denorm_image = (denorm_image - self._resnet_mean.to(denorm_image.device)) / self._resnet_std.to(denorm_image.device)
#                     feats, aux_feats = self.spatial_model.model.da3.backbone(denorm_image.unsqueeze(1).to(torch.bfloat16),cam_token=None, export_feat_layers=[-1], ref_view_strategy="saddle_balanced")
#                     Bs, S, N, C = feats[0][0].shape
#                     spatial_tokens = feats[-1][0].reshape(Bs*S, N, C)

#                 # step 3: fuse spatial tokens and qwen tokens
#                 num_layers = 4
#                 qwenvl_interval = len(qwenvl_outputs.hidden_states) // num_layers  
#                 qwenvl_index = [i * qwenvl_interval for i in range(1, num_layers)] + [-1]
#                 qwenvl_hidden_states = torch.stack([qwenvl_outputs.hidden_states[i] for i in qwenvl_index])
#                 spatial_interval = len(aggregated_tokens_list) // num_layers
#                 spatial_index = [spatial_interval * i for i in range(1, num_layers)] + [-1]
#                 spatial_hidden_states = torch.stack([aggregated_tokens_list[i][:,0,ps_idx:,:] for i in spatial_index])
#                 spatial_hidden_states = spatial_hidden_states.to(self.spatial_projector.weight.dtype)
#                 spatial_hidden_states = self.spatial_projector(spatial_hidden_states)
                
#                 hiddens = []
#                 for i in range(num_layers):
#                     last_hidden = self.fuser(qwenvl_hidden_states[i], spatial_hidden_states[i])
#                     hiddens.append(last_hidden)
#             else:
#                 num_layers = 4
#                 qwenvl_interval = len(qwenvl_outputs.hidden_states) // num_layers  
#                 qwenvl_index = [i * qwenvl_interval for i in range(1, num_layers)] + [-1]
#                 hiddens = [qwenvl_outputs.hidden_states[i] for i in qwenvl_index]

        
#         # TODO：根据inputid取出图像token
#         bs = qwen_inputs['input_ids'].shape[0]
#         image_grid_thw = qwen_inputs['image_grid_thw']
#         primary_img_idx = qwen_inputs['input_ids'] == 151655
#         primary_img_tokens = [hidden[primary_img_idx].view(bs,primary_img_idx[0].sum(), -1) for hidden in hiddens]
#         token_len = len(primary_img_tokens[0][1])
#         primary_img_tokens = [hidden[:,:token_len//2].view(bs, token_len*2, -1).to(torch.float32) for hidden in primary_img_tokens]

#         depth, dep_conf = self.depth_head(primary_img_tokens, spatial_input[:,:1], patch_start_idx=0)

#         return {"depth": depth}



# if __name__ == "__main__":
#     from omegaconf import OmegaConf
#     import debugpy
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
#     args, clipargs = parser.parse_known_args()

#     # debugpy.listen(("0.0.0.0", 10092))
#     # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
#     # debugpy.wait_for_client()

#     cfg = OmegaConf.load(args.config_yaml)
#     # try get model
#     cfg.framework.action_model.action_hidden_dim = 2048

#     # cfg.framework.qwenvl.base_vlm = f"{CHECKPOINT_BASEDIR}/Florence-2-large"
    

#     model: Qwen_GR00T = Qwen_GR00T(cfg)
#     print(model)



#     # fake sample 
#     image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
#     # Create a sample
#     sample = {
#         "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
#         "image": [image], # three views
#         "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
#         # "state" : np.random.uniform(-1, 1, size=(1, 44)).astype(np.float16), # chunk, state_dim
#     }

#     batch  = [sample, sample]  # batch size 2
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     forward_output = model(batch)
#     action_loss = forward_output['action_loss']
#     print(f"Action Loss: {action_loss.item()}")

#     # test predict action
#     predict_output = model.predict_action(examples=[sample]) #, state=[batch[0]["state"]]
#     normalized_actions = predict_output['normalized_actions']
#     print(f"Unnormalized Action: {normalized_actions}")

#     # # Advance: try forward model with dataloader
#     # # can be fake sample， but here get from dataloader for simpler
#     vla_dataset_cfg = cfg.datasets.vla_data
#     dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

#     from torch.utils.data import DataLoader
#     from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    
#     train_dataloader = DataLoader(
#         dataset,
#         batch_size=2,
#         num_workers=1,  # For Debug
#         collate_fn=collate_fn,
#     )
#     # 
#     for batch in tqdm(train_dataloader, desc="Processing Batches"):
#         batch
#         break

#     # try get model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
#     model(batch)

#     action = model.predict_action(examples=batch)
#     print("Finished")

# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))
import os
CHECKPOINT_BASEDIR = os.getenv('CHECKPOINT_BASEDIR', None)
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as TF
import cv2
import math

from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

from vggt.models.vggt import VGGT
from diffusers import QwenImageEditPlusPipeline
from starVLA.model.modules.projector.QFormer import get_layerwise_qformer
from vggt.heads.dpt_head import DPTHead

class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        nhead: int = 8,
        dropout: float = 0.0,
        kv_dim: int = 2048
    ):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden if d_hidden is not None else d_model
        self.nhead = nhead
        self.head_dim = self.d_hidden // nhead
        assert self.d_hidden % nhead == 0, "d_hidden must be divisible by nhead"

        # Projections
        self.q_proj = nn.Linear(d_model, self.d_hidden)
        self.k_proj = nn.Linear(kv_dim, self.d_hidden)
        self.v_proj = nn.Linear(kv_dim, self.d_hidden)
        self.out_proj = nn.Linear(self.d_hidden, d_model)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, image_feature: torch.Tensor, spatial_feature: torch.Tensor):
        """
        Args:
            image_feature: (B, N_img, d_model) — Query
            vggt_feature:   (B, N_vggt, kv_dim) — Key and Value

        Returns:
            fused_image_feature: (B, N_img, d_model)
        """
        B, N_img, _ = image_feature.shape
        _, N_spatial, _ = spatial_feature.shape

        # Project to d_hidden
        q = self.q_proj(image_feature)   # (B, N_img, d_hidden)
        k = self.k_proj(spatial_feature)     # (B, N_vggt, d_hidden)
        v = self.v_proj(spatial_feature)     # (B, N_vggt, d_hidden)

        # Reshape for multi-head: (B, N, d_hidden) -> (B, N, nhead, head_dim) -> (B, nhead, N, head_dim)
        q = q.view(B, N_img, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_img, head_dim)
        k = k.view(B, N_spatial, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_vggt, head_dim)
        v = v.view(B, N_spatial, self.nhead, self.head_dim).transpose(1, 2)  # (B, nhead, N_vggt, head_dim)

        # Scaled Dot-Product Attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, nhead, N_img, N_vggt)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)

        # Weighted sum over values
        attn_output = torch.matmul(attn_weights, v)  # (B, nhead, N_img, head_dim)

        # Concatenate heads and project back
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, N_img, nhead, head_dim)
        attn_output = attn_output.view(B, N_img, self.d_hidden)  # (B, N_img, d_hidden)

        # Final projection to d_model
        output = self.out_proj(attn_output)  # (B, N_img, d_model)
        output = self.dropout_out(output)

        # Residual connection + LayerNorm
        output = self.norm(image_feature + output)

        return output

class Attention(nn.Module):
    def __init__(self, dim, num_heads=40, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        # qkv shape: [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, embed_dim=2560, num_heads=16, dropout=0.1, bias=True):
        """
        Self-Attention Module with Residual Connection and Layer Norm
        
        Args:
            embed_dim (int): 输入 token 的维度
            num_heads (int): 注意力头的数量
            dropout (float): Dropout 概率
            bias (bool): 线性层是否使用 bias
        """
        super(SelfAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 确保维度可以整除
        assert self.head_dim * num_heads == self.embed_dim, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        # 【新增】层归一化 LayerNorm
        # Pre-LN 架构：在输入 Attention 之前进行归一化
        self.norm = nn.LayerNorm(embed_dim)

        # 定义 Q, K, V 的线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # 输出投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (bs, l, embed_dim)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (bs, l, embed_dim) with residual connection
        """
        bs, l, _ = x.shape

        # ==========================================
        # 【关键修改 1】: Pre-LN 归一化
        # ==========================================
        # 对输入进行归一化，有助于梯度流动
        x_normed = self.norm(x)

        # 1. 线性投影得到 Q, K, V (使用归一化后的输入)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        # 2. 多头 reshape 和 transpose
        q = q.view(bs, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, l, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算 Attention Scores
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask, 
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Fallback for older PyTorch versions
            scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        # 4. 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, l, self.embed_dim)

        # 5. 最终线性投影
        output_proj = self.out_proj(attn_output)
        
        # 应用 Dropout 到投影后的输出 (可选，但推荐)
        output_proj = self.dropout(output_proj)

        # ==========================================
        # 【关键修改 2】: 残差连接 (Residual Connection)
        # ==========================================
        # 将原始输入 x 加到注意力输出上
        # 注意：这里加的是原始的 x，而不是 x_normed
        out = x + output_proj
        
        return out

class MMDITBlock(nn.Module):
    def __init__(self, dim=2560, num_heads=40, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        
        # --- 模态 A 的组件 ---
        self.norm1_a = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_a = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp_a = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        # 固定 temb 后，调制参数变为可学习的固定向量 (Scale/Shift/Gate)
        self.mod_a = nn.Parameter(torch.zeros(6, dim)) 

        # --- 模态 B 的组件 ---
        self.norm1_b = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2_b = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp_b = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.mod_b = nn.Parameter(torch.zeros(6, dim))

        # --- 共享的注意力机制 ---
        self.attn = Attention(dim, num_heads=num_heads)
        
        self.init_weights()

    def init_weights(self):
        # 按照 DiT 的惯例，初始化 gate 为 0，使初始状态接近恒等映射
        nn.init.constant_(self.mod_a, 0)
        nn.init.constant_(self.mod_b, 0)

    def apply_modulate(self, x, norm_layer, shift, scale):
        # 对应：norm(x) * (1 + scale) + shift
        return norm_layer(x) * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)

    def forward(self, x_a, x_b):
        """
        x_a: 特征 A, 形状 [B, N_a, 2560]
        x_b: 特征 B, 形状 [B, N_b, 2560]
        """
        # 1. 拆分调制参数 (每个模态 6 个)
        # mod_a shape: [6, 2560]
        shft_msa_a, scl_msa_a, gate_msa_a, shft_mlp_a, scl_mlp_a, gate_mlp_a = self.mod_a.chunk(6, dim=0)
        shft_msa_b, scl_msa_b, gate_msa_b, shft_mlp_b, scl_mlp_b, gate_mlp_b = self.mod_b.chunk(6, dim=0)

        # 2. Attention 前的预归一化与调制
        norm_x_a = self.apply_modulate(x_a, self.norm1_a, shft_msa_a, scl_msa_a)
        norm_x_b = self.apply_modulate(x_b, self.norm1_b, shft_msa_b, scl_msa_b)

        # 3. 联合注意力处理 (Joint Attention)
        # 拼接两个序列
        combined = torch.cat([norm_x_a, norm_x_b], dim=1)
        attn_out = self.attn(combined)
        
        # 拆分回原序列
        attn_a = attn_out[:, :x_a.shape[1], :]
        attn_b = attn_out[:, x_a.shape[1]:, :]

        # 4. 残差连接与门控 (Gate)
        x_a = x_a + gate_msa_a.unsqueeze(0) * attn_a
        x_b = x_b + gate_msa_b.unsqueeze(0) * attn_b

        # 5. MLP 部分
        # 模态 A
        norm_mlp_a = self.apply_modulate(x_a, self.norm2_a, shft_mlp_a, scl_mlp_a)
        x_a = x_a + gate_mlp_a.unsqueeze(0) * self.mlp_a(norm_mlp_a)
        
        # 模态 B
        norm_mlp_b = self.apply_modulate(x_b, self.norm2_b, shft_mlp_b, scl_mlp_b)
        x_b = x_b + gate_mlp_b.unsqueeze(0) * self.mlp_b(norm_mlp_b)

        return x_a, x_b

def preprocess_images(image_list, target_size, mode='crop'): #  [B，[PLT]]
    batch_images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    # target_size = 518

    # First process all images and collect their shapes
    for imgs in image_list:
        epi_images = []
        img = imgs[0]
        # for img in imgs:
        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        # img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        epi_images.append(img)
        batch_images.append(torch.stack(epi_images))

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in batch_images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        batch_images = padded_images

    batch_images = torch.stack(batch_images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_list) == 1:
        # Verify shape is (1, C, H, W)
        if batch_images.dim() == 3:
            batch_images = batch_images.unsqueeze(0)
    return batch_images

def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
    
    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                  [-hard_max, hard_max] will be clamped. If None, 
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor
    
    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor

def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = '', valid_range=0.98, **kwargs):
    """
    Compute depth loss.
    
    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['depth']
    pred_depth_conf = predictions['depth_conf']

    gt_depth = batch['depths']
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]              # (B, H, W, 1)
    gt_depth_mask = batch['point_masks'].clone()   # 3D points derived from depth map, so we use the same mask

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_depth": dummy_loss,
                    f"loss_reg_depth": dummy_loss,
                    f"loss_grad_depth": dummy_loss,}
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,    
        f"loss_grad_depth": loss_grad,
    }

    return loss_dict

def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out

def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn='', gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


@FRAMEWORK_REGISTRY.register("QwenGR00TDPT")
class Qwen_GR00TDPT(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        # self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        # self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        # self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        # self.future_action_window_size = config.framework.action_model.future_action_window_size
        # self.past_action_window_size = config.framework.action_model.past_action_window_size
        # self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        if getattr(self.config.framework, 'spatial_model', None) is not None:
            self.spatial_model = self.get_spatial_model(config)
            self.spatial_projector = self.get_spatial_projector(config)

        if getattr(self.config.framework, 'image_edit_model', None) is not None:
            if not self.config.framework.image_edit_model.read_from_local:
                from starVLA.model.modules.longcat_image_edit_model import LongCatImageEditModel
                from diffusers import QwenImageEditPlusPipeline
                if 'Qwen' in config.framework.image_edit_model.model_name_or_path:
                    self.image_edit_model = QwenImageEditPlusPipeline.from_pretrained(config.framework.image_edit_model.model_name_or_path, torch_dtype=torch.bfloat16)
                elif 'LongCat' in config.framework.image_edit_model.model_name_or_path:
                    # self.image_edit_model = LongCatImageEditPipeline.from_pretrained(config.framework.image_edit_model.model_name_or_path, torch_dtype=torch.bfloat16)
                    self.image_edit_model = LongCatImageEditModel.from_pretrained(config.framework.image_edit_model.model_name_or_path, lora_path=config.framework.image_edit_model.lora_path, torch_dtype=torch.bfloat16)
                else:
                    raise NotImplementedError

            self.image_edit_projector = nn.Linear(64, 2560)

            if getattr(self.config.framework.image_edit_model, 'fuser_type', None) is not None:    
                print(self.config.framework.image_edit_model.fuser_type)
                if self.config.framework.image_edit_model.fuser_type == 'mmdit':
                    self.spatial_fuser = MMDITBlock()
                elif self.config.framework.image_edit_model.fuser_type == 'cross_attention' or self.config.framework.image_edit_model.fuser_type == 'inv_cross_attention':
                    self.spatial_fuser = self.get_cross_attention(d_model=config.framework.spatial_projector.output_dim,d_hidden=config.framework.spatial_projector.output_dim,kv_dim=2560)
                elif self.config.framework.image_edit_model.fuser_type == 'self_attention':
                    self.spatial_fuser = SelfAttention(embed_dim=config.framework.spatial_projector.output_dim)
                elif self.config.framework.image_edit_model.fuser_type == 'gated_fusion':
                    self.geo2mv = self.get_cross_attention(d_model=config.framework.spatial_projector.output_dim,d_hidden=config.framework.spatial_projector.output_dim,kv_dim=2560)
                    self.mv2geo = self.get_cross_attention(d_model=config.framework.spatial_projector.output_dim,d_hidden=config.framework.spatial_projector.output_dim,kv_dim=2560)
                    self.spatial_fuser = nn.Sequential(
                        nn.Linear(config.framework.spatial_projector.output_dim * 2, config.framework.spatial_projector.output_dim),
                        nn.GELU(),
                        nn.Linear(config.framework.spatial_projector.output_dim, 1),
                        nn.Sigmoid()
                    )
                elif self.config.framework.image_edit_model.fuser_type == 'mlp_fusion':
                    self.spatial_fuser = SelfAttention(embed_dim=config.framework.spatial_projector.output_dim)
                    self.view_selector = nn.Sequential(
                        nn.Linear(config.framework.spatial_projector.output_dim * 2, config.framework.spatial_projector.output_dim),
                        nn.GELU(),
                        nn.Linear(config.framework.spatial_projector.output_dim, 1),
                        nn.Sigmoid()
                    )
                elif self.config.framework.image_edit_model.fuser_type == 'residual_fusion':
                    self.spatial_fuser = SelfAttention(embed_dim=config.framework.spatial_projector.output_dim)
                    self.view_selector = nn.Sequential(
                        nn.Linear(config.framework.spatial_projector.output_dim * 2, config.framework.spatial_projector.output_dim),
                        nn.GELU(),
                        nn.Linear(config.framework.spatial_projector.output_dim, 1),
                        nn.Sigmoid()
                    )
                elif self.config.framework.image_edit_model.fuser_type == 'mlp_gated_tranformer':
                    self.spatial_fuser = SelfAttention(embed_dim=config.framework.spatial_projector.output_dim)
                    self.view_selector = nn.Sequential(
                        nn.Linear(config.framework.spatial_projector.output_dim * 2, config.framework.spatial_projector.output_dim),
                        nn.GELU(),
                        nn.Linear(config.framework.spatial_projector.output_dim, 1),
                        nn.Sigmoid()
                    )
                    self.spatial_fuser2 = SelfAttention(embed_dim=config.framework.spatial_projector.output_dim)

        self.depth_head = DPTHead(dim_in=2560, output_dim=2, features=512, activation="exp", conf_activation="expp1",intermediate_layer_idx=[0,0,0,0])
        
                # else:
                #     raise NotImplementedError
        # if getattr(self.config.framework, 'fuser', None) is None:
        #     self.config.framework.fuser = {'type':'cross_attention'}
        # print(self.config.framework.fuser.type)
        # if self.config.framework.fuser.type == 'cross_attention':
        #     self.fuser = self.get_cross_attention(d_model=config.framework.spatial_projector.output_dim,d_hidden=config.framework.spatial_projector.output_dim,kv_dim=config.framework.spatial_projector.output_dim)
        # elif self.config.framework.fuser.type == 'mlayer':
        #     self.fuser = get_layerwise_qformer(config=self.config)

    def get_cross_attention(self, d_model, d_hidden, kv_dim):
        model = CrossAttention(d_model=d_model,d_hidden=d_hidden,kv_dim=kv_dim)
        return model
        
    def get_spatial_model(self, config):
        spatial_model_cfg = config.framework.spatial_model
        if "vggt" in spatial_model_cfg.model_name_or_path:
            self.spatial_type = "vggt"
            spatial_model = VGGT.from_pretrained(spatial_model_cfg.model_name_or_path)
        else:
            raise NotImplementedError
        return spatial_model

    def get_spatial_projector(self, config):
        spatial_projector_cfg = config.framework.spatial_projector
        projector = nn.Linear(config.framework.spatial_model.output_dim, spatial_projector_cfg.output_dim)
        return projector

    def forward_pass_image_edit_model(self, images, prompt=None):
        view_num = getattr(self.config.framework.image_edit_model, 'view_num', 1)
        prompts = ['Rotate the camera view to the left', 'Rotate the camera view to the right']
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if view_num == 1:
                    inputs = {
                        "images": images,
                        "prompts": random.choices(prompts, k=len(images)),
                        "generator": torch.Generator("cuda").manual_seed(43),
                        "num_inference_steps": 4,
                        "guidance_scale": 1.0,
                        "output_type": "latent",
                        "device": 'cuda',
                        'height': 256,
                        'width': 256,
                    }
                    output = self.image_edit_model(**inputs)
                else:
                    outputs = []
                    for i in range(view_num):
                        inputs = {
                            "images": images,
                            "prompts": [prompts[i]] * len(images),
                            "generator": torch.Generator("cuda").manual_seed(43),
                            "num_inference_steps": self.config.framework.image_edit_model.num_inference_steps,
                            "guidance_scale": 1.0,
                            "output_type": "latent",
                            "device": 'cuda',
                            'height': 256,
                            'width': 256,
                        }
                        output = self.image_edit_model(**inputs)
                        outputs.append(output)
                    output = torch.cat(outputs, dim=1)

        return output
        
    def forward_pass_VGGT(self, batch_images, instructions, mv_feat=None):

        # Step 1: QWenVL input format
        # qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # qwenvl_outputs = self.qwen_vl_interface(
            #     **qwen_inputs,
            #     output_attentions=False,
            #     output_hidden_states=True,
            #     return_dict=True,
            # )    
            if getattr(self, 'spatial_model', None) is not None:
                # step 2: encode spatial feature
                with torch.no_grad():
                    if self.spatial_type == "vggt":    
                        spatial_input = preprocess_images(batch_images, batch_images[0][0].size[0]).to('cuda')   
                        aggregated_tokens_list, ps_idx = self.spatial_model.aggregator(spatial_input)
                        
                    elif self.spatial_type == "depthanything3":
                        denorm_image = (denorm_image - self._resnet_mean.to(denorm_image.device)) / self._resnet_std.to(denorm_image.device)
                        feats, aux_feats = self.spatial_model.model.da3.backbone(denorm_image.unsqueeze(1).to(torch.bfloat16),cam_token=None, export_feat_layers=[-1], ref_view_strategy="saddle_balanced")
                        Bs, S, N, C = feats[0][0].shape
                        spatial_tokens = feats[-1][0].reshape(Bs*S, N, C)
                    else:
                        raise NotImplementedError
            # import ipdb
            # ipdb.set_trace()
            extra_latents = None
            if getattr(self.config.framework, 'image_edit_model', None) is not None:
                if mv_feat is not None:
                    extra_latents = torch.tensor(np.array(mv_feat), device=qwen_inputs['pixel_values'].device, dtype=qwen_inputs['pixel_values'].dtype)
                else:
                    primary_image = [image[0] for image in batch_images]
                    extra_latents = self.forward_pass_image_edit_model(primary_image)
            

        # step 3: fuse spatial tokens and qwen tokens
        with torch.autocast("cuda", dtype=torch.float32):
            if self.config.framework.fuser.type == 'cross_attention':
                # last_hidden_state: [B, seq_len, H]
                # last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
                spatial_tokens = None
                if getattr(self, 'spatial_model', None) is not None:
                    if self.spatial_type == "vggt":
                        spatial_tokens = aggregated_tokens_list[-1][:,0,ps_idx:,:]
                    else:
                        raise NotImplementedError

                    spatial_tokens = self.spatial_projector(spatial_tokens)

                if extra_latents is not None:
                    extra_latents = self.image_edit_projector(extra_latents)
                    if spatial_tokens is not None:
                        extra_latents = extra_latents.to(spatial_tokens.dtype)
                        if getattr(self, 'spatial_fuser', None) is not None:
                            if self.config.framework.image_edit_model.fuser_type == 'mmdit':
                                spatial_tokens, extra_latents = self.spatial_fuser(spatial_tokens, extra_latents)
                                spatial_tokens = torch.cat([spatial_tokens, extra_latents], dim=1)
                            elif self.config.framework.image_edit_model.fuser_type == 'cross_attention':
                                view_num = getattr(self.config.framework.image_edit_model, 'view_num', 1)
                                B, L, D = extra_latents.shape

                                # 确保长度可以被整除
                                assert L % view_num == 0, f"extra_latents length {L} is not divisible by view_num {view_num}"

                                # 将 extra_latents 沿 L 维度切分成 view_num 个 chunk
                                # 每个 chunk 的形状为 (B, L // view_num, D)
                                latent_chunks = torch.chunk(extra_latents, chunks=view_num, dim=1)

                                fusion_results = []
                                for i in range(view_num):
                                    # 对每一份执行融合操作
                                    # 注意：如果 spatial_tokens 是共享的，直接传入；如果是多视角的，可能也需要拆分
                                    ri = self.spatial_fuser(spatial_tokens, latent_chunks[i])
                                    fusion_results.append(ri)

                                # 将结果拼接回来
                                extra_latents = torch.cat(fusion_results, dim=1) 
                                spatial_tokens = torch.cat([spatial_tokens, extra_latents], dim=1)
                            elif self.config.framework.image_edit_model.fuser_type == 'inv_cross_attention':
                                view_num = getattr(self.config.framework.image_edit_model, 'view_num', 1)
                                B, L, D = extra_latents.shape

                                # 确保长度可以被整除
                                assert L % view_num == 0, f"extra_latents length {L} is not divisible by view_num {view_num}"

                                # 将 extra_latents 沿 L 维度切分成 view_num 个 chunk
                                # 每个 chunk 的形状为 (B, L // view_num, D)
                                latent_chunks = torch.chunk(extra_latents, chunks=view_num, dim=1)

                                fusion_results = []
                                for i in range(view_num):
                                    # 对每一份执行融合操作
                                    ri = self.spatial_fuser(latent_chunks[i], spatial_tokens)
                                    fusion_results.append(ri)

                                # 将结果拼接回来
                                extra_latents = torch.cat(fusion_results, dim=1) 
                                spatial_tokens = torch.cat([spatial_tokens, extra_latents], dim=1)
                            elif self.config.framework.image_edit_model.fuser_type == 'concat':
                                spatial_tokens = torch.cat([spatial_tokens, extra_latents], dim=1)
                            elif self.config.framework.image_edit_model.fuser_type == 'self_attention':
                                extra_latents = torch.cat([spatial_tokens, extra_latents], dim=1)
                                spatial_tokens = self.spatial_fuser(extra_latents)
                            elif self.config.framework.image_edit_model.fuser_type == "gated_fusion":
                                B, L, D = extra_latents.shape
                                view_num = getattr(self.config.framework.image_edit_model, 'view_num', 1)
                                latent_chunks = torch.chunk(extra_latents, chunks=view_num, dim=1)
                                # fuse spatial feature to multi-view feature
                                fusion_results = [spatial_tokens]
                                spatial_aware_feat = []
                                for i in range(view_num):
                                    # 对每一份执行融合操作
                                    ri = self.mv2geo(latent_chunks[i], spatial_tokens)
                                    spatial_aware_feat.append(ri)
                                spatial_aware_feat = torch.cat(spatial_aware_feat, dim=1)
                                refined_spatial_tokens = self.geo2mv(spatial_tokens, spatial_aware_feat)
                                gate = self.spatial_fuser(torch.cat([spatial_tokens,refined_spatial_tokens], dim=-1))
                                spatial_tokens = gate * spatial_tokens + (1 - gate) * refined_spatial_tokens
                                
                                # 将结果拼接回来
                                spatial_tokens = torch.cat([spatial_tokens, spatial_aware_feat], dim=1)
                            elif self.config.framework.image_edit_model.fuser_type == 'mlp_fusion':
                                view_num = getattr(self.config.framework.image_edit_model, 'view_num', 1)
                                assert view_num == 2, f"view num should be 2"
                                spatial_token_num = spatial_tokens.shape[1]
                                fused_tokens = torch.cat([spatial_tokens, extra_latents], dim=1)
                                fused_tokens = self.spatial_fuser(fused_tokens)
                                fused_spatial_tokens = fused_tokens[:,:spatial_token_num,:]
                                fused_extra_latents = torch.chunk(fused_tokens[:,spatial_token_num:,:], chunks=view_num, dim=1)
                                gate = self.view_selector(torch.cat(fused_extra_latents, dim=-1))
                                fused_view = gate * fused_extra_latents[0] + (1 - gate) * fused_extra_latents[1]
                                spatial_tokens = torch.cat([fused_spatial_tokens, fused_view], dim=1)
                            elif self.config.framework.image_edit_model.fuser_type == 'mlp_gated_tranformer':
                                view_num = getattr(self.config.framework.image_edit_model, 'view_num', 1)
                                assert view_num == 2, f"view num should be 2"
                                spatial_token_num = spatial_tokens.shape[1]
                                fused_tokens = torch.cat([spatial_tokens, extra_latents], dim=1)
                                fused_tokens = self.spatial_fuser(fused_tokens)
                                fused_spatial_tokens = fused_tokens[:,:spatial_token_num,:]
                                fused_extra_latents = torch.chunk(fused_tokens[:,spatial_token_num:,:], chunks=view_num, dim=1)
                                gate = self.view_selector(torch.cat(fused_extra_latents, dim=-1))
                                fused_view = gate * fused_extra_latents[0] + (1 - gate) * fused_extra_latents[1]
                                spatial_tokens = torch.cat([fused_spatial_tokens, fused_view], dim=1)
                                spatial_tokens = self.spatial_fuser2(spatial_tokens)
                                spatial_tokens = spatial_tokens[:,:spatial_token_num,:]
                            elif self.config.framework.image_edit_model.fuser_type == 'residual_fusion':
                                view_num = getattr(self.config.framework.image_edit_model, 'view_num', 1)
                                assert view_num == 2, f"view num should be 2"
                                spatial_token_num = spatial_tokens.shape[1]
                                fused_tokens = torch.cat([spatial_tokens, extra_latents], dim=1)
                                fused_tokens = self.spatial_fuser(fused_tokens)
                                fused_spatial_tokens = fused_tokens[:,:spatial_token_num,:]
                                fused_extra_latents = fused_tokens[:,spatial_token_num:,:]
                                residual = fused_extra_latents - extra_latents
                                residual = torch.chunk(residual, chunks=view_num, dim=1) # 
                                fused_extra_latents = torch.chunk(fused_extra_latents, chunks=view_num, dim=1)
                                gate = self.view_selector(torch.cat(residual, dim=-1))
                                fused_view = gate * fused_extra_latents[0] + (1 - gate) * fused_extra_latents[1]
                                spatial_tokens = torch.cat([fused_spatial_tokens, fused_view], dim=1)

                    else:
                        spatial_tokens = extra_latents
                # last_hidden = self.fuser(last_hidden, spatial_tokens)
            elif self.config.framework.fuser.type == 'concat':
                # last_hidden_state: [B, seq_len, H]
                last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
                spatial_tokens = None
                if getattr(self, 'spatial_model', None) is not None:
                    if self.spatial_type == "vggt":
                        spatial_tokens = aggregated_tokens_list[-1][:,0,ps_idx:,:]
                    else:
                        raise NotImplementedError

                    spatial_tokens = self.spatial_projector(spatial_tokens)

                if extra_latents is not None:
                    extra_latents = self.image_edit_projector(extra_latents)
                    if spatial_tokens is not None:
                        spatial_tokens = torch.cat([spatial_tokens, extra_latents.to(spatial_tokens.dtype)], dim=1)
                    else:
                        spatial_tokens = extra_latents
                last_hidden = torch.cat([last_hidden, spatial_tokens], dim=1)
                if extra_latents is not None:
                    last_hidden = torch.cat([last_hidden, extra_latents.to(last_hidden.dtype)], dim=1)
            elif self.config.framework.fuser.type == 'mlayer':
                num_layers = self.config.framework.layer_qformer.num_layers
                qwenvl_interval = len(qwenvl_outputs.hidden_states) // num_layers  
                qwenvl_index = [i * qwenvl_interval for i in range(1, num_layers)] + [-1]
                qwenvl_hidden_states = torch.stack([qwenvl_outputs.hidden_states[i] for i in qwenvl_index])
                spatial_interval = len(aggregated_tokens_list) // num_layers
                spatial_index = [spatial_interval * i for i in range(1, num_layers)] + [-1]
                spatial_hidden_states = torch.stack([aggregated_tokens_list[i][:,0,ps_idx:,:] for i in spatial_index])
                spatial_hidden_states = self.spatial_projector(spatial_hidden_states)

                cat_conditions = []
                for layer_index in range(num_layers):
                    if extra_latents is not None:
                        layer_features = torch.cat(
                            [qwenvl_hidden_states[layer_index], spatial_hidden_states[layer_index], extra_latents.to(spatial_hidden_states[layer_index].dtype)], dim=1
                        )
                    else:
                        layer_features = torch.cat(
                            [qwenvl_hidden_states[layer_index], spatial_hidden_states[layer_index]], dim=1
                        )
                    cat_conditions.append(layer_features)

                last_hidden = self.fuser(cat_conditions)[0]
            else:
                raise NotImplementedError
           
        return spatial_tokens, spatial_input

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        # import ipdb
        # ipdb.set_trace()
        batch_images = [example["image"][:2] for example in examples]  #  [B，[PLT]]
        batch_depth = [example["image"][2:3] for example in examples]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        use_state = getattr(self.config.framework.action_model, 'use_state', False)
        if not use_state:
            state = None

        mv_feat = [example["mv_feat"] for example in examples] if "mv_feat" in examples[0] else None

        spatial_tokens, spatial_input = self.forward_pass_VGGT(batch_images, instructions, mv_feat)
        spatial_tokens = spatial_tokens.to(torch.float32)
        # token_len = len(primary_img_tokens[0][1])
        # primary_img_tokens = [hidden[:,:token_len//2].view(bs, token_len*2, -1) for hidden in primary_img_tokens]
        depth, dep_conf = self.depth_head(spatial_tokens[None,:,None], spatial_input[:,:1], patch_start_idx=0)

        predictions = {
            "depth": depth,
            "depth_conf": dep_conf
        }
        to_tensor = TF.ToTensor()
        gt_depth = torch.stack([to_tensor(depth_img[0]).to('cuda') for depth_img in batch_depth])[:,:1,:,:]
        batch = {
            "depths": gt_depth,
            "point_masks": torch.ones_like(gt_depth).to(torch.bool)
        }

        loss_dict = compute_depth_loss(predictions, batch)

        depth_loss = loss_dict["loss_reg_depth"] + loss_dict["loss_conf_depth"] #+loss_dict["loss_grad_depth"] 
        return {"action_loss": depth_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        # actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        # state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        # use_state = getattr(self.config.framework.action_model, 'use_state', False)
        # if not use_state:
        #     state = None

        mv_feat = [example["mv_feat"] for example in examples] if "mv_feat" in examples[0] else None

        spatial_tokens, spatial_input = self.forward_pass_VGGT(batch_images, instructions, mv_feat)
        # import ipdb
        # ipdb.set_trace()
        spatial_tokens = spatial_tokens.to(torch.float32)
        depth, dep_conf = self.depth_head(spatial_tokens[None,:,None], spatial_input[:,:1], patch_start_idx=0)

       
        # depth, dep_conf = self.depth_head(spatial_tokens, spatial_input[:,:1], patch_start_idx=0)
        return {"depth": depth}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.action_model.action_hidden_dim = 2048

    # cfg.framework.qwenvl.base_vlm = f"{CHECKPOINT_BASEDIR}/Florence-2-large"
    

    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        # "state" : np.random.uniform(-1, 1, size=(1, 44)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(examples=[sample]) #, state=[batch[0]["state"]]
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    vla_dataset_cfg = cfg.datasets.vla_data
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    # 
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        batch
        break

    # try get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model(batch)

    action = model.predict_action(examples=batch)
    print("Finished")
