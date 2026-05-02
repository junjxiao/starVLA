import os
import torch
from PIL import Image
# from diffusers import LongCatImageEditPipeline
import requests
import torch
import torch.nn as nn
import time
from starVLA.model.modules.longcat_image_edit_model import LongCatImageEditModel
pipeline = LongCatImageEditModel.from_pretrained("/mnt/xlab-nas-1/junjin/pretrained_models/LongCat-Image-Edit", lora_path='/mnt/workspace/junjin/code/LongCat-Image/output/edit_lora_model_robotwin_10000step/checkpoints-2000',torch_dtype=torch.bfloat16)

# pipeline = LongCatImageEditPipeline.from_pretrained("/mnt/xlab-nas-1/junjin/pretrained_models/LongCat-Image-Edit", torch_dtype= torch.bfloat16)
# pipeline = QwenImageEditPlusPipeline.from_pretrained("/mnt/xlab-nas-1/junjin/pretrained_models/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
# del pipeline.vae.decoder
# del pipeline.vae.post_quant_conv
# pipeline.vae.decoder = None
# pipeline.vae.post_quant_conv = None
print("pipeline loaded")
# pipeline.enable_model_cpu_offload()
# pipeline.load_lora_weights("/mnt/xlab-nas-1/junjin/pretrained_models/Qwen-Image-Edit-2511-Lightning", weight_name='Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors')
pipeline.to('cuda')
# pipeline.set_progress_bar_config(disable=None)
image1 = Image.open('a.jpg')
# image2 = Image.open('d.jpg')
image1 = image1.resize((224, 224))
# image2 = image2.resize((224, 224))
if image1.mode == 'RGBA':
    image1 = image1.convert('RGB')
# if image2.mode == 'RGBA':
#     image2 = image2.convert('RGB')


# prompt = "Preserve scene layout, object positions, and spatial relationships; only slightly adjust camera viewpoint, background color, and lighting."
prompt = 'Rotate the camera view to the right'
inputs = {
    "images": [image1],
    "prompts": [prompt] * 1,
    "generator": torch.Generator("cuda").manual_seed(43),
    "num_inference_steps": 50,
    "guidance_scale": 1.0,
    "output_type": "pil", #latent
    "device": 'cuda',
    'width': 256,
    'height': 256
}

# projector = TokenDownsampler().to('cuda',torch.bfloat16)


with torch.inference_mode():
    output = pipeline(**inputs)
    
    output[1][0].save("a_out_right.jpg")
    # output[1].save("d_out_right.jpg")
    

