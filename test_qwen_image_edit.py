import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from io import BytesIO
import requests
import torch
import torch.nn as nn
import time
pipeline = QwenImageEditPlusPipeline.from_pretrained("/mnt/xlab-nas-1/junjin/pretrained_models/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16)
# del pipeline.vae.decoder
# del pipeline.vae.post_quant_conv
# pipeline.vae.decoder = None
# pipeline.vae.post_quant_conv = None
print("pipeline loaded")
# pipeline.enable_model_cpu_offload()
# pipeline.load_lora_weights("/mnt/xlab-nas-1/junjin/pretrained_models/Qwen-Image-Edit-2511-Lightning", weight_name='Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors')
pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open('img1.jpg')


prompt = "Rotate the camera to left."
inputs = {
    "image": [image1],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
    "output_type": "pil"
}

# projector = TokenDownsampler().to('cuda',torch.bfloat16)
import ipdb
ipdb.set_trace()
for epoch in range(5):
    with torch.inference_mode():
        start = time.time()
        output = pipeline(**inputs)
        
        # latents = projector(output.images)
        print(f"model time: {time.time-starts}")
        output_image = output.images[0]
        output_image.save("output_image_edit_2511_1.png")
        # output_image = output.images[1]
        # output_image.save("output_image_edit_2511_2.png")
