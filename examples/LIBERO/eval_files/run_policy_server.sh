#!/bin/bash

your_ckpt=/mnt/workspace/junjin/code/starVLA/checkpoints/0111_liberoall_Qwen3vlGR00T_no_vggt_longcat_image_edit_cross_bs8/checkpoints/steps_3000_pytorch_model.pt
base_port=9879
export star_vla_python=/mnt/workspace/junjin/conda/starvla/bin/python

export DEBUG=1

CUDA_VISIBLE_DEVICES=0 python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16