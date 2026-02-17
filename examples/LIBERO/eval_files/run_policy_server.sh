#!/bin/bash

your_ckpt=/mnt/workspace/junjin/code/starVLA/checkpoints/0215_liberoall_Qwen3vlGR00T_vggt_view2_cross_bs16/checkpoints/steps_30000_pytorch_model.pt
base_port=9880
export star_vla_python=/mnt/workspace/junjin/conda/starvla/bin/python

export DEBUG=1

CUDA_VISIBLE_DEVICES=2 python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16