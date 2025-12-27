#!/bin/bash

# your_ckpt=/mnt/workspace/junjin/code/starVLA/checkpoints/1222_liberoall_Qwen3vlGR00T_vggt_mlayer/checkpoints/steps_20000_pytorch_model.pt
your_ckpt=/mnt/workspace/zengshuang.zs/checkpoints/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt
base_port=9882
export star_vla_python=/mnt/workspace/junjin/conda/starVLA/bin/python

export DEBUG=1

CUDA_VISIBLE_DEVICES=2 python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16