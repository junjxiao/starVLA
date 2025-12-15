#!/bin/bash

your_ckpt=/mnt/workspace/zengshuang.zs/checkpoints/Qwen2.5-VL-GR00T-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt
base_port=10012
export star_vla_python=/mnt/workspace/zengshuang.zs/env/starVLA/bin/python

export DEBUG=1

CUDA_VISIBLE_DEVICES=3 python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16