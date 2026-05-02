#!/bin/bash

your_ckpt=/mnt/workspace/zengshuang.zs/output/roboarena/0315_roboarena_QwenJAT_sft_joint_worker2/checkpoints/steps_100000_pytorch_model.pt
base_port=9980
export star_vla_python=/mnt/workspace/junjin/conda/starvla/bin/python

# export DEBUG=1

CUDA_VISIBLE_DEVICES=1 python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16