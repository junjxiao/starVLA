#!/bin/bash

your_ckpt=/mnt/workspace/zengshuang.zs/output/libero_all/1212_libero4in1_QwenGR00T/checkpoints/steps_10000_pytorch_model.pt
base_port=10010
export star_vla_python=/mnt/workspace/zengshuang.zs/env/starVLA/bin/python

export DEBUG=1

CUDA_VISIBLE_DEVICES=3 python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16