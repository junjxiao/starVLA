#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/mnt/workspace/junjin/conda/starvla/bin/python
your_ckpt=/mnt/workspace/junjin/code/starVLA/checkpoints/0127_robotwin_Qwen3vlGR00T_vggt_cross_bs16/checkpoints/steps_10000_pytorch_model.pt
gpu_id=3
port=5695
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
