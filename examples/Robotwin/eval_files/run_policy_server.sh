#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/mnt/workspace/junjin/conda/starvla/bin/python
your_ckpt=/mnt/workspace/junjin/code/starVLA/checkpoints/0115_robotwin_Qwen3vlGR00T_vggt_cross_bs16/checkpoints/steps_20000_pytorch_model.pt
gpu_id=0
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
