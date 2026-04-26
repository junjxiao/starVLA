#!/bin/bash

your_ckpt=/mnt/workspace/junjin/code/starVLA/checkpoints/0423_liberoall_Qwen3vlGR00TAML_vggt_longcat_view2_cross_mlp_gated_tranformer_ck10_JAT2048_14k_bs16_4gpus_reload_vlm_action/checkpoints/steps_40000_pytorch_model.pt
base_port=9880
export star_vla_python=/mnt/workspace/junjin/conda/starvla/bin/python

# export DEBUG=1

CUDA_VISIBLE_DEVICES=2 python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${base_port} \
    --use_bf16