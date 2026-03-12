#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd) 
ckpt=/mnt/workspace/junjin/code/starVLA/checkpoints/0303_liberoall_Qwen3vlGR00T_vggt_longcat_view2_cross_bs16_4gpus/checkpoints/steps_30000_pytorch_model.pt
log_dir=/mnt/workspace/junjin/code/starVLA/outputs/droid/test
port=9880
scene=1
unnorm_key="franka"
# export CUDA_VISIBLE_DEVICES=3
python examples/sim-evals/run_eval_starvla.py --episodes 10 --scene $scene --headless --log_dir $log_dir --port $port --ckpt $ckpt --unnorm_key $unnorm_key


