
# sim_python=~/Envs/miniconda3/envs/starVLA/bin/python
port=56705
# export DEBUG=true
export star_vla_python=/mnt/workspace/junjin/conda/starVLA/bin/python

# your_ckpt=/mnt/workspace/zengshuang.zs/checkpoints/Qwen-GR00T-Bridge/checkpoints/steps_45000_pytorch_model.pt
your_ckpt=/mnt/workspace/junjin/code/starVLA/checkpoints/1216_simpler_Qwen3vlGR00T_vggt_concat/checkpoints/steps_24000_pytorch_model.pt

CUDA_VISIBLE_DEVICES=3 ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16